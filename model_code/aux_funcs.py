""" Functions for inverting and running water-terminating glaciers in OGGM """
import logging
import warnings
import numpy as np
import pandas as pd
import copy
from scipy import optimize

from oggm.utils import entity_task, global_task
from oggm.core import inversion, climate
from oggm.workflow import execute_entity_task
from oggm import utils, tasks
from oggm.core.flowline import FileModel
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   PastMassBalance,
                                   AvgClimateMassBalance,
                                   RandomMassBalance)
                                   
import oggm.cfg as cfg
from oggm.core.inversion import (_inversion_poly, calving_flux_from_depth,
                                 _inversion_simple, _vol_below_water,
                                 mass_conservation_inversion,
                                 prepare_for_inversion)

from oggm.exceptions import MassBalanceCalibrationError

# Constants
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR
from oggm.cfg import G, GAUSSIAN_KERNEL

from model_code.FluxModel import FluxBasedModelWaterFront
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)

@entity_task(log)
def flowline_model_run_wt(gdir, output_filesuffix=None, mb_model=None,
                       ys=None, ye=None, zero_initial_glacier=False,
                       init_model_fls=None, store_monthly_step=False,
                       fixed_geometry_spinup_yr=None,
                       store_model_geometry=None,
                       store_fl_diagnostics=None,
                       water_level=None,
                       evolution_model=FluxBasedModelWaterFront, stop_criterion=None,
                       init_model_filesuffix=None, init_model_yr=None,
                       **kwargs):
    """Runs a model simulation with the default time stepping scheme.
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    mb_model : :py:class:`core.MassBalanceModel`
        a MassBalanceModel instance
    ys : int
        start year of the model run (default: from the config file)
    ye : int
        end year of the model run (default: from the config file)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory)
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        (new in OGGM v1.4.1: default is to follow
        cfg.PARAMS['store_model_geometry'])
    store_fl_diagnostics : bool
        whether to store the model flowline diagnostics to disk or not.
        (default is to follow cfg.PARAMS['store_fl_diagnostics'])
    evolution_model : :class:oggm.core.FlowlineModel
        which evolution model to use. Default: FluxBasedModel
    water_level : float
        the water level. It should be zero m a.s.l, but:
        - sometimes the frontal elevation is unrealistically high (or low).
        - lake terminating glaciers
        - other uncertainties
        The default is to take the water level obtained from the ice
        thickness inversion.
    stop_criterion : func
        a function which decides on when to stop the simulation. See
        `run_until_and_store` documentation for more information.
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    fixed_geometry_spinup_yr : int
        if set to an integer, the model will artificially prolongate
        all outputs of run_until_and_store to encompass all time stamps
        starting from the chosen year. The only output affected are the
        glacier wide diagnostic files - all other outputs are set
        to constants during "spinup"
     """

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        init_model_fls = fmod.fls

    mb_elev_feedback = kwargs.get('mb_elev_feedback', 'annual')
    if store_monthly_step and (mb_elev_feedback == 'annual'):
        warnings.warn("The mass-balance used to drive the ice dynamics model "
                      "is updated yearly. If you want the output to be stored "
                      "monthly and also reflect monthly processes, "
                      "set store_monthly_step=True and "
                      "mb_elev_feedback='monthly'. This is not recommended "
                      "though: for monthly MB applications, we recommend to "
                      "use the `run_with_hydro` task.")

    if cfg.PARAMS['use_inversion_params_for_run']:
        diag = gdir.get_diagnostics()
        fs = diag.get('inversion_fs', cfg.PARAMS['fs'])
        glen_a = diag.get('inversion_glen_a', cfg.PARAMS['glen_a'])
        calving_k = diag.get('calving_inversion_k', cfg.PARAMS['inversion_calving_k'])
    else:
        fs = cfg.PARAMS['fs']
        glen_a = cfg.PARAMS['glen_a']
        calving_k = cfg.PARAMS['calving_k']

    kwargs.setdefault('fs', fs)
    kwargs.setdefault('glen_a', glen_a)
    kwargs.setdefault('calving_k', calving_k)

    if store_model_geometry is None:
        store_model_geometry = cfg.PARAMS['store_model_geometry']

    if store_fl_diagnostics is None:
        store_fl_diagnostics = cfg.PARAMS['store_fl_diagnostics']

    if store_model_geometry:
        geom_path = gdir.get_filepath('model_geometry',
                                      filesuffix=output_filesuffix,
                                      delete=True)
    else:
        geom_path = False

    if store_fl_diagnostics:
        fl_diag_path = gdir.get_filepath('fl_diagnostics',
                                         filesuffix=output_filesuffix,
                                         delete=True)
    else:
        fl_diag_path = False

    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)

    if init_model_fls is None:
        fls = gdir.read_pickle('model_flowlines')
    else:
        fls = copy.deepcopy(init_model_fls)
    if zero_initial_glacier:
        for fl in fls:
            fl.thick = fl.thick * 0.

    if (cfg.PARAMS['use_kcalving_for_run'] and gdir.is_tidewater and
            water_level is None):
        # check for water level
        water_level = gdir.get_diagnostics().get('calving_water_level', None)
        if water_level is None:
            raise InvalidWorkflowError('This tidewater glacier seems to not '
                                       'have been inverted with the '
                                       '`find_inversion_calving` task. Set '
                                       "PARAMS['use_kcalving_for_run'] to "
                                       '`False` or set `water_level` '
                                       'to prevent this error.')

    model = evolution_model(fls, mb_model=mb_model, y0=ys,
                           inplace=True,
                           is_tidewater=gdir.is_tidewater,
                           is_lake_terminating=gdir.is_lake_terminating,
                           water_level=water_level,
                           **kwargs)

    with np.warnings.catch_warnings():
        # For operational runs we ignore the warnings
        np.warnings.filterwarnings('ignore', category=RuntimeWarning)
        model.run_until_and_store(ye,
                                  geom_path=geom_path,
                                  diag_path=diag_path,
                                  fl_diag_path=fl_diag_path,
                                  store_monthly_step=store_monthly_step,
                                  fixed_geometry_spinup_yr=fixed_geometry_spinup_yr,
                                  stop_criterion=stop_criterion)

    return model

@entity_task(log)
def run_random_climate_wt(gdir, nyears=1000, y0=None, halfsize=15,
                          bias=None, seed=None, temperature_bias=None,
                          precipitation_factor=None,
                          store_monthly_step=False,
                          store_model_geometry=None,
                          climate_filename='climate_historical',
                          climate_input_filesuffix='',
                          output_filesuffix='', init_model_fls=None,
                          zero_initial_glacier=False,
                          unique_samples=False, **kwargs):
    """Runs the random mass-balance model for a given number of years.

    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.flowline_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int
        length of the simulation
    y0 : int, optional
        central year of the random climate period. The default is to be
        centred on t*.
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1)
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero
    seed : int
        seed for the random generator. If you ignore this, the runs will be
        different each time. Setting it to a fixed seed across glaciers can
        be useful if you want to have the same climate years for all of them
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
        default is None and means that the precipitation factor from the
        calibration is applied which is cfg.PARAMS['prcp_scaling_factor']
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        (new in OGGM v1.4.1: default is to follow
        cfg.PARAMS['store_model_geometry'])
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    unique_samples: bool
        if true, chosen random mass-balance years will only be available once
        per random climate period-length
        if false, every model year will be chosen from the random climate
        period with the same probability
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """

    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=RandomMassBalance,
                                     y0=y0, halfsize=halfsize,
                                     bias=bias, seed=seed,
                                     filename=climate_filename,
                                     input_filesuffix=climate_input_filesuffix,
                                     unique_samples=unique_samples)

    if temperature_bias is not None:
        mb.temp_bias = temperature_bias
    if precipitation_factor is not None:
        mb.prcp_fac = precipitation_factor

    return flowline_model_run_wt(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb, ys=0, ye=nyears,
                              store_monthly_step=store_monthly_step,
                              store_model_geometry=store_model_geometry,
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
                              **kwargs)

@entity_task(log)
def run_from_climate_data_wt(gdir, ys=None, ye=None, min_ys=None, max_ys=None,
                          fixed_geometry_spinup_yr=None,
                          store_monthly_step=False,
                          store_model_geometry=None,
                          store_fl_diagnostics=None,
                          climate_filename='climate_historical',
                          climate_input_filesuffix='', output_filesuffix='',
                          init_model_filesuffix=None, init_model_yr=None,
                          init_model_fls=None, zero_initial_glacier=False,
                          bias=None, temperature_bias=None,
                          precipitation_factor=None, **kwargs):
    """ Runs a glacier with climate input from e.g. CRU or a GCM.
    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.flowline_model_run`.
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the glacier geometry
        date if init_model_filesuffix is None, else init_model_yr)
    ye : int
        end year of the model run (default: last year of the provided
        climate file)
    min_ys : int
        if you want to impose a minimum start year, regardless if the glacier
        inventory date is earlier (e.g. if climate data does not reach).
    max_ys : int
        if you want to impose a maximum start year, regardless if the glacier
        inventory date is later (e.g. if climate data does not reach).
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    store_model_geometry : bool
        whether to store the full model geometry run file to disk or not.
        (new in OGGM v1.4.1: default is to follow
        cfg.PARAMS['store_model_geometry'])
    store_fl_diagnostics : bool
        whether to store the model flowline diagnostics to disk or not.
        (default is to follow cfg.PARAMS['store_fl_diagnostics'])
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        for the output file
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory).
        Ignored if `init_model_filesuffix` is set
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
        default is None and means that the precipitation factor from the
        calibration is applied which is cfg.PARAMS['prcp_scaling_factor']
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    fixed_geometry_spinup_yr : int
        if set to an integer, the model will artificially prolongate
        all outputs of run_until_and_store to encompass all time stamps
        starting from the chosen year. The only output affected are the
        glacier wide diagnostic files - all other outputs are set
        to constants during "spinup"
    """

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        init_model_fls = fmod.fls
        if ys is None:
            ys = init_model_yr

    try:
        rgi_year = gdir.rgi_date.year
    except AttributeError:
        rgi_year = gdir.rgi_date

    # Take from rgi date if not set yet
    if ys is None:
        # The RGI timestamp is in calendar date - we convert to hydro date,
        # i.e. 2003 becomes 2004 if hydro_month is not 1 (January)
        # (so that we don't count the MB year 2003 in the simulation)
        # See also: https://github.com/OGGM/oggm/issues/1020
        # even if hydro_month is 1, we prefer to start from Jan 2004
        # as in the alps the rgi is from Aug 2003
        ys = rgi_year + 1

    if ys <= rgi_year and init_model_filesuffix is None:
        log.warning('You are attempting to run_with_climate_data at dates '
                    'prior to the RGI inventory date. This may indicate some '
                    'problem in your workflow. Consider using '
                    '`fixed_geometry_spinup_yr` for example.')

    # Final crop
    if min_ys is not None:
        ys = ys if ys > min_ys else min_ys
    if max_ys is not None:
        ys = ys if ys < max_ys else max_ys

    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=PastMassBalance,
                                     filename=climate_filename, bias=bias,
                                     input_filesuffix=climate_input_filesuffix)

    if temperature_bias is not None:
        mb.temp_bias = temperature_bias
    if precipitation_factor is not None:
        mb.prcp_fac = precipitation_factor

    if ye is None:
        # Decide from climate (we can run the last year with data as well)
        ye = mb.flowline_mb_models[0].ye + 1

    return flowline_model_run_wt(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb, ys=ys, ye=ye,
                              store_monthly_step=store_monthly_step,
                              store_model_geometry=store_model_geometry,
                              store_fl_diagnostics=store_fl_diagnostics,
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
                              fixed_geometry_spinup_yr=fixed_geometry_spinup_yr,
                              **kwargs)

                              
def _compute_thick_wt(a0s, a3, flux_a0, shape_factor, _inv_function):
    """Content of the original inner loop of the mass-conservation inversion.

    Put here to avoid code duplication.

    Parameters
    ----------
    a0s
    a3
    flux_a0
    shape_factor
    _inv_function

    Returns
    -------
    the thickness
    """

    a0s = a0s / (shape_factor ** 3)
    a3s = a3 / (shape_factor ** 3)
    if np.any(~np.isfinite(a0s)):
        raise RuntimeError('non-finite coefficients in the polynomial.')

    # Solve the polynomials
    try:
        out_thick = np.zeros(len(a0s))
        for i, (a0, a3, Q) in enumerate(zip(a0s, a3s, flux_a0)):
            out_thick[i] = _inv_function(a3, a0) if Q > 0 else 0
    except TypeError:
        # Scalar
        out_thick = _inv_function(a3, a0s) if flux_a0 > 0 else 0

    if np.any(~np.isfinite(out_thick)):
        raise RuntimeError('non-finite coefficients in the polynomial.')

    return out_thick
    
def sia_thickness_via_optim_wt(slope, width, flux, rel_h, a_factor, 
                               shape='rectangular', glen_a=None, fs=None, 
                               t_lambda=None):
    """Compute the thickness numerically instead of analytically.

    It's the only way that works for trapezoid shapes.

    Parameters
    ----------
    slope : -np.gradient(hgt, dx)
    width : section width in m
    flux : mass flux in m3 s-1
    shape : 'rectangular', 'trapezoid' or 'parabolic'
    glen_a : Glen A, defaults to PARAMS
    fs : sliding, defaults to PARAMS
    t_lambda: the trapezoid lambda, defaults to PARAMS

    Returns
    -------
    the ice thickness (in m)
    """

    if len(np.atleast_1d(slope)) > 1:
        shape = utils.tolist(shape, len(slope))
        t_lambda = utils.tolist(t_lambda, len(slope))
        out = []
        for sl, w, f, s, t in zip(slope, width, flux, shape, t_lambda):
            out.append(sia_thickness_via_optim_wt(sl, w, f, shape=s,
                                               glen_a=glen_a, fs=fs,
                                               t_lambda=t))
        return np.asarray(out)

    # Sanity
    if flux <= 0:
        return 0
    if width <= 10:
        return 0

    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']
    if t_lambda is None:
        t_lambda = cfg.PARAMS['trapezoid_lambdas']
    if shape not in ['parabolic', 'rectangular', 'trapezoid']:
        raise InvalidParamsError('shape must be `parabolic`, `trapezoid` '
                                 'or `rectangular`, not: {}'.format(shape))

    # Ice flow params
    n = cfg.PARAMS['glen_n']
    fd = 2 / (n+2) * glen_a
    rho = cfg.PARAMS['ice_density']
    rhogh = (rho * cfg.G * slope) ** n

    # To avoid geometrical inconsistencies
    max_h = width / t_lambda if shape == 'trapezoid' else 1e4

    def to_minimize(h):
        u_drag = (rho * cfg.G * slope * h * a_factor)**n * h * fd
        # u_slide = ((rho * cfg.G * slope * h * a_factor)**n * fs / 
                   # (utils.clip_min(10, h - (1028/rho) * water_depth)))
        u_slide = (rho * cfg.G * slope * a_factor)**n * fs * h**(n-1) * rel_h
        u = u_drag + u_slide
        #u = ((h ** (n + 1)) * fd * rhogh + ((h ** n) / (utils.clip_min(10, h -
        #     (1028/rho) * water_depth))) * fs * rhogh)
        if shape == 'parabolic':
            sect = 2./3. * width * h
        elif shape == 'trapezoid':
            w0m = width - t_lambda * h
            sect = (width + w0m) / 2 * h
        else:
            sect = width * h
        return sect * u - flux
    out_h, r = optimize.brentq(to_minimize, 0, max_h, full_output=True)
    return out_h

def sia_thickness_wt(slope, width, flux, rel_h, a_factor, #water_depth, f_b,
                     shape='rectangular', glen_a=None, fs=None, 
                     shape_factor=None):
    """Computes the ice thickness from mass-conservation.

    This is a utility function tested against the true OGGM inversion
    function. Useful for teaching and inversion with calving.

    Parameters
    ----------
    slope : -np.gradient(hgt, dx) (we don't clip for min slope!)
    width : section width in m
    flux : mass flux in m3 s-1
    shape : 'rectangular' or 'parabolic'
    glen_a : Glen A, defaults to PARAMS
    fs : sliding, defaults to PARAMS
    shape_factor: for lateral drag

    Returns
    -------
    the ice thickness (in m)
    """

    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']
    if shape not in ['parabolic', 'rectangular']:
        raise InvalidParamsError('shape must be `parabolic` or `rectangular`, '
                                 'not: {}'.format(shape))

    _inv_function = _inversion_simple if fs == 0 else _inversion_poly

    # Ice flow params
    fd = 2. / (cfg.PARAMS['glen_n']+2) * glen_a
    rho = cfg.PARAMS['ice_density']

    # Convert the flux to m2 s-1 (averaged to represent the sections center)
    flux_a0 = 1 if shape == 'rectangular' else 1.5
    flux_a0 *= flux / width

    # With numerically small widths this creates very high thicknesses
    try:
        flux_a0[width < 10] = 0
    except TypeError:
        if width < 10:
            flux_a0 = 0

    # Polynomial factors (a5 = 1)
    a0 = - flux_a0 / ((rho * cfg.G * slope * a_factor) ** 3 * fd)
    #a3 = fs / fd
    # a3 = (fs / fd) * ((water_depth+f_b) / utils.clip_min(10,(water_depth+f_b) -
                      # (1028/rho)*water_depth))# * a_factor    
    a3 = (fs / fd) * rel_h

    # Inversion with shape factors?
    sf_func = None
    if shape_factor == 'Adhikari' or shape_factor == 'Nye':
        sf_func = utils.shape_factor_adhikari
    elif shape_factor == 'Huss':
        sf_func = utils.shape_factor_huss

    sf = np.ones(slope.shape)  # Default shape factor is 1
    if sf_func is not None:

        # Start iteration for shape factor with first guess of 1
        i = 0
        sf_diff = np.ones(slope.shape)

        # Some hard-coded factors here
        sf_tol = 1e-2
        max_sf_iter = 20

        while i < max_sf_iter and np.any(sf_diff > sf_tol):
            out_thick = _compute_thick_wt(a0, a3, flux_a0, sf, _inv_function)
            is_rectangular = np.repeat(shape == 'rectangular', len(width))
            sf_diff[:] = sf[:]
            sf = sf_func(width, out_thick, is_rectangular)
            sf_diff = sf_diff - sf
            i += 1

        log.info('Shape factor {:s} used, took {:d} iterations for '
                 'convergence.'.format(shape_factor, i))

    return _compute_thick_wt(a0, a3, flux_a0, sf, _inv_function)
    
def mass_conservation_inversion_wt(gdir, glen_a=None, fs=None, write=True,
                                   filesuffix='', water_level=None,
                                   t_lambda=None, min_rel_h=None):
    """ Compute the glacier thickness along the flowlines

    More or less following Farinotti et al., (2009).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    glen_a : float
        glen's creep parameter A. Defaults to cfg.PARAMS.
    fs : float
        sliding parameter. Defaults to cfg.PARAMS.
    write: bool
        default behavior is to compute the thickness and write the
        results in the pickle. Set to False in order to spare time
        during calibration.
    filesuffix : str
        add a suffix to the output file
    water_level : float
        to compute volume below water level - adds an entry to the output dict
    t_lambda : float
        defining the angle of the trapezoid walls (see documentation). Defaults
        to cfg.PARAMS.
    """

    # Defaults
    if glen_a is None:
        glen_a = cfg.PARAMS['inversion_glen_a']
    if fs is None:
        fs = cfg.PARAMS['inversion_fs']
    if t_lambda is None:
        t_lambda = cfg.PARAMS['trapezoid_lambdas']

    # Check input
    _inv_function = _inversion_simple if fs == 0 else _inversion_poly

    # Ice flow params
    fd = 2. / (cfg.PARAMS['glen_n']+2) * glen_a
    a3 = fs / fd
    rho = cfg.PARAMS['ice_density']

    # Inversion with shape factors?
    sf_func = None
    use_sf = cfg.PARAMS.get('use_shape_factor_for_inversion', None)
    if use_sf == 'Adhikari' or use_sf == 'Nye':
        sf_func = utils.shape_factor_adhikari
    elif use_sf == 'Huss':
        sf_func = utils.shape_factor_huss

    # Clip the slope, in rad
    min_slope = 'min_slope_ice_caps' if gdir.is_icecap else 'min_slope'
    min_slope = np.deg2rad(cfg.PARAMS[min_slope])
    if min_rel_h is None:
        min_rel_h = 1
    out_volume = 0.

    cls = gdir.read_pickle('inversion_input')
    for cl in cls:
        # Clip slope to avoid negative and small slopes
        slope = cl['slope_angle']
        slope = utils.clip_array(slope, min_slope, np.pi/2.)
        sf = np.ones(slope.shape)  #  Default shape factor is 1
        h_tol = 1e-1
        max_iter = 20
        # Glacier width
        w = cl['width']
        k = 0
        h_diff = np.ones(slope.shape)
        a0s = - cl['flux_a0'] / ((rho*cfg.G*slope)**3*fd)
        out_thick = _compute_thick_wt(a0s, a3, cl['flux_a0'], sf, _inv_function) 
        while k < max_iter and np.any(h_diff > h_tol):
            h_diff = out_thick
            bed_h = cl['hgt']-out_thick
            water_depth = utils.clip_min(0,-bed_h+water_level)
            rel_h = out_thick / (out_thick-(1028/900)*water_depth)
            wrong_rel_h = ((rel_h < 1) | ~np.isfinite(rel_h))
            rel_h[wrong_rel_h & (water_depth > 0)] = min_rel_h
            rel_h[wrong_rel_h & (water_depth == 0)] = 1
            rel_h[-1] =  min_rel_h
            a3s = fs/fd*rel_h
            a_pull = a0s*0
            length = len(cl['width']) * cl['dx']
            stretch_dist = utils.clip_max(8e3,length)
            n_stretch = np.rint(stretch_dist/cl['dx']).astype(int)
            f_b = cl['hgt'][-1] - water_level
            pull_stress = utils.clip_min(0,0.5 * G * (900 * out_thick[-1]**2 - 
                                         1028 * water_depth[-1]**2))
            
            # Define stretch factor and add to driving stress
            stretch_factor = np.zeros(n_stretch)
            for j in range(n_stretch):
                stretch_factor[j] = 2*(j+1)/(n_stretch+1)
            if cl['dx'] > stretch_dist:
                stretch_factor = stretch_dist / cl['dx']
                n_stretch = 1
                
            a_pull[-(n_stretch-1):] = (stretch_factor[:-1] * (pull_stress / 
                                                       stretch_dist))
            a_factor = (a_pull / (rho*cfg.G*slope*out_thick)) + 1
            a_factor = np.nan_to_num(a_factor, nan=1, posinf=1, neginf=1)
            a0s = - cl['flux_a0'] / ((rho*cfg.G*slope*a_factor)**3*fd)
            out_thick = _compute_thick_wt(a0s, a3s, cl['flux_a0'], sf, _inv_function)

            if sf_func is not None:

                # Start iteration for shape factor with first guess of 1
                i = 0
                sf_diff = np.ones(slope.shape)

                # Some hard-coded factors here
                sf_tol = 1e-2
                max_sf_iter = 20

                while i < max_sf_iter and np.any(sf_diff > sf_tol):
                    out_thick = _compute_thick_wt(a0s, a3, cl['flux_a0'], sf,
                                               _inv_function)

                    sf_diff[:] = sf[:]
                    sf = sf_func(w, out_thick, cl['is_rectangular'])
                    sf_diff = sf_diff - sf
                    i += 1

                log.info('Shape factor {:s} used, took {:d} iterations for '
                         'convergence.'.format(use_sf, i))

                # TODO: possible shape factor optimisations
                # thick update could be used as iteration end criterion instead
                # we iterate for all grid points, even if some already converged

            # volume
            is_rect = cl['is_rectangular']
            fac = np.where(is_rect, 1, 2./3.)
            volume = fac * out_thick * w * cl['dx']

            # Now recompute thickness where parabola is too flat
            is_trap = cl['is_trapezoid']
            if cl['invert_with_trapezoid']:
                min_shape = cfg.PARAMS['mixed_min_shape']
                bed_shape = 4 * out_thick / w ** 2
                is_trap = ((bed_shape < min_shape) & ~ cl['is_rectangular'] &
                           (cl['flux'] > 0)) | is_trap
                for i in np.where(is_trap)[0]:
                    try:
                        out_thick[i] = sia_thickness_via_optim_wt(slope[i], w[i],
                                                               cl['flux'][i], 
                                                               rel_h[i],
                                                               a_factor[i],
                                                               shape='trapezoid',
                                                               t_lambda=t_lambda,
                                                               glen_a=glen_a,
                                                               fs=fs)
                        sect = (2*w[i] - t_lambda * out_thick[i]) / 2 * out_thick[i]
                        volume[i] = sect * cl['dx']
                    except ValueError:
                        # no solution error - we do with rect
                        out_thick[i] = sia_thickness_via_optim_wt(slope[i], w[i],
                                                               cl['flux'][i], 
                                                               rel_h[i],
                                                               a_factor[i],
                                                               shape='rectangular',
                                                               glen_a=glen_a,
                                                               fs=fs)
                        is_rect[i] = True
                        is_trap[i] = False
                        volume[i] = out_thick[i] * w[i] * cl['dx']
            h_diff = h_diff - out_thick
            k += 1
        # Sanity check
        if np.any(out_thick <= 0):
            log.warning("Found zero or negative thickness: "
                        "this should not happen.")

        if write:
            cl['is_trapezoid'] = is_trap
            cl['is_rectangular'] = is_rect
            cl['thick'] = out_thick
            cl['volume'] = volume

            # volume below sl
            try:
                bed_h = cl['hgt'] - out_thick
                bed_shape = 4 * out_thick / w ** 2
                if np.any(bed_h < 0):
                    cl['volume_bsl'] = _vol_below_water(cl['hgt'], bed_h,
                                                        bed_shape, out_thick, w,
                                                        cl['is_rectangular'],
                                                        cl['is_trapezoid'],
                                                        fac, t_lambda,
                                                        cl['dx'], 0)
                if water_level is not None and np.any(bed_h < water_level):
                    cl['volume_bwl'] = _vol_below_water(cl['hgt'], bed_h,
                                                        bed_shape, out_thick, w,
                                                        cl['is_rectangular'],
                                                        cl['is_trapezoid'],
                                                        fac, t_lambda,
                                                        cl['dx'],
                                                        water_level)
            except KeyError:
                # cl['hgt'] is not available on old prepro dirs
                pass

        out_volume += np.sum(volume)

    if write:
        gdir.write_pickle(cls, 'inversion_output', filesuffix=filesuffix)
        gdir.add_to_diagnostics('inversion_glen_a', glen_a)
        gdir.add_to_diagnostics('inversion_fs', fs)

    return out_volume

@utils.global_task(log)
def inversion_tasks(gdirs, glen_a=None, fs=None, filter_inversion_output=True,
                    ref_period='None'):
    """Run all ice thickness inversion tasks on a list of glaciers.

    Quite useful to deal with calving glaciers as well.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    """
    
    if ref_period is None:
        ref_period = cfg.PARAMS['geodetic_mb_period']
        y0, y1 = ref_period.split('_')
        y0 = int(y0.split('-')[0])
        y1 = int(y1.split('-')[0])
        mb_years = [y0, y1-1]
    else:
        y0, y1 = ref_period.split('_')
        y0 = int(y0.split('-')[0])
        y1 = int(y1.split('-')[0])
        mb_years = [y0, y1-1]       
    
    if cfg.PARAMS['use_kcalving_for_inversion']:
        # Differentiate between calving and non-calving glaciers
        gdirs_nc = []
        gdirs_c = []
        for gd in gdirs:
            if gd.is_tidewater:
                gdirs_c.append(gd)
            else:
                gdirs_nc.append(gd)

        log.workflow('Starting inversion tasks for {} tidewater and {} '
                     'non-tidewater glaciers.'.format(len(gdirs_c),
                                                      len(gdirs_nc)))

        if gdirs_nc:
            execute_entity_task(tasks.prepare_for_inversion, gdirs_nc)
            execute_entity_task(tasks.mass_conservation_inversion, gdirs_nc,
                                glen_a=glen_a, fs=fs)
            if filter_inversion_output:
                execute_entity_task(tasks.filter_inversion_output, gdirs_nc)

        if gdirs_c:
            execute_entity_task(find_inversion_calving_from_any_mb, gdirs_c,
                                glen_a=glen_a, fs=fs, mb_years=mb_years)
    else:
        execute_entity_task(tasks.prepare_for_inversion, gdirs)
        execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                            glen_a=glen_a, fs=fs)
        if filter_inversion_output:
            execute_entity_task(tasks.filter_inversion_output, gdirs)

@entity_task(log, writes=['diagnostics'])
def match_geodetic_mb_for_glacier(gdir, period='2000-01-01_2020-01-01',
                                  file_path=None, fail_safe=True,
                                  include_calving=True, corr_factor=1):
    """Shift the mass-balance residual to match geodetic mb observations.

    It is similar to match_regional_geodetic_mb but uses the raw, glacier
    per glacier tabular data.

    This method finds the "best mass-balance residual" to match all glaciers in
    gdirs with available OGGM mass balance and available geodetic mass-balance
    measurements from Hugonnet 2021 or any other file with the same format.

    The default is to use hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide_filled.hdf
    in  https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/

    Parameters
    ----------
    gdirs : the list of gdirs
    period : str
       One of
       '2000-01-01_2020-01-01',
       '2000-01-01_2010-01-01',
       '2010-01-01_2020-01-01'.
    file_path: str
       local file path to tabular file containing geodetic measurements, file must
       contain the columns:
           - 'rgiid': is the RGIId as in the RGI 6.0
           - 'period': time intervall of the measurements in the format shown
             above
           - 'dmdtda': the specific-mass change rate in meters water-equivalent
             per year,
           - 'area': is the glacier area (same as in RGI 6.0) in meters square
    fail_safe : bool
        some glaciers in the obs data have been corrected with the regional
        average. We don't use these values, unless there is no other choice and
        in which case you can set fail_safe to True
    """

    # Get the mass-balance OGGM would give out of the box
    df = utils.compile_fixed_geometry_mass_balance(gdir, path=False)
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # And also the Area and calving fluxes
    dfs = utils.compile_glacier_statistics(gdir, path=False)

    y0 = int(period.split('_')[0].split('-')[0])
    y1 = int(period.split('_')[1].split('-')[0]) - 1

    odf = pd.DataFrame(df.loc[y0:y1].mean(), columns=['SMB'])

    odf['AREA'] = dfs.rgi_area_km2 * 1e6
    # Just take the calving rate and change its units
    # Original units: km3 a-1, to change to mm a-1 (units of specific MB)
    rho = cfg.PARAMS['ice_density']
    if 'calving_flux' in dfs and include_calving:
        odf['CALVING'] = dfs['calving_flux'].fillna(0) * 1e9 * rho / odf['AREA']
    else:
        odf['CALVING'] = 0
    
    # We have to drop nans here, which occur when calving glaciers fail to run
    odf = odf.dropna()

    # save all rgi_ids for which a valid OGGM mb is available
    rgi_ids_oggm = odf.index.values

    # fetch the file online or read custom file
    if file_path is None:
        base_url = 'https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/'
        file_name = 'hugonnet_2021_ds_rgi60_pergla_rates_10_20_worldwide_filled.hdf'
        df = pd.read_hdf(utils.file_downloader(base_url + file_name))
    else:
        extension = os.path.splitext(file_path)[1]
        if extension == '.csv':
            df = pd.read_csv(file_path, index_col='rgiid')
        elif extension == '.hdf':
            df = pd.read_hdf(file_path, index_col='rgiid')

    # get the correct period from the whole dataset
    df = df.loc[df['period'] == period]

    # get only geodetic measurements for which a valid OGGM mb is available
    rdf_all = df.loc[rgi_ids_oggm]
    if rdf_all.empty:
        raise InvalidWorkflowError('No geodetic MB measurements available for '
                                   'this glacier selection!')

    # drop glaciers with no valid geodetic measurements
    rdf = rdf_all.loc[~rdf_all['is_cor']]
    if rdf.empty:
        if not fail_safe:
            raise InvalidWorkflowError('No gedoetic MB measurements available for '
                                       'this glacier selection! Set '
                                       'fail_safe=True to use the '
                                       'corrected values.')
        rdf = rdf_all

    # the remaining glaciers now have a OGGM mb and geodetic measurements
    rgi_ids = rdf.index.values
    msg = ('Applying geodetic MB correction for {}')
    log.workflow(msg.format(rgi_ids))

    # Total MB OGGM, only using glaciers with OGGM mb and geodetic measurements
    odf = odf.loc[rgi_ids]
    out_smb = np.average(odf['SMB'], weights=odf['AREA'])  # for logging
    out_cal = np.average(odf['CALVING'], weights=odf['AREA'])  # for logging
    smb_oggm = np.average(odf['SMB'] - odf['CALVING'], weights=odf['AREA'])

    # Total geodetic MB, no need for indexing
    smb_ref = rdf.dmdtda.values * 1000 # m to mm conversion
    area_ref = rdf.area.values
    smb_ref = np.average(smb_ref, weights=area_ref) * corr_factor

    # Diff between the two
    residual = smb_ref - smb_oggm

    # Let's just shift
    # log.workflow('Shifting glacier MB bias by {}'.format(residual))
    # log.workflow('Observations give {}'.format(smb_ref))
    # log.workflow('OGGM SMB gives {}'.format(out_smb))
    # log.workflow('OGGM frontal ablation gives {}'.format(out_cal))

    # This time we shift over all glaciers
    try:
        df = gdir.read_json('local_mustar')
        gdir.add_to_diagnostics('mb_bias_before_geodetic_corr', df['bias'])
        df['bias'] = df['bias'] - residual
        gdir.write_json(df, 'local_mustar')
    except FileNotFoundError:
        pass

@utils.global_task(log)
def calibrate_inversion_from_consensus_fs(gdirs, ignore_missing=True, 
                                          fs=1e-20,
                                          fs_bounds=(0.1,100), a=2.4e-24,
                                          error_on_mismatch=True,
                                          filter_inversion_output=True):
    """Fit the total volume of the glaciers to the 2019 consensus estimate.

    This method finds the "best Glen A" to match all glaciers in gdirs with
    a valid inverted volume.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    ignore_missing : bool
        set this to true to silence the error if some glaciers could not be
        found in the consensus estimate.
    fs : float
        invert with sliding (default: no)
    a_bounds: tuple
        factor to apply to default A
    apply_fs_on_mismatch: false
        on mismatch, try to apply an arbitrary value of fs (fs = 5.7e-20 from
        Oerlemans) and try to optimize A again.
    error_on_mismatch: bool
        sometimes the given bounds do not allow to find a zero mismatch:
        this will normally raise an error, but you can switch this off,
        use the closest value instead and move on.
    filter_inversion_output : bool
        whether or not to apply terminus thickness filtering on the inversion
        output (needs the downstream lines to work).

    Returns
    -------
    a dataframe with the individual glacier volumes
    """

    gdirs = utils.tolist(gdirs)

    # Get the ref data for the glaciers we have
    df = pd.read_hdf(utils.get_demo_file('rgi62_itmix_df.h5'))
    rids = [gdir.rgi_id for gdir in gdirs]

    found_ids = df.index.intersection(rids)
    if not ignore_missing and (len(found_ids) != len(rids)):
        raise InvalidWorkflowError('Could not find matching indices in the '
                                   'consensus estimate for all provided '
                                   'glaciers. Set ignore_missing=True to '
                                   'ignore this error.')

    df = df.reindex(rids)

    # Optimize the diff to ref
    def_a = a
    def_fs = fs

    def compute_vol(x):
        inversion_tasks(gdirs, glen_a=def_a, fs=x*def_fs,
                        filter_inversion_output=filter_inversion_output)
        odf = df.copy()
        odf['oggm'] = execute_entity_task(tasks.get_inversion_volume, gdirs)
        return odf.dropna()

    def to_minimize(x):
        log.workflow('Consensus estimate optimisation with '
                     'A: {} and fs factor: {}'.format(a, x))
        odf = compute_vol(x)
        return odf.vol_itmix_m3.sum() - odf.oggm.sum()

    try:
        out_fac, r = optimize.brentq(to_minimize, *fs_bounds, rtol=1e-2,
                                     full_output=True)
        if r.converged:
            log.workflow('calibrate_inversion_from_consensus '
                         'converged after {} iterations and Glen A={}. The '
                         'resulting fs factor is {}.'
                         ''.format(r.iterations, a, out_fac))
        else:
            raise ValueError('Unexpected error in optimization.brentq')
            
    except ValueError:
        # Ok can't find an A. Log for debug:
        odf1 = compute_vol(fs_bounds[0]).sum() * 1e-9
        odf2 = compute_vol(fs_bounds[1]).sum() * 1e-9
        msg = ('calibration from consensus estimate CANT converge with Glen A={}.\n'
               'Bound values (km3):\nRef={:.3f} OGGM={:.3f} for fs factor {}\n'
               'Ref={:.3f} OGGM={:.3f} for fs factor {}'
               ''.format(a,
                         odf1.vol_itmix_m3, odf1.oggm, fs_bounds[0],
                         odf2.vol_itmix_m3, odf2.oggm, fs_bounds[1]))
        if error_on_mismatch:
            raise ValueError(msg)

        out_fac = fs_bounds[int(abs(odf1.vol_itmix_m3 - odf1.oggm) >
                               abs(odf2.vol_itmix_m3 - odf2.oggm))]
        log.workflow(msg)
        log.workflow('We use fs factor = {} and Glen A = {} and move on.'
                     ''.format(out_fac, def_a))

    # Compute the final volume with the correct A
    inversion_tasks(gdirs, glen_a=def_a, fs=out_fac*def_fs,
                    filter_inversion_output=filter_inversion_output)
    df['vol_oggm_m3'] = execute_entity_task(tasks.get_inversion_volume, gdirs)
    return df

@utils.global_task(log)
def calibrate_inversion_from_consensus(gdirs, ignore_missing=True,
                                       fs=0, a_bounds=(0.01, 1),
                                       apply_fs_on_mismatch=False,
                                       error_on_mismatch=True,
                                       filter_inversion_output=True):
    """Fit the total volume of the glaciers to the 2019 consensus estimate.

    This method finds the "best Glen A" to match all glaciers in gdirs with
    a valid inverted volume.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    ignore_missing : bool
        set this to true to silence the error if some glaciers could not be
        found in the consensus estimate.
    fs : float
        invert with sliding (default: no)
    a_bounds: tuple
        factor to apply to default A
    apply_fs_on_mismatch: false
        on mismatch, try to apply an arbitrary value of fs (fs = 5.7e-20 from
        Oerlemans) and try to optimize A again.
    error_on_mismatch: bool
        sometimes the given bounds do not allow to find a zero mismatch:
        this will normally raise an error, but you can switch this off,
        use the closest value instead and move on.
    filter_inversion_output : bool
        whether or not to apply terminus thickness filtering on the inversion
        output (needs the downstream lines to work).

    Returns
    -------
    a dataframe with the individual glacier volumes
    """

    gdirs = utils.tolist(gdirs)

    # Get the ref data for the glaciers we have
    df = pd.read_hdf(utils.get_demo_file('rgi62_itmix_df.h5'))
    rids = [gdir.rgi_id for gdir in gdirs]

    found_ids = df.index.intersection(rids)
    if not ignore_missing and (len(found_ids) != len(rids)):
        raise InvalidWorkflowError('Could not find matching indices in the '
                                   'consensus estimate for all provided '
                                   'glaciers. Set ignore_missing=True to '
                                   'ignore this error.')

    df = df.reindex(rids)

    # Optimize the diff to ref
    def_a = cfg.PARAMS['inversion_glen_a']

    def compute_vol(x):
        inversion_tasks(gdirs, glen_a=x*def_a, fs=fs,
                        filter_inversion_output=filter_inversion_output)
        odf = df.copy()
        odf['oggm'] = execute_entity_task(tasks.get_inversion_volume, gdirs)
        return odf.dropna()

    def to_minimize(x):
        log.workflow('Consensus estimate optimisation with '
                     'A factor: {} and fs: {}'.format(x, fs))
        odf = compute_vol(x)
        return odf.vol_itmix_m3.sum() - odf.oggm.sum()

    try:
        out_fac, r = optimize.brentq(to_minimize, *a_bounds, rtol=1e-2,
                                         full_output=True)
        if r.converged:
            log.workflow('calibrate_inversion_from_consensus '
                         'converged after {} iterations and fs={}. The '
                         'resulting Glen A factor is {}.'
                         ''.format(r.iterations, fs, out_fac))
        else:
            raise ValueError('Unexpected error in optimization.brentq')
            
    except ValueError:
        # Ok can't find an A. Log for debug:
        odf1 = compute_vol(a_bounds[0]).sum() * 1e-9
        odf2 = compute_vol(a_bounds[1]).sum() * 1e-9
        msg = ('calibration from consensus estimate CANT converge with fs={}.\n'
               'Bound values (km3):\nRef={:.3f} OGGM={:.3f} for A factor {}\n'
               'Ref={:.3f} OGGM={:.3f} for A factor {}'
               ''.format(fs,
                         odf1.vol_itmix_m3, odf1.oggm, a_bounds[0],
                         odf2.vol_itmix_m3, odf2.oggm, a_bounds[1]))
        if apply_fs_on_mismatch and fs != 0 and  odf2.oggm > odf2.vol_itmix_m3:
            log.workflow(msg)
            out_fac = a_bounds[int(abs(odf1.vol_itmix_m3 - odf1.oggm) >
                                   abs(odf2.vol_itmix_m3 - odf2.oggm))]
            return calibrate_inversion_from_consensus_fs(gdirs,a=out_fac*def_a,
                                                         fs=fs,
                                                         ignore_missing=\
                                                         ignore_missing,
                                                         error_on_mismatch=\
                                                         error_on_mismatch)
        if error_on_mismatch:
            raise ValueError(msg)

        out_fac = a_bounds[int(abs(odf1.vol_itmix_m3 - odf1.oggm) >
                               abs(odf2.vol_itmix_m3 - odf2.oggm))]
        log.workflow(msg)
        log.workflow('We use A factor = {} and fs = {} and move on.'
                     ''.format(out_fac, fs))

    # Compute the final volume with the correct A
    inversion_tasks(gdirs, glen_a=out_fac*def_a, fs=fs,
                    filter_inversion_output=filter_inversion_output)
    df['vol_oggm_m3'] = execute_entity_task(tasks.get_inversion_volume, gdirs)
    return df
    
    
@entity_task(log, writes=['inversion_flowlines'],
             fallback=climate._fallback_mu_star_calibration)
def mu_star_calibration_from_geodetic_mb(gdir, 
                                         fa_data_path=None,
                                         ref_mb=None,
                                         ref_period='',
                                         step_height_for_corr=25,
                                         max_height_change_for_corr=3000,
                                         ignore_hydro_months=False,
                                         min_mu_star=None,
                                         max_mu_star=None,
                                         corr_factor=0.75, unc='no'):
    """Compute the flowlines' mu* from the reference geodetic MB data.
    This is similar to mu_star_calibration but using the reference geodetic
    MB data instead, and this does NOT compute the apparent mass-balance at
    the same time - users need to run apparent_mb_from_any_mb separately.
    Currently only works for single flowlines.
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ref_mb : float
        the reference mass-balance to match (units: kg m-2 yr-1)
    ref_period : str, default: PARAMS['geodetic_mb_period']
        one of '2000-01-01_2010-01-01', '2010-01-01_2020-01-01',
        '2000-01-01_2020-01-01'. If `ref_mb` is set, this should still match
        the same format but can be any date.
    ignore_hydro_months: bool, optional
        do not raise and error if we are not working on calendar years.
    min_mu_star: bool, optional
        defaults to cfg.PARAMS['min_mu_star']
    max_mu_star: bool, optional
        defaults to cfg.PARAMS['max_mu_star']
    """

    diag = gdir.get_diagnostics()
    unc = diag.get('unc_fa', unc)
    log.workflow('({}) Unc. applied: {}'.format(gdir.rgi_id,unc))

    # mu* constraints
    if min_mu_star is None:
        min_mu_star = cfg.PARAMS['min_mu_star']
    if max_mu_star is None:
        max_mu_star = cfg.PARAMS['max_mu_star']
    _brentq_xtol = cfg.PARAMS['_brentq_xtol']
    
    MB_PARAMS = ['temp_default_gradient', 'temp_all_solid', 'temp_all_liq',
                 'temp_melt', 'prcp_scaling_factor', 'climate_qc_months',
                 'hydro_month_nh', 'hydro_month_sh']

    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    if sm != 1 and not ignore_hydro_months:
        raise InvalidParamsError('mu_star_calibration_from_geodetic_mb makes '
                                 'more sense when applied on calendar years '
                                 "(PARAMS['hydro_month_nh']=1 and "
                                 "`PARAMS['hydro_month_sh']=1). If you want "
                                 "to ignore this error, set "
                                 "ignore_hydro_months to True")

    if max_mu_star > 1000:
        raise InvalidParamsError('You seem to have set a very high '
                                 'max_mu_star for this run. This is not '
                                 'how this task is supposed to work, and '
                                 'we recommend a value lower than 1000 '
                                 '(or even 600).')

    if not ref_period:
        ref_period = cfg.PARAMS['geodetic_mb_period']

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines')

    # If someone called another task before we need to reset this
    for fl in fls:
        fl.mu_star_is_valid = False

    # Let's go
    # Climate period
    y0, y1 = ref_period.split('_')
    y0 = int(y0.split('-')[0])
    y1 = int(y1.split('-')[0])
    yr_range = [y0, y1-1]

    # Climate data on flowline
    heights = np.array([])
    widths = np.array([])
    for fl in fls:
        heights = np.append(heights, fl.surface_h)
        widths = np.append(widths, fl.widths)

    _, temp, prcp = climate.mb_yearly_climate_on_height(gdir, heights,
                                                year_range=yr_range,
                                                flatten=False)

    # Get the reference data
    if ref_mb is None:
        ref_mb = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
        ref_mb = float(ref_mb.loc[ref_mb['period'] == ref_period]['dmdtda'])
        # dmdtda: in meters water-equivalent per year -> we convert
        ref_mb *= 1000 # kg m-2 yr-1
    log.workflow('({}) ref. m.b.: {}'.format(gdir.rgi_id,ref_mb))        
    # Do we have a calving glacier?
    # cmb = calving_mb(gdir)
    # if cmb != 0:
    #    raise NotImplementedError('Calving with geodetic MB is not implemented '
    #                              'yet, but it should actually work. Well keep '
    #                              'you posted!')
    if gdir.is_tidewater: 
        fa_will = np.genfromtxt(fa_data_path, delimiter=',', 
                                usecols=np.arange(0, 10), dtype='unicode')
        rgi_list = fa_will[1:,0]
        rgi_id = gdir.rgi_id

        if ref_period[:4] == '2000' and ref_period[11:15] == '2010':
            calving_will = float(fa_will[np.where(fa_will[:, 0] == rgi_id),4])
            terminus_change_will = (float(fa_will[np.where(fa_will[:, 0] == rgi_id),8])*
                                    corr_factor)
            unc_will = float(fa_will[np.where(fa_will[:, 0] == rgi_id),5])
        elif ref_period[:4] == '2010' and ref_period[11:15] == '2020':
            calving_will = float(fa_will[np.where(fa_will[:, 0] == rgi_id),6])
            terminus_change_will = (float(fa_will[np.where(fa_will[:, 0] == rgi_id),9])*
                                    corr_factor)
            unc_will = float(fa_will[np.where(fa_will[:, 0] == rgi_id),7])
        if (ref_period[:4] == '2000' and ref_period[11:15] == '2020') or calving_will == 0:
            calving_will = (float(fa_will[np.where(fa_will[:, 0] == rgi_id),6]) +
                           float(fa_will[np.where(fa_will[:, 0] == rgi_id),4])) / 2
            terminus_change_will = ((float(fa_will[np.where(fa_will[:, 0] == rgi_id),8])*
                                    corr_factor) + (float(
                                    fa_will[np.where(fa_will[:, 0] == rgi_id),9])*
                                    corr_factor)) / 2
            unc_will = (float(fa_will[np.where(fa_will[:, 0] == rgi_id),5])**2 + 
                        float(fa_will[np.where(fa_will[:, 0] == rgi_id),7])**2)**0.5 / 2 
 
        if unc == 'low':
            calving_will_unc = calving_will - unc_will
            calving_will_unc = utils.clip_min(calving_will_unc, 0.1*calving_will)
            terminus_change_will = terminus_change_will*(calving_will_unc/calving_will)
            calving_will = calving_will_unc
        if unc == 'half':
            calving_will = 0.5*calving_will
            terminus_change_will = 0.5*terminus_change_will 
        if unc == 'vlow':
            calving_will = 0.1*calving_will
            terminus_change_will = 0.1*terminus_change_will        
        if unc == 'ulow':
            calving_will = 0.01*calving_will
            terminus_change_will = 0.01*terminus_change_will
        if unc == 'high':
            calving_will_unc = calving_will + unc_will
            terminus_change_will = terminus_change_will*(calving_will_unc/calving_will)
            calving_will = calving_will_unc

        cmb = (calving_will * 1e12 / gdir.rgi_area_m2)
        log.workflow('({}) Frontal ablation added: {}'.format(gdir.rgi_id,cmb)) 
        # _mu_star_per_minimization solves for 0, we add calving to the match
        ref_mb += cmb
        tmb = (terminus_change_will * 1e12 / gdir.rgi_area_m2)
        log.workflow('({}) Terminus change substracted: {}'.format(gdir.rgi_id,tmb))
        ref_mb += tmb

    try:
        mu_star = optimize.brentq(climate._mu_star_per_minimization,
                                  min_mu_star, max_mu_star,
                                  args=(fls, ref_mb, temp, prcp, widths),
                                  xtol=_brentq_xtol)
    except ValueError:
        # This happens when out of bounds

        # Funny enough, this bias correction is arbitrary.
        # Here I'm trying something arbitrary as well.
        # Let's try to find a range of corrections that would lead to an
        # allowed mu* and pick one

        # Here we ignore the previous QC correction - if any -
        # to ensure that results are the same even after previous correction
        fpath = gdir.get_filepath('climate_historical')
        with utils.ncDataset(fpath, 'a') as nc:
            start = getattr(nc, 'uncorrected_ref_hgt', nc.ref_hgt)
            nc.uncorrected_ref_hgt = start
            nc.ref_hgt = start

        # Read timeseries again after reset
        _, temp, prcp = climate.mb_yearly_climate_on_height(gdir, heights,
                                                    year_range=yr_range,
                                                    flatten=False)

        # Check in which direction we should correct the temp
        _lim0 = climate._mu_star_per_minimization(min_mu_star, fls, ref_mb, temp,
                                          prcp, widths)
        if _lim0 < 0:
            # The mass-balances are too positive to be matched - we need to
            # cool down the climate data
            step = -step_height_for_corr
            end = -max_height_change_for_corr
        else:
            # The other way around
            step = step_height_for_corr
            end = max_height_change_for_corr

        steps = np.arange(start, start + end, step, dtype=np.int64)
        mu_candidates = steps * np.NaN
        for i, h in enumerate(steps):
            with utils.ncDataset(fpath, 'a') as nc:
                nc.ref_hgt = h

            # Read timeseries
            _, temp, prcp = climate.mb_yearly_climate_on_height(gdir, heights,
                                                        year_range=yr_range,
                                                        flatten=False)

            try:
                mu_star = optimize.brentq(climate._mu_star_per_minimization,
                                          min_mu_star, max_mu_star,
                                          args=(fls, ref_mb, temp, prcp, widths),
                                          xtol=_brentq_xtol)
            except ValueError:
                mu_star = np.NaN

            # Done - store for later
            mu_candidates[i] = mu_star
            # if we find one working mu_star we can actually stop
            # the loop to make it faster.
            # We are here only interested in the candidate which
            # changes the ref_hgt the least!
            if np.isfinite(mu_star):
                break

        # the workflow below works in general when having more candidates
        # but also works for one candidate (as we stopped the loop)
        sel_steps = steps[np.isfinite(mu_candidates)]
        sel_mus = mu_candidates[np.isfinite(mu_candidates)]
        if len(sel_mus) == 0:
            # Yeah nothing we can do here
            raise MassBalanceCalibrationError('We could not find a way to '
                                              'correct the climate data and '
                                              'fit within the prescribed '
                                              'bounds for mu*.')

        # We have just picked the first, but to be fair it is arbitrary
        # We could also pick one randomly... but here we rather prefer to have
        # the smallest ref_hgt change as possible (hence smalles temp. bias change)
        mu_star = sel_mus[0]
        # Final correction of the data
        with utils.ncDataset(fpath, 'a') as nc:
            nc.ref_hgt = sel_steps[0]
        gdir.add_to_diagnostics('ref_hgt_calib_diff', sel_steps[0] - start)

    if not np.isfinite(mu_star):
        raise MassBalanceCalibrationError('{} '.format(gdir.rgi_id) +
                                          'has a non finite mu.')

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around without re-calibration
    out = gdir.get_climate_info()
    out['mb_calib_params'] = {k: cfg.PARAMS[k] for k in MB_PARAMS}
    gdir.write_json(out, 'climate_info')
    log.workflow('({}) Found mu*: {}'.format(gdir.rgi_id,mu_star))
    # Store diagnostics
    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['t_star'] = np.nan
    df['bias'] = 0
    df['mu_star_per_flowline'] = [mu_star] * len(fls)
    df['mu_star_glacierwide'] = mu_star
    df['mu_star_flowline_avg'] = mu_star
    df['mu_star_allsame'] = True
    # Write
    gdir.write_json(df, 'local_mustar')

@entity_task(log, writes=['diagnostics'])
def find_inversion_calving_from_any_mb(gdir, mb_model=None, mb_years=None,
                                       water_level=None, glen_a=None, fs=None):
    """Optimized search for a calving flux compatible with the bed inversion.
    See Recinos et al 2019 for details. This task is an update to
    `find_inversion_calving` but acting upon a MB residual (i.e. a shift)
    instead of the model temperature sensitivity.
    Parameters
    ----------
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
        the mass-balance model to use
    mb_years : array
        the array of years from which you want to average the MB for (for
        mb_model only).
    water_level : float
        the water level. It should be zero m a.s.l, but:
        - sometimes the frontal elevation is unrealistically high (or low).
        - lake terminating glaciers
        - other uncertainties
        With this parameter, you can produce more realistic values. The default
        is to infer the water level from PARAMS['free_board_lake_terminating']
        and PARAMS['free_board_marine_terminating']
    glen_a : float, optional
    fs : float, optional
    """
    from oggm.core import climate

    if not gdir.is_tidewater or not cfg.PARAMS['use_kcalving_for_inversion']:
        # Do nothing
        return

    diag = gdir.get_diagnostics()
    calving_k = diag.get('optimized_k', cfg.PARAMS['inversion_calving_k'])

    # Get the relevant variables
    cls = gdir.read_pickle('inversion_input')[-1]
    slope = cls['slope_angle'][-1]
    width = cls['width'][-1]

    # Stupidly enough the slope is clipped in the OGGM inversion, not
    # in inversion prepro - clip here
    min_slope = 'min_slope_ice_caps' if gdir.is_icecap else 'min_slope'
    min_slope = np.deg2rad(cfg.PARAMS[min_slope])
    slope = utils.clip_array(slope, min_slope, np.pi / 2.)

    gdir.inversion_calving_rate = 0
    with utils.DisableLogger():
        climate.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
                                        mb_years=mb_years)
        prepare_for_inversion(gdir)
        v_ref = mass_conservation_inversion(gdir,glen_a=glen_a,fs=fs)
        #tasks.filter_inversion_output(gdir)
    # Store for statistics        
    gdir.add_to_diagnostics('volume_before_water', v_ref)

    cl = gdir.read_pickle('inversion_output')[-1]
    log.workflow('hgt, thick before before: {}, {}'.format(cls['hgt'][-1], cl['thick'][-1]))  
    thick0 = cl['thick'][-1]
    th = cls['hgt'][-1]
    if water_level is None:
        water_level = th - thick0 if thick0 > 10*th else 0
        if gdir.is_lake_terminating:
            water_level = th - cfg.PARAMS['free_board_lake_terminating']    
    # Check that water level is within given bounds
    # vmin, vmax = cfg.PARAMS['free_board_marine_terminating'] # BUOYANCY???
    # if water_level is None:
        # th = cls['hgt'][-1]
        # if gdir.is_lake_terminating:
            # water_level = th - cfg.PARAMS['free_board_lake_terminating']
        # else:
            # water_level = utils.clip_scalar(0, th - thick0*vmax, th - thick0*vmin)
            # chosemax = False
            # chosemin = False

    # The functions all have the same shape: they decrease, then increase
    # We seek the absolute minimum first
    def to_minimize(rel_h):
        f_b = cls['hgt'][-1] - water_level
        thick = ((1028/900)*rel_h*f_b) / ((1028/900) * rel_h - rel_h + 1)
        #water_depth = (thick - thick/rel_h)/(1028/900)
        water_depth = thick - f_b
        fl = calving_flux_from_depth(gdir, water_level=water_level, k=calving_k,
                                     water_depth=water_depth)

        flux = fl['flux'] * 1e9 / cfg.SEC_IN_YEAR
        f_b = fl['free_board']
        thick = fl['thick']
        a_pull = cls['width']*0
        length = len(cls['width']) * cls['dx']
        stretch_dist = utils.clip_max(8e3,length)
        n_stretch = np.rint(stretch_dist/cls['dx']).astype(int)
        
        pull_stress = utils.clip_min(0,0.5 * G * (900 * thick**2 - 
                                     1028 * water_depth**2))
        
        # Define stretch factor and add to driving stress
        stretch_factor = np.zeros(n_stretch)
        for j in range(n_stretch):
            stretch_factor[j] = 2*(j+1)/(n_stretch+1)
        if cls['dx'] > stretch_dist:
            stretch_factor = stretch_dist / cls['dx']
            n_stretch = 1
        stretch_factor = stretch_factor[-2]    
        a_pull = (stretch_factor * (pull_stress / stretch_dist))
        a_factor = ((a_pull / (900*cfg.G*slope*thick)) + 1)
        a_factor = np.nan_to_num(a_factor, nan=1, posinf=1, neginf=1)
        sia_thick = sia_thickness_wt(slope, width, flux, rel_h, a_factor,
                                     glen_a=glen_a, fs=fs)
        is_trap = cl['is_trapezoid'][-1]
        if cl['invert_with_trapezoid']:
            t_lambda = cfg.PARAMS['trapezoid_lambdas']
            min_shape = cfg.PARAMS['mixed_min_shape']
            bed_shape = 4 * sia_thick / cl['width'][-1] ** 2
            is_trap = ((bed_shape < min_shape) & ~ cl['is_rectangular'][-1] &
                       (cl['flux'][-1] > 0)) | is_trap
            if is_trap:
                try:
                    sia_thick = sia_thickness_via_optim_wt(slope, width,
                                                           flux, 
                                                           rel_h,
                                                           a_factor,
                                                           shape='trapezoid',
                                                           t_lambda=t_lambda,
                                                           glen_a=glen_a,
                                                           fs=fs)
                except ValueError:
                    # no solution error - we do with rect
                    sia_thick = sia_thickness_via_optim_wt(slope, width,
                                                           flux, 
                                                           rel_h,
                                                           a_factor,
                                                           shape='rectangular',
                                                           glen_a=glen_a,
                                                           fs=fs)
        return fl['thick'] - sia_thick

    abs_min = optimize.minimize(to_minimize, [1.01], bounds=((1.01, 1e7), ),
                                tol=1e-1)
    # if not abs_min['success']:
        # raise RuntimeError('Could not find the absolute minimum in calving '
                           # 'flux optimization: {}'.format(abs_min))
    min_wl = -thick0
    success = 0
    step = th if th > thick0 else thick0
    step = step/100
    if thick0 > 10*th:
        while abs_min['fun'] > 0 or success == 0:
            if water_level <= min_wl:
                break
            abs_min = optimize.minimize(to_minimize, [1.01], bounds=((1.01, 1e7), ),
                                        tol=1e-1)
            abs_min0 = abs_min['x'][0]
               
            try:
                opt = optimize.brentq(to_minimize, abs_min0, 1e7)
                success = 1
            except ValueError:         
                water_level -= step
                pass
        
        while abs_min['fun'] > 0 or success == 0:
            if water_level >= th:
                break
            abs_min = optimize.minimize(to_minimize, [1.01], bounds=((1.01, 1e7), ),
                                        tol=1e-1)                                      
            abs_min0 = abs_min['x'][0]
            try:
                opt = optimize.brentq(to_minimize, abs_min0, 1e7)
                success = 1
            except ValueError:         
                water_level += step
                pass
        

    else:
        while abs_min['fun'] > 0 or success == 0:
            if water_level >= th:
                break
            abs_min = optimize.minimize(to_minimize, [1.01], bounds=((1.01, 1e7), ),
                                        tol=1e-1)                                      
            abs_min0 = abs_min['x'][0]
            try:
                opt = optimize.brentq(to_minimize, abs_min0, 1e7)
                success = 1
            except ValueError:         
                water_level += step
                pass
        
        while abs_min['fun'] > 0 or success == 0:
            if water_level <= min_wl:
                break
            abs_min = optimize.minimize(to_minimize, [1.01], bounds=((1.01, 1e7), ),
                                        tol=1e-1)                               
            abs_min0 = abs_min['x'][0]
            try:
                opt = optimize.brentq(to_minimize, abs_min0, 1e7)
                success = 1
            except ValueError:         
                water_level -= step
                pass

    # Give the flux to the inversion and recompute
    # This is the thick guaranteeing OGGM Flux = Calving Law Flux
    try:
        rel_h = opt
        f_b = cls['hgt'][-1] - water_level
        thick = ((1028/900)*rel_h*f_b) / ((1028/900) * rel_h - rel_h + 1)
        water_depth = thick - f_b
    except:
        log.workflow('inversion routine not working as expected. '
                     'We just take random values and proceed...')
        f_b = utils.clip_max(1*calving_k,0.9)
        water_level = th - f_b
        opt = 1e2#1/(1-(900/1028))
        rel_h = opt        
        thick = ((1028/900)*rel_h*f_b) / ((1028/900) * rel_h - rel_h + 1)
        water_depth = thick - f_b

    gdir.inversion_calving_rate = 0
    with utils.DisableLogger():
        climate.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
                                        mb_years=mb_years)
        prepare_for_inversion(gdir)
        v_ref = mass_conservation_inversion_wt(gdir, water_level=water_level,
                                               glen_a=glen_a, fs=fs,
                                               min_rel_h=opt)
        #tasks.filter_inversion_output(gdir)
    # Store for statistics
    gdir.add_to_diagnostics('volume_before_calving', v_ref)

    cl = gdir.read_pickle('inversion_output')[-1]
    log.workflow('hgt, thick before: {}, {}'.format(cls['hgt'][-1], cl['thick'][-1]))
    out = calving_flux_from_depth(gdir, water_level=water_level, k=calving_k,
                                  water_depth=water_depth) 
                                  
    f_calving = out['flux']
    gdir.inversion_calving_rate = f_calving
    with utils.DisableLogger():
        climate.apparent_mb_from_any_mb(gdir, mb_model=mb_model,
                                        mb_years=mb_years)
        prepare_for_inversion(gdir)
        mass_conservation_inversion_wt(gdir, water_level=water_level,
                                       glen_a=glen_a, fs=fs, min_rel_h=opt)
        #tasks.filter_inversion_output(gdir)
    cl = gdir.read_pickle('inversion_output')[-1]
    out = calving_flux_from_depth(gdir, water_level=water_level, k=calving_k)  

    out = calving_flux_from_depth(gdir, water_level=water_level, k=calving_k)
    fl = gdir.read_pickle('inversion_flowlines')[-1]
    cl = gdir.read_pickle('inversion_output')[-1]
    log.workflow('thick after: {}, {}'.format(cl['thick'][-1], out['thick']))
    f_calving = (fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9 /
                 cfg.PARAMS['ice_density'])
    
    log.workflow('({}) found frontal thickness, water depth, free-board, water level of '
                 '{}, {}, {}, {}'.format(gdir.rgi_id, out['thick'], out['water_depth'],out['free_board'], out['water_level']))
    log.workflow('({}) calving (law) flux of {} ({})'.format(gdir.rgi_id, f_calving, out['flux']))
    # Store results
    odf = dict()
    odf['calving_flux'] = f_calving
    odf['calving_rate_myr'] = f_calving * 1e9 / (out['thick'] * out['width'])
    odf['calving_law_flux'] = out['flux']
    odf['calving_water_level'] = out['water_level']
    odf['calving_inversion_k'] = out['inversion_calving_k']
    odf['calving_front_slope'] = slope
    odf['calving_front_water_depth'] = out['water_depth']
    odf['calving_front_free_board'] = out['free_board']
    odf['calving_front_thick'] = out['thick']
    odf['calving_front_width'] = out['width']
    for k, v in odf.items():
        gdir.add_to_diagnostics(k, v)

    return odf