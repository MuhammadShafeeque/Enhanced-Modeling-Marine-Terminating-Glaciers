""" Functions for inverting and running water-terminating glaciers in OGGM """
import logging
import warnings
import numpy as np
import pandas as pd
import copy
from scipy import optimize

from oggm.utils import entity_task
from oggm.core import inversion
from oggm import utils
from oggm.core.flowline import FileModel
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   PastMassBalance,
                                   AvgClimateMassBalance,
                                   RandomMassBalance)
import oggm.cfg as cfg
from oggm.core.inversion import (prepare_for_inversion, _inversion_poly, 
                                 _inversion_simple, _vol_below_water,
                                 calving_flux_from_depth)
# Constants
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR
from oggm.cfg import G, GAUSSIAN_KERNEL

from FluxModel import FluxBasedModelWaterFront
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)


@entity_task(log)
def flowline_model_run_wt(gdir, output_filesuffix=None, mb_model=None,
                          ys=None, ye=None, zero_initial_glacier=False,
                          init_model_fls=None, store_monthly_step=False,
                          store_model_geometry=None, water_level=None,
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
    water_level : float
        the water level. It should be zero m a.s.l, but:
        - sometimes the frontal elevation is unrealistically high (or low).
        - lake terminating glaciers
        - other uncertainties
        The default is to take the water level obtained from the ice
        thickness inversion.
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
     """
    mb_elev_feedback = kwargs.get('mb_elev_feedback', 'annual')
    if store_monthly_step and (mb_elev_feedback == 'annual'):
        warnings.warn("The mass-balance used to drive the ice dynamics model "
                      "is updated yearly. If you want the output to be stored "
                      "monthly and also reflect reflect monthly processes,"
                      "set store_monthly_step=True and "
                      "mb_elev_feedback='monthly'. This is not recommended "
                      "though: for monthly MB applications, we recommend to "
                      "use the `run_with_hydro` task.")

    if cfg.PARAMS['use_inversion_params_for_run']:
        diag = gdir.get_diagnostics()
        fs = diag.get('inversion_fs', cfg.PARAMS['fs'])
        glen_a = diag.get('inversion_glen_a', cfg.PARAMS['glen_a'])
    else:
        fs = cfg.PARAMS['fs']
        glen_a = cfg.PARAMS['glen_a']

    kwargs.setdefault('fs', fs)
    kwargs.setdefault('glen_a', glen_a)

    if store_model_geometry is None:
        store_model_geometry = cfg.PARAMS['store_model_geometry']

    if store_model_geometry:
        geom_path = gdir.get_filepath('model_geometry',
                                      filesuffix=output_filesuffix,
                                      delete=True)
    else:
        geom_path = False

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
                                       
    model = FluxBasedModelWaterFront(fls, mb_model=mb_model, y0=ys,
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
                                  store_monthly_step=store_monthly_step)

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
                             store_monthly_step=False,
                             store_model_geometry=None,
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

    # Take from rgi date if not set yet
    if ys is None:
        try:
            ys = gdir.rgi_date.year
        except AttributeError:
            ys = gdir.rgi_date
        # The RGI timestamp is in calendar date - we convert to hydro date,
        # i.e. 2003 becomes 2004 if hydro_month is not 1 (January)
        # (so that we don't count the MB year 2003 in the simulation)
        # See also: https://github.com/OGGM/oggm/issues/1020
        # even if hydro_month is 1, we prefer to start from Jan 2004
        # as in the alps the rgi is from Aug 2003
        ys += 1

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
                              init_model_fls=init_model_fls,
                              zero_initial_glacier=zero_initial_glacier,
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
    
def sia_thickness_via_optim_wt(slope, width, flux, water_depth, 
                                      shape='rectangular',
                                      glen_a=None, fs=None, t_lambda=None):
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
            out.append(sia_thickness_via_optim(sl, w, f, shape=s,
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
        u = ((h ** (n + 1)) * fd * rhogh + ((h ** n) / (utils.clip_min(10, h -
             (1028/rho) * water_depth))) * fs * rhogh)
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

def sia_thickness_wt(slope, width, flux, water_depth, f_b, shape='rectangular',
                  glen_a=None, fs=None, shape_factor=None):
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
    a0 = - flux_a0 / ((rho * cfg.G * slope) ** 3 * fd)
    #a3 = fs / fd
    a3 = (fs / fd) * ((water_depth+f_b) / utils.clip_min(10,(water_depth+f_b) -
                      (1028/rho)*water_depth))

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
                                filesuffix='', water_level=None, t_lambda=None):
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

    out_volume = 0.

    cls = gdir.read_pickle('inversion_input')
    for cl in cls:
        # Clip slope to avoid negative and small slopes
        slope = cl['slope_angle']
        slope = utils.clip_array(slope, min_slope, np.pi/2.)

        # Glacier width
        a0s = - cl['flux_a0'] / ((rho*cfg.G*slope)**3*fd)
        sf = np.ones(slope.shape)  #  Default shape factor is 1
        out_thick = _compute_thick_wt(a0s, a3, cl['flux_a0'], sf, _inv_function)
        w = cl['width']
        bed_h = cl['hgt'] - out_thick
        if water_level is None:
            water_depth = utils.clip_min(0,-bed_h)
        else:
            water_depth = utils.clip_min(0,-bed_h + water_level)
        a3s = ((fs / fd) * (out_thick / utils.clip_min(10,(out_thick -
               (1028/rho)*water_depth))))

        if sf_func is not None:

            # Start iteration for shape factor with first guess of 1
            i = 0
            sf_diff = np.ones(slope.shape)

            # Some hard-coded factors here
            sf_tol = 1e-2
            max_sf_iter = 20

            while i < max_sf_iter and np.any(sf_diff > sf_tol):
                out_thick = _compute_thick_wt(a0s, a3s, cl['flux_a0'], sf,
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

        out_thick = _compute_thick_wt(a0s, a3s, cl['flux_a0'], sf, 
                                      _inv_function) 

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
                                                           water_depth[i],
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
                                                           water_depth[i],
                                                           shape='rectangular',
                                                           glen_a=glen_a,
                                                           fs=fs)
                    is_rect[i] = True
                    is_trap[i] = False
                    volume[i] = out_thick[i] * w[i] * cl['dx']

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
                                                        bed_shape, out_thick,
                                                        w,
                                                        cl['is_rectangular'],
                                                        cl['is_trapezoid'],
                                                        fac, t_lambda,
                                                        cl['dx'], 0)
                if water_level is not None and np.any(bed_h < water_level):
                    cl['volume_bwl'] = _vol_below_water(cl['hgt'], bed_h,
                                                        bed_shape, out_thick,
                                                        w,
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


@entity_task(log, writes=['diagnostics'])
def find_inversion_calving(gdir, water_level=None, fixed_water_depth=None,
                           glen_a=None, fs=None, min_mu_star_frac=None):
    """Optimized search for a calving flux compatible with the bed inversion.

    See Recinos et al 2019 for details.

    Parameters
    ----------
    water_level : float
        the water level. It should be zero m a.s.l, but:
        - sometimes the frontal elevation is unrealistically high (or low).
        - lake terminating glaciers
        - other uncertainties
        With this parameter, you can produce more realistic values. The default
        is to infer the water level from PARAMS['free_board_lake_terminating']
        and PARAMS['free_board_marine_terminating']
    fixed_water_depth : float
        fix the water depth to an observed value and let the free board vary
        instead.
    glen_a : float, optional
    fs : float, optional
    min_mu_star_frac : float, optional
        fraction of the original (non-calving) mu* you are ready to allow
        for. Defaults cfg.PARAMS['calving_min_mu_star_frac'].
    """
    from oggm.core import climate
    from oggm.exceptions import MassBalanceCalibrationError
    
    if not gdir.is_tidewater or not cfg.PARAMS['use_kcalving_for_inversion']:
        # Do nothing
        return

    if min_mu_star_frac is None:
        min_mu_star_frac = cfg.PARAMS['calving_min_mu_star_frac']

    # Let's start from a fresh state
    gdir.inversion_calving_rate = 0

    with utils.DisableLogger():
        climate.local_t_star(gdir)
        climate.mu_star_calibration(gdir)
        prepare_for_inversion(gdir)
        v_ref = mass_conservation_inversion_wt(gdir, 
                                            water_level=water_level,
                                            glen_a=glen_a, fs=fs)

    # We have a stop condition on mu*
    prev_params = gdir.read_json('local_mustar')
    mu_star_orig = np.min(prev_params['mu_star_per_flowline'])

    # Store for statistics
    gdir.add_to_diagnostics('volume_before_calving', v_ref)
    gdir.add_to_diagnostics('mu_star_before_calving', mu_star_orig)

    # Get the relevant variables
    cls = gdir.read_pickle('inversion_input')[-1]
    slope = cls['slope_angle'][-1]
    width = cls['width'][-1]

    # Stupidly enough the slope is clipped in the OGGM inversion, not
    # in prepro - clip here
    min_slope = 'min_slope_ice_caps' if gdir.is_icecap else 'min_slope'
    min_slope = np.deg2rad(cfg.PARAMS[min_slope])
    slope = utils.clip_array(slope, min_slope, np.pi / 2.)

    # Check that water level is within given bounds
    if water_level is None:
        th = cls['hgt'][-1]
        if gdir.is_lake_terminating:
            water_level = th - cfg.PARAMS['free_board_lake_terminating']
        else:
            vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
            water_level = utils.clip_scalar(0, th - vmax, th - vmin)

    # The functions all have the same shape: they decrease, then increase
    # We seek the absolute minimum first
    def to_minimize(h):
        if fixed_water_depth is not None:
            fl = calving_flux_from_depth(gdir, thick=h,
                                         water_level=water_level,
                                         water_depth=fixed_water_depth,
                                         fixed_water_depth=True)
        else:
            fl = calving_flux_from_depth(gdir, water_level=water_level,
                                         water_depth=h)

        flux = fl['flux'] * 1e9 / cfg.SEC_IN_YEAR
        f_b = fl['free_board']
        sia_thick = sia_thickness_wt(slope, width, flux, h, f_b, glen_a=glen_a, 
                                     fs=fs)
        return fl['thick'] - sia_thick

    abs_min = optimize.minimize(to_minimize, [1], bounds=((1e-4, 1e4), ),
                                tol=1e-1)
  
    if not abs_min['success']:
        raise RuntimeError('Could not find the absolute minimum in calving '
                           'flux optimization: {}'.format(abs_min))

    if abs_min['fun'] > 0:
        # This happens, and means that this glacier simply can't calve
        # This is an indicator for physics not matching, often a unrealistic
        # slope of free-board. We just set the water level to the surface
        # height at the front and try again to get an estimate.
        water_level = th
        out = calving_flux_from_depth(gdir, water_level=water_level)                                                
        opt = out['thick']

    # OK, we now find the zero between abs min and an arbitrary high front
    else:
        abs_min = abs_min['x'][0]
        opt = optimize.brentq(to_minimize, abs_min, 1e4)

    # This is the thick guaranteeing OGGM Flux = Calving Law Flux
    # Let's see if it results in a meaningful mu_star

    # Give the flux to the inversion and recompute
    if fixed_water_depth is not None:
        out = calving_flux_from_depth(gdir, water_level=water_level, thick=opt,
                                      water_depth=fixed_water_depth,
                                      fixed_water_depth=True)
        f_calving = out['flux']
    else:
        out = calving_flux_from_depth(gdir, water_level=water_level,
                                      water_depth=opt)
        f_calving = out['flux']   

    gdir.inversion_calving_rate = f_calving

    with utils.DisableLogger():
        # We accept values down to zero before stopping
        min_mu_star = mu_star_orig * min_mu_star_frac

        # At this step we might raise a MassBalanceCalibrationError
        try:
            climate.local_t_star(gdir, clip_mu_star=False,
                                 min_mu_star=min_mu_star,
                                 continue_on_error=False,
                                 add_to_log_file=False)
            df = gdir.read_json('local_mustar')
        except MassBalanceCalibrationError as e:
            assert 'mu* out of specified bounds' in str(e)
            # When this happens we clip mu*
            climate.local_t_star(gdir, clip_mu_star=True,
                                 min_mu_star=min_mu_star)
            df = gdir.read_json('local_mustar')

        climate.mu_star_calibration(gdir, min_mu_star=min_mu_star)
        prepare_for_inversion(gdir)
        mass_conservation_inversion_wt(gdir, water_level=water_level,
                                    glen_a=glen_a, fs=fs)

    if fixed_water_depth is not None:
        out = calving_flux_from_depth(gdir, water_level=water_level,
                                      water_depth=fixed_water_depth,
                                      fixed_water_depth=True)
    else:
        out = calving_flux_from_depth(gdir, water_level=water_level)

    fl = gdir.read_pickle('inversion_flowlines')[-1]
    f_calving = (fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9 /
                 cfg.PARAMS['ice_density'])

    log.info('({}) find_inversion_calving_from_any_mb: found calving flux of '
             '{:.03f} km3 yr-1'.format(gdir.rgi_id, f_calving))

    # Store results
    odf = dict()
    odf['calving_flux'] = f_calving
    odf['calving_rate_myr'] = f_calving * 1e9 / (out['thick'] * out['width'])
    odf['calving_mu_star'] = df['mu_star_glacierwide']
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
