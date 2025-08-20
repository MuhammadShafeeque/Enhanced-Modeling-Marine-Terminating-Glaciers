""" Functions for inverting and running marine-terminating glaciers in OGGM """
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

from oggm_marine_terminating.flux_model import FluxBasedModelMarineFront
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)

def calculate_water_depth(bed_elevation, water_level):
    """Calculate water depth at glacier bed.
    
    Parameters
    ----------
    bed_elevation : ndarray
        Elevation of the glacier bed (m a.s.l.)
    water_level : float
        Elevation of the water level (m a.s.l.)
        
    Returns
    -------
    depth : ndarray
        Water depth at each point (m), with values < 0 clipped to 0
    """
    return utils.clip_min(0, water_level - bed_elevation)

@entity_task(log)
def flowline_model_run_mt(gdir, output_filesuffix=None, mb_model=None,
                       ys=None, ye=None, zero_initial_glacier=False,
                       init_model_fls=None, store_monthly_step=False,
                       fixed_geometry_spinup_yr=None,
                       store_model_geometry=None,
                       store_fl_diagnostics=None,
                       water_level=None,
                       evolution_model=FluxBasedModelMarineFront, stop_criterion=None,
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
    evolution_model : class
        the FlowlineModel class to use (default: FluxBasedModelMarineFront)
    stop_criterion : func
        a function which, if returns True, stops the evolution
    water_level : float
        the water level (for marine-terminating glaciers)
    kwargs : dict
        kwargs to pass to the FlowlineModel instance
    Returns
    -------
    :py:class:`oggm.core.flowline.FlowlineModel`
    """
    
    # Read flowlines
    if init_model_fls is None:
        fls = gdir.read_pickle('model_flowlines')
    else:
        fls = copy.deepcopy(init_model_fls)
        
    # Default start and end year
    if ys is None:
        ys = cfg.PARAMS['ys']
    if ye is None:
        ye = cfg.PARAMS['ye']
        
    # Default water level is zero
    if water_level is None:
        water_level = 0
        
    # Initialize model
    if init_model_filesuffix is not None:
        # Recursive call from init
        flowline_model_run_mt(gdir, mb_model=mb_model, ys=ys, ye=ys,
                            init_model_filesuffix=init_model_filesuffix,
                            init_model_yr=init_model_yr,
                            evolution_model=evolution_model,
                            water_level=water_level,
                            **kwargs)
        # Now get the init state
        model = FileModel(gdir.get_filepath('model_run', filesuffix='_from_init'))
        fls = model.fls
        
    if zero_initial_glacier:
        for fl in fls:
            fl.thick = fl.thick * 0.
            
    # Overrides for certain parameters if provided
    if store_model_geometry is None:
        store_model_geometry = cfg.PARAMS['store_model_geometry']
    if store_fl_diagnostics is None:
        store_fl_diagnostics = cfg.PARAMS['store_fl_diagnostics']
        
    # Mass balance model
    if mb_model is None:
        mb_model = MultipleFlowlineMassBalance(gdir, fls=fls)
        
    # Create the model
    model = evolution_model(fls, mb_model=mb_model, y0=ys,
                          water_level=water_level,
                          **kwargs)
    
    # Run with step scheme
    if fixed_geometry_spinup_yr is not None:
        # For now we just use a simple spinup/constant geometry option
        _simple_spinup(model, fixed_geometry_spinup_yr, ye)
    else:
        # Go
        model.run_until_and_store(ye, stop_criterion=stop_criterion,
                                  output_filesuffix=output_filesuffix,
                                  store_monthly_step=store_monthly_step,
                                  store_model_geometry=store_model_geometry,
                                  store_fl_diagnostics=store_fl_diagnostics)
    return model

@entity_task(log)
def run_from_climate_data_mt(gdir, ys=None, ye=None, min_ys=None, max_ys=None,
                          fixed_geometry_spinup_yr=None,
                          output_filesuffix=None,
                          climate_filename='climate_historical',
                          climate_input_filesuffix='',
                          evolution_model=FluxBasedModelMarineFront,
                          init_model_filesuffix=None,
                          init_model_yr=None,
                          bias=None, 
                          **kwargs):
    """Runs the flowline model for a specific glacier with climate input from e.g CRU or a GCM.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year
    ye : int
        end year
    min_ys : int
        if you want to impose a minimum start year, regardless of the climate
        data available.
    max_ys : int
        if you want to impose a maximum start year, regardless of the climate
        data available.
    fixed_geometry_spinup_yr : int
        if set to an integer, the model will artificially prolongate
        all outputs (mass-balance, frontal ablation) at the respective starting
        year and keep the geometry fixed for that given period. The actual
        model run starts after that date.
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    evolution_model : class
        the FlowlineModel class to use (default: FluxBasedModelMarineFront)
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    bias : float
        bias of the mb model (applied on the chosen mb model). Note that
        this bias is *substracted* from the computed MB, i.e. a negative bias
        will yield more positive MB.
    kwargs : dict
        kwargs to pass to the FlowlineModel instance
    """
    
    if init_model_filesuffix is not None:
        fls = gdir.read_pickle('model_flowlines')
        if output_filesuffix is None:
            output_filesuffix = 'from_init'
        else:
            output_filesuffix = 'from_init_' + output_filesuffix
            
    # Defaults
    if ys is None:
        ys = cfg.PARAMS['ys']
    if ye is None:
        ye = cfg.PARAMS['ye']
        
    # Check dates integrity
    rtwo = gdir.read_json(climate_filename, filesuffix=climate_input_filesuffix)
    if 'climate_historical' in climate_filename:
        # Historical run
        ci = gdir.get_climate_info(input_filesuffix=climate_input_filesuffix)
        yrp = ci['baseline_yr_0']
    else:
        # GCM run
        yrp = rtwo.get('baseline_yr_0', 0)
        
    # Historical data are typically 1801-2018
    if min_ys is not None:
        ys = max(ys, min_ys)
        
    if max_ys is not None:
        ys = min(ys, max_ys)
        
    # Reference glaciers starting in yrp
    if ys < yrp:
        # These glaciers need a special handling
        # We use their state at yrp as starting date
        log.info('(%s) starting from baseline_yr_0 since ys < baseline_yr_0',
                gdir.rgi_id)
        run_from_climate_data_mt(gdir, ys=yrp, ye=ye,
                              climate_filename=climate_filename,
                              climate_input_filesuffix=climate_input_filesuffix,
                              evolution_model=evolution_model,
                              output_filesuffix=output_filesuffix,
                              init_model_filesuffix=init_model_filesuffix,
                              init_model_yr=init_model_yr,
                              bias=bias,
                              **kwargs)
        
        return
        
    # Mass balance model
    mb = PastMassBalance(gdir, bias=bias,
                        filename=climate_filename,
                        input_filesuffix=climate_input_filesuffix)
                        
    # Flowline model
    return flowline_model_run_mt(gdir, output_filesuffix=output_filesuffix,
                             mb_model=mb, ys=ys, ye=ye,
                             evolution_model=evolution_model,
                             fixed_geometry_spinup_yr=fixed_geometry_spinup_yr,
                             init_model_filesuffix=init_model_filesuffix,
                             init_model_yr=init_model_yr,
                             **kwargs)

@entity_task(log)
def run_random_climate_mt(gdir, nyears=1000, y0=None, halfsize=15,
                       bias=None, seed=None, temperature_bias=None,
                       precipitation_factor=None, evolution_model=FluxBasedModelMarineFront,
                       climate_filename='climate_historical',
                       climate_input_filesuffix='',
                       output_filesuffix='', init_model_filesuffix=None,
                       init_model_yr=None, zero_initial_glacier=False,
                       unique_samples=False, **kwargs):
    """Runs the flowline model for a specific glacier with random MB.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int
        length of the simulation
    y0 : int
        central year of the random climate period. The default is to be
        centred on t*.
    halfsize : int
        the half-size of the time window (window size = 2 * halfsize + 1)
    bias : float
        bias of the mb model (applied on the chosen mb model residual)
    seed : int
        seed for the random generator - for reproducibility
    temperature_bias : float
        add a bias to the temperature timeseries
    precipitation_factor: float
        multiply a factor to the precipitation time series
    evolution_model : class
        the FlowlineModel to use
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
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    unique_samples: bool
        if true, chosen random mass-balance years will only be available once
        per random climate period-halfsize. That is, if the climate period is
        2000-2010, the default is to sample these years randomly with
        repetitions (each year has a probability of 1/11 to be chosen at each
        time step). Setting this to True will sample the climate period as a
        whole (i.e. one permutation of these 11 years), thus ensuring that the
        10-yr statistics of the random climate are the same as the original
        series.
    kwargs : dict
        kwargs to pass to the FlowlineModel instance
    """
    
    mb = RandomMassBalance(gdir, mb_model_class=PastMassBalance,
                         y0=y0, halfsize=halfsize,
                         bias=bias, seed=seed,
                         filename=climate_filename,
                         input_filesuffix=climate_input_filesuffix,
                         unique_samples=unique_samples,
                         temperature_bias=temperature_bias,
                         precipitation_factor=precipitation_factor)
                         
    return flowline_model_run_mt(gdir, output_filesuffix=output_filesuffix,
                             mb_model=mb, ys=0, ye=nyears,
                             evolution_model=evolution_model,
                             zero_initial_glacier=zero_initial_glacier,
                             init_model_filesuffix=init_model_filesuffix,
                             init_model_yr=init_model_yr,
                             **kwargs)

@entity_task(log)
def mass_conservation_inversion_mt(gdir, glen_a=None, fs=0., write_out=True,
                                 filesuffix='', water_level=None,
                                 apply_thickness_change=True):
    """Compute the glacier thickness along the flowlines

    This is a copy of the default OGGM inversion function, with special handling
    for water termininating glaciers.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    glen_a : float
        Glen's creep parameter A. Defaults to cfg.PARAMS.
    fs : float
        Oerlemans sliding parameter f. Defaults to 0.
    write_out : bool
        default behavior is to compute the thickness and write the
        results in the pickle. Set to False in order to spare time
        during calibration.
    filesuffix : str
        add a suffix to the output file
    water_level : float
        the water level (for marine-terminating glaciers)
    apply_thickness_change : bool
        whether to use the glacier_grid['thickness_change'] to adjust the 
        thickness of marine-terminating flowlines according to elevation change
        if available.
    """

    # Defaults
    if glen_a is None:
        glen_a = cfg.PARAMS['glen_a']
    if water_level is None:
        water_level = 0
        
    # Check input
    if 'inversion_flowlines' not in gdir.read_pickle('geometries'):
        raise InvalidWorkflowError('Inversion preprocessing not run.')
    
    # Prepare for the inversion
    towrite = prepare_for_inversion(gdir)
    
    # Thickness change rate (dn/dt) to apply?
    if apply_thickness_change:
        thickness_change = None
        if (hasattr(cfg, 'BASENAMES') and
            cfg.BASENAMES.get('thickness_change')):
            fp = gdir.get_filepath('thickness_change')
            if os.path.exists(fp):
                with utils.ncDataset(fp) as nc:
                    thickness_change = nc.variables['thickness_change'][:]
    else:
        thickness_change = None
    
    # Fill with other variables
    towrite['is_tidewater'] = gdir.is_tidewater
    towrite['is_lake_terminating'] = gdir.is_lake_terminating
    towrite['water_level'] = water_level
    towrite['glen_a'] = glen_a
    towrite['fs'] = fs
    towrite['inversion_glen_a'] = cfg.PARAMS['inversion_glen_a']
    towrite['inversion_fs'] = cfg.PARAMS['inversion_fs']
    
    if gdir.is_tidewater or gdir.is_lake_terminating:
        # Recompute thickness with current A and FS
        # We're being lazy here and assume that calving_k will remain constant
        _w = gdir.read_pickle('inversion_params')
        cfg.PARAMS['inversion_calving_k'] = _w['calving_k']
        # We need to compute everything again because A and FS might have changed
        mass_conservation_inversion(gdir, glen_a=towrite['inversion_glen_a'],
                                 fs=towrite['inversion_fs'],
                                 write_out=False)
        _w = gdir.read_pickle('inversion_output')
        towrite['calving_k'] = _w['calving_k']
        towrite['calving_law_flux'] = _w['calving_law_flux']
        
    # Go
    invert_parabolic_bed(gdir, towrite, thickness_change=thickness_change)
    
    if write_out:
        gdir.write_pickle(towrite, 'inversion_output', filesuffix=filesuffix)
        
    return towrite

@entity_task(log)
def sia_thickness_mt(gdir, dx=None, write=True, filesuffix=''):
    """Computes the 'SIA thickness' of the glacier.

    This is the glacier thickness that would result from the SIA in a
    world without numerical models. This is very useful for e.g. ice
    volume estimates or ice thickness maps. The output is a glacier-wide
    thickness map and the volume.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    dx : float
        the map resolution (if not gdir.grid.dx)
    write: bool
        default behavior is to compute the thickness and write the
        results as a NetCDF file in the glacier directory. Set to False in
        order to speed up calculation when the result is only needed in object
        form.
    filesuffix : str
        add a suffix to the output file
    """
    # Handle different cases
    return sia_thickness_via_optim_mt(gdir, dx=dx, write=write, filesuffix=filesuffix)

@entity_task(log)
def sia_thickness_via_optim_mt(gdir, dx=None, write=True, filesuffix=''):
    """Optimized version of sia_thickness for tidewater glaciers.

    This is useful for tidewater glaciers, where the water depth
    is known.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    dx : float
        the map resolution (if not gdir.grid.dx)
    write: bool
        default behavior is to compute the thickness and write the
        results as a NetCDF file in the glacier directory. Set to False in
        order to speed up calculation when the result is only needed in object
        form.
    filesuffix : str
        add a suffix to the output file
    """
    # Simply call the task in OGGM - we just have this as a convenience wrapper
    # since it exists in the model_code version
    return inversion.sia_thickness_via_optim(gdir, dx=dx, write=write, 
                                           filesuffix=filesuffix)

def _simple_spinup(model, spinup_yr, ye):
    """Run a simplified spinup for a model simulation.

    Parameters
    ----------
    model : :py:class:`core.flowline.FlowlineModel`
        the model to run
    spinup_yr : int
        year until which the model should run with fixed geometry
    ye : int
        year for simulation stop
    """
    
    # Spinup with constant geometry for a few years
    model.run_until(spinup_yr, stop_criterion=False)
    
    # Now run the model normally
    model.run_until_and_store(ye)

def invert_parabolic_bed(gdir, towrite, thickness_change=None):
    """Actual thickness computation (parabolic bed).

    This is separated from the main task so we can test different
    implementations by overriding only this.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    towrite : dict
        the dictionary to write out
    thickness_change : numpy.ndarray
        If given, thickness change to apply to the flowlines.
    """
    
    # Check input
    if not towrite['is_tidewater'] and not towrite['is_lake_terminating']:
        # Do the standard routine for non-tidewater glaciers
        _invert_parabolic_bed(gdir, towrite, thickness_change=thickness_change)
        return
        
    # Ok we have a tidewater glacier
    inv_fs = towrite['inversion_fs']
    inv_glen_a = towrite['inversion_glen_a']
    fs = towrite['fs']
    glen_a = towrite['glen_a']
    gled_a_ratio = glen_a / inv_glen_a
    
    # Get the previous calving flux and k
    if 'calving_law_flux' in towrite:
        calving_law_flux = towrite['calving_law_flux']
        calving_k = towrite['calving_k']
    else:
        # we still didn't compute it
        calving_law_flux = None
        calving_k = None
        
    # Compute a consensus thickness before applying another calving law
    towrite['thick'] = []
    for cl, fl in zip(towrite['cls'], towrite['fls']):
        w = cl['width']
        a0s = cl['flux'] / (gled_a_ratio * cl['glen_a']) * cl['shape']
        if np.any(~np.isfinite(a0s)):
            log.info('(%s) has non-finite coefficients', gdir.rgi_id)
            raise InvalidParamsError('({}) has non-finite coefficients'.format(gdir.rgi_id))
        thk = np.zeros(len(fl.surface_h))
        for i, (a0, q) in enumerate(zip(a0s, cl['flux'])):
            if q > 0.:
                # Parabolic
                thk[i] = np.sqrt(a0) * np.sqrt(cl['shape'])
        towrite['thick'].append(thk)
        
    # Compute a calving flux based on water depth
    if calving_law_flux is None:
        # OK so this is a first guess
        cl, fl = towrite['cls'][-1], towrite['fls'][-1]
        depth = calculate_water_depth(fl.bed_h, towrite['water_level'])
        calving_law_flux = calving_flux_from_depth(gdir, depth)
        calving_k = cfg.PARAMS['inversion_calving_k']
        
    # Now recompute the flux, thickness and velocity with calving
    towrite['calving_law_flux'] = calving_law_flux
    towrite['calving_k'] = calving_k
    
    # Added to accommodate the additional flux_mb term with v1.6.0
    if towrite['ela_h'] > 0 and 'flux_mb' in towrite['cls'][0]:
        # This is a tidewater glacier with some modifications along the length
        if np.min(towrite['fls'][-1].bed_h) > 0:
            # This is a lake terminating, let's set calving flux to zero
            # This might have happened because the user provided their own flowlines
            # or the simplified flowlines don't reach below sea level
            calving_law_flux = 0
            towrite['calving_law_flux'] = 0
            towrite['calving_k'] = 0
            _invert_parabolic_bed(gdir, towrite, thickness_change=thickness_change)
            return
        # Add the correct flux_mb to the calving flux
        flux_mbs = towrite['cls'][-1]['flux_mb']
        # Only take the ablation zone
        if np.any(flux_mbs < 0):
            # Careful: we put the unit back to yr here!
            m3_ice_yr = np.sum(flux_mbs[flux_mbs < 0]) * SEC_IN_YEAR
            calving_law_flux += m3_ice_yr
    
    for i, (cl, fl) in enumerate(zip(towrite['cls'], towrite['fls'])):
        # Add the flux and recompute thickness
        if i == len(towrite['fls']) - 1:
            # Last one includes calving
            cl['flux'][-1] += calving_law_flux
            
        # Everything from here is the same as above
        flux = cl['flux']
        w = cl['width']
        a0s = flux / (gled_a_ratio * cl['glen_a']) * cl['shape']
        if np.any(~np.isfinite(a0s)):
            log.info('(%s) has non-finite coefficients', gdir.rgi_id)
            raise InvalidParamsError('({}) has non-finite coefficients'.format(gdir.rgi_id))
        thk = np.zeros(len(fl.surface_h))
        for j, (a0, q) in enumerate(zip(a0s, flux)):
            if q > 0.:
                # Parabolic
                thk[j] = np.sqrt(a0) * np.sqrt(cl['shape'])
        towrite['thick'][i] = thk

def _invert_parabolic_bed(gdir, towrite, thickness_change=None):
    """Actual thickness computation (for non-tidewater only).

    Simpler version of invert_parabolic_bed for non-tidewater glaciers.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    towrite : dict
        the dictionary to write out
    thickness_change : numpy.ndarray
        If given, thickness change to apply to the flowlines.
    """
    
    # Read the params
    inv_fs = towrite['inversion_fs']
    inv_glen_a = towrite['inversion_glen_a']
    fs = towrite['fs']
    glen_a = towrite['glen_a']
    
    # Check input
    gled_a_ratio = glen_a / inv_glen_a
    # Ratio of "sliding" to "non-sliding" velocity
    slid_r = fs / inv_fs
    slid_r = np.ones(len(towrite['fls'])) * slid_r
    # Adjust for shape (parabola, trapezoidal, rectangular)
    # Nominal width = 1 (factor in OGGM)
    # For rectangular: w = 1, A = 1
    # For trapezoidal: w = 1, A = 2/3
    # For parabolic: w = 1, A = 2/3
    # For W = 1 in the trapezoid (OGGM), then S = 1 and H = 1 and A = 2/3
    shape_a = []
    for fl in towrite['fls']:
        w = fl.widths
        assert np.all(w > 0)
        w = np.where(w < 1e-8, 1e-8, w)
        shape_a.append(np.ones(len(w)) * 3./2 * 1./w)
    
    # Go
    towrite['thick'] = []
    for cl, fl, sr, sa in zip(towrite['cls'], towrite['fls'], slid_r, shape_a):
        flux = cl['flux']
        # OGGM inversion 
        a0s = flux / ((gled_a_ratio + sr) * cl['u'] * cl['width'])
        # Get optimized flux shape and flux factors
        factors = _inversion_poly(fl.surface_h, fl.widths,
                               fs + (fl.surface_h * 0) + 1e-11)
        factors = np.where(factors < 1e-8, 1e-8, factors)
        # Thickness with scaling
        cl['a0'] = a0s
        cl['shape'] = sa
        
        thk = np.zeros(len(fl.surface_h))
        for i, (a0, q) in enumerate(zip(a0s, flux)):
            if q > 0.:
                # Parabola
                thk[i] = np.sqrt(a0 * factors[i])
        
        # Apply elevation change if available and requested
        if thickness_change is not None:
            # Start with what is available and applicable
            # thickness changes are relative to the present-day surface elevation
            dh = np.zeros(len(fl.surface_h))
            # Add the elevation dependent part
            fl_surface_h = fl.surface_h
            for idx, h in enumerate(fl_surface_h):
                # Index of the glacier grid to pick from
                i0 = np.argmin(np.abs(gdir.grid.y_coord - h))
                dh[idx] = thickness_change[i0]
            # Check if dh makes sense (avoid thinning below 0)
            dh = utils.clip_max(dh, thk - 5)
            # Check if dh[0] makes sense - sometimes at the very top of the flowline
            # There are these weird values which come from how we defined thickness_change
            # by elevation bins. dh shouldn't be -1000 just because there's a bug in
            # the data structure. 
            if dh[0] < -100 or dh[0] < -5 * dh[1]:
                dh[0] = dh[1]
            # Smooth
            dh = utils.gaussian_smoothing(dh, 4)
            # Apply dh
            thk = utils.clip_min(thk + dh, 0)
            
        towrite['thick'].append(thk)
