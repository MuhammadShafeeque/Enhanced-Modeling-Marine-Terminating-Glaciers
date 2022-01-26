""" Add a dynamical model for water-terminating glaciers to OGGM """

# External libs
import logging
import numpy as np
import os
import warnings

from oggm.core.flowline import FlowlineModel
import oggm.cfg as cfg
from oggm import utils

# Constants
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR
from oggm.cfg import G, GAUSSIAN_KERNEL

# Module logger
log = logging.getLogger(__name__)

class FluxBasedModelWaterFront(FlowlineModel):
    """The flowline model used by OGGM in production.

    It solves for the SIA along the flowline(s) using a staggered grid. It
    computes the *ice flux* between grid points and transports the mass
    accordingly (also between flowlines).

    This model is numerically less stable than fancier schemes, but it
    is fast and works with multiple flowlines of any bed shape (rectangular,
    parabolic, trapeze, and any combination of them).

    We test that it conserves mass in most cases, but not on very stiff cliffs.
    """

    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
                 fs=0., inplace=False, fixed_dt=None, cfl_number=None,
                 min_dt=None, flux_gate_thickness=None,
                 flux_gate=None, flux_gate_build_up=100,
                 do_kcalving=None, calving_k=None, calving_use_limiter=None,
                 calving_limiter_frac=None, water_level=None,
                 **kwargs):
        """Instanciate the model.

        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBalanceModel
            the mass-balance model
        y0 : int
            initial year of the simulation
        glen_a : float
            Glen's creep parameter
        fs : float
            Oerlemans sliding parameter
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        fixed_dt : float
            set to a value (in seconds) to prevent adaptive time-stepping.
        cfl_number : float
            Defaults to cfg.PARAMS['cfl_number'].
            For adaptive time stepping (the default), dt is chosen from the
            CFL criterion (dt = cfl_number * dx / max_u).
            To choose the "best" CFL number we would need a stability
            analysis - we used an empirical analysis (see blog post) and
            settled on 0.02 for the default cfg.PARAMS['cfl_number'].
        min_dt : float
            Defaults to cfg.PARAMS['cfl_min_dt'].
            At high velocities, time steps can become very small and your
            model might run very slowly. In production, it might be useful to
            set a limit below which the model will just error.
        is_tidewater: bool, default: False
            is this a tidewater glacier?
        is_lake_terminating: bool, default: False
            is this a lake terminating glacier?
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass-balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        flux_gate_thickness : float or array
            flux of ice from the left domain boundary (and tributaries).
            Units of m of ice thickness. Note that unrealistic values won't be
            met by the model, so this is really just a rough guidance.
            It's better to use `flux_gate` instead.
        flux_gate : float or function or array of floats or array of functions
            flux of ice from the left domain boundary (and tributaries)
            (unit: m3 of ice per second). If set to a high value, consider
            changing the flux_gate_buildup time. You can also provide
            a function (or an array of functions) returning the flux
            (unit: m3 of ice per second) as a function of time.
            This is overriden by `flux_gate_thickness` if provided.
        flux_gate_buildup : int
            number of years used to build up the flux gate to full value
        do_kcalving : bool
            switch on the k-calving parameterisation. Ignored if not a
            tidewater glacier. Use the option from PARAMS per default
        calving_k : float
            the calving proportionality constant (units: yr-1). Use the
            one from PARAMS per default
        calving_use_limiter : bool
            whether to switch on the calving limiter on the parameterisation
            makes the calving fronts thicker but the model is more stable
        calving_limiter_frac : float
            limit the front slope to a fraction of the calving front.
            "3" means 1/3. Setting it to 0 limits the slope to sea-level.
        water_level : float
            the water level. It should be zero m a.s.l, but:
            - sometimes the frontal elevation is unrealistically high (or low).
            - lake terminating glaciers
            - other uncertainties
            The default is 0. For lake terminating glaciers,
            it is inferred from PARAMS['free_board_lake_terminating'].
            The best way to set the water level for real glaciers is to use
            the same as used for the inversion (this is what
            `flowline_model_run` does for you)
        """
        super(FluxBasedModelWaterFront, self).__init__(flowlines, 
                                                       mb_model=mb_model, y0=y0,
                                                       glen_a=glen_a, fs=fs,
                                                       inplace=inplace,
                                                       water_level=water_level,
                                                       **kwargs)

        self.fixed_dt = fixed_dt
        if min_dt is None:
            min_dt = cfg.PARAMS['cfl_min_dt']
        if cfl_number is None:
            cfl_number = cfg.PARAMS['cfl_number']
        self.min_dt = min_dt
        self.cfl_number = cfl_number
        
        # Do we want to use shape factors?
        self.sf_func = None
        use_sf = cfg.PARAMS.get('use_shape_factor_for_fluxbasedmodel')
        if use_sf == 'Adhikari' or use_sf == 'Nye':
            self.sf_func = utils.shape_factor_adhikari
        elif use_sf == 'Huss':
            self.sf_func = utils.shape_factor_huss

        # Calving params
        if do_kcalving is None:
            do_kcalving = cfg.PARAMS['use_kcalving_for_run']
        self.do_calving = do_kcalving and self.is_tidewater
        #if calving_k is None:
        #    calving_k = cfg.PARAMS['calving_k']
        self.calving_k = calving_k / cfg.SEC_IN_YEAR
        if calving_use_limiter is None:
            calving_use_limiter = cfg.PARAMS['calving_use_limiter']
        self.calving_use_limiter = calving_use_limiter
        if calving_limiter_frac is None:
            calving_limiter_frac = cfg.PARAMS['calving_limiter_frac']
        if calving_limiter_frac > 0:
            raise NotImplementedError('calving limiter other than 0 not '
                                      'implemented yet')
        self.calving_limiter_frac = calving_limiter_frac
        # Flux gate
        self.flux_gate = utils.tolist(flux_gate, length=len(self.fls))
        self.flux_gate_m3_since_y0 = 0.
        if flux_gate_thickness is not None:
            # Compute the theoretical ice flux from the slope at the top
            flux_gate_thickness = utils.tolist(flux_gate_thickness,
                                               length=len(self.fls))
            self.flux_gate = []
            for fl, fgt in zip(self.fls, flux_gate_thickness):
                # We set the thickness to the desired value so that
                # the widths work ok
                fl = copy.deepcopy(fl)
                fl.thick = fl.thick * 0 + fgt
                slope = (fl.surface_h[0] - fl.surface_h[1]) / fl.dx_meter
                if slope == 0:
                    raise ValueError('I need a slope to compute the flux')
                flux = find_sia_flux_from_thickness(slope,
                                                    fl.widths_m[0],
                                                    fgt,
                                                    shape=fl.shape_str[0],
                                                    glen_a=self.glen_a,
                                                    fs=self.fs)
                self.flux_gate.append(flux)

        # convert the floats to function calls
        for i, fg in enumerate(self.flux_gate):
            if fg is None:
                continue
            try:
                # Do we have a function? If yes all good
                fg(self.yr)
            except TypeError:
                # If not, make one
                self.flux_gate[i] = partial(flux_gate_with_build_up,
                                            flux_value=fg,
                                            flux_gate_yr=(flux_gate_build_up +
                                                          self.y0))

        # Optim
        self.slope_stag = []
        self.thick_stag = []
        self.section_stag = []
        self.depth_stag = []
        self.u_drag = []
        self.u_slide = []
        self.u_stag = []
        self.shapefac_stag = []
        self.flux_stag = []
        self.trib_flux = []
        for fl, trib in zip(self.fls, self._tributary_indices):
            nx = fl.nx
            # This is not staggered
            self.trib_flux.append(np.zeros(nx))
            # We add an additional fake grid point at the end of tributaries
            if trib[0] is not None:
                nx = fl.nx + 1
            # +1 is for the staggered grid
            self.slope_stag.append(np.zeros(nx+1))
            self.thick_stag.append(np.zeros(nx+1))
            self.section_stag.append(np.zeros(nx+1))
            self.depth_stag.append(np.zeros(nx+1))
            self.u_stag.append(np.zeros(nx+1))
            self.u_drag.append(np.zeros(nx+1))
            self.u_slide.append(np.zeros(nx+1))
            self.shapefac_stag.append(np.ones(nx+1))  # beware the ones!
            self.flux_stag.append(np.zeros(nx+1))
    def step(self, dt):
        """Advance one step."""

        # Just a check to avoid useless computations
        if dt <= 0:
            raise InvalidParamsError('dt needs to be strictly positive')

        # Simple container
        mbs = []
        # Loop over tributaries to determine the flux rate
        for fl_id, fl in enumerate(self.fls):

            # This is possibly less efficient than zip() but much clearer
            trib = self._tributary_indices[fl_id]
            slope_stag = self.slope_stag[fl_id]
            thick_stag = self.thick_stag[fl_id]
            section_stag = self.section_stag[fl_id]
            depth_stag = self.depth_stag[fl_id]
            sf_stag = self.shapefac_stag[fl_id]
            flux_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            u_stag = self.u_stag[fl_id]
            u_drag = self.u_drag[fl_id]
            u_slide = self.u_slide[fl_id]
            flux_gate = self.flux_gate[fl_id]

            # Flowline state
            surface_h = fl.surface_h
            thick = fl.thick
            width = fl.widths_m
            section = fl.section
            dx = fl.dx_meter
            depth = utils.clip_min(0,self.water_level - fl.bed_h)

            # If it is a tributary, we use the branch it flows into to compute
            # the slope of the last grid point
            is_trib = trib[0] is not None
            if is_trib:
                fl_to = self.fls[trib[0]]
                ide = fl.flows_to_indice
                surface_h = np.append(surface_h, fl_to.surface_h[ide])
                thick = np.append(thick, thick[-1])
                section = np.append(section, section[-1])
                width = np.append(width, width[-1])
                depth = np.append(depth, depth[-1])
            # elif self.do_calving and self.calving_use_limiter:
                # We lower the max possible ice deformation
                # by clipping the surface slope here. It is completely
                # arbitrary but reduces ice deformation at the calving front.
                # I think that in essence, it is also partly
                # a "calving process", because this ice deformation must
                # be less at the calving front. The result is that calving
                # front "free boards" are quite high.
                # Note that 0 is arbitrary, it could be any value below SL
                #
                # This is deprecated and should not have an effect anymore
                # with the implementation of frontal dynamics below.
                # surface_h = utils.clip_min(surface_h, self.water_level)
                
            # Staggered gradient
            stretch_dist_p = cfg.PARAMS.get('stretch_dist', 8e3)                                                                
            min_l = cfg.PARAMS.get('min_ice_thick_for_length', 0)
            slope_stag[0] = 0
            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
            slope_stag[-1] = slope_stag[-2]
            
            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
            thick_stag[[0, -1]] = thick[[0, -1]]
            
            depth_stag[1:-1] = (depth[0:-1] + depth[1:]) / 2.
            depth_stag[[0, -1]] = depth[[0, -1]]
            
            h=[]
            d=[]
            no_ice=[]
            last_ice=[]
            last_above_wl=[]
            has_ice=[]
            ice_above_wl=[]
            
            A = self.glen_a
            N = self.glen_n
            
            if self.sf_func is not None:
                # TODO: maybe compute new shape factors only every year?
                sf = self.sf_func(fl.widths_m, fl.thick, fl.is_rectangular)
                if is_trib:
                    # for inflowing tributary, the sf makes no sense
                    sf = np.append(sf, 1.)
                sf_stag[1:-1] = (sf[0:-1] + sf[1:]) / 2.
                sf_stag[[0, -1]] = sf[[0, -1]]
            
            # Determine if and where ice bodies are; if no there is no ice above
            # water_level, we fall back to the standard version in the else
            # below.
            ice_above_wl = np.any((fl.surface_h > self.water_level) & 
                                  (fl.thick > min_l) & \
                                  (fl.bed_h < self.water_level))

            has_ice = np.any(fl.thick > min_l)
                                  
            if has_ice and ice_above_wl and self.do_calving:
                last_above_wl = np.nonzero((fl.surface_h > self.water_level) &
                                           (fl.bed_h < self.water_level) &
                                           (fl.thick > min_l))[0][-1]
                # last_above_wl = np.nonzero((fl.surface_h > self.water_level) &
                                           # (fl.bed_h < self.water_level) &
                                           # (fl.thick >= (1028/900)*depth))[0][-1]
                no_ice = np.nonzero((fl.thick < min_l))[0]
                last_ice = np.where((fl.thick[no_ice-1] > min_l) & \
                                (fl.surface_h[no_ice-1] > self.water_level))[0]
                last_ice = no_ice[last_ice]-1

                if last_ice.size == 1:
                    first_ice = np.nonzero(fl.thick[0:last_above_wl+1]\
                                           > min_l)[0][0]
                elif last_ice.size > 1 and (last_ice[-2]+1 < last_above_wl+1):
                    first_ice = np.nonzero(fl.thick[(last_ice[-2]+1)\
                                           :(last_above_wl+1)] > min_l)[0][0]
                    first_ice = last_ice[-2]+first_ice
                elif last_ice.size > 1 :
                    first_ice = np.nonzero(fl.thick[0:(last_above_wl)]\
                                           > min_l)[0][-1]                       
                else:
                    first_ice = 0

                # Determine water depth at the front
                h = fl.thick[last_above_wl]
                d = h - (fl.surface_h[last_above_wl] - self.water_level)
                #d = -fl.bed_h[last_above_wl]
                thick_stag[last_above_wl+1] = h
                depth_stag[last_above_wl+1] = d
                last_thick = np.nonzero((fl.thick > 0) & 
                                        (fl.surface_h > self.water_level))[0][-1]
                # Determine height above buoancy
                z_a_b = utils.clip_min(0,thick_stag - depth_stag * 
                                         (1028/self.rho))

                # Compute net hydrostatic force at the front.
                # One could think about incorporating ice melange / sea ice here
                # as a backstress term. (And also in the frontal ablation 
                # formulation.)
                if last_above_wl == last_thick:
                    pull_last = utils.clip_min(0,0.5 * G * (self.rho * h**2 - 
                                               1028 * d**2))

                    # Determine distance over which the above stress is distributed
                    stretch_length = (last_above_wl - first_ice) * dx
                    stretch_length = utils.clip_min(stretch_length, dx)
                    stretch_dist = utils.clip_max(stretch_length, 
                                                  stretch_dist_p)
                    n_stretch = np.rint(stretch_dist/dx).astype(int)

                    # Define stretch factor and add to driving stress
                    stretch_factor = np.zeros(n_stretch)
                    for j in range(n_stretch):
                        stretch_factor[j] = 2*(j+1)/(n_stretch+1)
                    if dx > stretch_dist:
                        stretch_factor = stretch_dist / dx
                        n_stretch = 1
                        
                    stretch_first = utils.clip_min(0,(last_above_wl+2)-n_stretch).\
                                    astype(int)
                    stretch_last = last_above_wl+2
                    
                    # Take slope for stress calculation at boundary to last grid 
                    # cell as the mean over the "stretched" distance (see above)
                    if last_above_wl+1 < len(fl.bed_h) and \
                    stretch_first != stretch_last-1:
                        slope_stag[last_above_wl+1] = np.nanmean(slope_stag\
                                                                 [stretch_first:\
                                                                 stretch_last-1])
                stress = self.rho*G*slope_stag*thick_stag
                if last_above_wl == last_thick:
                    stress[stretch_first:stretch_last] = (stress[stretch_first:
                                                          stretch_last] + 
                                                          stretch_factor * 
                                                         (pull_last / stretch_dist))

                # Compute velocities 
                u_drag[:] = thick_stag * stress**N * self._fd * sf_stag**N

                # Arbitrarily manipulating u_slide for grid cells 
                # approaching buoancy in order to prevent it from going 
                # towards infinity...
                u_slide[:] = (stress**N / z_a_b) * self.fs * sf_stag**N
                u_slide = np.where(z_a_b < 0.1, 4*u_drag, u_slide)
                # Inhibit flow out of grid cell adjacent to last above 
                # sea-level in order to prevent shelf dynamics.
                #u_slide[last_above_wl+2:] = u_slide[last_above_wl+1]
                #u_drag[last_above_wl+2:] = u_drag[last_above_wl+1]

                u_stag[:] = u_drag + u_slide
                # Staggered section
                # For the flux out of the last grid cell, the staggered section
                # is set to the cross section of the calving front.
                section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
                section_stag[[0, -1]] = section[[0, -1]]
                #section_stag[last_above_wl+1] = h * fl.widths_m[last_above_wl]
                section_stag[last_above_wl+1] = section[last_above_wl]

            else:
                rhogh = (self.rho*G*slope_stag)**N
                u_stag[:] = ((thick_stag**(N+1)) * self._fd * rhogh 
                              + (thick_stag**(N-1)) * self.fs * rhogh) * \
                              sf_stag**N
                #u_stag[thick_stag < depth_stag] = 0

                # Staggered section
                section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
                section_stag[[0, -1]] = section[[0, -1]]

            # Staggered flux rate
            flux_stag[:] = u_stag * section_stag

            # Add boundary condition
            if flux_gate is not None:
                flux_stag[0] = flux_gate(self.yr)

            # CFL condition
            if not self.fixed_dt:
                maxu = np.max(np.abs(u_stag))
                if maxu > cfg.FLOAT_EPS:
                    cfl_dt = self.cfl_number * dx / maxu
                else:
                    cfl_dt = dt

                # Update dt only if necessary
                if cfl_dt < dt:
                    dt = cfl_dt
                    if cfl_dt < self.min_dt:
                        raise RuntimeError(
                            'CFL error: required time step smaller '
                            'than the minimum allowed: '
                            '{:.1f}s vs {:.1f}s. Happening at '
                            'simulation year {:.1f}, fl_id {}, '
                            'bin_id {} and max_u {:.3f} m yr-1.'
                            ''.format(cfl_dt, self.min_dt, self.yr, fl_id,
                                      np.argmax(np.abs(u_stag)),
                                      maxu * cfg.SEC_IN_YEAR))

            # Since we are in this loop, reset the tributary flux
            trib_flux[:] = 0

            # We compute MB in this loop, before mass-redistribution occurs,
            # so that MB models which rely on glacier geometry to decide things
            # (like PyGEM) can do wo with a clean glacier state
            mbs.append(self.get_mb(fl.surface_h, self.yr,
                                   fl_id=fl_id, fls=self.fls))

        # Time step
        if self.fixed_dt:
            # change only if step dt is larger than the chosen dt
            if self.fixed_dt < dt:
                dt = self.fixed_dt
                
        # A second loop for the mass exchange
        for fl_id, fl in enumerate(self.fls):

            flx_stag = self.flux_stag[fl_id]
            trib_flux = self.trib_flux[fl_id]
            tr = self._tributary_indices[fl_id]

            dx = fl.dx_meter

            is_trib = tr[0] is not None

            # For these we had an additional grid point
            if is_trib:
                flx_stag = flx_stag[:-1]

            # Mass-balance
            widths = fl.widths_m
            mb = mbs[fl_id]
            # Allow parabolic beds to grow
            mb = dt * mb * np.where((mb > 0.) & (widths == 0), 10., widths)
            # Prevent more melt than ice above the water-level and surface melt 
            # for grid cells with surface below sea-level
            if self.do_calving:
                bed_below_sl = (fl.bed_h < self.water_level) & (fl.thick > 0)
                mb[bed_below_sl] = utils.clip_min(mb[bed_below_sl],
                                                  -(fl.surface_h[bed_below_sl] 
                                                   - self.water_level) * 
                                                   widths[bed_below_sl])
                mb[fl.surface_h < self.water_level] = 0
            # Update section with ice flow and mass balance
            new_section = (fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt/dx +
                           trib_flux*dt/dx + mb)

            # Keep positive values only and store
            fl.section = utils.clip_min(new_section, 0)
            
            self.calving_rate_myr = 0.
            # Prevent remnants of detached ice below sea-level
            section = fl.section
            ice_above_wl = np.any((fl.surface_h > self.water_level) & 
                                    (fl.bed_h < self.water_level) &
                                    (fl.thick >= (1028/900)*depth))   
            if ice_above_wl and self.do_calving:
                last_last_wl = []
                above_wl = np.nonzero((fl.surface_h > self.water_level) &
                                      (fl.bed_h < self.water_level) &
                                      (fl.thick > (1028/900)*depth))[0]
                for i in above_wl:
                    if i+1 < len(fl.bed_h) and fl.thick[i+1] <= \
                                               (1028/900)*depth[i+1]:
                        last_last_wl = np.append(last_last_wl, i)
                if len(last_last_wl) > 0:
                    last_above_wl = int(last_last_wl[0])
                else:
                    last_above_wl = above_wl[-1]
                last_ice = np.nonzero(fl.thick > 0)[0][-1]
                if last_ice > last_above_wl+1:
                    for i in range(last_above_wl+2, last_ice+1):
                        if section[i] > 0 and fl.bed_h[i] < self.water_level:
                            add_calving = section[i] * dx
                            #fl.calving_bucket_m3 -= add_calving
                            #fl.calving_bucket_m3 = utils.clip_min(0, 
                            #                       fl.calving_bucket_m3)
                            self.calving_m3_since_y0 += add_calving
                            self.calving_rate_myr += (dx / cfg.SEC_IN_YEAR)
                            section[i] = 0
                    fl.section = section

            # If we use a flux-gate, store the total volume that came in
            self.flux_gate_m3_since_y0 += flx_stag[0] * dt

            # Add the last flux to the tributary
            # this works because the lines are sorted in order
            if is_trib:
                # tr tuple: line_index, start, stop, gaussian_kernel
                self.trib_flux[tr[0]][tr[1]:tr[2]] += \
                    utils.clip_min(flx_stag[-1], 0) * tr[3]

            # --- The rest is for calving only ---
            # If tributary, do calving only if we are not transferring mass
            if is_trib and flx_stag[-1] > 0:
                continue

            # No need to do calving in these cases either
            if not self.do_calving or not fl.has_ice():
                continue

            # We do calving only if the last glacier bed pixel is below water
            # (this is to avoid calving elsewhere than at the front)
            if fl.bed_h[fl.thick > 0][-1] > self.water_level:
                continue

            # We do calving only if there is some ice below wl
            # ice_above_wl = np.any((fl.surface_h > self.water_level) & 
                                  # (fl.bed_h < self.water_level) &
                                  # (fl.thick > 0))
            ice_above_wl = np.any((fl.surface_h > self.water_level) & 
                                  (fl.bed_h < self.water_level) &
                                  (fl.thick >= (1028/900)*depth))            
            to_remove=0
            add_calving=0
            first_below_sl = np.nonzero((fl.bed_h < self.water_level) &
                                        (fl.thick > 0))[0][0]
                                       
            if ice_above_wl:
                # last_above_wl = np.nonzero((fl.surface_h > self.water_level) &
                                           # (fl.bed_h < self.water_level) &
                                           # (fl.thick > 0))[0][-1]
                last_above_wl = np.nonzero((fl.surface_h > self.water_level) &
                                           (fl.bed_h < self.water_level) &
                                           (fl.thick >= (1028/900)*depth))[0][-1]
                                      
                if fl.bed_h[last_above_wl+1] > self.water_level:
                    continue

                # OK, we're really calving
                section = fl.section

                # Calving law
                h = fl.thick[last_above_wl]
                d = h - (fl.surface_h[last_above_wl] - self.water_level)
                k = self.calving_k
                q_calving = k * d * h * fl.widths_m[last_above_wl]
                q_calving = utils.clip_min(0,q_calving)
                # Add to the bucket and the diagnostics
                fl.calving_bucket_m3 += q_calving * dt
                self.calving_m3_since_y0 += q_calving * dt
                self.calving_rate_myr += (q_calving / section[last_above_wl] *
                                         cfg.SEC_IN_YEAR)
                # See if we have ice below sea-water to clean out first
                #below_sl = (fl.surface_h < self.water_level) & (fl.thick > 0)
                below_sl = ((fl.bed_h < self.water_level) & 
                            (fl.thick < (1028/900)*depth) &
                            (fl.thick > 0))
                to_remove = np.sum(section[below_sl]) * fl.dx_meter
                bed_below_sl = (fl.bed_h < self.water_level) & (fl.thick > 0)

                if 0 < to_remove < fl.calving_bucket_m3:
                    # This is easy, we remove everything
                    section[below_sl] = 0
                    fl.calving_bucket_m3 -= to_remove
                elif to_remove > 0 and fl.calving_bucket_m3 > 0:
                    # We can only remove part of if
                    section[below_sl] = 0
                    section[last_above_wl+1] = ((to_remove - fl.calving_bucket_m3)
                                                / fl.dx_meter)
                    fl.calving_bucket_m3 = 0
                elif to_remove > 0:
                    section[below_sl] = 0
                    section[last_above_wl+1] = to_remove / fl.dx_meter
                # The rest of the bucket might calve an entire grid point (or more)
                vol_last = section[last_above_wl] * fl.dx_meter
                while fl.calving_bucket_m3 > vol_last and \
                last_above_wl >= bed_below_sl[0]:
                    fl.calving_bucket_m3 -= vol_last
                    section[last_above_wl] = 0

                    # OK check if we need to continue (unlikely)
                    last_above_wl -= 1
                    vol_last = section[last_above_wl] * fl.dx_meter
                fl.section = section 

                # Deal with surface height at front becoming too high because of
                # elif above, i.e. when too much volume falls below sea-level and
                # is then accumulated in the "last" grid cell. Everything that 
                # is higher than the previous grid point is therefore calved off.  
                section = fl.section  
                while ((last_above_wl+1 < len(fl.bed_h)) and 
                      (fl.surface_h[last_above_wl+1] > fl.surface_h[last_above_wl])
                       and section[last_above_wl+1] > 0):
                    add_calving = ((fl.surface_h[last_above_wl+1] - 
                                    fl.surface_h[last_above_wl]) * 
                                   fl.widths_m[last_above_wl+1] * dx)

                    if ((last_above_wl+2 < len(fl.bed_h)) and 
                        ((add_calving / (fl.widths_m[last_above_wl+2] * dx)) > \
                        (1028/900)*depth[last_above_wl+2])):
                        section[last_above_wl+2] += add_calving / dx
                    else:
                        #fl.calving_bucket_m3 -= add_calving
                        #fl.calving_bucket_m3 = utils.clip_min(0, fl.calving_bucket_m3)
                        self.calving_m3_since_y0 += add_calving
                        self.calving_rate_myr += ((add_calving / 
                                                   section[last_above_wl+1]) / 
                                                   cfg.SEC_IN_YEAR)
                    section[last_above_wl+1] -= add_calving / dx
                    section[last_above_wl+1] = utils.clip_min(0, 
                                                       section[last_above_wl+1])
                    add_calving = 0
                    fl.section = section
                    section = fl.section
                    last_above_wl += 1

                # We update the glacier with our changes
                fl.section = section
            elif fl.thick[first_below_sl-1] == 0:
                section = fl.section
                leftover = ((fl.bed_h < self.water_level) & 
                            (fl.thick <= (1028/900)*depth) &
                            (fl.thick > 0))
                add_calving = np.sum(section[leftover] * dx)
                #fl.calving_bucket_m3 -= add_calving
                fl.calving_bucket_m3 = utils.clip_min(0, fl.calving_bucket_m3)
                self.calving_m3_since_y0 += add_calving
                self.calving_rate_myr += (np.size(section[leftover]) * dx / 
                                         cfg.SEC_IN_YEAR)
                section[leftover] = 0
                fl.section = section

        # Next step
        self.t += dt
        return dt