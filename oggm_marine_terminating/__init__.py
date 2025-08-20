"""OGGM Marine Terminating Glaciers package

This package extends OGGM to provide enhanced modeling capabilities for 
marine-terminating glaciers including improved physics for calving, 
frontal ablation, and ice-ocean interactions.
"""

from oggm_marine_terminating.flux_model import FluxBasedModelMarineFront
from oggm_marine_terminating.aux_funcs import (
    mass_conservation_inversion_mt,
    flowline_model_run_mt,
    run_random_climate_mt,
    run_from_climate_data_mt,
    sia_thickness_mt,
    sia_thickness_via_optim_mt,
    calculate_water_depth
)

__version__ = '0.1.0'

# Define which symbols are exported when doing 'from oggm_marine_terminating import *'
__all__ = [
    'FluxBasedModelMarineFront',
    'mass_conservation_inversion_mt',
    'flowline_model_run_mt',
    'run_random_climate_mt',
    'run_from_climate_data_mt',
    'sia_thickness_mt',
    'sia_thickness_via_optim_mt',
    'calculate_water_depth'
]
