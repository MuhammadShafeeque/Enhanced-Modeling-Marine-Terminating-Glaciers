## OGGM Marine Terminating Module

This module provides enhanced functionality for modeling marine-terminating glaciers 
within the Open Global Glacier Model (OGGM) framework.

### Overview

The `oggm_marine_terminating` package extends OGGM's capabilities to better handle 
the physics of tidewater glaciers, including:

- Improved ice dynamics for marine-terminating glaciers
- Frontal ablation processes
- Submarine melting
- Enhanced calving models
- Water pressure effects

### Usage

The module provides enhanced versions of standard OGGM functions, specifically 
designed for marine-terminating glaciers:

```python
# Import the module
from oggm_marine_terminating import FluxBasedModelMarineFront
from oggm_marine_terminating import (
    mass_conservation_inversion_mt,
    flowline_model_run_mt,
    run_random_climate_mt,
    run_from_climate_data_mt
)

# Use enhanced functions instead of standard OGGM functions
# For example, use flowline_model_run_mt instead of flowline_model_run
model = flowline_model_run_mt(gdir, water_level=0.0)
```

### Integration with OGGM

This module is designed to be fully compatible with OGGM's workflow. Simply replace
the standard OGGM functions with the enhanced marine-terminating versions when 
working with tidewater glaciers.

### Key Classes and Functions

- `FluxBasedModelMarineFront`: Enhanced flowline model for marine-terminating glaciers
- `mass_conservation_inversion_mt`: Modified inversion for tidewater glaciers
- `flowline_model_run_mt`: Run a model simulation with marine-terminating physics
- `run_random_climate_mt`: Run simulation with random climate for marine-terminating glaciers
- `run_from_climate_data_mt`: Run simulation with climate data for marine-terminating glaciers
- `sia_thickness_mt`: Compute SIA thickness for marine-terminating glaciers
- `calculate_water_depth`: Utility function to calculate water depth at glacier bed

For more details, refer to the function docstrings.
