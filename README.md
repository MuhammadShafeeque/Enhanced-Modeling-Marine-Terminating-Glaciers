# Enhanced Modeling of Marine-Terminating Glaciers

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/MuhammadShafeeque/Enhanced-Modeling-Marine-Terminating-Glaciers/blob/main/LICENSE.txt)
[![PyPI version](https://badge.fury.io/py/oggm-marine-terminating.svg)](https://badge.fury.io/py/oggm-marine-terminating)

## Overview

This repository contains a specialized module for modeling marine-terminating glaciers within the Open Global Glacier Model (OGGM) framework. The code extends OGGM's capabilities by implementing enhanced physics for marine-terminating glaciers, including frontal ablation, calving dynamics, and ice-ocean interactions.

## Credits

This project is built upon the original work from [water_terminating_module](https://github.com/jmalles/water_terminating_module) by Jan-Hendrik Malles et al. The implementation is based on the methodology described in:

Malles J-H, Maussion F, Ultee L, Kochtitzky W, Copland L, Marzeion B. Exploring the impact of a frontal ablation parameterization on projected 21st-century mass change for Northern Hemisphere glaciers. Journal of Glaciology. 2023;69(277):1317-1332. [DOI: 10.1017/jog.2023.19](https://doi.org/10.1017/jog.2023.19)

## Features

- Improved ice dynamics for tidewater glaciers
- Submarine melting parameterization
- Enhanced calving processes modeling
- Frontal ablation calculation
- Water pressure effects modeling
- Height-above-buoyancy model implementation

## Structure

- `oggm_marine_terminating/`: Python package implementing enhanced models for marine-terminating glaciers
  - `flux_model.py`: Implementation of the flux-based model for marine-terminating glaciers
  - `aux_funcs.py`: Auxiliary functions for inverting and running marine-terminating glaciers
- `input_data/`: Observational data for model calibration and validation

## Requirements

- OGGM = 1.5.3 (Open Global Glacier Model)
- NumPy
- SciPy
- Pandas
- Matplotlib (for visualization)

## Usage

The module can be imported and used as an extension to OGGM's standard functionality:

```python
# Import as a standard Python package
from oggm_marine_terminating import FluxBasedModelMarineFront
from oggm_marine_terminating import mass_conservation_inversion_mt, flowline_model_run_mt

# Alternative import from the internal modules
from oggm_marine_terminating.flux_model import FluxBasedModelMarineFront
from oggm_marine_terminating.aux_funcs import mass_conservation_inversion_mt, flowline_model_run_mt
```

If you want to use the enhanced functions for marine-terminating glaciers with OGGM, use these enhanced versions instead of the standard OGGM functions.

### Installation

Install from PyPI:

```bash
pip install oggm-marine-terminating
```

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/MuhammadShafeeque/Enhanced-Modeling-Marine-Terminating-Glaciers.git@Shafeeque
```

Or clone the repository and install in development mode:

```bash
git clone https://github.com/MuhammadShafeeque/Enhanced-Modeling-Marine-Terminating-Glaciers.git
cd Enhanced-Modeling-Marine-Terminating-Glaciers
git checkout Shafeeque
pip install -e .
```

## License

This project is licensed under the terms of the LICENSE.txt file included in this repository.

## Citing this work

If you use this code in your research, please cite:

1. The original implementation:
   - Malles J-H, Maussion F, Ultee L, Kochtitzky W, Copland L, Marzeion B. Exploring the impact of a frontal ablation parameterization on projected 21st-century mass change for Northern Hemisphere glaciers. Journal of Glaciology. 2023;69(277):1317-1332. [DOI: 10.1017/jog.2023.19](https://doi.org/10.1017/jog.2023.19)

2. The OGGM framework:
   - Maussion, F., Butenko, A., Champollion, N., Dusch, M., Eis, J., Fourteau, K., Gregor, P., Jarosch, A. H., Landmann, J., Oesterle, F., Recinos, B., Rothenpieler, T., Vlug, A., Wild, C. T., and Marzeion, B. (2019). The Open Global Glacier Model (OGGM) v1.1. *Geoscientific Model Development*, 12(3), 909–931. [DOI: 10.5194/gmd-12-909-2019](https://doi.org/10.5194/gmd-12-909-2019)
