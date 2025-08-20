# oggm_marine_terminating

[![PyPI version](https://badge.fury.io/py/oggm-marine-terminating.svg)](https://badge.fury.io/py/oggm-marine-terminating)
[![GitHub](https://img.shields.io/badge/GitHub-Shafeeque_Branch-blue.svg)](https://github.com/MuhammadShafeeque/Enhanced-Modeling-Marine-Terminating-Glaciers/tree/Shafeeque)

## Installation

Install from PyPI:

```bash
pip install oggm-marine-terminating
```

Install from GitHub:

```bash
pip install git+https://github.com/MuhammadShafeeque/Enhanced-Modeling-Marine-Terminating-Glaciers.git@Shafeeque
```

## Quick Start

```python
# Import as a standard Python package
from oggm_marine_terminating import FluxBasedModelMarineFront
from oggm_marine_terminating import mass_conservation_inversion_mt, flowline_model_run_mt

# Set up your OGGM run
# ...

# Use the enhanced marine-terminating model
model = FluxBasedModelMarineFront(flowlines, mb_model=mb_model, y0=0., 
                                 fs=0., glen_a=glen_a, water_level=water_level)
```

For more detailed usage examples, please refer to the [full documentation](https://github.com/MuhammadShafeeque/Enhanced-Modeling-Marine-Terminating-Glaciers/tree/Shafeeque).
