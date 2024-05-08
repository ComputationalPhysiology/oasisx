# Oasisx

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ComputationalPhysiology/oasisx/main)
[![MIT License](https://img.shields.io/github/license/computationalphysiology/oasisx)](LICENSE)

The documentation for this repository can be found [here](https://computationalphysiology.github.io/oasisx).

## Installation

### Requirements

[DOLFINx](https://github.com/FEniCS/dolfinx/), which can be installed through apt, conda, docker, spack or manually, see: [DOLFINx installation instructions](https://github.com/FEniCS/dolfinx/#installation) for details

Once you have DOLFINx installed, install a compatible version of Oasisx:

- `python3 -m pip install oasisx` (compatible with version 0.8.0 of DOLFINx)
- `python3 -m pip install git+https://github.com/computationalPhysiology/oasisx@main` (compatible with version the main branch of DOLFINx)

#### Other version compatability
- oasisx v1.0.0 is compatible with DOLFINx v0.7.x
- The main branch of oasisx aim to be compatible with the main branch of DOLFINx
