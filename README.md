# Hyperbolic Nanocavities for Single-Molecule Vibrational Strong Coupling

This repository contains the numerical calculations, processed data, and
analysis notebooks associated with a study of light–matter interaction in
hyperbolic hBN cavities. The workflow covers planar hyperbolic phonon
polaritons, cavity-resonance estimates, spectral-density fitting, perturbative
corrections and master-equation calculations.

The notebooks are intended to be readable records of the analysis as well as
reproducible calculations. Common numerical routines have therefore been
moved into the reusable `methods` Python package.

## Repository structure

```text
.
├── data/                    # Spectral densities and electric-field data
├── methods/                 # Reusable numerical and plotting routines
├── notebooks/               # Documented analysis notebooks
│   └── few_mode_fits/       # Serialized few-mode fitting results
├── pyproject.toml           # Package metadata and fitter dependency
├── requirements.txt         # Python environment dependencies
```

## Installation

Python 3.11 is recommended. Create the project environment, install the
dependencies, and register it as a Jupyter kernel with:

```bash
conda create -n vschyper python=3.11 pip
conda activate vschyper

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ipykernel install --user \
    --name vschyper \
    --display-name "Python (vschyper)"
```

The editable installation makes `methods` importable from the notebooks and
immediately exposes local changes without requiring reinstallation. It also
installs the pinned revision of
[`spectral_density_fit`](https://github.com/jfeist/spectral_density_fit), which
is declared in `pyproject.toml` and used by the few-mode fitting workflow.

Start Jupyter from the repository root so that the notebooks can resolve their
relative data paths:

```bash
conda activate vschyper
jupyter lab
```

Select **Python (vschyper)** as the notebook kernel.

## Notebook guide

The notebooks are organized around the following calculations:

- `planar_hpps.ipynb` introduces the hBN permittivity, planar polariton
  dispersion, wavelengths, and propagation lengths.
- `cavity_resonance_frequency_analysis.ipynb` uses the planar dispersion and a
  geometrical model to estimate how the cavity resonance changes with cavity
  dimensions.
- `DAchS-25-10-1.ipynb`, `DAchS-6.5-5-1.ipynb`, and
  `DAchS-25-40-1.ipynb` analyze representative cavity geometries. They include
  spectral-density processing, few-mode fitting, perturbative corrections
  and open-system calculations.
- `reference_calculation_sheinfux.ipynb` applies the same analysis workflow to
  an extended hBN slab above a gold film with an aperture.

Some cells load saved fits from `notebooks/few_mode_fits`. Their data paths are
written relative to the `notebooks` directory, which is the working directory
Jupyter normally assigns when these files are opened from the repository's
Jupyter session. The notebooks contain explanatory Markdown cells and can also
be read without executing every computational section.

## The `methods` package

The package currently contains:

- `hBN_permittivity.py`: Lorentz-model in-plane and out-of-plane hBN
  permittivities.
- `hpp_dispersion.py`: planar hBN slab dispersion, polariton wavelengths, and
  propagation lengths.
- `green_function.py`: numerically reflected Green-function and LDOS
  calculations.
- `apply_few_mode.py`: the `FewModeFit` interface to
  `spectral_density_fit` and calculation of light–matter figures of merit.
- `perturbative.py`: residual Lamb-shift and decay-rate corrections and a
  perturbative-validity estimate.
- `mastereq.py`: the QuTiP-based `MasterEquationQtp` model.
- `plotcolored.py`: colour-mapped line plotting.
- `tools.py`: shared physical constants and unit conversions.

Common routines can be imported directly from the package:

```python
from methods import FewModeFit, get_epsilon, slab_dispersion
```

The QuTiP-based master-equation class is kept as an explicit specialized
import:

```python
from methods.mastereq import MasterEquationQtp
```

## Citation

The preprint associated with this repository is available on
[arXiv](https://arxiv.org/abs/2611.06240).

If you use the code or data from this repository, please cite:

> Stavros Athanasiou, Olivier J. F. Martin, Francisco J. García-Vidal, and Johannes Feist, “Hyperbolic Nanocavities for Single-Molecule Vibrational Strong Coupling,” *arXiv:2611.06240*, 2026.  
> https://doi.org/10.48550/arXiv.2611.06240