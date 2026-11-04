"""Physical constants, unit conversions, and small notebook utilities.

The energy unit used throughout the repository is the electronvolt (eV),
lengths are generally in nanometres, and spectroscopic wavenumbers are in
inverse centimetres (cm^-1).
"""

from collections.abc import Mapping

import numpy as np
from scipy.constants import c, elementary_charge, h, hbar


# Physical constants in the units used by the calculations.
hbar_eV_s = hbar / elementary_charge
hc_eV_nm = h * c / elementary_charge * 1e9

# One debye in units of elementary-charge nanometres (e nm).
debye_e_nm = 3.33564e-30 / (elementary_charge * 1e-9)

# E = h c nu_tilde, with nu_tilde expressed in cm^-1.
CM_INVERSE_TO_EV = h * c / elementary_charge * 100.0


def nu2E(wavenumber_cm):
    """Convert a spectroscopic wavenumber in cm^-1 to energy in eV."""
    return np.asarray(wavenumber_cm) * CM_INVERSE_TO_EV


def E2nu(energy_eV):
    """Convert an energy in eV to a spectroscopic wavenumber in cm^-1."""
    return np.asarray(energy_eV) / CM_INVERSE_TO_EV


def print_pars(parameters=None, /, **named_parameters):
    """Print named parameters in a compact, deterministic form.

    Parameters may be supplied as a mapping and/or as keyword arguments.
    This helper is retained for compatibility with the analysis notebooks.
    """
    if parameters is not None:
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        named_parameters = {**parameters, **named_parameters}

    for name, value in named_parameters.items():
        print(f"{name}: {value}")


__all__ = [
    "CM_INVERSE_TO_EV",
    "E2nu",
    "debye_e_nm",
    "hbar_eV_s",
    "hc_eV_nm",
    "nu2E",
    "print_pars",
]
