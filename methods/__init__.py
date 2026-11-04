"""Numerical methods used in the hyperbolic-cavities calculations.

The commonly used numerical and fitting helpers are available directly from
this namespace. ``mastereq`` remains an explicit import because it requires
QuTiP and is used only by the open-system calculations.
"""

from .apply_few_mode import FewModeFit
from .hBN_permittivity import get_epsilon
from .hpp_dispersion import compute_lengths, slab_dispersion
from .perturbative import check_perturb_fm, decay_rate_fm, lamb_shift_fm
from .plotcolored import plotcolored
from .tools import E2nu, debye_e_nm, hbar_eV_s, hc_eV_nm, nu2E

__all__ = [
    "check_perturb_fm",
    "compute_lengths",
    "decay_rate_fm",
    "debye_e_nm",
    "E2nu",
    "FewModeFit",
    "get_epsilon",
    "hbar_eV_s",
    "hc_eV_nm",
    "lamb_shift_fm",
    "nu2E",
    "plotcolored",
    "slab_dispersion",
]
