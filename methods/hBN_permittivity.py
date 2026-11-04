"""Lorentz-model permittivity of isotopically pure and natural hBN.

The tabulated phonon frequencies and linewidths are defined first in
inverse centimetres and then converted to photon energies in eV. Parameter
values are taken from the supporting information of Giles et al., Nature
Materials 17, 134--139 (2018).
"""

from __future__ import annotations

from .tools import CM_INVERSE_TO_EV


# Isotopically pure boron-11 hBN: in-plane parameters in cm^-1.
eps_inf_x_pure11b = 5.32
nu_TO_x_pure11b = 1359.8
nu_LO_x_pure11b = 1608.7
nu_gamma_x_pure11b = 2.1

# Isotopically pure boron-11 hBN: out-of-plane parameters in cm^-1.
eps_inf_z_pure11b = 3.15
nu_TO_z_pure11b = 755
nu_LO_z_pure11b = 814
nu_gamma_z_pure11b = 1

# Naturally abundant hBN: in-plane parameters in cm^-1.
eps_inf_x_nature = 4.9
nu_TO_x_nature = 1360
nu_LO_x_nature = 1614
nu_gamma_x_nature = 7

# Naturally abundant hBN: out-of-plane parameters in cm^-1.
eps_inf_z_nature = 2.95
nu_TO_z_nature = 760
nu_LO_z_nature = 825
nu_gamma_z_nature = 3


# Convert the spectroscopic parameters to the eV values used by the model.
omega_TO_x_pure11b = nu_TO_x_pure11b * CM_INVERSE_TO_EV
omega_LO_x_pure11b = nu_LO_x_pure11b * CM_INVERSE_TO_EV
gamma_x_pure11b = nu_gamma_x_pure11b * CM_INVERSE_TO_EV

omega_TO_z_pure11b = nu_TO_z_pure11b * CM_INVERSE_TO_EV
omega_LO_z_pure11b = nu_LO_z_pure11b * CM_INVERSE_TO_EV
gamma_z_pure11b = nu_gamma_z_pure11b * CM_INVERSE_TO_EV

omega_TO_x_nature = nu_TO_x_nature * CM_INVERSE_TO_EV
omega_LO_x_nature = nu_LO_x_nature * CM_INVERSE_TO_EV
gamma_x_nature = nu_gamma_x_nature * CM_INVERSE_TO_EV

omega_TO_z_nature = nu_TO_z_nature * CM_INVERSE_TO_EV
omega_LO_z_nature = nu_LO_z_nature * CM_INVERSE_TO_EV
gamma_z_nature = nu_gamma_z_nature * CM_INVERSE_TO_EV


def eps(eps_inf, omega_LO, omega_TO, gamma):
    """Return a Lorentz-oscillator permittivity function.

    All frequency-like arguments and the input to the returned function are
    photon energies in eV.
    """

    def permittivity(omega):
        numerator = omega_LO**2 - omega**2 - 1j * gamma * omega
        denominator = omega_TO**2 - omega**2 - 1j * gamma * omega
        return eps_inf * numerator / denominator

    return permittivity


def get_epsilon(hbn_type):
    """Return the in-plane and out-of-plane hBN permittivity functions.

    Parameters
    ----------
    hbn_type : {"pure", "nature"}
        Select isotopically pure boron-11 or naturally abundant hBN.
    """

    if hbn_type == "pure":
        eps_x = eps(
            eps_inf=eps_inf_x_pure11b,
            omega_LO=omega_LO_x_pure11b,
            omega_TO=omega_TO_x_pure11b,
            gamma=gamma_x_pure11b,
        )
        eps_z = eps(
            eps_inf=eps_inf_z_pure11b,
            omega_LO=omega_LO_z_pure11b,
            omega_TO=omega_TO_z_pure11b,
            gamma=gamma_z_pure11b,
        )
    elif hbn_type == "nature":
        eps_x = eps(
            eps_inf=eps_inf_x_nature,
            omega_LO=omega_LO_x_nature,
            omega_TO=omega_TO_x_nature,
            gamma=gamma_x_nature,
        )
        eps_z = eps(
            eps_inf=eps_inf_z_nature,
            omega_LO=omega_LO_z_nature,
            omega_TO=omega_TO_z_nature,
            gamma=gamma_z_nature,
        )
    else:
        raise ValueError("hbn_type must be 'pure' or 'nature'")

    return eps_x, eps_z


__all__ = ["eps", "get_epsilon"]
