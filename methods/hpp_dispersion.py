"""Dispersion and propagation lengths of polaritons in a uniaxial hBN slab."""

from __future__ import annotations

import numpy as np

from .hBN_permittivity import get_epsilon


_EPSILON_X, _EPSILON_Z = get_epsilon(hbn_type="pure")
# Retain the conversion used to generate the existing dispersion results.
_HC_EV_NM = 1240.0


def arctan(z, k=0):
    """Evaluate the complex arctangent on logarithmic branch ``k``.

    Parameters
    ----------
    z : array_like
        Real or complex argument.
    k : int, optional
        Integer logarithmic-branch offset. ``k=0`` gives the principal branch
        used by the slab dispersion relation.
    """

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    z = np.asarray(z, dtype=complex)
    z_minus = 1.0 - 1j * z
    z_plus = 1.0 + 1j * z
    log_magnitude = np.log(np.abs(z_minus) / np.abs(z_plus))
    phase_difference = np.angle(z_minus) - np.angle(z_plus) + 2.0 * np.pi * k
    result = 0.5j * (log_magnitude + 1j * phase_difference)
    return result[()] if result.ndim == 0 else result


def _validate_dispersion_inputs(omega_0_ls, l_val_ls, d, eps_1, eps_3):
    energies = np.atleast_1d(np.asarray(omega_0_ls, dtype=float))
    branches = np.atleast_1d(np.asarray(l_val_ls))

    if energies.ndim != 1 or energies.size == 0:
        raise ValueError("omega_0_ls must be a non-empty scalar or 1D array")
    if not np.all(np.isfinite(energies)) or np.any(energies <= 0.0):
        raise ValueError("omega_0_ls must contain finite, positive energies")
    if branches.ndim != 1 or branches.size == 0:
        raise ValueError("l_val_ls must be a non-empty scalar or 1D array")
    if not np.all(np.equal(branches, np.floor(branches))) or np.any(branches < 0):
        raise ValueError("l_val_ls must contain non-negative integers")
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("d must be a finite, positive slab thickness")
    if not np.isfinite(eps_1) or not np.isfinite(eps_3):
        raise ValueError("eps_1 and eps_3 must be finite")

    return energies, branches.astype(int)


def _dispersion_candidate(energy, branch, sign, d, eps_1, eps_3):
    """Evaluate one square-root branch of the large-momentum dispersion."""

    eps_2x = _EPSILON_X(energy)
    eps_2z = _EPSILON_Z(energy)
    k0 = 2.0 * np.pi * energy / _HC_EV_NM
    psi = sign * np.sqrt(eps_2z) / (1j * np.sqrt(eps_2x))
    z1 = eps_1 / (eps_2x * psi)
    z2 = eps_3 / (eps_2x * psi)
    return -psi * (arctan(z1) + arctan(z2) + np.pi * branch) / (k0 * d)


def slab_dispersion(omega_0_ls, l_val_ls, d, eps_1=1.0, eps_3=1.0):
    """Calculate the normalized complex wavevector of hBN slab modes.

    Parameters
    ----------
    omega_0_ls : float or array_like
        Photon energies in eV.
    l_val_ls : int or array_like
        Non-negative polariton branch indices.
    d : float
        Slab thickness in nm.
    eps_1, eps_3 : float, optional
        Relative permittivities of the upper and lower half-spaces.

    Returns
    -------
    numpy.ndarray
        Complex normalized in-plane wavevectors ``k_x / k_0`` with shape
        ``(n_branches, n_energies)``. An unavailable outgoing solution is
        represented by ``nan + 1j*nan``.
    """

    energies, branches = _validate_dispersion_inputs(
        omega_0_ls, l_val_ls, d, eps_1, eps_3
    )
    result = np.full((branches.size, energies.size), np.nan + 1j * np.nan)

    for branch_index, branch in enumerate(branches):
        for energy_index, energy in enumerate(energies):
            # Preserve the original radiation-condition selection: try the
            # positive square-root branch first, then the negative branch.
            for sign in (1.0, -1.0):
                candidate = _dispersion_candidate(
                    energy, branch, sign, d, eps_1, eps_3
                )
                if candidate.imag >= 0.0:
                    result[branch_index, energy_index] = candidate
                    break

    return result


def compute_lengths(kx, omega_0):
    """Convert normalized complex wavevectors to wavelength and decay length.

    Parameters
    ----------
    kx : array_like
        Complex normalized wavevector ``k_x / k_0``. Its final dimension must
        be broadcast-compatible with ``omega_0``.
    omega_0 : float or array_like
        Photon energies in eV.

    Returns
    -------
    wavelength_nm, propagation_length_nm : numpy.ndarray
        ``2 pi / Re(k_x)`` and ``1 / [2 Im(k_x)]`` after restoring the vacuum
        wavenumber. Infinite values are retained at zero real or imaginary part.
    """

    kx = np.asarray(kx, dtype=complex)
    energies = np.asarray(omega_0, dtype=float)
    if not np.all(np.isfinite(energies)) or np.any(energies <= 0.0):
        raise ValueError("omega_0 must contain finite, positive energies")

    try:
        k0 = 2.0 * np.pi * energies / _HC_EV_NM
        with np.errstate(divide="ignore", invalid="ignore"):
            wavelength_nm = 2.0 * np.pi / (kx.real * k0)
            propagation_length_nm = 1.0 / (2.0 * kx.imag * k0)
    except ValueError as exc:
        raise ValueError("kx and omega_0 have incompatible shapes") from exc

    return wavelength_nm, propagation_length_nm


__all__ = ["arctan", "compute_lengths", "slab_dispersion"]
