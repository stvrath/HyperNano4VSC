"""Sommerfeld Green tensor for a uniaxial slab.

The implementation assumes harmonic fields proportional to ``exp(-i omega t)``.
Consequently, longitudinal wavevectors are placed on the outgoing-wave branch:
``Im(k_z) >= 0``, with ``Re(k_z) >= 0`` on the real axis. This convention is
essential for evanescent waves to decay away from an interface.

Lengths are in nanometres, photon energies are in electronvolts, and the
dimensionless in-plane wavevector ``kap`` is ``k_parallel / k0``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from .tools import hc_eV_nm


_BRANCH_ATOL = 64.0 * np.finfo(float).eps


def _outgoing_sqrt(value):
    """Return the square root on the outgoing/decaying-wave branch."""

    root = np.sqrt(np.asarray(value, dtype=complex))
    almost_real = np.abs(root.imag) <= _BRANCH_ATOL * np.maximum(1.0, np.abs(root))
    flip = (root.imag < 0.0) | (almost_real & (root.real < 0.0))
    return np.where(flip, -root, root)


def _safe_ratio(numerator, denominator):
    """Divide complex arrays while allowing physical poles to remain visible."""

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator)


def _slab_reflection(r12, r23, phase):
    """Evaluate a finite-slab reflection coefficient without overflow.

    This evaluates ``(r12 + r23 exp(phase)) / (1 + r12 r23 exp(phase))``.
    When ``Re(phase) > 0``, numerator and denominator are instead multiplied by
    ``exp(-phase)``. The resulting expression is algebraically identical but
    contains no exponentially growing intermediate value.
    """

    r12, r23, phase = np.broadcast_arrays(
        np.asarray(r12, dtype=complex),
        np.asarray(r23, dtype=complex),
        np.asarray(phase, dtype=complex),
    )
    result = np.empty(r12.shape, dtype=complex)
    bounded = phase.real <= 0.0

    if np.any(bounded):
        propagation = np.exp(phase[bounded])
        result[bounded] = _safe_ratio(
            r12[bounded] + r23[bounded] * propagation,
            1.0 + r12[bounded] * r23[bounded] * propagation,
        )

    if np.any(~bounded):
        inverse_propagation = np.exp(-phase[~bounded])
        result[~bounded] = _safe_ratio(
            r12[~bounded] * inverse_propagation + r23[~bounded],
            inverse_propagation + r12[~bounded] * r23[~bounded],
        )

    return result[()] if result.ndim == 0 else result


def _validate_layer_inputs(kap, eps_2z, d, k0):
    kap = np.asarray(kap, dtype=float)
    if not np.all(np.isfinite(kap)) or np.any(kap < 0.0):
        raise ValueError("kap must contain finite, non-negative values")
    if not np.isfinite(d) or d < 0.0:
        raise ValueError("d must be a finite, non-negative thickness")
    if not np.isfinite(k0) or k0 <= 0.0:
        raise ValueError("k0 must be finite and positive")
    if np.any(np.asarray(eps_2z) == 0):
        raise ValueError("eps_2z must be nonzero")
    return kap


def ref_p(kap, eps_2x, eps_2z, eps_1, eps_3, d, k0):
    """Return the p-polarized reflection coefficient of a uniaxial slab.

    The slab has in-plane and out-of-plane permittivities ``eps_2x`` and
    ``eps_2z``, thickness ``d``, and lies between isotropic half-spaces with
    permittivities ``eps_1`` and ``eps_3``.
    """

    kap = _validate_layer_inputs(kap, eps_2z, d, k0)
    gamma_1 = k0 * _outgoing_sqrt(eps_1 - kap**2)
    gamma_2 = k0 * _outgoing_sqrt(eps_2x - (eps_2x / eps_2z) * kap**2)
    gamma_3 = k0 * _outgoing_sqrt(eps_3 - kap**2)

    r12 = _safe_ratio(
        eps_2x * gamma_1 - eps_1 * gamma_2,
        eps_2x * gamma_1 + eps_1 * gamma_2,
    )
    r23 = _safe_ratio(
        eps_3 * gamma_2 - eps_2x * gamma_3,
        eps_3 * gamma_2 + eps_2x * gamma_3,
    )
    return _slab_reflection(r12, r23, 2j * gamma_2 * d)


def ref_s(kap, eps_2x, eps_2z, eps_1, eps_3, d, k0):
    """Return the s-polarized reflection coefficient of a uniaxial slab."""

    kap = _validate_layer_inputs(kap, eps_2z, d, k0)
    gamma_1 = k0 * _outgoing_sqrt(eps_1 - kap**2)
    gamma_2 = k0 * _outgoing_sqrt(eps_2x - (eps_2x / eps_2z) * kap**2)
    gamma_3 = k0 * _outgoing_sqrt(eps_3 - kap**2)

    r12 = _safe_ratio(gamma_1 - gamma_2, gamma_1 + gamma_2)
    r23 = _safe_ratio(gamma_2 - gamma_3, gamma_2 + gamma_3)
    return _slab_reflection(r12, r23, 2j * gamma_2 * d)


def g_s_aniso(
    kap,
    eps_1,
    eps_2x_val,
    eps_2z_val,
    eps_3,
    z,
    d,
    k0,
    z0=None,
):
    """Return the reflected Green tensor resolved by in-plane wavevector.

    ``z`` and ``z0`` are observation and source distances above the slab in nm.
    If ``z0`` is omitted, source and observation are placed at the same height.
    The returned array has shape ``(3, 3, n_kap)``.
    """

    kap = np.atleast_1d(_validate_layer_inputs(kap, eps_2z_val, d, k0))
    z0 = z if z0 is None else z0
    if not np.isfinite(z) or not np.isfinite(z0) or z < 0.0 or z0 < 0.0:
        raise ValueError("z and z0 must be finite, non-negative distances")
    if eps_1 == 0:
        raise ValueError("eps_1 must be nonzero")

    rs = ref_s(kap, eps_2x_val, eps_2z_val, eps_1, eps_3, d, k0)
    rp = ref_p(kap, eps_2x_val, eps_2z_val, eps_1, eps_3, d, k0)
    gamma = _outgoing_sqrt(eps_1 - kap**2)

    tensor = np.zeros((3, 3, kap.size), dtype=complex)
    transverse = np.pi * (rs - gamma**2 * rp / eps_1)
    tensor[0, 0] = transverse
    tensor[1, 1] = transverse
    tensor[2, 2] = 2.0 * np.pi * kap**2 * rp / eps_1

    propagation = np.exp(1j * gamma * k0 * (z + z0))
    factor = _safe_ratio(1j * k0 * kap, 2.0 * (2.0 * np.pi) ** 2 * gamma)
    return factor[None, None, :] * tensor * propagation[None, None, :]


def _frequency_array(values, name, size):
    values = np.asarray(values, dtype=complex)
    if values.ndim == 0:
        return np.full(size, values, dtype=complex)
    if values.shape != (size,):
        raise ValueError(f"{name} must be scalar or have the same length as omega_ls")
    return values


def compute_ldos(
    omega_ls,
    eps_1,
    eps_2x,
    eps_2z,
    eps_3,
    d,
    hdist,
    dip_dir,
    kmax,
    num_int,
):
    """Integrate the reflected Green tensor over in-plane wavevector.

    This retains the dictionary format used by the notebooks. ``val`` contains
    the integrated projected reflected Green function; ``gf_k`` and ``k_vals``
    contain the evanescent integrand and its grid. The current light-line split
    is defined for an air upper half-space (``eps_1 = 1``).
    """

    omega_ls = np.atleast_1d(np.asarray(omega_ls, dtype=float))
    if (
        omega_ls.ndim != 1
        or not np.all(np.isfinite(omega_ls))
        or np.any(omega_ls <= 0)
    ):
        raise ValueError(
            "omega_ls must be a one-dimensional array of positive energies"
        )
    if not np.isscalar(eps_1) or not np.isclose(eps_1, 1.0):
        raise NotImplementedError(
            "compute_ldos currently requires eps_1 = 1"
        )
    if not np.isscalar(eps_3):
        raise ValueError("eps_3 must be frequency independent and scalar")
    if not np.isfinite(kmax) or kmax <= 1.0:
        raise ValueError("kmax must be finite and greater than the air light line")
    if int(num_int) != num_int or int(num_int) < 4:
        raise ValueError("num_int must be an integer of at least 4")

    direction = np.asarray(dip_dir, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("dip_dir must be a finite three-component vector")
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("dip_dir must be nonzero")
    direction = direction / norm

    eps_2x = _frequency_array(eps_2x, "eps_2x", omega_ls.size)
    eps_2z = _frequency_array(eps_2z, "eps_2z", omega_ls.size)
    k0_ls = 2.0 * np.pi * omega_ls / hc_eV_nm

    # Avoid evaluating exactly at the integrable light-line singularity. These
    # bounds retain the sampling convention used by the existing notebooks.
    propagating_k = np.linspace(0.0, 0.99, 100)
    evanescent_k = np.linspace(1.01, kmax, int(num_int))

    result = {"val": [], "gf_k": [], "k_vals": []}
    for energy_index, k0 in enumerate(k0_ls):
        common = dict(
            eps_1=eps_1,
            eps_2x_val=eps_2x[energy_index],
            eps_2z_val=eps_2z[energy_index],
            eps_3=eps_3,
            z=hdist,
            d=d,
            k0=k0,
        )

        green_propagating = g_s_aniso(kap=propagating_k, **common)
        projected_propagating = np.imag(
            np.einsum("i,ijo,j->o", direction, green_propagating, direction)
        )
        green_evanescent = g_s_aniso(kap=evanescent_k, **common)
        projected_evanescent = np.imag(
            np.einsum("i,ijo,j->o", direction, green_evanescent, direction)
        )

        integrated = simpson(projected_propagating, x=propagating_k)
        integrated += simpson(projected_evanescent, x=evanescent_k)
        result["val"].append(integrated)
        result["gf_k"].append(projected_evanescent)
        result["k_vals"].append(evanescent_k.copy())

    return result


__all__ = ["compute_ldos", "g_s_aniso", "ref_p", "ref_s"]
