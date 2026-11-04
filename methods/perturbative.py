"""Perturbative corrections to a fitted few-mode spectral density.

These routines operate on the residual spectral density
``delta_J = J_full - J_few_mode``. Energies, spectral densities, Lamb shifts,
and decay rates are expressed in eV throughout.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline


_QUAD_LIMIT = 600


def _residual_spline(omegas, residual):
    """Validate sampled residual data and return its cubic spline."""

    omegas = np.asarray(omegas, dtype=float)
    residual = np.asarray(residual, dtype=float)

    if omegas.ndim != 1 or residual.ndim != 1:
        raise ValueError("omegas and residual must be one-dimensional")
    if omegas.size < 2:
        raise ValueError("at least two spectral samples are required")
    if omegas.shape != residual.shape:
        raise ValueError("omegas and residual must have the same shape")
    if not np.all(np.isfinite(omegas)) or not np.all(np.isfinite(residual)):
        raise ValueError("omegas and residual must contain only finite values")
    if np.any(np.diff(omegas) <= 0.0):
        raise ValueError("omegas must be strictly increasing")

    return omegas, CubicSpline(omegas, residual)


def _frequency_scalar(omega_e, omegas):
    """Validate a scalar emitter energy within the sampled interval."""

    omega_e = np.asarray(omega_e)
    if omega_e.ndim != 0 or np.iscomplexobj(omega_e):
        raise TypeError("omega_e must be a real scalar")
    omega_e = float(omega_e)
    if not np.isfinite(omega_e):
        raise ValueError("omega_e must be finite")
    if not omegas[0] <= omega_e <= omegas[-1]:
        raise ValueError("omega_e must lie within the sampled frequency interval")
    return omega_e


def check_perturb_fm(omega_e, omegas, dJ):
    """Estimate perturbative validity above and below an emitter resonance.

    The residual spectrum is partitioned at ``omega_e``. For each side, this
    routine calculates an integrated spectral weight and its weighted mean
    frequency, then returns the dimensionless parameters
    ``beta = g_eff^2 / (omega_eff - omega_e)^2``.

    Returns
    -------
    list of float
        ``[beta_plus, beta_minus]`` for frequencies above and below the emitter.
        The list return type is retained for notebook compatibility.
    """

    omegas, spline = _residual_spline(omegas, dJ)
    omega_e = _frequency_scalar(omega_e, omegas)

    g2_plus = quad(spline, omega_e, omegas[-1], limit=_QUAD_LIMIT)[0]
    g2_minus = quad(spline, omegas[0], omega_e, limit=_QUAD_LIMIT)[0]

    def weighted_spectrum(omega):
        return omega * spline(omega)

    numerator_plus = quad(
        weighted_spectrum, omega_e, omegas[-1], limit=_QUAD_LIMIT
    )[0]
    numerator_minus = quad(
        weighted_spectrum, omegas[0], omega_e, limit=_QUAD_LIMIT
    )[0]

    if np.isclose(g2_plus, 0.0) or np.isclose(g2_minus, 0.0):
        raise ValueError(
            "the integrated residual spectral weight must be nonzero on both "
            "sides of omega_e"
        )

    omega_plus = numerator_plus / g2_plus
    omega_minus = numerator_minus / g2_minus
    beta_plus = g2_plus / (omega_plus - omega_e) ** 2
    beta_minus = g2_minus / (omega_minus - omega_e) ** 2
    return [beta_plus, beta_minus]


def lamb_shift_fm(omega_e, omegas, dJs):
    """Calculate the residual contribution to the emitter Lamb shift.

    The modified principal-value integral is evaluated as

    ``integral [(delta_J(omega) - delta_J(omega_e)) / (omega - omega_e)] d omega``.

    Subtracting the resonant value makes the singularity removable; its limiting
    value is the derivative of the interpolated residual spectrum.
    """

    omegas, spline = _residual_spline(omegas, dJs)
    omega_e = _frequency_scalar(omega_e, omegas)

    def integrand(omega):
        separation = omega - omega_e
        if abs(separation) < 1e-8:
            return float(spline(omega_e, 1))
        return float((spline(omega) - spline(omega_e)) / separation)

    shift, _ = quad(
        integrand,
        omegas[0],
        omegas[-1],
        points=[omega_e],
        limit=_QUAD_LIMIT,
    )
    return shift


def decay_rate_fm(omega_e, omegas, dJ):
    """Calculate ``2 pi delta_J(omega_e)`` for scalar or array energies."""

    omegas, spline = _residual_spline(omegas, dJ)
    omega_e = np.asarray(omega_e, dtype=float)
    if not np.all(np.isfinite(omega_e)):
        raise ValueError("omega_e must contain only finite values")
    if np.any((omega_e < omegas[0]) | (omega_e > omegas[-1])):
        raise ValueError("omega_e must lie within the sampled frequency interval")

    rate = 2.0 * np.pi * spline(omega_e)
    return float(rate) if rate.ndim == 0 else np.asarray(rate)


__all__ = ["check_perturb_fm", "decay_rate_fm", "lamb_shift_fm"]
