"""Fit a few-mode model to a spectral density and analyse its modes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eig

from .tools import hc_eV_nm


# Vacuum permittivity in the eV, elementary-charge, and nanometre unit system.
EPSILON_0_E2_PER_EV_NM = 55.264e-3


class FewModeFit:
    """Fit and analyse a few-mode representation of a spectral density.

    Parameters
    ----------
    frequencies : array_like
        Frequency/energy samples in eV.
    spectral_density : array_like
        Spectral-density values at ``frequencies``.
    hamiltonian : array_like
        Initial ``(n_modes, n_modes)`` few-mode Hamiltonian in eV.
    decay_rates : array_like
        Initial decay rate of each mode in eV.
    couplings : array_like
        Initial emitter-mode couplings with shape ``(n_emitters, n_modes)``.

    Notes
    -----
    The fitting backend is imported only when :meth:`initialize` is called,
    because ``spectral_density_fit`` is a separate research package.
    """

    def __init__(
        self,
        frequencies,
        spectral_density,
        hamiltonian,
        decay_rates,
        couplings,
    ):
        self.w = np.asarray(frequencies)
        self.J = np.asarray(spectral_density).squeeze()
        self.H = np.asarray(hamiltonian)
        self.K = np.asarray(decay_rates)
        self.g = np.asarray(couplings)

        self._validate_inputs()
        self.Nm = self.H.shape[0]
        self.eps0 = EPSILON_0_E2_PER_EV_NM
        self.opt = None
        self.ps = None

    def _validate_inputs(self):
        if self.w.ndim != 1 or self.J.ndim != 1:
            raise ValueError("frequencies and spectral_density must be one-dimensional")
        if self.w.shape != self.J.shape:
            raise ValueError("frequencies and spectral_density must have equal lengths")
        if self.H.ndim != 2 or self.H.shape[0] != self.H.shape[1]:
            raise ValueError("hamiltonian must be a square matrix")

        n_modes = self.H.shape[0]
        if self.K.shape != (n_modes,):
            raise ValueError("decay_rates must contain one value per mode")

        if self.g.ndim == 1:
            self.g = self.g[np.newaxis, :]
        if self.g.ndim != 2 or self.g.shape[1] != n_modes:
            raise ValueError("couplings must have shape (n_emitters, n_modes)")
        if self.g.shape[0] != 1:
            raise NotImplementedError(
                "Figures of merit are currently implemented for one emitter only"
            )

    def initialize(self, set_limits=True, plot=True):
        """Initialize the fitter from the supplied few-mode parameters."""
        try:
            from spectral_density_fit import spectral_density_fitter
        except ImportError as exc:
            raise ImportError(
                "Few-mode fitting requires the 'spectral_density_fit' package."
            ) from exc

        fitter_kwargs = {} if set_limits else {"λlims": False}
        self.opt = spectral_density_fitter(self.w, self.J, self.Nm, **fitter_kwargs)
        self.ps = self.opt.Hκg_to_ps(self.H, self.K, self.g)

        if plot:
            initial_spectrum = self.evaluate_spectral_density(self.w)
            return self._plot_spectral_density(initial_spectrum, "Initial guess")
        return None

    def fit(self, plot=True):
        """Optimize the few-mode parameters and return the fit error."""
        self._require_initialized()
        self.ps = self.opt.optimize(self.ps)
        fit_error = float(self.opt.obj_fun(self.ps, np.empty(0)))

        if plot:
            fitted_spectrum = self.evaluate_spectral_density(self.w)
            self._plot_spectral_density(fitted_spectrum, "Fit")

        print(f"Fit error: {fit_error:.6g}")
        return fit_error

    def evaluate_spectral_density(self, frequencies):
        """Evaluate the current few-mode spectral density."""
        self._require_initialized()
        frequencies = np.asarray(frequencies)
        return np.asarray(self.opt.Jfun(frequencies, self.ps)).squeeze()

    def create_interpolator(self, w_min, w_max, n_points=1000):
        """Return a cubic interpolator for the fitted spectral density.

        Values outside ``[w_min, w_max]`` are set to zero.
        """
        if w_min >= w_max:
            raise ValueError("w_min must be smaller than w_max")
        if n_points < 4:
            raise ValueError("n_points must be at least 4 for cubic interpolation")

        frequency_grid = np.linspace(w_min, w_max, n_points)
        density_grid = self.evaluate_spectral_density(frequency_grid)
        spline = CubicSpline(frequency_grid, density_grid, extrapolate=False)

        def interpolator(frequencies):
            values = spline(frequencies)
            return np.where(np.isnan(values), 0.0, values)

        return interpolator

    def fitted_parameters(self):
        """Return fitted ``(H, decay_rates, couplings)`` NumPy arrays.

        Modes are ordered by the real part of the diagonal Hamiltonian entries.
        """
        self._require_initialized()
        hamiltonian, decay_rates, couplings = map(
            np.asarray, self.opt.ps_to_Hκg(self.ps)
        )
        hamiltonian, decay_rates, couplings, _ = self._sort_by_frequency(
            hamiltonian, decay_rates, couplings
        )
        return hamiltonian, decay_rates, couplings

    def compute_figures_of_merit(self, plot=True):
        """Compute complex resonances, quality factors, and mode volumes.

        Returns
        -------
        resonances : numpy.ndarray
            Complex eigenfrequencies of the lossy few-mode Hamiltonian.
        quality_factors : numpy.ndarray
            Mode quality factors ``-Re(Omega) / (2 Im(Omega))``.
        normalized_mode_volumes : numpy.ndarray
            Effective mode volumes normalized by the cubic free-space
            wavelength corresponding to each resonance.
        """
        self.diagonalize_lossy_hamiltonian()

        quality_factors = -self.Omega.real / (2 * self.Omega.imag)
        effective_volumes = self.Omega.real / (2 * self.eps0 * self.g2)
        wavelengths_nm = hc_eV_nm / self.Omega.real
        normalized_mode_volumes = effective_volumes / wavelengths_nm**3

        self.quality_factors = quality_factors
        self.effective_mode_volumes = effective_volumes
        self.normalized_mode_volumes = normalized_mode_volumes

        if plot:
            self._plot_figures_of_merit(
                self.Omega, quality_factors, normalized_mode_volumes
            )

        return self.Omega, quality_factors, normalized_mode_volumes

    def display(self):
        """Print the fitted Hamiltonian, decay rates, and couplings."""
        hamiltonian, decay_rates, couplings = self.fitted_parameters()
        print("Hamiltonian:\n", hamiltonian)
        print("Decay rates:\n", decay_rates)
        print("Couplings:\n", couplings)

    def diagonalize_lossy_hamiltonian(self):
        hamiltonian, decay_rates, couplings = self.fitted_parameters()
        lossy_hamiltonian = hamiltonian.astype(complex, copy=True)
        diagonal = np.diag_indices_from(lossy_hamiltonian)
        lossy_hamiltonian[diagonal] -= 0.5j * decay_rates

        resonances, eigenvectors = eig(lossy_hamiltonian)
        order = np.argsort(resonances.real)
        resonances = resonances[order]
        eigenvectors = eigenvectors[:, order]
        transformed_couplings = eigenvectors.T @ couplings.squeeze(axis=0)

        self.Omega = resonances
        self.g_tilde = transformed_couplings
        self.g2 = transformed_couplings**2
        return resonances, transformed_couplings

    def _require_initialized(self):
        if self.opt is None or self.ps is None:
            raise RuntimeError("Call initialize() or init() before this operation.")

    @staticmethod
    def _sort_by_frequency(hamiltonian, decay_rates, couplings):
        order = np.argsort(np.diag(hamiltonian).real)
        sorted_hamiltonian = hamiltonian[np.ix_(order, order)]
        sorted_decay_rates = decay_rates[order]
        sorted_couplings = couplings[:, order]
        return sorted_hamiltonian, sorted_decay_rates, sorted_couplings, order

    def _plot_spectral_density(self, model_spectrum, model_label):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.w, self.J, label="Data")
        ax.plot(self.w, model_spectrum, "--", label=model_label)
        ax.set(xlabel=r"$\omega$ (eV)", ylabel=r"$J(\omega) / \mu^2$")
        ax.legend()
        return fig, ax

    def _plot_figures_of_merit(self, resonances, quality_factors, mode_volumes):
        energies = resonances.real
        fig, axes = plt.subplots(figsize=(10, 4), ncols=2)

        axes[0].plot(energies, quality_factors, "o")
        axes[0].set(xlabel=r"$\omega$ (eV)", ylabel=r"$Q$")

        axes[1].plot(energies, mode_volumes.real, "o")
        axes[1].set(
            xlabel=r"$\omega$ (eV)",
            ylabel=r"$V_\mathrm{eff}/\lambda^3$",
        )

        for axis in axes:
            for energy in energies:
                axis.axvline(energy, linestyle="--", linewidth=1, color="black")
            spectral_axis = axis.twinx()
            spectral_axis.plot(self.w, self.J, color="tab:green", alpha=0.7)
            spectral_axis.set_ylabel(r"$J(\omega)$")

        return fig, axes


__all__ = ["FewModeFit"]
