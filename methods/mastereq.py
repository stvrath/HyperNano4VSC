"""Few-mode master-equation models implemented with QuTiP."""

from __future__ import annotations

import numpy as np
from qutip import enr_destroy, enr_fock, mesolve
from scipy.linalg import eig

from .tools import hbar_eV_s


class MasterEquationQtp:
    """Master equation for emitters coupled to a lossy few-mode environment.

    The model uses an excitation-number-restricted Hilbert space. Energies,
    couplings, and decay rates are expressed in eV, while :meth:`solve` accepts
    and returns time in femtoseconds.

    The class retains the workflow used throughout the repository::

        model = MasterEquationQtp(Nm, Ne, Nexc)
        model.provide_fm_pars(H, gs, kappas)
        model.set_qe_pars(omega_e, mu, gamma_e)
        model.setup_ops()
        model.initial_state()
        times_fs, solution = model.solve(t_max, n_t)

    Parameters
    ----------
    Nm : int
        Number of photonic modes.
    Ne : int
        Number of two-level emitters.
    Nexc : int
        Maximum total number of excitations retained in the Hilbert space.
    """

    def __init__(self, Nm: int, Ne: int, Nexc: int):
        self.Nm = self._positive_integer(Nm, "Nm")
        self.Ne = self._positive_integer(Ne, "Ne")
        self.Nexc = self._positive_integer(Nexc, "Nexc")

        # Parameters populated by provide_fm_pars and set_qe_pars.
        self.Hfm = None
        self.gs = None
        self.kappas = None
        self.omega_e = None
        self.mu = None
        self.gamma_e = None

        # Objects populated when preparing a time-domain calculation.
        self.me_ops = None
        self.psi0 = None

        self._setup_basis()

    @staticmethod
    def _positive_integer(value, name):
        """Validate a positive integer constructor argument."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
        return int(value)

    def _setup_basis(self):
        """Create annihilation operators in the restricted Hilbert space."""
        # Photonic modes can hold up to Nexc quanta; emitters are two-level.
        self.part_dims = [self.Nexc + 1] * self.Nm + [2] * self.Ne
        ann_ops = list(enr_destroy(self.part_dims, self.Nexc))

        self.anh_ops = {
            "p": ann_ops[: self.Nm],
            "e": ann_ops[self.Nm :],
            "a": ann_ops,
        }

    def provide_fm_pars(self, H, gs, kappas):
        """Store the fitted few-mode parameters.

        Parameters
        ----------
        H : array_like, shape (Nm, Nm)
            Hermitian few-mode Hamiltonian in eV.
        gs : array_like, shape (Ne, Nm)
            Emitter-mode couplings per unit dipole moment.
        kappas : array_like, shape (Nm,)
            Non-negative photonic-mode decay rates in eV.

        Returns
        -------
        MasterEquationQtp
            The current instance, allowing optional method chaining.
        """
        H = np.asarray(H, dtype=complex)
        gs = np.asarray(gs, dtype=complex)
        kappas = np.asarray(kappas, dtype=float)

        if H.shape != (self.Nm, self.Nm):
            raise ValueError(f"H must have shape ({self.Nm}, {self.Nm})")
        if gs.shape != (self.Ne, self.Nm):
            raise ValueError(f"gs must have shape ({self.Ne}, {self.Nm})")
        if kappas.shape != (self.Nm,):
            raise ValueError(f"kappas must have shape ({self.Nm},)")
        if not np.all(np.isfinite(H)) or not np.all(np.isfinite(gs)):
            raise ValueError("H and gs must contain only finite values")
        if not np.all(np.isfinite(kappas)) or np.any(kappas < 0):
            raise ValueError("kappas must contain finite, non-negative rates")
        if not np.allclose(H, H.conj().T):
            raise ValueError("H must be Hermitian")

        self.Hfm = H
        self.gs = gs
        self.kappas = kappas
        self.me_ops = None
        return self

    def set_qe_pars(self, omega_e, mu, gamma_e=0.0):
        """Set the emitter parameters for a fixed transition energy.

        All emitters currently share the same transition energy and dipole
        moment. ``gamma_e`` may be a scalar shared by all emitters or an array
        containing one decay rate per emitter.

        Parameters
        ----------
        omega_e : float
            Emitter transition energy in eV.
        mu : float
            Emitter dipole moment in e nm.
        gamma_e : float or array_like, optional
            Total emitter decay rate in eV.

        Returns
        -------
        MasterEquationQtp
            The current instance, allowing optional method chaining.
        """
        omega_e = self._finite_scalar(omega_e, "omega_e")
        mu = self._finite_scalar(mu, "mu")
        gamma_e = np.asarray(gamma_e, dtype=float)

        if gamma_e.ndim == 0:
            gamma_e = np.full(self.Ne, gamma_e.item())
        elif gamma_e.shape != (self.Ne,):
            raise ValueError(
                f"gamma_e must be a scalar or have shape ({self.Ne},)"
            )
        if not np.all(np.isfinite(gamma_e)) or np.any(gamma_e < 0):
            raise ValueError("gamma_e must contain finite, non-negative rates")

        self.omega_e = omega_e
        self.mu = mu
        self.gamma_e = gamma_e
        self.me_ops = None
        return self

    @staticmethod
    def _finite_scalar(value, name):
        """Return a finite real scalar as a float."""
        array = np.asarray(value)
        if array.ndim != 0 or np.iscomplexobj(array):
            raise TypeError(f"{name} must be a real scalar")
        value = float(array)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _require_parameters(self):
        """Ensure that all parameters needed to construct operators exist."""
        if self.Hfm is None:
            raise RuntimeError("Call provide_fm_pars() before constructing operators")
        if self.omega_e is None:
            raise RuntimeError("Call set_qe_pars() before constructing operators")

    def setup_ops(self, return_ops=False, store=True):
        """Construct the Hamiltonian, collapse, and population operators.

        Parameters
        ----------
        return_ops : bool, optional
            Return ``(H, c_ops, e_ops)`` instead of only storing the operators.
        store : bool, optional
            Store the operators in ``me_ops`` for a subsequent call to
            :meth:`solve`. This argument is retained for notebook compatibility.

        Returns
        -------
        tuple, optional
            ``(H, c_ops, e_ops)`` when ``return_ops`` is true.
        """
        self._require_parameters()

        mode_ops = self.anh_ops["p"]
        emitter_ops = self.anh_ops["e"]

        Hham = sum(
            self.Hfm[i, j] * mode_ops[i].dag() * mode_ops[j]
            for i in range(self.Nm)
            for j in range(self.Nm)
        )
        Hham += sum(
            self.omega_e * sigma.dag() * sigma for sigma in emitter_ops
        )

        # Normal ordering is required in the excitation-restricted space.
        Hham += sum(
            self.mu
            * (
                self.gs[i, j] * mode_ops[j].dag() * emitter_ops[i]
                + self.gs[i, j].conjugate()
                * emitter_ops[i].dag()
                * mode_ops[j]
            )
            for i in range(self.Ne)
            for j in range(self.Nm)
        )

        c_ops = [
            np.sqrt(kappa) * mode
            for kappa, mode in zip(self.kappas, mode_ops)
        ]
        c_ops.extend(
            np.sqrt(gamma) * emitter
            for gamma, emitter in zip(self.gamma_e, emitter_ops)
        )
        e_ops = [operator.dag() * operator for operator in self.anh_ops["a"]]

        if store:
            self.me_ops = {"H": Hham, "c": c_ops, "e": e_ops}
        if return_ops:
            return Hham, c_ops, e_ops
        return None

    def initial_state(self, qe_indx=0):
        """Prepare a state with one selected emitter initially excited.

        Parameters
        ----------
        qe_indx : int, optional
            Zero-based index of the initially excited emitter.

        Returns
        -------
        qutip.Qobj
            The prepared initial state.
        """
        if isinstance(qe_indx, bool) or not isinstance(qe_indx, (int, np.integer)):
            raise TypeError("qe_indx must be an integer")
        if not 0 <= qe_indx < self.Ne:
            raise IndexError(f"qe_indx must lie between 0 and {self.Ne - 1}")

        emitter_occupations = np.zeros(self.Ne, dtype=int)
        emitter_occupations[qe_indx] = 1
        occupations = np.r_[np.zeros(self.Nm, dtype=int), emitter_occupations]
        self.psi0 = enr_fock(self.part_dims, self.Nexc, occupations)
        return self.psi0

    def solve(self, t_max, n_t, t_min=0.0):
        """Solve the Lindblad master equation.

        Parameters
        ----------
        t_max, t_min : float
            Final and initial times in femtoseconds.
        n_t : int
            Number of equally spaced output times.

        Returns
        -------
        times_fs : numpy.ndarray
            Output times in femtoseconds.
        solution : qutip.Result
            QuTiP master-equation result.
        """
        if self.me_ops is None:
            raise RuntimeError("Call setup_ops() before solve()")
        if self.psi0 is None:
            raise RuntimeError("Call initial_state() before solve()")

        t_min = self._finite_scalar(t_min, "t_min")
        t_max = self._finite_scalar(t_max, "t_max")
        n_t = self._positive_integer(n_t, "n_t")
        if n_t < 2:
            raise ValueError("n_t must be at least 2")
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min")

        times_fs = np.linspace(t_min, t_max, n_t)
        # QuTiP uses hbar = 1; convert fs to the corresponding inverse-eV unit.
        times_internal = times_fs * 1e-15 / hbar_eV_s

        solution = mesolve(
            self.me_ops["H"],
            self.psi0,
            times_internal,
            self.me_ops["c"],
            e_ops=self.me_ops["e"],
        )
        return times_fs, solution

    def get_Heff(self):
        """Return the non-Hermitian effective Hamiltonian."""
        Hham, c_ops, _ = self.setup_ops(return_ops=True, store=False)
        decay_term = sum(operator.dag() * operator for operator in c_ops)
        return Hham - 0.5j * decay_term

    def diag_Heff(self):
        """Diagonalize the non-Hermitian effective Hamiltonian."""
        return eig(self.get_Heff().full())

    def get_Heff_dispersion(self, omega_e_ls, mu, gamma_e=0.0):
        """Diagonalize the effective Hamiltonian across emitter energies.

        Parameters
        ----------
        omega_e_ls : array_like, shape (n_frequencies,)
            Emitter transition energies in eV. At each sweep point, all
            emitters are assigned the same transition energy.
        mu : float
            Emitter dipole moment in e nm.
        gamma_e : float or array_like, optional
            Total emitter decay rate in eV. Supply either one constant value or
            one value for every energy in ``omega_e_ls``.

        Returns
        -------
        eigenvalues : numpy.ndarray
            Complex eigenvalues, sorted by their real parts at each frequency.
        eigenvectors : numpy.ndarray
            Correspondingly ordered eigenvectors, stored column-wise.
        """
        if self.Hfm is None:
            raise RuntimeError("Call provide_fm_pars() before sweeping energies")

        omega_e_ls = np.asarray(omega_e_ls, dtype=float)
        if omega_e_ls.ndim != 1 or omega_e_ls.size == 0:
            raise ValueError("omega_e_ls must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(omega_e_ls)):
            raise ValueError("omega_e_ls must contain only finite values")

        gamma_e = np.asarray(gamma_e, dtype=float)
        if gamma_e.ndim == 0:
            gamma_e_ls = np.full(omega_e_ls.size, gamma_e.item())
        elif gamma_e.shape == omega_e_ls.shape:
            gamma_e_ls = gamma_e
        else:
            raise ValueError(
                "gamma_e must be a scalar or have the same shape as omega_e_ls"
            )

        eigenvalues = []
        eigenvectors = []
        for omega_e, gamma_at_omega in zip(omega_e_ls, gamma_e_ls):
            self.set_qe_pars(omega_e, mu, gamma_at_omega)
            values, vectors = self.diag_Heff()
            eigenvalues.append(values)
            eigenvectors.append(vectors)

        eigenvalues = np.asarray(eigenvalues)
        eigenvectors = np.asarray(eigenvectors)

        order = np.argsort(eigenvalues.real, axis=1)
        eigenvalues = np.take_along_axis(eigenvalues, order, axis=1)
        eigenvectors = np.take_along_axis(
            eigenvectors, order[:, np.newaxis, :], axis=2
        )
        return eigenvalues, eigenvectors


__all__ = ["MasterEquationQtp"]
