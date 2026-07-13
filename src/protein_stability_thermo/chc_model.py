"""
chc_model.py

Protein stability modeling in two stages, matching BPC PS6 (problems 6.3
and 6.4):

Stage 1 - chemical (Gdn-HCl) denaturation curve fitting, per temperature.
    Six-parameter two-state fit of a CD (or other optical) signal vs
    denaturant concentration, following the linear extrapolation model:

        dG(D) = dG_H2O + m * D
        K(D)  = exp(-dG(D) / RT)
        f_N(D) = K(D) / (1 + K(D))          [dominant-state population]
        y_N(D) = a_N + b_N * D              (native baseline)
        y_D(D) = a_D + b_D * D              (denatured baseline)
        y_obs(D) = y_N(D) * f_N(D) + y_D(D) * (1 - f_N(D))

    Fitting this at several temperatures gives a table of (T, dG_H2O(T),
    m(T)) - exactly the output of PS6 problem 6.3, repeated per melt.

Stage 2 - CHC (constant heat capacity) stability curve fit.
    The dG_H2O(T) values from stage 1 are fit to the constant-Cp
    Gibbs-Helmholtz form:

        dG(T) = dCp * (T - Th) - T * dCp * ln(T / Ts)

    where Th is the temperature of maximum enthalpy (dH = 0), Ts is the
    temperature of maximum entropy (dS = 0), and dCp is the (assumed
    temperature-independent) heat capacity change on unfolding. This is
    the "CHC model" referenced throughout PS6 problem 6.4.

Units: T in Kelvin, denaturant concentration in M, dG in kJ/mol (matches
the coursework), R = 0.0083145 kJ/(mol*K).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit

R = 0.0083145  # kJ / (mol * K), matches PS6 convention


# ---------------------------------------------------------------------
# Stage 1: two-state chemical denaturation curve (per temperature)
# ---------------------------------------------------------------------

@dataclass
class DenaturationFitResult:
    """Result of a single-temperature two-state Gdn-HCl melt fit."""

    T: float
    a_N: float
    b_N: float
    a_D: float
    b_D: float
    dG_H2O: float
    m: float
    dG_H2O_err: float
    m_err: float
    covariance: np.ndarray

    def summary(self) -> str:
        return (
            f"T = {self.T:.2f} K\n"
            f"dG(H2O) = {self.dG_H2O:.3f} +/- {self.dG_H2O_err:.3f} kJ/mol\n"
            f"m-value = {self.m:.3f} +/- {self.m_err:.3f} kJ/mol/M"
        )


def two_state_denaturation_signal(
    D: np.ndarray,
    a_N: float,
    b_N: float,
    a_D: float,
    b_D: float,
    dG_H2O: float,
    m: float,
    T: float,
) -> np.ndarray:
    """
    Evaluate the two-state Gdn-HCl denaturation model at denaturant
    concentrations D, following PS6 6.3's gdn_eqn convention:
    dG(D) = dG_H2O + m*D, K = exp(-dG/RT), f_N = K/(1+K).

    Note this sign convention (dG_H2O typically negative, m typically
    positive) is the one used in the coursework; it differs from the
    "dG = dG_H2O - m*D" convention sometimes seen elsewhere, so don't
    mix the two.
    """
    D = np.asarray(D, dtype=float)
    dG = dG_H2O + m * D
    K = np.exp(-dG / (R * T))
    f_N = K / (1.0 + K)
    y_N = a_N + b_N * D
    y_D = a_D + b_D * D
    return y_N * f_N + y_D * (1.0 - f_N)


def fit_denaturation_curve(
    D: np.ndarray,
    signal: np.ndarray,
    T: float,
    p0: Optional[tuple] = None,
) -> DenaturationFitResult:
    """
    Fit a single-temperature Gdn-HCl melt to the two-state model.

    Parameters
    ----------
    D : array-like
        Denaturant concentrations, M.
    signal : array-like
        Observed CD (or other) signal at each D.
    T : float
        Temperature of this melt, Kelvin (fixed, not fit - matches PS6
        workflow where each melt is fit at its own known T).
    p0 : tuple, optional
        Initial guess (a_N, b_N, a_D, b_D, dG_H2O, m). If omitted, a
        heuristic guess is built from the data.

    Returns
    -------
    DenaturationFitResult
    """
    D = np.asarray(D, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if D.shape != signal.shape:
        raise ValueError("D and signal must have the same shape")
    if D.size < 6:
        raise ValueError("Need at least 6 points to fit 6 parameters")

    if p0 is None:
        a_N0 = float(signal[0])
        a_D0 = float(signal[-1])
        Dm0 = float(D[np.argmin(np.abs(signal - (a_N0 + a_D0) / 2.0))])
        m0 = 6.0  # kJ/mol/M, typical Gdn-HCl m-value
        dG0 = -m0 * Dm0
        p0 = (a_N0, 0.0, a_D0, 0.0, dG0, m0)

    def model(D, a_N, b_N, a_D, b_D, dG_H2O, m):
        return two_state_denaturation_signal(D, a_N, b_N, a_D, b_D, dG_H2O, m, T)

    popt, pcov = curve_fit(model, D, signal, p0=p0, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    a_N, b_N, a_D, b_D, dG_H2O, m = popt

    return DenaturationFitResult(
        T=T,
        a_N=a_N,
        b_N=b_N,
        a_D=a_D,
        b_D=b_D,
        dG_H2O=dG_H2O,
        m=m,
        dG_H2O_err=perr[4],
        m_err=perr[5],
        covariance=pcov,
    )


def generate_synthetic_melt(
    T: float,
    a_N: float = -1.0,
    b_N: float = 0.0,
    a_D: float = 1.0,
    b_D: float = 0.0,
    dG_H2O: float = -20.0,
    m: float = 6.0,
    D_range: tuple = (0.0, 7.0),
    n_points: int = 30,
    noise_sd: float = 0.3,
    seed: Optional[int] = None,
) -> tuple:
    """Generate a synthetic single-temperature Gdn-HCl melt for testing."""
    rng = np.random.default_rng(seed)
    D = np.linspace(D_range[0], D_range[1], n_points)
    clean = two_state_denaturation_signal(D, a_N, b_N, a_D, b_D, dG_H2O, m, T)
    signal = clean + rng.normal(0, noise_sd, size=D.shape)
    return D, signal


# ---------------------------------------------------------------------
# Stage 2: CHC (constant heat capacity) stability curve
# ---------------------------------------------------------------------

@dataclass
class CHCStabilityFitResult:
    """Result of fitting dG_H2O(T) across temperatures to the CHC model."""

    Th: float
    Ts: float
    dCp: float
    Th_err: float
    Ts_err: float
    dCp_err: float
    covariance: np.ndarray

    def summary(self) -> str:
        return (
            f"Th   = {self.Th:.2f} +/- {self.Th_err:.2f} K  (enthalpy convergence temp)\n"
            f"Ts   = {self.Ts:.2f} +/- {self.Ts_err:.2f} K  (entropy convergence temp)\n"
            f"dCp  = {self.dCp:.3f} +/- {self.dCp_err:.3f} kJ/mol/K"
        )


def chc_stability_curve(T: np.ndarray, Th: float, Ts: float, dCp: float) -> np.ndarray:
    """
    Constant heat capacity (CHC) model for the temperature dependence of
    protein stability, matching PS6 problem 6.4:

        dG(T) = dCp * (T - Th) - T * dCp * ln(T / Ts)

    Parameters
    ----------
    T : array-like
        Temperature(s), Kelvin.
    Th : float
        Temperature at which dH = 0 (enthalpy convergence temperature).
    Ts : float
        Temperature at which dS = 0 (entropy convergence temperature).
    dCp : float
        Heat capacity change on unfolding, kJ/mol/K (typically negative
        for folding free energy defined as in this module's convention).

    Returns
    -------
    np.ndarray
        Predicted dG_H2O at each T.
    """
    T = np.asarray(T, dtype=float)
    return dCp * (T - Th) - T * dCp * np.log(T / Ts)


def fit_chc_stability_curve(
    T: Sequence[float],
    dG_H2O: Sequence[float],
    p0: Optional[tuple] = None,
) -> CHCStabilityFitResult:
    """
    Fit the CHC stability curve to a set of (T, dG_H2O) points, e.g. the
    per-temperature outputs of fit_denaturation_curve() collected across
    several melts (PS6 problem 6.4 workflow).

    Parameters
    ----------
    T : array-like
        Temperatures, Kelvin.
    dG_H2O : array-like
        Free energy of folding in water at each T, kJ/mol.
    p0 : tuple, optional
        Initial guess (Th, Ts, dCp). Defaults to a typical globular
        protein guess if omitted.

    Returns
    -------
    CHCStabilityFitResult
    """
    T = np.asarray(T, dtype=float)
    dG_H2O = np.asarray(dG_H2O, dtype=float)

    if T.shape != dG_H2O.shape:
        raise ValueError("T and dG_H2O must have the same shape")
    if T.size < 3:
        raise ValueError("Need at least 3 temperature points to fit 3 parameters")

    if p0 is None:
        p0 = (T.mean() - 10, T.mean() - 5, -2.0)

    popt, pcov = curve_fit(chc_stability_curve, T, dG_H2O, p0=p0, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    Th, Ts, dCp = popt

    return CHCStabilityFitResult(
        Th=Th,
        Ts=Ts,
        dCp=dCp,
        Th_err=perr[0],
        Ts_err=perr[1],
        dCp_err=perr[2],
        covariance=pcov,
    )


def generate_synthetic_stability_curve(
    Th: float = 288.0,
    Ts: float = 294.0,
    dCp: float = -3.5,
    T_range: tuple = (280.0, 335.0),
    n_points: int = 8,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> tuple:
    """Generate synthetic (T, dG_H2O) points for testing the CHC fit."""
    rng = np.random.default_rng(seed)
    T = np.linspace(T_range[0], T_range[1], n_points)
    clean = chc_stability_curve(T, Th, Ts, dCp)
    dG = clean + rng.normal(0, noise_sd, size=T.shape)
    return T, dG
