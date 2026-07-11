"""
chc_model.py

Two-state chemical denaturation curve (CHC) fitting using the linear
extrapolation model (LEM).

Model
-----
At denaturant concentration D, the observed signal (e.g. CD ellipticity or
fluorescence) is a population-weighted average of folded and unfolded
baselines, where the folded fraction is set by a two-state equilibrium
whose free energy varies linearly with denaturant concentration:

    dG(D) = dG_H2O - m * D

    K(D) = exp(-dG(D) / RT)
    f_unfolded(D) = K(D) / (1 + K(D))

    Y_folded(D)   = y_f + m_f * D      (folded baseline, linear in D)
    Y_unfolded(D) = y_u + m_u * D      (unfolded baseline, linear in D)

    Y_obs(D) = Y_folded(D) * (1 - f_unfolded(D)) + Y_unfolded(D) * f_unfolded(D)

Fit parameters: dG_H2O, m, y_f, m_f, y_u, m_u  (6 total)

Derived quantity: Cm = dG_H2O / m  (denaturant midpoint concentration)

Units
-----
- D in molar (M), e.g. urea or GdnHCl concentration
- dG_H2O in kcal/mol
- m in kcal/mol/M
- T in Kelvin (default 298.15 K)
- R = 0.0019872 kcal/(mol*K)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

R = 0.0019872  # kcal / (mol * K)


@dataclass
class CHCFitResult:
    """Container for a completed CHC fit."""

    dG_H2O: float
    m: float
    y_f: float
    m_f: float
    y_u: float
    m_u: float
    dG_H2O_err: float
    m_err: float
    Cm: float
    Cm_err: float
    covariance: np.ndarray

    def summary(self) -> str:
        return (
            f"dG(H2O) = {self.dG_H2O:.3f} +/- {self.dG_H2O_err:.3f} kcal/mol\n"
            f"m-value = {self.m:.3f} +/- {self.m_err:.3f} kcal/mol/M\n"
            f"Cm      = {self.Cm:.3f} +/- {self.Cm_err:.3f} M"
        )


def two_state_signal(
    D: np.ndarray,
    dG_H2O: float,
    m: float,
    y_f: float,
    m_f: float,
    y_u: float,
    m_u: float,
    T: float = 298.15,
) -> np.ndarray:
    """
    Evaluate the two-state LEM model at denaturant concentrations D.

    Parameters
    ----------
    D : array-like
        Denaturant concentration(s) in M.
    dG_H2O : float
        Free energy of unfolding in water, kcal/mol.
    m : float
        m-value, kcal/mol/M.
    y_f, m_f : float
        Folded baseline intercept and slope.
    y_u, m_u : float
        Unfolded baseline intercept and slope.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Predicted signal at each D.
    """
    D = np.asarray(D, dtype=float)
    dG = dG_H2O - m * D
    # clip to avoid overflow in exp for very negative dG (strongly unfolded)
    exponent = np.clip(-dG / (R * T), -700, 700)
    K = np.exp(exponent)
    f_u = K / (1.0 + K)

    Y_folded = y_f + m_f * D
    Y_unfolded = y_u + m_u * D
    return Y_folded * (1.0 - f_u) + Y_unfolded * f_u


def fit_chc_curve(
    D: np.ndarray,
    Y: np.ndarray,
    T: float = 298.15,
    p0: Optional[tuple] = None,
    sigma: Optional[np.ndarray] = None,
) -> CHCFitResult:
    """
    Fit a two-state CHC denaturation curve to data.

    Parameters
    ----------
    D : array-like
        Denaturant concentrations, M.
    Y : array-like
        Observed signal at each D.
    T : float
        Temperature in Kelvin.
    p0 : tuple, optional
        Initial guess (dG_H2O, m, y_f, m_f, y_u, m_u). If None, a heuristic
        guess is built from the data.
    sigma : array-like, optional
        Per-point measurement uncertainty, passed to curve_fit for
        weighted least squares.

    Returns
    -------
    CHCFitResult
    """
    D = np.asarray(D, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if D.shape != Y.shape:
        raise ValueError("D and Y must have the same shape")
    if D.size < 6:
        raise ValueError("Need at least 6 points to fit 6 parameters")

    if p0 is None:
        # heuristic initial guess:
        # folded baseline ~ first two points, unfolded baseline ~ last two points
        y_f0 = float(Y[0])
        y_u0 = float(Y[-1])
        m_f0 = 0.0
        m_u0 = 0.0
        Cm0 = float(D[np.argmin(np.abs(Y - (y_f0 + y_u0) / 2.0))])
        m0 = 2.0  # kcal/mol/M, typical small-protein m-value
        dG0 = m0 * Cm0
        p0 = (dG0, m0, y_f0, m_f0, y_u0, m_u0)

    def model(D, dG_H2O, m, y_f, m_f, y_u, m_u):
        return two_state_signal(D, dG_H2O, m, y_f, m_f, y_u, m_u, T=T)

    popt, pcov = curve_fit(model, D, Y, p0=p0, sigma=sigma, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    dG_H2O, m, y_f, m_f, y_u, m_u = popt
    dG_H2O_err, m_err = perr[0], perr[1]

    Cm = dG_H2O / m
    # propagate error on Cm = dG/m via standard error propagation,
    # including the covariance term between dG and m
    cov_dG_m = pcov[0, 1]
    Cm_err = abs(Cm) * np.sqrt(
        (dG_H2O_err / dG_H2O) ** 2
        + (m_err / m) ** 2
        - 2 * cov_dG_m / (dG_H2O * m)
    )

    return CHCFitResult(
        dG_H2O=dG_H2O,
        m=m,
        y_f=y_f,
        m_f=m_f,
        y_u=y_u,
        m_u=m_u,
        dG_H2O_err=dG_H2O_err,
        m_err=m_err,
        Cm=Cm,
        Cm_err=Cm_err,
        covariance=pcov,
    )


def generate_synthetic_curve(
    dG_H2O: float = 3.0,
    m: float = 2.0,
    y_f: float = 1.0,
    m_f: float = 0.0,
    y_u: float = -1.0,
    m_u: float = 0.0,
    D_range: tuple = (0.0, 6.0),
    n_points: int = 25,
    noise_sd: float = 0.02,
    T: float = 298.15,
    seed: Optional[int] = None,
) -> tuple:
    """
    Generate a synthetic CHC denaturation curve for testing/validation.

    Returns
    -------
    D, Y : np.ndarray
        Denaturant concentrations and noisy synthetic signal.
    """
    rng = np.random.default_rng(seed)
    D = np.linspace(D_range[0], D_range[1], n_points)
    Y_clean = two_state_signal(D, dG_H2O, m, y_f, m_f, y_u, m_u, T=T)
    Y = Y_clean + rng.normal(0, noise_sd, size=D.shape)
    return D, Y
