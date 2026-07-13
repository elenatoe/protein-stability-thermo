import numpy as np
import pytest

from protein_stability_thermo.chc_model import (
    two_state_denaturation_signal,
    fit_denaturation_curve,
    generate_synthetic_melt,
    chc_stability_curve,
    fit_chc_stability_curve,
    generate_synthetic_stability_curve,
)


# ---------------------------------------------------------------------
# Stage 1: two-state denaturation curve
# ---------------------------------------------------------------------

def test_denaturation_signal_matches_native_at_low_D():
    """At D=0 with a strongly folded protein (very negative dG_H2O),
    signal should sit at the native baseline."""
    D = np.array([0.0])
    y = two_state_denaturation_signal(
        D, a_N=-1.0, b_N=0.0, a_D=1.0, b_D=0.0, dG_H2O=-20.0, m=6.0, T=298.15
    )
    assert y[0] == pytest.approx(-1.0, abs=0.01)


def test_denaturation_signal_matches_denatured_at_high_D():
    D = np.array([10.0])
    y = two_state_denaturation_signal(
        D, a_N=-1.0, b_N=0.0, a_D=1.0, b_D=0.0, dG_H2O=-20.0, m=6.0, T=298.15
    )
    assert y[0] == pytest.approx(1.0, abs=0.01)


def test_fit_recovers_known_parameters_noiseless():
    """Fitting noiseless synthetic data should recover dG_H2O and m to
    high precision, matching the workflow in PS6 6.3."""
    true_dG, true_m, T = -20.8, 6.4, 298.15
    D, signal = generate_synthetic_melt(
        T=T, dG_H2O=true_dG, m=true_m, n_points=30, noise_sd=0.0, seed=1
    )
    result = fit_denaturation_curve(D, signal, T=T)

    assert result.dG_H2O == pytest.approx(true_dG, rel=1e-3)
    assert result.m == pytest.approx(true_m, rel=1e-3)
    assert result.T == T


def test_fit_tolerates_noise():
    """dG_H2O and m are correlated in this fit (this is a known real
    tradeoff in Gdn-HCl melts) - with noise, either can drift while
    their ratio Cm = -dG_H2O/m stays well-constrained near the actual
    transition midpoint. Check the well-constrained quantity."""
    true_dG, true_m, T = -21.3, 6.5, 303.15
    true_Cm = -true_dG / true_m
    D, signal = generate_synthetic_melt(
        T=T, dG_H2O=true_dG, m=true_m, n_points=30, noise_sd=0.3, seed=7
    )
    result = fit_denaturation_curve(D, signal, T=T)
    fit_Cm = -result.dG_H2O / result.m

    assert fit_Cm == pytest.approx(true_Cm, abs=0.3)
    assert result.dG_H2O_err > 0


def test_fit_rejects_too_few_points():
    D = np.array([0.0, 1.0, 2.0])
    signal = np.array([1.0, 0.5, -1.0])
    with pytest.raises(ValueError):
        fit_denaturation_curve(D, signal, T=298.15)


def test_fit_rejects_mismatched_shapes():
    D = np.linspace(0, 5, 10)
    signal = np.linspace(1, -1, 8)
    with pytest.raises(ValueError):
        fit_denaturation_curve(D, signal, T=298.15)


# ---------------------------------------------------------------------
# Stage 2: CHC stability curve
# ---------------------------------------------------------------------

def test_chc_stability_curve_matches_ps6_reference_values():
    """PS6 6.4 fit dG_H2O(T) data from real melts at 8 temperatures and
    obtained Th=288.71 K, Ts=294.29 K, dCp=-3.81 kJ/mol/K. Reproduce
    that fit here as a regression check against the real coursework
    result."""
    T = np.array([293.15, 298.15, 303.15, 313.15, 318.15, 323.15, 328.15, 333.15])
    dG = np.array(
        [-21.285983, -20.799230, -21.355305, -18.047234,
         -17.637077, -16.917975, -13.863625, -11.701674]
    )

    result = fit_chc_stability_curve(T, dG)

    assert result.Th == pytest.approx(288.71, abs=0.5)
    assert result.Ts == pytest.approx(294.29, abs=0.5)
    assert result.dCp == pytest.approx(-3.81, abs=0.1)


def test_chc_fit_recovers_known_parameters_noiseless():
    true_Th, true_Ts, true_dCp = 288.0, 294.0, -3.5
    T, dG = generate_synthetic_stability_curve(
        Th=true_Th, Ts=true_Ts, dCp=true_dCp, noise_sd=0.0, seed=3
    )
    result = fit_chc_stability_curve(T, dG)

    assert result.Th == pytest.approx(true_Th, abs=0.1)
    assert result.Ts == pytest.approx(true_Ts, abs=0.1)
    assert result.dCp == pytest.approx(true_dCp, abs=0.05)


def test_chc_stability_curve_has_a_stability_maximum():
    """The CHC curve should have a single interior maximum (most stable
    temperature) between Th and Ts, not be monotonic - matches the
    U-shaped stability curves plotted in PS6 6.2A."""
    T = np.linspace(250, 360, 500)
    dG = chc_stability_curve(T, Th=288.0, Ts=294.0, dCp=-3.5)
    min_idx = np.argmin(dG)
    assert 0 < min_idx < len(T) - 1


def test_chc_fit_rejects_too_few_points():
    T = np.array([290.0, 300.0])
    dG = np.array([-20.0, -18.0])
    with pytest.raises(ValueError):
        fit_chc_stability_curve(T, dG)


def test_chc_fit_rejects_mismatched_shapes():
    T = np.linspace(280, 330, 8)
    dG = np.linspace(-20, -12, 6)
    with pytest.raises(ValueError):
        fit_chc_stability_curve(T, dG)
