import numpy as np
import pytest

from protein_stability_thermo.chc_model import (
    two_state_signal,
    fit_chc_curve,
    generate_synthetic_curve,
)


def test_two_state_signal_limits():
    """At D=0 with a strongly stable protein, signal should sit near the
    folded baseline; at very high D it should sit near the unfolded
    baseline."""
    D = np.array([0.0, 10.0])
    Y = two_state_signal(D, dG_H2O=5.0, m=2.0, y_f=1.0, m_f=0.0, y_u=-1.0, m_u=0.0)
    assert Y[0] == pytest.approx(1.0, abs=0.05)
    assert Y[1] == pytest.approx(-1.0, abs=0.05)


def test_two_state_signal_midpoint():
    """At D = Cm = dG_H2O/m, folded and unfolded populations are equal,
    so the signal should sit halfway between the two baselines."""
    dG_H2O, m = 4.0, 2.0
    Cm = dG_H2O / m
    Y = two_state_signal(
        np.array([Cm]), dG_H2O=dG_H2O, m=m, y_f=1.0, m_f=0.0, y_u=-1.0, m_u=0.0
    )
    assert Y[0] == pytest.approx(0.0, abs=1e-6)


def test_fit_recovers_known_parameters():
    """Fitting noiseless synthetic data should recover the input
    parameters to high precision."""
    true_params = dict(dG_H2O=3.5, m=1.8, y_f=1.2, m_f=0.05, y_u=-0.8, m_u=-0.02)
    D, Y = generate_synthetic_curve(**true_params, n_points=30, noise_sd=0.0, seed=1)

    result = fit_chc_curve(D, Y)

    assert result.dG_H2O == pytest.approx(true_params["dG_H2O"], rel=1e-3)
    assert result.m == pytest.approx(true_params["m"], rel=1e-3)
    assert result.Cm == pytest.approx(true_params["dG_H2O"] / true_params["m"], rel=1e-3)


def test_fit_tolerates_noise():
    """With realistic noise, the fit should still recover Cm within a
    reasonable tolerance."""
    true_params = dict(dG_H2O=3.0, m=2.0, y_f=1.0, m_f=0.0, y_u=-1.0, m_u=0.0)
    D, Y = generate_synthetic_curve(**true_params, n_points=25, noise_sd=0.03, seed=42)

    result = fit_chc_curve(D, Y)
    true_Cm = true_params["dG_H2O"] / true_params["m"]

    assert result.Cm == pytest.approx(true_Cm, abs=0.1)
    assert result.dG_H2O_err > 0
    assert result.m_err > 0


def test_fit_rejects_too_few_points():
    D = np.array([0.0, 1.0, 2.0])
    Y = np.array([1.0, 0.5, -1.0])
    with pytest.raises(ValueError):
        fit_chc_curve(D, Y)


def test_fit_rejects_mismatched_shapes():
    D = np.linspace(0, 5, 10)
    Y = np.linspace(1, -1, 8)
    with pytest.raises(ValueError):
        fit_chc_curve(D, Y)


def test_generate_synthetic_curve_shape():
    D, Y = generate_synthetic_curve(n_points=17)
    assert D.shape == (17,)
    assert Y.shape == (17,)
