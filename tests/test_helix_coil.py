import numpy as np
import pytest

from protein_stability_thermo.helix_coil import (
    s_from_ddG,
    propensity_scale_to_s,
    helix_probabilities,
    mean_helix_content,
    PACE_SCHOLTZ_DDG,
)


def test_s_from_ddG_zero_gives_one():
    """ddG = 0 (Ala reference) should give s = 1 exactly."""
    assert s_from_ddG(0.0) == pytest.approx(1.0)


def test_s_from_ddG_positive_gives_less_than_one():
    """A positive ddG (less helix-favorable than Ala) should give s < 1."""
    assert s_from_ddG(1.0) < 1.0


def test_probabilities_shape_and_bounds():
    s = np.full(20, 1.5)
    probs = helix_probabilities(s, sigma=1e-3)
    assert probs.shape == (20,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_empty_sequence_returns_empty():
    probs = helix_probabilities(np.array([]))
    assert probs.size == 0


def test_high_s_favors_helix():
    """A chain with strongly helix-favoring s at every position should
    have high mean helix content; a chain with strongly coil-favoring s
    should have low mean helix content."""
    n = 30
    sigma = 1e-3

    s_helix_favoring = np.full(n, 3.0)
    s_coil_favoring = np.full(n, 0.3)

    high = mean_helix_content(s_helix_favoring, sigma=sigma)
    low = mean_helix_content(s_coil_favoring, sigma=sigma)

    assert high > 0.7
    assert low < 0.3
    assert high > low


def test_low_sigma_sharpens_transition():
    """Lower nucleation parameter (sigma) makes it harder to start a
    helix, which for a marginal (s close to 1) sequence should reduce
    mean helix content relative to a higher sigma."""
    n = 25
    s = np.full(n, 1.05)

    content_low_sigma = mean_helix_content(s, sigma=1e-5)
    content_high_sigma = mean_helix_content(s, sigma=1e-1)

    assert content_low_sigma < content_high_sigma


def test_proline_lowers_local_helix_probability():
    """Proline has the worst helix propensity in the Pace-Scholtz scale;
    inserting one into an otherwise strongly helical poly-Ala-like chain
    should locally suppress helix probability at that position relative
    to its neighbors."""
    seq = list("AAAAAAAAAA")
    seq[5] = "P"
    seq = "".join(seq)

    s = propensity_scale_to_s(seq)
    probs = helix_probabilities(s, sigma=1e-3)

    neighbor_avg = (probs[4] + probs[6]) / 2.0
    assert probs[5] < neighbor_avg


def test_propensity_scale_to_s_unknown_residue_falls_back_to_ala():
    s = propensity_scale_to_s("X")
    assert s[0] == pytest.approx(s_from_ddG(PACE_SCHOLTZ_DDG["A"]))


def test_propensity_scale_to_s_length_matches_sequence():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    s = propensity_scale_to_s(seq)
    assert s.shape == (len(seq),)
