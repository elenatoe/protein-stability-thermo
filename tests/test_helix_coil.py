import numpy as np
import pytest

from protein_stability_thermo.helix_coil import (
    partition_noncooperative,
    population_noncooperative,
    entropy_noncooperative,
    fraction_helix_residue_specific,
    sequence_to_k_values,
    partition_zipper,
    population_zipper,
    mean_helix_content_zipper,
    partition_matrix,
    fraction_helix_matrix,
)


# ---------------------------------------------------------------------
# 1. Non-cooperative (IIM) model - PS8 8.1A, 8.2A
# ---------------------------------------------------------------------

def test_partition_noncooperative_matches_ps8_8_1a():
    """PS8 8.1A: N=4, rho = (1+k)^4."""
    assert partition_noncooperative(N=4, k=1.0) == pytest.approx(16.0)


def test_population_noncooperative_sums_to_one():
    N, k = 4, 1.0
    total = sum(population_noncooperative(N, k, j) for j in range(N + 1))
    assert total == pytest.approx(1.0)


def test_population_noncooperative_matches_ps8_8_1a_table():
    """PS8 8.1A table at k=1, rho=16: populations are 1/16, 4/16, 6/16,
    4/16, 1/16 for j=0..4."""
    N, k = 4, 1.0
    expected = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]
    for j, exp in enumerate(expected):
        assert population_noncooperative(N, k, j) == pytest.approx(exp)


def test_entropy_noncooperative_peaks_at_k_equals_one():
    """PS8 8.2A: molar entropy has a maximum at k=1 (~2.079 R = 3 ln2 R)."""
    N = 3
    k_vals = np.linspace(0.01, 10, 500)
    entropies = [entropy_noncooperative(N, k) for k in k_vals]
    peak_k = k_vals[np.argmax(entropies)]
    assert peak_k == pytest.approx(1.0, abs=0.05)
    assert max(entropies) == pytest.approx(N * np.log(2), rel=1e-2)


def test_entropy_noncooperative_zero_at_extremes():
    """At k->0 (all coil) or k->inf (all helix), entropy should vanish
    (only one accessible configuration)."""
    N = 3
    assert entropy_noncooperative(N, k=1e-8) == pytest.approx(0.0, abs=1e-3)
    assert entropy_noncooperative(N, k=1e8) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------
# 2. Residue-specific non-cooperative model - PS8 8.3
# ---------------------------------------------------------------------

def test_fraction_helix_residue_specific_uniform_k():
    """With identical k for every residue, this should match the
    simple k/(1+k) non-cooperative fraction."""
    k = 2.0
    values = np.full(10, k)
    assert fraction_helix_residue_specific(values) == pytest.approx(k / (1 + k))


def test_fraction_helix_residue_specific_empty():
    assert fraction_helix_residue_specific(np.array([])) == 0.0


def test_sequence_to_k_values_matches_lookup():
    seq = "AAEE"
    table = {"A": 3.0, "E": 0.5}
    values = sequence_to_k_values(seq, table)
    assert values.tolist() == [3.0, 3.0, 0.5, 0.5]


def test_sequence_to_k_values_raises_on_unknown_without_default():
    with pytest.raises(KeyError):
        sequence_to_k_values("AX", {"A": 1.0})


def test_ps8_8_3_block_sequence_higher_kA_gives_more_helix_with_kE_20():
    """PS8 8.3B pattern: AAAAAAAEEEEEEAAAAAAA with kE=20 (E strongly
    helix-favoring) should give higher fraction helix than kE=0 at the
    same kA, since the E block becomes pre-folded."""
    seq = "AAAAAAAEEEEEEAAAAAAA"
    kA = 2.0
    k_values_kE0 = sequence_to_k_values(seq, {"A": kA, "E": 0.0})
    k_values_kE20 = sequence_to_k_values(seq, {"A": kA, "E": 20.0})

    fh_kE0 = fraction_helix_residue_specific(k_values_kE0)
    fh_kE20 = fraction_helix_residue_specific(k_values_kE20)

    assert fh_kE20 > fh_kE0


# ---------------------------------------------------------------------
# 3. Zipper model - PS8 8.1B-8.1I
# ---------------------------------------------------------------------

def test_partition_zipper_matches_ps8_8_1b_formula():
    """PS8 8.1B, N=4: rho = 1 + 4k*sigma + 3k^2*sigma + 2k^3*sigma + k^4*sigma."""
    N, k, sigma = 4, 0.5, 1.0
    expected = 1 + 4 * k * sigma + 3 * k**2 * sigma + 2 * k**3 * sigma + k**4 * sigma
    assert partition_zipper(N, k, sigma) == pytest.approx(expected)


def test_partition_zipper_matches_ps8_8_1e_numeric_value():
    """PS8 8.1E: k=0.00164, sigma=5000, N=4 gives rho ~ 33.84."""
    rho = partition_zipper(N=4, k=0.00164, sigma=5000)
    assert rho == pytest.approx(33.84, abs=0.01)


def test_population_zipper_sums_to_one():
    N, k, sigma = 4, 0.00164, 5000
    total = sum(population_zipper(N, k, sigma, j) for j in range(N + 1))
    assert total == pytest.approx(1.0)


def test_population_zipper_matches_ps8_8_1g_table():
    """PS8 8.1G, zipper model at k=0.00164, sigma=5000, rho=33.84:
    P(j=1) ~ 32.8/33.84."""
    N, k, sigma = 4, 0.00164, 5000
    p1 = population_zipper(N, k, sigma, j=1)
    assert p1 == pytest.approx(32.8 / 33.84, abs=0.01)


def test_zipper_j_out_of_range_raises():
    with pytest.raises(ValueError):
        population_zipper(N=4, k=1.0, sigma=1.0, j=5)


# ---------------------------------------------------------------------
# 4. Matrix (Zimm-Bragg-style) model - PS8 8.4A-8.4C
# ---------------------------------------------------------------------

def test_partition_matrix_matches_noncooperative_at_t_equals_one():
    """At t=1, the matrix model's transfer matrix reduces to the same
    growth as the non-cooperative model: rho_N(k, t=1) = (1+k)^N."""
    N, k = 5, 0.7
    rho_matrix = partition_matrix(N, k, t=1.0)
    rho_noncoop = partition_noncooperative(N, k)
    assert rho_matrix == pytest.approx(rho_noncoop, rel=1e-6)


def test_fraction_helix_matrix_matches_ps8_8_4_pattern():
    """PS8 8.4B: at N=20, t=1000, low k, fraction helix should show a
    sharp sigmoidal rise (cooperative transition) rather than a smooth
    gradual increase - matches the sharp jump seen around k~0.0015 in
    the coursework plot."""
    N, t = 20, 1000.0
    k_low = 0.0005
    k_high = 0.003
    fh_low = fraction_helix_matrix(N, k_low, t)
    fh_high = fraction_helix_matrix(N, k_high, t)

    assert fh_low < 0.3
    assert fh_high > 0.7


def test_fraction_helix_matrix_scalar_and_array_consistent():
    N, t = 10, 5.0
    k_vals = np.array([0.5, 1.0, 2.0])
    arr_result = fraction_helix_matrix(N, k_vals, t)
    scalar_results = [fraction_helix_matrix(N, float(k), t) for k in k_vals]

    assert arr_result.shape == (3,)
    for a, s in zip(arr_result, scalar_results):
        assert a == pytest.approx(s)


def test_fraction_helix_matrix_bounded_between_zero_and_one():
    N, t = 8, 3.0
    k_vals = np.linspace(0.01, 10, 20)
    fh = fraction_helix_matrix(N, k_vals, t)
    assert np.all(fh >= 0.0)
    assert np.all(fh <= 1.0)


def test_fraction_helix_matrix_increases_with_t_at_fixed_marginal_k():
    """Higher cooperativity t should push a marginally helix-favoring
    chain (k slightly > threshold) toward more helix, matching the
    qualitative behavior explored across the 8.4C k-t surface."""
    N = 15
    k = 0.8
    fh_low_t = fraction_helix_matrix(N, k, t=1.0)
    fh_high_t = fraction_helix_matrix(N, k, t=10.0)
    assert fh_high_t > fh_low_t
