"""
helix_coil.py

Helix-coil transition models, matching BPC PS8 problems 8.1-8.4. Four
models of increasing sophistication are implemented, in the same order
developed in the coursework:

1. Non-cooperative / independent-identical-molecules (IIM) model (8.1A,
   8.2A): every residue is independently helix or coil with the same
   equilibrium constant k. Partition function rho = (1+k)^N.

2. Residue-specific non-cooperative model (8.3): same independence
   assumption, but each residue type has its own k (e.g. k_A for Ala,
   k_E for Glu), so fraction helix is a per-sequence mean-field average.

3. Zipper model (8.1B-8.1I): only a single contiguous helical run is
   allowed. A run of length j has (N-j+1) possible positions along the
   chain, and starting a run carries one nucleation penalty regardless
   of its length.

4. Matrix (Zimm-Bragg-style) model (8.4A-8.4C): the exact statistical
   count via a 2x2 transfer matrix, allowing any arrangement of helix
   and coil (not just one contiguous run). Built with sympy exactly as
   in the coursework, since exact symbolic differentiation is how the
   ensemble-average fraction helix was obtained.

All models use k as the per-residue helix/coil equilibrium constant
(propagation-like parameter) and, where relevant, a second parameter
(sigma or t) controlling cooperativity/nucleation.
"""

from typing import Dict, Optional, Sequence

import math

import numpy as np
import sympy as sym


# ---------------------------------------------------------------------
# 1. Non-cooperative (independent, identical molecules) model
# ---------------------------------------------------------------------

def partition_noncooperative(N: int, k: float) -> float:
    """
    Partition function for N independent, identical helix/coil residues
    (PS8 8.1A): rho = (1 + k)^N.
    """
    return (1.0 + k) ** N


def population_noncooperative(N: int, k: float, j: int) -> float:
    """
    Probability of exactly j helical residues out of N, under the
    non-cooperative model (PS8 8.2A): binomial in k.

        P(j) = C(N, j) * k^j / (1 + k)^N
    """
    if not (0 <= j <= N):
        raise ValueError("j must be between 0 and N")
    return math.comb(N, j) * (k ** j) / partition_noncooperative(N, k)


def entropy_noncooperative(N: int, k: float) -> float:
    """
    Molar entropy (in units of R, i.e. S/R) of the non-cooperative model
    as a function of k (PS8 8.2A): peaks at k=1, where coil and helix
    states are equally likely and disorder is maximized.

        S/R = -N * [p_c*ln(p_c) + p_h*ln(p_h)],  p_c = 1/(1+k), p_h = k/(1+k)
    """
    p_c = 1.0 / (1.0 + k)
    p_h = k / (1.0 + k)
    s_c = p_c * np.log(p_c) if p_c > 0 else 0.0
    s_h = p_h * np.log(p_h) if p_h > 0 else 0.0
    return -N * (s_c + s_h)


# ---------------------------------------------------------------------
# 2. Residue-specific non-cooperative (mean-field) model
# ---------------------------------------------------------------------

def fraction_helix_residue_specific(k_values: Sequence[float]) -> float:
    """
    Fraction helix for a sequence of residues each with their own
    independent equilibrium constant k_i, under the non-cooperative
    model (PS8 8.3): mean-field average, no cooperativity between
    neighbors.

        <f_h> = (1/N) * sum_i [ k_i / (1 + k_i) ]
    """
    k_values = np.asarray(k_values, dtype=float)
    if k_values.size == 0:
        return 0.0
    return float(np.mean(k_values / (1.0 + k_values)))


def sequence_to_k_values(
    sequence: str,
    k_by_residue: Dict[str, float],
    default: Optional[float] = None,
) -> np.ndarray:
    """
    Map a sequence string to per-residue k values using a lookup table,
    e.g. {'A': k_A, 'E': k_E} as in PS8 8.3 (poly-Ala/Glu block
    sequences). Residues not in the table fall back to `default` if
    given, else raise KeyError.
    """
    values = []
    for aa in sequence.upper():
        if aa in k_by_residue:
            values.append(k_by_residue[aa])
        elif default is not None:
            values.append(default)
        else:
            raise KeyError(f"No k value for residue {aa!r} and no default given")
    return np.array(values, dtype=float)


# ---------------------------------------------------------------------
# 3. Zipper model (single contiguous helical run)
# ---------------------------------------------------------------------

def partition_zipper(N: int, k: float, sigma: float) -> float:
    """
    Partition function for the zipper model (PS8 8.1B-8.1E): only a
    single contiguous run of helical residues is allowed, anywhere
    along the chain. A run of length j has (N-j+1) possible positions
    and carries a single nucleation weight sigma regardless of length.

        rho = 1 + sum_{j=1}^{N} (N-j+1) * k^j * sigma

    (the "1" is the all-coil state, j=0).
    """
    total = 1.0
    for j in range(1, N + 1):
        total += (N - j + 1) * (k ** j) * sigma
    return total


def population_zipper(N: int, k: float, sigma: float, j: int) -> float:
    """
    Probability of a helical run of exactly length j (0 <= j <= N)
    under the zipper model (PS8 8.1G):

        P(0) = 1 / rho
        P(j) = (N-j+1) * k^j * sigma / rho,   j >= 1
    """
    if not (0 <= j <= N):
        raise ValueError("j must be between 0 and N")
    rho = partition_zipper(N, k, sigma)
    if j == 0:
        return 1.0 / rho
    return (N - j + 1) * (k ** j) * sigma / rho


def mean_helix_content_zipper(N: int, k: float, sigma: float) -> float:
    """Average fraction of helical residues under the zipper model,
    computed directly from the j-population distribution."""
    weighted = sum(j * population_zipper(N, k, sigma, j) for j in range(0, N + 1))
    return weighted / N


# ---------------------------------------------------------------------
# 4. Matrix (Zimm-Bragg-style) model, exact via sympy
# ---------------------------------------------------------------------
#
# Transfer matrix and boundary vectors exactly as used in PS8 8.4A:
#   W = [[k*t, 1], [k, 1]]
#   n = [0, 1]              (row vector)
#   c = [[1], [1]]           (column vector)
#   rho_N = (n * W**N * c)[0]
#
# k is the propagation-like equilibrium constant, t is the
# cooperativity parameter (t=1 recovers the non-cooperative model,
# since the matrix product's growth is then governed by k alone in the
# same way as (1+k)^N).

_k_sym, _t_sym = sym.symbols("k t", positive=True)


def _rho_symbolic(N: int):
    """Build the symbolic partition function rho_N(k, t) via the exact
    transfer-matrix product used in the coursework."""
    W = sym.Matrix([[_k_sym * _t_sym, 1], [_k_sym, 1]])
    n = sym.Matrix([[0, 1]])
    c = sym.Matrix([[1], [1]])
    return (n * W**N * c)[0]


def partition_matrix(N: int, k: float, t: float) -> float:
    """
    Numeric partition function for the matrix model at given N, k, t,
    matching PS8 8.4A (rho_N = n W^N c).
    """
    rho_expr = _rho_symbolic(N)
    rho_func = sym.lambdify([_k_sym, _t_sym], rho_expr, "numpy")
    return float(rho_func(k, t))


def fraction_helix_matrix(N: int, k, t: float) -> np.ndarray:
    """
    Ensemble-average fraction helix for the matrix model, matching PS8
    8.4B exactly:

        <f_h> = (k / (N * rho)) * d(rho)/dk

    Parameters
    ----------
    N : int
        Chain length.
    k : float or array-like
        Propagation equilibrium constant(s) to evaluate at.
    t : float
        Cooperativity parameter.

    Returns
    -------
    float or np.ndarray
        Fraction helix at each k (scalar in, scalar out; array in,
        array out).
    """
    rho_expr = _rho_symbolic(N)
    fh_expr = _k_sym / (N * rho_expr) * sym.diff(rho_expr, _k_sym)
    fh_func = sym.lambdify([_k_sym, _t_sym], fh_expr, "numpy")

    k_arr = np.asarray(k, dtype=float)
    result = fh_func(k_arr, t)
    if k_arr.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=float)
