"""
helix_coil.py

Helix-coil transition prediction using the Zimm-Bragg transfer matrix method.

Model
-----
Each residue i is in state helix (h) or coil (c). The Zimm-Bragg model
assigns:
    - s_i  : propagation parameter for residue i (statistical weight of
             extending a helix once nucleated). Residue-specific if a
             per-residue propensity scale is supplied, else uniform.
    - sigma: nucleation parameter (statistical weight penalty for starting
             a new helical segment), typically a single small constant
             (~1e-3 to 1e-4) shared across the chain.

Transfer matrix per residue (rows = state at i-1, columns = state at i,
both ordered [coil, helix]):

    W_i = [[1,        sigma*s_i ],
           [1,        s_i       ]]

i.e. coil->coil = 1, coil->helix = sigma*s_i (nucleation penalty applied
once, when a new helical run starts), helix->coil = 1, helix->helix = s_i
(propagation, no further nucleation penalty). The partition function is
the product of per-residue matrices between boundary vectors, and
per-residue helix probability is obtained from the ratio of forward *
backward partial partition weights to the total, which is the standard
way to get marginals out of a 1D transfer-matrix model.

This module supports:
    - uniform s (single propagation parameter for the whole sequence)
    - per-residue s from a propensity table (e.g. Pace-Scholtz helix
      propensity scale), so real sequences can be scored residue-by-residue
"""

from typing import Dict, Optional, Sequence

import numpy as np

# Pace & Scholtz (1998) helix propensity scale, delta-delta-G (kcal/mol)
# relative to Ala (Ala = 0, most helix-favorable). Lower = more
# helix-favorable. Used here only as an optional convenience table to
# derive per-residue s values; callers may supply their own.
PACE_SCHOLTZ_DDG = {
    "A": 0.00, "L": 0.21, "R": 0.21, "M": 0.24, "K": 0.26,
    "Q": 0.39, "E": 0.40, "I": 0.41, "W": 0.49, "S": 0.50,
    "Y": 0.53, "F": 0.54, "H": 0.61, "V": 0.61, "N": 0.65,
    "T": 0.66, "C": 0.68, "D": 0.69, "G": 1.00, "P": 3.16,
}

R_GAS = 0.0019872  # kcal / (mol * K)


def s_from_ddG(ddG: float, T: float = 298.15) -> float:
    """Convert a helix-propensity ddG (kcal/mol, relative) to a
    Boltzmann propagation weight s = exp(-ddG / RT)."""
    return float(np.exp(-ddG / (R_GAS * T)))


def propensity_scale_to_s(
    sequence: str,
    scale: Optional[Dict[str, float]] = None,
    T: float = 298.15,
) -> np.ndarray:
    """
    Convert a sequence to an array of per-residue s values using a
    ddG propensity scale (default: Pace & Scholtz).

    Unknown residues fall back to Ala's value (ddG = 0, s = 1).
    """
    if scale is None:
        scale = PACE_SCHOLTZ_DDG
    return np.array(
        [s_from_ddG(scale.get(aa.upper(), 0.0), T=T) for aa in sequence],
        dtype=float,
    )


def helix_probabilities(
    s: Sequence[float],
    sigma: float = 5e-4,
) -> np.ndarray:
    """
    Compute per-residue helix probability via the Zimm-Bragg transfer
    matrix method.

    Parameters
    ----------
    s : array-like of float, length N
        Per-residue propagation parameters. Use a constant array for a
        uniform-propensity chain, or propensity_scale_to_s() output for
        a real sequence.
    sigma : float
        Nucleation parameter, shared across the chain (typical range
        1e-3 to 1e-4).

    Returns
    -------
    np.ndarray, length N
        Probability that residue i is in the helical state, marginalized
        over all other residues' states.
    """
    s = np.asarray(s, dtype=float)
    n = s.size
    if n == 0:
        return np.array([])

    # Transfer matrix W_i, rows = state of residue i-1, columns = state of
    # residue i, state order [coil, helix]:
    #   coil  -> coil  : 1
    #   coil  -> helix : sigma * s_i   (nucleation)
    #   helix -> coil  : 1
    #   helix -> helix : s_i           (propagation)
    matrices = np.empty((n, 2, 2))
    for i in range(n):
        matrices[i] = np.array(
            [[1.0, sigma * s[i]], [1.0, s[i]]]
        )

    # alpha_i = weight vector over states AFTER processing residue i.
    # alpha_0 is the virtual "before residue 1" state: coil, weight 1.
    alpha = np.zeros((n + 1, 2))
    alpha[0] = np.array([1.0, 0.0])
    for i in range(n):
        alpha[i + 1] = alpha[i] @ matrices[i]

    # beta_i = weight of completing the chain from state at position i to
    # the end. beta_n is the boundary: either final state allowed, weight 1.
    beta = np.zeros((n + 1, 2))
    beta[n] = np.array([1.0, 1.0])
    for i in range(n - 1, -1, -1):
        beta[i] = matrices[i] @ beta[i + 1]

    Z = float(alpha[n] @ np.array([1.0, 1.0]))
    if Z <= 0 or not np.isfinite(Z):
        raise ValueError("Partition function is non-positive or non-finite; check s and sigma")

    # P(residue i is helix) = alpha_i[helix] * beta_i[helix] / Z, using
    # alpha/beta both indexed at position i (i.e. alpha[i+1], beta[i+1]
    # in the 0-indexed arrays above, since alpha[0]/beta[0] are the
    # boundary vectors before residue 1 / at the very end).
    helix_state = 1
    probs = (alpha[1 : n + 1, helix_state] * beta[1 : n + 1, helix_state]) / Z

    return np.clip(probs, 0.0, 1.0)


def mean_helix_content(s: Sequence[float], sigma: float = 5e-4) -> float:
    """Fraction of residues helical, averaged over the whole chain."""
    probs = helix_probabilities(s, sigma=sigma)
    return float(np.mean(probs)) if probs.size else 0.0
