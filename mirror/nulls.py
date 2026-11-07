"""Matched null structure for breaking shares and random surgery.

A matched null has exactly the counts of the fitted structure, so it is equally "easy" to satisfy:
  - a random orthogonal involution with the same d+ and d- as the fitted residual operator;
  - per block, a random neuron pairing with the same number of pairs and self-twins.
If the fitted structure is close to trivial (A near I and P near I), the matched null is also near
trivial and its breaking share is also low, which exposes the identity trap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import random_involution
from .matching import random_pairing


@dataclass
class MatchedNull:
    A: np.ndarray
    perms: list
    d_plus: int
    d_minus: int
    n_pairs: list
    n_self: list


def matched_null(d: int, d_plus: int, n_pairs_per_block: list[int], h: int, seed: int) -> MatchedNull:
    """Random involution with the given d+ (and d- = d - d+) plus one matched pairing per block.

    The operator depends only on (d, d_plus, seed), so it equals `random_involution(d, d_plus, seed)`.
    """
    A = random_involution(d, d_plus, seed)
    perms = [random_pairing(h, int(n), seed=seed * 1000 + 17 + b) for b, n in enumerate(n_pairs_per_block)]
    eig = np.linalg.eigvalsh(A)
    dp, dm = int((eig > 0).sum()), int((eig < 0).sum())
    pairs = [int((p != np.arange(h)).sum() // 2) for p in perms]
    selfs = [int((p == np.arange(h)).sum()) for p in perms]
    assert (dp, dm) == (d_plus, d - d_plus), (dp, dm, d_plus)
    assert pairs == [int(n) for n in n_pairs_per_block]
    return MatchedNull(A=A, perms=perms, d_plus=dp, d_minus=dm, n_pairs=pairs, n_self=selfs)
