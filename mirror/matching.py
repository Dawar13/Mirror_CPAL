"""Mirror-twin matching of MLP neurons.

Given S = (R + R^T)/2 with R[i, j] = corr(zt_i, z_j), find the involution pi maximising
sum_i S[i, pi(i)] = sum_i S_ii + sum_{pairs (i,j)} (2 S_ij - S_ii - S_jj).
This is a maximum-weight matching with edge weights w_ij = 2 S_ij - S_ii - S_jj; only positive
edges among each neuron's top-k candidates (ranked by w_ij) are kept. Unmatched neurons are
self-twins.

networkx's blossom matcher is pure Python (about 45 s for h = 3072 with clear structure and about
5 min for pure noise on a laptop CPU), so blocks are solved in parallel worker processes.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    perm: np.ndarray            # perm[i] = pi(i); an involution
    pairs: list                 # [(i, j)] with i < j
    self_twins: np.ndarray      # indices with pi(i) = i
    score: np.ndarray           # S[i, pi(i)]

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_self(self) -> int:
        return int(len(self.self_twins))

    @property
    def trace(self) -> int:
        return self.n_self


def symmetrise(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    return 0.5 * (R + R.T)


def candidate_edges(S: np.ndarray, top_k: int = 10):
    """Undirected positive edges (i < j) among each neuron's top-k by w_ij."""
    S = np.asarray(S, dtype=np.float64)
    h = S.shape[0]
    if h < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    diag = np.diag(S)
    W = 2.0 * S - diag[:, None] - diag[None, :]
    np.fill_diagonal(W, -np.inf)
    k = min(top_k, h - 1)
    top = np.argpartition(-W, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(h), k)
    cols = top.ravel()
    w = W[rows, cols]
    keep = w > 0
    i = np.minimum(rows, cols)[keep]
    j = np.maximum(rows, cols)[keep]
    w = w[keep]
    _, first = np.unique(i * h + j, return_index=True)
    return i[first], j[first], w[first]


def solve_matching(i: np.ndarray, j: np.ndarray, w: np.ndarray, scale: float = 1e6) -> list:
    """Exact max-weight matching (networkx blossom) on integer-scaled weights, per component."""
    import networkx as nx

    wi = np.rint(np.asarray(w) * scale).astype(np.int64)
    keep = wi > 0
    G = nx.Graph()
    G.add_weighted_edges_from(zip(np.asarray(i)[keep].tolist(), np.asarray(j)[keep].tolist(),
                                  wi[keep].tolist()))
    pairs = []
    for comp in nx.connected_components(G):
        if len(comp) == 2:
            a, b = tuple(comp)
            pairs.append((a, b))
            continue
        pairs.extend(nx.max_weight_matching(G.subgraph(comp), maxcardinality=False, weight="weight"))
    return sorted((min(a, b), max(a, b)) for a, b in pairs)


def build_result(S: np.ndarray, pairs: list) -> MatchResult:
    h = S.shape[0]
    perm = np.arange(h)
    for a, b in pairs:
        perm[a], perm[b] = b, a
    self_twins = np.flatnonzero(perm == np.arange(h))
    return MatchResult(perm=perm, pairs=[tuple(map(int, p)) for p in pairs], self_twins=self_twins,
                       score=S[np.arange(h), perm])


def match_neurons(R: np.ndarray, top_k: int = 10) -> MatchResult:
    S = symmetrise(R)
    return build_result(S, solve_matching(*candidate_edges(S, top_k)))


def _solve_job(args):
    return solve_matching(*args)


def match_many(R_list: list[np.ndarray], top_k: int = 10, workers: int = 1) -> list[MatchResult]:
    """Match several blocks; the blossom step runs in `workers` processes."""
    S_list = [symmetrise(R) for R in R_list]
    jobs = [candidate_edges(S, top_k) for S in S_list]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            pair_lists = list(ex.map(_solve_job, jobs))
    else:
        pair_lists = [_solve_job(j) for j in jobs]
    return [build_result(S, p) for S, p in zip(S_list, pair_lists)]


def random_pairing(h: int, n_pairs: int, seed: int) -> np.ndarray:
    """Random involution on h neurons with exactly n_pairs 2-cycles (null model)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(h)
    perm = np.arange(h)
    a, b = idx[:n_pairs], idx[n_pairs:2 * n_pairs]
    perm[a], perm[b] = b, a
    return perm
