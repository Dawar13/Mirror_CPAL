"""Twin-validation helpers for MUST-05 (twins are working mirror pairs) and MUST-06 (can ordinary
redundancy detectors see twins?), PROJECT_NEXT_PHASE.md Section 5.

Conventions (as in mirror/stats.py and mirror/variants.py):
- Blocks are 0-based in code (block b = model.blocks[b]); the plan's "block 1" is b = 0. Block b reads
  residual layer b and writes layer b + 1 (layer 0 = tokens entering block 0).
- Tokens are rows; a residual operator A acts on the right: for the mirrored image, un-flipped to the
  original token positions, Xt ~ X A. A is a symmetric orthogonal involution with A 1 = 1.
- A twin pairing is an involution `perm` on the h hidden neurons of one MLP: neuron perm[i] on the
  mirrored image tracks neuron i on the original at the mirrored token.
- "pre" activations are the fc1 outputs z (what the twin matching correlates); "post" activations are
  act(z), the GELU outputs that fc2 reads (what the switch-off test zeroes).

Contents
  Structure    get_structure: the fit run's structure, or a fresh fit (variants.fit_structure) on the
               fit set saved in the run_e1.py layout and reused by later runs; parse_blocks.
  Subsets      balanced_subset (class-balanced, seeded).
  Pairs        twin_pairs, strongest_twin_pairs, random pairs, random partners, same-image pairing
               (the twin matcher applied to the same-image correlation, self-pairs scored 0).
  Statistics   ActStats / collect_act_stats: float64 sums over every token of a paired pass, giving the
               mirror correlation R[i, j] = corr(zt_i, z_j) and the same-image correlation
               Q[i, j] = corr(z_i, z_j) for pre and post activations.
  Detectors    partner_ranks: rank of the twin among a neuron's same-image correlated partners.
  Spectra      effective_rank (entropy of the normalised singular values, Roy and Vetterli 2007
               (verify)) and mlp_spectra before and after the mirror projection.
  Switch-off   block_inputs, switch_off_block (batched single-neuron ablations, deltas at the output of
               block b + 1 and at the final CLS), summarise_pairs.
  Crops        token_crop (a size x size window centred on a token).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import variants as V
from .data import mirror, normalize_for
from .matching import candidate_edges, random_pairing, solve_matching, symmetrise
from .stats import center_channels, run_paired, unflip_tokens
from .surgery import breaking_share, fold_mlp, project_mlp_two


# ----------------------------------------------------------------------------------------------
# Structure: fit run, or a fresh fit on the fit set
# ----------------------------------------------------------------------------------------------
def default_structure_dir(cache_dir, model_name: str, fit_seed: int) -> Path:
    """Shared location of a fresh fit, so MUST-05 and MUST-06 on the same model fit only once."""
    return Path(cache_dir) / "structures" / f"{model_name}_fit{int(fit_seed)}"


def sha256_file(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_structure(*, adapter, proto, device, model_name: str, fit_run=None, fit_seed: int | None = None,
                  blocks=(), structure_dir=None, top_k: int = 10, workers: int = 1, log=print):
    """(Structure, info) for `model_name`.

    With `fit_run` (a run_e1.py output directory or its e1_operators.npz): load it (variants.load_fit_run)
    and check model, fit seed (None: the run's own) and sizes. Otherwise (fit seed None means 0) use
    `structure_dir`: reuse the structure saved there
    if it is for this model and fit seed and its twin matching covers `blocks` (0-based); if not, fit
    per-layer operators and the twin matching of blocks 0..max(blocks) on proto.fit_loader(fit_seed)
    with variants.fit_structure (the run_e1.py protocol: paired pass, channel-centred cross-moments,
    max-weight involution matching on the symmetrised fc1 mirror correlation, top_k candidates), save it
    in the run_e1.py layout (<dir>/results/e1_operators.npz and e1_meta.json) and load it back through
    load_fit_run, so a resumed run uses exactly what was saved.
    """
    blocks = sorted({int(b) for b in blocks})
    if fit_run:
        struct = V.load_fit_run(fit_run)
        run_model = struct.meta.get("model")
        if run_model is not None and run_model != model_name:
            raise SystemExit(f"the fit run is for {run_model}, not {model_name}")
        seed = V.resolve_fit_seed(fit_seed, struct)
        source = {"kind": "fit run", "path": str(struct.source), "sha256": sha256_file(struct.source),
                  "fit_seed": seed, "reused": True}
    else:
        if structure_dir is None:
            raise ValueError("structure_dir is needed when no fit run is given")
        fit_seed = 0 if fit_seed is None else int(fit_seed)
        sdir = Path(structure_dir)
        res = sdir / "results"
        npz, meta_path = res / "e1_operators.npz", res / "e1_meta.json"
        reuse = False
        if npz.exists() and meta_path.exists():
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            reuse = (m.get("model") == model_name and int(m.get("fit_seed", -1)) == int(fit_seed)
                     and set(blocks) <= set(int(b) for b in m.get("matched_blocks", [])))
        if not reuse:
            match = list(range(max(blocks) + 1)) if blocks else list(range(adapter.depth))
            log(f"structure: fitting operators and twins of blocks {[b + 1 for b in match]} on fit{fit_seed} "
                f"({workers} matching workers)")
            st = V.fit_structure(adapter, proto.fit_loader(int(fit_seed)), device, match_blocks=match,
                                 top_k=top_k, workers=workers)
            res.mkdir(parents=True, exist_ok=True)
            tmp = res / f"e1_operators.{os.getpid()}.tmp.npz"
            V.save_structure(tmp, st, with_moments=False)
            os.replace(tmp, npz)
            from .utils import git_info

            meta = {"fit_seed": int(fit_seed), "model": model_name, "protocol": "v2",
                    "writer": "mirror/twins.py get_structure (variants.fit_structure)",
                    "config": {"stats": {"center": True}, "model": {"name": model_name},
                               "data": {"fit_seed": int(fit_seed)}},
                    "matched_blocks": match, "top_k": int(top_k), "n_tokens": st.meta.get("n_tokens"),
                    "git": git_info()}
            tmp_meta = res / f"e1_meta.{os.getpid()}.tmp.json"
            tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            os.replace(tmp_meta, meta_path)
        struct = V.load_fit_run(sdir)
        struct.meta["matched_blocks"] = json.loads(meta_path.read_text(encoding="utf-8"))["matched_blocks"]
        source = {"kind": "fresh fit" if not reuse else "fresh fit (reused)", "path": str(npz),
                  "sha256": sha256_file(npz), "fit_seed": int(fit_seed), "reused": bool(reuse),
                  "matched_blocks": struct.meta["matched_blocks"]}
    if (struct.depth, struct.d, struct.h) != (adapter.depth, adapter.embed_dim, adapter.mlp_hidden):
        raise SystemExit(f"structure sizes {(struct.depth, struct.d, struct.h)} do not match the model "
                         f"{(adapter.depth, adapter.embed_dim, adapter.mlp_hidden)}")
    if struct.match_scores is None:
        raise SystemExit(f"{struct.source} has no match_scores (needed to rank twin pairs)")
    return struct, source


def reference_check(struct, reference, blocks) -> dict:
    """Compare a structure with an earlier run_e1.py run of the same model and fit seed (its e1_mlp.csv
    and e1_layers.csv): twin pair counts of `blocks` (0-based) and d+ of the layers they read and write.
    A fresh fit of the same fit set should reproduce them (up to near-ties of the matching)."""
    import pandas as pd

    ref = Path(reference)
    res = ref if (ref / "e1_mlp.csv").exists() else ref / "results"
    out = {"reference": str(res)}
    mlp_csv, lay_csv = res / "e1_mlp.csv", res / "e1_layers.csv"
    if mlp_csv.exists():
        mlp = pd.read_csv(mlp_csv).set_index("block")
        rows = {int(b) + 1: {"fresh": struct.n_pairs(b), "reference": int(mlp.loc[int(b) + 1, "n_pairs"])}
                for b in blocks if int(b) + 1 in mlp.index}
        out["n_pairs"] = rows
        out["n_pairs_equal"] = all(r["fresh"] == r["reference"] for r in rows.values())
    if lay_csv.exists():
        lay = pd.read_csv(lay_csv).set_index("layer")
        layers = sorted({int(b) for b in blocks} | {int(b) + 1 for b in blocks})
        rows = {l: {"fresh": int(struct.d_plus[l]), "reference": int(lay.loc[l, "d_plus"])}
                for l in layers if l in lay.index}
        out["d_plus"] = rows
        out["d_plus_equal"] = all(r["fresh"] == r["reference"] for r in rows.values())
    meta = res / "e1_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        out["reference_model"] = ((m.get("config") or {}).get("model") or {}).get("name")
        out["reference_fit_seed"] = m.get("fit_seed")
    return out


def parse_blocks(text: str, depth: int) -> list:
    """'1,3,6' or '1-6' or '1-3,6' (blocks numbered from 1) -> sorted 0-based indices; checks 1..depth."""
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    blocks = sorted(out)
    if not blocks or blocks[0] < 1 or blocks[-1] > depth:
        raise SystemExit(f"blocks must lie in 1..{depth}, got {text!r}")
    return [b - 1 for b in blocks]


# ----------------------------------------------------------------------------------------------
# Subsets
# ----------------------------------------------------------------------------------------------
def balanced_subset(labels, n: int, seed: int, pool=None) -> np.ndarray:
    """n indices (sorted) spread over classes as evenly as possible: classes are visited in a seeded
    random order, round after round, each round taking one more seeded random image per class. With
    n <= number of classes every chosen class gives exactly one image. `pool` restricts the indices."""
    labels = np.asarray(labels)
    pool = np.arange(len(labels)) if pool is None else np.asarray(pool, dtype=np.int64)
    if n > len(pool):
        raise ValueError(f"asked for {n} images, the pool has {len(pool)}")
    rng = np.random.default_rng(seed)
    per = {c: rng.permutation(pool[labels[pool] == c]) for c in np.unique(labels[pool])}
    order = rng.permutation(np.asarray(sorted(per)))
    out, r = [], 0
    while len(out) < n:
        for c in order:
            if r < len(per[c]):
                out.append(int(per[c][r]))
                if len(out) == n:
                    break
        r += 1
    return np.sort(np.asarray(out, dtype=np.int64))


# ----------------------------------------------------------------------------------------------
# Pairs (arrays of shape [n, 2] of neuron indices)
# ----------------------------------------------------------------------------------------------
def twin_pairs(perm) -> np.ndarray:
    """Every 2-cycle (u, v), u < v, of an involution, ordered by u."""
    perm = np.asarray(perm, dtype=np.int64)
    u = np.flatnonzero(perm > np.arange(len(perm)))
    return np.stack([u, perm[u]], axis=1) if len(u) else np.zeros((0, 2), dtype=np.int64)


def strongest_twin_pairs(perm, scores, n: int):
    """The n twin pairs with the highest fit-set twin correlation S_uv (scores[u] = S[u, perm[u]]),
    strongest first, ties broken by u. Returns (pairs [m, 2], S_uv [m]), m = min(n, number of pairs)."""
    pairs = twin_pairs(perm)
    s = np.asarray(scores, dtype=np.float64)[pairs[:, 0]] if len(pairs) else np.zeros(0)
    order = np.lexsort((pairs[:, 0], -s)) if len(pairs) else np.zeros(0, dtype=np.int64)
    order = order[:int(n)]
    return pairs[order], s[order]


def rank_twin_pairs(perm, S, n: int, by: str = "weight"):
    """The n strongest twin pairs with the full symmetrised fit-set mirror correlation S (h x h), ranked by
    the matching weight w_uv = 2 S_uv - S_uu - S_vv (by="weight", the order of scripts/f7_visuals.py
    rank_pairs: the gain of pairing over two self-twins, which avoids near-symmetric look-alike pairs) or
    by S_uv (by="score"); ties by S_uv, then u. Returns (pairs [m, 2], S_uv [m], w_uv [m])."""
    if by not in ("weight", "score"):
        raise ValueError(f"unknown ranking {by!r}")
    S = np.asarray(S, dtype=np.float64)
    pairs = twin_pairs(perm)
    u, v = pairs[:, 0], pairs[:, 1]
    s_uv = S[u, v]
    w = 2.0 * s_uv - S[u, u] - S[v, v]
    order = np.lexsort((u, -s_uv, -(w if by == "weight" else s_uv)))[:int(n)]
    return pairs[order], s_uv[order], w[order]


def matched_random_pairs(h: int, n_pairs: int, seed: int) -> np.ndarray:
    """n_pairs disjoint random pairs (matching.random_pairing, the null with matched counts)."""
    return twin_pairs(random_pairing(int(h), int(n_pairs), seed=int(seed)))


def random_partners_for(perm, idx, seed: int) -> np.ndarray:
    """For each neuron i in idx a uniformly random partner that is neither i nor its twin perm[i]."""
    perm = np.asarray(perm, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    h = len(perm)
    if h < 3:
        raise ValueError("need at least 3 neurons")
    rng = np.random.default_rng(seed)
    j = (idx + rng.integers(1, h, size=len(idx))) % h            # never i
    bad = j == perm[idx]
    while bad.any():
        j[bad] = (idx[bad] + rng.integers(1, h, size=int(bad.sum()))) % h
        bad = j == perm[idx]
    return j


def random_neuron_pairs(perm, n: int, seed: int) -> np.ndarray:
    """n pairs (i, j): i drawn without replacement from all neurons, j a random partner (not i, not the
    twin of i). The pairs need not be disjoint."""
    h = len(perm)
    rng = np.random.default_rng(seed)
    i = np.sort(rng.choice(h, size=min(int(n), h), replace=False)).astype(np.int64)
    return np.stack([i, random_partners_for(perm, i, seed + 1)], axis=1)


def truncate_pairs(pairs, score: np.ndarray, n: int) -> np.ndarray:
    """Keep the n pairs with the highest score[u, v]; if there are fewer, add disjoint pairs greedily by
    score among the unmatched neurons (as variants._exact_pair_count does for the cosine pairing)."""
    pairs = [tuple(map(int, p)) for p in pairs]
    w = np.array([score[a, b] for a, b in pairs]) if pairs else np.zeros(0)
    order = np.argsort(-w, kind="stable")
    out = [pairs[k] for k in order[:n]]
    if len(out) < n:
        used = {x for p in out for x in p}
        free = np.array([i for i in range(score.shape[0]) if i not in used], dtype=np.int64)
        sub = score[np.ix_(free, free)]
        iu, ju = np.triu_indices(len(free), 1)
        for t in np.argsort(-sub[iu, ju], kind="stable"):
            if len(out) >= n:
                break
            a, b = int(free[iu[t]]), int(free[ju[t]])
            if a in used or b in used:
                continue
            out.append((a, b))
            used.update((a, b))
    out = sorted((min(a, b), max(a, b)) for a, b in out)
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def _match_job(args):
    return solve_matching(*args)


def same_image_pairings(Q_list, n_pairs_list, top_k: int = 10, workers: int = 1) -> list:
    """Pairs matched by same-image activation correlation: the twin matcher (matching.match_neurons:
    max-weight matching over each neuron's top_k candidates, weight w_ij = 2 S_ij - S_ii - S_jj) applied
    to S = Q with the diagonal set to 0, so leaving a neuron unpaired scores 0, as a self-pair would if
    it carried no signal (with the diagonal of a correlation matrix, 1, no edge would ever be positive).
    Then the n strongest pairs by Q are kept, n = the twin matching's pair count."""
    from concurrent.futures import ProcessPoolExecutor

    S_list = []
    for Q in Q_list:
        S = symmetrise(np.asarray(Q, dtype=np.float64))
        np.fill_diagonal(S, 0.0)
        S_list.append(S)
    jobs = [candidate_edges(S, top_k) for S in S_list]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            found = list(ex.map(_match_job, jobs))
    else:
        found = [_match_job(j) for j in jobs]
    return [truncate_pairs(p, S, int(n)) for p, S, n in zip(found, S_list, n_pairs_list)]


def cosine_pairings(W1_list, n_pairs_list, top_k: int = 10, workers: int = 1) -> list:
    """Pairs matched by plain cosine of the folded fc1 rows (variants.generic_pairings, matched count)."""
    perms = V.generic_pairings(W1_list, [int(n) for n in n_pairs_list], top_k, workers)
    return [twin_pairs(p) for p in perms]


def pair_values(M, pairs, symmetric: bool = True) -> np.ndarray:
    """M[u, v] per pair, or (M[u, v] + M[v, u]) / 2 with symmetric=True."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if isinstance(M, torch.Tensor):
        u = torch.as_tensor(pairs[:, 0], device=M.device)
        v = torch.as_tensor(pairs[:, 1], device=M.device)
        a = M[u, v].double()
        a = 0.5 * (a + M[v, u].double()) if symmetric else a
        return a.cpu().numpy()
    M = np.asarray(M, dtype=np.float64)
    a = M[pairs[:, 0], pairs[:, 1]]
    return 0.5 * (a + M[pairs[:, 1], pairs[:, 0]]) if symmetric else a


# ----------------------------------------------------------------------------------------------
# Activation statistics over a paired pass
# ----------------------------------------------------------------------------------------------
KINDS = ("post", "pre")


class ActStats:
    """Float64 sums of MLP hidden activations of chosen blocks over every token (prefix and patch).

    For each (block, kind) with kind "pre" (fc1 output z) or "post" (act(z)), on the originals z and on
    the mirrored images un-flipped to the original token positions zt:
        n, sums and squared sums of z and of zt (per neuron),
        xm = sum zt^T z (h x h; mirror_cross), the statistic of the twin matching,
        xs = sum z^T z  (h x h; same_cross), the same image at the same token.
    """

    def __init__(self, blocks, h: int, device, kinds=KINDS, mirror_cross: bool = True,
                 same_cross: bool = True, chunk: int = 8192):
        self.blocks, self.kinds, self.h, self.chunk = [int(b) for b in blocks], tuple(kinds), int(h), int(chunk)
        bad = [k for k in self.kinds if k not in KINDS]
        if bad:
            raise ValueError(f"unknown activation kinds {bad}")
        kw = dict(dtype=torch.float64, device=device)
        self.acc = {}
        for b in self.blocks:
            for k in self.kinds:
                a = {"n": torch.zeros((), **kw)}
                for key in ("s", "s2", "st", "st2"):
                    a[key] = torch.zeros(h, **kw)
                if mirror_cross:
                    a["xm"] = torch.zeros(h, h, **kw)
                if same_cross:
                    a["xs"] = torch.zeros(h, h, **kw)
                self.acc[(b, k)] = a

    @torch.no_grad()
    def update(self, block: int, kind: str, z: torch.Tensor, zt: torch.Tensor) -> None:
        a = self.acc[(int(block), kind)]
        Z, Zt = z.reshape(-1, self.h), zt.reshape(-1, self.h)
        for s in range(0, Z.shape[0], self.chunk):
            A = Z[s:s + self.chunk].to(torch.float64)
            T = Zt[s:s + self.chunk].to(torch.float64)
            a["n"] += A.shape[0]
            a["s"] += A.sum(0)
            a["s2"] += (A * A).sum(0)
            a["st"] += T.sum(0)
            a["st2"] += (T * T).sum(0)
            if "xm" in a:
                a["xm"] += T.T @ A
            if "xs" in a:
                a["xs"] += A.T @ A

    def n_tokens(self, block: int) -> int:
        return int(self.acc[(int(block), self.kinds[0])]["n"].item())

    def _moments(self, a):
        n = a["n"]
        m, mt = a["s"] / n, a["st"] / n
        v = (a["s2"] / n - m * m).clamp_min(0)
        vt = (a["st2"] / n - mt * mt).clamp_min(0)
        return n, m, mt, v, vt

    @staticmethod
    def _corr(cov, va, vb, eps: float = 1e-12):
        den = torch.sqrt(torch.outer(va, vb))
        R = torch.where(den > eps, cov / den.clamp_min(eps), torch.zeros_like(cov))
        return R.clamp(-1.0, 1.0)

    def mirror_corr(self, block: int, kind: str) -> torch.Tensor:
        """R[i, j] = corr(zt_i, z_j): neuron i on the mirrored image at the mirrored token against neuron
        j on the original (the mirror.stats.MLPStats convention). Twins: R[perm[j], j] high."""
        a = self.acc[(int(block), kind)]
        n, m, mt, v, vt = self._moments(a)
        return self._corr(a["xm"] / n - torch.outer(mt, m), vt, v)

    def same_corr(self, block: int, kind: str) -> torch.Tensor:
        """Q[i, j] = corr(z_i, z_j) on the original images, same token (Pearson over all tokens)."""
        a = self.acc[(int(block), kind)]
        n, m, _, v, _ = self._moments(a)
        Q = self._corr(a["xs"] / n - torch.outer(m, m), v, v)
        Q.fill_diagonal_(1.0)
        return Q

    def free(self) -> None:
        self.acc = {}


def _act_of(block_module: nn.Module):
    act = getattr(block_module.mlp, "act", None)
    if act is None:
        raise ValueError("the MLP has no `act` module (GELU); the twin code needs fc1 -> act -> fc2")
    return act


@torch.no_grad()
def collect_act_stats(adapter, loader, device, blocks, kinds=KINDS, mirror_cross: bool = True,
                      same_cross: bool = True, model: nn.Module | None = None,
                      desc: str = "activation statistics") -> ActStats:
    """One paired pass (mirror.stats.run_paired: each batch with its mirror, float32 forward) into ActStats.
    Post activations are the block's own `mlp.act` applied to the fc1 output in float32 (float64 models
    keep float64)."""
    m = adapter.model if model is None else model
    blocks = sorted({int(b) for b in blocks})
    st = ActStats(blocks, adapter.mlp_hidden, device, kinds, mirror_cross, same_cross)
    acts = {b: _act_of(m.blocks[b]) for b in blocks}

    def up(t):
        return t if t.dtype == torch.float64 else t.float()

    def cb(b, z, zt):
        if b not in acts:
            return
        for kind in st.kinds:
            if kind == "pre":
                st.update(b, "pre", z, zt)
            else:
                st.update(b, "post", acts[b](up(z)), acts[b](up(zt)))

    run_paired(adapter, loader, device, fc1_cb=cb, model=m, desc=desc)
    return st


# ----------------------------------------------------------------------------------------------
# Same-image redundancy detector
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def partner_ranks(Q, pairs, absolute: bool = False, chunk: int = 1024):
    """For each pair (u, v): the rank of v among u's same-image partners (1 = most correlated; u itself
    excluded; ties count against the twin) and the rank of u among v's. absolute=True ranks by |Q|."""
    Q = torch.as_tensor(Q)
    M = Q.abs() if absolute else Q
    pairs = torch.as_tensor(np.asarray(pairs, dtype=np.int64).reshape(-1, 2), device=M.device)
    out = []
    for a_col, b_col in ((0, 1), (1, 0)):
        ranks = []
        for s in range(0, pairs.shape[0], chunk):
            a, b = pairs[s:s + chunk, a_col], pairs[s:s + chunk, b_col]
            rows = M[a].clone()
            ar = torch.arange(len(a), device=M.device)
            rows[ar, a] = -math.inf
            val = rows[ar, b]
            ranks.append((rows >= val[:, None]).sum(1))          # the twin itself counts once
        out.append(torch.cat(ranks).cpu().numpy() if ranks else np.zeros(0, dtype=np.int64))
    return out[0], out[1]


# ----------------------------------------------------------------------------------------------
# Spectra and effective rank
# ----------------------------------------------------------------------------------------------
def effective_rank(s, kind: str = "l1") -> float:
    """exp(entropy) of the normalised singular values. kind "l1": p_k = s_k / sum s (Roy and Vetterli 2007
    (verify)); kind "energy": p_k = s_k^2 / sum s^2."""
    s = np.asarray(s, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return float("nan")
    w = s if kind == "l1" else s * s
    p = w / w.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def mlp_spectra(block: nn.Module, A_in, A_out, perm) -> dict:
    """Singular values of the folded fc1 (W1 [h, d]) and fc2 (W2 [d, h]) of one block before and after the
    mirror projection with A_in (fc1) and A_out (fc2) (surgery.project_mlp_two), in float64."""
    f = fold_mlp(block)
    e = project_mlp_two(**f, A_in=A_in, A_out=A_out, perm=perm)
    out = {}
    for name, W, We in (("fc1", f["W1"], e["W1"]), ("fc2", f["W2"], e["W2"])):
        out[name] = {"before": torch.linalg.svdvals(W).cpu().numpy(),
                     "after": torch.linalg.svdvals(We).cpu().numpy(),
                     "breaking_share": breaking_share(W, We)}
    return out


# ----------------------------------------------------------------------------------------------
# Switch-off test
# ----------------------------------------------------------------------------------------------
class ActMask:
    """Forward hook on a block's mlp.act that multiplies its output by `mask` ([B, 1, h]) when set."""

    def __init__(self, act: nn.Module):
        self.mask = None
        self.handle = act.register_forward_hook(self._hook)

    def _hook(self, _mod, _inp, out):
        if self.mask is None:
            return None
        return out * self.mask.to(out.dtype)

    def remove(self):
        self.handle.remove()


def _autocast(amp: str | None, device):
    if amp == "bf16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@contextlib.contextmanager
def matmul_precision(precision: str):
    """fp32: TF32 off (the repo default); tf32: TF32 on inside the block; bf16 is handled by autocast."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.get_float32_matmul_precision())
    tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[0], old[1]
        torch.set_float32_matmul_precision(old[2])


@torch.no_grad()
def block_inputs(adapter, images, device, block: int, batch_size: int = 32, model: nn.Module | None = None,
                 amp: str | None = None):
    """Residual stream entering `block` for uint8 images [N, H, W, 3] and for their mirrors.

    Returns (X, Xm) [N, T, d] in the model's dtype on `device`; Xm is in the mirrored image's own token
    order (not un-flipped), ready to be pushed through the remaining blocks."""
    m = adapter.model if model is None else model
    xs, xms = [], []
    for s in range(0, len(images), batch_size):
        imgs = torch.from_numpy(np.ascontiguousarray(images[s:s + batch_size])).permute(0, 3, 1, 2)
        x = normalize_for(adapter, imgs.to(device))
        B = x.shape[0]
        with _autocast(amp, device):
            t = adapter.layer0_tokens(torch.cat([x, mirror(x)], dim=0), m)
            for k in range(block):
                t = m.blocks[k](t)
        t = t.to(adapter.dtype)
        xs.append(t[:B])
        xms.append(t[B:])
    return torch.cat(xs), torch.cat(xms)


SWITCH_KEYS = ("mid_dot", "mid_ni", "mid_nj", "patch_dot", "patch_ni", "patch_nj",
               "cls_dot", "cls_ni", "cls_nj")


@torch.no_grad()
def switch_off_block(adapter, block: int, X, Xm, pairs: dict, A_mid, A_last=None, *,
                     model: nn.Module | None = None, chunk_images: int = 4, abl_batch: int = 256,
                     amp: str | None = None, center: bool = True, pair_chunk: int = 64,
                     progress: bool = True) -> dict:
    """Single-neuron switch-off in `block` (0-based) on N images (X) and their mirrors (Xm, from block_inputs).

    pairs maps a pair type to an int array [n, 2] of (i, j): neuron i is zeroed after the activation, at
    every token, on image x; neuron j is zeroed on mirror(x). For each pair and image:
        D_i = Delta_i(x) at the output of block + 1 (all tokens),
        D_j = Delta_j(mirror x) there, un-flipped to the original token positions (mirror.stats.unflip_tokens),
    both channel-centred (center=True; LayerNorm removes that part, and the operators were fitted on
    centred tokens), and
        mid_dot = <D_i A_mid, D_j>, mid_ni = ||D_i A_mid||^2, mid_nj = ||D_j||^2   (all tokens),
        patch_*  the same over patch tokens only,
        cls_*    the same for the CLS token of the residual stream after the last block with A_last
                 (pre final norm; skipped when A_last is None).
    A_mid is the operator of the stream at the output of block + 1 (layer block + 2), A_last that of the
    last layer. Each forward batch holds the clean originals and mirrors of the image chunk followed by
    the ablated copies, so every delta is taken against a clean run with the same batch shape.
    Returns {type: {key: float64 [n, N]}} with keys SWITCH_KEYS (cls_* are NaN without A_last).
    """
    from tqdm import tqdm

    m = adapter.model if model is None else model
    L = len(m.blocks)
    if not 0 <= block < L - 1:
        raise ValueError(f"block {block} has no following block (depth {L})")
    mid = block + 1
    want_cls = A_last is not None and adapter.n_prefix_tokens > 0
    stop = L if want_cls else mid + 1
    grid, npre = adapter.grid_hw, adapter.n_prefix_tokens
    N, T, d = X.shape
    dev, dt = X.device, X.dtype
    h = int(m.blocks[block].mlp.fc1.out_features)
    A_m = torch.as_tensor(np.asarray(A_mid, dtype=np.float64), dtype=dt, device=dev)
    A_l = torch.as_tensor(np.asarray(A_last, dtype=np.float64), dtype=dt, device=dev) if want_cls else None

    pairs = {k: np.asarray(v, dtype=np.int64).reshape(-1, 2) for k, v in pairs.items()}
    orig = sorted({int(i) for P in pairs.values() for i in P[:, 0]})
    mirr = sorted({int(j) for P in pairs.values() for j in P[:, 1]})
    units = [(n, 0) for n in orig] + [(n, 1) for n in mirr]
    slot = {u: k for k, u in enumerate(units)}
    n_o = len(orig)
    idx_i = {k: torch.as_tensor([slot[(int(i), 0)] for i in P[:, 0]], dtype=torch.long, device=dev)
             for k, P in pairs.items()}
    idx_j = {k: torch.as_tensor([slot[(int(j), 1)] - n_o for j in P[:, 1]], dtype=torch.long, device=dev)
             for k, P in pairs.items()}
    out = {k: {key: np.full((len(P), N), np.nan) for key in SWITCH_KEYS} for k, P in pairs.items()}

    c = max(1, int(chunk_images))
    g = max(1, (int(abl_batch) - 2 * c) // c)
    hook = ActMask(_act_of(m.blocks[block]))
    try:
        it = range(0, N, c)
        for s in tqdm(it, desc=f"switch-off block {block + 1}", leave=False, disable=not progress):
            x0, x1 = X[s:s + c], Xm[s:s + c]
            cc = x0.shape[0]
            D = torch.empty(len(units), cc, T, d, dtype=dt, device=dev)
            Dc = torch.empty(len(units), cc, d, dtype=dt, device=dev) if want_cls else None
            for gs in range(0, len(units), g):
                grp = units[gs:gs + g]
                inp = torch.cat([x0, x1] + [x0 if side == 0 else x1 for _, side in grp], dim=0)
                mask = torch.ones(inp.shape[0], 1, h, dtype=dt, device=dev)
                for k, (n, _) in enumerate(grp):
                    mask[(2 + k) * cc:(3 + k) * cc, 0, n] = 0
                hook.mask = mask
                with _autocast(amp, dev):
                    t = inp
                    t_mid = None
                    for kb in range(block, stop):
                        t = m.blocks[kb](t)
                        if kb == mid:
                            t_mid = t
                hook.mask = None
                t_mid = t_mid.to(dt).reshape(2 + len(grp), cc, T, d)
                last = t[:, 0].to(dt).reshape(2 + len(grp), cc, d) if want_cls else None
                for k, (_, side) in enumerate(grp):
                    delta = t_mid[2 + k] - t_mid[side]
                    if side == 1:
                        delta = unflip_tokens(delta, grid, npre)
                    D[gs + k] = center_channels(delta) if center else delta
                    if want_cls:
                        dc = last[2 + k] - last[side]
                        Dc[gs + k] = center_channels(dc) if center else dc
                del t, t_mid, inp, mask
            DA = D[:n_o] @ A_m                                   # orig-side deltas mapped by the operator
            DcA = Dc[:n_o] @ A_l if want_cls else None
            Dm = D[n_o:]
            Dcm = Dc[n_o:] if want_cls else None
            for k in pairs:
                ii, jj = idx_i[k], idx_j[k]
                for ps in range(0, len(ii), pair_chunk):
                    a, b = DA[ii[ps:ps + pair_chunk]], Dm[jj[ps:ps + pair_chunk]]
                    sl = slice(ps, ps + len(a))
                    rows = out[k]
                    rows["mid_dot"][sl, s:s + cc] = (a * b).sum((-1, -2)).double().cpu().numpy()
                    rows["mid_ni"][sl, s:s + cc] = (a * a).sum((-1, -2)).double().cpu().numpy()
                    rows["mid_nj"][sl, s:s + cc] = (b * b).sum((-1, -2)).double().cpu().numpy()
                    ap, bp = a[:, :, npre:], b[:, :, npre:]
                    rows["patch_dot"][sl, s:s + cc] = (ap * bp).sum((-1, -2)).double().cpu().numpy()
                    rows["patch_ni"][sl, s:s + cc] = (ap * ap).sum((-1, -2)).double().cpu().numpy()
                    rows["patch_nj"][sl, s:s + cc] = (bp * bp).sum((-1, -2)).double().cpu().numpy()
                    if want_cls:
                        ac, bc = DcA[ii[ps:ps + pair_chunk]], Dcm[jj[ps:ps + pair_chunk]]
                        rows["cls_dot"][sl, s:s + cc] = (ac * bc).sum(-1).double().cpu().numpy()
                        rows["cls_ni"][sl, s:s + cc] = (ac * ac).sum(-1).double().cpu().numpy()
                        rows["cls_nj"][sl, s:s + cc] = (bc * bc).sum(-1).double().cpu().numpy()
            del D, Dc, DA, DcA, Dm, Dcm
    finally:
        hook.remove()
    return out


def _cos(dot, ni, nj) -> float:
    den = math.sqrt(float(ni) * float(nj))
    return float(dot) / den if den > 0 else float("nan")


def summarise_pairs(res: dict) -> dict:
    """Per pair, from the per-image arrays of switch_off_block (one type):
        cos            sum_n dot / sqrt(sum_n ni * sum_n nj)  (all images pooled; large effects weigh more)
        cos_img_median median over images of the per-image cosine (images where both deltas are non-zero)
        norm_ratio     sqrt(sum nj / sum ni) = ||Delta_j(mirror x)|| / ||A Delta_i(x)|| over all images
        rms_i, rms_j   root-mean-square delta norms per image
    for the prefixes mid (all tokens), patch and cls."""
    out = {}
    n = res["mid_dot"].shape[0]
    for pre in ("mid", "patch", "cls"):
        dot, ni, nj = res[f"{pre}_dot"], res[f"{pre}_ni"], res[f"{pre}_nj"]
        cos, med, ratio, rms_i, rms_j = (np.full(n, np.nan) for _ in range(5))
        for p in range(n):
            if not np.isfinite(dot[p]).all():
                continue
            Si, Sj = ni[p].sum(), nj[p].sum()
            cos[p] = _cos(dot[p].sum(), Si, Sj)
            ratio[p] = math.sqrt(Sj / Si) if Si > 0 else np.nan
            ok = (ni[p] > 0) & (nj[p] > 0)
            if ok.any():
                med[p] = float(np.median(dot[p][ok] / np.sqrt(ni[p][ok] * nj[p][ok])))
            rms_i[p], rms_j[p] = math.sqrt(Si / ni.shape[1]), math.sqrt(Sj / nj.shape[1])
        out[f"{pre}_cos"], out[f"{pre}_cos_img_median"], out[f"{pre}_norm_ratio"] = cos, med, ratio
        out[f"{pre}_rms_i"], out[f"{pre}_rms_j"] = rms_i, rms_j
    return out


# ----------------------------------------------------------------------------------------------
# Crops and small statistics
# ----------------------------------------------------------------------------------------------
def token_crop(img: np.ndarray, pos: int, grid_w: int, patch: int, size: int = 42, fill: int = 228) -> np.ndarray:
    """size x size window of an [H, W, 3] uint8 image centred on patch token `pos` (row-major); parts
    outside the image are filled with `fill`. For patch 14 and size 42 this is the token's 3 x 3 patch
    neighbourhood, the crop of scripts/f7_visuals.py."""
    r, c = divmod(int(pos), int(grid_w))
    y0 = r * patch + (patch - size) // 2
    x0 = c * patch + (patch - size) // 2
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    H, W = img.shape[:2]
    ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + size, H), min(x0 + size, W)
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out


def _sig_path(path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".sig.json")


def output_current(path, signature: dict) -> bool:
    """True when `path` exists and was written with the same signature (the settings that shape it), so a
    resumed run may reuse it; a changed setting forces a recompute."""
    sp = _sig_path(path)
    if not Path(path).exists() or not sp.exists():
        return False
    return json.loads(sp.read_text(encoding="utf-8")) == json.loads(json.dumps(signature, sort_keys=True, default=str))


def write_signature(path, signature: dict) -> None:
    _sig_path(path).write_text(json.dumps(signature, sort_keys=True, indent=1, default=str), encoding="utf-8")


def describe(x) -> dict:
    """n, mean, median and the 10/25/75/90th percentiles of the finite values."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "q10": None, "q25": None, "q75": None, "q90": None}
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(q[2]), "q10": float(q[0]),
            "q25": float(q[1]), "q75": float(q[3]), "q90": float(q[4])}


def verdict(value, op: str, threshold: float) -> dict:
    """{"value", "op", "threshold", "pass"}; pass is None when the value is missing."""
    ok = None
    if value is not None and np.isfinite(value):
        ok = bool(value >= threshold) if op == ">=" else bool(value <= threshold)
    return {"value": None if value is None or not np.isfinite(value) else float(value), "op": op,
            "threshold": float(threshold), "pass": ok}


def peak_memory_gb(device) -> float | None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    return None
