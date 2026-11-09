#!/usr/bin/env python
"""F7, visuals: mirror-twin neurons and layer-0 filters (PAPER_EXPERIMENTS.md Track F, "F7, visuals";
paper Figure 1, the teaser).

    python scripts/f7_visuals.py --model dinov2_vitb14_reg --fit-run runs/x2/perlayer_reg_s0 \
        --blocks 1,3,6 --pairs 8 --images 5000 --out runs/f7
    python scripts/f7_visuals.py --tiny --out tiny_out/f7          # CPU test mode (tiny random ViT)
    python scripts/f7_visuals.py --replot runs/f7                  # re-draw the figures only (laptop fonts)

Blocks are numbered from 1 as in e1_mlp.csv: block b is model.blocks[b - 1] and reads stream layer b - 1.

1. Strongest twin pairs, read from the fit run (scripts/run_e1.py, results/e1_operators.npz): the
   mirror matching perms[b - 1], match_scores (S[i, pi(i)]) and self_corr (S[i, i]). Here
   S = (R + R^T) / 2 and R[i, j] = corr(zt_i, z_j) over all tokens of the fit images, with z the fc1
   pre-activations and zt those of the mirrored image un-flipped to the original positions. A pair
   (u, v) is ranked by its matching weight w_uv = 2 S_uv - S_uu - S_vv: the gain of pairing the two
   neurons over leaving both as self-twins, the quantity the matching maximises. It is high when the
   twins follow each other under the mirror (S_uv near 1) and neither neuron is mirror-symmetric on
   its own (S_uu, S_vv low), that is for neurons with a left-right preference. Ties are broken by S_uv,
   then by u. (--rank-by score ranks by S_uv alone; it can favour pairs of near-symmetric neurons,
   S_uu and S_vv near 1, whose patches look the same with or without the mirror.)
2. Visualisation set: --images ImageNet train images drawn without replacement (seed --seed) from the
   k-NN bank rows of the train cache (--rows bank, the default; disjoint from the fit images that
   chose the pairs, so the check below is out of sample) or from the fit rows of the run's fit seed
   (--rows fit). Never validation images.
3. One paired pass (each batch together with its mirror, as mirror.stats.run_paired; bf16 autocast on
   CUDA). In the chosen blocks the fc1 output goes through the block's own activation (GELU) in
   float32; only patch tokens are used (CLS and registers are dropped). Float64 accumulators per
   block, with a = post-GELU activation on the original and at = on the mirror, un-flipped (at_j at
   patch p is neuron j on the mirrored image at the mirrored position of p):
     - top patches: for each neuron of each chosen pair, the --topk images with the largest activation
       at any patch, at most one patch per image (its arg-max position), with the twin's and a random
       partner's mirror activation at that position;
     - correlations over all patch tokens of the set: corr(at_j, a_c) for every neuron j and every
       chosen neuron c, and for every neuron i corr(at_pi(i), a_i) (twin), corr(at_i, a_i) (self) and
       corr(at_rho(i), a_i), rho(i) a random partner that is neither i nor its twin (fixed seed).
     For an exactly equivariant block, at_pi(i) = a_i, so the twin correlation is exactly 1.
4. Layer 0: the patch-embedding filters W (d x 3 x p x p, centred over channels as the fit statistics
   are) in the eigenbasis of the layer-0 operator A_0. For each parity subspace (A_0 = +1 even,
   A_0 = -1 odd): the principal filters (right singular vectors of V^T W, V an orthonormal basis of
   the subspace), their energy shares and their symmetric-energy fractions ||(g + flip g) / 2||^2 /
   ||g||^2; the same for a random involution with the same d+ (control). Exact equivariance
   (A_0 W = W F, F the left-right flip of the kernel) makes every even filter symmetric (fraction 1)
   and every odd filter antisymmetric (fraction 0). Channels are not paired one to one because A_0 is
   dense in the channel basis (recorded: the median over rows of the largest squared entry).
5. Figures (vector PDF and 300 dpi PNG in --fig-dir, style of scripts/make_figures.py): fig0_teaser
   (the top --teaser-pairs pairs of every block by the ranking above, no hand-picking; a strip of
   even and odd layer-0 filters; twin r against random r over every pair of each block), figF7_twins_full (every chosen pair) and figF7_layer0_filters.

Writes <out>/f7_pairs.csv (chosen pairs), f7_blocks.csv (per block), f7_all_pairs.csv (every twin pair of the
chosen blocks), f7_top_patches.csv, f7_layer0_filters.csv, f7_layer0.npz, f7_crops.npz, f7_meta.json and
F7_SUMMARY.md. --tiny writes the figures to <out>/figures unless --fig-dir is given.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.15)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirror import variants as V  # noqa: E402
from mirror.data import make_loader, mirror, normalize_for  # noqa: E402
from mirror.findings import model_label, sha256_file  # noqa: E402
from mirror.stats import PairedHooks  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402

PAPER_FIG_DIR = REPO_ROOT / "results" / "paper" / "figures_paper"
DEFAULT_BLOCKS, TINY_BLOCKS = "1,3,6", "1,2,4"
ZERO_SHARE = 1e-12                 # filter directions with less energy share are numerically zero (e.g. the
                                   # all-ones direction after channel centring); kept in the CSV, not plotted
CROP_FILL = 228                    # padding outside the image (light grey, C_GRID)
CROP_UP = 2                        # crops are upsampled (bicubic) by this factor for the figures
FILTER_UP = 6                      # filters are upsampled (nearest) by this factor


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--fit-run", default=None, help="run_e1.py output directory or its e1_operators.npz")
    p.add_argument("--blocks", default=None, help=f"blocks numbered from 1 (default {DEFAULT_BLOCKS}; tiny {TINY_BLOCKS})")
    p.add_argument("--pairs", type=int, default=8, help="strongest twin pairs per block")
    p.add_argument("--images", type=int, default=5000, help="train images in the visualisation set")
    p.add_argument("--rows", choices=("bank", "fit"), default="bank", help="train rows the images are drawn from")
    p.add_argument("--rank-by", choices=("weight", "score"), default="weight",
                   help="weight: w = 2 S_uv - S_uu - S_vv (default); score: S_uv")
    p.add_argument("--topk", type=int, default=8, help="top patches kept per neuron")
    p.add_argument("--context", type=int, default=1, help="crop radius in patches (1: the 3 x 3 neighbourhood)")
    p.add_argument("--teaser-pairs", type=int, default=2, help="top-ranked pairs per block in the teaser")
    p.add_argument("--teaser-patches", type=int, default=4, help="patches per neuron in the teaser")
    p.add_argument("--full-patches", type=int, default=6, help="patches per neuron in figF7_twins_full")
    p.add_argument("--filters", type=int, default=6, help="layer-0 filters per parity in the teaser")
    p.add_argument("--full-filters", type=int, default=12, help="layer-0 filters per parity in figF7_layer0_filters")
    p.add_argument("--out", default="runs/f7")
    p.add_argument("--fig-dir", default=None, help=f"default {PAPER_FIG_DIR.relative_to(REPO_ROOT)} (tiny: <out>/figures)")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("bf16", "none"), default="bf16", help="autocast of the forward on CUDA")
    p.add_argument("--batch-size", type=int, default=32, help="images per batch (the pass runs 2x this)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-model-mismatch", action="store_true",
                   help="allow a fit run of another model (diagnostics only; recorded in the meta file)")
    p.add_argument("--tiny", action="store_true", help="tiny random timm ViT and synthetic images on the CPU")
    p.add_argument("--replot", default=None, metavar="RUN_DIR",
                   help="only re-draw the figures from the outputs of a finished run (no model, no images)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Fit run: operators, pairing and twin scores
# ----------------------------------------------------------------------------------------------
def operators_npz(fit_run) -> Path:
    p = Path(fit_run)
    return p if p.suffix == ".npz" else p / "results" / "e1_operators.npz"


def make_tiny_fit(adapter, proto, device, run_dir: Path) -> Path:
    """Tiny synthetic fit in the run_e1.py layout (with match_scores and self_corr), CPU tests only."""
    res = run_dir / "results"
    npz = res / "e1_operators.npz"
    if npz.exists():
        return run_dir
    res.mkdir(parents=True, exist_ok=True)
    st = V.fit_structure(adapter, proto.fit_loader(0), device, top_k=10, workers=1)
    S = V.collect_mirror_scores(adapter, proto.fit_loader(0), device, range(adapter.depth))
    np.savez_compressed(
        npz, A_layers=np.stack(st.A_layers).astype(np.float32), perms=np.stack(st.perms).astype(np.int32),
        match_scores=np.asarray(st.match_scores, dtype=np.float32),
        self_corr=np.stack([np.diag(S[b]) for b in range(adapter.depth)]).astype(np.float32),
        C=st.C.astype(np.float32), G=st.G.astype(np.float32))
    (res / "e1_meta.json").write_text(json.dumps({
        "fit_seed": 0, "seed": 0, "protocol": "tiny", "note": "tiny synthetic fit (CPU test only)",
        "config": {"stats": {"center": True}, "model": {"name": "tiny"}}}), encoding="utf-8")
    return run_dir


def load_fit(args, adapter, proto, device, out: Path) -> dict:
    fit_run = args.fit_run
    if fit_run is None:
        if not args.tiny:
            raise SystemExit("--fit-run is required (a run_e1.py output directory, e.g. runs/x2/perlayer_reg_s0)")
        fit_run = make_tiny_fit(adapter, proto, device, out / "tiny_fit")
    npz = operators_npz(fit_run)
    struct = V.load_fit_run(npz)
    with np.load(npz) as z:
        missing = [k for k in ("match_scores", "self_corr") if k not in z.files]
        if missing:
            raise SystemExit(f"{npz} has no {', '.join(missing)} (needs an operators file written by run_e1.py)")
        scores = np.asarray(z["match_scores"], dtype=np.float64)
        self_corr = np.asarray(z["self_corr"], dtype=np.float64)
    expected = "tiny" if args.tiny else args.model
    run_model = struct.meta.get("model")
    mismatch = run_model is not None and run_model != expected
    if mismatch and not args.allow_model_mismatch:
        raise SystemExit(f"the fit run is for {run_model}, not {expected} (pass --allow-model-mismatch for diagnostics)")
    if run_model is None:
        log(f"WARNING: {npz.parent} has no e1_meta.json; the fit run's model and fit seed are not recorded")
    if struct.d != adapter.embed_dim or struct.h != adapter.mlp_hidden or struct.depth != adapter.depth:
        raise SystemExit("the fit run's sizes do not match the model")
    return {"struct": struct, "npz": npz, "scores": scores, "self_corr": self_corr, "model": run_model,
            "mismatch": bool(mismatch), "sha256": sha256_file(npz), "fit_seed": struct.meta.get("fit_seed")}


def rank_pairs(perm, scores, self_corr, rank_by: str = "weight") -> pd.DataFrame:
    """Every twin pair (u < v) of a block with its fit-run scores, ranked (rank 1 = strongest)."""
    perm = np.asarray(perm, dtype=np.int64)
    idx = np.arange(len(perm))
    u = idx[perm > idx]
    v = perm[u]
    s_uv, s_vu = np.asarray(scores)[u], np.asarray(scores)[v]
    s_uu, s_vv = np.asarray(self_corr)[u], np.asarray(self_corr)[v]
    w = 2.0 * s_uv - s_uu - s_vv
    key = w if rank_by == "weight" else s_uv
    order = np.lexsort((u, -s_uv, -key))
    df = pd.DataFrame({"u": u[order], "v": v[order], "fit_w": w[order], "fit_S_uv": s_uv[order],
                       "fit_S_vu": s_vu[order], "fit_S_uu": s_uu[order], "fit_S_vv": s_vv[order]})
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def random_partners(perm, seed: int) -> np.ndarray:
    """rho(i): a uniformly random neuron that is neither i nor its twin perm[i] (fixed seed)."""
    perm = np.asarray(perm, dtype=np.int64)
    h = len(perm)
    if h < 3:
        raise ValueError("need at least 3 neurons")
    rng = np.random.default_rng(seed)
    idx = np.arange(h)
    rho = (idx + rng.integers(1, h, size=h)) % h      # never i
    bad = rho == perm
    while bad.any():
        rho[bad] = (idx[bad] + rng.integers(1, h, size=int(bad.sum()))) % h
        bad = rho == perm
    return rho


# ----------------------------------------------------------------------------------------------
# Paired pass: top patches and correlations
# ----------------------------------------------------------------------------------------------
class TopPatches:
    """Running top-k over images of each tracked neuron's largest patch activation.

    At most one patch per image (the image's arg-max position), so the k entries come from k different
    images. `extras` are other activation maps read at that same image and position."""

    def __init__(self, n: int, k: int, extras=(), device="cpu"):
        self.n, self.k = int(n), int(k)
        self.extras = tuple(extras)
        self.val = torch.full((self.n, self.k), -math.inf, dtype=torch.float64, device=device)
        self.img = torch.full((self.n, self.k), -1, dtype=torch.int64, device=device)
        self.pos = torch.full((self.n, self.k), -1, dtype=torch.int64, device=device)
        self.extra = {e: torch.full((self.n, self.k), math.nan, dtype=torch.float64, device=device)
                      for e in self.extras}

    @torch.no_grad()
    def update(self, acts: torch.Tensor, offset: int, extras: dict | None = None) -> None:
        """acts: [B, P, n], the tracked neurons on images offset .. offset + B - 1 (P patches each)."""
        extras = extras or {}
        if set(extras) != set(self.extras):
            raise ValueError(f"extras must be {self.extras}")
        B = acts.shape[0]
        vmax, pos = acts.max(dim=1)                                           # [B, n]
        img = torch.arange(offset, offset + B, device=acts.device).unsqueeze(1).expand(B, self.n)
        val = torch.cat([self.val, vmax.T.to(torch.float64)], dim=1)
        top = val.topk(self.k, dim=1).indices                                 # sorted, largest first
        self.val = val.gather(1, top)
        self.img = torch.cat([self.img, img.T.to(self.img.device)], dim=1).gather(1, top)
        self.pos = torch.cat([self.pos, pos.T], dim=1).gather(1, top)
        for e, m in extras.items():
            g = m.gather(1, pos.unsqueeze(1)).squeeze(1)                      # [B, n] at the arg-max
            self.extra[e] = torch.cat([self.extra[e], g.T.to(torch.float64)], dim=1).gather(1, top)

    def result(self) -> dict:
        out = {"val": self.val.cpu().numpy(), "img": self.img.cpu().numpy(), "pos": self.pos.cpu().numpy()}
        out.update({e: t.cpu().numpy() for e, t in self.extra.items()})
        return out


def _corr(cross_mean, m_x, m_y, v_x, v_y, eps: float = 1e-12):
    den = torch.sqrt(v_x * v_y)
    c = (cross_mean - m_x * m_y) / den.clamp_min(eps)
    return torch.where(den > eps, c, torch.full_like(c, math.nan)).clamp(-1.0, 1.0)


class BlockTracker:
    """Float64 accumulators of one block over the patch tokens of the visualisation set.

    `tracked` lists the chosen neurons (u0, v0, u1, v1, ...); `perm` is the fit run's twin pairing and
    `rho` the random partners."""

    def __init__(self, block: int, perm, rho, tracked, k: int, device, chunk: int = 4096):
        self.block = int(block)
        kw = dict(dtype=torch.float64, device=device)
        self.perm = torch.as_tensor(np.asarray(perm), dtype=torch.long, device=device)
        self.rho = torch.as_tensor(np.asarray(rho), dtype=torch.long, device=device)
        self.tracked = torch.as_tensor(np.asarray(tracked), dtype=torch.long, device=device)
        h, m = len(self.perm), len(self.tracked)
        self.h, self.chunk = h, int(chunk)
        self.n = torch.zeros((), **kw)
        self.s_a, self.s_a2, self.s_t, self.s_t2 = (torch.zeros(h, **kw) for _ in range(4))
        self.cross = torch.zeros(h, m, **kw)                                  # sum at_j a_c
        self.d_twin, self.d_self, self.d_rand = (torch.zeros(h, **kw) for _ in range(3))
        self.top = TopPatches(m, k, extras=("twin", "rand"), device=device)

    @torch.no_grad()
    def update(self, a: torch.Tensor, at: torch.Tensor, offset: int) -> None:
        """a, at: [B, P, h] post-activation patch tokens of the originals and of the un-flipped mirrors."""
        h = a.shape[-1]
        A2, T2 = a.reshape(-1, h), at.reshape(-1, h)
        for s in range(0, A2.shape[0], self.chunk):
            A = A2[s:s + self.chunk].to(torch.float64)
            T = T2[s:s + self.chunk].to(torch.float64)
            self.n += A.shape[0]
            self.s_a += A.sum(0)
            self.s_a2 += (A * A).sum(0)
            self.s_t += T.sum(0)
            self.s_t2 += (T * T).sum(0)
            self.cross += T.T @ A[:, self.tracked]
            self.d_twin += (T[:, self.perm] * A).sum(0)
            self.d_self += (T * A).sum(0)
            self.d_rand += (T[:, self.rho] * A).sum(0)
        tr = self.tracked
        self.top.update(a[:, :, tr], offset, {"twin": at[:, :, self.perm[tr]], "rand": at[:, :, self.rho[tr]]})

    def correlations(self) -> dict:
        n = self.n
        ma, mt = self.s_a / n, self.s_t / n
        va = (self.s_a2 / n - ma * ma).clamp_min(0)
        vt = (self.s_t2 / n - mt * mt).clamp_min(0)
        tr = self.tracked
        partner = _corr(self.cross / n, mt[:, None], ma[tr][None, :], vt[:, None], va[tr][None, :])
        ar = torch.arange(self.h, device=n.device)

        def diag(d, idx):
            return _corr(d / n, mt[idx], ma, vt[idx], va).cpu().numpy()

        return {"partner": partner.cpu().numpy(), "twin": diag(self.d_twin, self.perm),
                "self": diag(self.d_self, ar), "rand": diag(self.d_rand, self.rho),
                "n_tokens": int(n.item()), "mean_a": ma.cpu().numpy(), "std_a": va.sqrt().cpu().numpy()}


@torch.no_grad()
def collect(adapter, loader, device, trackers: dict, amp: str | None = None, model=None,
            desc: str = "F7 paired pass") -> int:
    """Run every batch with its mirror; `trackers` maps a 0-based block index to its BlockTracker.
    Returns the number of images seen."""
    from tqdm import tqdm

    m = adapter.model if model is None else model
    npre = adapter.n_prefix_tokens
    acts = {}
    for b in trackers:
        act = getattr(m.blocks[b].mlp, "act", None)
        if act is None:
            raise ValueError(f"block {b + 1} has no mlp.act")
        acts[b] = act
    state = {"offset": 0}

    def to_stats(t):
        return t if t.dtype == torch.float64 else t.float()

    def cb(b, z, zt):
        tr = trackers.get(b)
        if tr is not None:
            tr.update(acts[b](to_stats(z[:, npre:])), acts[b](to_stats(zt[:, npre:])), state["offset"])

    device = torch.device(device)
    use_amp = amp == "bf16" and device.type == "cuda"
    with PairedHooks(adapter, m, fc1_cb=cb):
        for imgs, _ in tqdm(loader, desc=desc, leave=False):
            x = normalize_for(adapter, imgs.to(device, non_blocking=True))
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
            with ctx:
                adapter.forward_tokens(torch.cat([x, mirror(x)], dim=0), m)
            state["offset"] += int(imgs.shape[0])
    return state["offset"]


def tracked_neurons(pairs: pd.DataFrame) -> np.ndarray:
    return np.asarray([n for u, v in zip(pairs["u"], pairs["v"]) for n in (u, v)], dtype=np.int64)


# ----------------------------------------------------------------------------------------------
# Analysis of one block
# ----------------------------------------------------------------------------------------------
def _rank_of(col: np.ndarray, j: int) -> int:
    c = np.where(np.isfinite(col), col, -np.inf)
    return 1 + int((c > c[j]).sum())


def _ratio(num, den) -> float:
    return float(num / den) if den > 0 else math.nan


def analyse_block(tracker: BlockTracker, pairs: pd.DataFrame, perm, rho, all_pairs: pd.DataFrame) -> tuple:
    """(rows of the chosen pairs, the block row, a table of every twin pair) from a filled tracker."""
    corr = tracker.correlations()
    P = corr["partner"]
    top = tracker.top.result()
    perm, rho = np.asarray(perm), np.asarray(rho)
    h = len(perm)
    rows = []
    for i, r in enumerate(pairs.itertuples(index=False)):
        u, v = int(r.u), int(r.v)
        cu, cv = 2 * i, 2 * i + 1
        row = {"block": tracker.block, "rank": int(r.rank), "u": u, "v": v, "fit_w": r.fit_w,
               "fit_S_uv": r.fit_S_uv, "fit_S_uu": r.fit_S_uu, "fit_S_vv": r.fit_S_vv,
               "corr_twin_vu": float(P[v, cu]), "corr_twin_uv": float(P[u, cv]),
               "corr_self_u": float(P[u, cu]), "corr_self_v": float(P[v, cv]),
               "rand_u": int(rho[u]), "rand_v": int(rho[v]),
               "corr_rand_u": float(P[rho[u], cu]), "corr_rand_v": float(P[rho[v], cv]),
               "twin_rank_u": _rank_of(P[:, cu], v), "twin_rank_v": _rank_of(P[:, cv], u)}
        row["corr_twin"] = 0.5 * (row["corr_twin_vu"] + row["corr_twin_uv"])
        row["corr_rand"] = 0.5 * (row["corr_rand_u"] + row["corr_rand_v"])
        others = np.ones(h, dtype=bool)
        others[[u, v]] = False
        for tag, c in (("u", cu), ("v", cv)):
            col = P[others, c]
            row[f"max_other_{tag}"] = float(np.nanmax(col)) if np.isfinite(col).any() else math.nan
            row[f"mean_abs_other_{tag}"] = float(np.nanmean(np.abs(col))) if np.isfinite(col).any() else math.nan
            ok = top["img"][c] >= 0
            act = top["val"][c][ok]
            row[f"top_act_mean_{tag}"] = float(act.mean()) if ok.any() else math.nan
            row[f"top_ratio_twin_{tag}"] = _ratio(top["twin"][c][ok].mean(), act.mean()) if ok.any() else math.nan
            row[f"top_ratio_rand_{tag}"] = _ratio(top["rand"][c][ok].mean(), act.mean()) if ok.any() else math.nan
        row["top_ratio_twin"] = 0.5 * (row["top_ratio_twin_u"] + row["top_ratio_twin_v"])
        row["top_ratio_rand"] = 0.5 * (row["top_ratio_rand_u"] + row["top_ratio_rand_v"])
        row["n_tokens"] = corr["n_tokens"]
        rows.append(row)

    paired = perm != np.arange(h)
    tw_pair = 0.5 * (corr["twin"][all_pairs["u"].to_numpy()] + corr["twin"][all_pairs["v"].to_numpy()])
    rd_pair = 0.5 * (corr["rand"][all_pairs["u"].to_numpy()] + corr["rand"][all_pairs["v"].to_numpy()])
    tw, rd, sf = corr["twin"][paired], corr["rand"][paired], corr["self"][paired]
    both = np.isfinite(tw) & np.isfinite(rd)
    selfs = corr["self"][~paired]
    chosen = np.asarray([r["corr_twin"] for r in rows])
    fin = tw_pair[np.isfinite(tw_pair)]
    tr = tracker.tracked.cpu().numpy()
    diag_check = [abs(P[perm[n], c] - corr["twin"][n]) for c, n in enumerate(tr)
                  if np.isfinite(P[perm[n], c]) and np.isfinite(corr["twin"][n])]

    def q(x, p):
        x = x[np.isfinite(x)]
        return float(np.quantile(x, p)) if x.size else math.nan

    block_row = {
        "block": tracker.block, "h": h, "n_pairs": int(paired.sum() // 2), "n_self": int((~paired).sum()),
        "n_tokens": corr["n_tokens"], "n_chosen": len(rows),
        "pairs_twin_corr_median": q(tw_pair, 0.5), "pairs_twin_corr_q10": q(tw_pair, 0.1),
        "pairs_twin_corr_q90": q(tw_pair, 0.9), "pairs_rand_corr_median": q(rd_pair, 0.5),
        "pairs_rand_corr_q10": q(rd_pair, 0.1), "pairs_rand_corr_q90": q(rd_pair, 0.9),
        "paired_neurons_twin_corr_median": q(tw, 0.5), "paired_neurons_rand_corr_median": q(rd, 0.5),
        "paired_neurons_self_corr_median": q(sf, 0.5), "self_twins_self_corr_median": q(selfs, 0.5),
        "frac_paired_neurons_twin_gt_rand": float(np.mean(tw[both] > rd[both])) if both.any() else math.nan,
        "fit_w_median_all_pairs": float(all_pairs["fit_w"].median()) if len(all_pairs) else math.nan,
        "fit_w_min_chosen": float(pairs["fit_w"].min()) if len(pairs) else math.nan,
        "chosen_twin_corr_mean": float(np.nanmean(chosen)) if len(rows) else math.nan,
        "chosen_twin_corr_min": float(np.nanmin(chosen)) if len(rows) else math.nan,
        "chosen_rand_corr_mean": float(np.nanmean([r["corr_rand"] for r in rows])) if rows else math.nan,
        "chosen_twin_rank1_both": int(sum(r["twin_rank_u"] == 1 and r["twin_rank_v"] == 1 for r in rows)),
        "chosen_twin_corr_percentile_median": float(np.median([100.0 * np.mean(fin <= c) for c in chosen]))
        if rows and fin.size else math.nan,
        "partner_vs_diag_max_abs_diff": float(max(diag_check)) if diag_check else math.nan,
    }
    u_all, v_all = all_pairs["u"].to_numpy(), all_pairs["v"].to_numpy()
    every = all_pairs[["rank", "u", "v", "fit_w", "fit_S_uv", "fit_S_uu", "fit_S_vv"]].copy()
    every.insert(0, "block", tracker.block)
    every["corr_twin_vu"], every["corr_twin_uv"] = corr["twin"][u_all], corr["twin"][v_all]
    every["corr_twin"] = tw_pair
    every["corr_rand_u"], every["corr_rand_v"] = corr["rand"][u_all], corr["rand"][v_all]
    every["corr_rand"] = rd_pair
    every["corr_self_u"], every["corr_self_v"] = corr["self"][u_all], corr["self"][v_all]
    every["chosen"] = every["rank"] <= len(rows)
    return rows, block_row, every


# ----------------------------------------------------------------------------------------------
# Crops
# ----------------------------------------------------------------------------------------------
def patch_crop(img: np.ndarray, pos: int, grid_w: int, patch: int, context: int, fill: int = CROP_FILL) -> np.ndarray:
    """The (2 context + 1)^2 patch neighbourhood of patch `pos` (row-major) of an [H, W, 3] uint8 image;
    the parts outside the image are filled with `fill`."""
    r, c = divmod(int(pos), int(grid_w))
    size = (2 * context + 1) * patch
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    y0, x0 = (r - context) * patch, (c - context) * patch
    H, W = img.shape[:2]
    ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + size, H), min(x0 + size, W)
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out


def block_crops(tracker: BlockTracker, pairs: pd.DataFrame, images, vis_rows, grid_w, patch, context):
    """Crops [n_pairs, 2, k, S, S, 3] uint8, validity [n_pairs, 2, k] and the top-patch table rows."""
    top = tracker.top.result()
    k = tracker.top.k
    S = (2 * context + 1) * patch
    crops = np.full((len(pairs), 2, k, S, S, 3), CROP_FILL, dtype=np.uint8)
    valid = np.zeros((len(pairs), 2, k), dtype=bool)
    perm = tracker.perm.cpu().numpy()
    rho = tracker.rho.cpu().numpy()
    table = []
    for i, r in enumerate(pairs.itertuples(index=False)):
        for j, (role, n) in enumerate((("u", int(r.u)), ("v", int(r.v)))):
            c = 2 * i + j
            for t in range(k):
                vi = int(top["img"][c, t])
                if vi < 0:
                    continue
                row = int(vis_rows[vi])
                pos = int(top["pos"][c, t])
                crops[i, j, t] = patch_crop(np.asarray(images[row]), pos, grid_w, patch, context)
                valid[i, j, t] = True
                pr, pc = divmod(pos, grid_w)
                table.append({"block": tracker.block, "rank": int(r.rank), "role": role, "neuron": n,
                              "twin": int(perm[n]), "rand_partner": int(rho[n]), "top": t + 1, "vis_index": vi,
                              "train_row": row, "patch": pos, "patch_row": pr, "patch_col": pc,
                              "activation": float(top["val"][c, t]),
                              "twin_on_mirror": float(top["twin"][c, t]),
                              "rand_on_mirror": float(top["rand"][c, t])})
    return crops, valid, table


# ----------------------------------------------------------------------------------------------
# Layer 0
# ----------------------------------------------------------------------------------------------
def flip_kernels(M: np.ndarray, p: int) -> np.ndarray:
    """Rows are flattened [3, p, p] kernels; mirror each left-right (the kernel width axis)."""
    return np.ascontiguousarray(M.reshape(M.shape[0], -1, p, p)[..., ::-1]).reshape(M.shape)


def sym_fraction_rows(M: np.ndarray, p: int) -> np.ndarray:
    """||(g + flip g) / 2||^2 / ||g||^2 per row: 1 for a symmetric kernel, 0 for an antisymmetric one."""
    S = 0.5 * (M + flip_kernels(M, p))
    den = (M * M).sum(1)
    return np.where(den > 0, (S * S).sum(1) / np.where(den > 0, den, 1.0), np.nan)


def _sym_energy(M: np.ndarray, p: int) -> float:
    S = 0.5 * (M + flip_kernels(M, p))
    tot = float((M * M).sum())
    return float((S * S).sum()) / tot if tot > 0 else math.nan


def _parity_filters(W: np.ndarray, A: np.ndarray, p: int, tot: float) -> dict:
    lam, Q = np.linalg.eigh(0.5 * (A + A.T))
    out = {}
    for parity, mask in (("even", lam > 0), ("odd", lam < 0)):
        M = Q[:, mask].T @ W
        _, s, Gt = np.linalg.svd(M, full_matrices=False)
        sign = np.sign(Gt[np.arange(len(Gt)), np.abs(Gt).argmax(1)])         # largest entry positive
        Gt = Gt * np.where(sign == 0, 1.0, sign)[:, None]
        shares = s ** 2 / tot
        out[parity] = {"dim": int(mask.sum()), "energy_share": float((M * M).sum()) / tot,
                       "sym_fraction": _sym_energy(M, p), "singular_values": s, "energy_shares": shares,
                       "zero": shares < ZERO_SHARE, "sym_fractions": sym_fraction_rows(Gt, p), "filters": Gt}
    return out


def layer0_analysis(W_pe, A0, null_seed: int) -> dict:
    """Patch-embedding filters in the eigenbasis of A_0 and of a random involution with the same d+.

    W_pe: [d, C, p, p]. The filters are centred over channels first (the fit statistics are channel
    centred and A_0 fixes the all-ones direction, so the channel mean carries no mirror information)."""
    W = np.asarray(W_pe, dtype=np.float64)
    d, p = W.shape[0], W.shape[-1]
    Wf = W.reshape(d, -1)
    Wc = Wf - Wf.mean(0, keepdims=True)
    tot = float((Wc * Wc).sum())
    A = V.clean_involution(A0)[0]
    d_plus = V.d_plus_of(A)
    R = V.random_operator(d, d_plus, null_seed)

    def share(Aop):
        Weq = 0.5 * (Wc + Aop @ flip_kernels(Wc, p))
        return float(((Wc - Weq) ** 2).sum()) / tot

    return {"d": d, "patch": p, "d_plus": d_plus, "d_minus": d - d_plus,
            "fitted": _parity_filters(Wc, A, p, tot), "null": _parity_filters(Wc, R, p, tot),
            "all_sym_fraction": _sym_energy(Wc, p), "breaking_share": share(A), "breaking_share_null": share(R),
            "a0_row_max_sq_median": float(np.median((A * A).max(1))), "null_seed": int(null_seed)}


def layer0_table(l0: dict) -> pd.DataFrame:
    rows = []
    for op in ("fitted", "null"):
        for parity in ("even", "odd"):
            f = l0[op][parity]
            for k in range(len(f["singular_values"])):
                rows.append({"operator": "A_0" if op == "fitted" else "random involution", "parity": parity,
                             "rank": k + 1, "singular_value": float(f["singular_values"][k]),
                             "energy_share": float(f["energy_shares"][k]),
                             "sym_fraction": float(f["sym_fractions"][k]), "numerically_zero": bool(f["zero"][k])})
    return pd.DataFrame(rows)


def filter_rgb(g: np.ndarray, p: int) -> np.ndarray:
    """A flattened [C, p, p] filter as an RGB image in [0, 1], zero at mid grey (0.5 + 0.5 g / max|g|)."""
    f = g.reshape(-1, p, p)
    m = float(np.abs(f).max())
    img = 0.5 + 0.5 * f / (m if m > 0 else 1.0)
    img = np.transpose(img, (1, 2, 0))
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return np.clip(img[:, :, :3], 0.0, 1.0)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def _style():
    """scripts/make_figures.py: its rcParams, fonts and palette (loaded without running its main)."""
    spec = importlib.util.spec_from_file_location("f7_make_figures_style", ROOT / "scripts" / "make_figures.py")
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    mf.setup_style()
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial may be missing on the VM
    return mf


def _save(fig, fig_dir: Path, stem: str) -> list[Path]:
    import matplotlib.pyplot as plt

    pdf, png = fig_dir / f"{stem}.pdf", fig_dir / f"{stem}.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Creator": "scripts/f7_visuals.py"})
    fig.savefig(png, dpi=300, metadata={"Software": "scripts/f7_visuals.py"})
    w, h = fig.get_size_inches()
    plt.close(fig)
    log(f"wrote {pdf} and .png ({w:.2f} x {h:.2f} in)")
    return [pdf, png]


def _up(crop: np.ndarray, factor: int = CROP_UP) -> np.ndarray:
    from PIL import Image

    if factor == 1:
        return crop.astype(np.float32) / 255.0
    im = Image.fromarray(crop).resize((crop.shape[1] * factor, crop.shape[0] * factor), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def _paste(canvas: np.ndarray, img: np.ndarray, y: int, x: int) -> None:
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img


def _paint_pair(canvas, y, x, crops, valid, k, S, gap, mid) -> dict:
    """Paste u's and v's first k crops side by side; return the crop corners and the mirror axis x."""
    corners = []
    group = k * S + (k - 1) * gap
    for j, x0 in enumerate((x, x + group + mid)):
        for t in range(k):
            if t < crops.shape[1] and valid[j, t]:
                xx = x0 + t * (S + gap)
                _paste(canvas, _up(crops[j, t]), y, xx)
                corners.append((xx, y))
    return {"corners": corners, "axis_x": x + group + mid / 2.0, "end_x": x + 2 * group + mid}


def _centre_boxes(ax, corners, S, context, colour):
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle

    n = 2 * context + 1
    side = S / n
    for xx, yy in corners:
        rect = Rectangle((xx + context * side, yy + context * side), side, side, fill=False, edgecolor=colour,
                         linewidth=0.45, zorder=3)
        rect.set_path_effects([pe.withStroke(linewidth=0.9, foreground="black")])
        ax.add_patch(rect)


def _image_axes(fig, rect, canvas):
    ax = fig.add_axes(rect)
    ax.imshow(canvas, extent=(0, canvas.shape[1], canvas.shape[0], 0), interpolation="antialiased")
    ax.set_xlim(0, canvas.shape[1])
    ax.set_ylim(canvas.shape[0], 0)
    ax.set_axis_off()
    return ax


def _filter_strip(canvas, l0, n, FS, gap, LM, top, row_h) -> list:
    """Paste the first n even and odd principal filters into `canvas` (one row per parity); return the
    placements of their symmetric-energy fractions."""
    texts = []
    p = l0["patch"]
    for r, parity in enumerate(("even", "odd")):
        f = l0["fitted"][parity]
        y = top + r * row_h
        for k in range(min(n, len(f["filters"]))):
            img = np.repeat(np.repeat(filter_rgb(f["filters"][k], p), FILTER_UP, 0), FILTER_UP, 1)
            x = LM + k * (FS + gap)
            _paste(canvas, img.astype(np.float32), y, x)
            texts.append((x + FS / 2.0, y + FS + 6, f"{f['sym_fractions'][k]:.2f}"))
    return texts


def fig_teaser(mf, blocks_data, blocks_df, l0, args, fig_dir, model_name) -> list[Path]:
    """(a) the top-ranked pairs of each block; (b) layer-0 filters, one row per parity; (c) twin r of every
    pair of each block against the random-partner control (median and 10th to 90th percentile)."""
    import matplotlib.pyplot as plt

    from mirror.report import C_BLUE, C_INK, C_MUTED, C_NULL

    k, n_t = args.teaser_patches, args.teaser_pairs
    S = (2 * args.context + 1) * (blocks_data[0]["patch"] if blocks_data else 14) * CROP_UP
    gap, mid, LM, RM, TM, RG, BG = 6, 44, 150, 230, 54, 10, 30
    group = k * S + (k - 1) * gap
    Wc = LM + 2 * group + mid + RM
    y, placements = TM, []
    for bd in blocks_data:
        y_block = y
        for pr in bd["pairs"][:n_t]:
            placements.append((bd, pr, y))
            y += S + RG
        if bd["pairs"][:n_t]:
            y -= RG
        bd["_band"] = (y_block, y)
        y += BG
    Hc = max(y - BG + 8, TM + 10)
    canvas = np.ones((Hc, Wc, 3), dtype=np.float32)
    infos = [(pr, yy, _paint_pair(canvas, yy, LM, pr["crops"], pr["valid"], k, S, gap, mid)) for _, pr, yy in placements]

    nf = args.filters
    FS, fgap, FLM, FTOP, FROW = l0["patch"] * FILTER_UP, 12, 210, 6, l0["patch"] * FILTER_UP + 40
    W2 = FLM + nf * FS + (nf - 1) * fgap + 8
    H2 = FTOP + 2 * FROW
    canvas2 = np.ones((H2, W2, 3), dtype=np.float32)
    texts2 = _filter_strip(canvas2, l0, nf, FS, fgap, FLM, FTOP, FROW)

    fig_w, title_h, bottom = mf.DOUBLE, 0.22, 0.04
    left_w = fig_w * 0.585
    scale = left_w / Wc                                                        # inches per canvas pixel
    fig_h = Hc * scale + title_h + bottom
    rx = left_w + 0.14
    right_w = fig_w - rx
    b_h = H2 * right_w / W2
    fig = plt.figure(figsize=(fig_w, fig_h))
    axa = _image_axes(fig, [0.0, bottom / fig_h, left_w / fig_w, Hc * scale / fig_h], canvas)
    axb = _image_axes(fig, [rx / fig_w, (fig_h - title_h - b_h) / fig_h, right_w / fig_w, b_h / fig_h], canvas2)
    mf.panel_title(axa, "a", f"Mirror-twin MLP neurons, {model_label(model_name, short=True)}")
    mf.panel_title(axb, "b", "Layer-0 patch filters, eigenbasis of $A_0$")
    small = mf.FONT - 1.0

    ycol = TM - 10
    axa.text(LM + group / 2.0, ycol, f"neuron $u$: top {k} train patches", ha="center", va="bottom",
             fontsize=small, color=C_INK)
    axa.text(LM + group + mid + group / 2.0, ycol, f"its mirror twin $v$: top {k}", ha="center", va="bottom",
             fontsize=small, color=C_INK)
    axa.text(LM + 2 * group + mid + 22, ycol, "twin $r$ (random)", ha="left", va="bottom", fontsize=small,
             color=C_MUTED)
    for bd in blocks_data:
        y0, y1 = bd["_band"]
        if y1 > y0:
            axa.text(LM - 18, 0.5 * (y0 + y1), f"Block {bd['block']}", ha="right", va="center", fontsize=mf.FONT,
                     color=C_INK)
    axis_x = LM + group + mid / 2.0
    axa.plot([axis_x, axis_x], [TM - 2, Hc - 6], color=C_MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    for pr, yy, info in infos:
        _centre_boxes(axa, info["corners"], S, args.context, "white")
        axa.text(info["end_x"] + 22, yy + S / 2.0, f"{pr['corr_twin']:.2f}  ({pr['corr_rand']:.2f})", ha="left",
                 va="center", fontsize=small, color=C_INK)

    for r, lab in enumerate(("even\n$A_0 = +1$", "odd\n$A_0 = -1$")):
        axb.text(FLM - 16, FTOP + r * FROW + FS / 2.0, lab, ha="right", va="center", fontsize=small, color=C_INK)
    for x, yy, t in texts2:
        axb.text(x, yy, t, ha="center", va="top", fontsize=mf.FONT - 1.5, color=C_MUTED)
    f = l0["fitted"]
    axb.text(FLM, H2 + 2, "number: symmetric-energy fraction\n"
             f"whole subspace: even {f['even']['sym_fraction']:.2f}, "
             f"odd {f['odd']['sym_fraction']:.2f}", ha="left", va="top", fontsize=mf.FONT - 1.5, color=C_MUTED,
             clip_on=False, linespacing=1.3)

    # (c) every twin pair of each block
    c_top = fig_h - title_h - b_h - 0.36 - title_h
    c_bottom = 0.30
    if c_top - c_bottom > 0.3 and len(blocks_df):
        axc = fig.add_axes([(rx + 0.42) / fig_w, c_bottom / fig_h, (right_w - 0.56) / fig_w, (c_top - c_bottom) / fig_h])
        bl = list(blocks_df["block"])
        for i, r in enumerate(blocks_df.itertuples(index=False)):
            yy = len(bl) - 1 - i
            for off, med, lo, hi, colour, marker in (
                    (0.17, r.pairs_twin_corr_median, r.pairs_twin_corr_q10, r.pairs_twin_corr_q90, C_BLUE, "o"),
                    (-0.17, r.pairs_rand_corr_median, r.pairs_rand_corr_q10, r.pairs_rand_corr_q90, C_NULL, "s")):
                axc.plot([lo, hi], [yy + off] * 2, color=colour, linewidth=1.2, solid_capstyle="butt")
                axc.plot([med], [yy + off], marker=marker, color=colour, markersize=3.6, markeredgecolor="white",
                         markeredgewidth=0.4, linestyle="none")
        axc.set_yticks(range(len(bl)))
        axc.set_yticklabels([f"Block {b}" for b in reversed(bl)])
        axc.set_ylim(-0.6, len(bl) - 0.4)
        axc.set_xlim(-0.25, 1.0)
        axc.grid(True, axis="x")
        axc.set_xlabel("correlation over all patches", labelpad=1.5)
        first = blocks_df.iloc[0]
        axc.text(first["pairs_twin_corr_q10"], len(bl) - 1 + 0.17, "twin  ", ha="right", va="center",
                 fontsize=mf.FONT - 1.5, color=C_INK)
        axc.text(first["pairs_rand_corr_q90"], len(bl) - 1 - 0.17, "  random partner", ha="left", va="center",
                 fontsize=mf.FONT - 1.5, color=C_INK)
        mf.panel_title(axc, "c", "Every twin pair of the block")
    return _save(fig, fig_dir, "fig0_teaser")


def fig_twins_full(mf, blocks_data, args, fig_dir, model_name, n_images) -> list[Path]:
    import matplotlib.pyplot as plt

    from mirror.report import C_GRID, C_INK, C_MUTED

    k = args.full_patches
    S = (2 * args.context + 1) * blocks_data[0]["patch"] * CROP_UP if blocks_data else 84
    gap, mid, LM, CG, CAP, RG, BG, TM = 5, 34, 90, 90, 46, 12, 34, 60
    group = k * S + (k - 1) * gap
    CW = 2 * group + mid
    ncol = 2
    Wc = LM + ncol * CW + (ncol - 1) * CG + 6
    y = TM
    layout = []
    for bd in blocks_data:
        nrow = int(math.ceil(len(bd["pairs"]) / ncol))
        y_block = y
        for i, pr in enumerate(bd["pairs"]):
            rr, cc = divmod(i, ncol)
            layout.append((bd, pr, y + rr * (CAP + S + RG) + CAP, LM + cc * (CW + CG)))
        y += nrow * (CAP + S + RG) - RG if nrow else 0
        bd["_band_full"] = (y_block, y)
        y += BG
    Hc = y - BG + 10
    canvas = np.ones((Hc, Wc, 3), dtype=np.float32)
    infos = [(bd, pr, yy, xx, _paint_pair(canvas, yy, xx, pr["crops"], pr["valid"], k, S, gap, mid))
             for bd, pr, yy, xx in layout]
    fig_w = mf.DOUBLE
    scale = fig_w / Wc
    title_h = 0.24
    fig_h = Hc * scale + title_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = _image_axes(fig, [0.0, 0.0, 1.0, Hc * scale / fig_h], canvas)
    ax.set_title(f"Mirror-twin pairs of {model_label(model_name)}: top {k} train patches of each neuron "
                 f"({n_images:,} images; ranked by matching weight)", loc="left", fontsize=mf.FONT, color=C_INK)
    small = mf.FONT - 1.5
    for cc in range(ncol):
        x0 = LM + cc * (CW + CG)
        ax.text(x0 + group / 2.0, TM - 14, "neuron $u$", ha="center", va="bottom", fontsize=mf.FONT - 1, color=C_INK)
        ax.text(x0 + group + mid + group / 2.0, TM - 14, "twin $v$", ha="center", va="bottom", fontsize=mf.FONT - 1,
                color=C_INK)
    for i, bd in enumerate(blocks_data):
        y0, y1 = bd["_band_full"]
        if y1 > y0:
            ax.text(LM / 2.0 - 8, 0.5 * (y0 + y1), f"Block {bd['block']}", ha="center", va="center",
                    rotation=90, fontsize=mf.FONT, color=C_INK)
        if i > 0:
            ax.plot([6, Wc - 6], [y0 - BG / 2.0, y0 - BG / 2.0], color=C_GRID, linewidth=0.6)
    for bd, pr, yy, xx, info in infos:
        _centre_boxes(ax, info["corners"], S, args.context, "white")
        ax.plot([info["axis_x"]] * 2, [yy - 2, yy + S + 2], color=C_MUTED, linewidth=0.5, linestyle=(0, (2, 2)))
        ax.text(xx, yy - 8, f"#{pr['rank']}  neurons {pr['u']} and {pr['v']}    twin $r$ = {pr['corr_twin']:.2f}    "
                f"random $r$ = {pr['corr_rand']:.2f}    $w$ = {pr['fit_w']:.2f}", ha="left", va="bottom",
                fontsize=small, color=C_INK)
    return _save(fig, fig_dir, "figF7_twins_full")


def fig_layer0(mf, l0, args, fig_dir, model_name) -> list[Path]:
    import matplotlib.pyplot as plt

    from mirror.report import C_BLUE, C_INK, C_MUTED, C_NULL, C_ORANGE

    nf = args.full_filters
    FS, fgap, FLM, FTOP, FROW = l0["patch"] * FILTER_UP, 12, 200, 8, l0["patch"] * FILTER_UP + 40
    W2 = FLM + nf * FS + (nf - 1) * fgap + 10
    H2 = FTOP + 2 * FROW
    canvas = np.ones((H2, W2, 3), dtype=np.float32)
    texts = _filter_strip(canvas, l0, nf, FS, fgap, FLM, FTOP, FROW)
    fig_w = mf.DOUBLE
    strip_h = H2 * fig_w / W2
    plot_h, gap_h, title_h = 1.75, 0.30, 0.22
    fig_h = title_h + strip_h + gap_h + plot_h + 0.32
    fig = plt.figure(figsize=(fig_w, fig_h))
    axa = _image_axes(fig, [0.0, (fig_h - title_h - strip_h) / fig_h, 1.0, strip_h / fig_h], canvas)
    mf.panel_title(axa, "a", f"Principal patch-embedding filters of {model_label(model_name, short=True)} in the "
                             "even and odd subspaces of $A_0$ (number: symmetric-energy fraction)")
    for r, lab in enumerate(("even ($A_0 = +1$)", "odd ($A_0 = -1$)")):
        axa.text(FLM - 14, FTOP + r * FROW + FS / 2.0, lab, ha="right", va="center", fontsize=mf.FONT - 1, color=C_INK)
    for x, y, t in texts:
        axa.text(x, y, t, ha="center", va="top", fontsize=mf.FONT - 1.5, color=C_MUTED)

    axb = fig.add_axes([0.075, 0.32 / fig_h, 0.42, plot_h / fig_h])
    for op, parity, colour, marker, filled in (("null", "even", C_NULL, "o", False), ("null", "odd", C_NULL, "s", False),
                                               ("fitted", "even", C_BLUE, "o", True), ("fitted", "odd", C_ORANGE, "s", True)):
        f = l0[op][parity]
        keep = ~f["zero"]
        lab = (f"{parity}, {'$A_0$' if op == 'fitted' else 'random involution'} "
               f"(whole subspace {f['sym_fraction']:.2f})")
        axb.scatter(f["energy_shares"][keep], f["sym_fractions"][keep], s=4 if filled else 5, marker=marker,
                    facecolors=colour if filled else "white", edgecolors=colour, linewidths=0.4 if filled else 0.5,
                    label=lab, zorder=3 if filled else 2)
    axb.set_xscale("log")
    from matplotlib.ticker import NullFormatter

    axb.xaxis.set_minor_formatter(NullFormatter())
    axb.set_ylim(-0.03, 1.03)
    axb.set_xlabel("energy share of the filter direction")
    axb.set_ylabel("symmetric-energy fraction")
    axb.grid(True, axis="y")
    mf.panel_title(axb, "b", "Every principal filter, by parity")
    axb.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=mf.FONT - 1, markerscale=2.0)
    f = l0["fitted"]
    note = (f"Patch embedding breaking share {l0['breaking_share']:.3f} (random involution "
            f"{l0['breaking_share_null']:.3f}); all filters {l0['all_sym_fraction']:.2f} symmetric.\n"
            f"$A_0$: d+ = {l0['d_plus']}, d- = {l0['d_minus']}; dense in the channel basis (median largest "
            f"squared entry per row {l0['a0_row_max_sq_median']:.3f}), so channels are not paired one to one.")
    fig.text(0.53, 0.32 / fig_h + 0.08, note, ha="left", va="bottom", fontsize=mf.FONT - 1.5, color=C_MUTED,
             wrap=True)
    return _save(fig, fig_dir, "figF7_layer0_filters")


# ----------------------------------------------------------------------------------------------
# Saved layer-0 arrays and re-drawing the figures from a finished run (no model, no images)
# ----------------------------------------------------------------------------------------------
L0_SCALARS = ("d", "patch", "d_plus", "d_minus", "all_sym_fraction", "breaking_share", "breaking_share_null",
              "a0_row_max_sq_median", "null_seed")
L0_ARRAYS = ("singular_values", "energy_shares", "zero", "sym_fractions")


def save_layer0(path: Path, l0: dict, n_filters: int) -> None:
    """f7_layer0.npz: the scalars, every principal direction's numbers and the first n_filters filters."""
    arrays = {k: np.asarray(l0[k]) for k in L0_SCALARS}
    for op in ("fitted", "null"):
        for parity in ("even", "odd"):
            f = l0[op][parity]
            for k in L0_ARRAYS:
                arrays[f"{op}_{parity}_{k}"] = np.asarray(f[k])
            for k in ("dim", "energy_share", "sym_fraction"):
                arrays[f"{op}_{parity}_{k}"] = np.asarray(f[k])
            if op == "fitted":
                arrays[f"{op}_{parity}_filters"] = f["filters"][:n_filters]
    np.savez_compressed(path, **arrays)


def load_layer0(path: Path) -> dict:
    with np.load(path) as z:
        l0 = {k: z[k].item() for k in L0_SCALARS}
        for op in ("fitted", "null"):
            l0[op] = {}
            for parity in ("even", "odd"):
                f = {k: z[f"{op}_{parity}_{k}"] for k in L0_ARRAYS}
                f.update({k: z[f"{op}_{parity}_{k}"].item() for k in ("dim", "energy_share", "sym_fraction")})
                if op == "fitted":
                    f["filters"] = z[f"{op}_{parity}_filters"]
                l0[op][parity] = f
    return l0


def blocks_for_figures(pairs_df: pd.DataFrame, blocks: list, crops: dict, patch: int) -> list:
    """Per block: the chosen pairs (rows of f7_pairs.csv) with their crops, in rank order."""
    out = []
    for b in blocks:
        rows = pairs_df[pairs_df["block"] == b].sort_values("rank").to_dict("records")
        c, v = crops[f"b{b}_crops"], crops[f"b{b}_valid"]
        out.append({"block": b, "patch": patch,
                    "pairs": [{**r, "crops": c[i], "valid": v[i]} for i, r in enumerate(rows)]})
    return out


def draw_figures(args, blocks_data, blocks_df, l0, fig_dir: Path, model_name: str, n_images: int) -> list:
    import matplotlib

    with matplotlib.rc_context():                       # the paper style is restored afterwards
        mf = _style()
        figures = []
        figures += fig_teaser(mf, blocks_data, blocks_df, l0, args, fig_dir, model_name)
        figures += fig_twins_full(mf, blocks_data, args, fig_dir, model_name, n_images)
        figures += fig_layer0(mf, l0, args, fig_dir, model_name)
    return figures


def replot(args) -> dict:
    """Re-draw the figures from the outputs of a finished run in args.replot (for example on the laptop, so the
    fonts match scripts/make_figures.py). Crop geometry, model and image count come from the run; the layout
    options (--teaser-pairs, --teaser-patches, --full-patches, --filters, --full-filters) from this call."""
    run = Path(args.replot)
    meta = json.loads((run / "f7_meta.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    args.context, args.blocks_list = int(cfg["context"]), list(meta["blocks"])
    pairs_df = pd.read_csv(run / "f7_pairs.csv")
    blocks_df = pd.read_csv(run / "f7_blocks.csv")
    l0 = load_layer0(run / "f7_layer0.npz")
    with np.load(run / "f7_crops.npz") as z:
        crops = {k: z[k] for k in z.files}
    blocks_data = blocks_for_figures(pairs_df, meta["blocks"], crops, int(l0["patch"]))
    fig_dir = Path(args.fig_dir) if args.fig_dir else (run / "figures" if meta.get("tiny") else PAPER_FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = draw_figures(args, blocks_data, blocks_df, l0, fig_dir, meta["model"], int(meta["images"]["n"]))
    info = {"experiment": "F7 replot", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_info(),
            "run": str(run), "run_git": meta.get("git"), "config": vars(args),
            "outputs": {str(f): sha256_file(f) for f in figures}}
    (run / "f7_replot_meta.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log(f"F7 replot: {len(figures)} files in {fig_dir}")
    return info


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------
def _f(x, spec=".2f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:{spec}}"


def write_summary(path: Path, pairs_df: pd.DataFrame, blocks_df: pd.DataFrame, l0: dict, info: dict) -> None:
    a = info["args"]
    L = ["# F7: mirror-twin neurons and layer-0 filters (visuals)", ""]
    L.append(f"Generated by `scripts/f7_visuals.py` on {info['date']}, commit {info['git']['git_hash'][:10]}"
             f"{' (uncommitted changes)' if info['git'].get('git_dirty') else ''}. Model `{info['model']}`; fit run "
             f"`{info['fit_npz']}` (fit seed {info['fit_seed']}, SHA-256 {info['fit_sha256'][:12]}); seed {a['seed']}."
             f"{' TINY MODE: random tiny ViT and synthetic images, not a result.' if a['tiny'] else ''}")
    L.append(f"Visualisation set: {info['n_images']:,} {info['image_kind']} from the {info['rows_desc']} "
             f"({info['n_tokens_per_block']:,} patch tokens per block). Every number below is read from the CSV "
             "files in this folder.")
    L.append("")
    L.append("## Result in brief")
    L.append("")
    if len(pairs_df):
        n = len(pairs_df)
        both1 = int(((pairs_df["twin_rank_u"] == 1) & (pairs_df["twin_rank_v"] == 1)).sum())
        L.append(f"- Chosen pairs: {n} ({', '.join(f'block {b}: {c}' for b, c in pairs_df.groupby('block').size().items())}).")
        L.append(f"- Twin correlation (mean of both directions) over the chosen pairs: min {_f(pairs_df['corr_twin'].min())}, "
                 f"median {_f(pairs_df['corr_twin'].median())}, max {_f(pairs_df['corr_twin'].max())}. Random-partner "
                 f"control: min {_f(pairs_df['corr_rand'].min())}, median {_f(pairs_df['corr_rand'].median())}, "
                 f"max {_f(pairs_df['corr_rand'].max())}.")
        L.append(f"- In {both1} of {n} chosen pairs the twin is the best-correlated partner of both neurons among all "
                 f"{int(blocks_df['h'].iloc[0])} neurons of the block.")
        L.append(f"- On each neuron's own top patches, the twin's activation on the mirrored image at the mirrored "
                 f"position is on average {_f(pairs_df['top_ratio_twin'].mean())} of the neuron's activation "
                 f"(random partner {_f(pairs_df['top_ratio_rand'].mean())}).")
    else:
        L.append("- No twin pairs were found in the chosen blocks.")
    f = l0["fitted"]
    L.append(f"- Layer 0: the even subspace of A_0 holds {_f(f['even']['energy_share'])} of the (channel-centred) "
             f"patch-embedding energy and its filters are {_f(f['even']['sym_fraction'])} symmetric by energy; the odd "
             f"subspace holds {_f(f['odd']['energy_share'])} and is {_f(f['odd']['sym_fraction'])} symmetric (all "
             f"filters together: {_f(l0['all_sym_fraction'])}). With a random involution of the same d+ the two "
             f"subspaces are {_f(l0['null']['even']['sym_fraction'])} and {_f(l0['null']['odd']['sym_fraction'])} symmetric.")
    L.append("")
    L.append("## Chosen pairs")
    L.append("")
    L.append("Rank by matching weight w (fit run). r: correlation over all patch tokens of the visualisation set, "
             "post-GELU, mean of both directions. Rank of twin: position of the twin among all neurons as partners "
             "of u, then of v (1 = best). Top ratio: twin's mirror activation over the neuron's activation on the "
             "neuron's own top patches, mean of u and v.")
    L.append("")
    L.append("| block | rank | u | v | w (fit) | S_uv (fit) | S_uu, S_vv (fit) | twin r | random r | self r (u, v) | "
             "rank of twin | top ratio twin | top ratio random |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in pairs_df.itertuples(index=False):
        L.append(f"| {r.block} | {r.rank} | {r.u} | {r.v} | {_f(r.fit_w)} | {_f(r.fit_S_uv)} | {_f(r.fit_S_uu)}, "
                 f"{_f(r.fit_S_vv)} | {_f(r.corr_twin)} | {_f(r.corr_rand)} | {_f(r.corr_self_u)}, {_f(r.corr_self_v)} | "
                 f"{r.twin_rank_u}, {r.twin_rank_v} | {_f(r.top_ratio_twin)} | {_f(r.top_ratio_rand)} |")
    L.append("")
    L.append("## All twin pairs of each block (not only the chosen ones)")
    L.append("")
    L.append("| block | pairs | self-twins | twin r median (q10, q90) | random r median (q90) | paired neurons: twin r > "
             "random r | self r of paired neurons, median | chosen pairs: twin r mean | chosen pairs: percentile "
             "(median) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in blocks_df.itertuples(index=False):
        L.append(f"| {r.block} | {r.n_pairs} | {r.n_self} | {_f(r.pairs_twin_corr_median)} ({_f(r.pairs_twin_corr_q10)}, "
                 f"{_f(r.pairs_twin_corr_q90)}) | {_f(r.pairs_rand_corr_median)} ({_f(r.pairs_rand_corr_q90)}) | "
                 f"{_f(100 * r.frac_paired_neurons_twin_gt_rand, '.1f')}% | {_f(r.paired_neurons_self_corr_median)} | "
                 f"{_f(r.chosen_twin_corr_mean)} | {_f(r.chosen_twin_corr_percentile_median, '.0f')} |")
    L.append("")
    L.append("## Layer 0")
    L.append("")
    L.append(f"- A_0: d+ = {l0['d_plus']}, d- = {l0['d_minus']}. Median over rows of the largest squared entry: "
             f"{_f(l0['a0_row_max_sq_median'], '.3f')} (1 for a permutation), so A_0 is dense in the channel basis and "
             "channels are not paired one to one; the filters are shown along principal directions of the even and "
             "odd subspaces instead.")
    L.append(f"- Breaking share of the patch embedding, ||W - W_eq||^2 / ||W||^2 with W_eq = (W + A_0 W F) / 2: "
             f"{_f(l0['breaking_share'], '.3f')} (random involution with the same d+: {_f(l0['breaking_share_null'], '.3f')}).")
    for parity in ("even", "odd"):
        g = f[parity]
        top = ", ".join(_f(x) for x in g["sym_fractions"][:min(8, len(g["sym_fractions"]))])
        L.append(f"- {parity} subspace (dimension {g['dim']}): energy share {_f(g['energy_share'])}, symmetric-energy "
                 f"fraction {_f(g['sym_fraction'])}; first principal filters: {top}.")
    L.append("")
    L.append("## Definitions")
    L.append("")
    L.append("- Twin pairing and scores: from the fit run (run_e1.py): S = (R + R^T) / 2, R[i, j] = corr(zt_i, z_j) over "
             "all tokens (CLS, registers and patches) of the fit images, z = fc1 pre-activation, zt = the same on the "
             "mirrored image un-flipped. Matching weight w_uv = 2 S_uv - S_uu - S_vv. "
             f"Pairs ranked by {'w' if a['rank_by'] == 'weight' else 'S_uv'} (ties: S_uv, then u).")
    L.append("- Twin r of a pair (u, v): corr(v on the mirrored image at the mirrored position, u on the original) "
             "and the same with u and v swapped, averaged; post-GELU activations, patch tokens only, all patches of "
             "the visualisation set (float64 sums). It is 1 for an exactly equivariant block.")
    L.append(f"- Random r: the same with a random partner rho(u) that is neither u nor v (seed {a['seed']} * 1000 + "
             "block), for each neuron of the pair; one random partner per neuron.")
    L.append("- Top patches: the images with the largest activation of the neuron at any patch, at most one patch "
             f"per image; crops show the {2 * a['context'] + 1} x {2 * a['context'] + 1} patch neighbourhood (grey "
             "outside the image), the outlined square is the patch itself.")
    L.append("- Layer-0 filters: W = patch-embedding kernels as a d x (3 p p) matrix, centred over channels. "
             "Principal filters of a parity subspace: right singular vectors of V^T W (V an orthonormal basis of the "
             "subspace); energy share s_k^2 / ||W||^2; symmetric-energy fraction ||(g + flip g) / 2||^2 / ||g||^2 "
             "(flip = left-right mirror of the kernel); for a whole subspace the same ratio for V^T W. Filters are "
             "drawn as 0.5 + 0.5 g / max|g| in the (normalised) input colour channels.")
    L.append("")
    L.append("## Draft captions")
    L.append("")
    blocks_txt = ", ".join(str(b) for b in a["blocks_list"][:-1]) + (" and " if len(a["blocks_list"]) > 1 else "") \
        + str(a["blocks_list"][-1])
    held = "not used to fit the pairing" if a["rows"] == "bank" else "(the fit images)"
    L.append(f"- fig0_teaser: Mirror twins in {model_label(info['model'])}. (a) In blocks {blocks_txt}, the "
             f"{a['teaser_pairs']} highest-ranked twin pairs of MLP neurons (ranked by the "
             f"{'matching weight' if a['rank_by'] == 'weight' else 'twin score'} of the fit run, not hand-picked): the "
             f"top {a['teaser_patches']} patches of neuron u and of its twin v among {info['n_images']:,} "
             f"{info['image_kind']} {held}. [Check on the figure before keeping this sentence: the twin's patches "
             "look like mirror images of the neuron's.] Numbers: correlation between the twin's activation on the "
             "mirrored image and the neuron's own activation over all patches (in brackets: a random partner). "
             "(b) Patch-embedding filters along the principal directions of the even (A_0 = +1) and odd (A_0 = -1) "
             "subspaces of the layer-0 mirror operator; numbers: symmetric-energy fraction. (c) Over every twin pair "
             "of each block: median (marker) and 10th to 90th percentile (bar) of the twin correlation and of the "
             "random-partner control.")
    L.append(f"- figF7_twins_full: all {len(pairs_df)} chosen pairs, top {a['full_patches']} patches of each neuron.")
    L.append("- figF7_layer0_filters: (a) the first principal filters of each parity; (b) symmetric-energy fraction "
             "of every principal filter against its energy share, for A_0 and for a random involution.")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- The figure is qualitative; the correlations above are the evidence. The pairing was fitted on "
             "pre-activations over all tokens of the fit images; the check uses post-GELU patch tokens of other "
             "train images, so it is a held-out check of the same pairing" + (" (here the images are the fit "
                                                                             "images themselves, so it is not held out)"
                                                                             if a["rows"] == "fit" else "") + ".")
    L.append("- The twin's top patches are searched independently over the original images; they are not the "
             "mirrored crops of the neuron's patches.")
    L.append("- The forward pass runs in bf16 autocast on CUDA; the statistics are float64 sums of float32 "
             "activations." if info["amp"] == "bf16" else "- The forward pass runs in full precision.")
    L.append("- One fit seed (the fit run's); the random-partner control uses one random partner per neuron.")
    L.append("")
    L.append("Files: `f7_pairs.csv`, `f7_blocks.csv`, `f7_all_pairs.csv`, `f7_top_patches.csv`, `f7_layer0_filters.csv`, `f7_layer0.npz`, `f7_crops.npz`, "
             f"`f7_meta.json`; figures in `{info['fig_dir']}`: " + ", ".join(f"`{Path(p).name}`" for p in info["figures"]) + ".")
    text = "\n".join(L) + "\n"
    assert chr(0x2014) not in text and chr(0x2013) not in text      # no em or en dashes in reports
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.replot:
        return replot(args)
    timer = StageTimer()
    set_seed(args.seed)
    device = torch.device("cpu" if args.tiny else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but no GPU is visible (use --tiny for the CPU test mode)")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    amp = args.amp if device.type == "cuda" else "none"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir) if args.fig_dir else (out / "figures" if args.tiny else PAPER_FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    git = git_info()

    with timer.stage("load"):
        adapter, proto, _ = V.build_runtime(args.model, args.tiny, device, args.cache_dir, batch_size=args.batch_size,
                                            stats_batch_size=args.batch_size, num_workers=args.num_workers)
        fit = load_fit(args, adapter, proto, device, out)
    struct = fit["struct"]
    model_name = "tiny" if args.tiny else args.model
    blocks = [int(b) for b in (args.blocks or (TINY_BLOCKS if args.tiny else DEFAULT_BLOCKS)).split(",") if b.strip()]
    if not blocks or any(b < 1 or b > adapter.depth for b in blocks):
        raise SystemExit(f"blocks must lie in 1..{adapter.depth}")
    args.blocks_list = blocks

    # Pairs, random partners and images
    sel, all_pairs, rhos = {}, {}, {}
    for b in blocks:
        ranked = rank_pairs(struct.perms[b - 1], fit["scores"][b - 1], fit["self_corr"][b - 1], args.rank_by)
        all_pairs[b] = ranked
        sel[b] = ranked.head(args.pairs).reset_index(drop=True)
        rhos[b] = random_partners(struct.perms[b - 1], seed=args.seed * 1000 + b)
        if len(sel[b]) < args.pairs:
            log(f"WARNING: block {b} has only {len(sel[b])} twin pairs")
    fit_seed = fit["fit_seed"] if fit["fit_seed"] is not None else 0
    if args.rows == "bank":
        pool, rows_desc = np.asarray(proto.index["bank_rows"]), "k-NN bank rows (disjoint from the fit images)"
    else:
        pool, rows_desc = np.asarray(proto.index[f"fit{fit_seed}_rows"]), f"fit rows of fit seed {fit_seed}"
    n_img = min(int(args.images), len(pool))
    if n_img < args.images:
        log(f"only {len(pool)} rows available: using {n_img} images")
    vis_rows = np.sort(np.random.default_rng(args.seed).choice(pool, size=n_img, replace=False)).astype(np.int64)
    pin = device.type == "cuda"
    loader = make_loader(proto.train, proto.train_labels, vis_rows, args.batch_size, 0 if args.tiny else args.num_workers, pin)
    log(f"F7: {model_name}, blocks {blocks}, {sum(len(s) for s in sel.values())} pairs, {n_img} images, amp {amp}")

    trackers = {b - 1: BlockTracker(b, struct.perms[b - 1], rhos[b], tracked_neurons(sel[b]), args.topk, device)
                for b in blocks}
    with timer.stage("paired pass"):
        n_seen = collect(adapter, loader, device, trackers, amp=amp)
    if n_seen != n_img:
        raise RuntimeError(f"saw {n_seen} images, expected {n_img}")

    grid_w, patch = adapter.grid_hw[1], adapter.patch_size
    pair_rows, block_rows, top_rows, every_rows, crop_arrays = [], [], [], [], {}
    with timer.stage("analysis"):
        for b in blocks:
            tr = trackers[b - 1]
            rows, brow, every = analyse_block(tr, sel[b], struct.perms[b - 1], rhos[b], all_pairs[b])
            every_rows.append(every)
            crops, valid, table = block_crops(tr, sel[b], proto.train, vis_rows, grid_w, patch, args.context)
            pair_rows += rows
            block_rows.append({**brow, "n_images": n_img})
            top_rows += table
            crop_arrays[f"b{b}_crops"], crop_arrays[f"b{b}_valid"] = crops, valid
            crop_arrays[f"b{b}_pairs"] = sel[b][["u", "v"]].to_numpy()
        W_pe = adapter.model.patch_embed.proj.weight.detach().cpu().double().numpy()
        l0 = layer0_analysis(W_pe, struct.A_layers[0], null_seed=args.seed + 7)

    pairs_df = pd.DataFrame(pair_rows)
    blocks_df = pd.DataFrame(block_rows)
    for df in (pairs_df, blocks_df):
        df.insert(0, "model", model_name)
        df["fit_seed"] = fit["fit_seed"]
        df["seed"] = args.seed
        df["git_hash"] = git["git_hash"]
    pairs_df.to_csv(out / "f7_pairs.csv", index=False)
    blocks_df.to_csv(out / "f7_blocks.csv", index=False)
    pd.DataFrame(top_rows).to_csv(out / "f7_top_patches.csv", index=False)
    every_df = pd.concat(every_rows, ignore_index=True) if every_rows else pd.DataFrame()
    every_df["git_hash"] = git["git_hash"]
    every_df.to_csv(out / "f7_all_pairs.csv", index=False)
    l0_df = layer0_table(l0)
    l0_df["git_hash"] = git["git_hash"]
    l0_df.to_csv(out / "f7_layer0_filters.csv", index=False)
    np.savez_compressed(out / "f7_crops.npz", vis_rows=vis_rows, blocks=np.asarray(blocks), **crop_arrays)
    save_layer0(out / "f7_layer0.npz", l0, n_filters=max(64, args.filters, args.full_filters))

    with timer.stage("figures"):
        figures = draw_figures(args, blocks_for_figures(pairs_df, blocks, crop_arrays, patch), blocks_df, l0,
                               fig_dir, model_name, n_img)

    info = {"date": time.strftime("%Y-%m-%d %H:%M"), "git": git, "model": model_name, "fit_npz": str(fit["npz"]),
            "fit_seed": fit["fit_seed"], "fit_sha256": fit["sha256"], "n_images": n_img, "rows_desc": rows_desc,
            "n_tokens_per_block": int(blocks_df["n_tokens"].iloc[0]) if len(blocks_df) else 0, "amp": amp,
            "image_kind": "synthetic images" if args.tiny else "ImageNet train images",
            "args": {**vars(args)}, "fig_dir": str(fig_dir), "figures": [str(p) for p in figures]}
    write_summary(out / "F7_SUMMARY.md", pairs_df, blocks_df, l0, info)

    outputs = [out / n for n in ("f7_pairs.csv", "f7_blocks.csv", "f7_all_pairs.csv", "f7_top_patches.csv",
                                 "f7_layer0_filters.csv", "f7_layer0.npz", "f7_crops.npz",
                                 "F7_SUMMARY.md")] + list(figures)
    meta = {
        "experiment": "F7", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "seed": args.seed,
        "config": vars(args), "env": env_info(device), "model": model_name, "tiny": bool(args.tiny), "amp": amp,
        "fit_run": {"npz": str(fit["npz"]), "sha256": fit["sha256"], "model": fit["model"], "fit_seed": fit["fit_seed"],
                    "model_mismatch": fit["mismatch"], "load_checks": struct.meta.get("checks"),
                    "run_git": struct.meta.get("run_git"),
                    "max_abs_change_from_cleaning": struct.meta.get("max_abs_change_from_cleaning")},
        "images": {"source": args.rows, "description": rows_desc, "n": n_img, "pool": int(len(pool)),
                   "seed": args.seed, "rows_sha256_of_int64": hashlib.sha256(vis_rows.tobytes()).hexdigest()},
        "blocks": blocks, "block_numbering": "1-based as in e1_mlp.csv; block b = model.blocks[b - 1]",
        "random_partner_seed": "seed * 1000 + block", "layer0_null_seed": args.seed + 7,
        "definitions": {
            "rank": "w_uv = 2 S_uv - S_uu - S_vv from the fit run (match_scores, self_corr)" if args.rank_by == "weight"
            else "S_uv from the fit run (match_scores)",
            "twin_r": "corr(at_v, a_u) and corr(at_u, a_v) averaged; post-GELU, patch tokens, float64 sums",
            "top_patches": "top-k images by the neuron's largest patch activation, one patch per image",
            "layer0": "right singular vectors of V^T W_c per parity subspace of A_0; symmetric-energy fraction"},
        "layer0": {k: v for k, v in l0.items() if k not in ("fitted", "null")} | {
            f"{op}_{par}_{q}": l0[op][par][q] for op in ("fitted", "null") for par in ("even", "odd")
            for q in ("dim", "energy_share", "sym_fraction")},
        "timing": {"stages": timer.stages, "total_s": timer.total()},
        "outputs": {str(p): sha256_file(p) for p in outputs if Path(p).exists()},
    }
    (out / "f7_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log(f"F7 wrote {out} and figures in {fig_dir} ({timer.total():.0f} s)")
    return meta


if __name__ == "__main__":
    main()
