#!/usr/bin/env python
"""F3, artefact tokens and pre-registered hypothesis P1 (PAPER_EXPERIMENTS.md Track F; claim C3; gate G3).

One paired pass (original + mirror, as stats.run_paired) over the held-out validation images
(Protocol.heldout_loader: the first validation image of each class, 1,000 images) with per-layer
operators from an X2 run (cleaned to exact involutions). For every layer and token it records the
raw token norms of the original and of the un-flipped mirror, and the error ||x~ - x A_l||^2 and
||x~||^2 after channel centring (the E1 definition). Per layer it then reports:
  - token norms (CLS, register and patch tokens separately);
  - high-norm tokens: the top 1% by norm within the layer (original and mirror tokens pooled, all
    token types), and separately tokens above 3x the layer's median norm;
  - e(A_l) = sum ||x~ - x A||^2 / sum ||x~||^2 over all tokens, and over the token positions that
    are high-norm in neither the original nor the mirror (also per token type);
  - among patch positions: the fraction of high-norm positions of the (un-flipped) mirror that are
    also high-norm in the original (artefacts follow the content), the same at the raw grid
    position (artefacts stay at fixed grid positions), and the chance level.
P1 (PREREG_E2.md, thresholds in mirror/findings.py) is evaluated for the model's own bullet:
  plain:     mean e over layers 9 to 11 (all tokens) / (without the top 1%) >= 2;
  registers: mean e over layers 9 to 11 / mean e over layers 1 to 6 (all tokens) <= 2.
When both models' JSON files are in --out, gate G3 is written as well.

    python scripts/f3_artefacts.py --model dinov2_vitb14 --operators runs/x2/perlayer_plain_s0 --out runs/f3
    python scripts/f3_artefacts.py --model dinov2_vitb14_reg --operators runs/x2/perlayer_reg_s0 --out runs/f3
    python scripts/f3_artefacts.py --tiny [--tiny-registers 2] --out tiny_out/f3   # CPU test mode

Writes <out>/f3_<model>.json, f3_<model>_layers.csv, f3_<model>_tokens.npz, f3_gate_G3.json (when
both models are done) and plots/fig3b_artefacts.png.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.08)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.adapters import TimmTinyAdapter, ViTAdapter  # noqa: E402
from mirror.findings import (P1_EARLY_LAYERS, P1_LATE_LAYERS, P1_TOP_FRAC, clean_involution,  # noqa: E402
                             evaluate_p1, gate_g3, model_colours, model_label, sha256_file, sort_models)
from mirror.stats import center_channels, run_paired  # noqa: E402
from mirror.utils import REPO_ROOT, env_info, git_info, log, set_seed  # noqa: E402

MEDIAN_MULT = 3.0  # "tokens above 3x the median norm" (PAPER_EXPERIMENTS.md F3)
DEFAULT_OPERATORS = {"dinov2_vitb14": "runs/x2/perlayer_plain_s0", "dinov2_vitb14_reg": "runs/x2/perlayer_reg_s0"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14", choices=sorted(DEFAULT_OPERATORS))
    p.add_argument("--operators", default=None,
                   help="X2 run folder or e1_operators.npz (default: the fit-seed-0 run of the model's per-layer arm)")
    p.add_argument("--out", default="runs/f3")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16, help="images per batch (the pass runs 2x this)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-save-tokens", action="store_true", help="skip the per-token npz")
    p.add_argument("--allow-model-mismatch", action="store_true",
                   help="allow operators fitted on another model (diagnostics only; recorded in the JSON)")
    p.add_argument("--allow-uncommitted-prereg", action="store_true",
                   help="dry runs only: run although PREREG_E2.md is not committed (recorded in the JSON)")
    p.add_argument("--tiny", action="store_true", help="tiny random timm ViT and synthetic images on the CPU")
    p.add_argument("--tiny-registers", type=int, default=0, help="register tokens of the tiny model")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Per-token pass
# ----------------------------------------------------------------------------------------------
class TokenRecorder:
    """Per layer and token: raw norms of x and of the un-flipped mirror x~, and after channel centring
    (as in E1) the error ||x~ - x A_l||^2 and ||x~||^2. Kept on the CPU in float64."""

    KEYS = ("norm_x", "norm_xt", "err", "sq_xt")

    def __init__(self, ops: list[torch.Tensor], center: bool = True):
        self.ops, self.center = ops, center
        self.buf = {k: [[] for _ in ops] for k in self.KEYS}

    @torch.no_grad()
    def update(self, layer: int, x: torch.Tensor, xt: torch.Tensor) -> None:
        X, Xt = x.to(torch.float64), xt.to(torch.float64)
        self.buf["norm_x"][layer].append(X.norm(dim=-1).cpu().numpy())
        self.buf["norm_xt"][layer].append(Xt.norm(dim=-1).cpu().numpy())
        if self.center:
            X, Xt = center_channels(X), center_channels(Xt)
        diff = Xt - X @ self.ops[layer]
        self.buf["err"][layer].append((diff * diff).sum(-1).cpu().numpy())
        self.buf["sq_xt"][layer].append((Xt * Xt).sum(-1).cpu().numpy())

    def arrays(self) -> dict:
        """Each key: [n_layers, N images, T tokens]."""
        return {k: np.stack([np.concatenate(v, axis=0) for v in self.buf[k]]) for k in self.KEYS}


def collect_tokens(adapter, loader, device, ops: list[np.ndarray], center: bool = True) -> dict:
    ops_t = [torch.as_tensor(A, dtype=torch.float64, device=device) for A in ops]
    rec = TokenRecorder(ops_t, center=center)
    run_paired(adapter, loader, device, residual_cb=rec.update, desc="F3 paired pass")
    return rec.arrays()


# ----------------------------------------------------------------------------------------------
# Per-layer analysis (pure numpy)
# ----------------------------------------------------------------------------------------------
def top_threshold(values: np.ndarray, frac: float) -> float:
    """Norm of the n_top-th largest value, n_top = round(frac * n) (at least 1); `>=` selects the top."""
    v = np.asarray(values, dtype=np.float64).ravel()
    n_top = max(1, int(round(frac * v.size)))
    return float(np.partition(v, v.size - n_top)[v.size - n_top])


def _rel(err: np.ndarray, sq: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        err, sq = err[mask], sq[mask]
    s = float(sq.sum())
    return float(err.sum()) / s if s > 0 else float("nan")


def coincidence(hi_x: np.ndarray, hi_xt: np.ndarray, grid_hw: tuple[int, int]) -> dict:
    """hi_x, hi_xt: [N, H*W] high-norm flags of the original's and the un-flipped mirror's patch tokens.

    coincide_mirrored: share of the mirror's high positions (un-flipped, so position j holds the same
    content as the original's position j) that are also high in the original;
    coincide_raw: the same at the mirror image's own grid position (flipped back);
    coincide_chance: expected share if each image's high positions were placed at random.
    """
    H, W = grid_hw
    N = hi_x.shape[0]
    ho, hm = hi_x.reshape(N, H, W), hi_xt.reshape(N, H, W)
    n_m, n_o = int(hm.sum()), int(ho.sum())
    out = {"n_high_orig": n_o, "n_high_mirror": n_m}
    if n_m == 0:
        return {**out, "coincide_mirrored": float("nan"), "coincide_raw": float("nan"), "coincide_chance": float("nan")}
    chance = float((ho.sum((1, 2)) * hm.sum((1, 2)) / (H * W)).sum())
    return {**out, "coincide_mirrored": float((ho & hm).sum()) / n_m,
            "coincide_raw": float((ho & hm[:, :, ::-1]).sum()) / n_m, "coincide_chance": chance / n_m}


def analyse_layer(nx, nxt, err, sq, n_prefix: int, grid_hw, top_frac: float = P1_TOP_FRAC,
                  med_mult: float = MEDIAN_MULT) -> dict:
    """nx, nxt, err, sq: [N, T] for one layer. Token order: CLS, registers, patches (row-major)."""
    N, T = nx.shape
    groups = {"cls": np.arange(0, min(1, n_prefix)), "registers": np.arange(1, n_prefix),
              "patches": np.arange(n_prefix, T)}
    pooled = np.concatenate([nx.ravel(), nxt.ravel()])
    med = float(np.median(pooled))
    out = {"n_images": N, "tokens_per_image": T, "n_registers": max(0, n_prefix - 1),
           "norm_median_all": med, "thr_top1": top_threshold(pooled, top_frac), "thr_3x_median": med_mult * med,
           "e_all": _rel(err, sq)}
    for tag, thr, strict in (("top1", out["thr_top1"], False), ("3xmed", out["thr_3x_median"], True)):
        hx = nx > thr if strict else nx >= thr
        hxt = nxt > thr if strict else nxt >= thr
        excl = hx | hxt
        n_high = int(hx.sum() + hxt.sum())
        out[f"e_excl_{tag}"] = _rel(err, sq, ~excl)
        out[f"frac_high_{tag}"] = n_high / (2 * N * T)
        out[f"frac_positions_excluded_{tag}"] = float(excl.mean())
        for g, idx in groups.items():
            k = int(hx[:, idx].sum() + hxt[:, idx].sum())
            out[f"frac_high_{tag}_{g}"] = k / (2 * N * len(idx)) if len(idx) else float("nan")
            out[f"share_of_high_{tag}_{g}"] = k / n_high if n_high else float("nan")
        c = coincidence(hx[:, n_prefix:], hxt[:, n_prefix:], grid_hw)
        out.update({f"{k}_{tag}": v for k, v in c.items()})
    # Secondary reading of "top 1%": among patch tokens only (CLS and registers are never excluded).
    pn = np.concatenate([nx[:, n_prefix:].ravel(), nxt[:, n_prefix:].ravel()])
    out["thr_top1_patches"] = top_threshold(pn, top_frac)
    excl_p = np.zeros_like(nx, dtype=bool)
    excl_p[:, n_prefix:] = (nx[:, n_prefix:] >= out["thr_top1_patches"]) | (nxt[:, n_prefix:] >= out["thr_top1_patches"])
    out["e_excl_top1_patchthr"] = _rel(err, sq, ~excl_p)
    for g, idx in groups.items():
        if len(idx) == 0:
            for k in ("e", "norm_mean", "norm_median", "norm_p99", "norm_max"):
                out[f"{k}_{g}"] = float("nan")
            continue
        out[f"e_{g}"] = _rel(err[:, idx], sq[:, idx])
        gn = np.concatenate([nx[:, idx].ravel(), nxt[:, idx].ravel()])
        out.update({f"norm_mean_{g}": float(gn.mean()), f"norm_median_{g}": float(np.median(gn)),
                    f"norm_p99_{g}": float(np.quantile(gn, 0.99)), f"norm_max_{g}": float(gn.max())})
    keep = np.ones(T, dtype=bool)
    keep[groups["registers"]] = False
    out["e_excl_registers"] = _rel(err[:, keep], sq[:, keep])
    return out


def layer_table(arrays: dict, n_prefix: int, grid_hw) -> pd.DataFrame:
    n = arrays["err"].shape[0]
    return pd.DataFrame([{"layer": l, **analyse_layer(arrays["norm_x"][l], arrays["norm_xt"][l], arrays["err"][l],
                                                      arrays["sq_xt"][l], n_prefix, grid_hw)} for l in range(n)])


# ----------------------------------------------------------------------------------------------
# Models, data and operators
# ----------------------------------------------------------------------------------------------
class TinyRegAdapter(TimmTinyAdapter):
    """TimmTinyAdapter with register tokens (timm reg_tokens; order CLS, registers, patches)."""

    def __init__(self, img_size: int = 56, seed: int = 0, reg_tokens: int = 2):
        from timm.models.vision_transformer import VisionTransformer

        from mirror.adapters import _randomise_affine

        torch.manual_seed(seed)
        model = VisionTransformer(img_size=img_size, patch_size=14, embed_dim=64, depth=4, num_heads=4,
                                  init_values=1e-5, reg_tokens=reg_tokens)
        _randomise_affine(model, seed)
        ViTAdapter.__init__(self, model, img_size=img_size, patch_size=14,
                            n_prefix_tokens=int(model.num_prefix_tokens))


def build_tiny(args):
    from mirror.data import make_loader, make_splits, synthetic_dataset

    adapter = (TinyRegAdapter(seed=args.seed, reg_tokens=args.tiny_registers) if args.tiny_registers > 0
               else TimmTinyAdapter(img_size=56, seed=args.seed))
    images, labels = synthetic_dataset(num_classes=5, per_class_total=8, img_size=56, seed=args.seed)
    splits = make_splits(labels, {"fit": 4, "heldout": 4})
    loaders = {s: make_loader(images, labels, splits[s], 8, 0, False) for s in ("fit", "heldout")}
    name = f"tiny_reg{args.tiny_registers}" if args.tiny_registers > 0 else "tiny"
    return adapter, loaders, name


def fit_tiny_operators(adapter, loader, device, path: Path) -> Path:
    """Tiny mode only: per-layer operators fitted on synthetic fit images, stored as float32 like X2."""
    from mirror.fit import fit_involution
    from mirror.stats import collect_fit_stats

    C = collect_fit_stats(adapter, loader, device, center=True, mlp=False).residual.numpy()["C"]
    A = np.stack([fit_involution(C[l])[0] for l in range(C.shape[0])]).astype(np.float32)
    np.savez_compressed(path, A_layers=A)
    return path


def resolve_npz(path: str | Path) -> Path:
    p = Path(path)
    for cand in (p, p / "results" / "e1_operators.npz", p / "e1_operators.npz"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no operators npz at {path}")


def load_operators(path, n_layers: int, d: int) -> tuple[list[np.ndarray], dict]:
    npz = resolve_npz(path)
    with np.load(npz) as z:
        A_raw = z["A_layers"]
    if A_raw.shape != (n_layers, d, d):
        raise SystemExit(f"operators {A_raw.shape} do not fit the model ({n_layers}, {d}, {d})")
    meta_p = npz.parent / "e1_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    cleaned = [clean_involution(A_raw[l], return_info=True) for l in range(n_layers)]
    info = {"path": str(npz), "sha256": sha256_file(npz),
            "source_model": ((meta.get("config") or {}).get("model") or {}).get("name"),
            "source_fit_seed": meta.get("fit_seed"), "source_git_hash": (meta.get("git") or {}).get("git_hash"),
            "source_operator_mode": meta.get("operator_mode"), "source_protocol": meta.get("protocol"),
            "center": bool(((meta.get("config") or {}).get("stats") or {}).get("center", True)),
            "cleaning_max_eig_dev": max(c[3]["max_eig_dev"] for c in cleaned),
            "cleaning_max_rel_change": max(c[3]["rel_change"] for c in cleaned),
            "d_plus": [c[3]["d_plus"] for c in cleaned]}
    return [c[0] for c in cleaned], info


def prereg_record() -> dict:
    """Identity of PREREG_E2.md: sha256 always; the last commit touching it and whether the working
    copy is clean when git is available (the VM may have no git binary)."""
    path = REPO_ROOT / "PREREG_E2.md"
    rec = {"file": "PREREG_E2.md", "sha256": sha256_file(path) if path.exists() else None}
    try:
        def git(*a):
            r = subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                raise OSError(r.stderr)
            return r.stdout.strip()

        commit = git("log", "-1", "--format=%H", "--", "PREREG_E2.md") or None
        status = git("status", "--porcelain", "--", "PREREG_E2.md")
        rec.update({"git_available": True, "commit": commit, "committed_and_clean": bool(commit) and status == ""})
    except (OSError, subprocess.SubprocessError):
        rec.update({"git_available": False, "commit": None, "committed_and_clean": None,
                    "note": "git not available here: compare sha256 with the committed file "
                            "(git show <commit>:PREREG_E2.md | sha256sum)"})
    return rec


# ----------------------------------------------------------------------------------------------
# Gate and figure
# ----------------------------------------------------------------------------------------------
def maybe_write_gate(out_dir: Path, tiny: bool, git: dict) -> dict | None:
    recs = []
    for f in sorted(out_dir.glob("f3_*.json")):
        if f.name.startswith("f3_gate"):
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        if "p1" in j and bool(j.get("tiny")) == tiny:
            recs.append((f, j))
    plain = sorted([r for r in recs if r[1]["p1"]["bullet"] == "plain"], key=lambda r: r[1]["date"])
    reg = sorted([r for r in recs if r[1]["p1"]["bullet"] == "registers"], key=lambda r: r[1]["date"])
    if not plain or not reg:
        return None
    if len(plain) > 1 or len(reg) > 1:
        log("WARNING: several F3 results per bullet in this folder; the gate uses the latest of each")
    (fp, jp), (fr, jr) = plain[-1], reg[-1]
    g = gate_g3(jp["p1"]["pass"], jr["p1"]["pass"])
    g.update({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "tiny": tiny,
              "plain_json": fp.name, "registers_json": fr.name,
              "plain_value": jp["p1"]["value"], "registers_value": jr["p1"]["value"],
              "note": "Stop and report at gate G3 (PAPER_EXPERIMENTS.md 0.3)."})
    (out_dir / "f3_gate_G3.json").write_text(json.dumps(g, indent=2, default=str), encoding="utf-8")
    return g


def plot_fig3b(out_dir: Path) -> Path | None:
    files = sorted(out_dir.glob("f3_*_layers.csv"))
    if not files:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mirror.report import C_GRID, C_INK, C_NULL, _style

    from matplotlib.ticker import FuncFormatter

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    models = sort_models(df["model"])
    colours = model_colours(models)
    bg = "#fcfcfb"
    shade = df["layer"].max() >= 11
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.1), facecolor=bg)
    for ax in axes:
        ax.set_facecolor(bg)
        if shade:
            ax.axvspan(8.5, 11.5, color=C_GRID, alpha=0.6, linewidth=0)

    def plain_log(ax):
        from matplotlib.ticker import LogLocator

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))

    ax = axes[0]
    for m in models:
        d, c, s = df[df["model"] == m].sort_values("layer"), colours[m], model_label(m, short=True)
        ax.plot(d["layer"], d["e_all"], "-", color=c, marker="o", markersize=5, linewidth=2, label=f"{s}: all tokens")
        ax.plot(d["layer"], d["e_excl_top1"], "--", color=c, marker="s", markersize=5, linewidth=2,
                label=f"{s}: without top 1% norm")
        ax.plot(d["layer"], d["e_excl_3xmed"], ":", color=c, marker="^", markersize=5, linewidth=2,
                label=f"{s}: without norm > 3x median")
    plain_log(ax)
    ax.set_xlabel("layer (shaded: P1 layers 9 to 11)" if shade else "layer")
    ax.set_ylabel("held-out e(A_l) = ||X~ - XA||² / ||X~||²")
    ax.set_title("Mirror error with and without high-norm tokens", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)

    ax = axes[1]
    for m in models:
        d, c, s = df[df["model"] == m].sort_values("layer"), colours[m], model_label(m, short=True)
        ax.plot(d["layer"], d["norm_median_patches"], "-", color=c, marker="o", markersize=5, linewidth=2,
                label=f"{s}: patch tokens, median")
        ax.plot(d["layer"], d["norm_p99_patches"], "--", color=c, marker="s", markersize=5, linewidth=2,
                label=f"{s}: patch tokens, 99th percentile")
        if d["norm_mean_registers"].notna().any():
            ax.plot(d["layer"], d["norm_mean_registers"], ":", color=c, marker="D", markersize=5, linewidth=2,
                    label=f"{s}: register tokens, mean")
    plain_log(ax)
    ax.set_xlabel("layer")
    ax.set_ylabel("token norm")
    ax.set_title("Token norms", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)

    ax = axes[2]
    for m in models:
        d, c, s = df[df["model"] == m].sort_values("layer"), colours[m], model_label(m, short=True)
        ax.plot(d["layer"], d["coincide_mirrored_top1"], "-", color=c, marker="o", markersize=5, linewidth=2,
                label=f"{s}: at the mirrored position")
        ax.plot(d["layer"], d["coincide_raw_top1"], "--", color=c, marker="s", markersize=5, linewidth=2,
                label=f"{s}: at the same grid position")
        ax.plot(d["layer"], d["coincide_chance_top1"], ":", color=C_NULL, marker="o" if m == models[0] else "s",
                markersize=4, linewidth=1.5, label=f"{s}: chance")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("share of the mirror's top-1% patch tokens")
    ax.set_title("Do artefacts follow the mirrored content?", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    fig.tight_layout()
    path = out_dir / "plots" / "fig3b_artefacts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    args = parse_args(argv)
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    git = git_info()
    t0 = time.time()
    prereg = prereg_record()

    if args.tiny:
        device = torch.device("cpu")
        adapter, loaders, model_name = build_tiny(args)
        loader = loaders["heldout"]
        ops_src = args.operators or fit_tiny_operators(adapter, loaders["fit"], device,
                                                       out / f"f3_{model_name}_operators.npz")
        n_l = adapter.depth + 1
        late, early = (n_l - 2,), tuple(range(1, max(2, (n_l - 1) // 2 + 1)))  # stand-ins for 9-11 and 1-6
        data_note = "synthetic images (tiny test mode)"
    else:
        if prereg.get("committed_and_clean") is False and not args.allow_uncommitted_prereg:
            raise SystemExit("PREREG_E2.md is not committed or has local changes. Commit it before running F3 "
                             "(PAPER_EXPERIMENTS.md X5); --allow-uncommitted-prereg is for dry runs only.")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("no GPU visible; F3 runs on the VM (use --tiny for a CPU check)")
        from mirror.adapters import Dinov2Adapter
        from mirror.protocol import Protocol

        model_name = args.model
        adapter = Dinov2Adapter(hub_repo=args.hub_repo, name=args.model).to(device=device)
        proto = Protocol(args.cache_dir, stats_batch_size=args.batch_size, num_workers=args.num_workers,
                         pin=device.type == "cuda")
        loader = proto.heldout_loader()
        ops_src = args.operators or DEFAULT_OPERATORS[args.model]
        late, early = P1_LATE_LAYERS, P1_EARLY_LAYERS
        data_note = "ImageNet validation, first image of each class (Protocol.heldout_loader)"

    n_layers, d = adapter.depth + 1, adapter.embed_dim
    ops, ops_info = load_operators(ops_src, n_layers, d)
    if ops_info["source_model"] and ops_info["source_model"] != model_name and not args.tiny:
        if not args.allow_model_mismatch:
            raise SystemExit(f"operators were fitted on {ops_info['source_model']}, not {model_name}")
        log(f"WARNING: operators from {ops_info['source_model']} used on {model_name}")
    n_prefix, n_reg = adapter.n_prefix_tokens, adapter.n_prefix_tokens - 1
    log(f"F3 {model_name}: {n_layers} layers, d {d}, prefix tokens {n_prefix} ({n_reg} registers), grid "
        f"{adapter.grid_hw}, operators {ops_info['path']} (fit seed {ops_info['source_fit_seed']}), device {device}")

    arrays = collect_tokens(adapter, loader, device, ops, center=ops_info["center"])
    df = layer_table(arrays, n_prefix, adapter.grid_hw)
    df.insert(0, "model", model_name)
    df["git_hash"], df["seed"] = git["git_hash"], args.seed
    df.to_csv(out / f"f3_{model_name}_layers.csv", index=False)

    e_all = dict(zip(df["layer"].astype(int), df["e_all"]))
    e_excl = dict(zip(df["layer"].astype(int), df["e_excl_top1"]))
    e_excl_p = dict(zip(df["layer"].astype(int), df["e_excl_top1_patchthr"]))
    bullet = "registers" if n_reg > 0 else "plain"
    p1 = evaluate_p1(e_all, e_excl, bullet, late, early)
    other = evaluate_p1(e_all, e_excl, "plain" if bullet == "registers" else "registers", late, early)
    patch_reading = evaluate_p1(e_all, e_excl_p, "plain", late, early)

    result = {
        "experiment": "F3", "model": model_name, "tiny": bool(args.tiny), "seed": args.seed,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "env": env_info(device), "args": vars(args),
        "prereg": prereg,
        "prereg_override": bool(args.allow_uncommitted_prereg and prereg.get("committed_and_clean") is False),
        "operators": ops_info,
        "data": {"source": data_note, "images": int(arrays["err"].shape[1]), "tokens_per_image": int(arrays["err"].shape[2]),
                 "n_prefix_tokens": n_prefix, "n_registers": n_reg, "grid_hw": list(adapter.grid_hw)},
        "definitions": {
            "e": "sum over tokens of ||x~ - x A_l||^2 / sum ||x~||^2, after channel centring (E1 definition); "
                 "x~ = mirrored image's token at the mirrored position",
            "token_norm": "L2 norm of the raw residual-stream token (before centring)",
            "top1": f"norm >= the round({P1_TOP_FRAC} n)-th largest norm of the layer, original and mirror tokens "
                    "pooled, all token types (CLS, registers, patches)",
            "3xmed": f"norm > {MEDIAN_MULT:g} x the median norm of the layer (same pool)",
            "excluded": "token positions that are high-norm in the original or in the (un-flipped) mirror",
            "layers": "layer l = output of block l; layer 0 = tokens entering block 1",
            "p1_late_layers": list(late), "p1_early_layers": list(early),
        },
        "p1": p1,
        "p1_other_bullet_info": {**other, "note": "not pre-registered for this model; information only"},
        "p1_patch_only_reading_info": {**patch_reading, "note": "top 1% among patch tokens only; information only"},
        "per_layer": {"e_all": [float(v) for v in df["e_all"]], "e_excl_top1": [float(v) for v in df["e_excl_top1"]],
                      "e_excl_3xmed": [float(v) for v in df["e_excl_3xmed"]],
                      "coincide_mirrored_top1": [float(v) for v in df["coincide_mirrored_top1"]]},
        "timing_s": round(time.time() - t0, 1),
    }
    (out / f"f3_{model_name}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if not args.no_save_tokens:
        np.savez_compressed(out / f"f3_{model_name}_tokens.npz",
                            **{k: v.astype(np.float32) for k, v in arrays.items()},
                            n_prefix=n_prefix, grid_hw=np.array(adapter.grid_hw))
    gate = maybe_write_gate(out, bool(args.tiny), git)
    plot_fig3b(out)
    log(f"P1 ({bullet}): {p1['result']} (value {p1['value']:.3f} {p1['comparison']} {p1['threshold']:g})"
        + (f"; gate G3: {gate['status']}" if gate else ""))
    return result


if __name__ == "__main__":
    main()
