#!/usr/bin/env python
"""A15b cause probe, measurement: 90-degree rotation orbit structure before and after the rot90 fine-tune
(FINAL_PHASE_PLAN.md v2, Section 4 A15b; pre-registered rule in Section 6).

    python scripts/fp_a15b_measure.py --model dinov2_vitb14_reg \
        --finetuned runs/final/A15b/dinov2_vitb14_reg/rot90_probe/s0 --out results/A15b
    python scripts/fp_a15b_measure.py --dry-run --out tiny_out/A15b                  # CPU, tiny model

Both the original model and the fine-tuned copy are measured with the MUST-04 R90 code path itself: this script
imports scripts/measure_groups.py and calls its run_generator_pass, Ctx, analyse_group, run_twin_null and
build_metrics for group R90, with measure_groups' own argument parser (so every setting not listed here keeps its
MUST-04 default): fit set fit<--fit-seed> (2,000 ImageNet train images, mirror.protocol.Protocol), blocks 1 to 6,
20 null draws with base seed 0, top_k 10, attention statistics on, held-out errors off (reporting only in
MUST-04; they do not enter these numbers). Measured per model:
  * R90 ratio to null in blocks 1 to 6: mean breaking share (s_fc1 + s_fc2) / 2 of the folded MLP weights under
    the fitted order-4 operators and the Hungarian matching (cycles of 1, 2 and 4 kept), divided by the mean of
    the null (random order-4 operators with the fitted multiplicities, random permutations with the fitted cycle
    type);
  * share of hidden units in 4-cycles in blocks 1 to 6 (and the twin-fraction null if --twin-null-draws > 0).

--finetuned: a state-dict file of the same architecture (written by the healing agent's rot90 probe; plain or
wrapped in {"model": ...} / {"state_dict": ...}, torch.compile and DataParallel prefixes removed), or its run
directory (ckpt/step<S>.pt, the last step unless --step; if that state dict does not load,
mirror.final.heal.students.load_student is tried). A folded student (RMSNorm without affine) is loaded into a
folded copy and unfolded back to LayerNorm (mirror.controls.unfold_rmsnorm), which is exact and is what the MUST-04
path needs (its input moments read nn.LayerNorm).

Pre-registered A15b: the R90 ratio to null in blocks 1 to 6 falls by at least 0.1 after the fine-tune; reported
either way (PASS, FAIL, or PENDING when a number is missing). FINAL_PHASE_PLAN.md quotes the original as "about
0.82"; MUST-04 recorded 0.849 for dinov2_vitb14_reg fit 0 (results/MUST-04/dinov2_vitb14_reg/metrics.json); this
run measures the original again with the same code and reports the difference to MUST-04.

Outputs in --out: original/ and finetuned/ (measure_groups' own files: layers.csv, blocks.csv, null_draws.csv,
twin_null.csv, cache/, metrics.json), a15b.csv, summary.md, metrics.json, meta.json. Resumable through
measure_groups' caches; a changed checkpoint (sha256) clears the stale tables of finetuned/.

--dry-run: CPU, the tiny model and synthetic data of measure_groups.py --tiny (TimmTinyAdapter, TinyProtocol;
tested to give the same numbers), 3 null draws, and a stand-in "fine-tune" (seeded noise on every weight, written
as a wrapped state dict and loaded back through load_finetuned).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "A15b"
A15B_MIN_DROP = 0.1              # pre-registered (FINAL_PHASE_PLAN.md Section 6)
GPU_FRAC_DEFAULT = 0.15          # of an H200; MUST-04 measured DINOv2-B within 0.2 of an 80 GB H100
GROUP = "R90"
MEASURED_TABLES = ("layers.csv", "blocks.csv", "null_draws.csv", "twin_null.csv", "metrics.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--finetuned", default=None, help="state-dict file of the fine-tuned copy, or its run directory")
    p.add_argument("--step", type=int, default=None, help="checkpoint step in a run directory (default: the last)")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "A15b"))
    p.add_argument("--fit-seed", type=int, default=0)
    p.add_argument("--blocks", default="1-6", help="blocks measured (1-based)")
    p.add_argument("--range", default="1-6", help="block range of the A15b rule (1-based)")
    p.add_argument("--null-draws", type=int, default=None, help="null draws (default 20; dry run 3)")
    p.add_argument("--twin-null-draws", type=int, default=0, help="R90 twin-fraction null draws (0: off)")
    p.add_argument("--seed", type=int, default=0, help="base seed of the null draws (MUST-04 used 0)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--workers", default="auto", help="processes for the Hungarian matchings")
    p.add_argument("--no-attn", action="store_true", help="skip attention statistics (not used by the rule)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--stats-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--must04", default=None,
                   help="MUST-04 metrics.json of the original "
                        "(default results/MUST-04/<model>/metrics.json if present)")
    p.add_argument("--gpu-frac", type=float, default=GPU_FRAC_DEFAULT)
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny model, synthetic fine-tune and data")
    return p.parse_args(argv)


def load_measure_groups():
    """scripts/measure_groups.py as a module (the MUST-04 code path; as tests/test_measure_groups.py loads it)."""
    name = "measure_groups_for_a15b"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "measure_groups.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class MgRuntime:
    """The attributes of measure_groups.Runtime that its analysis functions read (a, device, adapter, model_name,
    fit_loader), around any adapter, so a fine-tuned state dict can be measured."""

    def __init__(self, a_mg, device, adapter, model_name: str, fit_loader):
        self.a, self.device, self.adapter, self.model_name = a_mg, device, adapter, model_name
        self._fit_loader = fit_loader
        self.proto = self.controlled = self.num_classes = None

    def fit_loader(self):
        return self._fit_loader()

    def heldout_loader(self):
        return None


def mg_args(mg, a, out_dir: Path, n_draws: int):
    argv = ["--out", str(out_dir), "--groups", GROUP, "--fit-seed", str(a.fit_seed), "--blocks", a.blocks,
            "--null-draws", str(n_draws), "--twin-null-draws", str(a.twin_null_draws), "--twin-null-groups", GROUP,
            "--twin-null-blocks", a.range, "--summary-ranges", a.range, "--top-k", str(a.top_k),
            "--workers", str(a.workers), "--seed", str(a.seed), "--experiment", EXPERIMENT, "--no-heldout",
            "--cache-dir", str(a.cache_dir), "--batch-size", str(a.batch_size),
            "--stats-batch-size", str(a.stats_batch_size), "--num-workers", str(a.num_workers),
            "--gpu-frac", str(a.gpu_frac)]
    if a.no_attn:
        argv.append("--no-attn")
    if a.dry_run:
        argv.append("--tiny")
    return mg.parse_args(argv)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# Fine-tuned state dict
# ----------------------------------------------------------------------------------------------
_WRAP_KEYS = ("model", "state_dict", "student", "model_state_dict", "weights")
_PREFIXES = ("_orig_mod.", "module.", "model.")


def unwrap_state_dict(obj) -> dict:
    """The tensor dictionary inside a checkpoint (plain, or under a key such as "model"), prefixes removed."""
    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        sd = dict(obj)
    elif isinstance(obj, dict):
        inner = next((obj[k] for k in _WRAP_KEYS if isinstance(obj.get(k), dict)), None)
        if inner is None:
            raise ValueError(f"no state dict found in the checkpoint (keys: {sorted(obj)[:10]})")
        return unwrap_state_dict(inner)
    else:
        raise ValueError(f"checkpoint holds {type(obj).__name__}, expected a state dict")
    changed = True
    while changed:
        changed = False
        for pre in _PREFIXES:
            if sd and all(k.startswith(pre) for k in sd):
                sd = {k[len(pre):]: v for k, v in sd.items()}
                changed = True
    return sd


def _is_folded(sd: dict) -> bool:
    return any(k.startswith("norm.affine.") for k in sd) or ("blocks.0.norm1.weight" not in sd
                                                             and "blocks.0.mlp.fc1.weight" in sd)


@torch.no_grad()
def load_finetuned(adapter, path, step=None, device="cpu") -> tuple:
    """(adapter around the fine-tuned model, info). See the module docstring for the accepted formats."""
    from mirror.controls import unfold_rmsnorm, with_model

    p = Path(path)
    file = p
    if p.is_dir():
        ck = p / "ckpt"
        files = sorted(ck.glob("step*.pt"), key=lambda f: int(f.stem[4:]) if f.stem[4:].isdigit() else -1)
        file = ck / f"step{int(step)}.pt" if step is not None else (files[-1] if files else None)
        if file is None or not file.exists():
            raise FileNotFoundError(f"no checkpoint step{'' if step is None else step}.pt in {ck}")
    info = {"path": str(file), "sha256": sha256_file(file), "loader": "state dict"}
    try:
        sd = unwrap_state_dict(torch.load(file, map_location="cpu", weights_only=True))
        model = copy.deepcopy(adapter.model)
        if _is_folded(sd):
            from mirror.fold import fold_layernorm_full
            from mirror.m1 import materialise_pos_embed

            materialise_pos_embed(model, adapter)
            fold_layernorm_full(model, final_norm_affine=False)
            info["loader"] = "state dict of a folded student, unfolded to LayerNorm (exact)"
        pe = getattr(model, "pos_embed", None)
        if "pos_embed" in sd and isinstance(pe, torch.Tensor) and tuple(sd["pos_embed"].shape) != tuple(pe.shape):
            from mirror.m1 import materialise_pos_embed

            materialise_pos_embed(model, adapter)
            info["pos_embed"] = "materialised to the evaluation grid before loading"
        res = model.load_state_dict({k: v.to(dtype=next(model.parameters()).dtype) if v.is_floating_point() else v
                                     for k, v in sd.items()}, strict=False)
        missing = [k for k in res.missing_keys if not k.endswith(".P")]
        if missing or res.unexpected_keys:
            raise ValueError(f"state dict does not fit the model: missing {missing[:5]}, "
                             f"unexpected {list(res.unexpected_keys)[:5]}")
        unfold_rmsnorm(model)
    except Exception as e:
        if not p.is_dir():
            raise
        try:
            from mirror.final.heal.students import load_student      # heal agent's interface (docs, Section 4.1)
        except ImportError:
            raise RuntimeError(f"{file} did not load as a state dict ({e}) and mirror.final.heal.students is "
                               f"not available") from e
        _, model, sinfo = load_student(p, step=step, device="cpu")
        unfold_rmsnorm(model)
        info.update({"loader": "mirror.final.heal.students.load_student", "student_info": repr(sinfo)[:500],
                     "state_dict_error": repr(e)[:300]})
    model = model.to(device).eval()
    return with_model(adapter, model), info


# ----------------------------------------------------------------------------------------------
# Measurement through measure_groups
# ----------------------------------------------------------------------------------------------
def measure_r90(mg, rt: MgRuntime, out_dir: Path, git: dict) -> dict:
    """measure_groups' R90 analysis of rt.adapter into out_dir (its own files), returning its metrics."""
    a = rt.a
    adapter = rt.adapter
    L = adapter.depth
    blocks = mg.parse_blocks(a.blocks, L)
    attn = not a.no_attn
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "source.json"
    if src.exists() and json.loads(src.read_text(encoding="utf-8")).get("model") != rt.model_name:
        for name in MEASURED_TABLES:                   # a different model in the same folder: drop stale rows
            (out_dir / name).unlink(missing_ok=True)
    src.write_text(json.dumps({"model": rt.model_name}), encoding="utf-8")
    want = {"blocks": [int(b) for b in blocks], "model": rt.model_name, "fit_seed": int(a.fit_seed),
            "top_k": int(a.top_k), "attn": bool(attn)}
    inputs_ok = mg._cache_ok(out_dir / "cache" / f"inputs_fit{a.fit_seed}.npz",
                             {"blocks": want["blocks"], "model": rt.model_name, "fit_seed": int(a.fit_seed)})
    timings = {}
    if not (mg._cache_ok(mg._pass_path(out_dir, GROUP, a.fit_seed), want) and inputs_ok):
        t0 = time.time()
        mg.run_generator_pass(rt, GROUP, blocks, attn, need_inputs=not inputs_ok, out=out_dir)
        timings[f"pass_{GROUP}"] = round(time.time() - t0, 1)
    ctx = mg.Ctx(rt, out_dir, blocks, attn)
    lay_csv = mg.Csv(out_dir / "layers.csv", mg.LAYER_KEY)
    blk_csv = mg.Csv(out_dir / "blocks.csv", mg.BLOCK_KEY)
    drw_csv = mg.Csv(out_dir / "null_draws.csv", mg.DRAW_KEY)
    if not (all(blk_csv.done((GROUP, b + 1)) for b in blocks) and all(lay_csv.done((GROUP, l)) for l in ctx.layers)):
        t0 = time.time()
        lrows, brows, drows = mg.analyse_group(ctx, GROUP, git)
        lay_csv.add_many(lrows)
        drw_csv.add_many(drows)
        blk_csv.add_many(brows)
        timings[f"analyse_{GROUP}"] = round(time.time() - t0, 1)
    if a.twin_null_draws > 0:
        t0 = time.time()
        if mg.run_twin_null(ctx, GROUP, mg.Csv(out_dir / "twin_null.csv", mg.TWIN_KEY), git):
            timings[f"twin_null_{GROUP}"] = round(time.time() - t0, 1)
    extra = {"model": rt.model_name, "depth": L, "d": adapter.embed_dim, "h": adapter.mlp_hidden,
             "grid": list(adapter.grid_hw), "n_prefix_tokens": adapter.n_prefix_tokens,
             "blocks_measured": [b + 1 for b in blocks], "timings_s": timings, "git": git,
             "n_fit_images": ctx.pass_(GROUP)["meta"]["n_images"]}
    metrics = mg.build_metrics(out_dir, [GROUP], blocks, L, a, extra)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


def r90_numbers(metrics: dict, range_name: str) -> dict:
    r = ((metrics.get("groups") or {}).get(GROUP) or {}).get("ranges", {}).get(range_name, {})
    keys = ("ratio_to_null", "share_4cycles", "twin_fraction_t50", "frac_4_t50", "frac_fixed", "s_mlp_mean",
            "s_mlp_null_mean", "s_mlp_null_lo", "s_mlp_null_hi", "s_fc1_ratio_to_null", "s_fc2_ratio_to_null",
            "s_attn_ratio_to_null", "tax_mlp", "blocks")
    out = {k: r.get(k) for k in keys}
    if "twin_null" in r:
        out["twin_null"] = r["twin_null"]
    return out


def score_a15b(ratio_original, ratio_finetuned, min_drop: float = A15B_MIN_DROP) -> dict:
    """PASS when the R90 ratio to null falls by at least min_drop (original minus fine-tuned, with a 1e-12 allowance
    for float rounding at the bar); PENDING if a number is missing or not finite."""
    def ok(v):
        return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))

    out = {"id": "A15b", "rule": f"R90 ratio to null falls by at least {min_drop} after the fine-tune",
           "ratio_original": ratio_original, "ratio_finetuned": ratio_finetuned, "min_drop": min_drop}
    if not (ok(ratio_original) and ok(ratio_finetuned)):
        return {**out, "drop": None, "status": "PENDING"}
    drop = float(ratio_original) - float(ratio_finetuned)
    return {**out, "drop": drop, "status": "PASS" if drop >= min_drop - 1e-12 else "FAIL"}


def must04_check(a, numbers: dict) -> dict | None:
    path = Path(a.must04) if a.must04 else REPO_ROOT / "results" / "MUST-04" / a.model / "metrics.json"
    if a.dry_run or not path.exists():
        return None
    sc = (json.loads(path.read_text(encoding="utf-8")).get("scores") or {}).get(GROUP) or {}
    out = {"file": str(path)}
    for key, col in (("ratio_to_null", f"ratio_to_null_{a.range}"), ("share_4cycles", f"share_4cycles_{a.range}")):
        ref, now = sc.get(col), numbers.get(key)
        out[f"{key}_must04"], out[f"{key}_now"] = ref, now
        out[f"{key}_diff"] = (now - ref) if (ref is not None and now is not None) else None
    return out


# ----------------------------------------------------------------------------------------------
def build_models(a, device, out: Path):
    """(original adapter, fine-tuned adapter, fine-tune info, fit loader factory, model names)."""
    if a.dry_run:                                        # the model and data of measure_groups.py --tiny
        from mirror.adapters import TimmTinyAdapter
        from mirror.variants import TinyProtocol

        ad = TimmTinyAdapter(img_size=56, seed=0)
        proto = TinyProtocol()
        ft_path = Path(a.finetuned) if a.finetuned else out / "dry_run_finetuned.pt"
        if not a.finetuned:                              # a stand-in "fine-tune": seeded relative noise on every weight
            g = torch.Generator().manual_seed(123)
            sd = {k: (v + 0.2 * v.std().nan_to_num(0.0) * torch.randn(v.shape, generator=g, dtype=v.dtype)
                      if v.is_floating_point() and v.numel() > 1 else v.clone())
                  for k, v in ad.model.state_dict().items()}
            torch.save({"model": sd, "step": 0}, ft_path)
        name = "tiny"
        fit_loader = lambda: proto.fit_loader(a.fit_seed)      # noqa: E731
    else:
        from mirror.adapters_timm import build_model_adapter
        from mirror.protocol import Protocol

        if not a.finetuned:
            raise SystemExit("--finetuned is required (state-dict file or run directory of the rot90 probe)")
        ad = build_model_adapter(a.model, device=device)
        proto = Protocol(a.cache_dir, a.batch_size, a.stats_batch_size, a.num_workers, device.type == "cuda")
        ft_path = Path(a.finetuned)
        name = a.model
        fit_loader = lambda: proto.fit_loader(a.fit_seed)      # noqa: E731
    ft, info = load_finetuned(ad, ft_path, a.step, device)
    return ad, ft, info, fit_loader, (name, f"{name}+finetuned:{info['sha256'][:12]}")


def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    prereg = require_prereg(allow_missing=True if a.dry_run else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    mg = load_measure_groups()        # before the memory cap: importing it sets its own cap (from sys.argv or 0.2)
    if a.dry_run:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
        torch.cuda.reset_peak_memory_stats()
    else:
        raise SystemExit("no GPU visible; use --dry-run for CPU runs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    git = git_info()
    n_draws = int(a.null_draws if a.null_draws is not None else (3 if a.dry_run else 20))
    orig, ft, ft_info, fit_loader, (name_o, name_f) = build_models(a, device, out)
    log(f"A15b: {name_o} against {name_f} ({ft_info['loader']}), group {GROUP}, blocks {a.blocks}, "
        f"{n_draws} null draws")
    measured, timings = {}, {}
    for tag, ad, name in (("original", orig, name_o), ("finetuned", ft, name_f)):
        t0 = time.time()
        rt = MgRuntime(mg_args(mg, a, out / tag, n_draws), device, ad, name, fit_loader)
        measured[tag] = measure_r90(mg, rt, out / tag, git)
        timings[tag] = round(time.time() - t0, 1)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    nums = {tag: r90_numbers(m, a.range) for tag, m in measured.items()}
    score = score_a15b(nums["original"]["ratio_to_null"], nums["finetuned"]["ratio_to_null"])
    check = must04_check(a, nums["original"])
    import pandas as pd

    pd.DataFrame([{"model_state": tag, "model": (name_o if tag == "original" else name_f),
                   **{k: v for k, v in n.items() if k not in ("twin_null", "blocks")}}
                  for tag, n in nums.items()]).to_csv(out / "a15b.csv", index=False)
    metrics = {"experiment": EXPERIMENT, "model": a.model if not a.dry_run else "tiny", "group": GROUP,
               "range": a.range, "blocks_measured": a.blocks, "null_draws": n_draws, "null_seed": a.seed,
               "fit_seed": a.fit_seed, "finetuned": ft_info, "original": nums["original"],
               "finetuned_numbers": nums["finetuned"], "scores": {"A15b": score}, "must04_check": check,
               "timings_s": timings, "git": git, "dry_run": bool(a.dry_run)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    def f(v, nd=3):
        return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"

    no, nf = nums["original"], nums["finetuned"]
    lines = [f"# A15b cause probe: 90-degree rotation structure, {metrics['model']}", "",
             f"**A15b (pre-registered): {score['status']}.** The R90 ratio to null in blocks {a.range} went from "
             f"{f(no['ratio_to_null'])} (original) to {f(nf['ratio_to_null'])} (fine-tuned), a drop of "
             f"{f(score['drop'])}; the rule needs a drop of at least {A15B_MIN_DROP}.", "",
             "| model | ratio to null | share in 4-cycles | twin fraction (2-cycles, S >= 0.5) | mean s_mlp "
             "| null mean |",
             "|---|---|---|---|---|---|",
             f"| original | {f(no['ratio_to_null'])} | {f(no['share_4cycles'])} | {f(no['twin_fraction_t50'])} | "
             f"{f(no['s_mlp_mean'])} | {f(no['s_mlp_null_mean'])} |",
             f"| fine-tuned | {f(nf['ratio_to_null'])} | {f(nf['share_4cycles'])} | {f(nf['twin_fraction_t50'])} | "
             f"{f(nf['s_mlp_mean'])} | {f(nf['s_mlp_null_mean'])} |", "",
             f"Fine-tuned weights: {ft_info['path']} (sha256 {ft_info['sha256'][:12]}, {ft_info['loader']}).",
             f"Code path: scripts/measure_groups.py (group R90, MUST-04), {n_draws} null draws, base seed {a.seed}, "
             f"fit set fit{a.fit_seed}."]
    if check:
        lines.append(f"MUST-04 recorded ratio {f(check.get('ratio_to_null_must04'))} and share in 4-cycles "
                     f"{f(check.get('share_4cycles_must04'))} for the original; difference now "
                     f"{f(check.get('ratio_to_null_diff'), 4)} and {f(check.get('share_4cycles_diff'), 4)}.")
    if a.dry_run:
        lines.append("Dry run: tiny random model and a noise stand-in for the fine-tune; the numbers mean nothing.")
    lines += ["", "Reading: a lower ratio means the MLP weights fit the measured 90-degree structure better than "
              "random structures of the same shape. The rule tests whether training for rotation invariance makes "
              "rotation structure appear; it is evidence about the cause of the pairing, not proof."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_final_meta(out, experiment=EXPERIMENT, config=vars(a),
                     seeds={"fit_seed": a.fit_seed, "null_seed": a.seed, "null_draws": n_draws},
                     models={name_o: {"registry": a.model} if not a.dry_run else {"note": "tiny random timm ViT"},
                             name_f: ft_info},
                     device=device, wall_seconds=time.time() - t_start,
                     extra={"prereg_check": prereg, "timings_s": timings, "verdict_A15b": score["status"]})
    log(f"A15b: {score['status']} (drop {score['drop']}); outputs in {out}")
    return metrics


if __name__ == "__main__":
    main()
