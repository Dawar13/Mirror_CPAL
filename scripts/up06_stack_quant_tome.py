#!/usr/bin/env python
"""UP-06 (PROJECT_NEXT_PHASE.md, Section 7): does Mirror stack with INT8 weight-only quantisation and token merging?

    python scripts/up06_stack_quant_tome.py --healed-run runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0
    python scripts/up06_stack_quant_tome.py --tiny      # CPU smoke test: heals a tiny Mirror-exact model first (~30 s)

Prediction (PREREG_NEXT_PHASE.md, Section 4; reported, not gating): the extra accuracy drop from INT8 weight-only
quantisation and from ToMe (r = 8) is the same for both models within 0.5 points.

Models (two bases, three forms each, six rows):
  original        the released model (the adapter of the healed run's spec, as MUST-10 uses it);
  mirror_exact    healed Mirror-exact at full conversion (MUST-09; mirror.heal_structured.load_healed_model);
  +int8           INT8 weight-only quantisation, symmetric, per output channel, of every nn.Linear inside the blocks
                  (qkv, proj, fc1, fc2): s = max|W_row| / 127 stored in fp32, q = round(W / s) in [-127, 127],
                  weights replaced by q s (simulated: the dequantised weights run in the usual float kernels; under
                  bf16 autocast they are rounded to bf16 at the matrix product, as every model's weights are).
                  Biases, norms, embeddings, the head and the even projection stay in float.
  +tome           ToMe token merging (mirror/tome.py) with r = 8 tokens per block in every block: bipartite soft
                  matching on the keys averaged over heads, size-weighted averaging, proportional attention; prefix
                  tokens (CLS and registers) are never merged. Applied to the bank and the queries alike.

Evaluation (PREREG_NEXT_PHASE.md Section 1, the paper protocol): validation k-NN top-1 with a 100k train bank and all
50k validation queries, CLS after the final norm (the even part for Mirror-exact), cosine, k = 20, T = 0.07, bf16
features (scripts/must07_compressors.Evaluator, the MUST-07 code path). Flip consistency (share of validation images
whose prediction equals that of the mirrored image) for every row in bf16, and in fp32 (TF32 off) for the Mirror-exact
rows (--fp32 all: every row), with the number of disagreements and the largest unit-feature difference. ToMe and INT8
are not mirror-equivariant operations, so they can break Mirror-exact's exact invariance; this is measured, not
assumed.

Scoring: extra drop of form X for base b = top1(b) - top1(b + X); difference = drop(mirror_exact) - drop(original).
The prediction holds for X if |difference| <= 0.5 points (point estimate; fixed here before any run). A paired
bootstrap over the validation images (mirror/bootstrap.py, 10,000 resamples, seed --seed) of the per-image
difference-in-differences gives a 95% interval, reported next to it.

Outputs under --out (default results/UP-06; tiny: tiny_out/results/UP-06): rows.csv, comparison.json, summary.md,
meta.json (mirror/meta.py), rows/<variant>.json (resumable per variant; --force redoes them), preds/<variant>.npz
(validation predictions for the images and their mirrors, labels).
"""
from __future__ import annotations

import sys

import torch

DEFAULT_GPU_FRAC = 0.2


def _early_gpu_frac(argv, default: float = DEFAULT_GPU_FRAC) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))  # shared GPU: cap before allocating

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.bootstrap import paired_bootstrap  # noqa: E402
from mirror.heal_structured import load_healed_model  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.tome import apply_tome, expected_tokens, r_schedule  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "UP-06"
BAR = 0.5                    # PREREG_NEXT_PHASE.md Section 4, UP-06: same extra drop within 0.5 points
TOME_R = 8                   # PROJECT_NEXT_PHASE.md UP-06
BASES = ("original", "mirror_exact")
FORMS = ("none", "int8", "tome")
FORM_LABEL = {"none": "as is", "int8": "INT8 weight-only, per output channel", "tome": "ToMe"}
BASE_LABEL = {"original": "original", "mirror_exact": "healed Mirror-exact"}
NOT_LINEAR_LIKE = ("LowRankLinear", "IndexedLinear", "SlicedLinear", "LowRankQKV")


def variant_name(base: str, form: str) -> str:
    return base if form == "none" else f"{base}+{form}"


ALL_VARIANTS = tuple(variant_name(b, f) for b in BASES for f in FORMS)


def _load(name: str):
    key = f"_up06_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--healed-run", default=None,
                   help="MUST-09 run folder of healed Mirror-exact at full conversion (spec.json, healed_state.pt); "
                        "with --tiny and no folder, a tiny one is healed first")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS), help=f"comma list from {', '.join(ALL_VARIANTS)}")
    p.add_argument("--tome-r", type=int, default=TOME_R, help="tokens merged per block (ToMe r)")
    p.add_argument("--fp32", default="mirror", choices=["mirror", "all", "none"],
                   help="fp32 flip-consistency passes: Mirror-exact rows (default), every row, or none")
    p.add_argument("--out", default=None, help="default results/UP-06 (tiny: tiny_out/results/UP-06)")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--eval-batch", type=int, default=None, help="feature batch (default 256; tiny 32)")
    p.add_argument("--knn-batch-size", type=int, default=128, help="loader batch of the protocol")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--preload", default="cpu", choices=["cpu", "gpu", "loader"],
                   help="keep the bank and validation images in host memory (cpu, about 23 GB), on the GPU, or "
                        "re-read them for every pass (loader)")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0, help="global seed and the bootstrap seed")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC, help="per-process GPU memory cap")
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="redo finished variants")
    p.add_argument("--tiny", action="store_true", help="tiny random ViT and synthetic data on CPU")
    return p.parse_args(argv)


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# ----------------------------------------------------------------------------------------------
# INT8 weight-only quantisation
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def int8_per_channel(W: torch.Tensor) -> tuple:
    """Symmetric INT8 per output channel of W [out, in]: s = max|W_row| / 127 (fp32; 1 for an all-zero row),
    q = round(W / s) clamped to [-127, 127] (int8), W_hat = q s in W's dtype. Round half to even is odd-symmetric,
    so W -> -W gives q -> -q. Bound: |W - W_hat| <= s / 2 per entry (up to fp32 rounding of the product q s)."""
    amax = W.detach().abs().amax(dim=1).to(torch.float64)
    s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).to(torch.float32)
    q = torch.round(W.detach().to(torch.float64) / s.to(torch.float64)[:, None]).clamp_(-127, 127)
    W_hat = (q.to(torch.float32) * s[:, None]).to(W.dtype)
    return q.to(torch.int8), s, W_hat


@torch.no_grad()
def quantise_blocks_int8(model: nn.Module) -> dict:
    """Replace the weight of every nn.Linear inside model.blocks by its INT8 round trip (in place)."""
    n_mat = n_w = n_rows = 0
    worst = 0.0
    for b, blk in enumerate(model.blocks):
        for name, mod in blk.named_modules():
            if type(mod).__name__ in NOT_LINEAR_LIKE:
                raise NotImplementedError(f"block {b} {name}: {type(mod).__name__} is not quantised by this script")
            if not isinstance(mod, nn.Linear):
                continue
            q, s, W_hat = int8_per_channel(mod.weight)
            err = (mod.weight.detach().to(torch.float64) - W_hat.to(torch.float64)).abs().amax(dim=1)
            worst = max(worst, float((err / s.to(torch.float64)).max()))
            mod.weight.copy_(W_hat)
            n_mat += 1
            n_w += int(q.numel())
            n_rows += int(s.numel())
    if n_mat == 0:
        raise ValueError("no nn.Linear found in model.blocks")
    return {"n_matrices": n_mat, "n_weights": n_w, "n_scales": n_rows, "bytes_fp32": 4 * n_w,
            "bytes_int8": n_w + 4 * n_rows, "max_abs_err_over_scale": worst,
            "scheme": "symmetric, per output channel, [-127, 127], fp32 scale max|W_row|/127, round half to even; "
                      "weights dequantised in place (simulated weight-only INT8)"}


# ----------------------------------------------------------------------------------------------
# Variants and evaluation
# ----------------------------------------------------------------------------------------------
def build_variant(base_model: nn.Module, form: str, r: int, n_protected: int) -> tuple:
    if form == "none":
        return base_model, {}
    if form == "int8":
        m = copy.deepcopy(base_model)
        return m, {"int8": quantise_blocks_int8(m)}
    if form == "tome":
        return apply_tome(base_model, r, n_protected, prop_attn=True), {"tome_r": int(r), "n_protected": int(n_protected)}
    raise ValueError(f"unknown form {form!r}")


def tiny_healed_run(out: Path) -> Path:
    """Heal a tiny Mirror-exact model with scripts/must09_heal.py --tiny (skipped if already there)."""
    M9 = _load("must09_heal")
    root = out / "_tiny_must09"
    M9.main(["--tiny", "--method", "mirror_exact", "--mirror-builder", "shim", "--lr", "1e-4",
             "--out-root", str(root / "runs"), "--results-root", str(root / "results")])
    return root / "runs" / "tiny" / "mirror_exact" / "s0"


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path: Path, obj) -> None:
    from mirror.meta import _jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list) -> None:
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    os.replace(tmp, path)


def evaluate(ev, model: nn.Module, fp32: bool, amp) -> tuple:
    """bf16 (or fp32 on CPU) k-NN with flip consistency, plus the fp32 pass when asked. Returns (row part, preds)."""
    bf, _, by = ev.feats(model, "bank")
    vv = ev.val(model, {"bank_f": bf, "bank_y": by}, flip=True)
    del bf
    pred, pflip, y = vv["val_pred"], vv["val_pred_flip"], vv["val_y"]
    part = {"top1": vv["top1"], "flip_consistency": vv["flip_consistency"],
            "n_flip_disagree": int((pred != pflip).sum()), "max_abs_unit_feature_diff": vv["max_abs_unit_feature_diff"],
            "n_bank": int(by.shape[0]), "n_val": vv["n_val"], "amp": amp or "fp32"}
    preds = {"pred": pred.numpy().astype(np.int16), "pred_flip": pflip.numpy().astype(np.int16),
             "labels": y.numpy().astype(np.int16)}
    if fp32:
        if amp is None:                       # the main pass already ran in fp32
            fp = {"flip_consistency_fp32": part["flip_consistency"], "top1_fp32": part["top1"],
                  "max_abs_unit_feature_diff_fp32": part["max_abs_unit_feature_diff"],
                  "n_flip_disagree_fp32": part["n_flip_disagree"], "n_bank_fp32": part["n_bank"],
                  "n_val_fp32": part["n_val"]}
            preds.update(pred_fp32=preds["pred"], pred_flip_fp32=preds["pred_flip"])
        else:
            f = ev.flip_fp32(model)
            fp = {k: f[k] for k in ("flip_consistency_fp32", "top1_fp32", "max_abs_unit_feature_diff_fp32",
                                    "n_bank_fp32", "n_val_fp32")}
            fp["n_flip_disagree_fp32"] = int((f["val_pred_fp32"] != f["val_pred_flip_fp32"]).sum())
            preds.update(pred_fp32=f["val_pred_fp32"].numpy().astype(np.int16),
                         pred_flip_fp32=f["val_pred_flip_fp32"].numpy().astype(np.int16))
        part.update(fp)
    return part, preds


def correct_of(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return (z["pred"] == z["labels"]).astype(np.float64)


def compare(out: Path, rows: dict, n_boot: int, seed: int) -> dict:
    """Extra drops, their difference against the 0.5 bar, and paired bootstraps over the validation images."""
    comp = {"bar_points": BAR, "reading": "point estimate |drop(mirror_exact) - drop(original)| <= bar", "forms": {}}
    for form in ("int8", "tome"):
        names = {b: (variant_name(b, "none"), variant_name(b, form)) for b in BASES}
        if not all(n in rows for pair in names.values() for n in pair):
            continue
        drops = {b: rows[n0]["top1"] - rows[nx]["top1"] for b, (n0, nx) in names.items()}
        diff = drops["mirror_exact"] - drops["original"]
        c = {b: (correct_of(out / "preds" / f"{n0}.npz"), correct_of(out / "preds" / f"{nx}.npz"))
             for b, (n0, nx) in names.items()}
        boot_drop = {b: paired_bootstrap(c[b][0], c[b][1], n_resamples=n_boot, seed=seed) for b in BASES}
        did = paired_bootstrap(c["mirror_exact"][0] - c["mirror_exact"][1], c["original"][0] - c["original"][1],
                               n_resamples=n_boot, seed=seed)
        comp["forms"][form] = {
            "drop_original": drops["original"], "drop_mirror_exact": drops["mirror_exact"],
            "difference": diff, "pass": bool(abs(diff) <= BAR),
            "bootstrap_difference": {k: did[k] for k in ("n", "n_resamples", "seed", "diff", "ci_low", "ci_high",
                                                         "excludes_zero")},
            "ci_inside_bar": bool(did["ci_low"] >= -BAR and did["ci_high"] <= BAR),
            "bootstrap_drop": {b: {k: v[k] for k in ("diff", "ci_low", "ci_high")} for b, v in boot_drop.items()},
        }
    forms = comp["forms"]
    comp["prediction_holds"] = (bool(all(f["pass"] for f in forms.values()))
                                if set(forms) == {"int8", "tome"} else None)
    return comp


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def _f(x, nd=2) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not np.isfinite(xf) else f"{xf:.{nd}f}"


def write_summary(path: Path, rows: dict, comp: dict, info: dict) -> None:
    L = []
    tiny = info["tiny"]
    L.append(f"# UP-06: Mirror stacked with INT8 quantisation and token merging"
             f"{' (TINY TEST: synthetic data, not a result)' if tiny else ''}\n")
    L.append("## Verdict\n")
    held = comp.get("prediction_holds")
    L.append(f"Prediction (PREREG_NEXT_PHASE.md, UP-06): the extra drop from each method is the same for both models "
             f"within {BAR} points. **{'HOLDS' if held else ('DOES NOT HOLD' if held is False else 'NOT SCORED (rows missing)')}**.\n")
    for form, c in comp["forms"].items():
        b = c["bootstrap_difference"]
        L.append(f"- {FORM_LABEL[form]}: extra drop {_f(c['drop_original'])} for the original, "
                 f"{_f(c['drop_mirror_exact'])} for healed Mirror-exact; difference {_f(c['difference'])} points "
                 f"(95% interval {_f(b['ci_low'])} to {_f(b['ci_high'])}, n = {b['n']}): "
                 f"**{'PASS' if c['pass'] else 'FAIL'}** (bar {BAR}; interval inside the bar: "
                 f"{'yes' if c['ci_inside_bar'] else 'no'}).")
    L.append("")
    L.append("## Rows\n")
    any_row = next(iter(rows.values()), {})
    L.append(f"Validation k-NN top-1 (bank of {any_row.get('n_bank', 'n/a')} train images, "
             f"{any_row.get('n_val', 'n/a')} validation queries, k = {info['k']}, T = {info['T']}, "
             f"{'fp32 on CPU' if tiny else 'bf16 features'}); flip consistency = share of validation images whose "
             "prediction equals that of the mirrored image; fp32 columns with TF32 off.\n")
    L.append("| variant | top-1 | flip consistency | disagreements | max unit-feature diff | top-1 fp32 | "
             "flip consistency fp32 | disagreements fp32 | tokens after last block |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name in ALL_VARIANTS:
        r = rows.get(name)
        if r is None:
            continue
        L.append(f"| {name} | {_f(r['top1'])} | {_f(r['flip_consistency'], 3)} | {r['n_flip_disagree']} | "
                 f"{_f(r['max_abs_unit_feature_diff'], 6)} | {_f(r.get('top1_fp32'))} | "
                 f"{_f(r.get('flip_consistency_fp32'), 3)} | {r.get('n_flip_disagree_fp32', 'n/a')} | "
                 f"{r.get('tokens_final', 'n/a')} |")
    L.append("")
    L.append("## Provenance\n")
    L.append(f"- Healed run: `{info['healed_run']}` (method {info['healed_method']}, healing seed {info['healed_seed']}); "
             f"model {info['model']}; git {info['git']['git_hash']}{' (dirty)' if info['git'].get('git_dirty') else ''}; "
             f"seed {info['seed']}; protocol {info['protocol']}.")
    if info.get("int8"):
        q = info["int8"]
        L.append(f"- INT8: {q['n_matrices']} matrices, {q['n_weights']} weights, largest |W - W_hat| / s = "
                 f"{_f(q['max_abs_err_over_scale'], 4)} (bound 0.5); {q['scheme']}.")
    if info.get("tome"):
        t = info["tome"]
        npro = t["n_protected"]
        L.append(f"- ToMe: r = {t['r']} per block, {npro} prefix token{'s' if npro != 1 else ''} (CLS and registers) "
                 f"never merged, proportional attention; tokens after each block "
                 f"{', '.join(str(n) for n in t['expected_tokens'])} (from {t['n_tokens']}).")
    if info.get("int8_mirror_checks"):
        viol = "; ".join(f"{k.split(':', 1)[-1]} {float(v):.2e}" for k, v in info["int8_mirror_checks"].items())
        L.append(f"- Mirror-exact weight constraints after INT8 (largest relative violation; about 1e-8 before): {viol}.")
    L.append("")
    L.append("## Caveats\n")
    for c in info["caveats"]:
        L.append(f"- {c}")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    tiny = a.tiny
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    timer = StageTimer()
    out = resolve(a.out or ("tiny_out/results/UP-06" if tiny else "results/UP-06"))
    out.mkdir(parents=True, exist_ok=True)
    wanted = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = sorted(set(wanted) - set(ALL_VARIANTS))
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {ALL_VARIANTS}")

    M7 = _load("must07_compressors")         # Evaluator: the MUST-07 k-NN code path
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))   # the import set its own cap

    with timer.stage("load"):
        if a.healed_run is None:
            if not tiny:
                raise SystemExit("--healed-run is required (or use --tiny)")
            run = tiny_healed_run(out)
        else:
            run = resolve(a.healed_run)
        adapter, healed, hinfo = load_healed_model(run, device)
        spec = hinfo["spec"]
        if spec.get("family") != "mirror" or not spec["base"].get("even_head"):
            raise SystemExit(f"{run} is not a full-conversion Mirror-exact run (family {spec.get('family')})")
        model_name = spec.get("model") or ("tiny" if tiny else "model")
        if tiny:
            from mirror.variants import TinyProtocol

            proto = TinyProtocol(img_size=adapter.img_size)
            n_cls, protocol = proto.num_classes, "tiny synthetic"
        else:
            from mirror.protocol import Protocol

            proto = Protocol(resolve(a.cache_dir), batch_size=a.knn_batch_size, num_workers=a.num_workers,
                             pin=device.type == "cuda")
            n_cls, protocol = 1000, "v2 (100k train bank, 50k validation)"
    amp = "bf16" if device.type == "cuda" else None
    ev = M7.Evaluator(adapter, proto, device, amp, a.eval_batch or (32 if tiny else 256), a.preload, n_cls,
                      a.knn_k, a.knn_temperature)
    bases = {"original": adapter.model, "mirror_exact": healed}
    npre = int(adapter.n_prefix_tokens)
    r_list = r_schedule(a.tome_r, adapter.depth)
    exp_tokens = expected_tokens(adapter.n_tokens, npre, r_list)
    settings = {"healed_run": str(run), "tome_r": int(a.tome_r), "protocol": protocol, "k": a.knn_k,
                "T": a.knn_temperature, "amp": amp or "fp32"}
    log(f"UP-06: model {model_name}, healed run {run}, device {device}, git {git['git_hash'][:10]}, "
        f"variants {wanted}, ToMe r {a.tome_r} (tokens {adapter.n_tokens} -> {exp_tokens[-1]})")

    rows, extra = {}, {}
    for name in ALL_VARIANTS:
        if name not in wanted:
            continue
        base, form = (name.split("+") + ["none"])[:2]
        rp = out / "rows" / f"{name}.json"
        pp = out / "preds" / f"{name}.npz"
        rec = None if a.force else read_json(rp)
        fp32 = a.fp32 == "all" or (a.fp32 == "mirror" and base == "mirror_exact")
        if rec and rec.get("settings") == settings and pp.exists() and (not fp32 or "flip_consistency_fp32" in rec["row"]):
            rows[name] = rec["row"]
            log(f"{name}: reused {rp}")
            continue
        with timer.stage(name):
            t0 = time.time()
            model, binfo = build_variant(bases[base], form, a.tome_r, npre)
            part, preds = evaluate(ev, model, fp32, amp)
            row = {"experiment": EXPERIMENT, "model": model_name, "variant": name, "base": base, "form": form,
                   **part, "tome_r": int(a.tome_r) if form == "tome" else None,
                   "seconds": round(time.time() - t0, 1), "healed_run": str(run), "healed_seed": spec.get("seed"),
                   "seed": a.seed, "git_hash": git["git_hash"], "git_dirty": git.get("git_dirty")}
            if form == "tome":
                seen = list(model.tome_state["tokens"])
                if seen != exp_tokens:
                    raise RuntimeError(f"ToMe token counts {seen} differ from the schedule {exp_tokens}")
                row["tokens_final"] = seen[-1]
                extra["tome"] = {"r": int(a.tome_r), "n_protected": npre, "n_tokens": int(adapter.n_tokens),
                                 "expected_tokens": exp_tokens}
            else:
                row["tokens_final"] = int(adapter.n_tokens)
            if form == "int8":
                q = binfo["int8"]
                row.update({f"int8_{k}": v for k, v in q.items() if k != "scheme"})
                extra["int8"] = q
                if base == "mirror_exact":
                    checks = {f"{i}:{type(c).__name__}": c.check(model) for i, c in enumerate(hinfo["constraints"])
                              if callable(getattr(c, "check", None)) and type(c).__name__ != "VExactEmbeddings"}
                    viol = {k: v.get("max_rel_violation") for k, v in checks.items()}
                    row["int8_mirror_max_rel_violation"] = max(float(v) for v in viol.values()) if viol else None
                    extra["int8_mirror_checks"] = viol
            pp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pp, **preds)
            write_json(rp, {"row": row, "settings": settings, "variant_info": binfo})
            rows[name] = row
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        log(f"{name}: top-1 {row['top1']:.2f}, flip consistency {row['flip_consistency']:.3f}%"
            + (f", fp32 {row['flip_consistency_fp32']:.3f}%" if "flip_consistency_fp32" in row else ""))
    if "tome" not in extra:
        extra["tome"] = {"r": int(a.tome_r), "n_protected": npre, "n_tokens": int(adapter.n_tokens),
                         "expected_tokens": exp_tokens}

    comp = compare(out, rows, a.n_boot, a.seed)
    write_csv(out / "rows.csv", [rows[n] for n in ALL_VARIANTS if n in rows])
    write_json(out / "comparison.json", comp)
    caveats = [
        "INT8 is simulated: weights are replaced by their dequantised INT8 values and run in the usual float kernels; "
        "the accuracy effect is that of the rounding, no INT8 kernel or speed is measured.",
        "ToMe is not mirror-equivariant (its alternating A/B partition swaps under a left-right flip of an even-width "
        "grid), so Mirror-exact + ToMe need not be exactly invariant; INT8 rounding of a dense, non-diagonal A* basis "
        "also breaks the exact weight constraints slightly. Both are measured in the flip-consistency columns.",
        "Prefix tokens (CLS and registers) are protected from merging; the official ToMe code protects the class "
        "token only, and registers are protected here in the same way.",
        f"One healed Mirror-exact model (healing seed {spec.get('seed')}); other seeds need another invocation.",
        "The verdict uses the point estimate of the difference in extra drops; the bootstrap interval is reported "
        "next to it and is not part of the verdict.",
    ]
    if tiny:
        caveats.insert(0, "TINY TEST: a tiny random ViT healed for a few steps on synthetic images; every number "
                          "checks the code, none is a result.")
    info = {"tiny": tiny, "healed_run": str(run), "healed_method": spec.get("method"), "healed_seed": spec.get("seed"),
            "model": model_name, "git": git, "seed": a.seed, "protocol": protocol, "caveats": caveats,
            "k": a.knn_k, "T": a.knn_temperature, **extra}
    write_summary(out / "summary.md", rows, comp, info)
    write_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"seed": a.seed, "bootstrap": a.seed,
                                                                   "healed": spec.get("seed")},
               models={model_name: spec.get("adapter")}, device=device, wall_seconds=time.time() - t_start,
               extra={"healed_run": str(run), "healed_method": spec.get("method"), "timing": timer.stages,
                      "bar_points": BAR, **extra})
    log(f"UP-06 done in {timer.total() / 60:.1f} min; prediction "
        f"{'holds' if comp.get('prediction_holds') else 'does not hold' if comp.get('prediction_holds') is False else 'not scored'}; "
        f"results in {out}")
    return {"rows": rows, "comparison": comp, "out": str(out), "healed_run": str(run)}


if __name__ == "__main__":
    main()
