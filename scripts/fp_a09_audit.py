#!/usr/bin/env python
"""A09: exactness audit at three levels (E1 exact in real arithmetic, E2 numerically exact, E3 bitwise) for the
dense-stored and the even/odd form (FINAL_PHASE_PLAN.md v2, item A09; PREREG_FINAL_PHASE.md Sections 1 and 2.6;
mirror/final/audit/exactness.py has the definitions).

    python scripts/fp_a09_audit.py --run runs/final/A05/dinov2_vitb14_reg/mirror_exact/s0      # a healed model
    python scripts/fp_a09_audit.py --model dinov2_vitb14_reg \
        --span-structure results/MUST-07/structure/dinov2_vitb14_reg_s0.npz                    # zero-shot
    python scripts/fp_a09_audit.py --dry-run --out tiny_out/A09                                 # CPU, tiny, synthetic

Steps (each resumable: a finished output file is reused unless --force):
 1. model and structure. --run: the heal agent's loader (mirror.final.heal.students.load_student) when it exists,
    otherwise mirror.heal_structured.load_healed_model (MUST-09 runs); the structure (A*, pairings, head maps after
    the gauge, register pairing) comes from the loader's info. --span-structure: the zero-shot full conversion
    (mirror.m1.build_vexact with the even head). --dry-run: a tiny random model and a random exact structure.
    DeiT-III: the classifier head is audited too (its logits must be invariant) and the flip check uses the
    predicted class;
 2. weight conditions on the stored weights (weights.csv);
 3. per-component audit on --n-images validation images (the first image of each class, as MUST-02), forms dense
    and even_odd, precisions float64 (reference), float64_exact (E1), float32 and bf16_autocast (and bf16 with
    --bf16-weights) in lockstep, plus the original model's own float32-versus-float64 floor (components.csv);
 4. flip checks per form in float32 and bf16 autocast: k-NN against the bank (--flip-bank-per-class, 0 = the full
    100k bank) on --flip-queries validation images (0 = all 50,000), the bank, queries and mirrored queries all in
    that precision (PREREG Section 2.1), predictions in knn/<form>_<precision>.npz (pred, pred_flip, labels: the
    MUST-09 format); for a classifier, the predicted class instead (flip.json);
 5. bitwise check of the even/odd form on --bitwise-images images (bitwise.json);
 6. A09-1, A09-2 and A09-3 (metrics.json), summary.md and meta.json (mirror.final.prereg.write_final_meta).
The script calls mirror.final.prereg.require_prereg() before touching the GPU (dry runs only record the status).
"""
from __future__ import annotations

import sys

import torch

DEFAULT_GPU_FRAC = 0.25

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.data import normalize  # noqa: E402
from mirror.final.audit import exactness as EX  # noqa: E402
from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "A09"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="healed run directory (spec.json, checkpoint)")
    p.add_argument("--step", type=int, default=None, help="checkpoint step for load_student (default: the last)")
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name for --span-structure")
    p.add_argument("--span-structure", default=None, help="SpanStructure file for the zero-shot full conversion")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--n-images", type=int, default=1000, help="validation images of the per-component audit")
    p.add_argument("--batch-size", type=int, default=16, help="images per audit batch (run with their mirrors)")
    p.add_argument("--forms", default="dense,even_odd")
    p.add_argument("--bf16-weights", action="store_true", help="also audit bf16 weights and inputs")
    p.add_argument("--align", type=int, default=8, help="alignment of the even/odd form (as timed)")
    p.add_argument("--no-original", action="store_true", help="skip the original model's floor")
    p.add_argument("--a09-2-floor", default="exact", choices=["cast", "exact"],
                   help="A09-2 floor: the float64 re-projected model (exact, pre-registered) or the same stored "
                        "weights in float64 (cast, diagnostic); both are reported")
    p.add_argument("--flip-queries", type=int, default=0, help="validation queries of the flip check (0: all)")
    p.add_argument("--flip-bank-per-class", type=int, default=0, help="bank images per class (0: the full bank)")
    p.add_argument("--flip-precisions", default="float32,bf16_autocast")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--knn-batch-size", type=int, default=256)
    p.add_argument("--knn-impl", default="builtin", choices=["builtin", "knn_fast"],
                   help="k-NN arithmetic: mirror.final.audit.exactness.make_knn_fn or the evaluation agent's "
                        "mirror.final.eval.knn_fast.knn_vote (both as mirror.knn.knn_predict, with vote margins)")
    p.add_argument("--bitwise-images", type=int, default=16)
    p.add_argument("--no-bitwise", action="store_true")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default results/A09/<model>/<variant> (dry run: tiny_out/A09)")
    p.add_argument("--force", action="store_true", help="redo every step")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny random model, synthetic images")
    p.add_argument("--dry-model", default="dinov2_reg", choices=["dinov2_reg", "deit3"],
                   help="tiny model family of the dry run (registers paired, or DeiT-III with its classifier)")
    a = p.parse_args(argv)
    if a.dry_run:
        a.n_images = min(a.n_images, 4)
        a.batch_size = min(a.batch_size, 2)
        a.bitwise_images = min(a.bitwise_images, 2)
    if sum(bool(x) for x in (a.run, a.span_structure, a.dry_run)) != 1:
        p.error("give exactly one of --run, --span-structure or --dry-run")
    return a


# ----------------------------------------------------------------------------------------------
# Model and structure
# ----------------------------------------------------------------------------------------------
def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_run(a, device) -> tuple:
    """(adapter, model, AuditStructure, provenance) of a healed run."""
    run = _resolve(a.run)
    try:
        from mirror.final.heal.students import load_student    # the heal agent's loader (final phase)
        adapter, model, info = load_student(run, step=a.step, device=device)
        loader = "mirror.final.heal.students.load_student"
    except ImportError:
        from mirror.heal_structured import load_healed_model
        adapter, model, info = load_healed_model(run, device)
        loader = "mirror.heal_structured.load_healed_model"
    info = info or {}
    if adapter is None:
        from mirror.heal_structured import build_adapter_from_spec
        adapter = build_adapter_from_spec(info["spec"]["adapter"], device)
    st = EX.AuditStructure.from_info(info, source=str(run))
    return adapter, model, st, {"run": str(run), "loader": loader, "method": info.get("method"),
                                "step": a.step, "family": info.get("family")}


def load_zero_shot(a, device) -> tuple:
    from mirror.adapters_timm import build_model_adapter
    from mirror.m1 import build_vexact
    from mirror.mirror_family import load_span_structure

    adapter = build_model_adapter(a.model, device, img_size=a.img_size)
    path = _resolve(a.span_structure)
    ss = load_span_structure(path)
    L = adapter.depth
    if ss.depth != L:
        raise SystemExit(f"{path}: structure for {ss.depth} blocks, the model has {L}")
    parts = ss.exact_parts("mirror_exact", L)
    model, binfo = build_vexact(adapter, parts["A"], list(range(L)), parts["perms"], parts["heads"],
                                reg_perm=parts["reg_perm"], even_head=True)
    st = EX.AuditStructure(A=parts["A"], perms=parts["perms"], maps=binfo["maps_after"], reg_perm=parts["reg_perm"],
                           source={"span_structure": str(path)})
    import hashlib
    return adapter, model, st, {"span_structure": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "fit_seed": ss.fit_seed, "structure_model": ss.model}


def load_dry(a) -> tuple:
    """A tiny random model (float32, like a stored model) with a random exact structure, full conversion."""
    from mirror.attention import attn_dims
    from mirror.final import adapters as FA
    from mirror.m1 import build_vexact, random_head_structure, random_involution_fixing_ones
    from mirror.matching import random_pairing

    make = {"dinov2_reg": lambda: FA.tiny_dinov2_reg(img_size=28, patch=7, dim=32, depth=2, heads=4, reg_tokens=4,
                                                       dtype=torch.float32),
            "deit3": lambda: FA.tiny_deit3(img_size=28, patch=7, dim=32, depth=2, heads=4, num_classes=5,
                                           dtype=torch.float32)}
    adapter = make[a.dry_model]()
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 3, seed=a.seed + 3)
    perms = [random_pairing(h, h // 3, seed=a.seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2)[b % 2], seed=a.seed + 20 + b) for b in range(L)]
    R = int(adapter.n_prefix_tokens) - int(adapter.has_cls_token)
    reg = np.array([1, 0] + list(range(2, R)), dtype=np.int64) if R >= 2 else np.zeros(0, dtype=np.int64)
    model, info = build_vexact(adapter, A, list(range(L)), perms, maps, reg_perm=reg if len(reg) else None,
                               even_head=adapter.has_cls_token)
    st = EX.AuditStructure(A=A, perms=perms, maps=info["maps_after"], reg_perm=reg, source={"dry_run": a.dry_model})
    return adapter, model, st, {"dry_run": a.dry_model}


def classifier_head(model):
    """The DeiT-style classifier head (norm -> class token -> head), or None."""
    from mirror.final.adapters import deit_head_path

    head = getattr(model, "head", None)
    if not isinstance(head, torch.nn.Linear):
        return None
    try:
        deit_head_path(model)
    except ValueError:
        return None
    return head


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def protocol(a, adapter, device):
    if a.dry_run:
        from mirror.variants import TinyProtocol
        return TinyProtocol(img_size=int(adapter.img_size), num_classes=5, batch_size=8), 5
    from mirror.protocol import Protocol
    pin = torch.device(device).type == "cuda"
    return Protocol(_resolve(a.cache_dir), a.knn_batch_size, 16, a.num_workers, pin), 1000


def audit_batches(a, adapter, proto) -> tuple:
    """Normalised float64 images (the first image of each class, as MUST-02) in batches, and their indices."""
    from mirror.data import make_loader

    if a.dry_run:
        idx = np.asarray(proto.query_idx)[:a.n_images]
    else:
        n_req = int(math.ceil(a.n_images / 1000.0)) * 1000
        idx = np.asarray(proto.query_loader(n_req).dataset.indices)[:a.n_images]
    loader = make_loader(proto.val, proto.val_labels, idx, a.batch_size, 0, False)
    mean, std = getattr(adapter, "mean", (0.485, 0.456, 0.406)), getattr(adapter, "std", (0.229, 0.224, 0.225))
    xs = [normalize(imgs, torch.float64, mean, std) for imgs, _ in loader]
    return xs, idx


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")
    tmp.replace(path)


def variant_name(a) -> str:
    if a.dry_run:
        return f"dry_{a.dry_model}"
    if a.run:
        r = _resolve(a.run)
        return f"{r.parent.name}_{r.name}" + (f"_step{a.step}" if a.step is not None else "")
    return "zeroshot"


def flip_checks(a, adapter, model, st, head, proto, n_cls, device, out: Path) -> dict:
    """Per form and precision: k-NN flip consistency (or the classifier's predicted class) with feature statistics."""
    import copy

    from mirror.data import make_loader, make_splits

    path = out / "flip.json"
    res = json.loads(path.read_text(encoding="utf-8")) if (path.exists() and not a.force) else {}
    mean, std = getattr(adapter, "mean", (0.485, 0.456, 0.406)), getattr(adapter, "std", (0.229, 0.224, 0.225))

    def norm(imgs):
        return normalize(imgs, torch.float32, mean, std)

    if a.dry_run:
        bank = proto.bank_loader()
        queries = proto.query_loader(a.flip_queries or None)
    else:
        rows = np.asarray(proto.index["bank_rows"])
        if a.flip_bank_per_class:
            sub = make_splits(proto.train_labels[rows], {"b": int(a.flip_bank_per_class)})["b"]
            bank = make_loader(proto.train, proto.train_labels, rows[sub], a.knn_batch_size, a.num_workers, True)
        else:
            bank = proto.bank_loader()
        queries = proto.query_loader(a.flip_queries or None)
    m32 = copy.deepcopy(model).float().eval()
    forms = [f for f in a.forms.split(",") if f]
    fv = EX.build_even_odd(m32, st, align=a.align) if "even_odd" in forms else None
    knn_dir = out / "knn"
    knn_dir.mkdir(exist_ok=True)
    for form in forms:
        base = m32 if form == "dense" else fv
        if base is None:
            continue
        for prec in [p for p in a.flip_precisions.split(",") if p]:
            key = f"{form}/{prec}"
            if key in res:
                continue
            if prec.startswith("bf16") and torch.device(device).type != "cuda" and not a.dry_run:
                continue
            if form == "dense":
                feat = (lambda x, m=base: head(m.norm(adapter.forward_tokens(x, m))[:, 0])) if head is not None else \
                    (lambda x, m=base: adapter.cls_feature(x, m))
            else:
                feat = (lambda x, m=base: head(m(x))) if head is not None else (lambda x, m=base: m(x))
            t0 = time.time()
            if head is not None:
                r = EX.feature_flip_check(feat, queries, norm, precision=prec, device=device, classifier=True)
            else:
                bf, by = EX.compute_features(feat, bank, norm, prec, device)
                if a.knn_impl == "knn_fast":
                    import torch.nn.functional as Fn
                    from mirror.final.eval.knn_fast import knn_vote
                    bank_u, bank_y = Fn.normalize(bf.float(), dim=-1), by.to(bf.device)
                    knn = (lambda q, bu=bank_u, byy=bank_y: knn_vote(bu, byy, Fn.normalize(q.float().to(bu.device), dim=-1),
                                                                     n_cls, a.knn_k, a.knn_temperature))
                else:
                    knn = EX.make_knn_fn(bf, by.to(bf.device), n_cls, a.knn_k, a.knn_temperature)
                r = EX.feature_flip_check(feat, queries, norm, precision=prec, device=device, knn_fn=knn)
                r["n_bank"] = int(bf.shape[0])
            if "pred" in r:
                np.savez(knn_dir / f"{form}_{prec}.npz", pred=r.pop("pred"), pred_flip=r.pop("pred_flip"),
                         labels=r.pop("labels"))
            r["seconds"] = round(time.time() - t0, 1)
            r["kind"] = "classifier (predicted class)" if head is not None else f"k-NN (cosine, k, T; {a.knn_impl})"
            res[key] = r
            log(f"A09 flip {key}: {r.get('flip_consistency')}% ({r.get('n_disagree')} differ), features identical "
                f"bit for bit for {100 * r['bitwise_identical_share']:.2f}% of images")
            _write_json(path, res)
    return res


def write_summary(out: Path, scores: dict, rows: list, weights: list, flip: dict, bitwise: dict | None,
                  meta: dict) -> Path:
    def f(x, nd=2):
        return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}e}"

    s1, s2, s3 = scores["A09-1"], scores["A09-2"], scores["A09-3"]
    orig = [r for r in rows if r["form"] == "original" and r["precision"] == "float32"
            and math.isfinite(r["dev_rel_tensor_max"])]
    of = max(orig, key=lambda r: r["dev_rel_tensor_max"]) if orig else None
    wmax = max((w["rel_residual"] for w in weights if math.isfinite(w["rel_residual"])), default=float("nan"))
    lines = [f"# A09 exactness audit: {meta['variant']}", "",
             f"Model {meta['model']}; {meta['n_images']} validation images; forms {', '.join(meta['forms'])}; "
             f"commit {meta['git'][:10]}. Levels: E1 exact in real arithmetic (a theorem, checked in float64 on "
             "weights re-projected in float64), E2 numerically exact (error no larger than the model's own "
             "rounding), E3 bitwise identical.", "",
             f"- **E1 (A09-1): {s1.get('verdict')}.** Largest relative equivariance error in float64 "
             f"{f(s1.get('max_rel_error'))} at {s1.get('where')} (bar 1e-10).",
             f"- **E2 (A09-2): {s2.get('verdict')}.** Float32 error within 2x the model's own float32-versus-float64 "
             f"deviation ({s2.get('floor')} floor) at {s2.get('n_tensors', 0) - s2.get('n_tensors_fail', 0)} of "
             f"{s2.get('n_tensors')} tensors (other floor: "
             f"{s2.get('n_tensors_fail_exact_floor', s2.get('n_tensors_fail_cast_floor'))} failing); "
             f"worst ratio {s2.get('worst_ratio') if s2.get('worst_ratio') is None else round(s2['worst_ratio'], 3)} "
             f"at {s2.get('worst_ratio_where')}; largest float32 error {f(s2.get('max_eq_error_fp32'))} at "
             f"{s2.get('max_eq_error_fp32_where')}; float32 flip consistency passes by form: "
             f"{s2.get('flip_consistency_fp32_pass')}.",
             "- **bfloat16 (A09-3, report):** " + "; ".join(
                 f"{k}: largest error {f(v['max_rel_error'])}, flip consistency {v['flip_consistency']}%"
                 for k, v in s3.get("by_form", {}).items()) + ".",
             f"- **Rounding floor of the original model:** largest float32-versus-float64 deviation "
             f"{f(of['dev_rel_tensor_max'] if of else None)}"
             + (f" at block {of['block']} {of['component']}" if of else "") + ".",
             f"- **Weight conditions:** largest relative residual {f(wmax)} "
             f"({weights[0]['dtype'] if weights else 'n/a'} storage)."]
    if bitwise is not None:
        lines += ["", "## Bitwise check of the even/odd form (E3)", ""] + [f"- {w}" for w in bitwise.get("why", [])]
    for k, v in flip.items():
        lines.append(f"- flip {k}: {v.get('flip_consistency')}% consistent, {v.get('n_disagree')} of "
                     f"{v.get('n_images')} differ; features identical bit for bit for "
                     f"{100 * v.get('bitwise_identical_share', 0):.2f}% of images")
    lines += ["", "Files: components.csv (one row per form, precision and component), weights.csv, flip.json, "
              "bitwise.json, metrics.json, meta.json."]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not a.dry_run else "cpu"))
    prereg = require_prereg(allow_missing=True if a.dry_run else None)     # before the GPU is touched
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)              # shared GPU: cap before allocating
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    if a.dry_run:
        adapter, model, st, prov = load_dry(a)
    elif a.run:
        adapter, model, st, prov = load_run(a, device)
    else:
        adapter, model, st, prov = load_zero_shot(a, device)
    model_tag = f"tiny_{a.dry_model}" if a.dry_run else (getattr(adapter, "model_name", None) or a.model)
    out = Path(a.out) if a.out else (REPO_ROOT / "tiny_out" / EXPERIMENT if a.dry_run
                                     else REPO_ROOT / "results" / EXPERIMENT / str(model_tag) / variant_name(a))
    out.mkdir(parents=True, exist_ok=True)
    adapter.model.to(device)
    model.to(device).eval()
    head = classifier_head(model)
    forms = [f for f in a.forms.split(",") if f]
    log(f"{EXPERIMENT}: {model_tag} {variant_name(a)} on {device}; forms {forms}; A* d+ {st.d_plus}, "
        f"converted blocks 1..{st.k}; classifier head: {head is not None}; out {out}")
    proto, n_cls = protocol(a, adapter, device)

    # 2. weight conditions
    wpath = out / "weights.csv"
    import pandas as pd
    if wpath.exists() and not a.force:
        weights = pd.read_csv(wpath).to_dict(orient="records")
    else:
        weights = EX.weight_conditions(adapter, model, st)
        pd.DataFrame(weights).to_csv(wpath, index=False)

    # 3. per-component audit
    cpath = out / "components.csv"
    xs, idx = audit_batches(a, adapter, proto)
    np.save(out / "val_indices.npy", np.asarray(idx, dtype=np.int64))
    if cpath.exists() and not a.force:
        rows = pd.read_csv(cpath).to_dict(orient="records")
        for r in rows:
            r["block"] = None if pd.isna(r["block"]) else int(r["block"])
            for flag in ("a09_2_pass", "a09_2_pass_exact_floor"):
                r[flag] = None if pd.isna(r.get(flag)) else bool(r[flag])
        notes = []
    else:
        precisions = list(EX.DEFAULT_PRECISIONS) + (["bf16"] if a.bf16_weights else [])
        res = EX.audit_forms(adapter, model, st, xs, forms=forms, precisions=precisions,
                             original=None if a.no_original else adapter.model, head=head, align=a.align,
                             device=device, log=log)
        rows, notes = res["rows"], res["notes"]
        df = pd.DataFrame(rows)
        for col in ("block", "code_index", "eq_worst_image"):
            df[col] = df[col].astype("Int64")
        df.insert(0, "model", str(model_tag))
        df["git_hash"] = git["git_hash"]
        df["seed"] = a.seed
        df.to_csv(cpath, index=False)

    # 4. flip checks
    flip = {} if a.no_flip else flip_checks(a, adapter, model, st, head, proto, n_cls, device, out)

    # 5. bitwise check of the even/odd form
    bitwise = None
    bpath = out / "bitwise.json"
    if "even_odd" in forms and not a.no_bitwise and int(st.k) == len(model.blocks):
        if bpath.exists() and not a.force:
            bitwise = json.loads(bpath.read_text(encoding="utf-8"))
        else:
            import copy
            fv = EX.build_even_odd(copy.deepcopy(model).float(), st, align=a.align)
            x = torch.cat(xs)[:a.bitwise_images].float()
            reg = st.reg_perm if len(st.reg_perm) else None
            from mirror.m1 import register_info
            bitwise = {}
            for name, mod in (("float32", fv), ("bf16", copy.deepcopy(fv).to(torch.bfloat16))):
                bitwise[name] = EX.bitwise_check(mod, x, dp=st.d_plus, grid_hw=adapter.grid_hw,
                                                 n_prefix=adapter.n_prefix_tokens, reg_perm=reg,
                                                 reg_start=int(register_info(adapter.model, adapter)["start"]),
                                                 head=None if head is None else copy.deepcopy(head).float())
            _write_json(bpath, bitwise)
            for w in bitwise["float32"]["why"]:
                log(f"A09 bitwise (float32): {w}")

    # 6. scores and outputs
    scores = EX.score_a09(rows, flip, floor=a.a09_2_floor)
    meta = {"variant": variant_name(a), "model": str(model_tag), "n_images": len(idx), "forms": forms,
            "git": git["git_hash"]}
    metrics = {"experiment": EXPERIMENT, "model": str(model_tag), "variant": variant_name(a), "scores": scores,
               "structure": {"d_plus": st.d_plus, "d_minus": st.d - st.d_plus, "k": int(st.k),
                             "reg_perm": [int(v) for v in st.reg_perm], "provenance": prov},
               "flip": flip, "bitwise_first_difference": None if bitwise is None else
               {k: v.get("first_difference") for k, v in bitwise.items()},
               "classifier_head": head is not None, "notes": notes, "n_images": len(idx),
               "prereg": {k: prereg.get(k) for k in ("ok", "reason", "sha256")} | {
                   "allowed_without_check": bool(prereg.get("allowed_without_check", False))},
               "tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                        "cudnn": bool(torch.backends.cudnn.allow_tf32)}}
    _write_json(out / "metrics.json", metrics)
    write_summary(out, scores, rows, weights, flip, None if bitwise is None else bitwise.get("float32"), meta)
    write_final_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"seed": a.seed},
                     models={str(model_tag): prov}, device=device, wall_seconds=time.time() - t_start,
                     extra={"n_images": len(idx), "forms": forms, "scores": {k: v.get("verdict") for k, v in scores.items()}})
    log(f"{EXPERIMENT}: A09-1 {scores['A09-1'].get('verdict')}, A09-2 {scores['A09-2'].get('verdict')}; wrote {out}")
    return {"out": str(out), "scores": scores, "rows": rows, "flip": flip, "bitwise": bitwise}


if __name__ == "__main__":
    main()
