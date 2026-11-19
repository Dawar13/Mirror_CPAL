#!/usr/bin/env python
"""A15a: SigLIP 2 B/16 projected with the activation-weighted ("whitened") projection, zero-shot, in the configuration
of the CORE-04 collapse (FINAL_PHASE_PLAN.md v2, item A15a; PREREG_FINAL_PHASE.md A15a: report the zero-shot k-NN
with whitened projection against the current collapse). The projection and its derivation are in
mirror/final/adapters.py (weighted_projection, build_whitened_vexact).

    python scripts/fp_a15a_siglip.py                                   # VM: pretrained SigLIP 2, the MUST-07 structure
    python scripts/fp_a15a_siglip.py --dry-run --out tiny_out/A15a     # CPU, tiny random SigLIP 2, synthetic images

Configuration (the collapse; results/CORE-04/surgery/rows_siglip2_b16.csv and info/siglip2_b16/*/mirror_exact_s0.json):
registry model siglip2_b16 (timm vit_base_patch16_siglip_224.v2_webli), structure
results/MUST-07/structure/siglip2_b16_s0.npz (fit seed 0: fit set 0, 2,000 train images), blocks 1 to 6 (k = 6) with
the span operator A*_6, sizes 6mlp (MLP only) and 6both (MLP and attention), embeddings projected, no even head; the
k-NN feature is the MAP-pooled embedding (adapter feature 'pooled'), 100k train bank, all 50k validation queries and
the 10k method-selection queries, k = 20, T = 0.07, bf16 autocast (scripts/must07_compressors.Evaluator).

Steps (rerunning skips finished models unless --force):
 1. adapter and structure (model name, fit seed and depth checked; sha256 recorded);
 2. input moments of every projected weight on the structure's fit set (fit0, no mirrors), float64;
 3. per size: the plain projection (mirror.mirror_family.build_mirror_variant, method mirror_exact: the CORE-04
    model) and the whitened one (mirror.final.adapters.build_whitened_vexact, same operator, pairings and head
    maps); span exactness of both on probe images; per-matrix weighted errors (weighted.csv);
 4. k-NN of the original and of every model: top-1 on validation and method selection, flip consistency (bf16),
    predictions in knn/<name>.npz;
 5. summary.csv (one row per model, with the CORE-04 values beside), metrics.json, summary.md and meta.json.
"""
from __future__ import annotations

import sys

import torch

DEFAULT_GPU_FRAC = 0.25

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.data import mirror, normalize_for  # noqa: E402
from mirror.final import adapters as FA  # noqa: E402
from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.knn import knn_predict  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "A15a"
STRUCTURE = "results/MUST-07/structure/siglip2_b16_s0.npz"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="siglip2_b16")
    p.add_argument("--structure", default=STRUCTURE)
    p.add_argument("--fit-seed", type=int, default=0)
    p.add_argument("--k", type=int, default=6, help="converted blocks 1..k (CORE-04: 6)")
    p.add_argument("--sizes", default="mlp,both", help="components per size: mlp (MLP only), both (MLP and attention)")
    p.add_argument("--damp", type=float, default=FA.DEFAULT_DAMP)
    p.add_argument("--no-weighted-patch", action="store_true", help="keep the plain projection of the patch embedding")
    p.add_argument("--queries", type=int, default=0, help="validation queries (0: all 50,000)")
    p.add_argument("--no-msel", action="store_true", help="skip the method-selection queries")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--probe-images", type=int, default=16)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--stats-batch-size", type=int, default=32)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--core04-rows", default=FA.CORE04_SIGLIP_ROWS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default results/A15a (dry run: tiny_out/A15a)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny random SigLIP 2, a structure fitted here")
    a = p.parse_args(argv)
    if a.dry_run:
        a.k = min(a.k, 2)
        a.probe_images = min(a.probe_images, 2)
    a.size_list = [s.strip() for s in a.sizes.split(",") if s.strip()]
    if set(a.size_list) - {"mlp", "both"}:
        p.error("--sizes takes mlp and/or both")
    return a


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


# ----------------------------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------------------------
def setup(a, device):
    """(adapter, span structure, protocol, provenance)."""
    from mirror import mirror_family as MF

    if a.dry_run:
        from mirror.variants import TinyProtocol

        from mirror.attention import attn_dims
        from mirror.m1 import random_head_structure, random_involution_fixing_ones
        from mirror.matching import random_pairing

        adapter = FA.tiny_siglip2(img_size=28, patch=7, dim=32, depth=3, heads=4, dtype=torch.float32)
        proto = TinyProtocol(img_size=28, num_classes=5, batch_size=16, stats_batch_size=8)
        L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
        H, dh = attn_dims(adapter.model.blocks[0])
        A = random_involution_fixing_ones(d, 19, seed=a.seed + 3)            # a planted structure (no fit needed)
        ss = MF.SpanStructure(model="tiny_siglip2", fit_seed=a.fit_seed, A_layers=[A] * (L + 1),
                              perms=[random_pairing(h, 40, seed=a.seed + b) for b in range(L)],
                              heads=[random_head_structure(H, dh, 1, seed=a.seed + 30 + b) for b in range(L)],
                              A_star={k: A for k in range(1, L + 1)},
                              reg_perm={k: np.zeros(0, dtype=np.int64) for k in range(1, L + 1)})
        return adapter, ss, proto, {"structure": "planted random structure (dry run)", "num_classes": 5}
    from mirror.adapters_timm import build_model_adapter
    from mirror.protocol import Protocol

    adapter = build_model_adapter(a.model, device)
    path = _resolve(a.structure)
    if not path.exists():
        raise SystemExit(f"{path} not found: upload the MUST-07 SigLIP 2 structure (fit seed {a.fit_seed})")
    ss = MF.load_span_structure(path)
    if ss.model != a.model or int(ss.fit_seed) != int(a.fit_seed) or ss.depth != adapter.depth:
        raise SystemExit(f"{path} is for {ss.model}, fit seed {ss.fit_seed}, depth {ss.depth}; this run is {a.model}, "
                         f"fit seed {a.fit_seed}, depth {adapter.depth}")
    proto = Protocol(_resolve(a.cache_dir), a.batch_size, a.stats_batch_size, a.num_workers,
                     torch.device(device).type == "cuda")
    prov = {"structure": str(path), "structure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "structure_meta": {k: ss.meta.get(k) for k in ("top_k", "register_margin", "n_pairs")}, "num_classes": 1000}
    return adapter, ss, proto, prov


def read_core04(a) -> dict:
    """The CORE-04 values of the collapse (results/CORE-04/surgery/rows_siglip2_b16.csv), keyed by size."""
    path = _resolve(a.core04_rows)
    if a.dry_run or not path.exists():
        return {}
    import pandas as pd

    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        if r["method"] in ("original", "mirror_exact", "svdllm"):
            out[f"{r['size']}/{r['method']}"] = {"val_top1": float(r["val_top1"]), "msel_top1": float(r["msel_top1"]),
                                                 "flip_consistency": float(r["flip_consistency"]),
                                                 "fit_seed": int(r["fit_seed"]), "source": str(path)}
    return out


# ----------------------------------------------------------------------------------------------
# k-NN
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def features(adapter, model, loader, device, flip=False):
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if torch.device(device).type == "cuda"
           else contextlib.nullcontext())
    fs, ys = [], []
    for imgs, y in loader:
        x = normalize_for(adapter, imgs.to(device, non_blocking=True))
        if flip:
            x = mirror(x)
        with ctx:
            f = adapter.cls_feature(x, model)
        fs.append(F.normalize(f.float(), dim=-1))
        ys.append(torch.as_tensor(y))
    return torch.cat(fs), torch.cat(ys).to(device)


def evaluate(a, adapter, model, proto, n_cls, device) -> dict:
    bf, by = features(adapter, model, proto.bank_loader(), device)
    qf, qy = features(adapter, model, proto.query_loader(a.queries or None), device)
    pred = knn_predict(bf, by, qf, n_cls, a.knn_k, a.knn_temperature)
    out = {"val_top1": 100.0 * float((pred == qy).float().mean()), "n_val": int(qf.shape[0]), "n_bank": int(bf.shape[0]),
           "pred": pred.cpu().numpy(), "labels": qy.cpu().numpy()}
    if not a.no_flip:
        qt, _ = features(adapter, model, proto.query_loader(a.queries or None), device, flip=True)
        pt = knn_predict(bf, by, qt, n_cls, a.knn_k, a.knn_temperature)
        out.update({"flip_consistency": 100.0 * float((pred == pt).float().mean()), "pred_flip": pt.cpu().numpy()})
    if not a.no_msel:
        mf, my = features(adapter, model, proto.msel_loader(), device)
        mp = knn_predict(bf, by, mf, n_cls, a.knn_k, a.knn_temperature)
        out.update({"msel_top1": 100.0 * float((mp == my).float().mean()), "n_msel": int(mf.shape[0])})
    return out


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not a.dry_run else "cpu"))
    prereg = require_prereg(allow_missing=True if a.dry_run else None)     # before the GPU is touched
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)              # shared GPU: cap before allocating
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(a.seed)
    git = git_info()
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out" if a.dry_run else "results") / EXPERIMENT
    out.mkdir(parents=True, exist_ok=True)
    (out / "knn").mkdir(exist_ok=True)
    import pandas as pd
    from mirror import mirror_family as MF
    from mirror.heal_vexact import span_exactness

    adapter, ss, proto, prov = setup(a, device)
    adapter.model.to(device)
    n_cls = int(prov["num_classes"])
    k = int(a.k)
    feat = FA.siglip_feature_path(adapter.model)
    log(f"{EXPERIMENT}: {a.model if not a.dry_run else 'tiny SigLIP 2'} on {device}, blocks 1..{k}, sizes {a.size_list}, "
        f"feature {feat['feature']}; out {out}")
    spath = out / "summary.csv"
    done = {} if (a.force or not spath.exists()) else {r["name"]: r for r in pd.read_csv(spath).to_dict(orient="records")}
    rows = list(done.values())
    core = read_core04(a)
    wrows = []
    exact = {}
    parts = ss.exact_parts("mirror_exact", k)
    probe = []
    for imgs, _ in proto.query_loader(None):                  # the first validation images only
        probe.append(normalize_for(adapter, imgs.to(device)))
        if sum(int(p.shape[0]) for p in probe) >= a.probe_images:
            break
    probe = torch.cat(probe)[:a.probe_images]

    def record(name, model, extra):
        t0 = time.time()
        ev = evaluate(a, adapter, model, proto, n_cls, device)
        np.savez(out / "knn" / f"{name}.npz", **{key: ev.pop(key) for key in ("pred", "pred_flip", "labels") if key in ev})
        row = {"name": name, **extra, **ev, "eval_s": round(time.time() - t0, 1), "git_hash": git["git_hash"],
               "seed": a.seed}
        rows.append(row)
        pd.DataFrame(rows).to_csv(spath, index=False)
        log(f"{EXPERIMENT} {name}: k-NN top-1 {ev['val_top1']:.2f}"
            + (f", method selection {ev['msel_top1']:.2f}" if "msel_top1" in ev else ""))

    if "original" not in done:
        record("original", adapter.model, {"size": "original", "projection": "none",
                                            "core04_val_top1": core.get("original/original", {}).get("val_top1")})
    need = [s for s in a.size_list if any(f"{k}{s}_{p}" not in done for p in ("plain", "whitened"))]
    moments = None
    if need:
        t0 = time.time()
        comps = ("mlp", "attn")
        moments = FA.collect_projection_moments(adapter, proto.fit_loader(a.fit_seed), device, range(k), comps)
        log(f"{EXPERIMENT}: input moments on fit set {a.fit_seed} ({moments['info']['n_images']} images) in "
            f"{time.time() - t0:.1f} s")
    for s in a.size_list:
        size = f"{k}{s}"
        comps = ("mlp",) if s == "mlp" else ("mlp", "attn")
        for proj in ("plain", "whitened"):
            name = f"{size}_{proj}"
            if name in done:
                continue
            t0 = time.time()
            if proj == "plain":
                model, info = MF.build_mirror_variant(adapter, "mirror_exact", range(k), comps, ss, seed=1)
                extra_info = {"params_total": info.get("params_total"), "budget_total": info.get("budget_total")}
            else:
                model, info = FA.build_whitened_vexact(adapter, parts["A"], range(k), parts["perms"], parts["heads"],
                                                       moments, reg_perm=parts["reg_perm"], components=comps,
                                                       even_head=False, damp=a.damp,
                                                       weighted_patch=not a.no_weighted_patch)
                for r in info["weighted"]:
                    wrows.append({"size": size, **r})
                extra_info = {}
            model.to(device).eval()
            ex = span_exactness(adapter, model, [probe], parts["A"], k,
                                parts["reg_perm"] if len(parts["reg_perm"]) else None)
            exact[name] = {"max_rel_inside": ex["max_rel_inside"], "pass": ex["pass"], "dtype": ex["dtype"]}
            ref = core.get(f"{size}/mirror_exact", {})
            record(name, model, {"size": size, "projection": proj, "components": "+".join(comps), "k": k,
                                 "d_plus": int(round((float(np.trace(parts["A"])) + parts["A"].shape[0]) / 2)),
                                 "span_exactness_max_rel": ex["max_rel_inside"], "build_s": round(time.time() - t0, 1),
                                 "core04_val_top1": ref.get("val_top1"), "core04_msel_top1": ref.get("msel_top1"),
                                 "core04_svdllm_val_top1": core.get(f"{size}/svdllm", {}).get("val_top1"),
                                 **extra_info})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if wrows:
        pd.DataFrame(wrows).to_csv(out / "weighted.csv", index=False)
    by = {r["name"]: r for r in rows}
    comparison = {}
    for s in a.size_list:
        size = f"{k}{s}"
        p, w = by.get(f"{size}_plain"), by.get(f"{size}_whitened")
        if p is not None and w is not None:
            comparison[size] = {"plain_val_top1": p["val_top1"], "whitened_val_top1": w["val_top1"],
                                "difference": w["val_top1"] - p["val_top1"],
                                "original_val_top1": by.get("original", {}).get("val_top1"),
                                "core04_collapse_val_top1": p.get("core04_val_top1")}
    metrics = {"experiment": EXPERIMENT, "rule": "A15a: report the zero-shot k-NN with whitened projection against "
                                                 "the current collapse (PREREG_FINAL_PHASE.md)",
               "comparison": comparison, "span_exactness": exact, "feature_path": feat, "provenance": prov,
               "core04": core, "k": k, "damp": a.damp, "weighted_patch": not a.no_weighted_patch,
               "prereg": {key: prereg.get(key) for key in ("ok", "reason", "sha256")}}
    (out / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    lines = [f"# A15a: SigLIP 2 with the activation-weighted projection (zero-shot, blocks 1 to {k})", "",
             "Same structure, operator, pairings and head maps as the CORE-04 collapse; only how each weight is "
             "projected changes (the equivariant matrix closest on the fit images' activations instead of in the "
             "plain Frobenius norm). With MLP and attention converted both projections are exactly equivariant over "
             "the converted span; with the MLP only, attention stays unprojected and neither is (as in CORE-04). "
             "The largest stream equivariance error over the span is in summary.csv (span_exactness_max_rel).", ""]
    for size, c in comparison.items():
        lines.append(f"- {size}: plain {c['plain_val_top1']:.2f}, whitened {c['whitened_val_top1']:.2f} k-NN top-1 "
                     f"(difference {c['difference']:+.2f}; original {c['original_val_top1']:.2f}"
                     + (f"; CORE-04 plain {c['core04_collapse_val_top1']:.2f}" if c.get("core04_collapse_val_top1")
                        is not None else "") + ")")
    lines += ["", "Files: summary.csv, weighted.csv (per-matrix weighted error, plain and whitened), knn/*.npz, "
              "metrics.json, meta.json."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_final_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"seed": a.seed, "fit_seed": a.fit_seed},
                     models={a.model: prov}, device=device, wall_seconds=time.time() - t_start,
                     extra={"comparison": comparison})
    log(f"{EXPERIMENT}: wrote {out}")
    return {"out": str(out), "rows": rows, "comparison": comparison, "exactness": exact}


if __name__ == "__main__":
    main()
