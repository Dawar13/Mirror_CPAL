#!/usr/bin/env python
"""A03 discovery null controls: does the discovery pipeline invent structure? (FINAL_PHASE_PLAN.md v2, Section 4
A03; rule A03-1 in Section 6; definitions in mirror/controls.py.)

    python scripts/fp_a03_controls.py --model dinov2_vitb14_reg --out results/A03
    python scripts/fp_a03_controls.py --dry-run --out tiny_out/A03          # CPU, tiny equivariant network

Rows: mirror (the real left-right mirror, reference), C1 (shuffled correspondence), C2 (no re-alignment), C3 (a
random involutive pixel permutation inside every patch) and C4 (the random-operator null). For each fitted row:
  1. two paired passes over the fit set with the row's pairing (mirror.controls.fit_control): the per-layer
     operators and their d-, the span operators A*_k, the eigenvalue spectra of the symmetrised normalised
     cross-moments, the twin matching, the head maps and register pairing; the full SpanStructure is written to
     <out>/structures/ with mirror_family.save_span_structure (about 128 MB for DINOv2-B: keep it on the VM);
  2. the MUST-04 measures (mirror.controls.measure_structure): twin fraction, breaking shares of the real MLP
     weights and the null of --null-draws random structures; ratio to null per block range;
  3. the split-half diagnostic (operators fitted on one half of the fit images, pairings on the other; see
     mirror/controls.py), unless --no-split-half;
  4. zero-shot k-NN after full Mirror-exact surgery with the row's own structure (MUST-09 protocol: 100k bank, all
     50k validation queries, k = 20, T = 0.07, bf16; C4: the random_structure method of the reference structure,
     seed --c4-knn-seed, 1 as in MUST-07 and MUST-09), unless --no-knn.
C4 has no pass: its row is the reference's own null draws (leave-one-out ratio, twin fraction of the random
pairings scored with the reference's correlations).

Scoring: rule A03-1 (pre-registered) on the same-sample ratio, exactly as MUST-04 defines it; the same rule applied
to the split-half ratio is reported as a diagnostic only.

Outputs in --out: controls.csv (one row per control), layers.csv (per control and stream layer, plus "span":
d+, d-, spectrum summary), blocks.csv (per control and block: pairs, twin fractions, breaking shares, null mean and
interval), null_draws.csv, knn.csv, spectra.npz (sorted eigenvalues: <row>_layers [L + 1, d], <row>_span [d]),
structures/<model>_<row>_fit<seed>.npz, summary.md, metrics.json, meta.json (mirror.final.prereg.write_final_meta).
Resumable: <out>/cache/ keeps every finished stage with its configuration; a rerun repeats only stages whose
configuration changed.

Data roles: the fit set (ImageNet train, fit<seed> rows of cache/train224_uint8.npy by default) for every
statistic; validation only for the k-NN report; nothing is chosen on validation.

--dry-run: CPU, no data. A tiny random timm ViT (embed 32, 3 blocks, 2 registers, 56 px) is made exactly
mirror-equivariant with the existing conversion (mirror.controls.make_equivariant_network) and measured on
synthetic images, so the dry run is a known case: the reference finds the planted structure, C1 to C3 find no
twins.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import controls as CT  # noqa: E402
from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "A03"
GPU_FRAC_DEFAULT = 0.15          # of an H200 (141 GB): about 21 GB; DINOv2-B needs about 3 GB here (estimate)
DRY = {"embed_dim": 32, "depth": 3, "num_heads": 2, "reg_tokens": 2, "img_size": 56, "net_images": (4, 8, 11),
       "fit_images": (4, 24, 12), "null_draws": 5}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--images", default=None,
                   help="fit set, uint8 [N, H, W, 3] .npy (default: the fit<--fit-seed> rows of "
                        "<cache-dir>/train224_uint8.npy via train_index.npz)")
    p.add_argument("--fit-seed", type=int, default=0, help="fit role when --images is not given; recorded")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "A03"))
    p.add_argument("--controls", default="mirror,C1,C2,C3,C4", help="rows; the reference (mirror) is always run")
    p.add_argument("--null-draws", type=int, default=None, help="null draws (default 20; dry run 5)")
    p.add_argument("--null-seed", type=int, default=0, help="base seed of the null draws (MUST-04 used 0)")
    p.add_argument("--control-seed", type=int, default=0, help="seed of the C1 derangement and the C3 pixel involution")
    p.add_argument("--top-k", type=int, default=10, help="matching candidates per neuron (as MUST-04)")
    p.add_argument("--workers", default="auto", help="processes for the matchings (auto: CPU count - 1, at most L)")
    p.add_argument("--batch-size", type=int, default=16, help="fit images per paired batch (MUST-04 used 16)")
    p.add_argument("--range", default="1-6", help="block range of rule A03-1 (1-based)")
    p.add_argument("--extra-ranges", default="7-12", help="further ranges reported (skipped if beyond the depth)")
    p.add_argument("--no-split-half", action="store_true", help="skip the split-half diagnostic")
    p.add_argument("--no-knn", action="store_true", help="skip the zero-shot k-NN (needs the VM caches)")
    p.add_argument("--queries", type=int, default=None, help="subsample validation queries (default: all 50k)")
    p.add_argument("--knn-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--c4-knn-seed", type=int, default=1, help="random_structure seed of the C4 k-NN row")
    p.add_argument("--reference-structure", default=None,
                   help="canonical SpanStructure npz (e.g. results/MUST-07/structure/dinov2_vitb14_reg_s0.npz): "
                        "agreement of the refitted reference with it is reported")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--gpu-frac", type=float, default=GPU_FRAC_DEFAULT,
                   help="torch.cuda.set_per_process_memory_fraction (shared GPU)")
    p.add_argument("--seed", type=int, default=0, help="global torch / numpy seed (no result depends on it)")
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny equivariant network, synthetic data")
    return p.parse_args(argv)


def parse_rows(spec: str) -> list:
    rows = [r.strip() for r in str(spec).split(",") if r.strip()]
    bad = [r for r in rows if r not in CT.ROWS]
    if bad:
        raise SystemExit(f"unknown control(s) {bad}; choose from {CT.ROWS}")
    return ["mirror"] + [r for r in CT.ROWS if r in rows and r != "mirror"]


def _workers(a, L: int) -> int:
    if str(a.workers) == "auto":
        w = 1 if a.dry_run else max(1, (os.cpu_count() or 2) - 1)
    else:
        w = int(a.workers)
    return max(1, min(w, L))


def _fingerprint(images) -> str:
    h = hashlib.sha1()
    arr = np.asarray(images)
    h.update(str(arr.shape).encode())
    flat = arr.reshape(len(arr), -1)
    for s in range(0, len(arr), 64):
        h.update(np.ascontiguousarray(flat[s:s + 64]).tobytes())
    return h.hexdigest()[:16]


def _free(device) -> None:
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------------------------------------------
# Runtime: model, fit set and k-NN loaders
# ----------------------------------------------------------------------------------------------
class Runtime:
    def __init__(self, adapter, images, device, model_name: str, fit_desc: str, num_classes=None, loaders=None,
                 amp=None, extra=None):
        self.adapter, self.images, self.device = adapter, images, device
        self.model_name, self.fit_desc = model_name, fit_desc
        self.num_classes, self._loaders, self.amp = num_classes, loaders, amp
        self.fingerprint = _fingerprint(images)
        self.extra = dict(extra or {})

    def loaders(self):
        if self._loaders is None:
            raise SystemExit("k-NN needs the evaluation caches; pass --no-knn")
        return self._loaders()


def build_dry_runtime(a) -> Runtime:
    """Tiny exactly mirror-equivariant network (mirror.controls.make_equivariant_network) and synthetic images."""
    from mirror.data import synthetic_dataset
    from mirror.variants import TinyProtocol

    ad = CT.tiny_adapter(img_size=DRY["img_size"], embed_dim=DRY["embed_dim"], depth=DRY["depth"],
                         num_heads=DRY["num_heads"], reg_tokens=DRY["reg_tokens"], seed=0, dtype=torch.float64)
    nc, per, seed = DRY["net_images"]
    net_imgs, _ = synthetic_dataset(nc, per, DRY["img_size"], seed=seed)
    eq, planted, conv = CT.make_equivariant_network(ad, net_imgs, "cpu", model_name="tiny_equivariant")
    nc, per, seed = DRY["fit_images"]
    images, _ = synthetic_dataset(nc, per, DRY["img_size"], seed=seed)
    proto = TinyProtocol(img_size=DRY["img_size"])
    L = ad.depth
    extra = {"planted": {"d_plus_span": CT._d_plus(planted.A_star[L]),
                         "n_pairs": [int(planted.n_pairs(b)) for b in range(L)],
                         "n_images_for_the_network": int(len(net_imgs))},
             "planted_structure": planted}
    return Runtime(eq, images, torch.device("cpu"), "tiny_equivariant",
                   f"synthetic uint8 images ({len(images)}, seed {seed})", num_classes=proto.num_classes,
                   loaders=lambda: (proto.bank_loader(), proto.query_loader()), amp=None, extra=extra)


def build_vm_runtime(a, device) -> Runtime:
    from mirror.adapters_timm import build_model_adapter

    adapter = build_model_adapter(a.model, device=device)
    cache = Path(a.cache_dir)
    if a.images:
        images = np.asarray(np.load(a.images, mmap_mode="r"))
        fit_desc = f"{a.images} (all {len(images)} rows)"
    else:
        train = np.load(cache / "train224_uint8.npy", mmap_mode="r")
        with np.load(cache / "train_index.npz") as z:
            rows = np.asarray(z[f"fit{a.fit_seed}_rows"])
        images = np.asarray(train[rows])
        fit_desc = f"{cache / 'train224_uint8.npy'} rows fit{a.fit_seed} ({len(rows)} images, stored order)"
    loaders = None
    if not a.no_knn:
        def loaders():
            from mirror.protocol import Protocol

            proto = Protocol(cache, a.knn_batch_size, a.batch_size, a.num_workers, pin=device.type == "cuda")
            return proto.bank_loader(), proto.query_loader(a.queries)
    return Runtime(adapter, images, device, a.model, fit_desc, num_classes=1000, loaders=loaders, amp="bf16")


# ----------------------------------------------------------------------------------------------
# Stages (cached)
# ----------------------------------------------------------------------------------------------
def _structure_path(out: Path, rt: Runtime, row: str, fit_seed: int) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in rt.model_name)
    return out / "structures" / f"{safe}_{row}_fit{fit_seed}.npz"


def stage_fit(rt: Runtime, row: str, a, out: Path, n_draws: int):
    from mirror.mirror_family import load_span_structure, save_span_structure

    control = CT.make_control(row, a.control_seed)
    key = {"model": rt.model_name, "images": rt.fingerprint, "control": control.describe(), "top_k": int(a.top_k),
           "batch_size": int(a.batch_size), "null_draws": int(n_draws), "null_seed": int(a.null_seed),
           "fit_seed": int(a.fit_seed)}
    spath, fpath = _structure_path(out, rt, row, a.fit_seed), out / "cache" / f"{row}_fit.npz"
    if spath.exists() and fpath.exists():
        try:
            with np.load(fpath, allow_pickle=False) as z:
                meta = json.loads(str(z["meta_json"]))
                arrs = {k: z[k] for k in z.files if k != "meta_json"}
            if meta.get("key") == _jsonable(key):
                st = load_span_structure(spath)
                log(f"{row}: fit cached ({spath.name})")
                return CT.ControlFit(control=control, structure=st, d_plus=[int(x) for x in arrs["d_plus"]],
                                     d_minus=[int(x) for x in arrs["d_minus"]], eig_layers=arrs["eig_layers"],
                                     eig_span=arrs["eig_span"], scores=arrs["scores"],
                                     self_scores=arrs["self_scores"],
                                     null_twin=arrs["null_twin"] if arrs["null_twin"].size else None,
                                     info=meta["info"])
        except Exception as e:     # unreadable cache: refit
            log(f"{row}: cache unreadable ({e}); refitting")
    log(f"{row}: fitting ({control.label})")
    fit = CT.fit_control(rt.adapter, rt.images, rt.device, control, model_name=rt.model_name, fit_seed=a.fit_seed,
                         top_k=a.top_k, workers=_workers(a, rt.adapter.depth), batch_size=a.batch_size,
                         n_null=n_draws, null_seed_base=a.null_seed)
    save_span_structure(spath, fit.structure)
    info = {k: v for k, v in fit.info.items() if k not in ("partner", "pixel_sigma")}
    fpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fpath, d_plus=np.asarray(fit.d_plus), d_minus=np.asarray(fit.d_minus), eig_layers=fit.eig_layers,
             eig_span=fit.eig_span, scores=fit.scores, self_scores=fit.self_scores,
             null_twin=fit.null_twin if fit.null_twin is not None else np.zeros(0),
             partner=np.asarray(fit.info["partner"]),
             pixel_sigma=np.asarray(fit.info["pixel_sigma"]) if fit.info["pixel_sigma"] is not None else np.zeros(0),
             meta_json=np.asarray(json.dumps(_jsonable({"key": key, "info": info}))))
    fit.info = info
    log(f"{row}: span d- {fit.span_d_minus}, pass 1 {info['seconds_pass1']} s, matching {info['seconds_match']} s, "
        f"pass 2 {info['seconds_pass2']} s")
    _free(rt.device)
    return fit


def stage_measure(rt: Runtime, row: str, fit, a, out: Path, n_draws: int) -> dict:
    path = out / "cache" / f"{row}_measure.json"
    key = {"structure": _structure_path(out, rt, row, a.fit_seed).name, "images": rt.fingerprint,
           "control": fit.control.describe(), "null_draws": int(n_draws), "null_seed": int(a.null_seed)}
    old = _read_json(path) if path.exists() else None
    if old and old.get("key") == _jsonable(key):
        log(f"{row}: measures cached")
        return old["measure"]
    t0 = time.time()
    st = fit.structure
    ms = CT.measure_structure(rt.adapter.model, st.A_layers, st.perms, fit.scores, n_draws=n_draws, seed=a.null_seed)
    _write_json(path, {"key": key, "measure": ms, "seconds": round(time.time() - t0, 1)})
    log(f"{row}: breaking shares and {n_draws} null draws in {time.time() - t0:.1f} s")
    return _read_json(path)["measure"]


def stage_split(rt: Runtime, row: str, a, out: Path, n_draws: int, blocks: list) -> dict:
    path = out / "cache" / f"{row}_split.json"
    control = CT.make_control(row, a.control_seed)
    key = {"model": rt.model_name, "images": rt.fingerprint, "control": control.describe(), "blocks": blocks,
           "top_k": int(a.top_k), "batch_size": int(a.batch_size), "null_draws": int(n_draws),
           "null_seed": int(a.null_seed)}
    old = _read_json(path) if path.exists() else None
    if old and old.get("key") == _jsonable(key):
        log(f"{row}: split-half cached")
        return old["summary"]
    t0 = time.time()
    halves = CT.split_half_fit(rt.adapter, rt.images, rt.device, control, blocks, batch_size=a.batch_size,
                               top_k=a.top_k, workers=_workers(a, len(blocks)))
    summ = CT.split_half_summary(rt.adapter.model, halves, blocks, n_draws=n_draws, seed=a.null_seed)
    summ["seconds"] = round(time.time() - t0, 1)
    _write_json(path, {"key": key, "summary": summ})
    log(f"{row}: split-half ratio {summ['ratio_to_null']:.3f} in {summ['seconds']} s")
    _free(rt.device)
    return summ


def stage_knn(rt: Runtime, row: str, structure, a, out: Path) -> dict:
    path = out / "cache" / f"{row}_knn.json"
    method, seed = ("random_structure", int(a.c4_knn_seed)) if row == CT.NULL_ROW else ("mirror_exact", 0)
    src = "mirror" if row == CT.NULL_ROW else row
    key = {"structure": _structure_path(out, rt, src, a.fit_seed).name, "images": rt.fingerprint, "method": method,
           "seed": seed, "queries": a.queries, "amp": rt.amp}
    old = _read_json(path) if path.exists() else None
    if old and old.get("key") == _jsonable(key):
        log(f"{row}: k-NN cached")
        return old["knn"]
    t0 = time.time()
    bank, query = rt.loaders()
    res, pred = CT.knn_after_surgery(rt.adapter, structure, bank, query, rt.device, rt.num_classes, method=method,
                                     seed=seed, amp=rt.amp)
    res["seconds"] = round(time.time() - t0, 1)
    np.save(out / "cache" / f"{row}_knn_pred.npy", pred)
    _write_json(path, {"key": key, "knn": res})
    log(f"{row}: zero-shot k-NN {res['top1']:.2f} (flip consistency {res['flip_consistency']:.2f}) "
        f"in {res['seconds']} s")
    _free(rt.device)
    return res


def original_knn(rt: Runtime, a, git: dict) -> dict | None:
    """The original model's k-NN (cache/knn_original JSON when present, as every earlier script)."""
    from mirror import variants as V

    bank, query = rt.loaders()
    try:
        return V.original_knn(rt.adapter, bank, query, rt.device, rt.num_classes, rt.model_name, rt.amp,
                              a.cache_dir, a.queries, a.dry_run, git["git_hash"])
    except Exception as e:   # reported, never fatal
        return {"error": repr(e)}


# ----------------------------------------------------------------------------------------------
# Tables and report
# ----------------------------------------------------------------------------------------------
def _fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def _join(xs, nd=None) -> str:
    return " ".join((f"{x:.{nd}g}" if nd else str(x)) for x in xs)


def assemble(rt: Runtime, a, rows: list, fits: dict, measures: dict, splits: dict, knn: dict, ranges: dict,
             out: Path) -> dict:
    import pandas as pd

    L = rt.adapter.depth
    prim = a.range
    pblocks = ranges[prim]
    ref = fits["mirror"]
    summaries = {r: {name: CT.range_summary(measures[r], bl) for name, bl in ranges.items()} for r in measures}
    if CT.NULL_ROW in rows:
        summaries[CT.NULL_ROW] = {name: CT.null_row_summary(measures["mirror"], ref.null_twin, bl)
                                  for name, bl in ranges.items()}
    agree = {r: CT.agreement_with(fits[r].structure, ref.structure, pblocks, ref.scores) for r in fits}
    ctl_rows, layer_rows, block_rows, draw_rows = [], [], [], []
    for r in rows:
        if r == CT.NULL_ROW:
            fitlike, label = ref, CT.NULL_LABEL
        else:
            fitlike, label = fits[r], fits[r].control.label
        row = {"control": r, "label": label, "n_images": int(len(rt.images)),
               "d_minus_span": fitlike.span_d_minus, "d_plus_span": fitlike.span_d_plus,
               "d_minus_layers": _join(fitlike.d_minus), "d_plus_layers": _join(fitlike.d_plus)}
        if r != CT.NULL_ROW:
            sp = CT.spectrum_summary(fitlike.eig_span)
            row.update({"span_abs_sum": sp["abs_sum"], "span_signed_sum": sp["signed_sum"],
                        "span_neg_mass_share": sp["neg_mass_share"], "span_participation": sp["participation"],
                        "span_top10": _join(sp["top_abs"], 4)})
        for name in ranges:
            s = summaries[r][name]
            row[f"twin_fraction_t50_{name}"] = s.get("twin_fraction_t50")
            row[f"ratio_to_null_{name}"] = s.get("ratio_to_null")
            if r == CT.NULL_ROW:
                row[f"ratio_lo_{name}"], row[f"ratio_hi_{name}"] = s.get("ratio_lo"), s.get("ratio_hi")
            else:
                for key in ("s_mlp_mean", "s_mlp_null_mean", "s_mlp_null_lo", "s_mlp_null_hi",
                            "s_fc1_ratio_to_null", "s_fc2_ratio_to_null", "frac_2"):
                    row[f"{key}_{name}"] = s.get(key)
        if r in splits:
            row[f"ratio_to_null_split_{prim}"] = splits[r]["ratio_to_null"]
            row[f"twin_fraction_t50_split_{prim}"] = splits[r]["twin_fraction_t50"]
        if r in agree:
            row.update({"operator_agreement_with_mirror": agree[r]["operator_agreement"],
                        "span_agreement_with_mirror": agree[r]["span_agreement"],
                        f"pair_recovery_{prim}": agree[r]["pair_recovery"],
                        f"twin_recovery_{prim}": agree[r]["twin_recovery"]})
        if r in knn:
            row.update({"knn_method": knn[r]["method"], "knn_seed": knn[r]["seed"], "knn_top1": knn[r]["top1"],
                        "knn_flip_consistency": knn[r]["flip_consistency"]})
        ctl_rows.append(row)
    for r, f in fits.items():
        for l in range(L + 1):
            sp = CT.spectrum_summary(f.eig_layers[l])
            layer_rows.append({"control": r, "layer": l, "d_plus": f.d_plus[l], "d_minus": f.d_minus[l],
                               **{k: sp[k] for k in ("abs_sum", "signed_sum", "neg_mass_share", "participation")},
                               "top10": _join(sp["top_abs"], 4)})
        sp = CT.spectrum_summary(f.eig_span)
        layer_rows.append({"control": r, "layer": "span", "d_plus": f.span_d_plus, "d_minus": f.span_d_minus,
                           **{k: sp[k] for k in ("abs_sum", "signed_sum", "neg_mass_share", "participation")},
                           "top10": _join(sp["top_abs"], 4)})
        block_rows += [{"control": r, **b} for b in measures[r]["blocks"]]
        draw_rows += [{"control": r, **d} for d in measures[r]["draws"]]
    pd.DataFrame(ctl_rows).to_csv(out / "controls.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(out / "layers.csv", index=False)
    pd.DataFrame(block_rows).to_csv(out / "blocks.csv", index=False)
    pd.DataFrame(draw_rows).to_csv(out / "null_draws.csv", index=False)
    if knn:
        pd.DataFrame([{"control": r, **v} for r, v in knn.items()]).to_csv(out / "knn.csv", index=False)
    np.savez(out / "spectra.npz", **{f"{r}_layers": f.eig_layers for r, f in fits.items()},
             **{f"{r}_span": f.eig_span for r, f in fits.items()})
    # PREREG_FINAL_PHASE.md Section 2.4: A03-1 is scored on the split-half ratio (operators fitted on one half of
    # the fit images, pairings on the other) with the same-sample twin fraction; without split halves it is PENDING.
    # The same-sample ratio (MUST-04 as computed so far) is reported as a diagnostic.
    same_sample = CT.score_a03({r: summaries[r][prim] for r in summaries}, prim)
    same_sample["note"] = ("diagnostic (not the pre-registered score): rule A03-1 with the same-sample MUST-04 ratio, "
                           "which is biased below 1 for any fitted structure")
    score = CT.score_a03({r: {"twin_fraction_t50": summaries[r][prim].get("twin_fraction_t50"),
                              "ratio_to_null": (splits.get(r) or {}).get("ratio_to_null")} for r in summaries}, prim)
    score["ratio_statistic"] = "split-half ratio to null (PREREG_FINAL_PHASE.md Section 2.4)"
    return {"summaries": summaries, "agreement": agree, "score": score, "split_score": same_sample,
            "controls": ctl_rows}


def write_summary(path: Path, rt: Runtime, a, rows: list, res: dict, splits: dict, knn: dict, orig: dict | None,
                  ranges: dict, extra: dict) -> None:
    prim = a.range
    sc = res["score"]
    blocks_txt = f"blocks {ranges[prim][0] + 1} to {ranges[prim][-1] + 1}"
    lines = [f"# A03 discovery null controls: {rt.model_name}", "",
             f"**Rule A03-1 (pre-registered): {sc['status']}.** For each of C1, C2 and C3: twin fraction in "
             f"{blocks_txt} at most {CT.A03_TWIN_MAX} and split-half ratio to null at least {CT.A03_RATIO_MIN} "
             f"(PREREG_FINAL_PHASE.md Section 2.4).", ""]
    for c, v in sc["controls"].items():
        lines.append(f"- {c}: {v['status']} (twin fraction {_fmt(v['twin_fraction_t50'])}, split-half ratio to "
                     f"null {_fmt(v['ratio_to_null'])})")
    if res.get("split_score"):
        ss = res["split_score"]
        lines += ["", f"Diagnostic, not the pre-registered score: the same rule with the same-sample MUST-04 ratio "
                  f"gives {ss['status']} ("
                  + ", ".join(f"{c} {_fmt(v['ratio_to_null'])}" for c, v in ss["controls"].items()) + ")."]
    lines += ["", f"Fit set: {rt.fit_desc}. Range: {blocks_txt} (1-based). Null: {extra['null_draws']} draws, "
              f"base seed "
              f"{a.null_seed}. Control seed {a.control_seed}.", "",
              f"| control | twin fraction | ratio to null | split-half ratio | d- of A* | spectrum sum abs | "
              f"odd share | zero-shot k-NN |", "|---|---|---|---|---|---|---|---|"]
    for row in res["controls"]:
        r = row["control"]
        lines.append(f"| {r} | {_fmt(row.get(f'twin_fraction_t50_{prim}'))} | "
                     f"{_fmt(row.get(f'ratio_to_null_{prim}'))} | "
                     f"{_fmt(row.get(f'ratio_to_null_split_{prim}'))} | {row['d_minus_span']} | "
                     f"{_fmt(row.get('span_abs_sum'))} | {_fmt(row.get('span_neg_mass_share'))} | "
                     f"{_fmt(row.get('knn_top1'), 2)} |")
    if orig and "top1" in orig:
        lines += ["", f"Original model k-NN: {orig['top1']:.2f} ({orig.get('source', '')})."]
    if extra.get("planted"):
        p = extra["planted"]
        lines += ["", f"Dry run: the network is exactly mirror-equivariant by construction (planted A* with d+ "
                  f"{p['d_plus_span']}, twin pairs per block {p['n_pairs']}), so the reference row must find it."]
    if extra.get("reference_check"):
        lines += ["", f"Reference against {a.reference_structure}: {json.dumps(extra['reference_check'])}."]
    lines += ["", "## How to read this", "",
              "- Twin fraction: share of hidden units in a matched pair whose symmetrised correlation is at least "
              "0.5, mean over the range's blocks. Ratio to null: mean breaking share of the real MLP weights under "
              "the row's fitted structure, divided by the mean over the same blocks of the null's mean (random "
              "involutions with the fitted d+, random pairings with the fitted cycle type); 1 means no better than "
              "random. Both exactly as MUST-04.",
              "- C4 is the null itself: its ratio is each draw against the mean of the other draws, and its twin "
              "fraction scores the random pairings with the reference's correlations.",
              "- The same-sample ratio of any fitted structure sits below 1 even without a mirror relation, because "
              "the matching and the operator are fitted to the same cross-moments (about 0.88 on the synthetic "
              "equivariant network of the CPU tests). The split-half ratio fits the operators on one half of the "
              "fit images and the pairings on the other; it removes that effect.",
              "- Prefix tokens (CLS, registers) keep their index in every row, and C1 keeps the true re-alignment, "
              "so any part of the tokens that does not depend on the image stays mirror paired in C1; C2 and C3 "
              "keep features that do not depend on position (C2) or on the pixel order inside a patch (C3).",
              "- d- and the spectrum: an involution fit always returns an involution, so d- alone does not show "
              "structure. The sum of absolute eigenvalues of the span matrix (1 for an exact symmetry at every "
              "layer) and the odd share (absolute mass on negative eigenvalues) do.",
              "- Zero-shot k-NN: full Mirror-exact surgery with the row's own structure (C4: the random-structure "
              "method), no healing."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------
def run(a, rt: Runtime, out: Path, prereg: dict, t_start: float) -> dict:
    git = git_info()
    rows = parse_rows(a.controls)
    n_draws = int(a.null_draws if a.null_draws is not None else (DRY["null_draws"] if a.dry_run else 20))
    if CT.NULL_ROW in rows and n_draws < 2:
        raise SystemExit("C4 needs at least 2 null draws")
    L = rt.adapter.depth
    ranges = {}
    for name in [a.range] + [x.strip() for x in str(a.extra_ranges).split(",") if x.strip()]:
        bl = CT.range_blocks(name, L)
        lo, _, hi = name.partition("-")
        if bl and (name == a.range or int(hi or lo) <= L):
            ranges[name] = bl
    if a.range not in ranges:
        raise SystemExit(f"range {a.range} has no block in a {L}-block model")
    log(f"A03 on {rt.model_name}: rows {rows}, {len(rt.images)} fit images, depth {L}, range {a.range} "
        f"(blocks {[b + 1 for b in ranges[a.range]]}), {n_draws} null draws, device {rt.device}")
    timings = {}
    fits, measures, splits, knn = {}, {}, {}, {}
    for r in [x for x in rows if x in CT.CONTROL_ROWS]:
        t0 = time.time()
        fits[r] = stage_fit(rt, r, a, out, n_draws)
        measures[r] = stage_measure(rt, r, fits[r], a, out, n_draws)
        if not a.no_split_half:
            splits[r] = stage_split(rt, r, a, out, n_draws, ranges[a.range])
        timings[f"structure_{r}"] = round(time.time() - t0, 1)
    orig = None
    if not a.no_knn:
        for r in rows:
            t0 = time.time()
            src = fits["mirror"] if r == CT.NULL_ROW else fits[r]
            knn[r] = stage_knn(rt, r, src.structure, a, out)
            timings[f"knn_{r}"] = round(time.time() - t0, 1)
        orig = original_knn(rt, a, git)
    res = assemble(rt, a, rows, fits, measures, splits, knn, ranges, out)
    extra = {"null_draws": n_draws, "planted": rt.extra.get("planted")}
    if a.reference_structure:
        from mirror.mirror_family import load_span_structure

        canon = load_span_structure(a.reference_structure)
        ag = CT.agreement_with(fits["mirror"].structure, canon, ranges[a.range])
        ag.update({"d_plus_span_refit": fits["mirror"].span_d_plus,
                   "d_plus_span_canonical": CT._d_plus(canon.A_star[L])})
        extra["reference_check"] = ag
    metrics = {
        "experiment": EXPERIMENT, "model": rt.model_name, "fit_set": rt.fit_desc, "n_fit_images": int(len(rt.images)),
        "images_fingerprint": rt.fingerprint, "fit_seed": int(a.fit_seed), "depth": L, "d": rt.adapter.embed_dim,
        "h": rt.adapter.mlp_hidden, "rows": rows, "ranges": {k: [b + 1 for b in v] for k, v in ranges.items()},
        "null_draws": n_draws, "null_seed": int(a.null_seed), "control_seed": int(a.control_seed),
        "controls": {r: (CT.make_control(r, a.control_seed).describe() if r != CT.NULL_ROW
                         else {"label": CT.NULL_LABEL}) for r in rows},
        "summaries": res["summaries"], "split_half": splits, "agreement_with_mirror": res["agreement"],
        "knn": knn, "knn_original": orig, "scores": {"A03-1": res["score"]},
        "diagnostics": {"A03-1_same_sample": res["split_score"]},
        "fits": {r: {"d_plus_layers": f.d_plus, "d_minus_layers": f.d_minus, "d_plus_span": f.span_d_plus,
                     "d_minus_span": f.span_d_minus, "spectrum_span": CT.spectrum_summary(f.eig_span),
                     "n_pairs": [int(f.structure.n_pairs(b)) for b in range(L)], "info": f.info}
                 for r, f in fits.items()},
        "definitions": {
            "twin_fraction_t50": "mean over the range's blocks of the share of hidden units in a matched pair whose "
                                 "symmetrised correlation is at least 0.5 (MUST-04)",
            "ratio_to_null": "mean over the range's blocks of (s_fc1 + s_fc2) / 2 under the fitted per-layer "
                             "structure, divided by the mean over the same blocks of its null mean (MUST-04)",
            "ratio_to_null_split": "the same with the operators fitted on one half of the fit images and the "
                                   "pairings on the other, both directions averaged (the statistic A03-1 scores, "
                                   "PREREG_FINAL_PHASE.md Section 2.4)",
            "C4": "the reference's null draws: ratio of each draw to the mean of the other draws (leave one out); "
                  "twin fraction of each random pairing scored with the reference's correlations",
            "spectrum": "eigenvalues of sym(Pc C_l Pc) / tr(G_l) per layer; span: their mean over layers 0..L",
            "blocks": "1-based (block 1 = code index 0)"},
        "extra": {k: v for k, v in extra.items() if v is not None},
        "timings_s": timings, "git": git, "dry_run": bool(a.dry_run),
    }
    _write_json(out / "metrics.json", metrics)
    write_summary(out / "summary.md", rt, a, rows, res, splits, knn, orig, ranges, extra)
    write_final_meta(out, experiment=EXPERIMENT, config=vars(a),
                     seeds={"fit_seed": a.fit_seed, "null_seed": a.null_seed, "null_draws": n_draws,
                            "control_seed": a.control_seed, "c4_knn_seed": a.c4_knn_seed, "global_seed": a.seed},
                     models={rt.model_name: ({"note": "tiny random timm ViT made exactly mirror-equivariant (dry run)"}
                                             if a.dry_run else
                                             {"registry": a.model,
                                              "note": "IDs and revisions: configs/models/*.yaml"})},
                     device=rt.device, wall_seconds=time.time() - t_start,
                     extra={"prereg_check": prereg, "fit_set": rt.fit_desc, "images_fingerprint": rt.fingerprint,
                            "timings_s": timings, "verdict_A03-1": res["score"]["status"]})
    log(f"A03-1: {res['score']['status']}; outputs in {out}")
    return metrics


def main(argv=None, runtime: Runtime | None = None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    prereg = require_prereg(allow_missing=True if a.dry_run else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
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
    rt = runtime if runtime is not None else (build_dry_runtime(a) if a.dry_run else build_vm_runtime(a, device))
    return run(a, rt, out, prereg, t_start)


if __name__ == "__main__":
    main()
