#!/usr/bin/env python
"""Track A zero-shot ablations A1 to A5 (PAPER_EXPERIMENTS.md, Track A; paper Table 3).

    python scripts/stage1_ablate.py --model dinov2_vitb14_reg --fit-run runs/x2/perlayer_reg_s0 \
        --ablation A1 --ks 6,9 --out runs/a/dinov2_vitb14_reg
    python scripts/stage1_ablate.py --tiny --ablation A1,A2,A3,A4,A5 --ks 1,2 --out tiny_out/a   # CPU

Every variant converts the MLPs of k blocks of a deep copy of the model and is scored with the
paper protocol (mirror/protocol.py: bank = 100 train images per class, queries = all 50k validation
images unless --queries, CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features for every
variant alike). The original model's k-NN is read from cache/knn_original/<model>_v2_bf16.json when
present (written by run_e1.py), otherwise computed and cached there.

Defaults for every variant: blocks 1..k, per-layer operators (fc1 with A_b, fc2 with A_(b+1)), the
fit run's mirror matching, alpha = 1. Each ablation changes one of these:
  A1 operator: per_layer, shared_span (one operator for the span layers 0..k, from sum C_l / tr(G_l)),
     shared_all (E1 design: plain sum of C_l over layers 1..12), identity (A = I), random_operator
     (matched d+ and d- per layer, A 1 = 1), plus the Track A baselines svd (rank matched per matrix
     to the per-layer mirror count) and random_structure (random_operator + random_pairing).
  A2 pairing: mirror, random_pairing (matched counts), self_twins, generic (plain cosine of fc1 rows,
     same number of pairs), hungarian (no involution constraint; projection
     (2W + PWA + P^T W A)/4, NOT exact for cycles longer than 2; flagged with params_exact = False)
     and hungarian_cycle (same permutation, exact projection onto {W : P W A = W}).
  A3 fit-set size: refit operators and twins from n fit images, n in 100, 500, 2000 (from fit seed
     s; nested subsets of a fixed class-spread order unless --a3-order first) and 6000 (fit seeds
     0 + 1 + 2 pooled, the largest set in the cache; the plan's 10,000 does not exist; run only with
     --fit-seed 0 and recorded with fit seed -1).
  A4 block selection: first k, last k, lowest breaking share first ((s1 + s2) / 2). Same k for all
     three; params_vs_mirror_first_k records each one's parameter count relative to first k.
  A5 projection strength: W <- W - alpha (W - W_eq), alpha in {0, 0.25, 0.5, 0.75, 1}.

Output: <out>/<ablation>_s<fit seed>.csv, one row per (ablation, variant, k, fit seed); a rerun
skips rows already present. <out>/<ablation>_s<fit seed>_meta.json holds provenance (git hash,
seed, arguments, fit-run checks). Generic and Hungarian pairings and A3 refits are cached under
<out>/cache/ so a resumed run does not recompute them. Several cells are the same model (A1
per_layer = A2 mirror = A4 first = A5 alpha1.00 at the same k and fit seed): each row carries a
content signature of its variant, and a variant whose signature was already scored in <out> (same
model and query set) reuses that result instead of being evaluated again; such rows name their
source in `reused_from`.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.08)  # shared GPU: cap before anything is allocated

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import variants as V  # noqa: E402
from mirror.data import make_loader  # noqa: E402
from mirror.knn import evaluate_knn  # noqa: E402
from mirror.utils import REPO_ROOT, env_info, git_info, log, set_seed  # noqa: E402

ABLATIONS = ("A1", "A2", "A3", "A4", "A5")
KEY = ["ablation", "variant", "k", "fit_seed"]
SEED = 0
RANDOM_VARIANTS = {("A1", "random_operator"), ("A1", "random_structure"), ("A2", "random_pairing")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--fit-run", default=None, help="run_e1.py output directory, e.g. runs/x2/perlayer_reg_s0")
    p.add_argument("--ablation", required=True, help="A1, A2, A3, A4, A5 or a comma list")
    p.add_argument("--ks", default="6,9", help="numbers of converted blocks")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--fit-seed", type=int, default=None, help="must agree with the fit run's fit seed")
    p.add_argument("--variants", default=None, help="comma list: run only these variants")
    p.add_argument("--a3-sizes", default=None, help="A3 fit-set sizes (default 100,500,2000,6000; tiny 4,10,30)")
    p.add_argument("--a3-order", choices=("spread", "first"), default="spread",
                   help="A3 subset order: spread (fixed permutation) or the literal first n rows (class-sorted)")
    p.add_argument("--queries", type=int, default=None, help="subsample validation queries (default: all)")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--workers", type=int, default=4, help="processes for the blossom matchings")
    p.add_argument("--top-k", type=int, default=10, help="matching candidates per neuron (as run_e1.py)")
    p.add_argument("--no-fix-ones", action="store_true", help="random involutions without A 1 = 1")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--stats-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT + synthetic data on CPU")
    return p.parse_args(argv)


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class Context:
    """Lazily computed inputs shared by several variants, cached in memory and under <out>/cache/."""

    def __init__(self, a, adapter, proto, struct, fit_seed, device, out: Path, max_k: int):
        self.a, self.adapter, self.proto, self.struct = a, adapter, proto, struct
        self.fit_seed, self.device, self.max_k = fit_seed, device, max_k
        self.cache = out / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.tag = struct.source
        self._mem: dict = {}
        self.notes: dict = {}

    # ---- cached block -> permutation dictionaries
    def _load_perms(self, path: Path, blocks):
        if not path.exists():
            return None
        with np.load(path) as z:
            if str(z["source"]) != self.tag or not set(blocks) <= set(z["blocks"].tolist()):
                return None
            return {int(b): np.asarray(p, dtype=np.int64) for b, p in zip(z["blocks"], z["perms"])}

    def _save_perms(self, path: Path, perms: dict, extra: dict):
        bl = sorted(perms)
        np.savez(path, blocks=np.asarray(bl), perms=np.stack([perms[b] for b in bl]).astype(np.int64),
                 source=np.asarray(self.tag), extra=np.asarray(json.dumps(extra, default=str)))

    def hungarian(self, blocks) -> dict:
        key = "hungarian"
        if key in self._mem and set(blocks) <= set(self._mem[key]):
            return self._mem[key]
        path = self.cache / f"hungarian_s{self.fit_seed}.npz"
        perms = self._load_perms(path, blocks)
        if perms is None:
            want = list(range(self.max_k))
            log(f"Hungarian: mirror-score pass over fit{self.fit_seed} for blocks 1..{self.max_k}")
            scores = V.collect_mirror_scores(self.adapter, self.proto.fit_loader(self.fit_seed), self.device, want)
            check = {}
            if self.struct.match_scores is not None:
                # The recomputed score at the stored twin of each neuron should equal the stored score.
                h = self.struct.h
                check = {"max_abs_diff_stored_match_scores": float(max(
                    np.abs(scores[b][np.arange(h), self.struct.perms[b]] - self.struct.match_scores[b]).max()
                    for b in want))}
                log(f"Hungarian: recomputed vs stored match scores, max abs diff "
                    f"{check['max_abs_diff_stored_match_scores']:.2e}")
            perms = {b: V.hungarian_permutation(scores[b]) for b in want}
            del scores
            self._save_perms(path, perms, check)
            self.notes["hungarian_score_check"] = check
        self._mem[key] = perms
        return perms

    def generic(self, blocks) -> dict:
        key = "generic"
        if key in self._mem and set(blocks) <= set(self._mem[key]):
            return self._mem[key]
        path = self.cache / f"generic_s{self.fit_seed}.npz"
        perms = self._load_perms(path, blocks)
        if perms is None:
            want = list(range(self.max_k))
            log(f"generic pairing: cosine matching for blocks 1..{self.max_k} ({self.a.workers} workers)")
            from mirror.surgery import fold_mlp

            W1 = [fold_mlp(self.adapter.model.blocks[b])["W1"] for b in want]
            got = V.generic_pairings(W1, [self.struct.n_pairs(b) for b in want], self.a.top_k,
                                     1 if self.a.tiny else self.a.workers)
            perms = dict(zip(want, got))
            self._save_perms(path, perms, {})
        self._mem[key] = perms
        return perms

    def shares(self) -> list:
        if "shares" not in self._mem:
            self._mem["shares"] = V.mlp_block_shares(self.adapter.model, self.struct)
        return self._mem["shares"]

    def a3_structure(self, n: int) -> V.Structure:
        rows, row_seed, desc = V.fit_subset_rows(self.proto, n, self.fit_seed, self.a.a3_order)
        key = f"a3_n{n}_s{row_seed}_{self.a.a3_order}"
        if key in self._mem:
            return self._mem[key]
        path = self.cache / f"{key}.npz"
        st = None
        if path.exists():
            st = V.load_structure_npz(path)
            if len(st.meta.get("matched_blocks", [])) < self.max_k:
                st = None
        if st is None:
            log(f"A3: refit from {n} images ({desc})")
            p = self.proto
            loader = make_loader(p.train, p.train_labels, rows, p.sbs, p.nw, p.pin)
            st = V.fit_structure(self.adapter, loader, self.device, match_blocks=range(self.max_k),
                                 top_k=self.a.top_k, workers=1 if self.a.tiny else self.a.workers)
            st.meta.update({"n_images": int(n), "fit_rows": desc, "row_fit_seed": row_seed})
            V.save_structure(path, st, with_moments=False)
            st = V.load_structure_npz(path)   # use exactly what a resumed run would use
        st.meta["agreement_with_fit_run"] = V.structure_agreement(st, self.struct, range(self.max_k))
        self._mem[key] = st
        return st


def known_results(out: Path, model_name: str) -> dict:
    """(signature, n_query) -> (k-NN result, label) for every variant already evaluated in `out`
    (any ablation, any fit seed), so an identical model is never scored twice. CSVs that another job
    is writing at this moment are skipped."""
    import pandas as pd

    known = {}
    for path in sorted(out.glob("A?_s*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "signature" not in df.columns:
            continue
        for row in df.to_dict("records"):
            if row.get("model") != model_name or isinstance(row.get("reused_from"), str):
                continue
            r = {key: row[key] for key in ("top1", "flip_consistency", "n_query", "n_bank")}
            label = f"{row['ablation']}/{row['variant']}/k{row['k']}/s{row['fit_seed']}"
            known.setdefault((str(row["signature"]), int(row["n_query"])), (r, label))
    return known


def row_fit_seed(ablation: str, variant: str, proto, fit_seed: int) -> int:
    if ablation == "A3" and int(variant[1:]) > len(proto.index[f"fit{fit_seed}_rows"]):
        return -1
    return fit_seed


def build_spec(ctx: Context, ablation: str, variant: str, k: int, rseed: int):
    """VariantSpec for one cell, or None if the cell is not run in this invocation."""
    st, a = ctx.struct, ctx.a
    kw = dict(seed=rseed, fix_ones=not a.no_fix_ones, top_k=a.top_k)
    if ablation == "A2" and variant == "generic":
        return V.build_variant("A2", variant, st, k, precomputed=ctx.generic(range(k)), **kw)
    if ablation == "A2" and variant in ("hungarian", "hungarian_cycle"):
        return V.build_variant("A2", variant, st, k, precomputed=ctx.hungarian(range(k)), **kw)
    if ablation == "A3":
        n = int(variant[1:])
        if row_fit_seed("A3", variant, ctx.proto, ctx.fit_seed) == -1 and ctx.fit_seed != 0:
            log(f"A3 {variant}: pooled over fit seeds 0, 1, 2; run with --fit-seed 0 only, skipped here")
            return None
        refit = ctx.a3_structure(n)
        spec = V.build_variant("A3", variant, refit, k, **kw)
        spec.info.update({"n_images": n, "fit_rows": refit.meta.get("fit_rows"),
                          "agreement_with_fit_run": refit.meta.get("agreement_with_fit_run")})
        return spec
    if ablation == "A4":
        return V.build_variant("A4", variant, st, k, shares=ctx.shares(), **kw)
    return V.build_variant(ablation, variant, st, k, **kw)


def main(argv=None) -> dict:
    a = parse_args(argv)
    ablations = [x.strip().upper() for x in a.ablation.split(",") if x.strip()]
    bad = [x for x in ablations if x not in ABLATIONS]
    if bad:
        raise SystemExit(f"unknown ablation(s) {bad}; choose from {ABLATIONS}")
    ks = list(dict.fromkeys(int(k) for k in a.ks.split(",")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    if a.tiny:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise SystemExit("no GPU visible; use --tiny for CPU runs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    amp = None if a.tiny else "bf16"
    git = git_info()
    model_name = "tiny" if a.tiny else a.model
    log(f"Track A {ablations} on {model_name}, ks {ks}, device {device}, commit {git['git_hash'][:10]}"
        f"{' (dirty)' if git['git_dirty'] else ''}")

    adapter, proto, num_classes = V.build_runtime(a.model, a.tiny, device, a.cache_dir, a.batch_size,
                                                  a.stats_batch_size, a.num_workers)
    L = adapter.depth
    struct = V.load_or_make_structure(a.fit_run, a.tiny, adapter, proto, device, out, a.fit_seed, a.top_k)
    if (struct.depth, struct.d, struct.h) != (L, adapter.embed_dim, adapter.mlp_hidden):
        raise SystemExit(f"fit run shape {(struct.depth, struct.d, struct.h)} does not match the model")
    run_model = struct.meta.get("model")
    if not a.tiny and run_model and run_model != a.model:
        raise SystemExit(f"fit run is for {run_model}, not {a.model}")
    fit_seed = V.resolve_fit_seed(a.fit_seed, struct)
    log(f"structure {struct.source}: fit seed {fit_seed}, checks {struct.meta.get('checks')}, "
        f"max change from cleaning {struct.meta.get('max_abs_change_from_cleaning', 0):.1e}")
    for k in ks:
        if not 1 <= k <= L:
            log(f"skip k = {k} (depth {L})")
    ks = [k for k in ks if 1 <= k <= L]
    if not ks:
        raise SystemExit("no valid k")

    bank, query = proto.bank_loader(), proto.query_loader(a.queries)
    t0 = time.time()
    orig = V.original_knn(adapter, bank, query, device, num_classes, model_name, amp, a.cache_dir,
                          a.queries, a.tiny, git["git_hash"])
    log(f"original: top-1 {orig['top1']:.2f}, flip consistency {orig['flip_consistency']:.2f} ({orig['source']})")
    ctx = Context(a, adapter, proto, struct, fit_seed, device, out, max(ks))
    rseed = 1000 * (fit_seed + 1)
    a3_sizes = ([int(x) for x in a.a3_sizes.split(",")] if a.a3_sizes
                else ([4, 10, 30] if a.tiny else list(V.A3_SIZES)))
    n_fit_default = len(proto.index[f"fit{fit_seed}_rows"])
    only = {v.strip() for v in a.variants.split(",")} if a.variants else None
    n_query = len(query.dataset)
    known = known_results(out, model_name)

    summary = {}
    for ablation in ablations:
        csv_path = out / f"{ablation}_s{fit_seed}.csv"
        csv = V.ResultsCsv(csv_path, KEY)
        names = [v for v in V.default_variants(ablation, a3_sizes) if only is None or v in only]
        n_new = 0
        for k in ks:
            for variant in names:
                rfs = row_fit_seed(ablation, variant, proto, fit_seed)
                if csv.done((ablation, variant, k, rfs)):
                    log(f"{ablation} {variant} k = {k} fit seed {rfs}: done, skipped")
                    continue
                t = time.time()
                spec = build_spec(ctx, ablation, variant, k, rseed)
                if spec is None:
                    continue
                sig = V.spec_signature(spec)
                label = f"{ablation}/{variant}/k{k}/s{rfs}"
                prev = known.get((sig, n_query))
                if prev is not None:      # the identical model was scored already: reuse, do not re-evaluate
                    r, reused = prev[0], prev[1]
                    log(f"{label}: same model as {reused}, result reused")
                else:
                    m = V.variant_copy(adapter.model, spec)
                    r = evaluate_knn(adapter, bank, query, device, num_classes, k=20, temperature=0.07,
                                     model=m, amp=amp)
                    del m
                    _free(device)
                    r = {key: r[key] for key in ("top1", "flip_consistency", "n_query", "n_bank")}
                    reused = ""
                    known[(sig, n_query)] = (r, label)
                # Reference: the per-layer mirror variant on blocks 1..k (A4 compares other selections to it).
                ref = sum(e1 + e2 for e1, e2 in V.mirror_param_counts(struct, range(k)))
                row = {
                    "model": model_name, "ablation": ablation, "variant": variant, "k": k, "fit_seed": rfs,
                    "blocks": " ".join(str(b + 1) for b in spec.blocks),
                    "method": spec.method, "projection": spec.projection if spec.method == "mirror" else "",
                    "alpha": spec.alpha if spec.method == "mirror" else "",
                    "top1": r["top1"], "flip_consistency": r["flip_consistency"],
                    "top1_original": orig["top1"], "flip_original": orig["flip_consistency"],
                    "drop_points": orig["top1"] - r["top1"],
                    "mlp_params_kept": spec.params_kept, "mlp_params_dense": spec.params_dense,
                    "mlp_params_kept_frac": spec.params_kept / spec.params_dense,
                    "params_exact": spec.params_exact,
                    "params_vs_mirror_first_k": spec.params_kept / ref,
                    "n_fit_images": int(variant[1:]) if ablation == "A3" else n_fit_default,
                    "n_query": r["n_query"], "n_bank": r["n_bank"],
                    "variant_seed": rseed if (ablation, variant) in RANDOM_VARIANTS else "",
                    "seed": SEED, "fit_run": struct.source, "git_hash": git["git_hash"],
                    "git_dirty": git["git_dirty"], "time_s": round(time.time() - t, 1),
                    "signature": sig, "reused_from": reused,
                    "info": json.dumps(spec.info, default=str),
                }
                csv.add(row)
                n_new += 1
                log(f"{ablation} {variant} k = {k}: top-1 {r['top1']:.2f} (drop {row['drop_points']:.2f}), "
                    f"flip {r['flip_consistency']:.2f}, MLP params kept {row['mlp_params_kept_frac']:.3f}")
        meta = {
            "ablation": ablation, "model": model_name, "ks": ks, "variants": names, "fit_seed": fit_seed,
            "seed": SEED, "git": git, "env": env_info(device), "args": vars(a), "fit_run": struct.source,
            "structure_meta": struct.meta, "original_knn": orig, "knn": {"k": 20, "temperature": 0.07, "amp": amp},
            "random_variant_seed": rseed, "fix_ones": not a.no_fix_ones, "a3_sizes": a3_sizes,
            "a3_order": a.a3_order, "rows_added": n_new, "rows_total": len(csv.rows), "notes": ctx.notes,
            "elapsed_s": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out / f"{ablation}_s{fit_seed}_meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                                               encoding="utf-8")
        summary[ablation] = {"csv": str(csv_path), "rows_added": n_new, "rows_total": len(csv.rows)}
        log(f"{ablation}: {n_new} new rows, {len(csv.rows)} in {csv_path}")
    return summary


if __name__ == "__main__":
    main()
