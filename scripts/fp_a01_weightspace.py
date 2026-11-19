"""A01: fit the weight-space operator A*_w, write its SpanStructure, compare it with the current operator and run
the zero-shot conversion of both (FINAL_PHASE_PLAN.md A01; PREREG_FINAL_PHASE.md Section 2.2; the method is in
mirror/fit_weightspace.py).

  python scripts/fp_a01_weightspace.py --model dinov2_vitb14_reg \\
      --structure results/MUST-07/structure/dinov2_vitb14_reg_s0.npz --mode both --out results/A01
  python scripts/fp_a01_weightspace.py ... --k 9 --mode matched      (span of blocks 1 to 9: the A06 point;
                                                                      default --out results/A01/k9)
  python scripts/fp_a01_weightspace.py --dry-run --out <dir>          (CPU, tiny random ViT, synthetic data)

Modes: "matched" (pre-registered main mode: the source operator's d- and every block's pair count, so the converted
model has the source's size) and "free" (diagnostic: the plain sign rule, pair counts free). The register pairing is
re-matched under A*_w on the structure's fit images (train role fit<fit_seed>, or --fit-images).

Outputs in --out:
  structure/<model>_wspace_<mode>[_k<k>].npz            the A*_w SpanStructure (mirror_family.save_span_structure)
  structure/<model>_wspace_<mode>[_k<k>]_objective.npz  the final objective matrix M and the eigenvalues of Pc M Pc
  operators.csv   one row per operator (current A*, A*_w per mode): d+, d-, pairs, J, damage (weighted sum of
                  breaking shares), agreement with the current A*, parameters, zero-shot k-NN
  rounds.csv      the alternation log (J, damage, d-, neurons that changed partner, blocks accepted or rejected)
  blocks.csv      per mode and block: pair counts, source pairs kept, MLP breaking share before and after
  terms.csv       per operator and damage term: energy, damage and breaking share
  knn/<operator>[_k<k>].json and .npz       zero-shot k-NN (top-1, flip consistency; pred, pred_flip, labels)
  summary.md, metrics.json, meta.json (mirror.final.prereg.write_final_meta)
Resume: a structure whose meta records the same source sha256, mode, k and settings is reused; k-NN results whose
recorded structure sha256 matches are reused. --force refits.
"""
from __future__ import annotations

import sys

import torch


def _early_gpu_frac(argv, default: float = 0.3) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if __name__ == "__main__" and torch.cuda.is_available() and "--dry-run" not in sys.argv[1:]:
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))  # shared GPU: cap before allocating

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import fit_weightspace as FW  # noqa: E402
from mirror import mirror_family as MF  # noqa: E402
from mirror import stability as STB  # noqa: E402
from mirror import variants as V  # noqa: E402
from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

MAIN_MODEL = "dinov2_vitb14_reg"
A01_ZERO_SHOT_BAR = 58.84     # PREREG_FINAL_PHASE.md Section 1, A01: zero-shot k-NN >= 58.84 (current 53.84 + 5)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MAIN_MODEL, help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--structure", default=None,
                   help="source SpanStructure (default results/MUST-07/structure/<model>_s0.npz; dry run: a tiny fit)")
    p.add_argument("--mode", default="both", choices=["matched", "free", "both"])
    p.add_argument("--k", type=int, default=None, help="span size, blocks 0..k-1 (default: every block)")
    p.add_argument("--out", default=None, help="output directory (default results/A01; with --k: results/A01/k<k>)")
    p.add_argument("--rounds", type=int, default=3, help="largest number of re-pairing rounds")
    p.add_argument("--tol", type=float, default=0.01, help="stop when fewer than this share of neurons change partner")
    p.add_argument("--top-k", type=int, default=None, help="matching candidates per neuron (default 10; dry run 3)")
    p.add_argument("--workers", type=int, default=None, help="processes for the matchings (default 8; dry run 1)")
    p.add_argument("--attn-terms", action="store_true", help="add the attention damage terms (default off)")
    p.add_argument("--term-weights", default=None, help="weights per term kind, e.g. mlp=1,patch=1 (default all 1)")
    p.add_argument("--register-margin", type=float, default=None,
                   help="register matching margin (default: the source structure's)")
    p.add_argument("--fit-images", default=None,
                   help="uint8 [N, S, S, 3] .npy for the register pass (default: train role fit<fit_seed>)")
    p.add_argument("--fit-index", default=None, help="npz with the rows of --fit-images to use")
    p.add_argument("--fit-rows-key", default=None, help="key of the rows in --fit-index")
    p.add_argument("--source-sidecar", default=None,
                   help="A02 sidecar of the source fit (scripts/fp_structure_fit.py): adds the energy-weighted "
                        "agreement in the direction current -> A*_w")
    p.add_argument("--skip-knn", action="store_true")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--eval-batch", type=int, default=None, help="feature batch (default 512; dry run 32)")
    p.add_argument("--stats-batch-size", type=int, default=16, help="paired passes (images with their mirrors)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu-frac", type=float, default=0.3, help="per-process GPU memory cap (set at start)")
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="refit and re-evaluate even when outputs exist")
    p.add_argument("--dry-run", action="store_true", help="tiny random ViT, synthetic data, CPU")
    return p.parse_args(argv)


def parse_weights(s) -> dict | None:
    if not s:
        return None
    out = {}
    for part in s.split(","):
        if part.strip():
            key, val = part.split("=")
            out[key.strip()] = float(val)
    return out


def _sha(path: Path) -> str:
    return FW.sha256_file(path)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(FW._jsonable(obj), indent=1, default=str), encoding="utf-8")


def _pairs(perm) -> set:
    return STB.matched_pairs(perm)


def struct_name(model: str, mode: str, k: int, L: int) -> str:
    return f"{model}_wspace_{mode}" + ("" if k == L else f"_k{k}")


# ----------------------------------------------------------------------------------------------
# Pieces
# ----------------------------------------------------------------------------------------------
def runtime(a, device):
    if a.dry_run:          # tiny random ViT with 2 registers (d 32, 3 blocks) and the synthetic tiny protocol
        proto = V.TinyProtocol()
        adapter, ncls, name = FW.dry_run_adapter(seed=0), proto.num_classes, "tiny"
    else:
        adapter, proto, ncls = V.build_runtime(a.model, False, device, a.cache_dir, a.eval_batch or 512,
                                               a.stats_batch_size, a.num_workers)
        name = a.model
    adapter.model.float().eval()
    return adapter, proto, ncls, name


def source_structure(a, adapter, proto, device, out: Path, model_name: str) -> Path:
    if a.structure:
        p = Path(a.structure)
        return p if p.is_absolute() else REPO_ROOT / p
    if not a.dry_run:
        return REPO_ROOT / "results" / "MUST-07" / "structure" / f"{model_name}_s0.npz"
    path = out / "structure" / "tiny_s0.npz"
    if not path.exists():
        log("dry run: fitting a tiny source structure on the synthetic fit set")
        st = MF.fit_span_structure(adapter, proto.fit_loader(0), device, model_name="tiny", fit_seed=0, top_k=3,
                                   workers=1, pairings=False)
        MF.save_span_structure(path, st)
    return path


def register_loader(a, proto, src):
    """The structure's fit images for the register pass: --fit-images, or the train role fit<fit_seed>."""
    if a.fit_images:
        images, rows = STB.load_image_rows(a.fit_images, a.fit_index, a.fit_rows_key)
        pin = torch.cuda.is_available() and not a.dry_run
        return STB.image_loader(images, rows, a.stats_batch_size, 0 if a.dry_run else a.num_workers, pin), \
            f"{a.fit_images} ({len(rows)} images)"
    if int(src.fit_seed) in (0, 1, 2):
        return proto.fit_loader(int(src.fit_seed)), f"train role fit{int(src.fit_seed)}"
    raise SystemExit("the source structure's fit images are not a train role fit0..fit2: pass --fit-images")


def _settings(a, top_k, weights) -> dict:
    return {"max_rounds": int(a.rounds), "tol": float(a.tol), "top_k": int(top_k), "attn_terms": bool(a.attn_terms),
            "weights": {**FW.DEFAULT_WEIGHTS, **(weights or {})}}


def reusable(path: Path, sha: str, mode: str, k: int, settings: dict):
    if not path.exists():
        return None
    st = MF.load_span_structure(path)
    of = st.meta.get("operator_fit", {}) or {}
    cfg = of.get("config", {}) or {}
    same = (of.get("source_sha256") == sha and of.get("mode") == mode and int(of.get("k", -1)) == int(k)
            and all(cfg.get(key) == val for key, val in settings.items()))
    return st if same else None


def operator_row(name: str, A, perms_k, terms, src_A, src_perms_k, S_self=None, S_src=None) -> tuple:
    vals = FW.term_values(terms, A, perms_k)
    J = float(sum(v["value"] for v in vals))
    damage = float(sum(v["weight"] * v["share"] for v in vals if v["weight"] and np.isfinite(v["share"])))
    dp, dm = STB.signature(A)
    mlp = [v["share"] for v in vals if v["kind"] == "mlp"]
    row = {"operator": name, "d_plus": dp, "d_minus": dm, "n_pairs_span": int(sum(FW.n_pairs_of(p) for p in perms_k)),
           "J": J, "damage": damage, "mlp_share_mean": float(np.mean(mlp)) if mlp else float("nan")}
    for kind in ("patch", "pos", "cls", "reg"):
        v = [x["share"] for x in vals if x["kind"] == kind]
        row[f"{kind}_share"] = float(v[0]) if v else float("nan")
    if name == "current":
        row.update({"operator_distance": 0.0, "unweighted_cos2": 1.0, "rho_to_current": 1.0, "rho_from_current": 1.0,
                    "weighted_agreement": 1.0, "pairs_kept_share": 1.0})
    else:
        oa = STB.odd_agreement(A, src_A, S_self, S_src)
        kept = sum(len(_pairs(p) & _pairs(q)) for p, q in zip(perms_k, src_perms_k))
        tot = sum(len(_pairs(q)) for q in src_perms_k)
        row.update({"operator_distance": STB.operator_distance(A, src_A), "unweighted_cos2": oa["unweighted_cos2"],
                    "rho_to_current": oa["rho_ab"], "rho_from_current": oa["rho_ba"],
                    "weighted_agreement": oa["weighted"], "pairs_kept_share": kept / tot if tot else float("nan")})
    return row, vals


def run_knn(a, adapter, proto, device, ncls, st, sha: str, name: str, k: int, out: Path) -> dict:
    jpath, npath = out / "knn" / f"{name}.json", out / "knn" / f"{name}.npz"
    if jpath.exists() and npath.exists() and not a.force:
        rec = json.loads(jpath.read_text(encoding="utf-8"))
        if rec.get("structure_sha256") == sha and int(rec.get("span", -1)) == int(k):
            log(f"k-NN {name}: reused ({rec['top1']:.2f})")
            return rec
    t = time.time()
    amp = "bf16" if torch.device(device).type == "cuda" else None
    r = FW.full_conversion_knn(adapter, st, proto.bank_loader(), proto.query_loader(), device, ncls, k=k,
                               knn_k=a.knn_k, temperature=a.knn_temperature, amp=amp)
    npath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npath, pred=r.pop("pred"), pred_flip=r.pop("pred_flip"), labels=r.pop("labels"))
    rec = {**r, "operator": name, "structure_sha256": sha, "seconds": round(time.time() - t, 1)}
    _write_json(jpath, rec)
    log(f"k-NN {name}: top-1 {rec['top1']:.2f}, flip {rec['flip_consistency']:.2f}, params {rec['params_model']:,} "
        f"({rec['seconds']:.0f} s)")
    return rec


def write_summary(out: Path, model: str, k: int, L: int, rows: list, rounds: pd.DataFrame, knn: dict,
                  gate: dict | None, src_path: Path, sha: str) -> Path:
    def f(x, nd=2):
        try:
            return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
        except (TypeError, ValueError):
            return str(x)

    lines = [f"# A01: weight-space operator A*_w ({model}, {'full model' if k == L else f'blocks 1 to {k}'})", ""]
    cur = knn.get("current", {})
    mat = knn.get("wspace_matched", {})
    if cur and mat:
        verdict = (f"Zero-shot k-NN: A*_w (matched) {f(mat.get('top1'))} against the current A* {f(cur.get('top1'))} "
                   f"({f(mat.get('top1', 0) - cur.get('top1', 0))} points).")
        if gate is not None:
            verdict += (f" Preview of the zero-shot part of the pre-registered A01 rule (at least {A01_ZERO_SHOT_BAR}): "
                        f"{'met' if gate['met'] else 'not met'}. The pre-registered value is step 0 of the A*_w "
                        "healing run, and the healed part is decided by the heals.")
        lines += [verdict, ""]
    elif not knn:
        lines += ["Zero-shot k-NN was not run (--skip-knn).", ""]
    lines += [f"Source structure: `{src_path}` (sha256 {sha[:16]}).", "",
              "| operator | d+ | d- | pairs (span) | damage | J | distance to A* | cos^2 (unweighted) | rho(A*_w -> A*) "
              "| pairs kept | params | zero-shot top-1 | flip (bf16) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        kn = knn.get(r["operator"], {})
        lines.append(f"| {r['operator']} | {r['d_plus']} | {r['d_minus']} | {r['n_pairs_span']:,} | {f(r['damage'], 4)} "
                     f"| {f(r['J'], 4)} | {f(r['operator_distance'], 4)} | {f(r['unweighted_cos2'], 3)} "
                     f"| {f(r['rho_to_current'], 3)} | {f(r['pairs_kept_share'], 3)} "
                     f"| {kn.get('params_model', 'n/a') if not kn else format(kn['params_model'], ',')} "
                     f"| {f(kn.get('top1'))} | {f(kn.get('flip_consistency'))} |")
    lines += ["", "Damage is the weighted sum of the terms' breaking shares ||X - X_eq||^2 / ||X||^2 (MLP blocks, patch "
              "embedding, position grid, class token, registers; mirror/fit_weightspace.py); J = tr(A M) and "
              "damage = (number of terms - J) / 2. rho is the energy-weighted share of A*_w's odd subspace inside the "
              "current A*'s (mirror/stability.py).", "", "## Alternation", "",
              "| mode | round | J after re-pairing | J | damage | d- | neurons changed | share changed |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in rounds.iterrows():
        lines.append(f"| {r['mode']} | {int(r['round'])} | {f(r['J_after_pairing'], 5)} | {f(r['J'], 5)} "
                     f"| {f(r['damage'], 5)} | {int(r['d_minus'])} | {int(r['neurons_changed'])} "
                     f"| {f(100 * r['share_changed'], 2)}% |")
    lines += ["", "Round 0 fits A on the source pairings (its first J column is the current operator's J). The free "
              "mode is a diagnostic only (not healed)."]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    prereg = require_prereg()
    if a.dry_run:
        device = torch.device("cpu")
    else:
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise SystemExit("no GPU visible; use --dry-run for CPU runs")
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(0)
    top_k = a.top_k if a.top_k is not None else (3 if a.dry_run else 10)
    workers = a.workers if a.workers is not None else (1 if a.dry_run else 8)
    weights = parse_weights(a.term_weights)
    adapter, proto, ncls, model_name = runtime(a, device)
    partial = a.k is not None and int(a.k) != adapter.depth
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "A01" / (f"k{int(a.k)}" if partial else "")
    out.mkdir(parents=True, exist_ok=True)
    src_path = source_structure(a, adapter, proto, device, out, model_name)
    src = MF.load_span_structure(src_path)
    L = adapter.depth
    if src.depth != L or src.d != adapter.embed_dim or src.h != adapter.mlp_hidden:
        raise SystemExit(f"{src_path} does not fit the model {model_name}")
    if src.model and src.model != model_name:
        log(f"note: the structure was fitted for {src.model!r}, this run is {model_name!r}")
    k = L if a.k is None else int(a.k)
    if not 1 <= k <= L:
        raise SystemExit(f"--k must be in 1..{L}")
    sha = _sha(src_path)
    modes = ["matched", "free"] if a.mode == "both" else [a.mode]
    settings = _settings(a, top_k, weights)
    margin = float(a.register_margin if a.register_margin is not None else src.meta.get("register_margin", 0.0))
    lin_dev = device                                      # float64 linear algebra on the GPU when there is one
    log(f"A01 {'DRY RUN ' if a.dry_run else ''}{model_name}: source {src_path.name} (sha {sha[:12]}), k = {k}, "
        f"modes {modes}, top_k {top_k}, workers {workers}")
    terms = FW.collect_terms(adapter, k, reg_perm=src.reg_perm[k], heads=src.heads, attn_terms=a.attn_terms,
                             weights=weights, device=lin_dev)
    A_src = FW.exact_involution(src.A_star[k])
    P_src = [np.asarray(src.perms[b], dtype=np.int64) for b in range(k)]
    R = int(FW.register_info(adapter.model, adapter)["R"])
    structs, fits_meta, round_rows, block_rows = {}, {}, [], []
    S_w = {}
    for mode in modes:
        name = struct_name(model_name, mode, k, L)
        path = out / "structure" / f"{name}.npz"
        st = None if a.force else reusable(path, sha, mode, k, settings)
        if st is not None:
            log(f"{mode}: reusing {path.name}")
        else:
            fit = FW.fit_weightspace(adapter, src, mode, k, max_rounds=a.rounds, tol=a.tol, top_k=top_k,
                                     workers=workers, attn_terms=a.attn_terms, weights=weights, device=lin_dev,
                                     terms=terms, log=log)
            if R:
                loader, desc = register_loader(a, proto, src)
                reg, cos = FW.rematch_registers(adapter, loader, device, fit.A, k, margin)
                reg_src = f"re-matched under A*_w (m1.RegisterScores, layers 0..{k}, margin {margin}) on {desc}"
            else:
                reg, cos, reg_src = None, None, "the model has no registers"
            st = FW.weightspace_structure(src, fit, reg, source_path=src_path, source_sha256=sha,
                                          register_source=reg_src,
                                          extra={"register_cosines": None if cos is None else np.asarray(cos)})
            MF.save_span_structure(path, st)
            np.savez(out / "structure" / f"{name}_objective.npz", M=fit.M, eigvals=np.asarray(fit.eigvals),
                     rounds_json=np.array(json.dumps(FW._jsonable(fit.rounds))))
            log(f"{mode}: saved {path.name}: d- {fit.d_minus}, J {fit.J_source:.5f} -> {fit.J:.5f}, {fit.stopped}")
            st = MF.load_span_structure(path)
        structs[mode] = (st, path)
        of = st.meta["operator_fit"]
        fits_meta[mode] = {key: of.get(key) for key in ("d_plus", "d_minus", "J", "J_source", "damage",
                                                        "damage_source", "J_rounds", "stopped", "register_pairing",
                                                        "register_perm", "n_pairs", "n_pairs_source")}
        for r in of["rounds"]:
            round_rows.append({"mode": mode, **{key: (json.dumps(v) if isinstance(v, list) else v)
                                                for key, v in r.items() if key != "note"}})
        perms_w = [np.asarray(st.perms[b], dtype=np.int64) for b in range(k)]
        S_w[mode] = FW.centre_matrix(FW.objective_matrix(terms, perms_w))
        A_w = FW.exact_involution(st.A_star[k])
        vs = FW.term_values(terms, A_src, P_src)
        vw = FW.term_values(terms, A_w, perms_w)
        for b in range(k):
            block_rows.append({"mode": mode, "block": b, "n_pairs_source": FW.n_pairs_of(P_src[b]),
                               "n_pairs": FW.n_pairs_of(perms_w[b]),
                               "pairs_kept": len(_pairs(P_src[b]) & _pairs(perms_w[b])),
                               "neurons_changed": int((P_src[b] != perms_w[b]).sum()),
                               "mlp_share_source": vs[b]["share"], "mlp_share": vw[b]["share"]})

    S_src = None
    if a.source_sidecar:
        sc = STB.load_sidecar(a.source_sidecar)
        if k == L and np.array_equal(np.asarray(sc["perms"]), np.stack([np.asarray(p) for p in src.perms])):
            S_src = np.asarray(sc["S_span"])
        else:
            log("note: the source sidecar does not match the source structure (or k < L); weighted agreement is "
                "reported in the direction A*_w -> A* only")
    rows, term_rows = [], []
    row, vals = operator_row("current", A_src, P_src, terms, A_src, P_src)
    rows.append(row)
    term_rows += [{"operator": "current", **v} for v in vals]
    for mode in modes:
        st, _ = structs[mode]
        A_w = FW.exact_involution(st.A_star[k])
        perms_w = [np.asarray(st.perms[b], dtype=np.int64) for b in range(k)]
        own = FW.with_register_perm(terms, st.reg_perm[k])   # damage of the saved conversion (re-matched registers)
        row, vals = operator_row(f"wspace_{mode}", A_w, perms_w, own, A_src, P_src, S_w[mode], S_src)
        rows.append(row)
        term_rows += [{"operator": f"wspace_{mode}", **v} for v in vals]

    knn = {}
    sfx = "" if k == L else f"_k{k}"
    if not a.skip_knn:
        knn["current"] = run_knn(a, adapter, proto, device, ncls, src, sha, f"current{sfx}", k, out)
        for mode in modes:
            st, path = structs[mode]
            knn[f"wspace_{mode}"] = run_knn(a, adapter, proto, device, ncls, st, _sha(path), f"wspace_{mode}{sfx}",
                                            k, out)
    for r in rows:
        kn = knn.get(r["operator"])
        r.update({"knn_top1": kn["top1"] if kn else float("nan"),
                  "flip_consistency_bf16": kn["flip_consistency"] if kn else float("nan"),
                  "params_model": kn["params_model"] if kn else None,
                  "params_matrices": kn["params_matrices"] if kn else None})
    gate = None
    if model_name == MAIN_MODEL and k == L and "wspace_matched" in knn:
        v = float(knn["wspace_matched"]["top1"])
        gate = {"rule": "A01 zero-shot part: A*_w (matched) zero-shot k-NN >= 58.84 (PREREG_FINAL_PHASE.md)",
                "bar": A01_ZERO_SHOT_BAR, "value": v, "met": bool(v >= A01_ZERO_SHOT_BAR),
                "current_same_run": float(knn["current"]["top1"]),
                "status": "preview: the pre-registered value is step 0 of the A*_w healing run (PREREG 2.2)",
                "healed_part": "decided by the A01 heals (2 seeds), not here"}

    pd.DataFrame(rows).to_csv(out / "operators.csv", index=False)
    rounds = pd.DataFrame(round_rows)
    rounds.to_csv(out / "rounds.csv", index=False)
    pd.DataFrame(block_rows).to_csv(out / "blocks.csv", index=False)
    pd.DataFrame(term_rows).to_csv(out / "terms.csv", index=False)
    summary = write_summary(out, model_name, k, L, rows, rounds, knn, gate, src_path, sha)
    metrics = {"experiment": "A01", "model": model_name, "k": k, "depth": L, "source_structure": str(src_path),
               "source_sha256": sha, "modes": modes, "settings": settings, "operators": rows, "fits": fits_meta,
               "knn": {n: {key: v for key, v in r.items()} for n, r in knn.items()}, "zero_shot_preview": gate,
               "structures": {m: {"path": str(p), "sha256": _sha(p)} for m, (st, p) in structs.items()},
               "dry_run": bool(a.dry_run)}
    _write_json(out / "metrics.json", metrics)
    wall = time.time() - t_start
    write_final_meta(out, experiment="A01", config=vars(a),
                     seeds={"fit_seed": int(src.fit_seed), "global": 0, "matching": "deterministic (blossom)"},
                     models={model_name: "tiny random timm ViT (dry run)" if a.dry_run else model_name},
                     device=device, wall_seconds=wall,
                     extra={"prereg_check": {key: prereg.get(key) for key in ("ok", "reason", "allowed_without_check")},
                            "source_structure": str(src_path), "source_sha256": sha, "k": k,
                            "structures": metrics["structures"], "git": git_info()})
    log(f"A01 done in {wall / 60:.1f} min: {summary}")
    return {"out": str(out), "operators": rows, "knn": knn, "gate": gate, "structures": metrics["structures"],
            "fits": fits_meta}


if __name__ == "__main__":
    main()
