#!/usr/bin/env python
"""M1: attention conversion (V-local and V-exact): head structure, exactness and zero-shot k-NN
(PAPER_EXPERIMENTS.md, M1; also A6 and the accuracy part of A7).

    python scripts/m1_attention.py --model dinov2_vitb14_reg \
        --operators runs/x2/perlayer_reg_s0/results/e1_operators.npz --fit-seed 0 --ks 3,6,9,12 --out runs/m1
    python scripts/m1_attention.py --tiny --out tiny_out/m1     # tiny random ViT, synthetic data, CPU

Stages (each writes its outputs to --out and is skipped when they exist, so a rerun resumes):
 1. structure: one paired pass over the fit images (Protocol.fit_loader(fit seed)) collecting
    attention statistics (AttnStats) and register alignment scores. Per block: head pairing and head
    maps, head-map errors, attention breaking shares and free parameters
      V-local: A_in = A_l for q, k, v; for Wo the mid-block stream has no fitted operator, so
               A_out = A_l and A_out = A_(l+1) are both scored and the one with the lower
               block-average attention breaking share (s_all, mean over all blocks) is kept;
      V-exact: one operator A* per span (blocks 1..k, stream layers 0..k), fitted with
               fit.fit_involution from sum_l C_l / tr(G_l) (each layer normalised to equal
               energy), shares measured on the fully folded model;
    plus a matched random-structure null and the register pairing for each span.
 2. exactness (fp32, TF32 off, validation images): for the V-exact model of every k, the mirrored
    image's tokens at mirrored positions against A* times the original tokens, at every layer;
    pass if the max relative error (max abs error / max abs token entry) is below 1e-4 for every
    layer inside the span.
 3. flip check (k = depth only): the fully converted model (V-exact, even head) in fp32 on a
    validation subset: k-NN flip consistency (must be 100%) and feature invariance.
 4. zero-shot k-NN (paper protocol: bank of 100 train images per class, all 50k validation
    queries, bf16, k = 20, T = 0.07; original from the cache written by run_e1.py), per k:
      vlocal    MLP + attention, V-local
      vexact    MLP + attention, V-exact (k = depth: with the even head)
      mlp_only  MLP only, V-local (A6; the E1b surgery)
      svd       SVD of fc1, fc2, Wq, Wk, Wv, Wo at the V-local parameter counts (matrices)
      random    matched random structure (operators, neuron pairings, head maps) on MLP + attention
    plus the folded original (k = 0) as a sanity row.
Outputs: m1_structure.npz (head maps, A_out choice, span operators, register pairings; read by
scripts/heal.py), m1_heads.csv, m1_heads_detail.csv, m1_ranks.csv, m1_vexact.csv,
m1_vexact_blocks.csv, m1_vexact_layers.csv, m1_exactness.csv, m1_flipcheck.json, m1_knn.csv,
m1_attn_stats.npz, m1_meta.json and M1_SUMMARY.md. Every CSV row carries the git hash and seeds.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.1)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.adapters import Dinov2Adapter, TimmTinyAdapter  # noqa: E402
from mirror.attention import (HeadMaps, apply_attention_surgery, attention_breaking_shares,  # noqa: E402
                              attention_free_params, attn_dims, collect_attn_stats, head_map_errors,
                              pair_and_fit_heads)
from mirror.data import make_loader, make_splits, normalize_for, synthetic_dataset  # noqa: E402
from mirror.fit import fit_layers  # noqa: E402
from mirror.fold import fold_layernorm_full  # noqa: E402
from mirror.heal import exact_involution  # noqa: E402
from mirror.knn import evaluate_knn, extract_features, knn_predict  # noqa: E402
from mirror.m1 import (RegisterScores, apply_attention_svd, attention_svd_ranks, build_vexact,  # noqa: E402
                       feature_invariance, fit_span_operator, involution_trace64, match_registers,
                       materialise_pos_embed, matched_random_structure, mlp_mirror_params, project_embeddings,
                       register_info, span_operator_report, stream_equivariance)
from mirror.matching import match_many  # noqa: E402
from mirror.stats import collect_fit_stats  # noqa: E402
from mirror.surgery import (apply_surgery_two, apply_svd_baseline, equivariant_param_count,  # noqa: E402
                            mlp_breaking_shares_two, perm_trace, svd_rank_for)
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402

VARIANTS = ("vlocal", "vexact", "mlp_only", "svd", "random")
SANITY = ("folded",)
VARIANT_LABELS = {
    "original": "original", "folded": "folded original (sanity)",
    "vlocal": "Mirror V-local, MLP + attention", "vexact": "Mirror V-exact, MLP + attention",
    "mlp_only": "Mirror V-local, MLP only (A6)", "svd": "SVD at matched parameters, MLP + attention",
    "random": "matched random structure, MLP + attention",
}
STRUCTURE_FORMAT = 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="torch.hub DINOv2 entry")
    p.add_argument("--hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--operators", default=None,
                   help="e1_operators.npz from scripts/run_e1.py (A_layers, perms, C, G); with --tiny it is "
                        "computed in-run when omitted")
    p.add_argument("--fit-seed", type=int, default=0, help="Protocol fit set used for the attention statistics")
    p.add_argument("--ks", default=None, help="comma-separated numbers of converted blocks (default 3,6,9,12; tiny 1,2,4)")
    p.add_argument("--out", default=None, help="output directory (default runs/m1; tiny_out/m1 with --tiny)")
    p.add_argument("--variants", default=",".join(SANITY + VARIANTS), help="k-NN variants to run")
    p.add_argument("--seed", type=int, default=0, help="global seed")
    p.add_argument("--random-seed", type=int, default=1,
                   help="random-structure control (structure.random_seed in configs/m3_pilot.yaml)")
    p.add_argument("--register-margin", type=float, default=0.0,
                   help="a register pair is kept only if its gain over two even registers exceeds this")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128, help="k-NN feature extraction")
    p.add_argument("--stats-batch-size", type=int, default=16, help="paired passes (run with their mirrors)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--knn-amp", default="bf16", choices=["bf16", "none"])
    p.add_argument("--exact-images", type=int, default=64, help="validation images for the exactness probe")
    p.add_argument("--exact-tol", type=float, default=1e-4)
    p.add_argument("--flip-queries", type=int, default=5000, help="validation queries for the fp32 flip check")
    p.add_argument("--flip-bank-per-class", type=int, default=10, help="bank images per class for the flip check")
    p.add_argument("--memory-fraction", type=float, default=None, help="override the 0.1 GPU memory cap")
    p.add_argument("--no-save-stats", action="store_true", help="do not write m1_attn_stats.npz")
    p.add_argument("--force", action="store_true", help="recompute every stage")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, synthetic data, CPU")
    return p.parse_args(argv)


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perm_str(perm) -> str:
    """Involution as a short string of 0-based indices: pairs 'a-b', fixed points 'a'."""
    perm = np.asarray(perm)
    parts = []
    for i, j in enumerate(perm):
        if j == i:
            parts.append(str(i))
        elif j > i:
            parts.append(f"{i}-{j}")
    return " ".join(parts)


# ----------------------------------------------------------------------------------------------
# Model, data, operators
# ----------------------------------------------------------------------------------------------
def build_adapter(a, device):
    if a.tiny:
        adapter = TimmTinyAdapter(img_size=56, seed=a.seed, randomise=True, dtype=torch.float32)
    elif a.model.startswith("dinov2"):
        adapter = Dinov2Adapter(hub_repo=a.hub_repo, name=a.model, img_size=a.img_size)
    else:
        from mirror.adapters_timm import build_model_adapter

        adapter = build_model_adapter(a.model, img_size=a.img_size)
    adapter.to(device=device)
    adapter.model.float().eval()
    return adapter


def load_data(a, device) -> dict:
    if a.tiny:
        per = {"fit": 4, "heldout": 2, "bank": 12, "query": 4}
        images, labels = synthetic_dataset(5, sum(per.values()), 56, seed=a.seed)
        sp = make_splits(labels, per)

        def ld(s, bs):
            return make_loader(images, labels, sp[s], bs, 0, False)

        return {"fit": ld("fit", 8), "probe": ld("heldout", 8), "bank": ld("bank", 32), "query": ld("query", 32),
                "flip_bank": ld("bank", 32), "flip_query": ld("query", 32), "num_classes": 5,
                "sizes": {k: int(len(v)) for k, v in sp.items()}, "protocol": "tiny synthetic"}
    from mirror.protocol import Protocol

    pin = device.type == "cuda"
    proto = Protocol(_resolve(a.cache_dir), a.batch_size, a.stats_batch_size, a.num_workers, pin)
    bank_rows = proto.index["bank_rows"]
    sub = make_splits(proto.train_labels[bank_rows], {"b": a.flip_bank_per_class})["b"]
    sizes = proto.sizes(a.fit_seed)
    sizes.update({"flip_bank": int(len(sub)), "flip_query": int(a.flip_queries)})
    return {"fit": proto.fit_loader(a.fit_seed), "probe": proto.heldout_loader(), "bank": proto.bank_loader(),
            "query": proto.query_loader(), "num_classes": 1000,
            "flip_bank": make_loader(proto.train, proto.train_labels, bank_rows[sub], a.batch_size, a.num_workers, pin),
            "flip_query": proto.query_loader(a.flip_queries), "sizes": sizes, "protocol": "v2"}


def tiny_operators(adapter, loader, device, path: Path) -> Path:
    """Tiny path: per-layer operators and twin pairings fitted in-run (run_e1 layout)."""
    fs = collect_fit_stats(adapter, loader, device, center=True, mlp=True)
    res = fs.residual.numpy()
    fit = fit_layers(res["C"], (1, adapter.depth), res["sq_x"], res["sq_xt"])
    matches = match_many([fs.mlp.correlation(b) for b in range(adapter.depth)], top_k=10, workers=1)
    np.savez_compressed(path, A_layers=np.stack(fit["A_layers"]).astype(np.float32),
                        C=res["C"].astype(np.float32), G=res["G"].astype(np.float32),
                        perms=np.stack([m.perm for m in matches]).astype(np.int32))
    return path


def load_operators(path: Path, L: int, d: int, h: int) -> dict:
    with np.load(path) as z:
        A_layers = [exact_involution(A) for A in z["A_layers"]]
        perms = [np.asarray(p, dtype=np.int64) for p in z["perms"]]
        C = np.asarray(z["C"], dtype=np.float64)
        G = np.asarray(z["G"], dtype=np.float64) if "G" in z.files else None
    if len(A_layers) != L + 1 or A_layers[0].shape != (d, d) or len(perms) != L or len(perms[0]) != h:
        raise ValueError(f"{path}: expected {L + 1} operators of {d} x {d} and {L} perms of length {h}")
    if C.shape != (L + 1, d, d):
        raise ValueError(f"{path}: C has shape {C.shape}, expected {(L + 1, d, d)}")
    for p in perms:
        if not np.array_equal(p[p], np.arange(h)):
            raise ValueError(f"{path}: a neuron pairing is not an involution")
    return {"A_layers": A_layers, "perms": perms, "C": C, "G": G, "path": str(path), "sha256": sha256(path)}


# ----------------------------------------------------------------------------------------------
# Stage 1: structure
# ----------------------------------------------------------------------------------------------
def _save_structure(path: Path, S: dict, a, ops: dict, git: dict) -> None:
    ks = sorted(S["A_star"])
    R = S["n_reg"]
    np.savez_compressed(
        path,
        format=np.array(STRUCTURE_FORMAT),
        sigma=np.stack([hm.sigma for hm in S["maps"]]).astype(np.int64),
        B=np.stack([hm.B for hm in S["maps"]]), C=np.stack([hm.C for hm in S["maps"]]),
        scores=np.stack([hm.scores for hm in S["maps"]]),
        a_out_offset=np.array(S["a_out_offset"]),
        ks=np.array(ks, dtype=np.int64),
        A_star=np.stack([S["A_star"][k] for k in ks]),
        reg_perm=np.stack([S["reg_perm"][k] for k in ks]).astype(np.int64) if R else np.zeros((len(ks), 0), np.int64),
        reg_scores=np.stack([S["reg_scores"][k] for k in ks]) if R else np.zeros((len(ks), 0, 0)),
        model=np.array(a.model if not a.tiny else "tiny"), fit_seed=np.array(a.fit_seed),
        operators=np.array(ops["path"]), operators_sha256=np.array(ops["sha256"]),
        random_seed=np.array(a.random_seed), register_margin=np.array(a.register_margin),
        git_hash=np.array(git["git_hash"]),
    )


def _load_structure(path: Path, a, ops: dict, ks: list):
    with np.load(path) as z:
        f = {k: z[k] for k in z.files}
    want = {"model": a.model if not a.tiny else "tiny", "fit_seed": a.fit_seed,
            "operators_sha256": ops["sha256"], "random_seed": a.random_seed, "register_margin": a.register_margin}
    for key, val in want.items():
        got = f[key].item()
        if (float(got) != float(val)) if isinstance(val, (int, float)) else (str(got) != str(val)):
            raise RuntimeError(f"{path} was made with {key} = {got}, this run has {val}: use another --out or --force")
    saved_ks = [int(k) for k in f["ks"]]
    if not set(ks) <= set(saved_ks):
        log(f"saved structure covers ks {saved_ks}, not {ks}: recomputing")
        return None
    maps = [HeadMaps(sigma=f["sigma"][b], B=f["B"][b], C=f["C"][b], scores=f["scores"][b]) for b in range(len(f["sigma"]))]
    R = f["reg_perm"].shape[1]
    return {"maps": maps, "a_out_offset": int(f["a_out_offset"]), "n_reg": int(R),
            "A_star": {k: f["A_star"][i] for i, k in enumerate(saved_ks)},
            "reg_perm": {k: f["reg_perm"][i] for i, k in enumerate(saved_ks)},
            "reg_scores": {k: f["reg_scores"][i] for i, k in enumerate(saved_ks)}}


def structure_stage(a, adapter, data, ops, ks, out: Path, device, git, tags: dict) -> dict:
    path = out / "m1_structure.npz"
    model = adapter.model
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    H, dh = attn_dims(model.blocks[0])
    reg = register_info(model, adapter)
    A, perms = ops["A_layers"], ops["perms"]
    if path.exists() and not a.force:
        S = _load_structure(path, a, ops, ks)
        if S is not None:
            log(f"structure from {path} (A_out offset {S['a_out_offset']})")
            S["random"] = matched_random_structure(A, perms, a.random_seed, fix_ones=True, heads=S["maps"])
            return S

    # Span operators (V-exact), one per k: stream layers 0..k, each normalised to equal energy.
    A_star, star_info = {}, {}
    for k in ks:
        A_star[k], star_info[k] = fit_span_operator(ops["C"], range(k + 1), ops["G"])
        log(f"V-exact k = {k}: A* over layers 0..{k}: d+ {star_info[k]['d_plus']}, d- {star_info[k]['d_minus']}")

    # One paired pass: attention statistics for all blocks, register scores for every span.
    rs = (RegisterScores({k: (A_star[k], k) for k in ks}, L + 1, reg["R"], reg["start"], device)
          if reg["R"] else None)
    st = collect_attn_stats(adapter, data["fit"], device, residual_cb=rs.update if rs is not None else None)
    if not a.no_save_stats:
        arrs = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and k != "n" else v) for k, v in st.numpy().items()}
        np.savez_compressed(out / "m1_attn_stats.npz", **arrs)

    maps, errs, n_tok = [], [], []
    for b in range(L):
        mom = st.moments(b)
        hm = pair_and_fit_heads(mom)
        hm.check(tol=1e-8)
        maps.append(hm)
        errs.append(head_map_errors(mom, hm))
        n_tok.append(mom["n"])
    del st
    _free(device)

    # V-local: A_in = A_b; A_out = A_b (offset 0) or A_(b+1) (offset 1), lower block-average share kept.
    sh = {0: [attention_breaking_shares(model.blocks[b], A[b], A[b], maps[b]) for b in range(L)],
          1: [attention_breaking_shares(model.blocks[b], A[b], A[b + 1], maps[b]) for b in range(L)]}
    mean_sh = {o: float(np.mean([s["s_all"] for s in sh[o]])) for o in (0, 1)}
    off = 1 if mean_sh[1] < mean_sh[0] else 0
    log(f"A_out choice: block-average s_all {mean_sh[0]:.4f} (A_l) vs {mean_sh[1]:.4f} (A_l+1): "
        f"keep {'A_l+1' if off else 'A_l'}")
    rnd = matched_random_structure(A, perms, a.random_seed, fix_ones=True, heads=maps)
    rA = rnd["A_layers"]

    rows, detail, ranks = [], [], []
    for b in range(L):
        blk, hm, err = model.blocks[b], maps[b], errs[b]
        s0, s1 = sh[0][b], sh[1][b]
        sc = sh[off][b]
        srnd = attention_breaking_shares(blk, rA[b], rA[b + off], rnd["heads"][b])
        mlp = mlp_breaking_shares_two(blk, A[b], A[b + 1], perms[b])
        mlp_r = mlp_breaking_shares_two(blk, rA[b], rA[b + 1], rnd["perms"][b])
        free = attention_free_params(d, hm, A[b], A[b + off])
        S_pick = hm.scores[np.arange(H), hm.sigma]
        rows.append({
            "block": b + 1, "H": H, "head_dim": dh, "n_self": hm.n_self, "n_pairs": hm.n_pairs,
            "sigma": perm_str(hm.sigma), "mean_pair_score": float(S_pick.mean()),
            "err_qk": float(err["qk"].mean()), "err_v": float(err["v"].mean()),
            "err_qk_identity": float(err["qk_identity"].mean()), "err_v_identity": float(err["v_identity"].mean()),
            "s_q": sc["s_q"], "s_k": sc["s_k"], "s_v": sc["s_v"], "s_o": sc["s_o"], "s_all": sc["s_all"],
            "s_o_out_same": s0["s_o"], "s_all_out_same": s0["s_all"], "s_o_out_next": s1["s_o"], "s_all_out_next": s1["s_all"],
            "s_all_random": srnd["s_all"], "s_all_over_random": sc["s_all"] / srnd["s_all"] if srnd["s_all"] > 0 else float("nan"),
            "mlp_s1": mlp["s1"], "mlp_s2": mlp["s2"], "mlp_s1_random": mlp_r["s1"], "mlp_s2_random": mlp_r["s2"],
            "attn_free_matrices": free["matrices"], "attn_matrices_dense": free["matrices_original"],
            "attn_kept_frac": free["matrices"] / free["matrices_original"], "attn_free_total": free["total"],
            "attn_dense_total": free["original"], "tr_A_in": involution_trace64(A[b]),
            "tr_A_out": involution_trace64(A[b + off]), "a_out": "A_l+1" if off else "A_l", "n_tokens": n_tok[b],
        })
        for hh in range(H):
            g = int(hm.sigma[hh])
            detail.append({"block": b + 1, "head": hh, "partner": g, "kind": "self" if g == hh else "pair",
                           "score": float(hm.scores[hh, g]), "self_score": float(hm.scores[hh, hh]),
                           "err_qk": float(err["qk"][hh]), "err_v": float(err["v"][hh]),
                           "err_qk_identity": float(err["qk_identity"][hh]), "err_v_identity": float(err["v_identity"][hh]),
                           "tr_B": involution_trace64(hm.B[hh]) if g == hh else None,
                           "tr_C": involution_trace64(hm.C[hh]) if g == hh else None})
        eq1 = equivariant_param_count(h, d, perm_trace(perms[b]), involution_trace64(A[b]))
        eq2 = equivariant_param_count(h, d, perm_trace(perms[b]), involution_trace64(A[b + 1]))
        r_att = attention_svd_ranks(free, d, H * dh)
        ranks.append({"block": b + 1, "mlp_eq_fc1": eq1, "mlp_eq_fc2": eq2, "svd_r_fc1": svd_rank_for(h, d, eq1),
                      "svd_r_fc2": svd_rank_for(h, d, eq2), "attn_free_q": free["Wq"], "attn_free_k": free["Wk"],
                      "attn_free_v": free["Wv"], "attn_free_o": free["Wo"], "svd_r_q": r_att[0], "svd_r_k": r_att[1],
                      "svd_r_v": r_att[2], "svd_r_o": r_att[3]})

    # V-exact: register pairing per span, shares on the folded model, embedding shares.
    reg_perm, reg_scores = {}, {}
    for k in ks:
        if reg["R"]:
            reg_scores[k] = rs.cosines(k, range(k + 1))
            reg_perm[k] = match_registers(reg_scores[k], a.register_margin)
        else:
            reg_scores[k] = np.zeros((0, 0))
            reg_perm[k] = np.zeros(0, dtype=np.int64)
    folded = copy.deepcopy(model)
    materialise_pos_embed(folded, adapter)
    fold_layernorm_full(folded, final_norm_affine=False)
    vx, vx_blocks, vx_layers = [], [], []
    for k in ks:
        As = A_star[k]
        span_rows = span_operator_report(As, A, ops["C"], range(k + 1))
        for r in span_rows:
            vx_layers.append({"k": k, **r})
        blk_rows = []
        for b in range(k):
            s = attention_breaking_shares(folded.blocks[b], As, As, maps[b])
            m_ = mlp_breaking_shares_two(folded.blocks[b], As, As, perms[b])
            free = attention_free_params(d, maps[b], As, As)
            blk_rows.append({"k": k, "block": b + 1, "s_q": s["s_q"], "s_k": s["s_k"], "s_v": s["s_v"], "s_o": s["s_o"],
                             "s_all": s["s_all"], "mlp_s1": m_["s1"], "mlp_s2": m_["s2"],
                             "attn_free_matrices": free["matrices"],
                             "mlp_free": mlp_mirror_params(h, d, perms[b], As, As)})
        vx_blocks += blk_rows
        emb = project_embeddings(folded, adapter, As, reg_perm[k], write=False)
        R = reg["R"]
        vx.append({
            "k": k, "layers": f"0-{k}", "d_plus": star_info[k]["d_plus"], "d_minus": star_info[k]["d_minus"],
            "n_zero": star_info[k]["n_zero"], "normalisation": star_info[k]["normalisation"],
            "mean_agreement": float(np.mean([r["agreement"] for r in span_rows])),
            "min_agreement": float(np.min([r["agreement"] for r in span_rows])),
            "mean_alignment": float(np.mean([r["alignment"] for r in span_rows])),
            "min_alignment": float(np.min([r["alignment"] for r in span_rows])),
            "mean_s_all": float(np.mean([r["s_all"] for r in blk_rows])),
            "mean_mlp_s1": float(np.mean([r["mlp_s1"] for r in blk_rows])),
            "mean_mlp_s2": float(np.mean([r["mlp_s2"] for r in blk_rows])),
            "n_registers": R, "register_perm": perm_str(reg_perm[k]) if R else "",
            "register_pairs": int((reg_perm[k] != np.arange(R)).sum() // 2) if R else 0,
            "register_scores": json.dumps(np.round(reg_scores[k], 4).tolist()) if R else "",
            **{f"emb_share_{key}": v for key, v in emb.items()},
        })
    del folded
    _free(device)

    for df_rows in (rows, detail, ranks, vx, vx_blocks, vx_layers):
        for r in df_rows:
            r.update(tags)
    pd.DataFrame(rows).to_csv(out / "m1_heads.csv", index=False)
    pd.DataFrame(detail).to_csv(out / "m1_heads_detail.csv", index=False)
    pd.DataFrame(ranks).to_csv(out / "m1_ranks.csv", index=False)
    pd.DataFrame(vx).to_csv(out / "m1_vexact.csv", index=False)
    pd.DataFrame(vx_blocks).to_csv(out / "m1_vexact_blocks.csv", index=False)
    pd.DataFrame(vx_layers).to_csv(out / "m1_vexact_layers.csv", index=False)
    with open(out / "m1_a_out_choice.json", "w", encoding="utf-8") as fh:
        json.dump({"a_out_offset": off, "kept": "A_l+1" if off else "A_l", "mean_s_all_A_l": mean_sh[0],
                   "mean_s_all_A_l+1": mean_sh[1], "rule": "lower block-average attention breaking share s_all "
                   "over all blocks", **tags}, fh, indent=2)
    S = {"maps": maps, "a_out_offset": off, "n_reg": reg["R"], "A_star": A_star, "reg_perm": reg_perm,
         "reg_scores": reg_scores, "random": rnd}
    _save_structure(path, S, a, ops, git)
    log(f"saved structure to {path}")
    return S


# ----------------------------------------------------------------------------------------------
# Variants and parameter counts
# ----------------------------------------------------------------------------------------------
def make_variant(name: str, k: int, adapter, S: dict, ops: dict):
    """A converted deep copy of the model for one k-NN row."""
    L = adapter.depth
    A, perms, maps, off = ops["A_layers"], ops["perms"], S["maps"], S["a_out_offset"]
    blocks = list(range(k))
    if name == "folded":
        m = copy.deepcopy(adapter.model)
        fold_layernorm_full(m, final_norm_affine=False)
        return m
    if name == "vexact":
        m, _ = build_vexact(adapter, S["A_star"][k], blocks, [perms[b] for b in blocks], [maps[b] for b in blocks],
                            reg_perm=S["reg_perm"][k], even_head=(k == L))
        return m
    m = copy.deepcopy(adapter.model)
    if name in ("vlocal", "mlp_only"):
        apply_surgery_two(m, blocks, [A[b] for b in blocks], [A[b + 1] for b in blocks], [perms[b] for b in blocks])
        if name == "vlocal":
            apply_attention_surgery(m, blocks, [A[b] for b in blocks], [A[b + off] for b in blocks],
                                    [maps[b] for b in blocks])
    elif name == "random":
        r = S["random"]
        rA = r["A_layers"]
        apply_surgery_two(m, blocks, [rA[b] for b in blocks], [rA[b + 1] for b in blocks], [r["perms"][b] for b in blocks])
        apply_attention_surgery(m, blocks, [rA[b] for b in blocks], [rA[b + off] for b in blocks],
                                [r["heads"][b] for b in blocks])
    elif name == "svd":
        mlp_r, att_r = svd_ranks(adapter, S, ops, blocks)
        apply_svd_baseline(m, blocks, mlp_r)
        apply_attention_svd(m, blocks, att_r)
    else:
        raise ValueError(name)
    return m


def svd_ranks(adapter, S, ops, blocks):
    """Ranks matched to the V-local mirror counts: MLP as in run_e1 (fc1 with A_b, fc2 with A_b+1),
    attention per matrix (Wq, Wk, Wv with A_b; Wo with the chosen A_out)."""
    d, h = adapter.embed_dim, adapter.mlp_hidden
    H, dh = attn_dims(adapter.model.blocks[0])
    A, perms, maps, off = ops["A_layers"], ops["perms"], S["maps"], S["a_out_offset"]
    mlp_r, att_r = [], []
    for b in blocks:
        tp = perm_trace(perms[b])
        mlp_r.append((svd_rank_for(h, d, equivariant_param_count(h, d, tp, involution_trace64(A[b]))),
                      svd_rank_for(h, d, equivariant_param_count(h, d, tp, involution_trace64(A[b + 1])))))
        att_r.append(attention_svd_ranks(attention_free_params(d, maps[b], A[b], A[b + off]), d, H * dh))
    return mlp_r, att_r


def variant_params(name: str, k: int, adapter, S: dict, ops: dict, n_model: int) -> dict:
    """Matrix parameters of the converted blocks (MLP fc1, fc2; attention Wq, Wk, Wv, Wo), kept and
    dense, and the whole-model count after conversion (biases counted as dense)."""
    d, h = adapter.embed_dim, adapter.mlp_hidden
    H, dh = attn_dims(adapter.model.blocks[0])
    D = H * dh
    A, perms, maps, off = ops["A_layers"], ops["perms"], S["maps"], S["a_out_offset"]
    blocks = list(range(k))
    mlp_dense, att_dense = 2 * h * d * k, 4 * D * d * k
    if name in ("original", "folded") or k == 0:
        mlp_kept, att_kept = mlp_dense, att_dense
    elif name in ("vlocal", "mlp_only"):
        mlp_kept = sum(mlp_mirror_params(h, d, perms[b], A[b], A[b + 1]) for b in blocks)
        att_kept = (sum(attention_free_params(d, maps[b], A[b], A[b + off])["matrices"] for b in blocks)
                    if name == "vlocal" else att_dense)
    elif name == "random":
        r = S["random"]
        rA = r["A_layers"]
        mlp_kept = sum(mlp_mirror_params(h, d, r["perms"][b], rA[b], rA[b + 1]) for b in blocks)
        att_kept = sum(attention_free_params(d, r["heads"][b], rA[b], rA[b + off])["matrices"] for b in blocks)
    elif name == "vexact":
        As = S["A_star"][k]
        mlp_kept = sum(mlp_mirror_params(h, d, perms[b], As, As) for b in blocks)
        att_kept = sum(attention_free_params(d, maps[b], As, As)["matrices"] for b in blocks)
    elif name == "svd":
        mlp_r, att_r = svd_ranks(adapter, S, ops, blocks)
        mlp_kept = sum((r1 + r2) * (h + d) for r1, r2 in mlp_r)
        att_kept = sum(sum(r) * (D + d) for r in att_r)
    else:
        raise ValueError(name)
    dense, kept = mlp_dense + att_dense, mlp_kept + att_kept
    return {"mlp_params_kept": int(mlp_kept), "attn_params_kept": int(att_kept), "converted_params_dense": int(dense),
            "converted_params_kept": int(kept), "converted_kept_frac": kept / dense if dense else 1.0,
            "model_params": int(n_model - dense + kept), "model_params_frac": (n_model - dense + kept) / n_model}


# ----------------------------------------------------------------------------------------------
# Stage 2 and 3: exactness and the full-conversion flip check
# ----------------------------------------------------------------------------------------------
def probe_batches(adapter, loader, n: int, device) -> list:
    out, got = [], 0
    for imgs, _ in loader:
        x = normalize_for(adapter, imgs.to(device))
        out.append(x[: n - got])
        got += out[-1].shape[0]
        if got >= n:
            break
    return out


def exactness_stage(a, adapter, data, S, ops, ks, out: Path, device, tags: dict) -> pd.DataFrame:
    path = out / "m1_exactness.csv"
    if path.exists() and not a.force:
        df = pd.read_csv(path)
        if set(ks) <= set(df["k"].astype(int)):
            log(f"exactness from {path}")
            return df
    L = adapter.depth
    perms, maps = ops["perms"], S["maps"]
    xs = probe_batches(adapter, data["probe"], a.exact_images, device)
    reg = register_info(adapter.model, adapter)
    rows = []
    for k in ks:
        m, _ = build_vexact(adapter, S["A_star"][k], range(k), [perms[b] for b in range(k)], [maps[b] for b in range(k)],
                            reg_perm=S["reg_perm"][k], even_head=(k == L))
        acc = {}
        for x in xs:
            for r in stream_equivariance(adapter, m, x, S["A_star"][k], S["reg_perm"][k], reg["start"]):
                c = acc.setdefault(r["layer"], {"max_abs_err": 0.0, "max_abs_ref": 0.0, "err_sq": 0.0, "ref_sq": 0.0})
                c["max_abs_err"] = max(c["max_abs_err"], r["max_abs_err"])
                c["max_abs_ref"] = max(c["max_abs_ref"], r["max_abs_ref"])
                c["err_sq"] += r["err_sq"]
                c["ref_sq"] += r["ref_sq"]
        for layer, c in sorted(acc.items()):
            rel = c["max_abs_err"] / c["max_abs_ref"] if c["max_abs_ref"] > 0 else float("nan")
            rows.append({"k": k, "layer": layer, "in_span": layer <= k, "max_abs_err": c["max_abs_err"],
                         "max_abs_ref": c["max_abs_ref"], "rel_max": rel,
                         "rel_fro": math.sqrt(c["err_sq"] / c["ref_sq"]) if c["ref_sq"] > 0 else float("nan"),
                         "pass": bool(rel < a.exact_tol) if layer <= k else None,
                         "n_images": int(sum(x.shape[0] for x in xs)), "dtype": str(adapter.dtype), **tags})
        span = [r for r in rows if r["k"] == k and r["in_span"]]
        log(f"exactness k = {k}: max relative error inside the span {max(r['rel_max'] for r in span):.2e} "
            f"({'PASS' if all(r['pass'] for r in span) else 'FAIL'}, tol {a.exact_tol:g})")
        del m
        _free(device)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def flip_stage(a, adapter, data, S, ops, out: Path, device, tags: dict):
    L = adapter.depth
    path = out / "m1_flipcheck.json"
    if L not in S["A_star"]:
        return None
    if path.exists() and not a.force:
        log(f"flip check from {path}")
        return json.loads(path.read_text())
    perms, maps = ops["perms"], S["maps"]
    res = {"k": L, "dtype": "float32", "knn_k": a.knn_k, "temperature": a.knn_temperature, **tags}
    xs = probe_batches(adapter, data["probe"], a.exact_images, device)
    for even in (True, False):
        m, _ = build_vexact(adapter, S["A_star"][L], range(L), perms, maps, reg_perm=S["reg_perm"][L], even_head=even)
        bf, by = extract_features(adapter, data["flip_bank"], device, m)
        qf, qy = extract_features(adapter, data["flip_query"], device, m)
        qft, _ = extract_features(adapter, data["flip_query"], device, m, flip=True)
        pred = knn_predict(bf, by, qf, data["num_classes"], a.knn_k, a.knn_temperature)
        pred_t = knn_predict(bf, by, qft, data["num_classes"], a.knn_k, a.knn_temperature)
        inv = [feature_invariance(adapter, m, x) for x in xs]
        r = {"flip_consistency": 100.0 * float((pred == pred_t).float().mean()),
             "top1_subset": 100.0 * float((pred == qy).float().mean()),
             "max_abs_unit_feature_diff": float((qft - qf).abs().max()),
             "cls_feature_rel_max": max(i["rel_max"] for i in inv),
             "n_bank": int(bf.shape[0]), "n_query": int(qf.shape[0])}
        if even:
            r["pass"] = bool(r["flip_consistency"] == 100.0)
        res["even_head" if even else "no_even_head"] = r
        del m, bf, qf, qft
        _free(device)
    log(f"full conversion (k = {L}, fp32): flip consistency {res['even_head']['flip_consistency']:.2f}% "
        f"({'PASS' if res['even_head']['pass'] else 'FAIL'}); without the even head "
        f"{res['no_even_head']['flip_consistency']:.2f}%")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    return res


# ----------------------------------------------------------------------------------------------
# Stage 4: zero-shot k-NN
# ----------------------------------------------------------------------------------------------
def knn_stage(a, adapter, data, S, ops, ks, variants, out: Path, device, tags: dict) -> pd.DataFrame:
    path = out / "m1_knn.csv"
    rows = pd.read_csv(path).to_dict("records") if (path.exists() and not a.force) else []
    done = {(int(r["k"]), str(r["variant"])) for r in rows}
    amp = None if a.knn_amp == "none" else a.knn_amp
    n_model = int(sum(p.numel() for p in adapter.model.parameters()))

    def knn(model=None):
        o = evaluate_knn(adapter, data["bank"], data["query"], device, data["num_classes"], k=a.knn_k,
                         temperature=a.knn_temperature, model=model, amp=amp)
        return o["top1"], o["flip_consistency"]

    def save():
        pd.DataFrame(rows).to_csv(path, index=False)

    if (0, "original") not in done:
        cache = _resolve(a.cache_dir) / "knn_original" / f"{a.model}_v2_{amp}.json"
        if not a.tiny and cache.exists():
            c = json.loads(cache.read_text())
            t1, fc, src = c["top1"], c["flip_consistency"], f"cache {cache}"
        else:
            t1, fc = knn()
            src = "computed"
            if not a.tiny:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"top1": t1, "flip_consistency": fc, "git": tags["git_hash"]}))
        rows.append({"k": 0, "variant": "original", "top1": t1, "flip_consistency": fc, "drop_points": 0.0,
                     "source": src, **variant_params("original", 0, adapter, S, ops, n_model), **tags})
        save()
        log(f"original: top-1 {t1:.2f}, flip consistency {fc:.2f} ({src})")
    top1_0 = float(next(r["top1"] for r in rows if r["variant"] == "original"))
    plan = ([(0, "folded")] if "folded" in variants else []) + [(k, v) for k in ks for v in VARIANTS if v in variants]
    for k, v in plan:
        if (k, v) in done:
            continue
        m = make_variant(v, k, adapter, S, ops)
        t1, fc = knn(m)
        del m
        _free(device)
        rows.append({"k": k, "variant": v, "top1": t1, "flip_consistency": fc, "drop_points": top1_0 - t1,
                     "source": "computed", **variant_params(v, k, adapter, S, ops, n_model), **tags})
        save()
        log(f"k = {k} {v}: top-1 {t1:.2f} (drop {top1_0 - t1:.2f}), flip consistency {fc:.2f}")
    df = pd.DataFrame(rows)
    order = {v: i for i, v in enumerate(("original",) + SANITY + VARIANTS)}
    df = df.sort_values(["k", "variant"], key=lambda s: s.map(order) if s.name == "variant" else s).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------
def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _e(x):
    try:
        return f"{float(x):.1e}"
    except (TypeError, ValueError):
        return str(x)


def write_summary(out: Path, meta: dict) -> Path:
    heads = pd.read_csv(out / "m1_heads.csv")
    vx = pd.read_csv(out / "m1_vexact.csv")
    ex = pd.read_csv(out / "m1_exactness.csv") if (out / "m1_exactness.csv").exists() else None
    knn = pd.read_csv(out / "m1_knn.csv") if (out / "m1_knn.csv").exists() else None
    flip = json.loads((out / "m1_flipcheck.json").read_text()) if (out / "m1_flipcheck.json").exists() else None
    choice = json.loads((out / "m1_a_out_choice.json").read_text())
    g = meta["git"]
    L = int(meta["depth"])
    lines = [
        "# M1: attention conversion, zero-shot" + (" (TINY TEST, random weights and synthetic data)" if meta["tiny"] else ""),
        "",
        f"Model `{meta['model']}` ({L} blocks, {meta['heads']} heads of dimension {meta['head_dim']}, "
        f"{meta['n_registers']} registers). Operators and neuron pairings: `{meta['operators']}` "
        f"(sha256 {meta['operators_sha256'][:12]}). Attention statistics: fit seed {meta['fit_seed']} "
        f"({meta['sizes'].get('fit', 'n/a')} images, with their mirrors). Random-structure seed {meta['random_seed']}. "
        f"Git `{g['git_hash'][:10]}`{' (uncommitted changes)' if g.get('git_dirty') else ''}. Written {meta['date']}.",
        "",
        f"Protocol: {meta['protocol']}. Statistics and operators on train images; exactness probe, flip check and "
        f"k-NN queries on validation; k-NN bank of train images. k-NN features in {meta['knn_precision']} for every "
        "variant; exactness and flip check in fp32 with TF32 off.",
        "",
        "## Head structure per block (V-local)",
        "",
        "Pairing score: cosine of the best orthogonal map between head h on the original and head sigma(h) on the "
        "mirror (q and k together; 1 = exact). Head-map error: relative error of that map on q, k and on v, "
        "with the identity map (no head swap, B = C = I) in brackets. Breaking share s_all: share of the "
        "attention matrix energy (Wq, Wk, Wv, Wo) removed by the projection; random: the matched random "
        "structure. Kept: free matrix parameters of the equivariant attention over dense.",
        "",
        "| block | self-heads | twin pairs | pairing score | error q,k (identity) | error v (identity) | s_all | random | MLP s1 / s2 | kept |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in heads.iterrows():
        lines.append(f"| {int(r.block)} | {int(r.n_self)} | {int(r.n_pairs)} | {_f(r.mean_pair_score)} | "
                     f"{_f(r.err_qk)} ({_f(r.err_qk_identity)}) | {_f(r.err_v)} ({_f(r.err_v_identity)}) | "
                     f"{_f(r.s_all)} | {_f(r.s_all_random)} | {_f(r.mlp_s1)} / {_f(r.mlp_s2)} | {_f(r.attn_kept_frac)} |")
    lines += [
        "",
        f"A_out choice for Wo (the mid-block stream has no fitted operator): block-average s_all is "
        f"{_f(choice['mean_s_all_A_l'], 4)} with A_out = A_l and {_f(choice['mean_s_all_A_l+1'], 4)} with "
        f"A_out = A_l+1; kept **{choice['kept']}**. The MLP keeps the E1b convention (fc1 with A_l, fc2 with A_l+1).",
        "",
        "## V-exact: one operator per span",
        "",
        "A* is fitted from the cross-moments of stream layers 0 to k, each divided by its energy tr(G_l). "
        "Agreement: tr(A* A_l)/d averaged over the span. Alignment: tr(A* C_l)/tr(A_l C_l) (fit data, 1 = as "
        "good as the per-layer operator). Shares: means over the span's blocks on the folded model. Exactness: "
        "largest relative error over the span's layers (fp32).",
        "",
        "| k | d+ | d- | agreement | alignment (min) | attention s_all | MLP s1 / s2 | registers | max rel. error | pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in vx.iterrows():
        k = int(r.k)
        err, ok = "n/a", "n/a"
        if ex is not None and (ex["k"] == k).any():
            span = ex[(ex["k"] == k) & (ex["in_span"].astype(bool))]
            err = _e(span["rel_max"].max())
            ok = "yes" if span["pass"].astype(bool).all() else "NO"
        regs = (f"{r.register_perm} ({int(r.register_pairs)} pairs)" if int(r.n_registers) else "none")
        lines.append(f"| {k} | {int(r.d_plus)} | {int(r.d_minus)} | {_f(r.mean_agreement)} | "
                     f"{_f(r.mean_alignment)} ({_f(r.min_alignment)}) | {_f(r.mean_s_all)} | "
                     f"{_f(r.mean_mlp_s1)} / {_f(r.mean_mlp_s2)} | {regs} | {err} | {ok} |")
    emb_cols = [c for c in vx.columns if c.startswith("emb_share_")]
    if emb_cols:
        lines += ["", "Embedding breaking shares (part of each embedding removed by the V-exact projection):", ""]
        for _, r in vx.iterrows():
            lines.append(f"- k = {int(r.k)}: " + ", ".join(f"{c[len('emb_share_'):]} {_f(r[c])}" for c in emb_cols))
    if flip is not None:
        e, n = flip["even_head"], flip["no_even_head"]
        lines += [
            "",
            f"## Full conversion (k = {flip['k']}, fp32)",
            "",
            f"With the final features projected onto the even subspace of A*: flip consistency "
            f"{_f(e['flip_consistency'], 2)}% on {e['n_query']} validation queries (bank {e['n_bank']} train images), "
            f"**{'PASS' if e['pass'] else 'FAIL'}** (must be 100%); largest relative CLS feature difference between an "
            f"image and its mirror {_e(e['cls_feature_rel_max'])}. Without the even projection: flip consistency "
            f"{_f(n['flip_consistency'], 2)}% (the features of the mirror are A* times the original's, not equal). "
            f"Top-1 on this small subset protocol: {_f(e['top1_subset'], 2)}% (not comparable with the main table).",
        ]
    if knn is not None:
        lines += [
            "",
            "## Zero-shot k-NN",
            "",
            "Kept: matrix parameters of the converted blocks (MLP fc1, fc2 and attention Wq, Wk, Wv, Wo) over "
            "dense; SVD and the random structure are matched to Mirror V-local. Model: whole-model parameters "
            "after conversion (biases counted as dense). Flip consistency here is in "
            f"{meta['knn_precision']} over all queries.",
            "",
            "| k | variant | top-1 | drop | flip consistency | kept (converted blocks) | model params |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in knn.iterrows():
            lines.append(f"| {int(r.k)} | {VARIANT_LABELS.get(r.variant, r.variant)} | {_f(r.top1, 2)} | "
                         f"{_f(r.drop_points, 2)} | {_f(r.flip_consistency, 2)} | {_f(r.converted_kept_frac)} | "
                         f"{int(r.model_params):,} ({_f(r.model_params_frac)}) |")
    lines += [
        "",
        "## Notes",
        "",
        "- One fit seed; headline numbers need 3 seeds (X3).",
        "- V-exact uses the fully folded model (every norm an RMSNorm); for DINOv2 the position embedding "
        f"interpolated to the {meta['grid'][0]} x {meta['grid'][1]} grid is written into the model first, so the "
        f"converted model is tied to {meta['img_size']} px inputs. The folded original row checks that folding "
        "alone does not change k-NN.",
        "- Registers: paired by the involution matching on register alignment scores averaged over the span's "
        "layers (layer 0 is the register parameters themselves); unpaired registers are projected to the even "
        "subspace.",
        "- In bf16 the mirrored computation rounds differently, so the fully converted model's bf16 flip "
        "consistency can fall slightly below 100%; the exact check is the fp32 one above.",
        "- SVD keeps biases dense and is matched on matrix parameters, like the E1b MLP comparison.",
        "",
        f"Files: m1_heads.csv, m1_heads_detail.csv, m1_ranks.csv, m1_vexact*.csv, m1_exactness.csv, "
        f"m1_flipcheck.json, m1_knn.csv, m1_structure.npz, m1_meta.json (seed {meta['seed']}, git hash, config).",
    ]
    text = "\n".join(lines) + "\n"
    assert "\u2014" not in text
    path = out / "M1_SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    tiny = a.tiny
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/m1" if tiny else "runs/m1")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    if device.type == "cuda" and a.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(a.memory_fraction))
    torch.backends.cuda.matmul.allow_tf32 = False   # fp32 exactness checks need full fp32 matmuls
    torch.backends.cudnn.allow_tf32 = False         # (the patch embedding is a convolution)
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    timer = StageTimer()
    tags = {"git_hash": git["git_hash"], "seed": a.seed, "fit_seed": a.fit_seed, "random_seed": a.random_seed}
    log(f"M1 {'TINY ' if tiny else ''}run on {device}, out {out}, commit {git['git_hash'][:10]}"
        f"{' (dirty)' if git['git_dirty'] else ''}")

    with timer.stage("load"):
        adapter = build_adapter(a, device)
        data = load_data(a, device)
        L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
        H, dh = attn_dims(adapter.model.blocks[0])
        ks_req = [int(k) for k in (a.ks or ("1,2,4" if tiny else "3,6,9,12")).split(",") if k.strip()]
        ks = [k for k in ks_req if 1 <= k <= L]
        if ks != ks_req:
            log(f"skipping ks outside 1..{L}: {sorted(set(ks_req) - set(ks))}")
        variants = [v.strip() for v in a.variants.split(",") if v.strip()]
        bad = set(variants) - set(SANITY + VARIANTS)
        if bad:
            raise ValueError(f"unknown variants {sorted(bad)}")
        if a.operators:
            ops_path = _resolve(a.operators)
        elif tiny:
            ops_path = out / "tiny_operators.npz"
            if not ops_path.exists() or a.force:
                tiny_operators(adapter, data["fit"], device, ops_path)
        else:
            raise SystemExit("--operators is required (runs/x2/<arm>/results/e1_operators.npz)")
        ops = load_operators(ops_path, L, d, h)
    log(f"model depth {L}, d {d}, h {h}, heads {H} x {dh}, prefix tokens {adapter.n_prefix_tokens}; ks {ks}; "
        f"operators {ops_path}")

    with timer.stage("structure"):
        S = structure_stage(a, adapter, data, ops, ks, out, device, git, tags)
    with timer.stage("exactness"):
        ex = exactness_stage(a, adapter, data, S, ops, ks, out, device, tags)
    flip = None
    if "vexact" in variants or tiny:
        with timer.stage("flip_check"):
            flip = flip_stage(a, adapter, data, S, ops, out, device, tags)
    with timer.stage("knn"):
        knn_df = knn_stage(a, adapter, data, S, ops, ks, variants, out, device, tags)

    reg = register_info(adapter.model, adapter)
    meta = {
        "experiment": "M1 attention conversion (zero-shot)", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": a.model if not tiny else "tiny random timm ViT", "tiny": tiny, "seed": a.seed,
        "fit_seed": a.fit_seed, "random_seed": a.random_seed, "git": git, "env": env_info(device),
        "args": vars(a), "operators": str(ops_path), "operators_sha256": ops["sha256"], "ks": ks,
        "variants": variants, "depth": L, "embed_dim": d, "mlp_hidden": h, "heads": H, "head_dim": dh,
        "n_registers": reg["R"], "protocol": data["protocol"], "sizes": data["sizes"],
        "grid": list(adapter.grid_hw), "img_size": adapter.img_size,
        "knn_amp": a.knn_amp, "exact_tol": a.exact_tol, "a_out_offset": S["a_out_offset"],
        "knn_precision": "bf16" if (device.type == "cuda" and a.knn_amp == "bf16") else "fp32 (autocast only on CUDA)",
        "exactness_pass": {int(k): bool(g[g["in_span"].astype(bool)]["pass"].astype(bool).all())
                           for k, g in ex.groupby("k")},
        "flip_check_pass": None if flip is None else flip["even_head"]["pass"],
        "timing": {"stages": timer.stages, "total_s": timer.total()},
    }
    with open(out / "m1_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    summary = write_summary(out, meta)
    log(f"wrote {summary} in {timer.total() / 60:.1f} min")
    return {"out": str(out), "meta": meta, "knn": knn_df}


if __name__ == "__main__":
    main()
