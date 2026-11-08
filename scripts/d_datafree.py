#!/usr/bin/env python
"""Track D: data-free readout pilot (D1, D2, D3) and the pre-registered P5 check (gate G4).

    python scripts/d_datafree.py --model dinov2_vitb14_reg \
        --operators runs/x2/perlayer_reg_s0/results/e1_operators.npz --out runs/d
    python scripts/d_datafree.py --tiny --out tiny_out/d     # planted symmetric tiny ViT, CPU, seconds

Inputs: the model weights (torch.hub; downloads, if any, go under $HF_HOME/torch_hub) and one
protocol-v2 operators file from scripts/run_e1.py (A_layers [L+1, d, d], perms [L, h], G). The
run's e1_meta.json next to the file is read for provenance (model name, fit seed, git hash) and for
the number of fit images. Optional --compare-operators: the same file from other fit seeds, to show
how much the data-driven answers themselves move between seeds.

Everything runs on the CPU in float64; the script never touches the GPU.

D1  layer-0 operator from the patch embedding and position embeddings (mirror/datafree.py). Primary
    choice, fixed before any run on real weights: isotropic N(0, I) prior on the normalised pixels
    of a patch (sigma^2 = 1, the unit variance that ImageNet mean/std normalisation gives each
    pixel). Also reported, never used for the verdict: a sweep over the content energy share rho
    (including rho = 0.5, equal energy), the literal position-only term, a data-calibrated rho, and
    random involutions with the same d+ as the data-driven A_0 (the chance level of the measure).
D2  twin pairing from fc1 rows given A: data-driven A_b for every requested block, the data-free
    A_0^w for block 1, and a random involution (null) for block 1.
D3  (stretch) moment propagation to the outputs of blocks 1 to --d3-blocks, with the weight-only
    pairing (fully data-free chain) and with the data-driven pairing; baselines: copying A_0^w
    forward, and the data-driven drift between neighbouring layers.

P5 (PREREG_E2.md): part 1, mean cosine of the principal angles between the +1 eigenspaces of
A_0^w and the data-driven A_0 >= 0.8; part 2, weight-only pairing in block 1 recovers >= 70% of the
data-driven pairs. Part 2 is evaluated as D2 defines it ("given A": the data-driven A_0) and,
separately, for the fully data-free chain (A_0^w); both are reported.

Block numbers in outputs are 1-based (block 1 = model.blocks[0], whose fc1 reads layer 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.datafree import (clean_involution, compare_operators, content_share_from_data,  # noqa: E402
                             fit_layer0, layer0_parts, match_twins, pair_score_summary,
                             pairing_agreement, plant_mirror_symmetry, propagate_chain,
                             random_involution_fixing_ones, twin_scores)
from mirror.fit import fit_involution, random_involution  # noqa: E402
from mirror.matching import match_many  # noqa: E402
from mirror.surgery import fold_mlp  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402

# Pre-registered in PREREG_E2.md (P5); never edit.
P5_MEAN_COS_EVEN = 0.8
P5_PAIR_RECOVERY = 0.70
SIGMA2_PRIMARY = 1.0   # D1 weighting, fixed before any run on real weights
RHO_SWEEP = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
N_NULL = 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="torch.hub entry")
    p.add_argument("--hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--img-size", type=int, default=224, help="crop size used by the operator run")
    p.add_argument("--operators", default=None, help="e1_operators.npz from scripts/run_e1.py")
    p.add_argument("--meta", default=None, help="e1_meta.json of that run (default: next to the npz)")
    p.add_argument("--compare-operators", nargs="*", default=[],
                   help="operators files from other fit seeds (secondary comparison)")
    p.add_argument("--n-fit-images", type=int, default=None,
                   help="fit images behind G (default: split_sizes.fit from the meta)")
    p.add_argument("--out", default=str(REPO_ROOT / "runs" / "d"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10, help="matching candidates per neuron (E1b used 10)")
    p.add_argument("--workers", default="auto", help="processes for the blossom matchings")
    p.add_argument("--d2-blocks", default="all", help="1-based blocks for D2 with data-driven A, e.g. 1,2,3 (block 1 always)")
    p.add_argument("--d3-blocks", type=int, default=3, help="D3 propagates through blocks 1..n (0 skips D3)")
    p.add_argument("--prereg-commit", default=None, help="commit hash of PREREG_E2.md (if git is missing)")
    p.add_argument("--allow-model-mismatch", action="store_true")
    p.add_argument("--tiny", action="store_true", help="planted mirror-symmetric tiny timm ViT, synthetic data")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Provenance helpers
# ----------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prereg_info(override: str | None) -> dict:
    path = REPO_ROOT / "PREREG_E2.md"
    info = {"file_sha256": sha256(path) if path.exists() else None, "commit": override, "uncommitted": None}

    def run(*a):
        try:
            r = subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    if info["commit"] is None:
        info["commit"] = run("log", "-n", "1", "--format=%H", "--", "PREREG_E2.md") or "unknown (verify)"
    st = run("status", "--porcelain", "--", "PREREG_E2.md")
    info["uncommitted"] = None if st is None else bool(st)
    return info


def parse_blocks(spec: str, L: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(L))
    out = sorted({int(s) - 1 for s in spec.split(",") if s.strip()} | {0})
    if any(b < 0 or b >= L for b in out):
        raise ValueError(f"blocks must be in 1..{L}")
    return out


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------
def load_operators(path: Path) -> dict:
    with np.load(path) as z:
        A_raw = z["A_layers"]
        perms = z["perms"].astype(np.int64)
        trace_G0 = float(np.trace(z["G"][0].astype(np.float64))) if "G" in z.files else None
    A, info = [], []
    for l in range(A_raw.shape[0]):
        a, i = clean_involution(A_raw[l])
        A.append(a)
        info.append(i)
    return {"A": A, "perms": perms, "trace_G0": trace_G0, "clean_info": info, "path": str(path),
            "sha256": sha256(path)}


def load_meta(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_tiny(seed: int):
    """Tiny timm ViT (d 64, h 256, 4 blocks, 56 px) in float64 with a planted mirror symmetry in
    layer 0 and block 1 (d+ = 36; 80 twin pairs and 96 self-twins in block 1)."""
    from mirror.adapters import TimmTinyAdapter

    ad = TimmTinyAdapter(img_size=56, seed=seed, randomise=True, dtype=torch.float64)
    A = random_involution_fixing_ones(ad.embed_dim, 36, seed=seed + 3)
    perm = plant_mirror_symmetry(ad, A, seed=seed + 4, n_pairs=80, block=0)
    return ad, {"A": A, "perm": perm}


def tiny_operators(adapter, out: Path, seed: int, top_k: int) -> tuple[Path, Path]:
    """Data-driven operators for the tiny model from the real data path on synthetic images."""
    from mirror.data import make_loader, synthetic_dataset
    from mirror.stats import collect_fit_stats

    images, labels = synthetic_dataset(4, 12, adapter.img_size, seed=seed + 1)
    loader = make_loader(images, labels, np.arange(len(labels)), 16)
    fs = collect_fit_stats(adapter, loader, torch.device("cpu"), center=True, mlp=True)
    res = fs.residual.numpy()
    L = adapter.depth
    A_layers = np.stack([fit_involution(res["C"][l])[0] for l in range(L + 1)])
    matches = match_many([fs.mlp.correlation(b) for b in range(L)], top_k=top_k, workers=1)
    npz = out / "tiny_operators.npz"
    np.savez_compressed(npz, A_layers=A_layers.astype(np.float32), G=res["G"].astype(np.float32),
                        perms=np.stack([m.perm for m in matches]).astype(np.int32))
    meta = out / "tiny_meta.json"
    meta.write_text(json.dumps({"seed": seed, "git": git_info(), "tiny": True, "protocol": "tiny synthetic",
                                "split_sizes": {"fit": int(len(labels))},
                                "config": {"model": {"name": "tiny_planted"}, "data": {"fit_seed": None}}}, indent=2))
    return npz, meta


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return "nan" if not np.isfinite(x) else f"{float(x):.{nd}f}"
    return str(x)


def table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(fmt(r.get(c[0])) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def write_report(path: Path, R: dict) -> None:
    p5 = R["p5"]
    tiny = R["tiny"]
    pf = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    L = []
    title = "Track D: data-free readout pilot (P5, gate G4)"
    L.append(f"# {title}{' (TINY TEST, planted symmetric model, not a result)' if tiny else ''}\n")
    L.append("## Verdict\n")
    L.append(f"- P5 part 1, layer-0 even subspace: mean cosine of principal angles "
             f"{fmt(p5['part1']['value'])} (bar {P5_MEAN_COS_EVEN}): **{pf(p5['part1']['pass'])}**. "
             f"Chance level (random involutions with the same d+): {fmt(p5['part1']['null_mean'])}. "
             f"d+ {p5['part1']['d_plus_w']} from the weights against {p5['part1']['d_plus_data']} from data; "
             f"overlap {fmt(p5['part1']['overlap_even'])}; tr(A A^w)/d {fmt(p5['part1']['trace_agreement'])}.")
    L.append(f"- P5 part 2, block-1 pairing from the weights given the data-driven A_0: recovers "
             f"{fmt(p5['part2']['value'])} of the data-driven pairs (bar {P5_PAIR_RECOVERY}): **{pf(p5['part2']['pass'])}**.")
    L.append(f"- Fully data-free chain (pairing given A_0^w): recovers {fmt(p5['part2_strict']['value'])}: "
             f"**{pf(p5['part2_strict']['pass'])}**.")
    L.append(f"- P5 as D2 defines it (parts 1 and 2): **{pf(p5['pass'])}**; strict reading (part 1 and the "
             f"fully data-free chain): **{pf(p5['pass_strict'])}**. Gate G4: "
             f"{'passed' if p5['pass'] else 'not passed: claim C6 is dropped unless Dawar decides otherwise'}.\n")
    L.append("PREREG_E2.md does not say whether the block-1 pairing may use the data-driven A_0 (\"given A\" in "
             "PAPER_EXPERIMENTS.md D2) or must use A_0^w; both are shown so the reading can be chosen "
             "without looking at which one passes.\n")

    pv = R["provenance"]
    L.append("## Provenance\n")
    L.append(f"- Model: `{pv['model']}`; operators: `{pv['operators']['path']}` (sha256 {pv['operators']['sha256'][:16]}), "
             f"operator run git {fmt(pv['operators'].get('run_git_hash'))}, fit seed {fmt(pv['operators'].get('fit_seed'))}, "
             f"protocol {fmt(pv['operators'].get('protocol'))}.")
    L.append(f"- This script: git {pv['git']['git_hash']}{' (dirty)' if pv['git'].get('git_dirty') else ''}, seed {pv['seed']}, "
             f"date {pv['date']}, CPU float64.")
    L.append(f"- PREREG_E2.md commit: {pv['prereg']['commit']}; file sha256 {fmt((pv['prereg']['file_sha256'] or '')[:16])}"
             f"{'; WARNING: PREREG_E2.md has uncommitted changes or is untracked' if pv['prereg']['uncommitted'] else ''}.")
    L.append(f"- Layer-0 map affine check (weights against the adapter's own layer-0 tokens): max relative "
             f"difference {R['d1']['affine_rel_err']:.2e}. Stored operators cleaned to exact involutions "
             f"(largest eigenvalue deviation {R['operators_max_eig_dev']:.2e}).\n")

    L.append("## D1: layer-0 operator from the weights\n")
    L.append("C_w = sigma^2 HW J W F W^T J + sum over tokens of J k_t kt_t^T J (isotropic N(0, sigma^2 I) prior on "
             "the normalised pixels of a patch; k = bias, position, CLS and registers), channel-centred as in the data "
             "path. rho is the resulting content share of the token energy. The primary row (sigma^2 = 1) is the only "
             "one used for P5; the others are sensitivity checks. Agreement is with the data-driven A_0.\n")
    L.append(table(R["d1"]["rows"], [("variant", "variant"), ("rho", "rho"), ("d_plus_test", "d+ (w)"),
                                    ("d_plus_ref", "d+ (data)"), ("mean_cos_even", "mean cos even"),
                                    ("mean_cos_odd", "mean cos odd"), ("overlap_even", "overlap even"),
                                    ("trace_agreement", "tr(A A^w)/d")]))
    L.append("")
    if R["d1"].get("calibration"):
        c = R["d1"]["calibration"]
        L.append(f"Data-calibrated content share: rho = {fmt(c['rho'])} (raw {fmt(c['rho_raw'])}, from tr(G_0) and "
                 f"{c['n_images']} fit images). This uses data and is not part of the verdict.\n")
    L.append("Mean cosine is over min(d+ (w), d+ (data)) angles; overlap = ||U^T U^w||_F^2 / max(d+), which also "
             "penalises a dimension mismatch.\n")

    L.append("## D2: twin pairing from the weights\n")
    L.append("S[i, j] = cos(w_i, A w_j) over folded fc1 rows, then the E1b involution matching (top-k "
             f"{R['top_k']}). Recovery = data-driven pairs found / data-driven pairs. Chance = expected recovery of a "
             "random pairing with the same number of pairs.\n")
    L.append(table(R["d2"]["rows"], [("block", "block"), ("operator", "A used"), ("n_pairs_ref", "pairs (data)"),
                                    ("n_pairs_test", "pairs (w)"), ("recovery", "recovery"), ("precision", "precision"),
                                    ("self_kept", "self-twins kept"), ("chance_recovery", "chance"),
                                    ("mean_S_pairs_data", "mean S, data pairs"),
                                    ("mean_S_self_data", "mean S, data self")]))
    L.append("")

    if R.get("seeds"):
        L.append("## Other fit seeds (secondary)\n")
        L.append("The data-driven answers from other fit seeds, compared with the primary seed's data-driven answers "
                 "(the ceiling for any readout) and with the weight-only answers.\n")
        L.append(table(R["seeds"], [("file", "operators"), ("data_A0_vs_primary_mean_cos_even", "A_0 vs primary A_0"),
                                    ("A0w_mean_cos_even", "A_0^w vs this A_0"),
                                    ("data_pairs_vs_primary_recovery", "pairs: primary seed recovers"),
                                    ("w_given_dataA0_recovery", "pairs: weights given A_0"),
                                    ("w_given_A0w_recovery", "pairs: weights given A_0^w")]))
        L.append("")

    if R.get("d3"):
        L.append("## D3 (stretch): propagation to later layers\n")
        L.append("Moment model per token: attention replaced by its value-output map (attention mixing taken as the "
                 "identity), MLP with independent Gaussian pre-activations and twins swapping under the mirror, no "
                 "correlation between the MLP output and the stream it reads. The layer-0 state is the D1 primary one "
                 "(sigma^2 = 1). Agreement with the data-driven A_l (l = output of block l). The propagation is only "
                 "useful where it beats copying A_0^w forward. The scale-check rows show how much the answer depends on "
                 "the absolute stream energy set at layer 0. Exploratory: no pre-registered bar.\n")
        L.append(table(R["d3"]["rows"], [("layer", "layer"), ("variant", "operator"), ("mean_cos_even", "mean cos even"),
                                        ("mean_cos_odd", "mean cos odd"), ("trace_agreement", "tr(A A^w)/d"),
                                        ("d_plus_test", "d+ (w)"), ("d_plus_ref", "d+ (data)")]))
        L.append("")
        if R["d3"]["pairing_rows"]:
            L.append("Pairing in later blocks from the propagated data-free operator:\n")
            L.append(table(R["d3"]["pairing_rows"], [("block", "block"), ("recovery", "recovery"),
                                                    ("precision", "precision"), ("self_kept", "self-twins kept"),
                                                    ("n_pairs_test", "pairs (w)"), ("n_pairs_ref", "pairs (data)")]))
            L.append("")

    L.append("## Caveats\n")
    for c in R["caveats"]:
        L.append(f"- {c}")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)
    set_seed(seed)
    timer = StageTimer()
    git = git_info()
    workers = max(1, min(8, (os.cpu_count() or 2) - 1)) if args.workers == "auto" else int(args.workers)
    log(f"Track D {'TINY ' if args.tiny else ''}run, commit {git['git_hash'][:10]}"
        f"{' (dirty)' if git['git_dirty'] else ''}, seed {seed}, workers {workers}")

    # ------------------------------------------------------------ model and operators
    with timer.stage("load"):
        planted = None
        if args.tiny:
            adapter, planted = build_tiny(seed)
            model_name = "tiny_planted"
            if args.operators is None:
                ops_path, meta_path = tiny_operators(adapter, out, seed, args.top_k)
            else:
                ops_path, meta_path = Path(args.operators), Path(args.meta) if args.meta else None
        else:
            if args.operators is None:
                raise SystemExit("--operators is required (or use --tiny)")
            from mirror.adapters import Dinov2Adapter

            adapter = Dinov2Adapter(hub_repo=args.hub_repo, name=args.model, img_size=args.img_size)
            adapter.to(device=torch.device("cpu"), dtype=torch.float64)
            model_name = args.model
            ops_path = Path(args.operators)
            meta_path = Path(args.meta) if args.meta else ops_path.parent / "e1_meta.json"
        ops = load_operators(ops_path)
        meta = load_meta(meta_path)
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    if len(ops["A"]) != L + 1 or ops["A"][0].shape != (d, d) or ops["perms"].shape != (L, h):
        raise SystemExit(f"operators do not fit the model: A {len(ops['A'])} x {ops['A'][0].shape}, "
                         f"perms {ops['perms'].shape}; model L {L}, d {d}, h {h}")
    run_model = (meta or {}).get("config", {}).get("model", {}).get("name")
    if run_model is not None and run_model != model_name and not args.allow_model_mismatch:
        raise SystemExit(f"operators were fitted on {run_model}, not {model_name}")
    n_fit = args.n_fit_images or (meta or {}).get("split_sizes", {}).get("fit")
    blocks = adapter.blocks
    log(f"model {model_name}: L {L}, d {d}, h {h}, grid {adapter.grid_hw}, prefix tokens {adapter.n_prefix_tokens}; "
        f"operators {ops_path} (fitted on {run_model}, {n_fit} fit images)")

    caveats = [
        "The isotropic prior on patch content is the only data-free choice made; natural patches are far from "
        "isotropic, so D1 tests whether the embedding's own mirror structure dominates.",
        f"The D1 weighting (sigma^2 = {SIGMA2_PRIMARY}: unit-variance isotropic prior on normalised pixels) was fixed in "
        "code before any run on real weights; the rho sweep (with rho = 0.5, equal energy) and the data-calibrated rho "
        "are sensitivity checks only. Real patches are correlated, so the prior understates the content energy that "
        "passes through low-frequency kernels; the calibrated row shows by how much.",
        "The mean-cosine measure has a high chance level when d+ is near d/2 (see the null row); the overlap and "
        "trace columns are harder to satisfy by chance.",
        "Directions that neither the data nor the weights reach (zero eigenvalues of C) are set to +1 in both "
        "operators, which adds a few shared even directions.",
        "Block 1 means model.blocks[0]; its fc1 is compared under A_0 (the block input), as in E1b.",
    ]

    # ------------------------------------------------------------ D1
    with timer.stage("d1"):
        A0 = ops["A"][0]
        parts = layer0_parts(adapter, check_seed=seed)
        if parts["affine_rel_err"] > 1e-8:
            raise SystemExit(f"layer-0 map is not the patch embedding plus a constant (rel err {parts['affine_rel_err']:.2e})")
        d1_rows = []

        def d1_row(variant, terms="full", sigma2=None, rho=None, used_for_p5=False):
            r = fit_layer0(parts, sigma2=sigma2, rho=rho, terms=terms)
            c = compare_operators(A0, r["A"])
            d1_rows.append({"variant": variant, "rho": r["moments"]["rho"], "sigma2": r["moments"]["sigma2"],
                            "terms": terms, "used_for_p5": used_for_p5, "n_zero": r["n_zero"], **c})
            return r, c

        primary, primary_cmp = d1_row("primary: isotropic prior, sigma^2 = 1", sigma2=SIGMA2_PRIMARY, used_for_p5=True)
        for rho in RHO_SWEEP:
            d1_row("sweep: equal energy" if rho == 0.5 else "sweep", rho=rho)
        d1_row("position term only (no bias, no prefix), sigma^2 = 1", terms="position", sigma2=SIGMA2_PRIMARY)
        calib, calib_moments = None, None
        if ops["trace_G0"] is not None and n_fit:
            calib = content_share_from_data(ops["trace_G0"], int(n_fit), parts)
            calib["n_images"] = int(n_fit)
            calib_moments = d1_row("data-calibrated rho (not data-free)", rho=calib["rho"])[0]["moments"]
        d_plus0 = int(round((np.trace(A0) + d) / 2))
        nulls = [compare_operators(A0, random_involution(d, d_plus0, seed=seed + 101 + s)) for s in range(N_NULL)]
        null_row = {"variant": f"null: random involution, same d+ (mean of {N_NULL})", "rho": None,
                    **{k: (int(round(np.mean([n[k] for n in nulls]))) if k.startswith("d_") else
                           float(np.mean([n[k] for n in nulls]))) for k in nulls[0]}}
        d1_rows.append(null_row)
        A0w = primary["A"]
        log(f"D1 primary: d+ {primary['d_plus']} (data {primary_cmp['d_plus_ref']}), mean cos even "
            f"{primary_cmp['mean_cos_even']:.4f}, odd {primary_cmp['mean_cos_odd']:.4f}, "
            f"tr/d {primary_cmp['trace_agreement']:.4f}; null mean cos even {null_row['mean_cos_even']:.4f}")

    # ------------------------------------------------------------ D2
    with timer.stage("d2"):
        d2_blocks = parse_blocks(args.d2_blocks, L)
        W1 = {b: fold_mlp(blocks[b])["W1"] for b in sorted(set(d2_blocks) | {0})}
        jobs = [(b, "data A_b", ops["A"][b]) for b in d2_blocks]
        jobs.append((0, "data-free A_0^w", A0w))
        jobs.append((0, "null: random involution", random_involution(d, d_plus0, seed=seed + 211)))
        d2_rows, results = [], {}
        for start in range(0, len(jobs), workers):
            batch = jobs[start:start + workers]
            S_list = [twin_scores(W1[b], A) for b, _, A in batch]
            matches = match_twins(S_list, top_k=args.top_k, workers=workers)
            for (b, name, _), S, m in zip(batch, S_list, matches):
                ag = pairing_agreement(ops["perms"][b], m.perm)
                sc = pair_score_summary(S, ops["perms"][b])
                d2_rows.append({"block": b + 1, "operator": name, **ag,
                                "mean_S_pairs_data": sc["mean_S_pairs"], "mean_S_self_data": sc["mean_S_self"]})
                results[(b, name)] = m.perm
                log(f"D2 block {b + 1} ({name}): recovery {ag['recovery']:.4f}, precision {ag['precision']:.4f}, "
                    f"self kept {ag['self_kept']:.4f}, pairs {ag['n_pairs_test']} (data {ag['n_pairs_ref']})")
            del S_list
        P1_dataA = results[(0, "data A_b")]
        P1_w = results[(0, "data-free A_0^w")]
        rec_dataA = pairing_agreement(ops["perms"][0], P1_dataA)["recovery"]
        rec_w = pairing_agreement(ops["perms"][0], P1_w)["recovery"]

    # ------------------------------------------------------------ other fit seeds (secondary)
    seed_rows = []
    with timer.stage("seeds"):
        for f in args.compare_operators:
            o = load_operators(Path(f))
            seed_rows.append({
                "file": str(f),
                "data_A0_vs_primary_mean_cos_even": compare_operators(o["A"][0], A0)["mean_cos_even"],
                "A0w_mean_cos_even": compare_operators(o["A"][0], A0w)["mean_cos_even"],
                "A0w_trace_agreement": compare_operators(o["A"][0], A0w)["trace_agreement"],
                "data_pairs_vs_primary_recovery": pairing_agreement(o["perms"][0], ops["perms"][0])["recovery"],
                "w_given_dataA0_recovery": pairing_agreement(o["perms"][0], P1_dataA)["recovery"],
                "w_given_A0w_recovery": pairing_agreement(o["perms"][0], P1_w)["recovery"],
                "sha256": o["sha256"],
            })

    # ------------------------------------------------------------ D3 (stretch)
    d3 = None
    n3 = min(int(args.d3_blocks), L)
    if n3 > 0:
        with timer.stage("d3"):
            state0 = primary["moments"]
            data_perms = [ops["perms"][b] for b in range(n3)]
            chain_w = propagate_chain(blocks, state0, n3, perms=[P1_w] + [None] * (n3 - 1), top_k=args.top_k)
            chain_p = propagate_chain(blocks, state0, n3, perms=data_perms)
            # Scale sensitivity (data-driven pairing, so no extra matchings): the stream's absolute energy
            # sets how much the attention and MLP outputs weigh against the carried stream.
            chain_eq = propagate_chain(blocks, fit_layer0(parts, rho=0.5)["moments"], n3, perms=data_perms)
            chain_cal = propagate_chain(blocks, calib_moments, n3, perms=data_perms) if calib_moments else None
            rows, prow = [], []
            for l in range(1, n3 + 1):
                Ad = ops["A"][l]
                variants = [("propagated, weight-only pairing", chain_w["A"][l]),
                            ("propagated, data-driven pairing", chain_p["A"][l]),
                            ("scale check: data-driven pairing, equal-energy layer 0", chain_eq["A"][l])]
                if chain_cal is not None:
                    variants.append(("scale check: data-driven pairing, data-calibrated layer 0", chain_cal["A"][l]))
                variants += [("baseline: A_0^w copied forward", A0w),
                             ("context: data-driven A_(l-1)", ops["A"][l - 1])]
                for name, Aw in variants:
                    rows.append({"layer": l, "variant": name, **compare_operators(Ad, Aw)})
                by = {r["variant"]: r for r in rows if r["layer"] == l}
                log(f"D3 layer {l}: propagated mean cos even {by[variants[0][0]]['mean_cos_even']:.4f}, "
                    f"tr/d {by[variants[0][0]]['trace_agreement']:.4f} (copy tr/d "
                    f"{by['baseline: A_0^w copied forward']['trace_agreement']:.4f}, data drift tr/d "
                    f"{by['context: data-driven A_(l-1)']['trace_agreement']:.4f})")
            for b in range(1, n3):
                ag = pairing_agreement(ops["perms"][b], chain_w["perms"][b])
                prow.append({"block": b + 1, **ag})
                log(f"D3 pairing block {b + 1}: recovery {ag['recovery']:.4f}")
            d3 = {"rows": rows, "pairing_rows": prow,
                  "energy_mlp_share": [st.get("energy_mlp", float("nan")) / max(np.trace(st["G"]), 1e-300)
                                       for st in chain_w["states"][1:]]}
            np.savez_compressed(out / "d_operators.npz", A0_w=A0w.astype(np.float32),
                                A_w_chain=np.stack(chain_w["A"]).astype(np.float32),
                                A_w_chain_dataP=np.stack(chain_p["A"]).astype(np.float32),
                                perms_w_chain=np.stack(chain_w["perms"]).astype(np.int32),
                                P1_given_dataA0=P1_dataA.astype(np.int32))
    if d3 is None:
        np.savez_compressed(out / "d_operators.npz", A0_w=A0w.astype(np.float32),
                            P1_w=P1_w.astype(np.int32), P1_given_dataA0=P1_dataA.astype(np.int32))

    # ------------------------------------------------------------ P5
    part1 = primary_cmp["mean_cos_even"]
    p5 = {
        "thresholds": {"mean_cos_even": P5_MEAN_COS_EVEN, "pair_recovery": P5_PAIR_RECOVERY},
        "part1": {"value": part1, "pass": bool(part1 >= P5_MEAN_COS_EVEN), "null_mean": null_row["mean_cos_even"],
                  "mean_cos_odd": primary_cmp["mean_cos_odd"], "trace_agreement": primary_cmp["trace_agreement"],
                  "overlap_even": primary_cmp["overlap_even"], "d_plus_w": primary_cmp["d_plus_test"],
                  "d_plus_data": primary_cmp["d_plus_ref"]},
        "part2": {"value": rec_dataA, "pass": bool(rec_dataA >= P5_PAIR_RECOVERY), "operator": "data-driven A_0"},
        "part2_strict": {"value": rec_w, "pass": bool(rec_w >= P5_PAIR_RECOVERY), "operator": "data-free A_0^w"},
    }
    p5["pass"] = bool(p5["part1"]["pass"] and p5["part2"]["pass"])
    p5["pass_strict"] = bool(p5["part1"]["pass"] and p5["part2_strict"]["pass"])
    p5["gate_G4"] = "pass" if p5["pass"] else "fail"
    log(f"P5 part 1 {part1:.4f} ({'PASS' if p5['part1']['pass'] else 'FAIL'}); part 2 {rec_dataA:.4f} "
        f"({'PASS' if p5['part2']['pass'] else 'FAIL'}); strict {rec_w:.4f} ({'PASS' if p5['part2_strict']['pass'] else 'FAIL'})")

    # ------------------------------------------------------------ outputs
    run_git = (meta or {}).get("git", {}).get("git_hash")
    provenance = {
        "model": model_name, "seed": seed, "git": git, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": env_info(torch.device("cpu")), "args": vars(args), "prereg": prereg_info(args.prereg_commit),
        "operators": {"path": str(ops_path), "sha256": ops["sha256"], "run_git_hash": run_git,
                      "fit_seed": (meta or {}).get("fit_seed", (meta or {}).get("config", {}).get("data", {}).get("fit_seed")),
                      "protocol": (meta or {}).get("protocol"), "model": run_model, "n_fit_images": n_fit,
                      "meta_path": str(meta_path) if meta_path else None},
    }
    if args.tiny:
        caveats.insert(0, "TINY TEST: a random tiny ViT with a planted exact mirror symmetry in layer 0 and block 1; "
                          "every number checks the code, none is a result.")
        planted_cmp = {"A0w_vs_planted": compare_operators(planted["A"], A0w),
                       "P1w_vs_planted": pairing_agreement(planted["perm"], P1_w)}
    else:
        planted_cmp = None
    R = {
        "tiny": bool(args.tiny), "provenance": provenance, "top_k": args.top_k, "p5": p5,
        "d1": {"rows": d1_rows, "affine_rel_err": parts["affine_rel_err"], "calibration": calib,
               "primary": {"sigma2": SIGMA2_PRIMARY, "rho": primary["moments"]["rho"], "terms": "full",
                           "d_plus": primary["d_plus"],
                           "d_minus": primary["d_minus"], "n_zero": primary["n_zero"]}},
        "d2": {"rows": d2_rows, "blocks": [b + 1 for b in d2_blocks]},
        "d3": d3, "seeds": seed_rows, "planted": planted_cmp, "caveats": caveats,
        "operators_max_eig_dev": float(max(i["max_eig_dev"] for i in ops["clean_info"])),
        "timing": {"stages": timer.stages, "total_s": timer.total()},
    }
    with open(out / "d_datafree.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    write_report(out / "D_DATAFREE_REPORT.md", R)
    log(f"wrote {out / 'd_datafree.json'} and {out / 'D_DATAFREE_REPORT.md'} in {timer.total():.1f} s")
    return R


if __name__ == "__main__":
    main()
