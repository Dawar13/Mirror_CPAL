#!/usr/bin/env python
"""CORE-02: the law predicts the refund for new groups (predict, then verify). PROJECT_NEXT_PHASE.md CORE-02;
PREREG_NEXT_PHASE.md P2.1 to P2.3.

    # Step 1 (before any surgery): fit every group's structure on the final CORE-01 checkpoints and write the
    # predicted tax of blocks 1 to 6 to results/CORE-02/predictions.json (UTC timestamps). Commit that file.
    python scripts/core02_refund.py --predict-only --gpu-frac 0.1
    # Step 2: zero-shot group-projection surgery on blocks 1 to 6 for every group on every final checkpoint,
    # validation top-1 of the model's own head, the law check and the scores
    python scripts/core02_refund.py --gpu-frac 0.1
    # CPU smoke test (synthetic data, tiny checkpoints, both steps)
    python scripts/core02_refund.py --tiny --out tiny_out/core02

Models: the final checkpoint of every finished run under --runs-root (runs/CORE-01) whose condition is in
--conditions (default none, LR, UD, LRUD, R90: the complete matrix). Scored (PREREG P2.1 to P2.3): the LR+UD
(condition LRUD, trained group Klein) and R90 models. Every other row is reported, not tested.

Structure (fit seed 0, 2,000 ImageNet-100 train images, the machinery of scripts/measure_groups.py): per-layer
operators (fitted_layer_ops) and hidden permutations (fitted_block_perms) for LR, UD, R90 and Klein (R180 on
request). The paired passes come from the CORE-01 measurement of the same checkpoint
(results/CORE-01/<run>/e<NNN>/cache/, so CORE-02 uses exactly the structure CORE-01 reports) when that cache is
valid (same model, fit seed, fit images, top_k, blocks), else they are run here into <out>/passes/<run>/. The
structure is frozen in <out>/structure/<run>__e<NNN>.npz; its sha256 and the checkpoint's go into
predictions.json, and the surgery step refuses to run on anything else.

Prediction (the law, mirror.groups.mlp_free_params): per block b of blocks 1 to 6 and group G,
free = (1/|G|) sum_g tr(rho(g)) fix(sigma(g)) for fc1 (rho_in = the operator of layer b) and fc2 (rho_out = layer
b + 1); tax = 1 - free / (d h). Written per block and summed over the blocks.

Surgery: mirror.groups.apply_group_surgery on the MLPs of blocks 1 to 6 (folded; per-layer operators), on a copy of
the checkpoint. Accuracy: top-1 of the model's own classifier head on the ImageNet-100 validation split (all
images, the centre view of mirror.controlled.val_loader), fp32 by default; also on the LR, UD and R90 views. The
drop against the unmodified model has a paired bootstrap interval (10,000 resamples, seed 0, mirror/bootstrap.py).

Achieved saving (`compressed_mlp`): the surgery output rewritten in compressed form, independently of the formula.
For every orbit of hidden units under the group, the representative unit's fc1 row is stored as coordinates in the
subspace fixed by its stabiliser (an eigenbasis of the stabiliser average of rho_in) and its fc2 column likewise
with rho_out; the other units of the orbit follow by the group action. The free count is the number of stored
coordinates (from the shapes of these arrays), and the compressed form is expanded again and compared with the
surgery output (relative reconstruction error, about 1e-7 for float32 weights). P2.1 (operational): the achieved
and predicted free-parameter counts are equal for fc1 and fc2 in every block of every scored model and group (an
identity: a difference is a bug), with every reconstruction error below 1e-4.

Relations: a surgery group is "trained" if it is the model's training group, "subgroup" if it is a proper subgroup
of it (LR, UD and R180 inside Klein; R180 inside R90), "wrong" otherwise (every group for the condition none).
P2.2: trained-group surgery drops validation top-1 by at most 3 points. P2.3: every wrong-group surgery drops it
by at least 20 points. Subgroup surgeries are reported, not scored.

Outputs (<out>, default results/CORE-02; resumable: finished rows are skipped): predictions.json, structure/,
passes/ (if needed), preds/ (per-image predictions), surgery.csv (one row per model and group), law_per_block.csv,
scores.json, summary.md, meta.json.
"""
from __future__ import annotations

import sys

import torch

GPU_FRAC_DEFAULT = 0.1


def _early_gpu_frac(argv, default: float) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:], GPU_FRAC_DEFAULT))

import argparse  # noqa: E402
import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirror import controlled as C  # noqa: E402
from mirror import groups as GR  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import git_info, log  # noqa: E402

EXPERIMENT = "CORE-02"
TRAINED_GROUP = {"none": None, "LR": "LR", "UD": "UD", "LRUD": "Klein", "R90": "R90", "UD_p0.1": "UD",
                 "UD_p0.25": "UD"}
SUBGROUPS = {"LR": {"LR"}, "UD": {"UD"}, "R180": {"R180"}, "Klein": {"LR", "UD", "R180", "Klein"},
             "R90": {"R180", "R90"}}
SCORED_CONDITIONS = ("LRUD", "R90")
P22_MAX_DROP = 3.0          # PREREG P2.2: trained-group surgery keeps top-1 within 3 points
P23_MIN_DROP = 20.0         # PREREG P2.3: every wrong-group surgery loses at least 20 points
RECON_TOL = 1e-4            # compressed form must reproduce the surgery output (float32 weights)
VIEWS = ("orig", "LR", "UD", "R90")
SURGERY_KEY = ["run", "group"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default=str(ROOT / "runs" / "CORE-01"))
    p.add_argument("--arch", default="vit_s8")
    p.add_argument("--conditions", default="none,LR,UD,LRUD,R90")
    p.add_argument("--groups", default="LR,UD,R90,Klein", help="surgery groups (R180 also possible)")
    p.add_argument("--blocks", default="1-6", help="1-based surgery blocks")
    p.add_argument("--fit-seed", type=int, default=0)
    p.add_argument("--fit-images", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--workers", default="auto", help="processes for the neuron matchings (fresh passes only)")
    p.add_argument("--structure-from", choices=("auto", "core01", "fresh"), default="auto",
                   help="auto: the CORE-01 measurement cache when valid, else fresh passes")
    p.add_argument("--core01-results", default=str(ROOT / "results" / "CORE-01"))
    p.add_argument("--predict-only", action="store_true", help="step 1: write predictions.json and stop")
    p.add_argument("--allow-incomplete", action="store_true", help="use the latest checkpoint of unfinished runs")
    p.add_argument("--amp", choices=("none", "bf16"), default="none", help="evaluation precision (fp32 default)")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--cache-dir", default=None, help="ImageNet-100 cache (default: cache/ or $IN100_CACHE_DIR)")
    p.add_argument("--out", default=str(ROOT / "results" / "CORE-02"))
    p.add_argument("--gpu-frac", type=float, default=GPU_FRAC_DEFAULT)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--tiny", action="store_true", help="synthetic data and tiny checkpoints on the CPU")
    return p.parse_args(argv)


def _mg():
    """scripts/measure_groups.py as a module (its structure machinery is reused unchanged)."""
    spec = importlib.util.spec_from_file_location("measure_groups_core02", ROOT / "scripts" / "measure_groups.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(p)


def _sha(path) -> str:
    from mirror.controlled_sym import file_sha256

    return file_sha256(path)


def relation(group: str, trained: str | None) -> str:
    if trained is None:
        return "wrong"
    if group == trained:
        return "trained"
    return "subgroup" if group in SUBGROUPS[trained] else "wrong"


# ----------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------
def discover_runs(root: Path, arch: str, conditions: list, allow_incomplete: bool) -> list:
    """Finished runs <arch>_<condition>_s<seed> under root with their final checkpoint."""
    from mirror.controlled_sym import final_checkpoint

    out = []
    for d in sorted(Path(root).glob(f"{arch}_*_s*")):
        name = d.name
        rest, _, seed = name[len(arch) + 1:].rpartition("_s")
        if rest not in conditions or not seed.isdigit():
            continue
        try:
            ck = final_checkpoint(d, require_complete=not allow_incomplete)
        except (FileNotFoundError, RuntimeError) as exc:
            log(f"skip {name}: {exc}")
            continue
        epoch = int(ck.stem.split("_e")[-1])
        out.append({"name": name, "dir": d, "condition": rest, "seed": int(seed), "ckpt": ck, "epoch": epoch,
                    "trained_group": TRAINED_GROUP[rest]})
    return out


# ----------------------------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------------------------
class StructCtx:
    """The parts of measure_groups.Ctx that fitted_layer_ops and fitted_block_perms read, with one pass folder per
    group element."""

    def __init__(self, mg, sources: dict, blocks: list, fit_seed: int):
        self.mg, self.sources, self.blocks = mg, sources, list(blocks)
        self.layers = sorted(set(self.blocks) | {b + 1 for b in self.blocks})
        self.a = SimpleNamespace(fit_seed=int(fit_seed))
        self._passes: dict = {}

    def pass_(self, element: str) -> dict:
        if element not in self._passes:
            self._passes[element] = self.mg.load_pass(self.sources[element], element, self.a.fit_seed)
        return self._passes[element]

    def moments(self, element: str, l: int, heldout: bool = False):
        if heldout:
            return None
        src = self.pass_(element)
        i = src["layers"].index(l)
        res = src["res"]
        return {"C": res["C"][i], "sq_x": float(res["sq_x"][i]), "sq_xt": float(res["sq_xt"][i]),
                "n": float(res["n"][i])}


def _pass_usable(path: Path, model: str, fit_seed: int, fit_images: int, top_k: int, blocks: list) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta_json"]))
    except Exception:
        return False
    return (meta.get("model") == model and int(meta.get("fit_seed", -1)) == int(fit_seed)
            and int(meta.get("n_images", -1)) == int(fit_images) and int(meta.get("top_k", -1)) == int(top_k)
            and set(blocks) <= set(int(b) for b in meta.get("blocks", [])))


def build_structure(run: dict, a, mg, device, groups: list, blocks: list, out: Path) -> Path:
    """Fit (or reuse) the structure of every group for one run and freeze it in an npz; returns its path."""
    path = out / "structure" / f"{run['name']}__e{run['epoch']:03d}.npz"
    if path.exists():
        info = load_structure(path)["info"]
        if [int(b) for b in info["blocks"]] != list(blocks) or not set(groups) <= set(info["groups"]):
            raise SystemExit(f"{path} holds blocks {info['blocks']} and groups {sorted(info['groups'])}; "
                             "use another --out for other blocks or groups")
        return path
    t0 = time.time()
    pass_dir = out / "passes" / run["name"]
    ns = mg.parse_args(["--checkpoint", str(run["ckpt"]), "--arch", "controlled", "--out", str(pass_dir),
                        "--fit-seed", str(a.fit_seed), "--fit-images", str(a.fit_images), "--top-k", str(a.top_k),
                        "--workers", str(a.workers), "--no-attn"] + (["--tiny"] if a.tiny else []))
    rt = mg.Runtime(ns, device)
    core_dir = Path(a.core01_results) / run["name"] / f"e{run['epoch']:03d}"
    sources = {}
    for el in GR.generators_needed(groups):
        core_pass = core_dir / "cache" / f"pass_{el}_fit{a.fit_seed}.npz"
        own_pass = pass_dir / "cache" / f"pass_{el}_fit{a.fit_seed}.npz"
        ok = dict(model=rt.model_name, fit_seed=a.fit_seed, fit_images=a.fit_images, top_k=a.top_k, blocks=blocks)
        if a.structure_from in ("auto", "core01") and _pass_usable(core_pass, **ok):
            sources[el] = core_dir
        elif a.structure_from == "core01":
            raise SystemExit(f"{core_pass} missing or not usable (--structure-from core01)")
        else:
            if not _pass_usable(own_pass, **ok):
                mg.run_generator_pass(rt, el, blocks, attn=False, need_inputs=False, out=pass_dir)
            sources[el] = pass_dir
    ctx = StructCtx(mg, sources, blocks, a.fit_seed)
    arrays, info = {}, {"groups": {}, "layers": ctx.layers, "blocks": blocks, "model": rt.model_name,
                        "checkpoint": _rel(run["ckpt"]), "pass_sources": {k: _rel(v) for k, v in sources.items()},
                        "pass_sha256": {el: _sha(Path(src) / "cache" / f"pass_{el}_fit{a.fit_seed}.npz")
                                        for el, src in sources.items()},
                        "fit_seed": a.fit_seed, "fit_images": a.fit_images, "top_k": a.top_k}
    for g in groups:
        ops = mg.fitted_layer_ops(ctx, g)
        perms = mg.fitted_block_perms(ctx, g)
        gi = {"layers": {}, "blocks": {}}
        for l in ctx.layers:
            for i, M in enumerate(ops[l]["gens"]):
                arrays[f"{g}__op__{l}__{i}"] = np.asarray(M, dtype=np.float64)
            gi["layers"][str(l)] = {k: ops[l][k] for k in ("d_plus", "d_minus", "m_plus", "m_minus", "m_i_pairs",
                                                           "m_pp", "m_pm", "m_mp", "m_mm") if k in ops[l]}
        for b in blocks:
            for i, P in enumerate(perms[b]["gens"]):
                arrays[f"{g}__perm__{b}__{i}"] = np.asarray(P, dtype=np.int64)
            gi["blocks"][str(b)] = {k: perms[b][k] for k in ("n_turned", "n_raw_longer", "commute_share_before",
                                                             "n_orbits_fixed") if k in perms[b]}
        gi["n_gens"] = len(GR.GROUP_GENERATORS[g])
        info["groups"][g] = gi
    info["seconds"] = round(time.time() - t0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, meta_json=np.asarray(json.dumps(info, default=float)), **arrays)
    tmp.replace(path)
    del rt
    log(f"structure of {run['name']} ({', '.join(groups)}): {path.name}, {info['seconds']} s")
    return path


def load_structure(path: Path) -> dict:
    """{"info", "ops": {g: {layer: [gens]}}, "perms": {g: {block: [gens]}}}."""
    with np.load(path) as z:
        info = json.loads(str(z["meta_json"]))
        ops, perms = {}, {}
        for g, gi in info["groups"].items():
            n = int(gi["n_gens"])
            ops[g] = {int(l): [z[f"{g}__op__{l}__{i}"] for i in range(n)] for l in info["layers"]}
            perms[g] = {int(b): [z[f"{g}__perm__{b}__{i}"] for i in range(n)] for b in info["blocks"]}
    return {"info": info, "ops": ops, "perms": perms}


def block_actions(st: dict, g: str, blocks: list) -> list:
    return [GR.block_action(g, st["ops"][g][b], st["ops"][g][b + 1], st["perms"][g][b]) for b in blocks]


# ----------------------------------------------------------------------------------------------
# Prediction and the achieved count
# ----------------------------------------------------------------------------------------------
def predict_run(st: dict, groups: list, blocks: list, trained: str | None) -> dict:
    out = {}
    for g in groups:
        rows = []
        for b, act in zip(blocks, block_actions(st, g, blocks)):
            d = int(st["ops"][g][b][0].shape[0])
            h = int(len(st["perms"][g][b][0]))
            f = GR.mlp_free_params(act, h, d)
            rows.append({"block": b + 1, "h": h, "d": d, "free_fc1": f["free_fc1"], "free_fc2": f["free_fc2"],
                         "free_b1": f["free_b1"], "free_b2": f["free_b2"], "dense_fc": f["dense_fc"],
                         "tax_fc1": f["tax_fc1"], "tax_fc2": f["tax_fc2"], "tax_mlp": f["tax_mlp"]})
        free = sum(r["free_fc1"] + r["free_fc2"] for r in rows)
        dense = sum(2 * r["dense_fc"] for r in rows)
        out[g] = {"relation": relation(g, trained), "per_block": rows,
                  "blocks_total": {"free_mlp": free, "dense_mlp": dense, "tax_mlp": 1.0 - free / dense}}
    return out


def _fixed_basis(mats: list, tol: float = 0.5) -> np.ndarray:
    """Orthonormal basis [n, r] of the vectors fixed by every matrix in `mats` (a finite group of orthogonal
    matrices): the eigenvectors of their average with eigenvalue 1."""
    P = np.mean(np.stack(mats), axis=0)
    lam, V = np.linalg.eigh(0.5 * (P + P.T))
    return V[:, lam > tol]


def compressed_mlp(W1, b1, W2, b2, act: GR.BlockAction) -> dict:
    """The MLP weights (folded; h x d and d x h) written in compressed form under the group action `act`:
    per hidden orbit, the representative's fc1 row in the basis fixed by its stabiliser (rows: w = w rho_in(s)),
    its fc2 column likewise (columns: rho_out(s) c = c), one b1 value per orbit and b2 in the basis fixed by the
    whole group. Returns the stored counts (free_fc1, free_fc2, free_b1, free_b2) and the relative errors of the
    expanded form against the given weights. Uses P(g) W1 = W1 rho_in(g), i.e. W1[perm_g[i]] = W1[i] rho_in(g),
    and rho_out(g) W2 = W2 P(g), i.e. W2[:, perm_g[i]] = rho_out(g)^T W2[:, i]."""
    W1, W2 = np.asarray(W1, dtype=np.float64), np.asarray(W2, dtype=np.float64)
    b1, b2 = np.asarray(b1, dtype=np.float64), np.asarray(b2, dtype=np.float64)
    h, d = W1.shape
    G = act.order
    perms = np.stack([np.arange(h) if P is None else np.asarray(P, dtype=np.int64) for P in act.perms])
    rin = [np.eye(d) if R is None else np.asarray(R, dtype=np.float64) for R in act.rho_in]
    rout = [np.eye(d) if R is None else np.asarray(R, dtype=np.float64) for R in act.rho_out]
    seen = np.zeros(h, dtype=bool)
    by_stab: dict = {}
    n_orbits = 0
    for i in range(h):
        if seen[i]:
            continue
        seen[np.unique(perms[:, i])] = True
        n_orbits += 1
        by_stab.setdefault(tuple(int(k) for k in np.flatnonzero(perms[:, i] == i)), []).append(i)
    W1r, W2r, b1r = np.zeros_like(W1), np.zeros_like(W2), np.zeros_like(b1)
    free1 = free2 = 0
    stab_rows = []
    for stab, reps in sorted(by_stab.items()):
        reps = np.asarray(reps, dtype=np.int64)
        B1 = _fixed_basis([rin[k].T for k in stab])
        B2 = _fixed_basis([rout[k] for k in stab])
        c1 = W1[reps] @ B1                      # stored fc1 coordinates [orbits, r1]
        c2 = W2[:, reps].T @ B2                 # stored fc2 coordinates [orbits, r2]
        w0, v0 = c1 @ B1.T, B2 @ c2.T
        for g in range(G):
            W1r[perms[g, reps]] = w0 @ rin[g]
            W2r[:, perms[g, reps]] = rout[g].T @ v0
            b1r[perms[g, reps]] = b1[reps]
        free1 += int(c1.size)
        free2 += int(c2.size)
        stab_rows.append({"stabiliser": list(stab), "orbits": int(len(reps)), "r_fc1": int(B1.shape[1]),
                          "r_fc2": int(B2.shape[1])})
    Bb = _fixed_basis(rout)
    b2r = Bb @ (Bb.T @ b2)

    def rel(a, b):
        n = float(np.linalg.norm(a))
        return float(np.linalg.norm(a - b)) / n if n > 0 else float(np.linalg.norm(a - b))

    return {"free_fc1": free1, "free_fc2": free2, "free_b1": n_orbits, "free_b2": int(Bb.shape[1]),
            "recon_fc1": rel(W1, W1r), "recon_fc2": rel(W2, W2r), "recon_b1": rel(b1, b1r), "recon_b2": rel(b2, b2r),
            "stabilisers": stab_rows}


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def load_val(cache_dir, batch_size: int) -> tuple:
    """Every validation image (centre view, uint8 [N, 3, 64, 64]) and its label, via controlled.val_loader."""
    xs, ys = [], []
    for x, y in C.val_loader(None, batch_size=batch_size, cache_dir=cache_dir):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys).numpy()


@torch.no_grad()
def evaluate(model, val_u8: torch.Tensor, labels: np.ndarray, device, amp: str, batch_size: int) -> dict:
    """Per-image predictions and top-1 (%) of the model's own head on each view (the views of controlled.val_loader:
    centre crop, then the transform)."""
    model.eval()
    preds = {v: [] for v in VIEWS}
    for s in range(0, len(labels), int(batch_size)):
        base = val_u8[s:s + batch_size].to(device)
        for v in VIEWS:
            with C.autocast_ctx(device, amp):
                logits = model(C.normalise(C.apply_view(base, v)))
            preds[v].append(logits.argmax(-1).cpu())
    preds = {v: torch.cat(p).numpy() for v, p in preds.items()}
    return {"pred": preds, "top1": {v: 100.0 * float((p == labels).mean()) for v, p in preds.items()},
            "n": int(len(labels))}


# ----------------------------------------------------------------------------------------------
# Scores and summary
# ----------------------------------------------------------------------------------------------
def score(rows: list, law: list, scored_conditions=SCORED_CONDITIONS) -> dict:
    out = {"definitions": {
        "P2.1": "achieved free parameters (compressed form of the surgery output, from array shapes) equal the "
                "predicted ones (character formula) for fc1 and fc2 in every block 1 to 6 of every group, and every "
                f"reconstruction error is below {RECON_TOL:g}",
        "P2.2": f"trained-group surgery on blocks 1 to 6: drop in validation top-1 at most {P22_MAX_DROP:g} points",
        "P2.3": f"every wrong-group surgery on blocks 1 to 6: drop at least {P23_MIN_DROP:g} points",
        "scored models": f"final checkpoints of the conditions {list(scored_conditions)} (LR+UD and R90)",
        "subgroup": "a proper subgroup of the trained group (LR, UD, R180 in Klein; R180 in R90): reported only"},
        "models": {}, "all_models_law": {}}
    by_run: dict = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    law_by: dict = {}
    for r in law:
        law_by.setdefault((r["run"], r["group"]), []).append(r)
    for run, rr in sorted(by_run.items()):
        cond = rr[0]["condition"]
        lw = [x for r in rr for x in law_by.get((run, r["group"]), [])]
        law_ok = bool(lw) and all(bool(x["counts_equal"]) and float(x["recon_max"]) < RECON_TOL for x in lw)
        entry = {"condition": cond, "trained_group": rr[0]["trained_group"],
                 "P2.1": {"pass": law_ok, "n_block_rows": len(lw),
                          "max_abs_diff_pp": max((abs(float(x["diff_pp"])) for x in lw), default=float("nan")),
                          "max_recon": max((float(x["recon_max"]) for x in lw), default=float("nan"))}}
        trained = [r for r in rr if r["relation"] == "trained"]
        wrong = [r for r in rr if r["relation"] == "wrong"]
        entry["P2.2"] = ({"pass": all(float(r["drop_points"]) <= P22_MAX_DROP for r in trained),
                          "drops": {r["group"]: float(r["drop_points"]) for r in trained}} if trained else
                         {"pass": None, "note": "no trained-group row"})
        entry["P2.3"] = ({"pass": all(float(r["drop_points"]) >= P23_MIN_DROP for r in wrong),
                          "drops": {r["group"]: float(r["drop_points"]) for r in wrong}} if wrong else
                         {"pass": None, "note": "no wrong-group row"})
        entry["subgroup_drops"] = {r["group"]: float(r["drop_points"]) for r in rr if r["relation"] == "subgroup"}
        if cond in scored_conditions:
            out["models"][run] = entry
        out["all_models_law"][run] = entry["P2.1"]
    for key in ("P2.1", "P2.2", "P2.3"):
        vals = [m[key]["pass"] for m in out["models"].values()]
        conds = {m["condition"] for m in out["models"].values()}
        out[key] = {"pass": (all(v is True for v in vals) if vals and None not in vals else None),
                    "n_models": len(vals), "conditions_present": sorted(conds),
                    "complete": set(scored_conditions) <= conds}
    return out


def write_summary(out: Path, rows: list, scores: dict, preds_meta: dict, groups: list, blocks_tag: str) -> None:
    def fmt(x, nd=2):
        try:
            return f"{float(x):.{nd}f}"
        except (TypeError, ValueError):
            return "n/a"

    def verdict(v):
        return {True: "PASS", False: "FAIL", None: "not scored (incomplete)"}[v]

    by_run: dict = {}
    for r in rows:
        by_run.setdefault(r["run"], {})[r["group"]] = r
    lines = ["# CORE-02: the law predicts the refund for new groups", "",
             f"Zero-shot group-projection surgery on the MLPs of blocks {blocks_tag} (per-layer fitted operators and",
             "hidden permutations, fit seed 0) on the final CORE-01 checkpoints. Top-1 of the model's own head on the",
             "ImageNet-100 validation split; drop = original minus surgery, in points, with a paired bootstrap 95%",
             "interval. Predictions were written to predictions.json before any surgery (timestamps below).", "",
             "## Scores (PREREG_NEXT_PHASE.md P2.1 to P2.3)", ""]
    for key in ("P2.1", "P2.2", "P2.3"):
        s = scores[key]
        lines.append(f"- {key}: {verdict(s['pass'])} on {s['n_models']} scored models "
                     f"(conditions present: {', '.join(s['conditions_present']) or 'none'}). {scores['definitions'][key]}.")
    lines += ["", "## Drop per model and surgery group (points; t = trained, s = subgroup, w = wrong)", "",
              "| model | original top-1 | " + " | ".join(groups) + " |", "|---|---|" + "---|" * len(groups)]
    tag = {"trained": "t", "subgroup": "s", "wrong": "w"}
    for run, gr in sorted(by_run.items()):
        r0 = next(iter(gr.values()))
        cells = []
        for g in groups:
            r = gr.get(g)
            cells.append("" if r is None else
                         f"{fmt(r['drop_points'])} [{fmt(r['drop_ci_low'], 1)}, {fmt(r['drop_ci_high'], 1)}] "
                         f"({tag[r['relation']]})")
        lines.append(f"| {run} | {fmt(r0['top1_original'])} | " + " | ".join(cells) + " |")
    lines += ["", "## The law (predicted against achieved MLP tax, blocks together)", "",
              "| model | group | predicted tax | achieved tax | counts equal | max reconstruction error |",
              "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["run"], x["group"])):
        lines.append(f"| {r['run']} | {r['group']} | {fmt(r['tax_predicted'], 4)} | {fmt(r['tax_achieved'], 4)} | "
                     f"{bool(r['counts_equal_all_blocks'])} | {float(r['recon_max']):.1e} |")
    lines += ["", "## Timing of predictions and surgery (UTC)", ""]
    for run, m in sorted(preds_meta.items()):
        first = min((r["surgery_utc"] for r in rows if r["run"] == run), default="not run")
        lines.append(f"- {run}: predicted {m['predicted_utc']}; first surgery {first}")
    lines += ["", "Per-block law rows: law_per_block.csv. Rows: surgery.csv. Scores: scores.json.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Tiny set-up
# ----------------------------------------------------------------------------------------------
def setup_tiny(out: Path) -> tuple:
    """Synthetic ImageNet-100 stand-in and tiny finished 'runs': UD exactly UD-equivariant (canonical projection of
    mirror.controlled_sym), the others at their random initialisation."""
    from mirror import controlled_sym as S

    cache = out / "tiny_data" / "in100_synth"
    if not (cache / "in100_meta.json").exists():
        C.write_synthetic_cache(cache, num_classes=10, train_per_class=8, val_per_class=4, seed=0)
    root = out / "tiny_runs" / "CORE-01"
    for i, cond in enumerate(("UD", "LRUD", "R90", "LR", "none")):
        d = root / f"vit_test8_{cond}_s0"
        if (d / "last_done.flag").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(100 + i)
        model = C.build_model("vit_test8", num_classes=10)
        if cond == "UD":
            S.project_(model, S.canonical_structure(model), "param")
        torch.save({"format": C.CKPT_FORMAT, "kind": "checkpoint", "exp": "CORE-01", "arch": "vit_test8",
                    "condition": cond, "seed": 0, "epoch": 2, "step": 0, "img_size": 64, "patch_size": 8,
                    "num_classes": 10, "model": {k: v.detach().clone() for k, v in model.state_dict().items()}},
                   d / C.checkpoint_name(2))
        (d / "meta.json").write_text(json.dumps({"extra": {"status": "complete"}}))
        (d / "last_done.flag").write_text("tiny\n")
    return root, cache


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.tiny or a.device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    else:
        raise SystemExit("no GPU visible; use --tiny for CPU runs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if a.tiny:
        runs_root, cache = setup_tiny(out)
        a.runs_root, a.arch, a.cache_dir = str(runs_root), "vit_test8", str(cache)
        a.fit_images, a.workers = min(a.fit_images, 20), "1"
        a.n_boot, a.allow_incomplete, a.core01_results = min(a.n_boot, 200), False, str(out / "no_core01")
    if a.cache_dir:
        os.environ["IN100_CACHE_DIR"] = str(a.cache_dir)     # read by controlled.fit_loader inside measure_groups
    git = git_info()
    mg = _mg()
    groups = GR.parse_groups(a.groups)
    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in TRAINED_GROUP]
    if bad:
        raise SystemExit(f"unknown conditions {bad}")
    runs = discover_runs(Path(a.runs_root), a.arch, conditions, a.allow_incomplete)
    if not runs:
        raise SystemExit(f"no finished runs under {a.runs_root} for {conditions}")
    L = int(C.ARCHS[a.arch]["depth"])
    blocks = mg.parse_blocks(a.blocks, L)
    timings = {}

    # ---- step 1: structure and predictions -------------------------------------------------------
    pred_path = out / "predictions.json"
    preds = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else {
        "experiment": EXPERIMENT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definition": "predicted tax of the MLPs of the surgery blocks by the character formula "
                      "(mirror.groups.mlp_free_params): free = (1/|G|) sum_g tr(rho(g)) fix(sigma(g)) for fc1 "
                      "(rho_in, layer b) and fc2 (rho_out, layer b + 1); tax = 1 - free / (d h)",
        "fit_seed": a.fit_seed, "fit_images": a.fit_images, "top_k": a.top_k, "blocks_1based": a.blocks,
        "scored_conditions": list(SCORED_CONDITIONS), "git": git, "models": {}}
    new = 0
    structures = {}
    for run in runs:
        t0 = time.time()
        sp = build_structure(run, a, mg, device, groups, blocks, out)
        structures[run["name"]] = sp
        ck_sha, st_sha = _sha(run["ckpt"]), _sha(sp)
        entry = preds["models"].get(run["name"])
        if entry is not None:
            if entry["checkpoint_sha256"] != ck_sha or entry["structure_sha256"] != st_sha:
                raise SystemExit(f"{run['name']}: checkpoint or structure differs from predictions.json; "
                                 "the prediction must not be rewritten")
            missing = [g for g in groups if g not in entry["groups"]]
            if missing:
                raise SystemExit(f"{run['name']}: groups {missing} were not predicted; predict them in a new folder")
            continue
        st = load_structure(sp)
        preds["models"][run["name"]] = {
            "predicted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checkpoint": _rel(run["ckpt"]),
            "checkpoint_sha256": ck_sha, "epoch": run["epoch"], "condition": run["condition"], "seed": run["seed"],
            "trained_group": run["trained_group"], "scored": run["condition"] in SCORED_CONDITIONS,
            "structure_file": _rel(sp), "structure_sha256": st_sha, "pass_sources": st["info"]["pass_sources"],
            "groups": predict_run(st, groups, blocks, run["trained_group"])}
        new += 1
        timings[f"predict_{run['name']}"] = round(time.time() - t0, 1)
        pred_path.write_text(json.dumps(preds, indent=2, default=float), encoding="utf-8")
        log(f"predicted {run['name']}: " + ", ".join(
            f"{g} {preds['models'][run['name']]['groups'][g]['blocks_total']['tax_mlp']:.4f}" for g in groups))
    log(f"predictions.json: {len(preds['models'])} models ({new} new)")
    if a.predict_only:
        write_meta(out, name="meta_predict.json", experiment=EXPERIMENT, config=vars(a),
                   seeds={"fit_seed": a.fit_seed, "bootstrap": 0}, models={r["name"]: _rel(r["ckpt"]) for r in runs},
                   device=device, wall_seconds=time.time() - t_start,
                   extra={"step": "predict-only", "predictions_sha256": _sha(pred_path), "timings_s": timings})
        return {"predictions": preds, "out": str(out)}

    # ---- step 2: surgery ---------------------------------------------------------------------
    from mirror import variants as V
    from mirror.bootstrap import paired_bootstrap
    from mirror.surgery import fold_mlp

    pred_sha = _sha(pred_path)
    csv = V.ResultsCsv(out / "surgery.csv", SURGERY_KEY)
    law_csv = V.ResultsCsv(out / "law_per_block.csv", ["run", "group", "block"])
    val_u8, labels = load_val(a.cache_dir, a.batch_size)
    (out / "preds").mkdir(exist_ok=True)
    for run in runs:
        name = run["name"]
        if all(csv.done((name, g)) for g in groups):
            continue
        sp = structures[name]
        entry = preds["models"][name]
        st = load_structure(sp)
        model, _ = C.load_checkpoint_model(run["ckpt"], device)
        t0 = time.time()
        orig_path = out / "preds" / f"{name}__original.npz"
        if orig_path.exists():
            with np.load(orig_path) as z:
                orig = {"pred": {v: z[v] for v in VIEWS}, "top1": json.loads(str(z["top1_json"]))}
        else:
            orig = evaluate(model, val_u8, labels, device, a.amp, a.batch_size)
            np.savez(orig_path, top1_json=np.asarray(json.dumps(orig["top1"])), **orig["pred"])
        tag = f"{blocks[0] + 1}-{blocks[-1] + 1}"
        for g in groups:
            if csv.done((name, g)):
                continue
            tg = time.time()
            utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            m = copy.deepcopy(model)
            acts = GR.apply_group_surgery(m, blocks, [st["ops"][g][b] for b in blocks],
                                          [st["ops"][g][b + 1] for b in blocks], [st["perms"][g][b] for b in blocks],
                                          group=g)
            pred_rows = {r["block"]: r for r in entry["groups"][g]["per_block"]}
            free_ach = free_pred = dense = 0
            recon_max, equal_all = 0.0, True
            for b, act in zip(blocks, acts):
                f = {k: v.detach().cpu().numpy() for k, v in fold_mlp(m.blocks[b]).items()}
                cm = compressed_mlp(f["W1"], f["b1"], f["W2"], f["b2"], act)
                pr = pred_rows[b + 1]
                h, d = f["W1"].shape
                eq = cm["free_fc1"] == pr["free_fc1"] and cm["free_fc2"] == pr["free_fc2"]
                rmax = max(cm["recon_fc1"], cm["recon_fc2"], cm["recon_b1"], cm["recon_b2"])
                tax_a = 1.0 - (cm["free_fc1"] + cm["free_fc2"]) / (2.0 * h * d)
                law_csv.add({"run": name, "group": g, "block": b + 1, "condition": run["condition"],
                             "relation": relation(g, run["trained_group"]),
                             "free_fc1_predicted": pr["free_fc1"], "free_fc1_achieved": cm["free_fc1"],
                             "free_fc2_predicted": pr["free_fc2"], "free_fc2_achieved": cm["free_fc2"],
                             "free_b1_predicted": pr["free_b1"], "free_b1_achieved": cm["free_b1"],
                             "free_b2_predicted": pr["free_b2"], "free_b2_achieved": cm["free_b2"],
                             "tax_predicted": pr["tax_mlp"], "tax_achieved": tax_a,
                             "diff_pp": 100.0 * (pr["tax_mlp"] - tax_a), "counts_equal": bool(eq),
                             "recon_fc1": cm["recon_fc1"], "recon_fc2": cm["recon_fc2"], "recon_max": rmax,
                             "stabilisers": json.dumps(cm["stabilisers"]), "git_hash": git["git_hash"]})
                free_ach += cm["free_fc1"] + cm["free_fc2"]
                free_pred += pr["free_fc1"] + pr["free_fc2"]
                dense += 2 * h * d
                recon_max = max(recon_max, rmax)
                equal_all = equal_all and eq
            r = evaluate(m, val_u8, labels, device, a.amp, a.batch_size)
            del m
            np.save(out / "preds" / f"{name}__{g}.npy", r["pred"]["orig"])
            bs = paired_bootstrap(orig["pred"]["orig"] == labels, r["pred"]["orig"] == labels,
                                  n_resamples=a.n_boot, seed=0)
            row = {"run": name, "group": g, "condition": run["condition"], "seed": run["seed"], "epoch": run["epoch"],
                   "trained_group": run["trained_group"] or "none", "relation": relation(g, run["trained_group"]),
                   "scored": run["condition"] in SCORED_CONDITIONS, "blocks": tag,
                   "top1_original": orig["top1"]["orig"], "top1_surgery": r["top1"]["orig"],
                   "drop_points": bs["diff"], "drop_ci_low": bs["ci_low"], "drop_ci_high": bs["ci_high"],
                   "n_val": r["n"], "bootstrap_resamples": bs["n_resamples"],
                   **{f"top1_original_{v}": orig["top1"][v] for v in VIEWS if v != "orig"},
                   **{f"top1_surgery_{v}": r["top1"][v] for v in VIEWS if v != "orig"},
                   "free_predicted": free_pred, "free_achieved": free_ach, "dense_mlp": dense,
                   "tax_predicted": 1.0 - free_pred / dense, "tax_achieved": 1.0 - free_ach / dense,
                   "counts_equal_all_blocks": bool(equal_all), "recon_max": recon_max,
                   "predicted_utc": entry["predicted_utc"], "surgery_utc": utc, "predictions_sha256": pred_sha,
                   "structure_sha256": entry["structure_sha256"], "amp": a.amp, "fit_seed": a.fit_seed,
                   "git_hash": git["git_hash"], "time_s": round(time.time() - tg, 1)}
            csv.add(row)
            log(f"{name} {g} ({row['relation']}): top-1 {r['top1']['orig']:.2f} against {orig['top1']['orig']:.2f} "
                f"(drop {bs['diff']:.2f} [{bs['ci_low']:.2f}, {bs['ci_high']:.2f}]); tax predicted "
                f"{row['tax_predicted']:.4f}, achieved {row['tax_achieved']:.4f}, counts equal {equal_all}, "
                f"reconstruction {recon_max:.1e}")
        timings[f"surgery_{name}"] = round(time.time() - t0, 1)
        del model

    rows = [r for r in csv.rows if r["run"] in {x["name"] for x in runs}]
    law = [r for r in law_csv.rows if r["run"] in {x["name"] for x in runs}]
    scores = score(rows, law)
    scores.update({"predictions_sha256": pred_sha, "git": git, "n_rows": len(rows)})
    (out / "scores.json").write_text(json.dumps(scores, indent=2, default=str), encoding="utf-8")
    write_summary(out, rows, scores, {k: v for k, v in preds["models"].items() if k in {x["name"] for x in runs}},
                  groups, f"{blocks[0] + 1} to {blocks[-1] + 1}")
    write_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"fit_seed": a.fit_seed, "bootstrap": 0},
               models={r["name"]: {"checkpoint": _rel(r["ckpt"]), "condition": r["condition"]} for r in runs},
               device=device, wall_seconds=time.time() - t_start,
               extra={"predictions_sha256": pred_sha, "timings_s": timings,
                      "scores": {k: scores[k] for k in ("P2.1", "P2.2", "P2.3")}})
    log(f"done in {time.time() - t_start:.1f} s: P2.1 {scores['P2.1']['pass']}, P2.2 {scores['P2.2']['pass']}, "
        f"P2.3 {scores['P2.3']['pass']}; {out / 'summary.md'}")
    return {"scores": scores, "rows": rows, "law": law, "out": str(out)}


if __name__ == "__main__":
    main()
