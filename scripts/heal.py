#!/usr/bin/env python
"""M3 healing distillation (PAPER_EXPERIMENTS.md, M3-pilot and M3-grid).

    python scripts/heal.py --config configs/m3_pilot.yaml --method mirror
    python scripts/heal.py --config configs/m3_pilot.yaml --method svd
    python scripts/heal.py --config configs/m3_pilot.yaml --method random
    python scripts/heal.py --config configs/m3_pilot.yaml --tiny --method mirror    # CPU smoke test

Steps: build the teacher (original model); build the student for the method (surgery plus its
constraints); choose the learning rate on the held-out TRAIN images (healval, never validation)
with a short budget per rate; run the full budget with the chosen rate. Writes, under
<out_dir>/<method>/: lr_selection/ (one sub-run per rate and lr_selection.json), final/
(train_log.csv, heldout_log.csv, metrics.json, trained_blocks.pt) and summary.json.

Data (built by scripts/build_train_cache.py): cache/heal256_uint8.npy, uint8 [105000, 256, 256, 3],
and cache/train_index.npz with heal_rows (100k) and healval_rows (5k) as row indices into it.

Setup parts that depend on unfinished experiments are marked TODO: the X2-protocol operator file
(load_structure) and the M1 attention conversion (attention_constraints and the SVD attention
baseline). The trainer itself (mirror/heal.py) is complete.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.2)  # shared GPU: cap before anything is allocated; config sets the final cap

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.adapters import Dinov2Adapter, TimmTinyAdapter  # noqa: E402
from mirror.data import synthetic_dataset  # noqa: E402
from mirror.fit import involution_trace  # noqa: E402
from mirror.heal import (DEFAULT_LRS, HealConfig, HealData, LowRankMLP, MirrorMLPConstraint,  # noqa: E402
                         exact_involution, heal, heal_with_lr_selection)
from mirror.matching import random_pairing  # noqa: E402
from mirror.surgery import (apply_surgery_two, apply_svd_baseline, equivariant_param_count,  # noqa: E402
                            perm_trace, svd_rank_for)
from mirror.utils import REPO_ROOT, env_info, git_info, load_config, log, set_seed, setup_torch  # noqa: E402

METHODS = ("mirror", "svd", "random")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_pilot.yaml"))
    p.add_argument("--method", choices=METHODS, default=None, help="overrides `method` in the config")
    p.add_argument("--tiny", action="store_true", help="tiny random ViT + synthetic data on CPU")
    p.add_argument("--out-dir", default=None, help="overrides `out_dir`; results go to <out_dir>/<method>/")
    p.add_argument("--lr", type=float, default=None, help="skip learning-rate selection and use this rate")
    p.add_argument("--max-steps", type=int, default=None, help="cap the full run (debugging only)")
    p.add_argument("--memory-fraction", type=float, default=None, help="overrides compute.gpu_memory_fraction")
    return p.parse_args(argv)


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# ----------------------------------------------------------------------------------------------
# Teacher and data
# ----------------------------------------------------------------------------------------------
def build_teacher(cfg: dict, device: torch.device):
    crop = int(cfg["heal"]["crop_size"])
    if cfg.get("tiny_mode"):
        adapter = TimmTinyAdapter(img_size=crop, seed=int(cfg["seed"]))
    else:
        m = cfg["model"]
        adapter = Dinov2Adapter(hub_repo=m["hub_repo"], name=m["name"], img_size=int(m.get("img_size", crop)))
    return adapter.to(device=device)


def load_heal_data(cfg: dict) -> HealData:
    d = cfg["data"]
    if cfg.get("tiny_mode"):
        n_train, n_held = int(d["n_train"]), int(d["n_heldout"])
        n_cls = 4
        per = -(-(n_train + n_held) // n_cls)
        images, _ = synthetic_dataset(n_cls, per, int(d["image_size"]), seed=int(cfg["seed"]))
        return HealData(images, np.arange(n_train), np.arange(n_train, n_train + n_held))
    meta = _resolve(d.get("meta", "cache/train_meta.json"))
    if meta.exists() and not json.loads(meta.read_text()).get("complete"):
        raise RuntimeError(f"{meta} says the train cache is incomplete; rerun scripts/build_train_cache.py")
    images = np.load(_resolve(d["images"]), mmap_mode="r")
    with np.load(_resolve(d["index"])) as z:
        train, held = z[d["train_key"]], z[d["heldout_key"]]
    if d.get("max_train"):
        train = train[:int(d["max_train"])]
    if d.get("max_heldout"):
        held = held[:int(d["max_heldout"])]
    log(f"data: {images.shape} {images.dtype}; {len(train)} train rows, {len(held)} held-out rows")
    return HealData(images, train, held)


# ----------------------------------------------------------------------------------------------
# Structure (operators and twin pairings)
# ----------------------------------------------------------------------------------------------
def random_involution_fixing_ones(d: int, d_plus: int, seed: int) -> np.ndarray:
    """Random orthogonal involution with d_plus eigenvalues +1 and A 1 = 1 (so it commutes with
    LayerNorm centring, like a fitted operator). Needs d_plus >= 1."""
    if not 1 <= d_plus <= d:
        raise ValueError("d_plus must be in [1, d]")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(d), rng.standard_normal((d, d - 1))]))
    signs = np.concatenate([[1.0], np.ones(d_plus - 1), -np.ones(d - d_plus)])
    A = (Q * signs) @ Q.T
    return exact_involution(0.5 * (A + A.T))


def load_structure(cfg: dict, adapter) -> dict:
    """Fitted mirror structure: A_layers (L + 1 operators; layer 0 = tokens entering block 0, layer
    l = output of block l - 1) and perms (one neuron involution per block).

    TODO (X2/X3): set structure.operators to the X2-protocol fit (train images, the chosen fit seed).
    This loader reads the E1/E1b layout (e1_operators.npz: A_layers [L+1, d, d], A_shared [d, d],
    perms [L, h]); if the X2 script writes another layout, adapt this function only. For V-exact,
    point structure.shared_key at the energy-normalised span operator once M1 or A1 writes it.
    """
    s = cfg["structure"]
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    if cfg.get("tiny_mode"):
        seed = int(cfg["seed"])
        A = [random_involution_fixing_ones(d, d // 2 + 4 * (l % 3), seed=seed * 100 + l) for l in range(L + 1)]
        perms = [random_pairing(h, h // 3, seed=seed * 100 + 50 + b) for b in range(L)]
        return {"A_layers": A, "perms": perms, "mode": "per_layer", "source": "tiny random structure (smoke test)"}
    if not s.get("operators"):
        raise FileNotFoundError("TODO (X2): set structure.operators to the X2 operator file")
    path = _resolve(s["operators"])
    mode = s.get("mode", "per_layer")
    with np.load(path) as z:
        perms = [np.asarray(p, dtype=np.int64) for p in z["perms"]]
        if mode == "per_layer":
            A_layers = [exact_involution(a) for a in z["A_layers"]]
        elif mode == "shared":
            A = exact_involution(z[s.get("shared_key", "A_shared")])
            A_layers = [A] * (L + 1)
        else:
            raise ValueError("structure.mode must be per_layer or shared")
    if len(A_layers) != L + 1 or len(perms) != L or A_layers[0].shape != (d, d) or len(perms[0]) != h:
        raise ValueError(f"{path}: expected {L + 1} operators of {d} x {d} and {L} perms of length {h}")
    return {"A_layers": A_layers, "perms": perms, "mode": mode, "source": str(path)}


def matched_random_structure(structure: dict, adapter, seed: int, fix_ones: bool = True) -> dict:
    """Random-structure control: per layer a random involution with the fitted d+ and d-, per block a
    random pairing with the fitted numbers of pairs and self-twins."""
    from mirror.fit import random_involution

    d, h = adapter.embed_dim, adapter.mlp_hidden

    def rand_A(A, s):
        dp = int(round((np.trace(A) + d) / 2))
        return random_involution_fixing_ones(d, dp, s) if fix_ones else exact_involution(random_involution(d, dp, s))

    if structure["mode"] == "shared":
        A_layers = [rand_A(structure["A_layers"][0], seed * 1000)] * len(structure["A_layers"])
    else:
        A_layers = [rand_A(A, seed * 1000 + l) for l, A in enumerate(structure["A_layers"])]
    perms = [random_pairing(h, int((p != np.arange(h)).sum() // 2), seed=seed * 1000 + 500 + b)
             for b, p in enumerate(structure["perms"])]
    for p, q in zip(perms, structure["perms"]):
        assert perm_trace(p) == perm_trace(q)
    return {"A_layers": A_layers, "perms": perms, "mode": structure["mode"],
            "source": f"matched random (seed {seed}, fix_ones {fix_ones})"}


def attention_constraints(cfg: dict, structure: dict, blocks: list) -> list:
    """TODO (M1): return the attention projector(s) for the converted blocks, as callables
    c(model) with optional prepare / frozen_parameters / check (see mirror/heal.py). The student
    builder must also apply the attention surgery before training."""
    return []


# ----------------------------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------------------------
def method_setup(method: str, adapter, structure: dict, blocks: list, cfg: dict):
    """Return (make_student, info). make_student() gives a fresh (student_model, constraints)."""
    s = cfg["structure"]
    if s.get("attention", True):
        raise NotImplementedError(
            "TODO (M1): attention conversion is not implemented yet. The M3 pilot converts MLP plus attention: "
            "add the M1 attention surgery and projector (attention_constraints) and a parameter-matched SVD of "
            "the attention weights for the svd arm, then remove this check. For an MLP-only run (A6), set "
            "structure.attention: false.")
    h, d = adapter.mlp_hidden, adapter.embed_dim
    fit_in = [structure["A_layers"][b] for b in blocks]
    fit_out = [structure["A_layers"][b + 1] for b in blocks]
    fit_perms = [structure["perms"][b] for b in blocks]
    eq = [(equivariant_param_count(h, d, perm_trace(p), involution_trace(Ai)),
           equivariant_param_count(h, d, perm_trace(p), involution_trace(Ao)))
          for Ai, Ao, p in zip(fit_in, fit_out, fit_perms)]
    info = {"method": method, "blocks": blocks, "mlp_params_dense": 2 * h * d * len(blocks),
            "structure_source": structure["source"], "structure_mode": structure["mode"]}

    if method in ("mirror", "random"):
        st = structure if method == "mirror" else matched_random_structure(
            structure, adapter, seed=int(s.get("random_seed", 1)), fix_ones=bool(s.get("random_fix_ones", True)))
        A_in = [st["A_layers"][b] for b in blocks]
        A_out = [st["A_layers"][b + 1] for b in blocks]
        perms = [st["perms"][b] for b in blocks]

        def make_student():
            m = copy.deepcopy(adapter.model)
            apply_surgery_two(m, blocks, A_in, A_out, perms)
            return m, [MirrorMLPConstraint(blocks, A_in, A_out, perms), *attention_constraints(cfg, st, blocks)]

        kept = [(equivariant_param_count(h, d, perm_trace(p), involution_trace(a)),
                 equivariant_param_count(h, d, perm_trace(p), involution_trace(b)))
                for a, b, p in zip(A_in, A_out, perms)]
        info.update({"mlp_params_kept": int(sum(a + b for a, b in kept)), "structure_used": st["source"],
                     "n_pairs": [int((p != np.arange(h)).sum() // 2) for p in perms]})
    elif method == "svd":
        ranks = [(svd_rank_for(h, d, e1), svd_rank_for(h, d, e2)) for e1, e2 in eq]

        def make_student():
            m = copy.deepcopy(adapter.model)
            apply_svd_baseline(m, blocks, ranks)
            return m, [LowRankMLP(blocks, ranks)]

        info.update({"ranks": [list(r) for r in ranks],
                     "mlp_params_kept": int(sum((r1 + r2) * (h + d) for r1, r2 in ranks)),
                     "mlp_params_mirror": int(sum(a + b for a, b in eq))})
    else:
        raise ValueError(f"unknown method {method!r}")
    return make_student, info


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    cfg = load_config(a.config, tiny=a.tiny)
    if a.method:
        cfg["method"] = a.method
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    if a.memory_fraction is not None:
        cfg["compute"]["gpu_memory_fraction"] = float(a.memory_fraction)
    method = cfg["method"]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    device = setup_torch(cfg)  # sets the configured GPU memory cap (shared GPU)
    seed = int(cfg["seed"])
    set_seed(seed)
    out = _resolve(cfg["out_dir"]) / method
    out.mkdir(parents=True, exist_ok=True)
    git = git_info()
    log(f"M3 heal: method {method}, blocks {cfg['blocks']}, out {out}, git {git['git_hash'][:10]}")

    teacher = build_teacher(cfg, device)
    data = load_heal_data(cfg)
    blocks = [int(b) for b in cfg["blocks"]]
    structure = load_structure(cfg, teacher)
    make_student, info = method_setup(method, teacher, structure, blocks, cfg)
    hcfg = HealConfig.from_dict({**cfg.get("heal", {}), "blocks": blocks, "seed": seed})
    if a.max_steps:
        hcfg = replace(hcfg, max_steps=int(a.max_steps))

    sel_cfg = cfg.get("lr_selection", {}) or {}
    if a.lr is not None:
        student, cons = make_student()
        final = heal(teacher, student, data, replace(hcfg, lr=float(a.lr)), cons, out / "final")
        selection = {"skipped": True, "best_lr": float(a.lr), "lrs": [float(a.lr)]}
    else:
        res = heal_with_lr_selection(teacher, make_student, data, hcfg,
                                     lrs=[float(x) for x in sel_cfg.get("lrs", DEFAULT_LRS)],
                                     budget_steps=int(sel_cfg.get("budget_steps", 100)), out_dir=out)
        selection, final, student = res["lr_selection"], res["final"], res["student"]
    # TODO (Tier 1): k-NN top-1 and flip consistency of the healed student come from the evaluation
    # script, which loads final/trained_blocks.pt into a student built by the same make_student().

    final = {k: v for k, v in final.items() if k != "history"}
    summary = {"method": method, "blocks": blocks, "seed": seed, "method_info": info, "lr_selection": selection,
               "final": final, "config": cfg, **git, "env": env_info(device)}
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    ho = final.get("heldout_final") or {}
    log(f"M3 heal done: {method}, lr {selection['best_lr']:.0e}, held-out loss {ho.get('total', float('nan')):.5f}")
    return summary


if __name__ == "__main__":
    main()
