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

Structure: operators and neuron pairings from the X2-protocol fit (structure.operators, the
run_e1.py e1_operators.npz layout); with structure.attention, head pairings, head maps and the
A_out choice from M1 (structure.heads, the m1_structure.npz written by scripts/m1_attention.py).
The pilot is MLP plus attention, V-local: the attention of block b is projected with A_in = A_b
and A_out = A_b or A_b+1 (the M1 choice) and kept exact by mirror.heal_attn.MirrorAttnConstraint;
the svd arm truncates fc1, fc2, Wq, Wk, Wv and Wo at the mirror parameter counts and keeps the
ranks exact (LowRankMLP, heal_attn.LowRankAttention).

After healing, the healed student (and, with knn.zero_shot, the unhealed one) is scored with the
paper's Tier-1 k-NN protocol (100 train images per class as the bank, all 50k validation images as
queries, bf16, k = 20, T = 0.07); the original's score comes from the cache written by run_e1.py.
Results go to <out_dir>/<method>/knn.json and summary.json.

E3 options (PREREG_E3.md; configs/e3_svd_raw.yaml, configs/e3_svd_sym.yaml), all off by default:
  structure.mode span          one operator A* over blocks 0..k-1 (k = len(blocks)), read from the M1
                               structure file as scripts/heal_vexact.py reads it (load_span_operator).
                               The svd arm then takes its ranks from V-exact's per-block parameter
                               counts (mirror.heal_vexact.vexact_block_params, matched_svd_ranks).
  student.materialise_pos_embed write the position grid of the evaluation size into the student, as
                               V-exact does (same function at that size; mirror.m1.materialise_pos_embed).
  student.train_span_only      with heal.train_scope "all": only the converted blocks and the embeddings
                               train (mirror.heal_vexact.SpanTrainable), V-exact's trainable set.
  targets.final                raw (default: the teacher's own outputs) or flip_average (V-exact's
                               full-conversion targets: the final CLS and patch targets are the flip
                               average of the teacher's features, the loss normalisers are measured on
                               them; block targets stay raw). Needs structure.mode span and k = depth.
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
from mirror.attention import HeadMaps, apply_attention_surgery, attention_free_params, attn_dims  # noqa: E402
from mirror.data import make_loader, make_splits, synthetic_dataset  # noqa: E402
from mirror.fit import involution_trace  # noqa: E402
from mirror.heal import (DEFAULT_LRS, HealConfig, HealData, LowRankMLP, MirrorMLPConstraint,  # noqa: E402
                         exact_involution, heal, heal_with_lr_selection)
from mirror.heal_attn import LowRankAttention, MirrorAttnConstraint  # noqa: E402
from mirror.heal_vexact import SpanTrainable, VExactTeacher, matched_svd_ranks, vexact_block_params  # noqa: E402
from mirror.knn import evaluate_knn  # noqa: E402
from mirror.m1 import apply_attention_svd, attention_svd_ranks, random_head_maps_like, random_head_structure  # noqa: E402
from mirror.m1 import materialise_pos_embed, register_info  # noqa: E402
from mirror.m1 import random_involution_fixing_ones as span_random_involution  # noqa: E402
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
    p.add_argument("--extra-lrs", default=None,
                   help="comma-separated extra rates for the secondary learning-rate check (overrides "
                        "lr_selection.extra_lrs; short trials only, the chosen rate is unchanged)")
    return p.parse_args(argv)


def extra_lrs_of(cfg: dict, cli: str | None = None) -> list:
    """Extra rates of the secondary learning-rate check (E3): --extra-lrs, else lr_selection.extra_lrs. Each
    gets only the short selection trial (mirror.heal.probe_extra_lrs); none may repeat a grid rate."""
    sel = cfg.get("lr_selection", {}) or {}
    vals = [float(v) for v in cli.split(",") if v.strip()] if cli else [float(v) for v in (sel.get("extra_lrs") or [])]
    grid = [float(v) for v in sel.get("lrs", DEFAULT_LRS)]
    if any(v in grid for v in vals):
        raise ValueError(f"extra rates {vals} repeat a grid rate {grid}")
    return vals


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


def load_heads(cfg: dict, adapter, operators_path: str | None) -> dict:
    """M1 head structure for structure.attention: per block a HeadMaps (head involution sigma, maps B
    on q, k and C on v, as fitted; attention.apply_attention_surgery gauges them) and the A_out
    offset (0: Wo projected with A_b, 1: with A_b+1). Read from structure.heads (m1_structure.npz
    from scripts/m1_attention.py); structure.attn_a_out (auto | same | next) can override the
    offset. Tiny: random head structure."""
    s = cfg["structure"]
    L = adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    choice = str(s.get("attn_a_out", "auto"))
    if choice not in ("auto", "same", "next"):
        raise ValueError("structure.attn_a_out must be auto, same or next")
    if cfg.get("tiny_mode"):
        seed = int(cfg["seed"])
        heads = [random_head_structure(H, dh, 1 + b % 2, seed=seed * 100 + 70 + b) for b in range(L)]
        off, src = 1, "tiny random head structure (smoke test)"
    else:
        if not s.get("heads"):
            raise FileNotFoundError("structure.attention is on: set structure.heads to the M1 structure file "
                                    "(runs/m1/m1_structure.npz from scripts/m1_attention.py)")
        path = _resolve(s["heads"])
        with np.load(path) as z:
            heads = [HeadMaps(sigma=np.asarray(z["sigma"][b], dtype=np.int64), B=z["B"][b], C=z["C"][b])
                     for b in range(len(z["sigma"]))]
            off = int(z["a_out_offset"])
            fit_ops = str(z["operators"]) if "operators" in z.files else None
            fit_sha = str(z["operators_sha256"]) if "operators_sha256" in z.files else None
        if len(heads) != L or heads[0].H != H or heads[0].dh != dh:
            raise ValueError(f"{path}: expected {L} blocks of {H} heads of dimension {dh}")
        if operators_path is not None and fit_sha is not None:
            import hashlib

            with open(operators_path, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
            if sha != fit_sha:
                log(f"WARNING: {path} chose A_out with operators {fit_ops}, not {operators_path}")
        src = str(path)
    for hm in heads:
        hm.check(tol=1e-8)
    if choice != "auto":
        off = 0 if choice == "same" else 1
    return {"heads": heads, "a_out_offset": off, "heads_source": src}


def _sha256_file(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_span_operator(cfg: dict, adapter, k: int) -> dict:
    """The V-exact span operator A* and register pairing for blocks 0..k-1 (stream layers 0..k), from the
    M1 structure file (structure.heads, runs/m1/m1_structure.npz: A_star, ks, reg_perm, model), as
    scripts/heal_vexact.py uses them. Returns A_star (exact involution), reg_perm, span_source and
    span_sha256. Tiny: a random span operator (d+ = d/2 + 4) and the identity register pairing."""
    d = adapter.embed_dim
    R = register_info(adapter.model, adapter)["R"]
    if cfg.get("tiny_mode"):
        A = span_random_involution(d, d // 2 + 4, seed=int(cfg["seed"]) * 100 + 90 + int(k))
        return {"A_star": A, "reg_perm": np.arange(R, dtype=np.int64),
                "span_source": "tiny random span operator (smoke test)", "span_sha256": None}
    path = _resolve(cfg["structure"]["heads"])
    with np.load(path) as z:
        if "A_star" not in z.files:
            raise ValueError(f"{path} has no span operators (A_star); rerun scripts/m1_attention.py")
        ks = [int(v) for v in z["ks"]]
        model_name = str(z["model"]) if "model" in z.files else None
        if k not in ks:
            raise ValueError(f"{path} holds span operators for k in {ks}, not {k}: rerun scripts/m1_attention.py "
                             f"with --ks including {k}")
        i = ks.index(k)
        A = exact_involution(z["A_star"][i])
        reg_perm = np.asarray(z["reg_perm"][i], dtype=np.int64)
    if model_name is not None and model_name != cfg["model"]["name"]:
        raise ValueError(f"{path} was fitted on {model_name}, not {cfg['model']['name']}")
    if len(reg_perm) != R:
        raise ValueError(f"{path}: register pairing has {len(reg_perm)} entries, the model has {R} registers")
    return {"A_star": A, "reg_perm": reg_perm, "span_source": str(path), "span_sha256": _sha256_file(path)}


def _with_span_operator(cfg: dict, adapter, out: dict) -> dict:
    """structure.mode span: every layer's operator is the span operator A* for k = len(blocks)."""
    blocks = [int(b) for b in cfg.get("blocks", [])]
    k = len(blocks)
    if not blocks or blocks != list(range(k)):
        raise ValueError("structure.mode span (one operator A* over the span) needs blocks 0..k-1")
    if out.get("heads") is None:
        raise ValueError("structure.mode span needs structure.attention true (V-exact converts MLP and attention)")
    sp = load_span_operator(cfg, adapter, k)
    out.update(sp)
    out.update({"A_layers": [sp["A_star"]] * (adapter.depth + 1), "mode": "span", "span_k": k,
                "source": f"{out['source']} (neuron pairings); span operator A* for k = {k}: {sp['span_source']}"})
    return out


def load_structure(cfg: dict, adapter) -> dict:
    """Fitted mirror structure: A_layers (L + 1 operators; layer 0 = tokens entering block 0, layer
    l = output of block l - 1) and perms (one neuron involution per block), from structure.operators
    (the run_e1.py layout: e1_operators.npz with A_layers [L+1, d, d], A_shared [d, d], perms [L, h]).
    With structure.attention also the M1 head structure (load_heads). With structure.mode span every
    layer gets the V-exact span operator A* (load_span_operator), plus A_star and reg_perm.
    """
    s = cfg["structure"]
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    mode = s.get("mode", "per_layer")
    if cfg.get("tiny_mode"):
        seed = int(cfg["seed"])
        A = [random_involution_fixing_ones(d, d // 2 + 4 * (l % 3), seed=seed * 100 + l) for l in range(L + 1)]
        perms = [random_pairing(h, h // 3, seed=seed * 100 + 50 + b) for b in range(L)]
        out = {"A_layers": A, "perms": perms, "mode": "per_layer", "source": "tiny random structure (smoke test)"}
        if s.get("attention", True):
            out.update(load_heads(cfg, adapter, None))
        return _with_span_operator(cfg, adapter, out) if mode == "span" else out
    if not s.get("operators"):
        raise FileNotFoundError("set structure.operators to the X2 operator file (e1_operators.npz)")
    path = _resolve(s["operators"])
    with np.load(path) as z:
        perms = [np.asarray(p, dtype=np.int64) for p in z["perms"]]
        if mode == "per_layer":
            A_layers = [exact_involution(a) for a in z["A_layers"]]
        elif mode == "shared":
            A = exact_involution(z[s.get("shared_key", "A_shared")])
            A_layers = [A] * (L + 1)
        elif mode == "span":
            A_layers = [np.eye(d)] * (L + 1)   # placeholder for the shape check; replaced by A* below
        else:
            raise ValueError("structure.mode must be per_layer, shared or span")
    if len(A_layers) != L + 1 or len(perms) != L or A_layers[0].shape != (d, d) or len(perms[0]) != h:
        raise ValueError(f"{path}: expected {L + 1} operators of {d} x {d} and {L} perms of length {h}")
    out = {"A_layers": A_layers, "perms": perms, "mode": mode, "source": str(path)}
    if s.get("attention", True):
        out.update(load_heads(cfg, adapter, str(path)))
    return _with_span_operator(cfg, adapter, out) if mode == "span" else out


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
    out = {"A_layers": A_layers, "perms": perms, "mode": structure["mode"],
           "source": f"matched random (seed {seed}, fix_ones {fix_ones})"}
    if structure.get("heads") is not None:
        # Random head involutions with the fitted numbers of twin pairs and self-heads, random maps
        # with the fitted self-head traces (same free-parameter count); same as mirror.m1.
        out["heads"] = [random_head_maps_like(hm, seed * 1000 + 900 + b) for b, hm in enumerate(structure["heads"])]
        out["a_out_offset"] = structure["a_out_offset"]
    return out


def attention_constraints(cfg: dict, structure: dict, blocks: list, model) -> list:
    """M1 attention for the converted blocks (V-local). Projects the attention of `blocks` in `model`
    in place (attention.apply_attention_surgery, A_in = A_b for q, k, v and A_out = A_b or A_b+1 for
    Wo, as chosen by M1) and returns the projector that keeps it exact while training
    (heal_attn.MirrorAttnConstraint, built with the head maps valid after surgery; it freezes norm1's
    affine and ls1). Empty when structure.attention is false."""
    if not cfg["structure"].get("attention", True):
        return []
    if structure["mode"] != "per_layer":
        raise NotImplementedError("attention healing is wired for V-local only; V-exact also needs the full "
                                  "LayerNorm folding and the embedding projection (mirror.m1.build_vexact)")
    off = int(structure["a_out_offset"])
    A_in = [structure["A_layers"][b] for b in blocks]
    A_out = [structure["A_layers"][b + off] for b in blocks]
    maps = apply_attention_surgery(model, blocks, A_in, A_out, [structure["heads"][b] for b in blocks], gauge=True)
    return [MirrorAttnConstraint(blocks, A_in, A_out, maps)]


# ----------------------------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------------------------
def method_setup(method: str, adapter, structure: dict, blocks: list, cfg: dict):
    """Return (make_student, info). make_student() gives a fresh (student_model, constraints).
    The student options (student.materialise_pos_embed, student.train_span_only) apply to every method."""
    make_student, info = _method_setup(method, adapter, structure, blocks, cfg)
    st = cfg.get("student") or {}
    materialise, span_only = bool(st.get("materialise_pos_embed", False)), bool(st.get("train_span_only", False))
    if not (materialise or span_only):
        return make_student, info
    if span_only and str((cfg.get("heal") or {}).get("train_scope", "converted")) != "all":
        raise ValueError("student.train_span_only trains the embeddings too: set heal.train_scope to all")
    base = make_student

    def make_student():
        m, cons = base()
        if materialise:
            materialise_pos_embed(m, adapter)
        return m, ([*cons, SpanTrainable(blocks, embeddings=True)] if span_only else cons)

    info["student"] = {"materialise_pos_embed": materialise, "train_span_only": span_only}
    return make_student, info


def _method_setup(method: str, adapter, structure: dict, blocks: list, cfg: dict):
    s = cfg["structure"]
    attention = bool(s.get("attention", True))
    h, d = adapter.mlp_hidden, adapter.embed_dim
    H, dh = attn_dims(adapter.model.blocks[0])
    if structure["mode"] == "span" and method != "svd":
        raise ValueError("structure.mode span is for the SVD arm matched to V-exact (E3); heal V-exact and its "
                         "random-structure control with scripts/heal_vexact.py")
    fit_in = [structure["A_layers"][b] for b in blocks]
    fit_out = [structure["A_layers"][b + 1] for b in blocks]
    fit_perms = [structure["perms"][b] for b in blocks]
    eq = [(equivariant_param_count(h, d, perm_trace(p), involution_trace(Ai)),
           equivariant_param_count(h, d, perm_trace(p), involution_trace(Ao)))
          for Ai, Ao, p in zip(fit_in, fit_out, fit_perms)]
    info = {"method": method, "blocks": blocks, "mlp_params_dense": 2 * h * d * len(blocks),
            "structure_source": structure["source"], "structure_mode": structure["mode"], "attention": attention}
    att_free = []
    if attention:
        off = int(structure["a_out_offset"])
        att_free = [attention_free_params(d, structure["heads"][b], structure["A_layers"][b],
                                          structure["A_layers"][b + off]) for b in blocks]
        info.update({"attn_params_dense": 4 * H * dh * d * len(blocks), "attn_a_out_offset": off,
                     "attn_a_out": "A_b+1" if off else "A_b", "heads_source": structure["heads_source"],
                     "attn_params_mirror": int(sum(f["matrices"] for f in att_free))})

    if method in ("mirror", "random"):
        st = structure if method == "mirror" else matched_random_structure(
            structure, adapter, seed=int(s.get("random_seed", 1)), fix_ones=bool(s.get("random_fix_ones", True)))
        A_in = [st["A_layers"][b] for b in blocks]
        A_out = [st["A_layers"][b + 1] for b in blocks]
        perms = [st["perms"][b] for b in blocks]

        def make_student():
            m = copy.deepcopy(adapter.model)
            apply_surgery_two(m, blocks, A_in, A_out, perms)
            return m, [MirrorMLPConstraint(blocks, A_in, A_out, perms), *attention_constraints(cfg, st, blocks, m)]

        kept = [(equivariant_param_count(h, d, perm_trace(p), involution_trace(a)),
                 equivariant_param_count(h, d, perm_trace(p), involution_trace(b)))
                for a, b, p in zip(A_in, A_out, perms)]
        info.update({"mlp_params_kept": int(sum(a + b for a, b in kept)), "structure_used": st["source"],
                     "n_pairs": [int((p != np.arange(h)).sum() // 2) for p in perms]})
        if attention:
            off = int(st["a_out_offset"])
            info.update({"attn_params_kept": int(sum(
                attention_free_params(d, st["heads"][b], st["A_layers"][b], st["A_layers"][b + off])["matrices"]
                for b in blocks)), "n_head_pairs": [int(st["heads"][b].n_pairs) for b in blocks]})
    elif method == "svd":
        ranks = [(svd_rank_for(h, d, e1), svd_rank_for(h, d, e2)) for e1, e2 in eq]
        att_ranks = [attention_svd_ranks(f, d, H * dh) for f in att_free]
        if structure["mode"] == "span":
            # E3: ranks from V-exact's own per-block accounting (the code behind vexact_param_counts), checked
            # against the pilot's rule above, which gives the same numbers when every layer's operator is A*.
            per = vexact_block_params(d, h, structure["A_star"], fit_perms, [structure["heads"][b] for b in blocks])
            match = matched_svd_ranks(per, h, d, H * dh)
            if [tuple(r["mlp_ranks"]) for r in match] != ranks or [tuple(r["attn_ranks"]) for r in match] != att_ranks:
                raise AssertionError("SVD ranks from vexact_block_params differ from the pilot rule")
            mlp_gap = max(r["mlp_rel_gap"] for r in match)
            att_gap = max(r["attn_rel_gap"] for r in match)
            info["vexact_match"] = {
                "rule": "per block and matrix, the largest rank r with r (rows + cols) <= the V-exact count",
                "accounting": "mirror.heal_vexact.vexact_block_params (summed by vexact_param_counts)",
                "d_plus": int(round((np.trace(structure["A_star"]) + d) / 2)), "span_k": len(blocks),
                "mlp_kept_vexact": int(sum(r["mlp_kept_vexact"] for r in match)),
                "attn_kept_vexact": int(sum(r["attn_kept_vexact"] for r in match)),
                "mlp_kept_svd": int(sum(r["mlp_kept_svd"] for r in match)),
                "attn_kept_svd": int(sum(r["attn_kept_svd"] for r in match)),
                "max_block_rel_gap_mlp": mlp_gap, "max_block_rel_gap_attn": att_gap, "per_block": match}
            log(f"SVD matched to V-exact over blocks 0..{len(blocks) - 1}: MLP {info['vexact_match']['mlp_kept_svd']:,} of "
                f"{info['vexact_match']['mlp_kept_vexact']:,}, attention {info['vexact_match']['attn_kept_svd']:,} of "
                f"{info['vexact_match']['attn_kept_vexact']:,}; largest per-block shortfall {mlp_gap:.2%} (MLP), "
                f"{att_gap:.2%} (attention)")

        def make_student():
            m = copy.deepcopy(adapter.model)
            apply_svd_baseline(m, blocks, ranks)
            cons = [LowRankMLP(blocks, ranks)]
            if attention:
                apply_attention_svd(m, blocks, att_ranks)
                cons.append(LowRankAttention(blocks, att_ranks))
            return m, cons

        info.update({"ranks": [list(r) for r in ranks],
                     "mlp_params_kept": int(sum((r1 + r2) * (h + d) for r1, r2 in ranks)),
                     "mlp_params_mirror": int(sum(a + b for a, b in eq))})
        if attention:
            info.update({"attn_ranks": [list(r) for r in att_ranks],
                         "attn_params_kept": int(sum(sum(r) * (H * dh + d) for r in att_ranks))})
    else:
        raise ValueError(f"unknown method {method!r}")
    return make_student, info


FINAL_TARGET_CHOICES = ("raw", "flip_average")


def distill_teacher(cfg: dict, teacher, structure: dict, blocks: list):
    """The teacher the trainer distils from. targets.final raw (default): the original model's adapter
    itself. flip_average: mirror.heal_vexact.VExactTeacher with block_targets "raw" and final_target
    "symmetrised", so the final CLS and patch targets are the flip average of the teacher's final-norm
    features, (f(x) + align(f(mirror x))) / 2, and the fixed loss normalisers are measured on them (as in
    scripts/heal_vexact.py at full conversion); the block targets stay the teacher's own outputs. The
    A*-symmetrised, centred block targets of V-exact are not used: centring belongs to the folded stream,
    and the A* symmetrisation changes the loss only by a constant for a student that has the symmetry,
    so for any other student it adds a pull towards A*-equivariance (PREREG_E3.md, arm 3b)."""
    final = str((cfg.get("targets") or {}).get("final", "raw"))
    if final not in FINAL_TARGET_CHOICES:
        raise ValueError(f"targets.final must be one of {FINAL_TARGET_CHOICES}, not {final!r}")
    if final == "raw":
        return teacher
    L = int(teacher.depth)
    if structure.get("mode") != "span" or list(blocks) != list(range(L)):
        raise ValueError("targets.final flip_average is V-exact's full-conversion target: it needs structure.mode span "
                         f"and blocks 0..{L - 1} (k = depth)")
    vt = VExactTeacher(teacher, structure["A_star"], L, structure["reg_perm"], symmetrise=True,
                       final_target="symmetrised", block_targets="raw")
    log(f"targets: {vt.describe()}")
    return vt


# ----------------------------------------------------------------------------------------------
# Final Tier-1 evaluation (k-NN, paper protocol)
# ----------------------------------------------------------------------------------------------
def knn_setup(cfg: dict, device: torch.device):
    """(bank loader, query loader, number of classes, protocol) for the paper k-NN protocol
    (mirror.protocol.Protocol: bank = 100 train images per class, queries = all validation images).
    Tiny: synthetic images at the crop size."""
    kc = cfg.get("knn", {}) or {}
    if cfg.get("tiny_mode"):
        n_cls = int(kc.get("num_classes", 4))
        per = {"bank": int(kc.get("bank_per_class", 8)), "query": int(kc.get("query_per_class", 4))}
        images, labels = synthetic_dataset(n_cls, sum(per.values()), int(cfg["heal"]["crop_size"]), seed=int(cfg["seed"]) + 1)
        sp = make_splits(labels, per)
        bs = int(kc.get("batch_size", 16))
        return (make_loader(images, labels, sp["bank"], bs), make_loader(images, labels, sp["query"], bs), n_cls,
                "tiny synthetic")
    from mirror.protocol import Protocol

    proto = Protocol(_resolve(kc.get("cache_dir", "cache")), int(kc.get("batch_size", 128)), 16,
                     int(kc.get("num_workers", 4)), device.type == "cuda")
    return proto.bank_loader(), proto.query_loader(), 1000, "v2"


def evaluate_final_knn(cfg: dict, teacher, student, make_student, device: torch.device, out: Path, git: dict) -> dict:
    """k-NN top-1 and flip consistency of the original (cached when available), the unhealed student
    (knn.zero_shot) and the healed student. Writes <out>/knn.json."""
    kc = cfg.get("knn", {}) or {}
    amp = kc.get("amp", "bf16")
    amp = None if amp in (None, "none") else str(amp)
    k, T = int(kc.get("k", 20)), float(kc.get("temperature", 0.07))
    bank, query, n_cls, protocol = knn_setup(cfg, device)

    def run(model=None) -> dict:
        o = evaluate_knn(teacher, bank, query, device, n_cls, k=k, temperature=T, model=model, amp=amp)
        return {"top1": o["top1"], "flip_consistency": o["flip_consistency"], "n_bank": o["n_bank"], "n_query": o["n_query"]}

    res = {"protocol": protocol, "amp": amp, "k": k, "temperature": T, "seed": int(cfg["seed"]),
           "git_hash": git["git_hash"]}
    tiny = bool(cfg.get("tiny_mode"))
    cache = _resolve(kc.get("cache_dir", "cache")) / "knn_original" / f"{cfg['model']['name']}_v2_{amp}.json"
    if not tiny and cache.exists():
        c = json.loads(cache.read_text())
        res["original"] = {"top1": c["top1"], "flip_consistency": c["flip_consistency"], "source": str(cache)}
    else:
        res["original"] = {**run(None), "source": "computed"}
        if not tiny:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"top1": res["original"]["top1"],
                                         "flip_consistency": res["original"]["flip_consistency"], "git": git["git_hash"]}))
    if kc.get("zero_shot", True):
        zs, _ = make_student()
        res["zero_shot"] = run(zs)
        del zs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res["healed"] = run(student)
    for key in ("zero_shot", "healed"):
        if key in res:
            res[key]["drop_points"] = res["original"]["top1"] - res[key]["top1"]
    with open(out / "knn.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    log(f"k-NN: original {res['original']['top1']:.2f}"
        + (f", zero-shot {res['zero_shot']['top1']:.2f}" if "zero_shot" in res else "")
        + f", healed {res['healed']['top1']:.2f} (flip consistency {res['healed']['flip_consistency']:.2f})")
    return res


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
    heal_teacher = distill_teacher(cfg, teacher, structure, blocks)   # the plain teacher unless targets.final says otherwise

    sel_cfg = cfg.get("lr_selection", {}) or {}
    extra_lrs = extra_lrs_of(cfg, a.extra_lrs)
    if extra_lrs and a.lr is not None:
        log(f"--lr given: no grid to compare with, so the extra rates {extra_lrs} are skipped")
    lr_extra = None
    if a.lr is not None:
        student, cons = make_student()
        final = heal(heal_teacher, student, data, replace(hcfg, lr=float(a.lr)), cons, out / "final")
        selection = {"skipped": True, "best_lr": float(a.lr), "lrs": [float(a.lr)]}
    else:
        res = heal_with_lr_selection(heal_teacher, make_student, data, hcfg,
                                     lrs=[float(x) for x in sel_cfg.get("lrs", DEFAULT_LRS)],
                                     budget_steps=int(sel_cfg.get("budget_steps", 100)), out_dir=out,
                                     extra_lrs=extra_lrs)
        selection, final, student = res["lr_selection"], res["final"], res["student"]
        lr_extra = res.get("lr_extra")
    # Tier 1: k-NN of the healed student (and of the unhealed one) with the paper protocol.
    knn_res = None
    if (cfg.get("knn", {}) or {}).get("enabled", True):
        knn_res = evaluate_final_knn(cfg, teacher, student, make_student, device, out, git)

    final = {k: v for k, v in final.items() if k != "history"}
    summary = {"method": method, "blocks": blocks, "seed": seed, "method_info": info, "lr_selection": selection,
               "final": final, "knn": knn_res, "config": cfg, **git, "env": env_info(device)}
    if cfg.get("targets") is not None:
        summary["targets"] = (heal_teacher.describe() if heal_teacher is not teacher else
                              {"type": "raw", "block_targets": "raw (the teacher's own block outputs)",
                               "final_target": "raw (the teacher's own final-norm CLS and patch tokens)"})
    if lr_extra is not None:
        summary["lr_extra"] = {k: lr_extra[k] for k in ("chosen_lr", "extra_lrs", "extra_would_win",
                                                        "winning_extra_lrs", "best_lr_with_extra")}
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    ho = final.get("heldout_final") or {}
    log(f"M3 heal done: {method}, lr {selection['best_lr']:.0e}, held-out loss {ho.get('total', float('nan')):.5f}")
    return summary


if __name__ == "__main__":
    main()
