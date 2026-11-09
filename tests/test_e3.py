"""E3 (PREREG_E3.md): matched healed baselines at full conversion. CPU tests:
  - SVD ranks matched to V-exact per block within 1% at the real dimensions (d 768, hidden 3072, 12 heads of 64)
    with a synthetic structure of the same d+ / d- (700 / 68);
  - the random-structure control is exactly equivariant (tiny model) and has V-exact's parameter counts;
  - both SVD target variants (raw, flip-averaged) run, with V-exact's trainable set;
  - the healing seed changes the data order, crops and flips, and nothing else outside --tiny;
  - the secondary learning-rate check runs short trials only and never changes the chosen rate;
  - the Tier-2 rebuild (scripts/tier2_linear.py) loads the new checkpoints as they were healed."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mirror.heal as heal_mod
from mirror.attention import attn_dims
from mirror.data import mirror
from mirror.heal import HealConfig, HealData, configure_trainable, heal, probe_extra_lrs, select_lr
from mirror.heal_vexact import (SpanTrainable, VExactEmbeddings, VExactTeacher, build_vexact_student,
                                matched_svd_ranks, span_exactness, vexact_block_params, vexact_param_counts)
from mirror.m1 import (feature_invariance, matched_random_span_structure, random_head_structure,
                       random_involution_fixing_ones, random_register_pairing_like, stream_equivariance)
from mirror.matching import random_pairing
from mirror.stats import unflip_tokens
from mirror.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
FAST = ["--max-iter", "60", "--wd-grid", "1e-3,1e-2", "--no-wd-refine"]


def load_script(name):
    key = f"{name}_e3test"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod          # dataclasses need the module registered while it executes
        spec.loader.exec_module(mod)
    return sys.modules[key]


def d_plus(A):
    return int(round((np.trace(A) + A.shape[0]) / 2))


def n_pairs(p):
    p = np.asarray(p)
    return int((p != np.arange(len(p))).sum() // 2)


def tiny_structure(adapter, seed=0):
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=seed + 3)
    perms = [random_pairing(h, h // 3 + 7 * b, seed=seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2, 0, 1)[b % 4], seed=seed + 20 + b) for b in range(L)]
    return A, perms, maps


def heal_data(n_train=16, n_held=8, seed=0):
    rng = np.random.default_rng(seed)
    imgs = rng.integers(0, 256, size=(n_train + n_held, 64, 64, 3), dtype=np.uint8)
    rows = rng.permutation(n_train + n_held)
    return HealData(imgs, train_rows=rows[:n_train], heldout_rows=rows[n_train:])


def cfg(blocks, **kw):
    c = dict(blocks=list(blocks), seed=0, lr=1e-3, max_steps=4, batch_size=8, micro_batch=4, crop_size=56,
             warmup_steps=1, calib_images=8, log_every=0, prefetch=0)
    c.update(kw)
    return HealConfig(**c)


def images(n, seed=0):
    return torch.randn(n, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- parameter match at the real dimensions
D, HID, HEADS, DH = 768, 3072, 12, 64
NEURON_PAIRS = [0, 17, 93, 155, 406, 554, 672, 878, 1013, 1168, 1400, 1536]   # the whole range, both extremes
HEAD_PAIRS = [0, 1, 1, 0, 2, 1, 3, 0, 6, 1, 0, 4]


def real_dims_structure(seed=0):
    A = random_involution_fixing_ones(D, 700, seed=seed + 1)
    perms = [random_pairing(HID, n, seed=seed + 100 + b) for b, n in enumerate(NEURON_PAIRS)]
    heads = [random_head_structure(HEADS, DH, n, seed=seed + 200 + b) for b, n in enumerate(HEAD_PAIRS)]
    return A, perms, heads


def test_svd_ranks_match_vexact_per_block_within_one_percent_at_real_dims():
    A, perms, heads = real_dims_structure()
    assert d_plus(A) == 700 and D - d_plus(A) == 68
    per = vexact_block_params(D, HID, A, perms, heads)
    rows = matched_svd_ranks(per, HID, D, HEADS * DH)
    assert len(rows) == 12
    for p, r in zip(per, rows):
        # V-exact's counts: fc1 = fc2 = (h d + tr P tr A) / 2 with A_in = A_out = A*
        tp = HID - 2 * NEURON_PAIRS[p["block"]]
        assert p["fc1"] == p["fc2"] == (HID * D + tp * (2 * 700 - D)) // 2 and p["mlp"] == 2 * p["fc1"]
        assert p["attn"] == p["Wq"] + p["Wk"] + p["Wv"] + p["Wo"]
        # MLP and attention matched separately, per block, within 1%, never above V-exact
        assert 0 <= r["mlp_rel_gap"] < 0.01 and 0 <= r["attn_rel_gap"] < 0.01, r
        assert r["mlp_kept_svd"] == sum(r["mlp_ranks"]) * (HID + D)
        assert r["attn_kept_svd"] == sum(r["attn_ranks"]) * (HEADS * DH + D)
        # each rank is the largest whose factorisation fits the V-exact count of its matrix
        for rank, key in zip(r["mlp_ranks"], ("fc1", "fc2")):
            assert rank * (HID + D) <= p[key] < (rank + 1) * (HID + D) and 1 <= rank <= D
        for rank, key in zip(r["attn_ranks"], ("Wq", "Wk", "Wv", "Wo")):
            assert rank * (2 * D) <= p[key] < (rank + 1) * (2 * D) and 1 <= rank <= D
    tot_vx = sum(p["mlp"] + p["attn"] for p in per)
    tot_svd = sum(r["mlp_kept_svd"] + r["attn_kept_svd"] for r in rows)
    assert 0 <= (tot_vx - tot_svd) / tot_vx < 0.01


def test_heal_svd_span_mode_at_real_dims_uses_the_vexact_accounting():
    """scripts/heal.py method_setup (svd, structure.mode span) on a real-size ViT-B (meta device, no weights):
    ranks and totals equal matched_svd_ranks on vexact_block_params, whose sums are vexact_param_counts."""
    from timm.models.vision_transformer import VisionTransformer

    from mirror.adapters import ViTAdapter

    with torch.device("meta"):
        m = VisionTransformer(img_size=224, patch_size=14, embed_dim=D, depth=12, num_heads=HEADS, init_values=1e-5,
                              reg_tokens=4)
    ad = ViTAdapter(m, img_size=224, patch_size=14, n_prefix_tokens=int(m.num_prefix_tokens))
    assert (ad.embed_dim, ad.mlp_hidden, attn_dims(m.blocks[0])) == (D, HID, (HEADS, DH))
    A, perms, heads = real_dims_structure()
    hm = load_script("heal")
    c = load_config(CONFIGS / "e3_svd_raw.yaml")
    structure = {"mode": "span", "A_layers": [A] * 13, "A_star": A, "perms": perms, "heads": heads, "a_out_offset": 1,
                 "source": "synthetic", "heads_source": "synthetic"}
    _, info = hm.method_setup("svd", ad, structure, list(range(12)), c)
    vm = info["vexact_match"]
    pc = vexact_param_counts(ad, A, 12, perms, heads)
    assert vm["mlp_kept_vexact"] == pc["mlp_params_kept"] and vm["attn_kept_vexact"] == pc["attn_params_kept"]
    assert vm["mlp_kept_svd"] == info["mlp_params_kept"] and vm["attn_kept_svd"] == info["attn_params_kept"]
    assert vm["max_block_rel_gap_mlp"] < 0.01 and vm["max_block_rel_gap_attn"] < 0.01 and vm["d_plus"] == 700
    want = matched_svd_ranks(vexact_block_params(D, HID, A, perms, heads), HID, D, D)
    assert info["ranks"] == [r["mlp_ranks"] for r in want] and info["attn_ranks"] == [r["attn_ranks"] for r in want]
    assert info["student"] == {"materialise_pos_embed": True, "train_span_only": True}
    # The pilot's own path (no span mode) reports none of the E3 keys.
    _, pilot = hm.method_setup("svd", ad, {**structure, "mode": "per_layer"}, list(range(12)),
                               load_config(CONFIGS / "m3_pilot.yaml"))
    assert "vexact_match" not in pilot and "student" not in pilot and pilot["ranks"] == info["ranks"]


def test_vexact_param_counts_sum_the_block_accounting(tiny64):
    A, perms, maps = tiny_structure(tiny64)
    for k in (2, 4):
        per = vexact_block_params(tiny64.embed_dim, tiny64.mlp_hidden, A, perms[:k], maps[:k])
        pc = vexact_param_counts(tiny64, A, k, perms, maps)
        assert pc["mlp_params_kept"] == sum(p["mlp"] for p in per)
        assert pc["attn_params_kept"] == sum(p["attn"] for p in per)


# ---------------------------------------------------------------- random-structure control
def test_random_span_structure_is_matched_and_random(tiny64):
    A, perms, maps = tiny_structure(tiny64)
    reg = np.array([1, 0, 2, 3])
    r = matched_random_span_structure(A, perms, maps, reg, seed=1)
    assert d_plus(r["A_star"]) == d_plus(A) and np.allclose(r["A_star"] @ np.ones(A.shape[0]), 1.0)
    assert np.allclose(r["A_star"] @ r["A_star"], np.eye(A.shape[0])) and not np.allclose(r["A_star"], A)
    for p, q, hm, hq in zip(perms, r["perms"], maps, r["heads"]):
        assert n_pairs(q) == n_pairs(p) and not np.array_equal(p, q)
        assert hq.n_pairs == hm.n_pairs and hq.n_self == hm.n_self
    assert n_pairs(r["reg_perm"]) == 1 and np.array_equal(r["reg_perm"][r["reg_perm"]], np.arange(4))
    d, h = tiny64.embed_dim, tiny64.mlp_hidden
    assert vexact_block_params(d, h, A, perms, maps) == vexact_block_params(d, h, r["A_star"], r["perms"], r["heads"])
    # Deterministic in its seed, different for another seed.
    r2 = matched_random_span_structure(A, perms, maps, reg, seed=1)
    assert np.array_equal(r2["A_star"], r["A_star"]) and all(np.array_equal(a, b) for a, b in zip(r2["perms"], r["perms"]))
    assert not np.allclose(matched_random_span_structure(A, perms, maps, reg, seed=2)["A_star"], r["A_star"])
    assert np.array_equal(random_register_pairing_like(np.arange(4), 5), np.arange(4))          # no pairs: identity
    with pytest.raises(ValueError):
        random_register_pairing_like(np.array([1, 2, 0]), 0)


def test_random_control_is_exactly_equivariant_after_healing(tiny64):
    """The V-exact construction and recipe under a random structure: exact over the span with the random A*
    (and the flip-invariant features of full conversion), not with the fitted one."""
    L = tiny64.depth
    A, perms, maps = tiny_structure(tiny64)
    r = matched_random_span_structure(A, perms, maps, np.zeros(0, dtype=np.int64), seed=1)   # no registers
    student, cons, _ = build_vexact_student(tiny64, r["A_star"], L, r["perms"], r["heads"], r["reg_perm"])
    res = heal(VExactTeacher(tiny64, r["A_star"], L, r["reg_perm"]), student, heal_data(),
               cfg(range(L), train_scope="all", decay_1d=True), cons, None)
    assert res["steps_done"] == 4 and np.isfinite(res["heldout_final"]["total"])
    x = images(3, seed=5)
    ex = span_exactness(tiny64, student, [x], r["A_star"], L, r["reg_perm"])
    assert ex["pass"] and ex["max_rel_inside"] < 1e-10
    assert feature_invariance(tiny64, student, x)["rel_max"] < 1e-10
    checks = res["constraint_checks"]
    assert checks["0:MirrorMLPConstraint"]["max_rel_violation"] < 1e-12
    assert checks["1:MirrorAttnConstraint"]["max_rel_violation"] < 1e-12
    assert checks["2:VExactEmbeddings"]["max_share"] < 1e-24
    # Teeth: the random student is not equivariant under the fitted operator.
    assert max(row["rel_max"] for row in stream_equivariance(tiny64, student, x, A)) > 1e-3


def test_random_control_with_registers_is_exact():
    from test_m1_run import _timm_with_registers

    ad = _timm_with_registers(n_reg=4)
    A, perms, maps = tiny_structure(ad, seed=2)
    reg = np.array([1, 0, 2, 3])
    r = matched_random_span_structure(A, perms, maps, reg, seed=3)
    m, _, _ = build_vexact_student(ad, r["A_star"], ad.depth, r["perms"], r["heads"], r["reg_perm"])
    x = images(2, seed=6)
    assert span_exactness(ad, m, [x], r["A_star"], ad.depth, r["reg_perm"])["pass"]
    assert feature_invariance(ad, m, x)["rel_max"] < 1e-10


# ---------------------------------------------------------------- SVD arm: targets and trainable set
def test_vexact_teacher_block_targets_option(tiny64):
    A, _, _ = tiny_structure(tiny64)
    L = tiny64.depth
    plan = SimpleNamespace(capture=frozenset(range(L)), n_prefix=tiny64.n_prefix_tokens, first_grad=0)
    assert VExactTeacher(tiny64, A, L).describe()["block_targets"] == "centred, symmetrised with A*"   # unchanged
    assert VExactTeacher(tiny64, A, L, symmetrise=False).describe()["block_targets"] == "centred"
    with pytest.raises(ValueError):
        VExactTeacher(tiny64, A, L, symmetrise=False, block_targets="symmetrised")
    with pytest.raises(ValueError):
        VExactTeacher(tiny64, A, L, block_targets="odd")
    x = images(3, seed=4)
    t = VExactTeacher(tiny64, A, L, symmetrise=True, final_target="symmetrised", block_targets="raw")
    fx, start = t.distill_targets(x, plan)
    fm, _ = t.distill_targets(mirror(x), plan)
    plain, _ = heal_mod._teacher_forward(tiny64, x, plan)
    assert start is None
    for b in range(L):                                       # block targets: the teacher's own outputs
        ref = plain[f"block{b}"]
        assert ((fx[f"block{b}"] - ref).abs().max() / ref.abs().max()).item() < 1e-12
        assert ref.mean(-1).abs().max().item() > 1e-6        # not centred (V-exact's block targets are)
    # final targets: the flip average, so an image and its mirror have equal CLS and grid-flipped patch targets
    assert (fm["cls"] - fx["cls"]).abs().max().item() < 1e-12
    assert (fm["patch"] - unflip_tokens(fx["patch"], tiny64.grid_hw, 0)).abs().max().item() < 1e-12
    assert (fx["cls"] - plain["cls"]).abs().max().item() > 1e-6                     # teeth: not the raw target


def test_span_trainable_matches_the_vexact_trainable_set(tiny64):
    """The SVD student with SpanTrainable and train_scope "all" trains the converted blocks and exactly the
    embedding tensors a V-exact student with train_embeddings trains; the final norm stays frozen."""
    hm = load_script("heal")
    L = tiny64.depth
    A, perms, maps = tiny_structure(tiny64)
    vx, vcons, _ = build_vexact_student(tiny64, A, L, perms, maps)
    for c in vcons:
        getattr(c, "prepare", lambda m: None)(vx)
    v_names = [n for n, _ in configure_trainable(vx, range(L), "all", vcons)]
    c = load_config(CONFIGS / "e3_svd_raw.yaml", tiny=True)
    structure = {"mode": "span", "A_layers": [A] * (L + 1), "A_star": A, "perms": perms, "heads": maps,
                 "a_out_offset": 1, "source": "test", "heads_source": "test"}
    make, info = hm.method_setup("svd", tiny64, structure, list(range(L)), c)
    s, scons = make()
    for cc in scons:
        getattr(cc, "prepare", lambda m: None)(s)
    s_names = [n for n, _ in configure_trainable(s, range(L), "all", scons)]
    outside = lambda names: sorted(n for n in names if not n.startswith("blocks."))  # noqa: E731
    assert outside(s_names) == outside(v_names) and outside(s_names)
    assert not any(n.startswith("norm.") for n in s_names) and isinstance(scons[-1], SpanTrainable)
    assert {n.split(".", 2)[1] for n in s_names if n.startswith("blocks.")} == {str(b) for b in range(L)}
    # Without train_scope "all" the option is refused (it would silently train nothing outside the blocks).
    c2 = copy.deepcopy(c)
    c2["heal"]["train_scope"] = "converted"
    with pytest.raises(ValueError, match="train_scope"):
        hm.method_setup("svd", tiny64, structure, list(range(L)), c2)
    # Embeddings of V-exact: same tensors as the VExactEmbeddings constraint keeps trainable.
    frozen = {id(p) for p in VExactEmbeddings(tiny64, A, range(L)).frozen_parameters(vx)}
    assert set(outside(v_names)) == {n for n, p in vx.named_parameters()
                                     if id(p) not in frozen and not n.startswith("blocks.")}


def test_both_svd_target_variants_run(tmp_path):
    hm = load_script("heal")
    runs = {}
    for name in ("raw", "sym"):
        s = hm.main(["--config", str(CONFIGS / f"e3_svd_{name}.yaml"), "--tiny", "--out-dir", str(tmp_path / name)])
        out = tmp_path / name / "svd"
        for f in ("summary.json", "knn.json", "final/metrics.json", "final/trained_blocks.pt",
                  "lr_selection/lr_selection.json", "lr_extra.json"):
            assert (out / f).is_file(), f
        assert s["method"] == "svd" and s["blocks"] == [0, 1, 2, 3] and s["seed"] == 0 and len(s["git_hash"]) >= 7
        assert s["final"]["steps_done"] == s["final"]["total_steps"] and s["final"]["config"]["train_scope"] == "all"
        runs[name] = s
    raw, sym = runs["raw"], runs["sym"]
    assert raw["targets"]["type"] == "raw"
    assert sym["targets"]["type"] == "VExactTeacher" and sym["targets"]["final_target"] == "symmetrised"
    assert sym["targets"]["block_targets"].startswith("raw")
    # Same student, ranks, trainable set and block normalisers; the final normalisers are measured on the
    # flip-averaged targets.
    assert raw["method_info"]["ranks"] == sym["method_info"]["ranks"]
    assert raw["final"]["trainable_names"] == sym["final"]["trainable_names"]
    nr, ns = raw["final"]["normalisers"], sym["final"]["normalisers"]
    for b in range(4):                     # float32 model; the 3b teacher runs the batch with its mirror appended
        assert ns[f"block{b}"] == pytest.approx(nr[f"block{b}"], rel=1e-6)
    assert abs(ns["patch"] - nr["patch"]) > 1e-6 * nr["patch"] and ns["cls"] != nr["cls"]
    names = raw["final"]["trainable_names"]
    assert {"patch_embed.proj.weight", "pos_embed", "cls_token"} <= set(names)
    assert not any(n.startswith("norm.") for n in names) and any(n.endswith("mlp.fc1.U") for n in names)
    # Ranks matched to the tiny V-exact run of the same structure (scripts/heal_vexact.py, k = 4).
    hv = load_script("heal_vexact")
    vcfg = load_config(CONFIGS / "m3_vexact.yaml", tiny=True)
    teacher = hm.build_teacher(vcfg, torch.device("cpu"))
    st = hv.load_vexact_structure(vcfg, teacher, 4)
    pc = vexact_param_counts(teacher, st["A_star"], 4, st["perms"], st["heads"])
    vm = raw["method_info"]["vexact_match"]
    assert (vm["mlp_kept_vexact"], vm["attn_kept_vexact"]) == (pc["mlp_params_kept"], pc["attn_params_kept"])
    assert vm["mlp_kept_svd"] <= vm["mlp_kept_vexact"] and vm["attn_kept_svd"] <= vm["attn_kept_vexact"]
    chk = raw["final"]["constraint_checks"]
    assert chk["1:LowRankAttention"]["numerical_ranks"] == chk["1:LowRankAttention"]["target_ranks"]
    # flip_average is V-exact's full-conversion target: refused for a partial span.
    c = load_config(CONFIGS / "e3_svd_sym.yaml", tiny=True)
    c["blocks"] = [0, 1]
    with pytest.raises(ValueError, match="k = depth"):
        hm.distill_teacher(c, teacher, hm.load_structure(c, teacher), [0, 1])


# ---------------------------------------------------------------- healing seed
def test_seed_changes_data_order_crops_and_flips(tiny64, monkeypatch):
    """HealConfig.seed (the config `seed`) sets the data order, the crops and the flips; the same seed
    repeats them exactly."""
    A, perms, maps = tiny_structure(tiny64)
    data = heal_data(n_train=24)
    seen = {}

    def run(seed):
        rows, crops = [], []
        read, sample = heal_mod._read_rows, heal_mod.sample_crop_params

        def rec_read(images, r, pin):
            if len(r) == 8:                                    # training batches (held-out and calibration: 4)
                rows.append(np.asarray(r).copy())
            return read(images, r, pin)

        def rec_sample(*a, **kw):
            out = sample(*a, **kw)
            crops.append(out.clone())
            return out

        monkeypatch.setattr(heal_mod, "_read_rows", rec_read)
        monkeypatch.setattr(heal_mod, "sample_crop_params", rec_sample)
        s, cons, _ = build_vexact_student(tiny64, A, 2, perms, maps)
        heal(VExactTeacher(tiny64, A, 2), s, data, cfg([0, 1], seed=seed, max_steps=3, train_scope="all"), cons, None)
        monkeypatch.undo()
        return rows, crops

    for seed in (0, 1, 0):
        seen.setdefault(seed, []).append(run(seed))
    (r0, c0), (r0b, c0b) = seen[0]
    r1, c1 = seen[1][0]
    assert len(r0) == len(r1) == 3 and len(c0) == len(c1) == 3
    assert all(np.array_equal(a, b) for a, b in zip(r0, r0b)) and all(torch.equal(a, b) for a, b in zip(c0, c0b))
    assert any(not np.array_equal(a, b) for a, b in zip(r0, r1))                       # data order
    assert not torch.equal(torch.stack(c0)[..., :4], torch.stack(c1)[..., :4])         # crop boxes
    assert not torch.equal(torch.stack(c0)[..., 4], torch.stack(c1)[..., 4])           # flips


def _write_structure_files(adapter, root: Path, model_name: str):
    """Synthetic structure files in the real layout (e1_operators.npz, m1_structure.npz) for the tiny model."""
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    A, perms, maps = tiny_structure(adapter, seed=5)
    ops = root / "e1_operators.npz"
    np.savez(ops, A_layers=np.stack([random_involution_fixing_ones(d, d // 2, seed=40 + l) for l in range(L + 1)]),
             A_shared=A, perms=np.stack(perms))
    m1 = root / "m1_structure.npz"
    np.savez(m1, sigma=np.stack([hm.sigma for hm in maps]), B=np.stack([hm.B for hm in maps]),
             C=np.stack([hm.C for hm in maps]), a_out_offset=np.array(1), ks=np.array([2, L]),
             A_star=np.stack([A, A]), reg_perm=np.zeros((2, 0), dtype=np.int64), model=np.array(model_name))
    return ops, m1, A, perms


def test_outside_tiny_the_seed_leaves_the_structure_alone(tmp_path):
    """Outside --tiny the structure comes from the files: seed 0 and seed 1 (configs/e3_vexact_s1.yaml) build
    the same A*, pairings and head maps, and the random control's structure depends on structure.random_seed
    only."""
    from mirror.adapters import TimmTinyAdapter

    hv = load_script("heal_vexact")
    ad = TimmTinyAdapter(img_size=56, seed=0)
    out = {}
    for name in ("m3_vexact", "e3_vexact_s1", "e3_random"):
        c = load_config(CONFIGS / f"{name}.yaml")
        ops, m1, A, perms = _write_structure_files(ad, tmp_path, c["model"]["name"])
        c["structure"].update(operators=str(ops), heads=str(m1))
        out[name] = (c["seed"], hv.load_vexact_structure(c, ad, ad.depth))
        if name == "e3_random":
            c["seed"] = 7                                     # another healing seed: the same random structure
            out["e3_random_seed7"] = (7, hv.load_vexact_structure(c, ad, ad.depth))
    (s0, a), (s1, b) = out["m3_vexact"], out["e3_vexact_s1"]
    assert (s0, s1) == (0, 1)
    assert np.array_equal(a["A_star"], b["A_star"]) and np.allclose(a["A_star"], A, atol=1e-12)
    assert all(np.array_equal(p, q) for p, q in zip(a["perms"], b["perms"]))
    assert all(np.array_equal(p.B, q.B) and np.array_equal(p.sigma, q.sigma) for p, q in zip(a["heads"], b["heads"]))
    r, r7 = out["e3_random"][1], out["e3_random_seed7"][1]
    assert np.array_equal(r["A_star"], r7["A_star"]) and not np.allclose(r["A_star"], a["A_star"])
    assert r["random_structure"]["seed"] == 1 and r["random_structure"]["d_plus"] == d_plus(A)
    assert "random_structure" not in a
    # The E3 V-exact configs differ from configs/m3_vexact.yaml only where intended.
    base = load_config(CONFIGS / "m3_vexact.yaml")
    for name, diff in (("e3_vexact_s1", {"seed", "out_dir", "config_path", "lr_selection"}),
                       ("e3_random", {"structure", "out_dir", "config_path", "lr_selection"})):
        c = load_config(CONFIGS / f"{name}.yaml")
        assert {k for k in set(c) | set(base) if c.get(k) != base.get(k)} == diff, name
        assert {k: v for k, v in c["lr_selection"].items() if k != "extra_lrs"} == base["lr_selection"]


def test_budget_script_reports_the_matched_counts(tmp_path):
    """scripts/e3_budget.py (the table in PREREG_E3.md) on structure files in the real layout."""
    from mirror.adapters import TimmTinyAdapter

    ad = TimmTinyAdapter(img_size=56, seed=0)
    ops, m1, A, perms = _write_structure_files(ad, tmp_path, "tiny")
    res = load_script("e3_budget").main(["--heads", str(m1), "--operators", str(ops), "--k", "4",
                                         "--json", str(tmp_path / "b.json")])
    st = load_vexact_structure_from_files(ad, ops, m1)
    pc = vexact_param_counts(ad, A, 4, perms, st["heads"])
    assert res["vexact_kept"] == pc["converted_params_kept"] and res["dense_matrix_params"] == pc["converted_params_dense"]
    assert res["d_plus"] == d_plus(A) and res["random_counts_equal"]
    assert [r["mlp_ranks"] for r in res["per_block"]] == [r["mlp_ranks"] for r in matched_svd_ranks(
        vexact_block_params(ad.embed_dim, ad.mlp_hidden, A, perms, st["heads"]), ad.mlp_hidden, ad.embed_dim, 64)]
    assert json.loads((tmp_path / "b.json").read_text())["svd_kept"] == res["svd_kept"] <= res["vexact_kept"]


def load_vexact_structure_from_files(ad, ops, m1):
    hv = load_script("heal_vexact")
    c = load_config(CONFIGS / "m3_vexact.yaml")
    c["model"]["name"] = "tiny"
    c["structure"].update(operators=str(ops), heads=str(m1))
    return hv.load_vexact_structure(c, ad, ad.depth)


# ---------------------------------------------------------------- secondary learning-rate check
def test_extra_lrs_run_short_trials_and_never_change_the_chosen_rate(tiny64, tmp_path):
    A, perms, maps = tiny_structure(tiny64)
    data = heal_data()
    teacher = VExactTeacher(tiny64, A, 2)

    def make():
        s, cons, _ = build_vexact_student(tiny64, A, 2, perms, maps)
        return s, cons

    c = cfg([0, 1], train_scope="all")
    grid, extra = [1e-5, 1e-4], [1e-3, 50.0]
    sel = select_lr(teacher, make, data, c, lrs=grid, budget_steps=3)
    res = probe_extra_lrs(teacher, make, data, c, sel, extra, out_dir=tmp_path)
    # The extra trials are exactly the trials select_lr would have run with those rates in its grid.
    both = select_lr(teacher, make, data, c, lrs=grid + extra[:1], budget_steps=3)
    assert res["extra"][0]["heldout_loss"] == pytest.approx(both["trials"][2]["heldout_loss"], rel=1e-12)
    assert [t["heldout_loss"] for t in res["grid"]] == pytest.approx([t["heldout_loss"] for t in both["trials"][:2]],
                                                                      rel=1e-12)
    assert res["chosen_lr"] == sel["best_lr"] and res["budget_steps"] == 3 and res["seed"] == 0
    assert res["extra_would_win"] == (both["best_lr"] == 1e-3) and res["best_lr_with_extra"] == both["best_lr"]
    assert res["extra"][1]["diverged"] or res["extra"][1]["heldout_loss"] > min(t["heldout_loss"] for t in res["grid"])
    saved = json.loads((tmp_path / "lr_extra.json").read_text())
    assert saved["chosen_lr"] == sel["best_lr"] and saved["extra_lrs"] == extra and saved["git_hash"]
    assert (tmp_path / "lr_extra" / "lr_1e-03" / "train_log.csv").is_file()
    with pytest.raises(ValueError):
        probe_extra_lrs(teacher, make, data, c, {"skipped": True, "best_lr": 1e-4}, extra)


def test_cli_extra_lrs_and_probe_only(tmp_path):
    hv = load_script("heal_vexact")
    s = hv.main(["--config", str(CONFIGS / "e3_vexact_s1.yaml"), "--tiny", "--out-dir", str(tmp_path / "s1")])
    lx = json.loads((tmp_path / "s1" / "k4" / "lr_extra.json").read_text())
    assert s["seed"] == 1 and s["final"]["config"]["seed"] == 1 and s["lr_selection"]["seed"] == 1
    assert s["lr_selection"]["best_lr"] == lx["chosen_lr"] == s["lr_extra"]["chosen_lr"]
    assert s["final"]["config"]["lr"] == s["lr_selection"]["best_lr"]                  # the full run used the grid's choice
    assert lx["extra_lrs"] == [3e-3] and "Secondary learning-rate check" in (tmp_path / "s1" / "k4" / "SUMMARY.md").read_text()
    # Probe only, for a finished run: grid and extra rates, no checkpoint, k-NN or summary; compared with the
    # recorded grid of the reference run.
    ref = hv.main(["--config", str(CONFIGS / "m3_vexact.yaml"), "--tiny", "--out-dir", str(tmp_path / "s0")])
    p = hv.main(["--config", str(CONFIGS / "m3_vexact.yaml"), "--tiny", "--extra-lrs", "3e-3", "--lr-probe-only",
                 "--reference-run", str(tmp_path / "s0" / "k4"), "--out-dir", str(tmp_path / "probe")])
    pd_ = tmp_path / "probe" / "k4"
    assert (pd_ / "lr_extra.json").is_file() and (pd_ / "lr_selection" / "lr_selection.json").is_file()
    assert not (pd_ / "summary.json").exists() and not (pd_ / "final").exists() and not (pd_ / "knn.json").exists()
    assert p["probe_only"] and p["chosen_lr"] == ref["lr_selection"]["best_lr"]
    assert p["reference"]["same_best_lr"] and p["reference"]["max_rel_diff_heldout_loss"] == 0.0
    with pytest.raises(FileExistsError):                      # never inside a finished run
        hv.main(["--config", str(CONFIGS / "m3_vexact.yaml"), "--tiny", "--extra-lrs", "3e-3", "--lr-probe-only",
                 "--out-dir", str(tmp_path / "s0")])
    with pytest.raises(ValueError, match="another grid, budget or seed"):
        hv.main(["--config", str(CONFIGS / "m3_vexact.yaml"), "--tiny", "--extra-lrs", "3e-3", "--lr-probe-only",
                 "--reference-run", str(tmp_path / "s1" / "k4"), "--out-dir", str(tmp_path / "probe2")])
    with pytest.raises(ValueError, match="repeat a grid rate"):
        hv.main(["--config", str(CONFIGS / "m3_vexact.yaml"), "--tiny", "--extra-lrs", "1e-3", "--lr-probe-only",
                 "--out-dir", str(tmp_path / "probe3")])


# ---------------------------------------------------------------- Tier 2 loads the E3 checkpoints
def test_tier2_rebuilds_the_e3_students(tmp_path):
    """scripts/tier2_linear.py (the jobs file's Tier-2 step) rebuilds the healed random-structure V-exact
    student (--vexact-dir) and the healed SVD student (--heal-dir) exactly as they were healed."""
    hv, hm = load_script("heal_vexact"), load_script("heal")
    captured = {}
    knn_v, knn_h = hv.heal_cli.evaluate_final_knn, hm.evaluate_final_knn

    def cap(tag, fn):
        def inner(cfg_, teacher, student, make_student, device, out, git):
            captured[tag] = copy.deepcopy(student)
            return fn(cfg_, teacher, student, make_student, device, out, git)
        return inner

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hv.heal_cli, "evaluate_final_knn", cap("random", knn_v))
        mp.setattr(hm, "evaluate_final_knn", cap("svd", knn_h))
        hv.main(["--config", str(CONFIGS / "e3_random.yaml"), "--tiny", "--lr", "1e-3", "--max-steps", "3",
                 "--out-dir", str(tmp_path / "random")])
        hm.main(["--config", str(CONFIGS / "e3_svd_sym.yaml"), "--tiny", "--lr", "1e-3", "--max-steps", "3",
                 "--out-dir", str(tmp_path / "svd_sym")])
    T2 = load_script("tier2_linear")
    ad = T2.build_adapter("tiny", True, "cpu")
    x = images(4, seed=9).float()
    fac = T2.VariantFactory(ad, "tiny", True, vexact_dir=tmp_path / "random", heal_dir=tmp_path / "svd_sym")
    for name, tag in (("healed_vexact_k4", "random"), ("healed_svd_k4", "svd")):
        m, info = fac.build(T2.parse_variant(name))
        with torch.no_grad():
            got, ref = T2.probe_feature(ad, x, m), T2.probe_feature(ad, x, captured[tag])
        assert (got - ref).abs().max().item() == 0.0, name
        if tag == "random":
            assert info["vexact_checks"]["pass"] and info["vexact_checks"]["full_conversion"]
            summ = json.loads((tmp_path / "random" / "k4" / "summary.json").read_text())
            assert summ["method"] == "vexact_random" and summ["random_structure"]["seed"] == 1
            assert info["converted_params_kept"] == summ["method_info"]["converted_params_kept"]
        else:
            assert info["method_info"]["vexact_match"]["span_k"] == 4
            assert info["checkpoint_meta"]["steps"] == 3
    # The CLI step of the jobs file runs on both (linear probe).
    res = T2.main(["--tiny", "--variants", "healed_vexact_k4,healed_svd_k4", "--vexact-dir", str(tmp_path / "random"),
                   "--heal-dir", str(tmp_path / "svd_sym"), "--out", str(tmp_path / "t2"), *FAST])
    rows = {r["variant"]: r for r in res["rows"]}
    assert rows["healed_vexact_k4"]["vexact_checks_pass"] and 0 <= rows["healed_svd_k4"]["top1"] <= 100
