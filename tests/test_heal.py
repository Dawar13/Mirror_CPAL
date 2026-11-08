"""Healing trainer (M3): CPU tests with the tiny random ViT and synthetic uint8 images."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mirror.heal import (HealConfig, HealData, LowRankLinear, LowRankMLP, MirrorMLPConstraint, augment_batch,
                         centre_view, crop_resize_flip, heal, load_trained_blocks, sample_crop_params, select_lr)
from mirror.matching import random_pairing
from mirror.surgery import apply_surgery_two, apply_svd_baseline, fold_mlp

ROOT = Path(__file__).resolve().parents[1]
S_IN, CROP = 64, 56   # tiny model takes 56 x 56; crops come from 64 x 64 arrays


def involution_fixing_ones(d, d_plus, seed):
    """Random involution with A 1 = 1 (needed for LayerNorm equivariance); as in test_surgery.py."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(d), rng.standard_normal((d, d - 1))]))
    signs = np.concatenate([[1.0], np.ones(d_plus - 1), -np.ones(d - d_plus)])
    A = (Q * signs) @ Q.T
    return 0.5 * (A + A.T)


def make_data(n_train=16, n_held=8, seed=0):
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(n_train + n_held, S_IN, S_IN, 3), dtype=np.uint8)
    rows = rng.permutation(n_train + n_held)
    return HealData(images, train_rows=rows[:n_train], heldout_rows=rows[n_train:])


def base_cfg(**kw):
    c = dict(blocks=[1, 2], seed=0, lr=1e-3, max_steps=4, batch_size=8, micro_batch=4, crop_size=CROP,
             warmup_steps=1, calib_images=8, log_every=0, prefetch=0)
    c.update(kw)
    return HealConfig(**c)


def mirror_student(adapter, blocks, seed=0):
    d, h = adapter.embed_dim, adapter.mlp_hidden
    A = {l: involution_fixing_ones(d, 24 + 4 * l, seed=100 + 10 * seed + l) for l in range(adapter.depth + 1)}
    A_in, A_out = [A[b] for b in blocks], [A[b + 1] for b in blocks]
    perms = [random_pairing(h, 90, seed=200 + 10 * seed + b) for b in blocks]
    student = copy.deepcopy(adapter.model)
    apply_surgery_two(student, blocks, A_in, A_out, perms)
    return student, MirrorMLPConstraint(blocks, A_in, A_out, perms), (A_in, A_out, perms)


def mlp_branch(blk):
    return lambda t: blk.ls2(blk.mlp(blk.norm2(t)))


# ---------------------------------------------------------------- augmentation
def test_augmentation_identity_flip_and_bounds():
    g = torch.Generator().manual_seed(0)
    x = torch.from_numpy(np.random.default_rng(1).integers(0, 256, (3, S_IN, S_IN, 3), dtype=np.uint8))
    xf = x.permute(0, 3, 1, 2).to(torch.float64)
    full = sample_crop_params(3, S_IN, S_IN, (1.0, 1.0), (1.0, 1.0), 0.0, g)
    assert torch.equal(full[:, :4], torch.tensor([[0.0, 0.0, S_IN, S_IN]] * 3, dtype=torch.float64))
    assert (crop_resize_flip(xf, full, S_IN) - xf).abs().max().item() < 1e-6
    full[:, 4] = 1.0
    assert (crop_resize_flip(xf, full, S_IN) - torch.flip(xf, dims=[-1])).abs().max().item() < 1e-6
    # random boxes stay inside the image and are reproducible from the seed
    p = sample_crop_params(500, S_IN, S_IN, (0.35, 1.0), (3 / 4, 4 / 3), 0.5, torch.Generator().manual_seed(3))
    q = sample_crop_params(500, S_IN, S_IN, (0.35, 1.0), (3 / 4, 4 / 3), 0.5, torch.Generator().manual_seed(3))
    assert torch.equal(p, q)
    top, left, h, w, flip = p.unbind(1)
    assert bool(((top >= 0) & (left >= 0) & (h >= 1) & (w >= 1) & (top + h <= S_IN) & (left + w <= S_IN)).all())
    assert 0.3 < flip.mean().item() < 0.7
    out = augment_batch(x, p[:3], CROP)
    assert out.shape == (3, 3, CROP, CROP) and torch.isfinite(out).all()
    assert centre_view(x, CROP).shape == (3, 3, CROP, CROP)


# ---------------------------------------------------------------- (a) loss decreases
def test_loss_decreases_on_fixed_batch(tiny64, tmp_path):
    data = make_data(n_train=8, n_held=8)
    student, con, _ = mirror_student(tiny64, [1, 2])
    # one batch of all 8 training images, full-image crop, no flip: the same batch every step
    cfg = base_cfg(max_steps=8, crop_scale=(1.0, 1.0), crop_ratio=(1.0, 1.0), hflip_p=0.0)
    res = heal(tiny64, student, data, cfg, [con], tmp_path)
    losses = res["history"]
    assert len(losses) == 8 and all(np.isfinite(losses))
    assert losses[-1] < 0.8 * losses[0], losses
    assert res["heldout_final"]["total"] < res["heldout_initial"]["total"]
    with open(tmp_path / "train_log.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8 and {"loss", "cls", "patch", "block1", "block2", "lr"} <= set(rows[0])
    meta = json.loads((tmp_path / "metrics.json").read_text())
    assert meta["seed"] == 0 and "git_hash" in meta and meta["config"]["blocks"] == [1, 2]
    assert (tmp_path / "trained_blocks.pt").exists()


def test_accumulation_prefix_sharing_and_checkpointing_do_not_change_training(tiny64):
    data = make_data()
    states = []
    for kw in (dict(micro_batch=8), dict(micro_batch=2, share_prefix=False), dict(micro_batch=4, grad_checkpoint=True)):
        student, con, _ = mirror_student(tiny64, [1, 2])
        res = heal(tiny64, student, data, base_cfg(max_steps=3, **kw), [con], None)
        states.append(student.state_dict())
        assert res["share_prefix_used"] == kw.get("share_prefix", True)
    for st in states[1:]:
        for k in st:
            assert (st[k] - states[0][k]).abs().max().item() < 1e-9, k


# ---------------------------------------------------------------- (b) exact equivariance
def test_mirror_constraint_keeps_mlp_exactly_equivariant(tiny64):
    blocks = [1, 2]
    data = make_data()
    student, con, (A_in, A_out, _) = mirror_student(tiny64, blocks)
    w0 = {b: student.blocks[b].mlp.fc1.weight.clone() for b in blocks}
    heal(tiny64, student, data, base_cfg(max_steps=4), [con], None)
    x = torch.randn(3, 17, tiny64.embed_dim, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    for i, b in enumerate(blocks):
        blk = student.blocks[b]
        f = mlp_branch(blk)
        Ai, Ao = torch.tensor(A_in[i]), torch.tensor(A_out[i])
        assert (f(x @ Ai) - f(x) @ Ao).abs().max().item() < 1e-10
        assert (blk.mlp.fc1.weight - w0[b]).abs().max().item() > 1e-5     # training did move the weights
        assert torch.equal(blk.norm2.weight, torch.ones_like(blk.norm2.weight))
        assert torch.equal(blk.ls2.gamma, torch.ones_like(blk.ls2.gamma))
    assert con.check(student)["max_rel_violation"] < 1e-12
    # Teeth: the same training without the constraint breaks the equivariance.
    free, _, _ = mirror_student(tiny64, blocks)
    heal(tiny64, free, data, base_cfg(max_steps=4), [], None)
    f = mlp_branch(free.blocks[1])
    Ai, Ao = torch.tensor(A_in[0]), torch.tensor(A_out[0])
    assert (f(x @ Ai) - f(x) @ Ao).abs().max().item() > 1e-6


# ---------------------------------------------------------------- (c) exact rank
def test_low_rank_mlp_keeps_exact_rank(tiny64, images64):
    blocks, ranks = [0, 3], [(10, 12), (7, 9)]
    data = make_data()
    student = copy.deepcopy(tiny64.model)
    apply_svd_baseline(student, blocks, ranks)
    dense = copy.deepcopy(student)
    con = LowRankMLP(blocks, ranks)
    con.prepare(student)                                     # heal calls it again: must be idempotent
    assert (tiny64.cls_feature(images64, dense) - tiny64.cls_feature(images64, student)).abs().max().item() < 1e-10
    u0 = {b: student.blocks[b].mlp.fc1.U.clone() for b in blocks}
    res = heal(tiny64, student, data, base_cfg(blocks=blocks, max_steps=4), [con], None)
    h, d = tiny64.mlp_hidden, tiny64.embed_dim
    for b, (r1, r2) in zip(blocks, ranks):
        blk = student.blocks[b]
        fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
        assert isinstance(fc1, LowRankLinear) and isinstance(fc2, LowRankLinear)
        assert fc1.U.shape == (h, r1) and fc1.V.shape == (r1, d) and fc2.U.shape == (d, r2)
        f = fold_mlp(blk)
        assert torch.linalg.matrix_rank(f["W1"]).item() == r1
        assert torch.linalg.matrix_rank(f["W2"]).item() == r2
        assert (fc1.U - u0[b]).abs().max().item() > 1e-5
    chk = res["constraint_checks"]["0:LowRankMLP"]
    assert chk["numerical_ranks"] == [list(r) for r in ranks]


# ---------------------------------------------------------------- (d) LR selection
def test_lr_selection_picks_from_list_and_is_deterministic(tiny64, tmp_path):
    data = make_data()

    def make():
        s, c, _ = mirror_student(tiny64, [1, 2])
        return s, [c]

    lrs = [1e-5, 1e-3, 3e-2]
    r1 = select_lr(tiny64, make, data, base_cfg(), lrs=lrs, budget_steps=3, out_dir=tmp_path / "a")
    r2 = select_lr(tiny64, make, data, base_cfg(), lrs=lrs, budget_steps=3, out_dir=None)
    assert r1["best_lr"] in lrs and r1["best_lr"] == r2["best_lr"]
    l1 = [t["heldout_loss"] for t in r1["trials"]]
    l2 = [t["heldout_loss"] for t in r2["trials"]]
    assert np.allclose(l1, l2, rtol=1e-12, atol=0)
    assert r1["best_lr"] == lrs[int(np.argmin(l1))]
    assert len(set(l1)) == len(l1)                           # the rates really give different losses
    saved = json.loads((tmp_path / "a" / "lr_selection.json").read_text())
    assert saved["best_lr"] == r1["best_lr"] and len(saved["trials"]) == 3


# ---------------------------------------------------------------- (e) frozen parameters
def test_frozen_parameters_do_not_change(tiny64, images64, tmp_path):
    blocks = [1, 2]
    data = make_data()
    student, con, _ = mirror_student(tiny64, blocks)
    before = {k: v.clone() for k, v in student.state_dict().items()}
    teacher_before = {k: v.clone() for k, v in tiny64.model.state_dict().items()}
    heal(tiny64, student, data, base_cfg(max_steps=3, decay_1d=True), [con], tmp_path)
    after = student.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    allowed = {k for k in before if k.startswith(("blocks.1.", "blocks.2."))
               and not any(s in k for s in ("norm2.", "ls2."))}
    assert changed and changed <= allowed, sorted(changed - allowed)
    assert any(k.endswith("attn.qkv.weight") for k in changed) and any(k.endswith("mlp.fc1.weight") for k in changed)
    assert all(torch.equal(v, tiny64.model.state_dict()[k]) for k, v in teacher_before.items())
    # The checkpoint holds exactly the converted blocks and restores the healed student.
    payload = torch.load(tmp_path / "trained_blocks.pt", weights_only=True)
    keys = set(payload["state_dict"])
    want = {f"blocks.{b}.{k}" for b in blocks for k in student.blocks[b].state_dict()}
    assert keys == want
    fresh, _, _ = mirror_student(tiny64, blocks)
    load_trained_blocks(fresh, tmp_path / "trained_blocks.pt")
    assert (tiny64.cls_feature(images64, fresh) - tiny64.cls_feature(images64, student)).abs().max().item() == 0.0


# ---------------------------------------------------------------- CLI smoke test
def test_cli_tiny_runs_all_three_methods(tmp_path):
    spec = importlib.util.spec_from_file_location("heal_cli", ROOT / "scripts" / "heal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for method in ("mirror", "svd", "random"):
        summary = mod.main(["--config", str(ROOT / "configs" / "m3_pilot.yaml"), "--tiny", "--method", method,
                            "--out-dir", str(tmp_path)])
        assert summary["lr_selection"]["best_lr"] in summary["lr_selection"]["lrs"]
        assert summary["final"]["steps_done"] == summary["final"]["total_steps"]
        assert (tmp_path / method / "summary.json").exists()
        assert (tmp_path / method / "final" / "trained_blocks.pt").exists()
