"""Lockstep multi-student healing (mirror/final/heal/multistudent.py, targets.py, data_gpu.py): CPU tests with the
tiny random timm ViT in float64 and synthetic uint8 images.

(a) one student trained in lockstep equals mirror.heal.heal() bit for bit (state, loss history, held-out losses,
logs); (b) N students in lockstep each equal their own single run; (c) the targets equal
VExactTeacher.distill_targets for every mode; (e) the rot90, logits and block_map options. Resume, hooks, divergence
and the learning-rate selection are in tests/test_fp_heal_lockstep.py."""
import copy
import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from mirror.heal import HealConfig, _make_plan, augment_batch, heal, sample_crop_params
from mirror.heal_vexact import VExactTeacher
from mirror.m1 import random_involution_fixing_ones

from mirror.final.heal.multistudent import DeviceAugment, LockstepStudent, train_lockstep
from mirror.final.heal.targets import SharedTeacher, TargetMode, rotate_back_tokens, student_features

from test_fp_heal_common import Student, TinyRegAdapter, cfg, heal_data, same_state, tiny_adapter


@pytest.fixture(scope="module")
def ad():
    return tiny_adapter()


@pytest.fixture(scope="module")
def kinds(ad):
    return {k: Student(k, ad) for k in ("mirror", "lowrank", "slice")}


# ---------------------------------------------------------------- augmentation on the device
def test_device_augmentation_equals_augment_batch():
    g = torch.Generator().manual_seed(0)
    x = torch.from_numpy(np.random.default_rng(1).integers(0, 256, (6, 64, 64, 3), dtype=np.uint8))
    p = sample_crop_params(6, 64, 64, (0.35, 1.0), (3 / 4, 4 / 3), 0.5, g)
    for dt in (torch.float32, torch.float64):
        assert torch.equal(DeviceAugment("cpu", dt, 56)(x, p), augment_batch(x, p, 56, dt))


# ---------------------------------------------------------------- (a) one student equals heal() bit for bit
@pytest.mark.parametrize("kind", ["mirror", "lowrank"])
def test_single_student_equals_heal(ad, kinds, kind, tmp_path):
    st = kinds[kind]
    data = heal_data()
    c = cfg(range(ad.depth), max_steps=3, eval_every=2, log_every=2)
    ref_model, ref = st.single(c, data, tmp_path / "ref")
    s = st.lockstep("s", c.lr, run_dir=tmp_path / "lock")
    res = train_lockstep(ad, [s], data, c, job_dir=tmp_path / "job")
    got = res["students"]["s"]
    assert same_state(ref_model, s.model)                                     # bit for bit
    assert got["history"] == ref["history"] and len(got["history"]) == 3
    assert not same_state(st.fresh()[0], s.model) and len(set(got["history"])) == 3   # training moved the weights
    assert got["heldout_initial"] == ref["heldout_initial"] and got["heldout_final"] == ref["heldout_final"]
    assert got["normalisers"] == ref["normalisers"] and got["n_trainable"] == ref["n_trainable"]
    assert got["trainable_names"] == ref["trainable_names"] and got["steps_done"] == 3
    assert got["constraint_checks"].keys() == ref["constraint_checks"].keys()
    for name in ("train_log.csv", "heldout_log.csv"):                          # the same rows (seconds aside)
        with open(tmp_path / "ref" / name, newline="") as fh:
            a = [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(fh)]
        with open(tmp_path / "lock" / name, newline="") as fh:
            b = [{k: v for k, v in r.items() if k != "seconds"} for r in csv.DictReader(fh)]
        assert a == b and len(a) >= 3, name
    held = json.loads((tmp_path / "lock" / "heldout.json").read_text())
    assert held["final"]["total"] == ref["heldout_final"]["total"] and [r["step"] for r in held["by_step"]] == [0, 2, 3]


# ---------------------------------------------------------------- (b) N students equal their single runs
def test_lockstep_students_equal_their_single_runs(ad, kinds, tmp_path):
    """Three target modes (symmetrised, raw and centred block targets, all with flip-averaged final targets) in one
    job: each student equals its own heal() run bit for bit. A fourth student without flip targets takes the first
    half of the flipped teacher pass; against its own run (teacher on x only) it agrees to rounding."""
    data = heal_data()
    c = cfg(range(ad.depth), max_steps=2)
    L, d = ad.depth, ad.embed_dim
    lrs = {"mirror": 1e-2, "lowrank": 5e-3, "slice": 2e-2, "noflip": 1e-2}
    studs = [kinds[k].lockstep(k, lrs[k]) for k in ("mirror", "lowrank")]
    sl = kinds["slice"].lockstep("slice", lrs["slice"])
    sl.mode = TargetMode("symmetrised", "centred")
    nf = kinds["lowrank"].lockstep("noflip", lrs["noflip"])
    nf.mode = TargetMode("raw", "centred")
    studs += [sl, nf]
    res = train_lockstep(ad, studs, data, c, job_dir=tmp_path / "job")
    summary = json.loads((tmp_path / "job" / "summary.json").read_text())
    assert summary["students"].keys() == lrs.keys()
    assert [res["students"][k]["target_mode"]["blocks"] for k in lrs] == ["symmetrised", "raw", "centred", "centred"]
    for s in studs[:2]:
        ref_model, ref = kinds[s.name].single(replace(c, lr=lrs[s.name]), data)
        assert same_state(ref_model, s.model), s.name
        assert res["students"][s.name]["history"] == ref["history"], s.name
        assert res["students"][s.name]["heldout_final"] == ref["heldout_final"], s.name
    teachers = {"slice": VExactTeacher(ad, np.eye(d), L, None, True, block_targets="centred"),
                "noflip": VExactTeacher(ad, np.eye(d), L, None, False)}
    for s, kind in ((sl, "slice"), (nf, "lowrank")):
        m, cons, _ = kinds[kind].fresh()
        ref = heal(teachers[s.name], m, data, replace(c, lr=lrs[s.name]), cons)
        if s.name == "slice":
            assert same_state(m, s.model) and res["students"]["slice"]["history"] == ref["history"]
        else:
            sa, sb = m.state_dict(), s.model.state_dict()
            assert max((sa[k] - sb[k]).abs().max().item() for k in sa) < 1e-9
            assert np.allclose(res["students"]["noflip"]["history"], ref["history"], rtol=1e-10, atol=0)


# ---------------------------------------------------------------- (c) targets equal VExactTeacher
def _plan(adapter, capture):
    from mirror.heal import _Plan

    return _Plan(capture=frozenset(capture), first_grad=0, prefix_grad=True, share=False, checkpoint=False,
                 n_prefix=int(adapter.n_prefix_tokens))


def _equal(a: dict, b: dict):
    assert list(a) == list(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("adapter_kind", ["plain", "registers"])
def test_targets_equal_vexact_teacher_for_every_mode(ad, adapter_kind):
    a = ad if adapter_kind == "plain" else TinyRegAdapter()
    L, d = a.depth, a.embed_dim
    A = random_involution_fixing_ones(d, d // 2 + 3, seed=7)
    reg = np.array([1, 0]) if adapter_kind == "registers" else None
    x = torch.randn(3, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    cap = list(range(L))
    plan = _plan(a, cap)
    sh = SharedTeacher(a)
    tp = sh.run(x, True, cap)
    tp_raw = sh.run(x, False, cap)
    k = L - 1
    cases = [  # (TargetMode, VExactTeacher with the MUST-09 / E3 semantics, flipped pass)
        (TargetMode("symmetrised", "symmetrised", A=A, reg_perm=reg), VExactTeacher(a, A, L, reg, True), True),
        (TargetMode("symmetrised", "raw", reg_perm=reg), VExactTeacher(a, A, L, reg, True, block_targets="raw"), True),
        (TargetMode("symmetrised", "centred", reg_perm=reg), VExactTeacher(a, A, L, reg, True, block_targets="centred"),
         True),
        (TargetMode("raw", "centred"), VExactTeacher(a, A, L, reg, False), False),
        (TargetMode("raw", "raw"), VExactTeacher(a, A, L, reg, False, block_targets="raw"), False),
        (TargetMode("raw", "symmetrised", A=A, reg_perm=reg, k=k), VExactTeacher(a, A, k, reg, True), True),
    ]
    for mode, vt, flipped in cases:
        want, _ = vt.distill_targets(x, plan)
        _equal(sh.targets(tp if flipped else tp_raw, mode, cap), want)
    # partial span with flip-averaged final targets: blocks as VExactTeacher(k), final as VExactTeacher(L)
    got = sh.targets(tp, TargetMode("symmetrised", "symmetrised", A=A, reg_perm=reg, k=k), cap)
    blocks_k, _ = VExactTeacher(a, A, k, reg, True).distill_targets(x, plan)
    final_L, _ = VExactTeacher(a, A, L, reg, True).distill_targets(x, plan)
    _equal(got, {**{n: blocks_k[n] for n in blocks_k if n.startswith("block")}, "cls": final_L["cls"],
                 "patch": final_L["patch"]})
    assert not torch.equal(got[f"block{L - 1}"], final_L[f"block{L - 1}"])     # the last block is only centred
    # block_map: student block j against teacher block map[j]
    got = sh.targets(tp, TargetMode("symmetrised", "raw", reg_perm=reg, block_map={0: 0, 1: L - 1}), [0, 1])
    want, _ = VExactTeacher(a, A, L, reg, True, block_targets="raw").distill_targets(x, plan)
    assert torch.equal(got["block1"], want[f"block{L - 1}"]) and torch.equal(got["block0"], want["block0"])
    # a symmetrised mode refuses a pass without the mirrored images
    with pytest.raises(ValueError, match="mirrored"):
        sh.targets(tp_raw, TargetMode("symmetrised", "raw"), cap)
    if adapter_kind == "registers":       # the register pairing enters the symmetrised block targets' register rows
        plain = sh.targets(tp, TargetMode("symmetrised", "symmetrised", A=A), cap)
        paired = sh.targets(tp, TargetMode("symmetrised", "symmetrised", A=A, reg_perm=reg), cap)
        n = a.n_prefix_tokens
        assert torch.equal(plain["cls"], paired["cls"]) and torch.equal(plain["patch"], paired["patch"])
        assert torch.equal(plain["block0"][:, n:], paired["block0"][:, n:])
        assert not torch.equal(plain["block0"][:, 1:n], paired["block0"][:, 1:n])


def test_logits_target_is_the_flip_average_of_the_teacher_logits(ad):
    x = torch.randn(2, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    sh = SharedTeacher(ad)
    tp = sh.run(x, True, [0], logits=True)
    got = sh.targets(tp, TargetMode("symmetrised", "raw", logits=True), [0])
    assert list(got) == ["block0", "cls", "patch", "logits"] and got["logits"].shape == (2, 1000)
    m = ad.model
    l1 = m.forward_head(m.norm(ad.forward_tokens(x)))
    l2 = m.forward_head(m.norm(ad.forward_tokens(torch.flip(x, dims=[-1]))))
    assert (got["logits"] - 0.5 * (l1 + l2)).abs().max().item() < 1e-12


def test_target_keys_group_equal_modes(ad):
    sh = SharedTeacher(ad)
    A = random_involution_fixing_ones(ad.embed_dim, 36, seed=1)
    cap = list(range(ad.depth))
    k1 = sh.key(TargetMode("symmetrised", "raw", A=A), cap)
    assert k1 == sh.key(TargetMode("symmetrised", "raw"), cap)                 # A unused by raw blocks
    assert k1 != sh.key(TargetMode("symmetrised", "centred"), cap)
    assert sh.key(TargetMode("symmetrised", "symmetrised", A=A), cap) != sh.key(
        TargetMode("symmetrised", "symmetrised", A=random_involution_fixing_ones(ad.embed_dim, 36, seed=2)), cap)
    assert k1 != sh.key(TargetMode("symmetrised", "raw"), cap[:-1])


# ---------------------------------------------------------------- (e) rot90, logits and block_map
def test_rotate_back_puts_patches_on_the_original_grid():
    x = torch.randn(2, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    def patch_means(img):     # [B, 16, 3]: a patch statistic that does not depend on the patch's orientation
        return img.unfold(2, 14, 14).unfold(3, 14, 14).mean(dim=(-1, -2)).permute(0, 2, 3, 1).reshape(2, 16, 3)

    back = rotate_back_tokens(patch_means(torch.rot90(x, 1, (2, 3))), (4, 4), 0)
    assert torch.allclose(back, patch_means(x), atol=1e-12, rtol=0)
    assert not torch.allclose(patch_means(torch.rot90(x, 1, (2, 3))), patch_means(x))   # the rotation matters
    cls = torch.randn(2, 1, 3, dtype=torch.float64)
    full = rotate_back_tokens(torch.cat([cls, patch_means(torch.rot90(x, 1, (2, 3)))], 1), (4, 4), 1)
    assert torch.equal(full[:, :1], cls) and torch.allclose(full[:, 1:], patch_means(x), atol=1e-12, rtol=0)


def test_rot90_logits_and_block_map_students_run(ad):
    data = heal_data()
    c = cfg(range(ad.depth), max_steps=2)
    L = ad.depth
    rot = LockstepStudent("rot90", copy.deepcopy(ad.model), [], TargetMode("raw", "raw", input_transform="rot90"), 1e-3)
    lg = LockstepStudent("logits", copy.deepcopy(ad.model), [], TargetMode("symmetrised", "raw", logits=True), 1e-3)
    dropped = copy.deepcopy(ad.model)
    dropped.blocks = torch.nn.Sequential(*[dropped.blocks[i] for i in (0, 2, 3)])
    bd = LockstepStudent("blockdrop", dropped, [], TargetMode("symmetrised", "raw", block_map={0: 0, 1: 2, 2: 3}), 1e-3)
    res = train_lockstep(ad, [rot, lg, bd], data, c)
    for name, terms in (("rot90", [f"block{b}" for b in range(L)]), ("logits", [f"block{b}" for b in range(L)]),
                        ("blockdrop", ["block0", "block1", "block2"])):
        r = res["students"][name]
        assert r["steps_done"] == 2 and all(np.isfinite(r["history"]))
        assert set(r["normalisers"]) == {"cls", "patch", *terms} | ({"logits"} if name == "logits" else set())
    assert res["students"]["logits"]["weights"]["logits"] == 1.0
    # shapes of the student features against the targets
    x = torch.randn(2, 3, 56, 56, dtype=torch.float64)
    sh = SharedTeacher(ad)
    tp = sh.run(x, True, range(L), logits=True)
    for s, cap in ((rot, range(L)), (lg, range(L)), (bd, range(3))):
        plan = _make_plan(ad.model, s.model, HealConfig(blocks=list(cap)), ad.n_prefix_tokens)
        with torch.no_grad():
            sf = student_features(ad, s.model, x, plan, s.mode)
        tf = sh.targets(tp, s.mode, list(cap))
        assert list(tf) == list(sf) and all(sf[k].shape == tf[k].shape for k in tf)
    # the rot90 student, given the teacher's own weights, is the teacher on rot90(x) with its grid rotated back
    with torch.no_grad():
        sf = student_features(ad, copy.deepcopy(ad.model), x, _make_plan(ad.model, ad.model, HealConfig(blocks=[0]), 1),
                              TargetMode("raw", "raw", input_transform="rot90"))
        ref = ad.model.norm(ad.forward_tokens(torch.rot90(x, 1, (2, 3))))
    assert torch.equal(sf["cls"], ref[:, 0]) and torch.equal(sf["patch"], rotate_back_tokens(ref[:, 1:], (4, 4), 0))
    with pytest.raises(ValueError, match="rot90"):
        TargetMode("symmetrised", "raw", input_transform="rot90")
