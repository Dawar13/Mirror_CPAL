"""V-exact healing (mirror/heal_vexact.py, scripts/heal_vexact.py): CPU tests with the tiny random ViT
(float64 unless stated), synthetic uint8 images."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mirror.attention import attn_dims
from mirror.data import make_loader, mirror
from mirror.fold import fold_copy
from mirror.heal import HealConfig, HealData, evaluate_heldout, heal, load_trained_blocks
from mirror.heal_vexact import (VExactEmbeddings, VExactTeacher, build_vexact_student, embedding_parameters,
                                span_exactness, train_scope_for, vexact_param_counts)
from mirror.knn import evaluate_knn
from mirror.m1 import feature_invariance, random_head_structure, random_involution_fixing_ones, stream_equivariance
from mirror.matching import random_pairing
from mirror.stats import unflip_tokens

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_vx_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def structure(adapter, seed=0):
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=seed + 3)
    perms = [random_pairing(h, h // 3, seed=seed + 10 + b) for b in range(L)]
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


def with_model(adapter, model):
    """The same adapter interface over another model (a teacher that is itself a converted model)."""
    ad = copy.copy(adapter)
    ad.model = model
    return ad


def span_errors(rows, k):
    return max(r["rel_max"] for r in rows if r["layer"] <= k), [r["rel_max"] for r in rows if r["layer"] > k]


def knn_flip_consistency(adapter, model, seed=3):
    rng = np.random.default_rng(seed)
    imgs = rng.integers(0, 256, size=(48, 56, 56, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(4), 12)
    bank = make_loader(imgs, labels, np.arange(32), 16)
    query = make_loader(imgs, labels, np.arange(32, 48), 16)
    return evaluate_knn(adapter, bank, query, "cpu", 4, k=5, model=model)["flip_consistency"]


# ---------------------------------------------------------------- targets
@pytest.mark.parametrize("k", [2, 4])
def test_targets_are_matched_exactly_by_an_exact_model(tiny64, k):
    """With an exactly equivariant V-exact model as the teacher, the centred and symmetrised targets are
    that model's own outputs: a copy of it as the student has zero loss on every term."""
    A, perms, maps = structure(tiny64)
    exact, _, _ = build_vexact_student(tiny64, A, k, perms, maps)
    student = copy.deepcopy(exact)
    teacher = VExactTeacher(with_model(tiny64, exact), A, k)
    assert teacher.even_head == (k == tiny64.depth)
    data = heal_data()
    terms = evaluate_heldout(teacher, student, data, cfg(range(k)))
    assert set(terms) == {"total", "cls", "patch", *[f"block{b}" for b in range(k)]}
    assert max(terms.values()) < 1e-20, terms
    # Teeth: symmetrising with another operator does not give the model's own stream.
    wrong = VExactTeacher(with_model(tiny64, exact), random_involution_fixing_ones(tiny64.embed_dim, 36, seed=99), k)
    bad = evaluate_heldout(wrong, student, data, cfg(range(k)))
    assert min(bad[f"block{b}"] for b in range(k)) > 1e-6


def test_centred_targets_match_the_folded_original(tiny64):
    """Block targets are the teacher's stream centred over channels: exactly the folded model's stream."""
    A, _, _ = structure(tiny64)
    folded, _ = fold_copy(tiny64.model, final_norm_affine=False)
    data = heal_data()
    terms = evaluate_heldout(VExactTeacher(tiny64, A, 2, symmetrise=False), folded, data, cfg([0, 1]))
    assert max(terms.values()) < 1e-20, terms
    # Teeth: the plain teacher's block outputs carry the channel mean the folded stream does not have.
    plain = evaluate_heldout(tiny64, folded, data, cfg([0, 1]))
    assert min(plain["block0"], plain["block1"]) > 1e-6 and max(plain["cls"], plain["patch"]) < 1e-20


def _register_adapter():
    from test_m1_run import _timm_with_registers

    return _timm_with_registers(n_reg=2)


@pytest.mark.parametrize("which", ["tiny", "registers"])
def test_symmetrised_targets_have_the_student_symmetry(tiny64, which):
    """T(mirror x) = align(T(x)) A* for block targets; at full conversion the final targets of x and
    mirror(x) are equal (CLS) and grid-flipped (patches), the symmetry of the even-head student."""
    ad = tiny64 if which == "tiny" else _register_adapter()
    reg_perm = None if which == "tiny" else [1, 0]
    A, _, _ = structure(ad)
    L = ad.depth
    plan = SimpleNamespace(capture=frozenset(range(L)), n_prefix=ad.n_prefix_tokens)
    x = images(3, seed=4)
    for final_target in ("symmetrised", "even"):
        t = VExactTeacher(ad, A, L, reg_perm, final_target=final_target)
        fx, start = t.distill_targets(x, plan)
        fm, _ = t.distill_targets(mirror(x), plan)
        assert start is None
        A64 = torch.as_tensor(A)
        for b in range(L):
            ref = fx[f"block{b}"].abs().max()
            assert ((fm[f"block{b}"] - t.align(fx[f"block{b}"]) @ A64).abs().max() / ref).item() < 1e-12
            assert fx[f"block{b}"].mean(-1).abs().max().item() < 1e-12            # centred, like the folded stream
        assert (fm["cls"] - fx["cls"]).abs().max().item() < 1e-12
        assert (fm["patch"] - unflip_tokens(fx["patch"], ad.grid_hw, 0)).abs().max().item() < 1e-12
        if final_target == "even":                                              # even before the final affine
            norm = ad.model.norm
            u = (fx["patch"] - norm.bias) / norm.weight
            Q = 0.5 * (torch.eye(A64.shape[0], dtype=torch.float64) - A64)
            assert ((u @ Q).abs().max() / u.abs().max()).item() < 1e-12
    # Teeth: without symmetrisation the original teacher has no such symmetry.
    raw = VExactTeacher(ad, A, L, reg_perm, symmetrise=False)
    fx, _ = raw.distill_targets(x, plan)
    fm, _ = raw.distill_targets(mirror(x), plan)
    assert (fm["cls"] - fx["cls"]).abs().max().item() > 1e-3


# ---------------------------------------------------------------- healing keeps the model exact
@pytest.mark.parametrize("k,train_emb", [(2, True), (4, True), (4, False)])
def test_vexact_healing_keeps_exact_equivariance(tiny64, k, train_emb):
    L = tiny64.depth
    A, perms, maps = structure(tiny64)
    student, cons, _ = build_vexact_student(tiny64, A, k, perms, maps, train_embeddings=train_emb)
    before = {n: v.clone() for n, v in student.state_dict().items()}
    teacher_before = {n: v.clone() for n, v in tiny64.model.state_dict().items()}
    res = heal(VExactTeacher(tiny64, A, k), student, heal_data(),
               cfg(range(k), train_scope=train_scope_for(train_emb), decay_1d=True), cons, None)
    assert res["steps_done"] == 4 and np.isfinite(res["heldout_final"]["total"])
    # Exact over the span after training: mirrored tokens = A* x original tokens at every layer 0..k.
    x = images(3, seed=5)
    inside, outside = span_errors(stream_equivariance(tiny64, student, x, A), k)
    assert inside < 1e-10
    if k < L:
        assert min(outside) > 1e-6                                       # teeth: unconverted blocks break it
    ex = span_exactness(tiny64, student, [x, images(2, seed=6)], A, k)
    assert ex["pass"] and ex["max_rel_inside"] < 1e-10 and ex["n_images"] == 5
    checks = res["constraint_checks"]
    assert checks["0:MirrorMLPConstraint"]["max_rel_violation"] < 1e-12
    assert checks["1:MirrorAttnConstraint"]["max_rel_violation"] < 1e-12
    assert checks["2:VExactEmbeddings"]["max_share"] < 1e-24
    # What trained: the converted blocks' matrices and biases, and the embeddings only if asked.
    after = student.state_dict()
    changed = {n for n in before if not torch.equal(before[n], after[n])}
    emb = {n for n, _ in embedding_parameters(student)}
    allowed = {n for n in before if n.startswith(tuple(f"blocks.{b}." for b in range(k)))
               and not any(s in n for s in (".norm1.", ".norm2.", ".ls1.", ".ls2."))}
    allowed |= emb if train_emb else set()
    assert changed <= allowed, sorted(changed - allowed)
    assert {"blocks.0.mlp.fc1.weight", "blocks.0.attn.qkv.weight", f"blocks.{k - 1}.mlp.fc2.weight"} <= changed
    assert ("patch_embed.proj.weight" in changed) == train_emb and ("pos_embed" in changed) == train_emb
    assert set(res["trainable_names"]) <= allowed
    assert all(torch.equal(v, tiny64.model.state_dict()[n]) for n, v in teacher_before.items())
    if k == L:                                                           # full conversion: invariant features
        assert checks["2:VExactEmbeddings"]["even_head_max_abs_err"] < 1e-12
        assert feature_invariance(tiny64, student, x)["rel_max"] < 1e-10
        assert knn_flip_consistency(tiny64, student) == 100.0


def test_constraints_have_teeth(tiny64):
    """Training without the projections breaks the exact equivariance: blocks without the MLP and
    attention constraints, embeddings trained without re-projection."""
    A, perms, maps = structure(tiny64)
    x = images(3, seed=5)
    data = heal_data()
    free, _, _ = build_vexact_student(tiny64, A, 2, perms, maps)
    heal(VExactTeacher(tiny64, A, 2), free, data, cfg([0, 1]), [], None)
    rows = stream_equivariance(tiny64, free, x, A)
    assert rows[0]["rel_max"] < 1e-10 and rows[1]["rel_max"] > 1e-6     # embeddings frozen, block 1 broken

    class NoProjection(VExactEmbeddings):
        def __call__(self, model):
            return None

    s, cons, _ = build_vexact_student(tiny64, A, 2, perms, maps, train_embeddings=True)
    loose = NoProjection(tiny64, A, [0, 1], train=True)
    heal(VExactTeacher(tiny64, A, 2), s, data, cfg([0, 1], train_scope="all"), cons[:2] + [loose], None)
    assert stream_equivariance(tiny64, s, x, A)[0]["rel_max"] > 1e-6
    assert loose.check(s)["max_share"] > 1e-12
    # A model whose embeddings were never projected is refused.
    folded, _ = fold_copy(tiny64.model, final_norm_affine=False)
    with pytest.raises(ValueError):
        VExactEmbeddings(tiny64, A, [0, 1]).prepare(folded)


def test_checkpoint_restores_the_healed_student(tiny64, tmp_path):
    A, perms, maps = structure(tiny64)
    L = tiny64.depth
    student, cons, _ = build_vexact_student(tiny64, A, L, perms, maps)
    heal(VExactTeacher(tiny64, A, L), student, heal_data(), cfg(range(L), max_steps=2, train_scope="all"), cons, tmp_path)
    fresh, _, _ = build_vexact_student(tiny64, A, L, perms, maps)
    load_trained_blocks(fresh, tmp_path / "trained_blocks.pt")
    x = images(3, seed=7)
    assert (tiny64.cls_feature(x, fresh) - tiny64.cls_feature(x, student)).abs().max().item() == 0.0


def test_param_counts_match_m1(tiny64):
    m1 = load_script("m1_attention")
    A, perms, maps = structure(tiny64)
    L = tiny64.depth
    n_model = int(sum(p.numel() for p in tiny64.model.parameters()))
    for k in (2, L):
        ours = vexact_param_counts(tiny64, A, k, perms, maps)
        S = {"A_star": {k: A}, "maps": maps, "a_out_offset": 1}
        ops = {"A_layers": [A] * (L + 1), "perms": perms}
        theirs = m1.variant_params("vexact", k, tiny64, S, ops, n_model)
        for key, v in theirs.items():
            assert ours[key] == pytest.approx(v, rel=0, abs=1e-12), key


# ---------------------------------------------------------------- CLI: scripts/heal_vexact.py --tiny
def test_cli_tiny_runs_end_to_end(tmp_path):
    cli = load_script("heal_vexact")
    config = str(ROOT / "configs" / "m3_vexact.yaml")
    s = cli.main(["--config", config, "--tiny", "--out-dir", str(tmp_path)])      # tiny k = 4 = full conversion
    out = tmp_path / "k4"
    assert s["method"] == "vexact" and s["k"] == 4 == s["depth"] and s["seed"] == 0 and len(s["git_hash"]) >= 7
    assert s["lr_selection"]["best_lr"] in s["lr_selection"]["lrs"]
    fin = s["final"]
    assert fin["steps_done"] == fin["total_steps"] and fin["config"]["train_scope"] == "all"
    assert "patch_embed.proj.weight" in fin["trainable_names"]
    checks = fin["constraint_checks"]
    assert checks["0:MirrorMLPConstraint"]["max_rel_violation"] < 1e-6             # float32 model
    assert checks["1:MirrorAttnConstraint"]["max_rel_violation"] < 1e-6
    assert checks["2:VExactEmbeddings"]["max_share"] < 1e-10
    assert s["exactness"]["pass"] and s["flipcheck"]["pass"] and s["flipcheck"]["flip_consistency"] == 100.0
    assert {"original", "zero_shot", "healed"} <= set(s["knn"]) and 0 <= s["knn"]["healed"]["top1"] <= 100
    for f in ("summary.json", "SUMMARY.md", "knn.json", "exactness.json", "flipcheck.json", "final/metrics.json",
              "final/trained_blocks.pt", "lr_selection/lr_selection.json"):
        assert (out / f).is_file(), f
    for f in ("summary.json", "knn.json", "exactness.json", "flipcheck.json", "final/metrics.json"):
        meta = json.loads((out / f).read_text())
        assert meta.get("git_hash") and meta.get("seed") == 0, f
    text = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "—" not in text and "PASS" in text and "full conversion" in text
    # Partial span with frozen embeddings: no flip check, still exact over the span.
    s2 = cli.main(["--config", config, "--tiny", "--k", "2", "--freeze-embeddings", "--lr", "1e-4",
                   "--out-dir", str(tmp_path)])
    assert s2["flipcheck"] is None and s2["exactness"]["pass"] and s2["final"]["config"]["train_scope"] == "converted"
    assert not any(n.startswith(("patch_embed", "pos_embed", "cls_token")) for n in s2["final"]["trainable_names"])
    assert s2["lr_selection"]["skipped"] and (tmp_path / "k2" / "summary.json").is_file()
    # The alternative final target runs too.
    s3 = cli.main(["--config", config, "--tiny", "--final-target", "even", "--lr", "1e-4", "--max-steps", "2",
                   "--out-dir", str(tmp_path / "even")])
    assert s3["teacher"]["final_target"] == "even" and s3["flipcheck"]["pass"]
