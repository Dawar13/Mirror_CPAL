"""Matched budgets (mirror/budget.py): CPU only, no downloads, no model weights beyond the tiny random ViT.

  * Mirror-exact's matrix count equals heal_vexact.vexact_param_counts for the same structure (both components, and
    the MLP and attention parts on their own);
  * basis changes are counted as specified (Mirror-exact: d^2 at a span end that is not a model boundary; Mirror-local:
    d^2 per block; SliceGPT: shortcuts, rotations, entry and exit);
  * biases are dense and separate; the embedding saving is separate and consistent;
  * every method's allocation reaches the total and stays within 1% at DINOv2-B sizes for the four MUST-07 sizes
    (dims only; the structure is random with test values, not fitted numbers);
  * at the dense budget every method keeps every weight.
"""
import numpy as np
import pytest
import torch

from mirror import budget as BU
from mirror import variants as V
from mirror.attention import attention_free_params
from mirror.heal_vexact import vexact_param_counts
from mirror.m1 import random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing

B_DIMS = {"d": 768, "h": 3072, "H": 12, "dh": 64, "depth": 12}     # DINOv2 ViT-B sizes
SIZES = [(("mlp",), 6), (("mlp", "attn"), 6), (("mlp", "attn"), 9), (("mlp", "attn"), 12)]   # MUST-07 sizes


def tiny_structure(adapter, seed=0, d_plus=None):
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    A = random_involution_fixing_ones(d, d // 2 + 5 if d_plus is None else d_plus, seed=seed + 1)
    perms = [random_pairing(h, h // 3 + 5 * b, seed=seed * 100 + 50 + b) for b in range(L)]
    heads = [random_head_structure(4, 16, 1 + (b % 2), seed=seed * 100 + 90 + b) for b in range(L)]
    return A, perms, heads


def b_structure(seed=0):
    """A random structure at DINOv2-B sizes (test values only)."""
    A = random_involution_fixing_ones(768, 500, seed=seed + 3)
    perms = [random_pairing(3072, 1100 - 50 * b, seed=seed * 100 + b) for b in range(12)]
    heads = [random_head_structure(12, 64, 3 + (b % 3), seed=seed * 100 + 40 + b) for b in range(12)]
    return A, perms, heads


# ---------------------------------------------------------------- Mirror-exact count
@pytest.mark.parametrize("k", [1, 2, 4])
def test_mirror_exact_count_equals_vexact_param_counts(tiny64, k):
    A, perms, heads = tiny_structure(tiny64)
    ref = vexact_param_counts(tiny64, A, k, perms, heads)
    both = BU.mirror_exact_budget(tiny64, A, range(k), perms, heads, ("mlp", "attn"))
    assert both.matrix_total == ref["converted_params_kept"]
    assert both.dense_total == ref["converted_params_dense"]
    mlp = BU.mirror_exact_budget(tiny64, A, range(k), perms, None, ("mlp",))
    att = BU.mirror_exact_budget(tiny64, A, range(k), None, heads, "attn")
    assert mlp.matrix_total == ref["mlp_params_kept"] and att.matrix_total == ref["attn_params_kept"]
    # perms and head maps aligned with the blocks give the same budget as indexed by block
    al = BU.mirror_exact_budget(tiny64, A, range(k), perms[:k], heads[:k])
    assert al.matrix_total == both.matrix_total
    assert both.total == both.matrix_total + both.basis_change


def test_per_matrix_counts_follow_attention_free_params(tiny64):
    A, perms, heads = tiny_structure(tiny64)
    bud = BU.mirror_exact_budget(tiny64, A, [0, 1, 2], perms, heads)
    for b in range(3):
        att = attention_free_params(tiny64.embed_dim, heads[b], A, A)
        for name, key in (("wq", "Wq"), ("wk", "Wk"), ("wv", "Wv"), ("wo", "Wo")):
            assert bud.matrix(b, name).mirror_exact == att[key]
        assert bud.matrix(b, "fc1").mirror_exact == bud.matrix(b, "fc2").mirror_exact
        assert bud.matrix(b, "fc1").shape == (tiny64.mlp_hidden, tiny64.embed_dim)
        assert bud.matrix(b, "wo").shape == (tiny64.embed_dim, 64)


def test_dims_dict_gives_the_same_budget_as_the_model(tiny64):
    A, perms, heads = tiny_structure(tiny64)
    dims = BU.model_dims(tiny64)
    a = BU.mirror_exact_budget(tiny64, A, range(3), perms, heads)
    b = BU.mirror_exact_budget(dims, A, range(3), perms, heads)
    assert a.total == b.total and [m.mirror_exact for m in a.matrices] == [m.mirror_exact for m in b.matrices]


# ---------------------------------------------------------------- basis changes
def test_mirror_exact_basis_change_rule(tiny64):
    A, perms, heads = tiny_structure(tiny64)
    d, L = tiny64.embed_dim, tiny64.depth
    cases = {(0, 1): d * d,           # starts at the model start, ends inside: one end
             (0, 1, 2, 3): 0,         # full model: none
             (1, 2): 2 * d * d,       # both ends inside the model
             (1, 2, 3): d * d}        # ends at the model end
    for blocks, want in cases.items():
        bud = BU.mirror_exact_budget(tiny64, A, list(blocks), perms, heads)
        assert bud.basis_change == want == BU.mirror_exact_basis_change(d, blocks, L), blocks
        assert bud.total == bud.matrix_total + want
    with pytest.raises(ValueError):
        BU.mirror_exact_budget(tiny64, A, [0, 2], perms, heads)      # not a span


def test_mirror_local_counts_and_basis(tiny64):
    struct = V.Structure(A_layers=[random_involution_fixing_ones(64, 30 + 2 * i, seed=i) for i in range(5)],
                         d_plus=[30 + 2 * i for i in range(5)], d_minus=[34 - 2 * i for i in range(5)],
                         perms=[random_pairing(256, 70 + b, seed=b) for b in range(4)])
    _, _, heads = tiny_structure(tiny64)
    for k in (1, 3):
        bud = BU.mirror_local_budget(tiny64, struct, range(k), ("mlp", "attn"), head_maps=heads)
        eq = V.mirror_param_counts(struct, range(k))
        assert [(bud.matrix(b, "fc1").mirror_exact, bud.matrix(b, "fc2").mirror_exact) for b in range(k)] == \
            [tuple(map(int, e)) for e in eq]
        for b in range(k):
            att = attention_free_params(64, heads[b], struct.A_layers[b], struct.A_layers[b + 1])
            assert bud.matrix(b, "wq").mirror_exact == att["Wq"] and bud.matrix(b, "wo").mirror_exact == att["Wo"]
        assert bud.basis_change == k * 64 * 64 and bud.total == bud.matrix_total + k * 64 * 64


def test_slicegpt_basis_changes(tiny64):
    A, perms, heads = tiny_structure(tiny64)
    d, h, D = 64, 256, 64
    for blocks in ([0, 1], [0, 1, 2, 3], [1, 2]):
        k = len(blocks)
        enter, leave = int(blocks[0] > 0), int(blocks[-1] < 3)
        bud = BU.mirror_exact_budget(tiny64, A, blocks, perms, heads, ("mlp", "attn"))
        for basis in ("per_block", "shared"):
            al = BU.allocate("slicegpt", bud, basis=basis)
            ds = al[0]["d_s"]
            tot = BU.allocation_totals(al)
            sc = 2 * k * ds * ds if basis == "per_block" else 0
            assert tot["basis_change"] == sc + (enter + leave) * d * ds
            assert tot["matrices"] == k * (4 * D + 2 * h) * ds
            names = sorted(a["name"] for a in al if a["kind"] == "basis")
            assert names.count("shortcut_attn") == names.count("shortcut_mlp") == (k if basis == "per_block" else 0)
            assert names.count("entry") == enter and names.count("exit") == leave
            # the smallest d_s that reaches the total
            lay = BU._slice_layout(bud, basis)
            assert tot["total"] >= bud.total or ds == d
            assert ds == 1 or lay["per_d"] * (ds - 1) + lay["basis_params"](ds - 1) < bud.total
        mlp = BU.mirror_exact_budget(tiny64, A, blocks, perms, None, ("mlp",))
        al = BU.allocate("slicegpt", mlp, basis="per_block")
        assert BU.allocation_totals(al)["basis_change"] == d * d * (k - 1 + enter + leave)
        assert all(a["shape"] == [d, d] for a in al if a["kind"] == "basis")
        al = BU.allocate("slicegpt", mlp, basis="shared")
        assert BU.allocation_totals(al)["basis_change"] == d * d * (enter + leave)
    for m in ("svd", "svdllm", "asvd_scale", "zipit"):
        assert BU.allocation_totals(BU.allocate(m, bud))["basis_change"] == 0
    assert BU.allocation_totals(BU.allocate("flap", bud, allocation="uniform"))["basis_change"] == 0


def test_biases_dense_and_separate(tiny64):
    A, perms, heads = tiny_structure(tiny64)
    for comps, per in ((("mlp",), 256 + 64), (("attn",), 3 * 64 + 64), (("mlp", "attn"), 256 + 64 + 3 * 64 + 64)):
        bud = BU.mirror_exact_budget(tiny64, A, range(3), perms, heads, comps)
        assert bud.biases_dense == 3 * per
        assert bud.total == bud.matrix_total + bud.basis_change      # biases never in the total


def test_embedding_saving_is_separate_and_consistent(tiny64):
    A, perms, heads = tiny_structure(tiny64)
    bud = BU.mirror_exact_budget(tiny64, A, range(2), perms, heads, grid_hw=tiny64.grid_hw)
    emb = bud.embedding
    d = 64
    dp = (d + int(round(np.trace(A)))) // 2
    assert emb["dense"] == sum(p["dense"] for p in emb["parts"].values())
    assert emb["mirror_exact"] == sum(p["mirror_exact"] for p in emb["parts"].values())
    W = tiny64.model.patch_embed.proj.weight
    assert emb["parts"]["patch_weight"]["dense"] == W.numel()
    assert emb["parts"]["patch_weight"]["mirror_exact"] == W.numel() // 2      # even kernel width: tr(flip) = 0
    assert emb["parts"]["cls_token"]["mirror_exact"] == dp and emb["parts"]["patch_bias"]["mirror_exact"] == dp
    assert emb["parts"]["pos_grid"]["mirror_exact"] == 16 * d // 2              # 4 x 4 grid, even width
    assert bud.total == bud.matrix_total + bud.basis_change                      # not in the matched total
    later = BU.mirror_exact_budget(tiny64, A, [1, 2], perms, heads, grid_hw=tiny64.grid_hw)
    assert later.embedding == {}                                                 # span not at the embeddings


# ---------------------------------------------------------------- allocation within 1% at DINOv2-B sizes
@pytest.fixture(scope="module")
def b_struct():
    return b_structure()


@pytest.mark.parametrize("comps,k", SIZES, ids=[f"{'+'.join(c)}-{k}" for c, k in SIZES])
def test_every_method_within_1pct_at_dinov2_b_sizes(b_struct, comps, k):
    A, perms, heads = b_struct
    bud = BU.mirror_exact_budget(B_DIMS, A, range(k), perms, heads, comps)
    assert bud.basis_change == (768 * 768 if k < 12 else 0)
    targets = BU.matrix_targets(bud)
    assert bud.total <= sum(targets.values()) < bud.total + len(targets)
    rng = np.random.default_rng(k)
    scores = {b: {"mlp": torch.tensor(rng.standard_normal(3072)), "attn": torch.tensor(rng.standard_normal(12))}
              for b in range(k)}
    cases = [("svd", {}), ("svdllm", {}), ("asvd_scale", {}), ("slicegpt", {"basis": "per_block"}),
             ("slicegpt", {"basis": "shared"}), ("flap", {"allocation": "uniform"}),
             ("flap", {"allocation": "global", "scores": scores}), ("zipit", {})]
    for m, hp in cases:
        al = BU.allocate(m, bud, **hp)
        tot = BU.allocation_totals(al)
        gap = BU.gap_of(tot["total"], bud.total)
        step = BU.allocation_step(m, bud, al, **hp)
        assert 0 <= gap <= 0.01, (m, hp.get("basis", hp.get("allocation")), gap)
        assert tot["total"] - bud.total < step, (m, tot["total"] - bud.total, step)    # at most one step above
        mats = [a for a in al if a["kind"] != "basis"]
        assert sorted((a["block"], a["name"]) for a in mats) == sorted((x.block, x.name) for x in bud.matrices)


def test_flap_global_keeps_the_best_items(b_struct):
    A, perms, heads = b_struct
    bud = BU.mirror_exact_budget(B_DIMS, A, range(3), perms, heads)
    rng = np.random.default_rng(5)
    scores = {b: {"mlp": torch.tensor(rng.standard_normal(3072)), "attn": torch.tensor(rng.standard_normal(12) + 0.5)}
              for b in range(3)}
    al = BU.allocate("flap", bud, allocation="global", scores=scores)
    kept_n = {(a["block"], j) for a in al if a["name"] == "fc1" for j in a["kept"]}
    kept_h = {(a["block"], j) for a in al if a["name"] == "wq" for j in a["heads_kept"]}
    s_kept = [float(scores[b]["mlp"][j]) for b, j in kept_n] + [float(scores[b]["attn"][j]) for b, j in kept_h]
    s_gone = [float(scores[b]["mlp"][j]) for b in range(3) for j in range(3072) if (b, j) not in kept_n]
    # every removed neuron scores below the weakest kept item except the one forced neuron per block
    forced = {b: max(range(3072), key=lambda j: float(scores[b]["mlp"][j])) for b in range(3)}
    assert all((b, forced[b]) in kept_n for b in range(3))
    assert max(s_gone) <= sorted(s_kept)[3]


def test_uniform_flap_and_zipit_follow_the_block_shares(b_struct):
    A, perms, heads = b_struct
    bud = BU.mirror_exact_budget(B_DIMS, A, range(6), perms, heads)
    bt = BU._block_targets(bud)
    for m, hp in (("flap", {"allocation": "uniform"}), ("zipit", {})):
        al = BU.allocate(m, bud, **hp)
        for b in range(6):
            got = sum(a["params"] for a in al if a["block"] == b)
            assert abs(got - (bt[b]["mlp"] + bt[b]["attn"])) <= 2 * 2 * 768, (m, b)   # within two neurons per block


# ---------------------------------------------------------------- the dense budget keeps everything
def test_dense_budget_keeps_every_weight(tiny64):
    bud = BU.dense_budget(tiny64, range(4), ("mlp", "attn"))
    assert bud.total == bud.dense_total and bud.basis_change == 0
    for m in ("svd", "svdllm", "asvd_scale"):
        al = BU.allocate(m, bud)
        assert all(a["dense_storage"] and a["rank"] == min(a["shape"]) for a in al)
        assert BU.allocation_totals(al)["total"] == bud.total
    al = BU.allocate("slicegpt", bud, basis="shared")
    assert all(a["d_s"] == 64 for a in al if a["kind"] == "slice") and BU.allocation_totals(al)["total"] == bud.total
    al = BU.allocate("flap", bud, allocation="uniform")
    assert all(a["n"] == (256 if a["name"] in ("fc1", "fc2") else 64) for a in al)
    al = BU.allocate("zipit", bud)
    assert all(a["n"] == (256 if a["name"] in ("fc1", "fc2") else 64) for a in al)
