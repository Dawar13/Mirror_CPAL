"""MUST-02 exactness audit (scripts/must02_exactness.py): CPU tests with tiny random timm ViTs (one with two
register tokens), synthetic images. Nothing here downloads data or weights."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror.adapters import TimmTinyAdapter, ViTAdapter, _randomise_affine
from mirror.attention import attn_dims
from mirror.m1 import align_mirror_tokens, build_vexact, random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs" / "m3_vexact.yaml")


def load_script(name):
    key = f"{name}_must02test"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


M = load_script("must02_exactness")
BLOCK_COMPONENTS = ["norm1", "q", "k", "v", "attn_logits", "attn_probs", "head_out", "attn_proj", "ls1", "resid_attn",
                    "norm2", "fc1", "act", "fc2", "ls2", "block_out"]


class TinyRegAdapter(TimmTinyAdapter):
    """The tiny random timm ViT with register tokens (timm reg_tokens: positions for CLS, registers, patches)."""

    def __init__(self, reg_tokens=2, img_size=56, seed=0, dtype=torch.float64):
        from timm.models.vision_transformer import VisionTransformer

        torch.manual_seed(seed)
        model = VisionTransformer(img_size=img_size, patch_size=14, embed_dim=64, depth=4, num_heads=4,
                                  init_values=1e-5, reg_tokens=reg_tokens)
        _randomise_affine(model, seed)
        with torch.no_grad():
            model.reg_token.normal_(0.0, 0.5)
        ViTAdapter.__init__(self, model.to(dtype), img_size=img_size, patch_size=14,
                            n_prefix_tokens=int(model.num_prefix_tokens))


def random_structure(adapter, seed=0, reg_perm=None):
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=seed + 3)
    perms = [random_pairing(h, h // 3, seed=seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2, 0, 1)[b % 4], seed=seed + 20 + b) for b in range(L)]
    return A, perms, maps, reg_perm


def exact_model(adapter, seed=0, reg_perm=None):
    A, perms, maps, rp = random_structure(adapter, seed, reg_perm)
    L = adapter.depth
    model, info = build_vexact(adapter, A, range(L), perms, maps, reg_perm=rp, even_head=True)
    st = M.ExactStructure(A=A, k=L, reg_perm=np.asarray(rp if rp is not None else [], dtype=np.int64),
                          perms=perms, maps=info["maps_after"])
    return model, st


def images(n=4, seed=0, dtype=torch.float64):
    return torch.randn(n, 3, 56, 56, dtype=dtype, generator=torch.Generator().manual_seed(seed))


def by_key(rows):
    return {(r["block"], r["component"]): r for r in rows}


# ---------------------------------------------------------------- alignment
def test_token_index_matches_align_mirror_tokens():
    g = torch.Generator().manual_seed(1)
    for grid, n_pre, rp in (((4, 4), 1, None), ((3, 5), 3, np.array([1, 0])), ((4, 4), 5, np.array([2, 1, 0, 3]))):
        t = torch.randn(2, n_pre + grid[0] * grid[1], 6, generator=g)
        idx = M.token_index(grid, n_pre, rp, 1)
        assert torch.equal(t[:, idx], align_mirror_tokens(t, grid, n_pre, rp, 1))
    assert torch.equal(M.token_index((2, 3), 0), torch.tensor([2, 1, 0, 5, 4, 3]))


# ---------------------------------------------------------------- the audit in float64
def test_float64_audit_reaches_rounding_level_with_registers():
    ad = TinyRegAdapter(reg_tokens=2)
    model, st = exact_model(ad, reg_perm=np.array([1, 0]))
    rows = M.audit_model(ad, model, st, [images(3), images(2, seed=1)], "cpu", tol=1e-5)
    got = by_key(rows)
    for b in range(1, ad.depth + 1):                        # every component of every block is measured
        for c in BLOCK_COMPONENTS:
            assert (b, c) in got, (b, c)
    for c in ("patch_embed", "tokens0", "final_rms", "even_proj", "final_affine", "cls_feature"):
        assert (None, c) in got, c
    assert all(r["in_span"] and r["n_images"] == 5 for r in rows)
    worst = max(r["rel_tensor_max"] for r in rows)
    worst_tok = max(r["rel_token_max"] for r in rows)
    assert worst < 1e-12 and worst_tok < 1e-12, (worst, worst_tok)
    s = M.summarise(rows, 1e-5)
    assert s["all_below_tol"] and s["all_below_tol_per_token"] and s["n_components"] == len(rows)
    # The wrong register pairing breaks the register tokens at layer 0 at once.
    st_bad = M.ExactStructure(A=st.A, k=st.k, reg_perm=np.array([0, 1]), perms=st.perms, maps=st.maps)
    bad = by_key(M.audit_model(ad, model, st_bad, [images(2)], "cpu"))
    assert bad[(None, "tokens0")]["rel_tensor_max"] > 1e-3
    assert bad[(None, "patch_embed")]["rel_tensor_max"] < 1e-12       # patch tokens do not involve registers


def test_audit_localises_a_broken_component():
    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)
    model, st = exact_model(ad, seed=2)
    with torch.no_grad():                                   # break block 3's fc2 only (code index 2)
        model.blocks[2].mlp.fc2.weight.add_(0.05 * torch.randn_like(model.blocks[2].mlp.fc2.weight))
    got = by_key(M.audit_model(ad, model, st, [images(3)], "cpu"))
    for b in (1, 2):
        assert max(got[(b, c)]["rel_tensor_max"] for c in BLOCK_COMPONENTS) < 1e-12
    for c in ("norm1", "q", "attn_probs", "head_out", "attn_proj", "resid_attn", "norm2", "fc1", "act"):
        assert got[(3, c)]["rel_tensor_max"] < 1e-12, c
    assert got[(3, "fc2")]["rel_tensor_max"] > 1e-3
    assert got[(3, "block_out")]["rel_tensor_max"] > 1e-4
    assert got[(None, "cls_feature")]["rel_tensor_max"] > 1e-4


def test_wrong_head_map_shows_in_q_k_v_but_not_in_the_stream():
    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)
    model, st = exact_model(ad, seed=4)
    hm = st.maps[0]
    pair = hm.pairs[0]
    wrong = type(hm)(sigma=np.arange(hm.H), B=np.broadcast_to(np.eye(hm.dh), hm.B.shape).copy(),
                     C=np.broadcast_to(np.eye(hm.dh), hm.C.shape).copy())
    st_bad = M.ExactStructure(A=st.A, k=st.k, reg_perm=st.reg_perm, perms=st.perms, maps=[wrong] + st.maps[1:])
    got = by_key(M.audit_model(ad, model, st_bad, [images(2)], "cpu"))
    assert pair and got[(1, "q")]["rel_tensor_max"] > 1e-2 and got[(1, "attn_probs")]["rel_tensor_max"] > 1e-3
    assert got[(1, "attn_proj")]["rel_tensor_max"] < 1e-12 and got[(2, "q")]["rel_tensor_max"] < 1e-12


# ---------------------------------------------------------------- structure read from the weights
def test_infer_structure_recovers_the_built_structure():
    ad = TinyRegAdapter(reg_tokens=2)
    model, st = exact_model(ad, seed=1, reg_perm=np.array([1, 0]))
    inf, res = M.infer_structure(ad, model, st.A)
    assert res["max"] < 1e-10
    assert np.array_equal(inf.reg_perm, [1, 0])
    for p, q in zip(inf.perms, st.perms):
        assert np.array_equal(p, q)
    for hi, hk in zip(inf.maps, st.maps):
        assert np.array_equal(hi.sigma, hk.sigma)
        assert np.abs(hi.B - hk.B).max() < 1e-8 and np.abs(hi.C - hk.C).max() < 1e-8
    rows = M.audit_model(ad, model, inf, [images(2)], "cpu")
    assert max(r["rel_tensor_max"] for r in rows) < 1e-12
    A_even = M.operator_from_even_head(model)
    assert np.abs(A_even - st.A).max() < 1e-12
    with pytest.raises(ValueError):                         # the original model is not exactly equivariant
        M.infer_structure(ad, ad.model, st.A)


# ---------------------------------------------------------------- flip check
def test_flip_check_full_and_on_the_original():
    from mirror.data import make_loader

    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    model, _ = exact_model(ad, seed=3)
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, size=(40, 56, 56, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(4), 10)
    bank, query = make_loader(imgs, labels, np.arange(24), 8), make_loader(imgs, labels, np.arange(24, 40), 8)
    r = M.flip_consistency_check(ad, model, bank, query, 4, "cpu", torch.float32, k=5)
    assert r["flip_consistency"] == 100.0 and r["pass"] and r["n_query"] == 16 and r["n_bank"] == 24
    assert r["max_abs_feature_diff"] < 1e-4 and r["precision"] == "float32"
    r0 = M.flip_consistency_check(ad, ad.model, bank, query, 4, "cpu", torch.float32, k=5)
    assert r0["max_abs_feature_diff"] > 1e-3


# ---------------------------------------------------------------- command line
def test_cli_tiny_fp32_then_float64_reuses_the_structure(tmp_path):
    out = tmp_path / "zs"
    res = M.main(["--tiny", "--out", str(out)])
    m = json.loads((out / "metrics.json").read_text())
    df = pd.read_csv(out / M.CSV_NAME)
    assert m["prediction_holds"] is True and m["flip_check"]["flip_consistency"] == 100.0
    assert m["summary"]["max_rel_tensor"] < 1e-5 and m["structure"]["source"]["fitted_here"]
    assert m["flip_check"]["n_query"] == 20 and m["flip_check"]["max_abs_feature_diff"] < 1e-4
    assert m["flip_check_bf16"] is None                     # bf16 only on CUDA
    assert len(m["val_indices"]) == 20 and np.load(out / "val_indices.npy").shape == (20,)
    assert len(df) == len(res["rows"]) == 2 + 4 * len(BLOCK_COMPONENTS) + 4
    assert set(df["component"]) >= set(BLOCK_COMPONENTS) and df["pass"].all()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["experiment"] == "MUST-02" and meta["git"]["git_hash"] and meta["seeds"]["seed"] == 0
    # float64 with the structure fitted above: the errors reach rounding level.
    s = out / "structure"
    out64 = tmp_path / "zs64"
    M.main(["--tiny", "--dtype", "float64", "--operators", str(s / "tiny_operators.npz"),
            "--m1-structure", str(s / "m1_structure.npz"), "--out", str(out64)])
    m64 = json.loads((out64 / "metrics.json").read_text())
    assert m64["structure"]["source"]["fitted_here"] is False
    assert m64["summary"]["max_rel_tensor"] < 1e-12 and m64["summary"]["max_rel_token"] < 1e-12
    assert m64["flip_check"]["max_abs_feature_diff"] < 1e-12 and m64["prediction_holds"] is True


def test_cli_span_structure_file(tmp_path):
    """--span-structure: A*, pairings, head maps and register pairing from a mirror.mirror_family SpanStructure
    file (the MUST-07 format), with its path and sha256 recorded; it cannot be combined with --m1-structure."""
    from mirror.mirror_family import SpanStructure, save_span_structure

    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    A, perms, maps, _ = random_structure(ad, seed=6)
    L = ad.depth
    path = save_span_structure(tmp_path / "structure" / "tiny_s0.npz", SpanStructure(
        model="tiny", fit_seed=0, A_layers=[A] * (L + 1), perms=perms, heads=maps,
        A_star={k: A for k in range(1, L + 1)}, reg_perm={k: np.zeros(0, dtype=np.int64) for k in range(1, L + 1)}))
    out = tmp_path / "ss"
    M.main(["--tiny", "--span-structure", str(path), "--out", str(out), "--no-flip-check"])
    m = json.loads((out / "metrics.json").read_text())
    src = m["structure"]["source"]
    assert src["span_structure"] == str(path) and src["span_structure_sha256"] == M._t2().sha256(path)
    assert src["fitted_here"] is False and not (out / "structure").exists()        # nothing fitted here
    assert m["structure"]["d_plus"] == int(round((np.trace(A) + A.shape[0]) / 2))
    assert m["structure"]["n_neuron_pairs"] == [int((p != np.arange(len(p))).sum() // 2) for p in perms]
    assert m["structure"]["n_head_pairs"] == [hm.n_pairs for hm in maps]
    assert m["components_below_tol"] and m["summary"]["max_rel_tensor"] < 1e-5
    meta = json.loads((out / "meta.json").read_text())
    assert meta["models"]["tiny"]["structure"]["span_structure_sha256"] == src["span_structure_sha256"]
    with pytest.raises(SystemExit):
        M.parse_args(["--span-structure", str(path), "--m1-structure", "x.npz"])


def test_cli_healed_vexact_checkpoint(tmp_path):
    hv = load_script("heal_vexact")
    root = tmp_path / "m3_vexact"
    hv.main(["--config", CONFIG, "--tiny", "--k", "4", "--lr", "1e-3", "--max-steps", "2", "--out-dir", str(root)])
    out = tmp_path / "healed"
    M.main(["--tiny", "--vexact-run", str(root / "k4"), "--out", str(out), "--no-flip-check"])
    m = json.loads((out / "metrics.json").read_text())
    src = m["structure"]["source"]
    assert src["kind"].startswith("healed V-exact") and src["checkpoint_check"]["all_equal"]
    assert m["components_below_tol"] and m["summary"]["max_rel_tensor"] < 1e-5
    assert m["prediction_holds"] is None and m["flip_check"] is None      # not decided without the flip check
    assert M.variant_name(M.parse_args(["--vexact-run", str(root / "k4")])) == "vexact_m3_vexact_k4"


def test_cli_healed_run_through_load_healed_model(tmp_path, monkeypatch):
    """--healed-run imports mirror.heal_structured lazily; here a stand-in returns an exact tiny model with no
    structure in its info, so A* comes from the even head and the rest from the weights."""
    calls, layouts = [], [{"method": "mirror_exact", "seed": 0}]

    def load_healed_model(run_dir, device):
        calls.append(Path(run_dir))
        ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
        model, st = exact_model(ad, seed=5)
        info = dict(layouts[-1])
        if info.get("family") == "mirror":         # the layout mirror.heal_structured returns for the mirror family
            info["structure"] = {"A_star": st.A, "perms": st.perms, "maps_after": st.maps, "reg_perm": st.reg_perm,
                                 "blocks": list(range(st.k))}
            info["constraints"] = [object()]
        return ad, model, info

    fake = types.ModuleType("mirror.heal_structured")
    fake.load_healed_model = load_healed_model
    monkeypatch.setitem(sys.modules, "mirror.heal_structured", fake)
    run = tmp_path / "runs" / "vexact_s0"
    run.mkdir(parents=True)
    out = tmp_path / "healed_run"
    M.main(["--tiny", "--healed-run", str(run), "--out", str(out)])
    m = json.loads((out / "metrics.json").read_text())
    assert calls == [run]
    src = m["structure"]["source"]
    assert src["A_star"].startswith("even head") and src["perms"] == "weights" and src["inference_residuals"]["max"] < 1e-4
    assert m["prediction_holds"] is True and m["summary"]["max_rel_tensor"] < 1e-5
    assert M.variant_name(M.parse_args(["--healed-run", str(run), "--dtype", "float64"])) == "healed_vexact_s0_fp64"
    # The mirror-family layout: A* and the span from info["structure"]; the pairings read from the weights agree.
    layouts.append({"method": "mirror_exact", "family": "mirror", "seed": 0})
    out2 = tmp_path / "healed_run2"
    M.main(["--tiny", "--healed-run", str(run), "--out", str(out2), "--no-flip-check"])
    src = json.loads((out2 / "metrics.json").read_text())["structure"]["source"]
    assert src["A_star"] == "info" and src["k"] == 4 and src["family"] == "mirror"
    assert src["agrees_with_info"] == {"reg_perm": True, "perms": True, "head_sigma": True}
