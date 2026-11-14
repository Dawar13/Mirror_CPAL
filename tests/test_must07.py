"""MUST-07 runner and the Mirror family (scripts/must07_compressors.py, mirror/mirror_family.py): CPU tests with the
tiny random ViT and synthetic uint8 images. Float64 where exactness is tested."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror import mirror_family as MF
from mirror.adapters import TimmTinyAdapter
from mirror.attention import attention_breaking_shares, fold_attn
from mirror.data import make_loader, synthetic_dataset
from mirror.fold import fold_copy
from mirror.heal_vexact import flip_check, span_exactness, vexact_param_counts
from mirror.knn import evaluate_knn
from mirror.surgery import fold_mlp

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_m07_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses in the script look their module up
    spec.loader.exec_module(mod)
    return mod


M = load_script("must07_compressors")


# ---------------------------------------------------------------- fixtures (module scope: one structure fit)
@pytest.fixture(scope="module")
def ad64():
    return TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)


@pytest.fixture(scope="module")
def data64():
    images, labels = synthetic_dataset(4, 12, 56, seed=3)
    idx = np.arange(len(labels))
    return {"fit": make_loader(images, labels, idx[:16], 8), "bank": make_loader(images, labels, idx[16:40], 16),
            "query": make_loader(images, labels, idx[40:], 16), "num_classes": 4}


@pytest.fixture(scope="module")
def struct64(ad64, data64):
    return MF.fit_span_structure(ad64, data64["fit"], "cpu", model_name="tiny64", top_k=3, workers=1)


def probe(n=3, seed=5):
    return torch.randn(n, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


# ---------------------------------------------------------------- structure
def test_structure_roundtrip_and_pairing_counts(struct64, tmp_path):
    L = struct64.depth
    for b in range(L):
        n = struct64.n_pairs(b)
        for perms in (struct64.actcorr_perms, struct64.wcos_perms):
            p = perms[b]
            assert np.array_equal(p[p], np.arange(len(p)))                        # an involution
            assert int((p != np.arange(len(p))).sum() // 2) == n                  # the mirror pairing's pair count
    assert set(struct64.A_star) == set(range(1, L + 1))
    path = MF.save_span_structure(tmp_path / "s.npz", struct64)
    back = MF.load_span_structure(path)
    assert back.model == "tiny64" and back.fit_seed == 0 and back.meta["n_pairs"] == struct64.meta["n_pairs"]
    for k in range(1, L + 1):
        assert np.array_equal(back.A_star[k], struct64.A_star[k])
    assert all(np.array_equal(x, y) for x, y in zip(back.actcorr_perms, struct64.actcorr_perms))
    assert all(np.array_equal(x.B, y.B) and np.array_equal(x.sigma, y.sigma) for x, y in zip(back.heads, struct64.heads))


def test_top_pairs_cuts_and_fills():
    S = np.array([[0, .9, .1, .2], [.9, 0, .3, .4], [.1, .3, 0, .8], [.2, .4, .8, 0]])
    assert MF.top_pairs([(0, 1), (2, 3)], S, 1) == [(0, 1)]
    assert MF.top_pairs([(2, 3)], S, 2) == [(0, 1), (2, 3)]


def test_parse_size():
    assert M.parse_size("6mlp", 12) == M.Size("6mlp", 6, ("mlp",))
    assert M.parse_size("9", 12) == M.Size("9both", 9, ("mlp", "attn"))
    assert M.parse_size("full", 12) == M.Size("full", 12, ("mlp", "attn"))
    assert M.parse_size("fullattn", 24) == M.Size("fullattn", 24, ("attn",))
    assert M.parse_size("12both", 12).name == "full"
    with pytest.raises(ValueError):
        M.parse_size("13mlp", 12)


# ---------------------------------------------------------------- budgets and realised structure
def _check_realised(model, info, method, k, comps, struct):
    """The converted weights lie exactly in the subspace whose dimension is the counted parameter number."""
    s = info["structure"]
    for b in range(k):
        if "mlp" in comps:
            f = fold_mlp(model.blocks[b])
            perm = torch.as_tensor(np.asarray(s["perms"][b]))
            if method == "mirror_local":
                A_in, A_out = (torch.as_tensor(struct.A_layers[b]), torch.as_tensor(struct.A_layers[b + 1]))
            else:
                A_in = A_out = torch.as_tensor(s["A"])
            assert rel(f["W1"][perm] @ A_in, f["W1"]) < 1e-10
            assert rel((A_out @ f["W2"])[:, perm], f["W2"]) < 1e-10
        if "attn" in comps:
            A_in = torch.as_tensor(struct.A_layers[b] if method == "mirror_local" else s["A"])
            sh = attention_breaking_shares(model.blocks[b], A_in, A_in, s["heads_after"][b])
            assert sh["s_all"] < 1e-20


@pytest.mark.parametrize("size", ["2mlp", "2attn", "2both", "full"])
def test_mirror_family_counts_meet_the_budget(ad64, struct64, size):
    L, d = ad64.depth, ad64.embed_dim
    s = M.parse_size(size, L)
    budget = MF.span_budget(ad64.model, struct64, range(s.k), s.components)
    bf = MF.budget_fields(budget)
    assert bf["budget_basis_change"] == (0 if s.k == L else d * d)
    for method in MF.MIRROR_METHODS:
        model, info = MF.build_mirror_variant(ad64, method, range(s.k), s.components, struct64, seed=1, budget=budget)
        assert info["params_total"] == info["params_matrices"] + info["params_basis_change"]
        assert info["span_dense"] == bf["span_dense"] and info["budget_total"] == bf["budget_total"]
        if method == "mirror_local":                                   # ablation: one d x d rotation per block
            assert info["params_basis_change"] == s.k * d * d
        else:                                                          # matched exactly: gap 0
            assert info["gap"] == 0 and info["params_total"] == bf["budget_total"]
            assert info["params_total_absorbed"] == info["params_matrices"]
        _check_realised(model, info, method, s.k, s.components, struct64)
    # Mirror-exact, both components, equals the V-exact accounting used by the healing runs.
    if s.components == MF.COMPONENTS:
        parts = struct64.exact_parts("mirror_exact", s.k)
        perms = parts["perms"] + list(struct64.perms[s.k:])
        heads = parts["heads"] + list(struct64.heads[s.k:])
        ref = vexact_param_counts(ad64, parts["A"], s.k, perms, heads)
        assert bf["budget_matrices"] == ref["converted_params_kept"]


def test_component_counts_add_up(ad64, struct64):
    for k in (2, ad64.depth):
        tot = {c: MF.build_mirror_variant(ad64, "mirror_exact", range(k), c, struct64)[1]["params_matrices"]
               for c in (("mlp",), ("attn",), MF.COMPONENTS)}
        assert tot[("mlp",)] + tot[("attn",)] == tot[MF.COMPONENTS]


# ---------------------------------------------------------------- exactness (float64)
@pytest.mark.parametrize("method", ["mirror_exact", "random_structure", "pair_actcorr"])
def test_full_model_is_exactly_flip_invariant_in_float64(ad64, struct64, data64, method):
    L = ad64.depth
    model, info = MF.build_mirror_variant(ad64, method, range(L), MF.COMPONENTS, struct64, seed=1)
    s = info["structure"]
    xs = [probe(3, 5), probe(2, 6)]
    ex = span_exactness(ad64, model, xs, s["A"], L, s["reg_perm"])
    assert ex["pass"] and ex["max_rel_inside"] < 1e-10
    fc = flip_check(ad64, model, data64["bank"], data64["query"], xs, data64["num_classes"], "cpu", k=5)
    assert fc["flip_consistency"] == 100.0 and fc["max_abs_unit_feature_diff"] < 1e-10 and fc["cls_feature_rel_max"] < 1e-10
    r = evaluate_knn(ad64, data64["bank"], data64["query"], "cpu", data64["num_classes"], k=5, model=model)
    assert torch.equal(r["pred"], r["pred_flip"])                    # identical predictions for x and mirror(x)
    # Teeth: the original model and Mirror-local are not invariant.
    for other in (ad64.model, MF.build_mirror_variant(ad64, "mirror_local", range(L), MF.COMPONENTS, struct64)[0]):
        assert flip_check(ad64, other, data64["bank"], data64["query"], xs, 4, "cpu", k=5)["max_abs_unit_feature_diff"] > 1e-6


def test_component_switches_leave_the_other_component_untouched(ad64, struct64):
    ref, _ = fold_copy(ad64.model, final_norm_affine=False)
    k, L = 2, ad64.depth
    m = {c: MF.build_mirror_variant(ad64, "mirror_exact", range(k), c, struct64)[0] for c in ("mlp", "attn", "both")}

    def mlp(model, b):
        return [model.blocks[b].mlp.fc1.weight, model.blocks[b].mlp.fc1.bias, model.blocks[b].mlp.fc2.weight,
                model.blocks[b].mlp.fc2.bias]

    def attn(model, b):
        a = model.blocks[b].attn
        return [a.qkv.weight, a.qkv.bias, a.proj.weight, a.proj.bias]

    def same(xs, ys):
        return all(torch.equal(x, y) for x, y in zip(xs, ys))

    for b in range(k):
        assert same(attn(m["mlp"], b), attn(ref, b)) and not same(mlp(m["mlp"], b), mlp(ref, b))
        assert same(mlp(m["attn"], b), mlp(ref, b)) and not same(attn(m["attn"], b), attn(ref, b))
        assert same(mlp(m["both"], b), mlp(m["mlp"], b)) and same(attn(m["both"], b), attn(m["attn"], b))
    for b in range(k, L):                                             # outside the span: the folded original
        for c in m.values():
            assert same(mlp(c, b), mlp(ref, b)) and same(attn(c, b), attn(ref, b))
    pe = [c.patch_embed.proj.weight for c in m.values()]               # embeddings projected the same way
    assert all(torch.equal(pe[0], p) for p in pe) and not torch.equal(pe[0], ref.patch_embed.proj.weight)


# ---------------------------------------------------------------- compressors (SHIM or the real methods)
def _rank(W):
    return int(torch.linalg.matrix_rank(W.double(), rtol=1e-5))     # weights are stored in float32


def _realised_shim(model, method, blocks, comps):
    """Weights as stored: rank x (rows + cols) for low-rank matrices, sliced dimensions times h for slicing, entries
    for pruning; attention by rank (plain SVD in the shim)."""
    total = 0
    for b in blocks:
        if "mlp" in comps:
            f = fold_mlp(model.blocks[b])
            W1, W2 = f["W1"], f["W2"]
            if method == "flap":
                total += W1.numel() + W2.numel()
            elif method == "slicegpt":
                total += W1.shape[0] * _rank(W1) + W2.shape[1] * _rank(W2)
            else:
                total += sum(_rank(W) * (W.shape[0] + W.shape[1]) for W in (W1, W2))
        if "attn" in comps:
            f = fold_attn(model.blocks[b])
            total += sum(_rank(f[n]) * (f[n].shape[0] + f[n].shape[1]) for n in ("Wq", "Wk", "Wv", "Wo"))
    return int(total)


@pytest.fixture(scope="module")
def comp_env():
    """The tiny float32 runtime, both compressor APIs (the SHIM, and the real one when it exists) and their
    statistics on fit set 0, collected once."""
    from mirror.variants import build_runtime

    ad, proto, _ = build_runtime("tiny", True, torch.device("cpu"), ROOT / "cache")
    apis = {a.kind: a for a in (M.compressor_api(force_shim=True), M.compressor_api())}
    stats = {k: a.collect_stats(ad, proto.fit_loader(0), torch.device("cpu"), list(range(ad.depth)), list(a.methods),
                                MF.COMPONENTS) for k, a in apis.items()}
    return ad, apis, stats


@pytest.mark.parametrize("size", ["2mlp", "2both", "full"])
def test_compressors_meet_the_budget_rule(struct64, comp_env, size):
    ad, apis, all_stats = comp_env
    L = ad.depth
    s = M.parse_size(size, L)
    blocks = list(range(s.k))
    budget = MF.span_budget(ad.model, struct64, blocks, s.components)
    bf = MF.budget_fields(budget)
    before = {n: v.clone() for n, v in ad.model.state_dict().items()}
    for kind, api in apis.items():
        stats = all_stats[kind]
        for method in api.methods:
            for hp in api.grids[method]:
                model, info = api.compress(ad.model, method, blocks, s.components, budget, stats, hp)
                gap = int(info["params_total"]) - bf["budget_total"]
                assert gap >= 0, (api.kind, method, hp, gap)               # rounds toward more parameters
                if api.kind == "shim":
                    assert gap < info["granularity"]                        # by less than one allocation step each
                    assert _realised_shim(model, method, blocks, s.components) == info["params_matrices"]
                else:
                    assert int(info["budget_total"]) == bf["budget_total"]
    assert all(torch.equal(v, ad.model.state_dict()[n]) for n, v in before.items())   # original untouched


# ---------------------------------------------------------------- gate B
def _gate_rows():
    rows = []
    msel = {"svd": 50.0, "svdllm": 61.0, "asvd_scale": 61.0, "slicegpt": 40.0, "flap": 55.0, "zipit": 30.0,
            "pair_actcorr": 20.0, "pair_wcos": 25.0, "pair_random": 10.0, "random_structure": 5.0,
            "mirror_exact": 99.0, "mirror_local": 98.0}
    for m, v in msel.items():
        rows.append({"model": "m", "size": "full", "method": m, "fit_seed": 0, "status": "ok", "msel_top1": v,
                     "val_top1": 100.0 - v, "hparams": "{}", "params_total": 100, "budget_total": 100, "gap_rel": 0.0,
                     "n_msel": 10})
        rows.append({**rows[-1], "size": "6both", "msel_top1": 100.0 - v})            # other size: ignored
        rows.append({**rows[-2], "fit_seed": 1, "msel_top1": 100.0 - v})              # other fit seed: ignored
        rows.append({**rows[-3], "model": "other", "msel_top1": 100.0 - v})           # other model: ignored
    rows.append({**rows[0], "method": "flap", "status": "budget_miss", "msel_top1": 99.5})   # not eligible
    return pd.DataFrame(rows)


def test_gate_b_uses_the_method_selection_set_only():
    df = _gate_rows()
    g = M.gate_b(df, "m")
    assert [r["method"] for r in g["ranking"]] == ["svdllm", "asvd_scale", "flap", "svd", "slicegpt", "zipit",
                                                   "pair_wcos", "pair_actcorr", "pair_random", "random_structure"]
    assert g["best"] == "svdllm" and g["second_best"] == "asvd_scale"           # tie: the lower method number first
    assert g["ties"] == [["svdllm", "asvd_scale"]] and g["complete"] and not g["missing_methods"]
    assert "mirror_exact" not in {r["method"] for r in g["ranking"]}
    # Validation is never read: the same result without the column, and a reversed validation order changes nothing.
    assert M.gate_b(df.drop(columns=["val_top1"]), "m") == g
    assert all("val" not in key for r in g["ranking"] for key in r)
    g2 = M.gate_b(df[df["method"] != "zipit"], "m")
    assert not g2["complete"] and g2["missing_methods"] == ["zipit"]


# ---------------------------------------------------------------- CLI, tiny, end to end
@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("must07")
    res = M.main(["--tiny", "--out", str(out)])
    return out, res


def test_cli_rows_budgets_and_flip(cli_run):
    out, res = cli_run
    df = pd.read_csv(out / "rows_tiny.csv")
    api = M.compressor_api()
    methods = list(MF.MIRROR_METHODS) + list(api.methods)
    sizes = ["2mlp", "2both", "3both", "full"]
    ok = df[df["status"] == "ok"]
    assert {(s, m) for s in sizes for m in methods} <= set(zip(ok["size"], ok["method"]))
    assert ((df["method"] == "original") & (df["fit_seed"] == -1)).sum() == 1
    star = ok[ok["method"].isin(MF.EXACT_FAMILY)]
    assert (star["gap"] == 0).all() and (star["params_total"] == star["budget_total"]).all()
    comp = ok[ok["method"].isin(api.methods)]
    assert (comp["gap"] >= 0).all()
    full = ok[ok["size"] == "full"]
    assert full["flip_consistency_fp32"].notna().all()                         # every full-model row, fp32
    assert ok[(ok["size"] != "full") & (ok["method"] != "original")]["flip_consistency_fp32"].isna().all()
    exact = full[full["claims_exact_invariance"].astype(bool)]
    assert set(exact["method"]) == set(MF.EXACT_FAMILY)
    assert (exact["flip_consistency_fp32"] == 100.0).all()
    # Costs: fewer parameters and FLOPs than the original for every matched method.
    orig = df[df["method"] == "original"].iloc[0]
    assert (star["model_params"] < orig["model_params"]).all() and (star["gflops"] < orig["gflops"]).all()
    # Per-image predictions, info files, bootstrap, gate B, summary and meta.
    for _, r in ok.iterrows():
        with np.load(M.preds_path(out, "tiny", r["size"], r["method"], int(r["fit_seed"]))) as z:
            assert len(z["val_pred"]) == r["n_val"] and len(z["msel_pred"]) == r["n_msel"]
    boot = pd.read_csv(out / "bootstrap.csv")
    want = {(s, m) for s in sizes for m in methods if m != "mirror_exact"}
    assert want <= set(zip(boot["size"], boot["method"])) and (boot["n_resamples"] == 10_000).all()
    gate = json.loads((out / "gate_b.json").read_text())
    cand = ok[(ok["size"] == "full") & ok["method"].isin(M.BASELINES) & (ok["fit_seed"] == 0)]
    top = cand.sort_values(["msel_top1", "number"], ascending=[False, True]).iloc[0]
    assert gate["best"] == top["method"] and "validation is not read" in gate["selection_data"]
    grid = pd.read_csv(out / "grid_tiny.csv")
    assert len(grid) == sum(len(api.grids[m]) for m in api.methods) * len(sizes)
    text = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in text and "P7.1" in text and "Gate B" in text
    meta = json.loads((out / "meta.json").read_text())
    assert meta["experiment"] == "MUST-07" and meta["seeds"]["fit_seeds"] == [0] and meta["git"]["git_hash"]
    assert res["rows_added"] == len(df)


def test_cli_resume_skips_finished_rows(cli_run):
    out, _ = cli_run
    before = pd.read_csv(out / "rows_tiny.csv")
    grid_before = pd.read_csv(out / "grid_tiny.csv")
    res = M.main(["--tiny", "--out", str(out)])
    assert res["rows_added"] == 0
    pd.testing.assert_frame_equal(pd.read_csv(out / "rows_tiny.csv"), before)
    # Drop one compressor row (its grid points stay): only that row is redone, from the logged grid choice.
    api = M.compressor_api()
    m = api.methods[0]
    keep = before[~((before["size"] == "2both") & (before["method"] == m))]
    keep.to_csv(out / "rows_tiny.csv", index=False)
    res = M.main(["--tiny", "--out", str(out)])
    assert res["rows_added"] == 1
    after = pd.read_csv(out / "rows_tiny.csv")
    old = before[(before["size"] == "2both") & (before["method"] == m)].iloc[0]
    new = after[(after["size"] == "2both") & (after["method"] == m)].iloc[0]
    assert new["val_top1"] == old["val_top1"] and new["hparams"] == old["hparams"]
    assert len(pd.read_csv(out / "grid_tiny.csv")) == len(grid_before)
