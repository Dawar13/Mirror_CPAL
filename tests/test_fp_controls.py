"""A03 discovery null controls and the A15b measurement (mirror/controls.py, scripts/fp_a03_controls.py,
scripts/fp_a15b_measure.py). CPU only, tiny random models, synthetic images, no downloads.

Synthetic equivariant network (one module fixture: the --dry-run of fp_a03_controls.py). A tiny random timm ViT
(embed 32, 3 blocks, 2 heads, 2 registers, 56 px, float64) gets its mirror structure fitted on 32 random images
(mirror_family.fit_span_structure) and is projected with build_mirror_variant("mirror_exact", every block, MLP and
attention), then unfolded back to LayerNorm. Every stream layer then satisfies aligned(mirror x) = x A* (checked to
1e-10) and every MLP has P W1 = W1 A*, P b1 = b1. The controls are measured on 96 other random images.

What is shown, and why the tolerances are what they are:
  * reference (real mirror): finds the planted structure exactly: operators equal to A*, pairings equal to the
    planted ones, twin fraction equal to the planted share (about 0.71), ratio to null 0, d- of A* > 0, every
    layer's spectrum summing to 1 in absolute value. Exact by construction, so the tolerances are 1e-8 or tighter.
  * C1, C2 and C3 find no twins: twin fraction <= 0.05, the A03-1 bar (measured 0.005 to 0.016). With 96 images of
    19 tokens a noise correlation has a standard deviation of about 1 / sqrt(1824) = 0.02, so the best of the 127
    candidates of a neuron stays near 0.1, far below the 0.5 threshold.
  * Their split-half ratio to null (operators fitted on one half of the images, pairings on the other) is within
    0.1 of 1 for C1 and C2 (measured 0.99 to 1.00) and at least 0.8 for C3 (measured 0.91: a fixed pixel
    permutation leaves a fixed linear relation between the patch embeddings of x and T(x), present in both halves).
  * Their same-sample ratio (the MUST-04 definition that rule A03-1 scores) is 0.87 to 0.89: below 1 without any
    mirror relation, because the matching and the operator are fitted to the same cross-moments. The test pins the
    mechanism (fitted operators with null pairings, and null operators with the fitted pairing, sit at the null
    within 0.05; both fitted together fall below 0.95) and checks the ratio stays in [0.75, 1.05], far above the
    reference's 0. Five null draws give the null mean to about 1% here (the spread of a draw is under 1%).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror import controls as CT
from mirror import groups as GR

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_fp_controls_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A03 = load_script("fp_a03_controls")
A15B = load_script("fp_a15b_measure")
RANGE = "1-6"


@pytest.fixture(scope="module", autouse=True)
def _allow_no_prereg():
    mp = pytest.MonkeyPatch()
    mp.setenv("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
    yield
    mp.undo()


def _images(n_per_class, seed, classes=4):
    from mirror.data import synthetic_dataset

    return synthetic_dataset(classes, n_per_class, 56, seed=seed)[0]


def _dp(A):
    A = np.asarray(A)
    return int(round((A.shape[0] + float(np.trace(A))) / 2))


# ---------------------------------------------------------------------------------------------- input actions
@pytest.mark.parametrize("n", [2, 3, 7, 2000])
def test_c1_derangement(n):
    p = CT.derangement(n, seed=0)
    assert sorted(p.tolist()) == list(range(n))                 # a permutation of the fit images
    assert not np.any(p == np.arange(n))                         # no image is paired with itself
    assert np.array_equal(p, CT.derangement(n, seed=0))          # fixed by its seed
    if n > 3:
        assert not np.array_equal(p, CT.derangement(n, seed=1))
    c1 = CT.make_control("C1", seed=0)
    assert np.array_equal(c1.partner(n), p)
    assert np.array_equal(CT.make_control("mirror").partner(n), np.arange(n))


def test_derangement_needs_two_images():
    with pytest.raises(ValueError):
        CT.derangement(1, 0)


@pytest.mark.parametrize("p", [14, 7, 4])
def test_c3_patch_involution(p):
    n = p * p
    s = CT.patch_involution(p, seed=3)
    flip = CT.flip_patch_index(p)
    assert sorted(s.tolist()) == list(range(n))
    assert np.array_equal(s[s], np.arange(n))                    # an involution
    assert np.array_equal(flip[flip], np.arange(n))
    fixed = int((s == np.arange(n)).sum())
    assert fixed == int((flip == np.arange(n)).sum()) == (p % 2) * p     # as many fixed pixels as the flip
    assert not np.array_equal(s, flip) and fixed < n                     # neither the flip nor the identity
    assert not np.array_equal(s, CT.patch_involution(p, seed=4))


def test_c3_transform_keeps_patches_and_the_token_grid():
    from timm.models.vision_transformer import VisionTransformer

    c3 = CT.make_control("C3", seed=0)
    p, S = 14, 56
    idx, sigma = c3.pixel_index(S, p), torch.as_tensor(c3.pixel_sigma(p))
    x = torch.randn(2, 3, S, S, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    t = c3.apply(x, idx)
    assert torch.equal(c3.apply(t, idx), x) and not torch.equal(t, x)          # T is an involution, not identity
    for R in range(S // p):                                                     # same sigma in every patch, channel
        for C in range(S // p):
            src = x[..., R * p:(R + 1) * p, C * p:(C + 1) * p].reshape(2, 3, p * p)
            dst = t[..., R * p:(R + 1) * p, C * p:(C + 1) * p].reshape(2, 3, p * p)
            assert torch.equal(dst, src[..., sigma])
    torch.manual_seed(0)
    m = VisionTransformer(img_size=S, patch_size=p, embed_dim=8, depth=1, num_heads=2).double()
    G = S // p
    assert np.array_equal(c3.token_index((G, G)), np.arange(G * G))
    for r, c in ((0, 0), (1, 3), (3, 2)):                                       # a marker patch stays on its token
        z = torch.zeros(1, 3, S, S, dtype=torch.float64)
        z[..., r * p:(r + 1) * p, c * p:(c + 1) * p] = torch.rand(3, p, p, dtype=torch.float64)
        with torch.no_grad():
            tok = m.patch_embed(c3.apply(z, idx)) - m.patch_embed(torch.zeros_like(z))
        assert int(tok.norm(dim=-1)[0].argmax()) == r * G + c


def test_each_pairing_feeds_the_intended_tokens():
    """Layer-0 cross-moment of every row against the explicit computation: the reference and C1 un-flip the grid of
    mirror(x') (C1 with x' the deranged partner), C2 meets token (r, c) of x with token (r, c) of mirror(x), C3 meets
    x with T(x) at the same positions; prefix tokens always keep their index."""
    from mirror.data import normalize_for
    from mirror.stats import center_channels, unflip_tokens

    ad = CT.tiny_adapter(depth=1, reg_tokens=1)
    imgs = _images(2, seed=5, classes=3)[:5]
    d = ad.embed_dim
    x = normalize_for(ad, torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous())
    G = ad.grid_hw
    assert np.array_equal(CT.make_control("C2").token_index(G), np.arange(G[0] * G[1]))
    assert np.array_equal(CT.make_control("C1").token_index(G), GR.token_index("LR", G))
    for name in CT.CONTROL_ROWS:
        c = CT.make_control(name, seed=2)
        got = CT.collect_control_pass(ad, imgs, "cpu", c, batch_size=2, mlp=False, layers=[0])["residual"]["C"][0]
        xp = x[torch.as_tensor(c.partner(len(imgs)))]
        with torch.no_grad():
            X, Xt = ad.layer0_tokens(x), ad.layer0_tokens(c.apply(xp, c.pixel_index(56, 14)))
        if c.align == "unflip":
            Xt = unflip_tokens(Xt, G, ad.n_prefix_tokens)
        X, Xt = center_channels(X.reshape(-1, d)), center_channels(Xt.reshape(-1, d))
        assert np.allclose(got, (X.T @ Xt).numpy(), rtol=1e-10, atol=1e-10), name
    assert not np.array_equal(CT.make_control("C1", 2).partner(5), np.arange(5))


def test_null_seeds_are_those_of_measure_groups():
    mg = A15B.load_measure_groups()
    for base, draw, slot in ((0, 0, 0), (0, 19, 12), (3, 7, 3005)):
        assert CT.null_seed(base, draw, slot) == mg.seed_for(base, "LR", draw, slot)


def test_score_a03_rule():
    ok = {"twin_fraction_t50": 0.05, "ratio_to_null": 0.9}                     # the bars themselves pass
    s = CT.score_a03({"C1": ok, "C2": ok, "C3": ok})
    assert s["status"] == "PASS" and all(v["status"] == "PASS" for v in s["controls"].values())
    s = CT.score_a03({"C1": ok, "C2": {"twin_fraction_t50": 0.051, "ratio_to_null": 1.0}, "C3": ok})
    assert s["status"] == "FAIL" and s["controls"]["C2"]["twin_ok"] is False
    s = CT.score_a03({"C1": ok, "C2": ok, "C3": {"twin_fraction_t50": 0.0, "ratio_to_null": 0.899}})
    assert s["status"] == "FAIL" and s["controls"]["C3"]["ratio_ok"] is False
    assert CT.score_a03({"C1": ok, "C2": ok})["status"] == "PENDING"
    s = CT.score_a03({"C1": ok, "C2": ok, "C3": {"twin_fraction_t50": float("nan"), "ratio_to_null": 1.0}})
    assert s["controls"]["C3"]["status"] == "PENDING" and s["status"] == "PENDING"
    assert (CT.A03_TWIN_MAX, CT.A03_RATIO_MIN) == (0.05, 0.9)


def test_score_a15b_rule():
    assert A15B.A15B_MIN_DROP == 0.1
    assert A15B.score_a15b(0.85, 0.70)["status"] == "PASS"
    s = A15B.score_a15b(0.85, 0.80)
    assert s["status"] == "FAIL" and s["drop"] == pytest.approx(0.05)
    assert A15B.score_a15b(0.85, 0.95)["status"] == "FAIL"                     # a rise fails
    assert A15B.score_a15b(None, 0.7)["status"] == "PENDING"
    assert A15B.score_a15b(0.85, float("nan"))["status"] == "PENDING"


def test_unwrap_state_dict():
    t = torch.zeros(2)
    sd = A15B.unwrap_state_dict({"model": {"_orig_mod.blocks.0.w": t, "_orig_mod.norm.weight": t}, "step": 780})
    assert set(sd) == {"blocks.0.w", "norm.weight"}
    assert set(A15B.unwrap_state_dict({"module.a": t})) == {"a"}
    with pytest.raises(ValueError):
        A15B.unwrap_state_dict({"step": 3})


def test_reference_equals_fit_span_structure():
    """The reference row is the existing pipeline: on the same images and batches its structure equals
    mirror_family.fit_span_structure's bit for bit (operators, span operators, pairings, head maps, registers)."""
    from mirror import mirror_family as MF
    from mirror.data import make_loader

    ad = CT.tiny_adapter()
    imgs = _images(6, seed=7)
    ref = MF.fit_span_structure(ad, make_loader(imgs, np.zeros(len(imgs), dtype=np.int64), np.arange(len(imgs)), 16),
                                "cpu", pairings=False)
    st = CT.fit_control(ad, imgs, "cpu", CT.make_control("mirror"), batch_size=16, n_null=0).structure
    L = ad.depth
    assert all(np.array_equal(a, b) for a, b in zip(ref.A_layers, st.A_layers))
    assert all(np.array_equal(ref.A_star[k], st.A_star[k]) for k in range(1, L + 1))
    assert all(np.array_equal(a, b) for a, b in zip(ref.perms, st.perms))
    for a, b in zip(ref.heads, st.heads):
        assert np.array_equal(a.sigma, b.sigma) and np.array_equal(a.B, b.B) and np.array_equal(a.C, b.C)
    assert all(np.array_equal(ref.reg_perm[k], st.reg_perm[k]) for k in range(1, L + 1))


# ---------------------------------------------------------------------------------------------- synthetic network
@pytest.fixture(scope="module")
def a03(tmp_path_factory):
    out = tmp_path_factory.mktemp("a03")
    argv = ["--dry-run", "--out", str(out)]
    rt = A03.build_dry_runtime(A03.parse_args(argv))
    metrics = A03.main(argv, runtime=rt)
    table = pd.read_csv(out / "controls.csv").set_index("control")
    return {"out": out, "rt": rt, "metrics": metrics, "argv": argv, "table": table}


def _row_structure(a03, row):
    from mirror.mirror_family import load_span_structure

    out = a03["out"]
    st = load_span_structure(out / "structures" / f"tiny_equivariant_{row}_fit0.npz")
    with np.load(out / "cache" / f"{row}_fit.npz") as z:
        arrs = {k: z[k] for k in z.files}
    return st, arrs


def test_synthetic_network_is_exactly_equivariant(a03):
    from mirror.m1 import stream_equivariance
    from mirror.surgery import fold_mlp

    ad, planted = a03["rt"].adapter, a03["rt"].extra["planted_structure"]
    L = ad.depth
    A = np.asarray(planted.A_star[L])
    x = torch.randn(3, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    rows = stream_equivariance(ad, ad.model, x, A)
    assert len(rows) == L + 1 and max(r["rel_max"] for r in rows) < 1e-10        # every stream layer
    At = torch.as_tensor(A)
    for b in range(L):
        f, p = fold_mlp(ad.model.blocks[b]), torch.as_tensor(planted.perms[b])
        assert torch.allclose(f["W1"][p], f["W1"] @ At, atol=1e-12)               # P W1 = W1 A*
        assert torch.allclose(f["b1"][p], f["b1"], atol=1e-12)                     # P b1 = b1
    raw = CT.tiny_adapter()                                                          # teeth: before the conversion
    assert max(r["rel_max"] for r in stream_equivariance(raw, raw.model, x, A)) > 1e-3


def test_reference_finds_the_planted_structure(a03):
    rt, tab = a03["rt"], a03["table"]
    planted, L, h = rt.extra["planted_structure"], rt.adapter.depth, rt.adapter.mlp_hidden
    A = np.asarray(planted.A_star[L])
    st, _ = _row_structure(a03, "mirror")
    assert all(np.abs(np.asarray(st.A_layers[l]) - A).max() < 1e-8 for l in range(L + 1))
    assert np.abs(np.asarray(st.A_star[L]) - A).max() < 1e-8
    assert all(np.array_equal(st.perms[b], planted.perms[b]) for b in range(L))
    r = tab.loc["mirror"]
    assert int(r["d_minus_span"]) == A.shape[0] - _dp(A) > 0
    share = float(np.mean([2 * planted.n_pairs(b) / h for b in range(L)]))
    assert share > 0.5 and r[f"twin_fraction_t50_{RANGE}"] == pytest.approx(share, abs=1e-12)
    assert r[f"ratio_to_null_{RANGE}"] < 1e-10 and r[f"ratio_to_null_split_{RANGE}"] < 1e-10
    lay = pd.read_csv(a03["out"] / "layers.csv")
    assert np.allclose(lay[lay["control"] == "mirror"]["abs_sum"], 1.0, atol=1e-8)


def test_controls_find_no_twins(a03):
    tab = a03["table"]
    for c in CT.TESTED_CONTROLS:
        assert tab.loc[c, f"twin_fraction_t50_{RANGE}"] <= CT.A03_TWIN_MAX, c
        assert tab.loc[c, f"twin_fraction_t50_split_{RANGE}"] <= CT.A03_TWIN_MAX, c
        assert tab.loc[c, "span_abs_sum"] < 0.5                                   # reference: 1.0
    assert tab.loc["C4", f"twin_fraction_t50_{RANGE}"] <= CT.A03_TWIN_MAX


def test_split_half_ratio_is_near_one_for_the_controls(a03):
    tab = a03["table"]
    for c in ("C1", "C2"):
        assert 0.9 <= tab.loc[c, f"ratio_to_null_split_{RANGE}"] <= 1.1, c
    assert tab.loc["C3", f"ratio_to_null_split_{RANGE}"] >= 0.8
    score = a03["metrics"]["scores"]["A03-1"]                 # the pre-registered score uses the split-half ratio
    assert score["controls"]["C1"]["status"] == score["controls"]["C2"]["status"] == "PASS"
    for c in ("C1", "C2", "C3"):
        assert score["controls"][c]["ratio_to_null"] == pytest.approx(tab.loc[c, f"ratio_to_null_split_{RANGE}"])
    assert "A03-1_same_sample" in a03["metrics"]["diagnostics"]


def test_same_sample_ratio_and_its_cause(a03):
    """The same-sample ratio of a structure fitted without any mirror relation sits near 0.88, not 1: the fitted
    operator alone (with the null's pairings) and the fitted pairing alone (with the null's operators) sit at the
    null; only both together fall below it."""
    from mirror.surgery import fold_mlp

    tab = a03["table"]
    for c in CT.TESTED_CONTROLS:
        assert 0.75 <= tab.loc[c, f"ratio_to_null_{RANGE}"] <= 1.05, c
    assert abs(tab.loc["C4", f"ratio_to_null_{RANGE}"] - 1.0) < 0.05             # the null against itself
    model = a03["rt"].adapter.model
    st, _ = _row_structure(a03, "C1")
    L, d, n = len(model.blocks), st.d, 10
    fold = {b: fold_mlp(model.blocks[b]) for b in range(L)}

    def s(A0, A1, p, b):
        return GR.mlp_group_shares(fold[b], GR.block_action("LR", [A0], [A1], [p]))["s_mlp"]

    fit, a_only, p_only, null = [], [], [], []
    for b in range(L):
        A0, A1, p = st.A_layers[b], st.A_layers[b + 1], st.perms[b]
        pr = CT.null_pairings(p, b, n, seed=0)
        ra = [(GR.random_c2(d, _dp(A0), CT.null_seed(0, k, b)), GR.random_c2(d, _dp(A1), CT.null_seed(0, k, b + 1)))
              for k in range(n)]
        fit.append(s(A0, A1, p, b))
        a_only.append(np.mean([s(A0, A1, q, b) for q in pr]))
        p_only.append(np.mean([s(R0, R1, p, b) for R0, R1 in ra]))
        null.append(np.mean([s(R0, R1, q, b) for (R0, R1), q in zip(ra, pr)]))
    nm = np.mean(null)
    assert abs(np.mean(a_only) / nm - 1) < 0.05 and abs(np.mean(p_only) / nm - 1) < 0.05
    assert np.mean(fit) / nm < 0.95


def test_a03_outputs_and_scoring(a03):
    from mirror.mirror_family import load_span_structure

    out, m, tab, L = a03["out"], a03["metrics"], a03["table"], a03["rt"].adapter.depth
    for name in ("controls.csv", "layers.csv", "blocks.csv", "null_draws.csv", "knn.csv", "spectra.npz",
                 "summary.md", "metrics.json", "meta.json"):
        assert (out / name).exists(), name
    assert tab.index.tolist() == list(CT.ROWS)
    s = CT.score_a03({c: {"twin_fraction_t50": float(tab.loc[c, f"twin_fraction_t50_{RANGE}"]),
                          "ratio_to_null": float(tab.loc[c, f"ratio_to_null_split_{RANGE}"])}
                      for c in CT.TESTED_CONTROLS})
    assert m["scores"]["A03-1"]["status"] == s["status"]
    same = CT.score_a03({c: {"twin_fraction_t50": float(tab.loc[c, f"twin_fraction_t50_{RANGE}"]),
                             "ratio_to_null": float(tab.loc[c, f"ratio_to_null_{RANGE}"])}
                         for c in CT.TESTED_CONTROLS})
    assert m["diagnostics"]["A03-1_same_sample"]["status"] == same["status"]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["experiment"] == "A03" and "prereg_check" in meta["extra"] and meta["git"]["git_hash"]
    with np.load(out / "spectra.npz") as z:
        d = a03["rt"].adapter.embed_dim
        for r in CT.CONTROL_ROWS:
            assert z[f"{r}_layers"].shape == (L + 1, d) and z[f"{r}_span"].shape == (d,)
            assert np.all(np.diff(z[f"{r}_span"]) >= 0)                          # sorted
    for r in CT.CONTROL_ROWS:
        st = load_span_structure(out / "structures" / f"tiny_equivariant_{r}_fit0.npz")
        assert st.depth == L and st.meta["control"]["name"] == r
    knn = pd.read_csv(out / "knn.csv").set_index("control")
    assert (knn["flip_consistency"] == 100.0).all()          # Mirror-exact is exactly invariant whatever the structure
    assert knn.loc["C4", "method"] == "random_structure" and knn.loc["C1", "method"] == "mirror_exact"
    blk, drw = pd.read_csv(out / "blocks.csv"), pd.read_csv(out / "null_draws.csv")
    assert len(blk) == 4 * L and len(drw) == 4 * L * m["null_draws"]
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "Rule A03-1 (pre-registered)" in text and chr(0x2014) not in text          # no em dash in the report


def test_a03_rerun_uses_the_cache(a03):
    out = a03["out"]
    files = sorted((out / "cache").glob("*")) + sorted((out / "structures").glob("*"))
    before = {p.name: p.stat().st_mtime_ns for p in files}
    table = (out / "controls.csv").read_text(encoding="utf-8")
    A03.main(a03["argv"], runtime=a03["rt"])
    assert {p.name: p.stat().st_mtime_ns for p in files} == before
    assert (out / "controls.csv").read_text(encoding="utf-8") == table


# ---------------------------------------------------------------------------------------------- A15b
def test_a15b_dry_run_follows_measure_groups(tmp_path):
    out = tmp_path / "a15b"
    m = A15B.main(["--dry-run", "--out", str(out)])
    for name in ("a15b.csv", "summary.md", "metrics.json", "meta.json", "original/blocks.csv", "finetuned/blocks.csv"):
        assert (out / name).exists(), name
    sc = m["scores"]["A15b"]
    assert sc["status"] in ("PASS", "FAIL")
    assert sc["drop"] == pytest.approx(m["original"]["ratio_to_null"] - m["finetuned_numbers"]["ratio_to_null"])
    assert m["finetuned"]["loader"] == "state dict"                               # the wrapped {"model": ...} file
    assert m["original"]["ratio_to_null"] != m["finetuned_numbers"]["ratio_to_null"]
    # the original's numbers are those of scripts/measure_groups.py --tiny itself (the MUST-04 code path)
    ref = A15B.load_measure_groups().main(["--tiny", "--groups", "R90", "--blocks", "1-6", "--null-draws", "3",
                                           "--twin-null-draws", "0", "--summary-ranges", "1-6", "--no-heldout",
                                           "--out", str(tmp_path / "mg")])["groups"]["R90"]["ranges"]["1-6"]
    for key in ("ratio_to_null", "share_4cycles", "twin_fraction_t50", "s_mlp_mean", "s_mlp_null_mean"):
        assert m["original"][key] == pytest.approx(ref[key], rel=1e-12, abs=1e-15), key
    # a different checkpoint in the same folder replaces the fine-tuned tables and keeps the original's
    orig_mtime = (out / "original" / "blocks.csv").stat().st_mtime_ns
    from mirror.adapters import TimmTinyAdapter

    sd = TimmTinyAdapter(img_size=56, seed=0).model.state_dict()
    torch.save({"_orig_mod." + k: v for k, v in sd.items()}, tmp_path / "same.pt")   # the original's own weights
    m2 = A15B.main(["--dry-run", "--out", str(out), "--finetuned", str(tmp_path / "same.pt")])
    assert (out / "original" / "blocks.csv").stat().st_mtime_ns == orig_mtime
    assert m2["finetuned_numbers"]["ratio_to_null"] == pytest.approx(m2["original"]["ratio_to_null"], rel=1e-12)
    assert m2["scores"]["A15b"]["drop"] == pytest.approx(0.0, abs=1e-12)
