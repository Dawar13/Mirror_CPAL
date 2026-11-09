"""Strong baselines (mirror/baselines.py, scripts/strong_baselines.py): CPU only, no downloads.

  * budgets: on the fitted counts of the real fit runs (DINOv2-B, d = 768, h = 3072; files committed under
    runs/x2/), every method lands within 1% of the Mirror V-local MLP-only count for every k, and the
    Mirror counts reproduce those recorded by the Track A runs; asvd uses exactly the plain SVD ranks;
  * statistics: the folded-model pass equals direct computations on the original model;
  * asvd at full rank and slicegpt without slicing reproduce the original outputs (float64, 1e-6);
  * asvd is the whitened rank-r optimum; slicegpt in absorbed form equals slicing the folded model;
  * prune_act keeps exactly the requested neurons (the top scores) and removes the rest;
  * the analytic parameter counts equal what the built models hold;
  * the --tiny path runs end to end, writes the Track A CSV format and resumes.
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror import baselines as B
from mirror import variants as V
from mirror.data import normalize_for
from mirror.fold import fold_copy
from mirror.matching import random_pairing
from mirror.surgery import equivariant_param_count, fold_mlp, svd_truncate, write_absorbed_mlp

ROOT = Path(__file__).resolve().parents[1]


def make_struct(adapter, seed=0):
    """Per-layer random involutions (A 1 = 1, varied d+) and random pairings with varied counts."""
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    A = [V.random_operator(d, d // 2 + 3 * (i % 3) + 1, seed=seed * 100 + i) for i in range(L + 1)]
    dps = [V.d_plus_of(a) for a in A]
    perms = [random_pairing(h, h // 3 + 5 * b, seed=seed * 100 + 50 + b) for b in range(L)]
    return V.Structure(A_layers=A, d_plus=dps, d_minus=[d - x for x in dps], perms=perms)


def uint8_loader(n=4, seed=2):
    imgs = torch.from_numpy(np.random.default_rng(seed).integers(0, 256, (n, 3, 56, 56), dtype=np.uint8))
    return [(imgs[:2], torch.zeros(2, dtype=torch.long)), (imgs[2:], torch.zeros(n - 2, dtype=torch.long))], imgs


def all_stats(adapter, blocks=None):
    loader, _ = uint8_loader()
    blocks = range(adapter.depth) if blocks is None else blocks
    return B.collect_baseline_stats(adapter, loader, "cpu", blocks)


# ---------------------------------------------------------------- budgets on the real fitted counts
REAL_RUNS = sorted(p for p in (ROOT / "runs" / "x2").glob("perlayer_*_s*") if (p / "results" / "e1_mlp.csv").exists())


@pytest.mark.parametrize("run", REAL_RUNS, ids=[p.name for p in REAL_RUNS])
def test_every_method_within_1pct_of_mirror_on_real_counts(run):
    lay = pd.read_csv(run / "results" / "e1_layers.csv")
    mlp = pd.read_csv(run / "results" / "e1_mlp.csv").sort_values("block")
    d, h = int(mlp["d"].iloc[0]), int(mlp["h"].iloc[0])
    assert (d, h) == (768, 3072)
    dp = lay.sort_values("layer")["d_plus"].astype(int).tolist()
    eq = []
    for b, r in enumerate(mlp.itertuples()):
        e1 = equivariant_param_count(h, d, int(r.n_self), 2 * dp[b] - d)
        e2 = equivariant_param_count(h, d, int(r.n_self), 2 * dp[b + 1] - d)
        assert (e1, e2) == (int(r.eq_params_fc1), int(r.eq_params_fc2))   # same count as run_e1.py
        eq.append((e1, e2))
    worst = {}
    for k in range(1, len(eq) + 1):
        target = sum(a + b for a, b in eq[:k])
        for m in B.METHODS:
            alloc, kept = B.allocate(m, h, d, eq[:k])
            assert kept == B.method_params(m, h, d, alloc)
            assert 0.99 <= kept / target <= 1.0, (m, k, kept / target)
            worst[m] = min(worst.get(m, 1.0), kept / target)
        # asvd keeps the plain SVD ranks that run_e1.py recorded
        ranks, _ = B.allocate("asvd", h, d, eq[:k])
        assert [list(r) for r in ranks] == mlp[["svd_rank", "svd_rank_fc2"]].iloc[:k].astype(int).values.tolist()
    assert min(worst.values()) > 0.995


def test_mirror_budget_reproduces_track_a_counts():
    """The budget the baselines match is the Track A per-layer mirror count (A1 per_layer rows), and asvd keeps
    exactly the plain SVD baseline's weights (A1 svd rows)."""
    run = ROOT / "runs" / "x2" / "perlayer_reg_s0" / "results"
    a1 = ROOT / "runs" / "a" / "dinov2_vitb14_reg" / "A1_s0.csv"
    if not (run / "e1_mlp.csv").exists() or not a1.exists():
        pytest.skip("fit run or Track A CSV not present")
    mlp = pd.read_csv(run / "e1_mlp.csv").sort_values("block")
    eq = list(zip(mlp["eq_params_fc1"].astype(int), mlp["eq_params_fc2"].astype(int)))
    df = pd.read_csv(a1)
    for k in sorted(df["k"].unique()):
        mirror = df[(df["variant"] == "per_layer") & (df["k"] == k)]["mlp_params_kept"].iloc[0]
        svd = df[(df["variant"] == "svd") & (df["k"] == k)]["mlp_params_kept"].iloc[0]
        assert sum(a + b for a, b in eq[:k]) == int(mirror)
        assert B.allocate("asvd", 3072, 768, eq[:k])[1] == int(svd)


def test_plan_matches_svd_spec_and_mirror_counts(tiny64):
    struct = make_struct(tiny64)
    for k in (1, 3):
        pl = B.plan_for("asvd", struct, k)
        assert [tuple(r) for r in pl.alloc] == [tuple(r) for r in V.svd_spec("svd", struct, range(k)).ranks]
        assert pl.mirror_budget == sum(a + b for a, b in V.mirror_param_counts(struct, range(k)))
        assert pl.params_dense == 2 * struct.h * struct.d * k
        for m in B.METHODS:
            p = B.plan_for(m, struct, k)
            assert p.params_kept <= p.mirror_budget and p.params_kept == B.method_params(m, struct.h, struct.d, p.alloc)


# ---------------------------------------------------------------- statistics
def test_statistics_match_direct_computation(tiny64):
    loader, imgs = uint8_loader()
    st = B.collect_baseline_stats(tiny64, loader, "cpu", [0, 2])
    ref = V.collect_mlp_input_moments(tiny64, loader, "cpu", blocks=[0, 2])
    d, h = tiny64.embed_dim, tiny64.mlp_hidden
    assert st.n_images == 4 and st.n_tokens == 4 * tiny64.n_tokens
    with torch.no_grad():
        t = tiny64.layer0_tokens(normalize_for(tiny64, imgs))
        for b, blk in enumerate(tiny64.model.blocks):
            mid = t + blk.ls1(blk.attn(blk.norm1(t)))
            if b in (0, 2):
                assert (st.M1[b] - ref["M1"][b]).abs().max() <= 1e-9 * ref["M1"][b].abs().max()
                assert (st.M2[b] - ref["M2"][b]).abs().max() <= 1e-9 * ref["M2"][b].abs().max()
                X = mid.reshape(-1, d)
                Xc = X - X.mean(dim=1, keepdim=True)                       # the folded stream is the centred one
                S = Xc.T @ Xc
                assert (st.S[b] - S).abs().max() <= 1e-9 * S.abs().max()
                H = blk.mlp.act(blk.mlp.fc1(blk.norm2(mid))).reshape(-1, h)
                assert (st.abs_g[b] - H.abs().sum(0)).abs().max() <= 1e-9 * H.abs().sum(0).max()
            t = blk(t)
    assert st.info["max_rel_channel_mean_of_folded_stream"] < 1e-10
    only = B.collect_baseline_stats(tiny64, loader, "cpu", [1], need=B.needs_for(["prune_act"]))
    assert set(only.abs_g) == {1} and not only.M1 and not only.M2 and not only.S


# ---------------------------------------------------------------- exactness at no compression
def _max_diff(adapter, images, model):
    with torch.no_grad():
        a = adapter.forward_tokens(images)
        b = adapter.forward_tokens(images, model)
        c = (adapter.cls_feature(images) - adapter.cls_feature(images, model)).abs().max().item()
    return max((a - b).abs().max().item(), c)


def test_asvd_full_rank_reproduces_original(tiny64, images64):
    st = all_stats(tiny64)
    r = min(tiny64.embed_dim, tiny64.mlp_hidden)
    blocks = list(range(tiny64.depth))
    m, info = B.baseline_copy(tiny64.model, "asvd", blocks, [(r, r)] * len(blocks), st)
    assert _max_diff(tiny64, images64, m) < 1e-6
    assert all(i["whitened_discarded_fc1"] < 1e-12 and i["whitened_discarded_fc2"] < 1e-12 for i in info)


def test_slicegpt_without_slicing_reproduces_original(tiny64, images64):
    st = all_stats(tiny64)
    d = tiny64.embed_dim
    blocks = list(range(tiny64.depth))
    m, info = B.baseline_copy(tiny64.model, "slicegpt", blocks, [(d, d)] * len(blocks), st)
    assert _max_diff(tiny64, images64, m) < 1e-6
    assert all(abs(i["pca_energy_kept_in"] - 1) < 1e-12 for i in info)


# ---------------------------------------------------------------- method properties
def test_asvd_is_the_whitened_optimum():
    rng = np.random.default_rng(3)
    for shape in ((48, 12), (12, 48)):               # fc1 and fc2 orientations
        n = shape[1]
        X = rng.standard_normal((300, n)) * rng.uniform(0.1, 3, n)
        X -= X.mean(1, keepdims=True)                # mean-zero inputs: the all-ones direction is null (as fc1)
        M = torch.tensor(X.T @ X)
        W = torch.tensor(rng.standard_normal(shape))
        r = 5
        Wn, info = B.asvd_truncate(W, M, r, damp=1e-6)
        assert int(torch.linalg.matrix_rank(Wn)) == r
        L = torch.linalg.cholesky(M + info["ridge"] * torch.eye(n, dtype=M.dtype))
        U = torch.linalg.svd(W @ L, full_matrices=False)[0][:, :r]
        assert (Wn - U @ U.T @ W).abs().max() < 1e-9 * W.abs().max()     # = U_r U_r^T W
        Xt = torch.tensor(X)

        def err(Wa):
            return float(((Xt @ (W - Wa).T) ** 2).sum())

        best = err(Wn)
        assert best <= err(svd_truncate(W, r))                       # beats plain SVD on the activations
        for _ in range(5):                                           # and other rank-r maps of this form
            Q = torch.linalg.qr(torch.tensor(rng.standard_normal((shape[0], r))))[0]
            assert best <= err(Q @ Q.T @ W)


def test_slicegpt_absorbed_form_equals_slicing_the_folded_model(tiny64, images64):
    st = all_stats(tiny64)
    blocks, alloc = [0, 1, 2], [(40, 24), (17, 50), (33, 33)]
    m, _ = B.baseline_copy(tiny64.model, "slicegpt", blocks, alloc, st)
    folded, _ = fold_copy(tiny64.model, final_norm_affine=False)
    for b, (di, do) in zip(blocks, alloc):
        B.slice_folded_block(folded.blocks[b], B.pca_basis(st.S[b])[0], di, do)
    with torch.no_grad():
        f_abs = tiny64.cls_feature(images64, m)
        f_fold = tiny64.cls_feature(images64, folded)
        f_orig = tiny64.cls_feature(images64)
    assert (f_abs - f_fold).abs().max().item() < 1e-10
    assert (f_abs - f_orig).abs().max().item() > 1e-3                  # slicing did change the model
    for b, (di, do) in zip(blocks, alloc):
        W1 = m.blocks[b].mlp.fc1.weight
        W2 = m.blocks[b].mlp.fc2.weight
        assert int(torch.linalg.matrix_rank(W1)) == di and int(torch.linalg.matrix_rank(W2)) == do
        assert W2.sum(0).abs().max() < 1e-10 * W2.abs().max()          # writes are mean-zero over channels
        Q = B.pca_basis(st.S[b])[0]
        assert (W1 @ Q[:, di:]).abs().max() < 1e-10 * W1.abs().max()   # fc1 reads only the top d_in directions
        assert (Q[:, do:].T @ W2).abs().max() < 1e-10 * W2.abs().max()  # fc2 writes only into the top d_out


def test_prune_act_keeps_exactly_the_requested_neurons(tiny64, images64):
    loader, imgs = uint8_loader()
    st = B.collect_baseline_stats(tiny64, loader, "cpu", range(4), need=B.needs_for(["prune_act"]))
    h = tiny64.mlp_hidden
    widths = [100, 7, 256, 131]
    m, info = B.baseline_copy(tiny64.model, "prune_act", range(4), widths, st)
    # independent importance: mean |GELU output| over the same tokens times the folded fc2 column norm
    with torch.no_grad():
        t = tiny64.layer0_tokens(normalize_for(tiny64, imgs))
        for b, blk in enumerate(tiny64.model.blocks):
            mid = t + blk.ls1(blk.attn(blk.norm1(t)))
            H = blk.mlp.act(blk.mlp.fc1(blk.norm2(mid))).reshape(-1, h)
            f = fold_mlp(blk)
            score = H.abs().mean(0) * f["W2"].norm(dim=0)
            keep = torch.sort(torch.argsort(-score, stable=True)[:widths[b]]).values
            fc1, fc2 = m.blocks[b].mlp.fc1, m.blocks[b].mlp.fc2
            assert fc1.out_features == widths[b] == fc2.in_features and info[b]["n_kept"] == widths[b]
            assert torch.equal(fc1.weight, f["W1"][keep].to(fc1.weight)) and torch.equal(fc1.bias, f["b1"][keep])
            assert torch.equal(fc2.weight, f["W2"][:, keep]) and torch.equal(fc2.bias, f["b2"])
            # removed neurons are the lowest scores
            dropped = torch.ones(h, dtype=torch.bool)
            dropped[keep] = False
            if dropped.any():
                assert score[dropped].max() <= score[keep].min()
            t = blk(t)
    # the pruned model computes the dense model with the removed neurons switched off
    ref = copy.deepcopy(tiny64.model)
    with torch.no_grad():
        for b, blk in enumerate(ref.blocks):
            f = fold_mlp(blk)
            kept = set(B.prune_keep(B.prune_scores(f["W2"], st.mean_abs_g(b)), widths[b]).tolist())
            off = torch.tensor([j not in kept for j in range(h)])
            W2 = f["W2"].clone()
            W2[:, off] = 0
            write_absorbed_mlp(blk, f["W1"], f["b1"], W2, f["b2"])
        assert (tiny64.cls_feature(images64, m) - tiny64.cls_feature(images64, ref)).abs().max().item() < 1e-10
    assert B.realised_params(m, "prune_act", range(4), widths) == B.method_params("prune_act", 0, 64, widths)


def test_built_models_hold_the_counted_parameters(tiny64):
    struct = make_struct(tiny64)
    st = all_stats(tiny64)
    for m in B.METHODS:
        for k in (1, 3):
            pl = B.plan_for(m, struct, k)
            model, _ = B.baseline_copy(tiny64.model, m, pl.blocks, pl.alloc, st)
            assert B.realised_params(model, m, pl.blocks, pl.alloc) == pl.params_kept, (m, k)
            for b in range(k, tiny64.depth):                            # blocks after the span are untouched
                for p, q in zip(model.blocks[b].parameters(), tiny64.model.blocks[b].parameters()):
                    assert torch.equal(p, q)
            sig = B.weights_signature(model, m, pl.blocks, pl.alloc)
            assert sig == B.weights_signature(B.baseline_copy(tiny64.model, m, pl.blocks, pl.alloc, st)[0], m,
                                              pl.blocks, pl.alloc)


# ---------------------------------------------------------------- CLI --tiny path
def run_script(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "strong_baselines.py"), *args],
                          capture_output=True, text=True, timeout=1200)


def test_strong_baselines_tiny_end_to_end_and_resumable(tmp_path):
    out = tmp_path / "sb"
    common = ["--tiny", "--ks", "1,2", "--out", str(out), "--top-k", "3"]
    proc = run_script(*common)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    df = pd.read_csv(out / "SB_s0.csv")
    a1 = ROOT / "runs" / "a" / "dinov2_vitb14_reg" / "A1_s0.csv"
    if a1.exists():                                                     # the Track A column format
        assert list(df.columns) == list(pd.read_csv(a1, nrows=1).columns)
    assert len(df) == 6 and set(df["variant"]) == set(B.METHODS) and (df["ablation"] == "SB").all()
    assert not df.duplicated(["ablation", "variant", "k", "fit_seed"]).any()
    assert df["git_hash"].notna().all() and (df["seed"] == 0).all() and (df["fit_seed"] == 0).all()
    assert df["top1"].between(0, 100).all() and df["flip_consistency"].between(0, 100).all()
    assert (df["params_vs_mirror_first_k"] <= 1).all() and (df["params_vs_mirror_first_k"] > 0.95).all()
    for _, r in df.iterrows():
        info = json.loads(r["info"])
        assert info["params_kept"] == r["mlp_params_kept"]
        assert abs(info["ratio_to_mirror"] - r["params_vs_mirror_first_k"]) < 1e-12
        assert abs(r["mlp_params_kept"] / info["mirror_budget"] - r["params_vs_mirror_first_k"]) < 1e-12
    meta = json.loads((out / "SB_s0_meta.json").read_text(encoding="utf-8"))
    assert "git_hash" in meta["git"] and meta["fit_seed"] == 0 and meta["seed"] == 0
    assert meta["data_checks"]["fit_bank_overlap"] == 0 and len(meta["budgets"]) == 6
    assert meta["statistics"]["n_images"] == 10 and meta["rows_added"] == 6
    summary = (out / "SB_SUMMARY.md").read_text(encoding="utf-8")
    assert "prune_act" in summary and "\u2014" not in summary
    # A second run finds every row, skips the statistics pass and adds nothing.
    proc = run_script(*common)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "skipped" in proc.stdout and "statistics" not in proc.stdout
    assert len(pd.read_csv(out / "SB_s0.csv")) == 6
    assert json.loads((out / "SB_s0_meta.json").read_text(encoding="utf-8"))["rows_added"] == 0
