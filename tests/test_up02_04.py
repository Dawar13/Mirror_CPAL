"""UP-02 and UP-04 (mirror/controlled_sym.py, scripts/train_controlled_sym.py). CPU only, synthetic data, the 3-block
test ViT at the real 64 px / patch 8 layout.

  * the canonical projection is exact: breaking share 0, logits of UD(x) equal those of x, aligned tokens of UD(x)
    equal A times the tokens of x after every block; free counts equal the number of distinct weight values;
  * a dense (fitted-type) structure on a norm-folded model is exact too, and folding keeps the function;
  * sym_init starts at breaking share 0 and training breaks it; tied training keeps it at 0 after every step;
  * a tied run stopped and resumed ends identical to an uninterrupted one;
  * refund_heal fits the structure of an unconstrained UD run, refunds it exactly and keeps it exact while healing;
  * checkpoints load through mirror.controlled.load_checkpoint_adapter and scripts/measure_groups.py.
"""
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mirror import controlled as C
from mirror import controlled_sym as S
from mirror import groups as GR

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_t", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _random_model(seed=0, dtype=torch.float64, scale=0.1):
    """Test ViT with every parameter perturbed (LayerNorm affines and biases included)."""
    torch.manual_seed(seed)
    m = C.build_model("vit_test8", num_classes=10).to(dtype)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn_like(p))
    return m.eval()


def _aligned_token_gap(model, st, x):
    """Max over blocks of ||tokens(UD x)[tau] - tokens(x) A|| / ||tokens(x)||."""
    ad = C.ControlledAdapter(model, "vit_test8")
    A = torch.as_tensor(st.A.matrix(), dtype=x.dtype)
    tok = torch.as_tensor(st.tokens)
    outs = {}
    hooks = [blk.register_forward_hook(lambda m, i, o, b=b: outs.setdefault(b, []).append(o))
             for b, blk in enumerate(model.blocks)]
    with torch.no_grad():
        ad.forward_tokens(x)
        ad.forward_tokens(torch.flip(x, dims=[-2]))
    for h in hooks:
        h.remove()
    return max(float((o[1][:, tok] - o[0] @ A).norm() / o[0].norm()) for o in outs.values())


# ----------------------------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------------------------
def test_canonical_projection_is_exact():
    m = _random_model(0)
    st = S.canonical_structure(m)
    assert st.A.is_perm and st.A.trace == 0 and st.info["d_minus"] == 24          # half and half (d = 48)
    assert np.array_equal(st.A.matrix() @ np.ones(48), np.ones(48))                # A 1 = 1
    before = S.breaking_shares(m, st)
    assert before["s_all"] > 1e-3 and before["s_mlp_b1_3"] > 0.1
    S.project_(m, st, "param")
    after = S.breaking_shares(m, st)
    assert after["s_all"] == 0.0 and after["s_max_tensor"] == 0.0                  # exact: pair averages
    for b in range(1, 4):
        assert after[f"s_fc1_b{b}"] < 1e-25 and after[f"s_fc2_b{b}"] < 1e-25 and after[f"s_attn_b{b}"] < 1e-25
    x = torch.randn(3, 3, 64, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    assert S.ud_function_gap(m, x)["logit_rel_gap"] < 1e-12
    assert _aligned_token_gap(m, st, x) < 1e-12
    w = {k: v.clone() for k, v in m.state_dict().items()}
    S.project_(m, st, "param")                                                     # idempotent, bit for bit
    assert all(torch.equal(w[k], v) for k, v in m.state_dict().items())


@pytest.mark.parametrize("d_minus,hidden_pairs", [(0, 1.0), (10, 0.6), (24, 0.0)])
def test_free_counts_equal_number_of_orbits(d_minus, hidden_pairs):
    """After projecting random weights, entries in one orbit are equal and different orbits differ, so the number
    of distinct values of each tensor is its number of free parameters."""
    m = _random_model(2, scale=1.0)
    st = S.canonical_structure(m, d_minus=d_minus, hidden_pairs=hidden_pairs)
    S.project_(m, st, "param")
    cnt = S.free_param_counts(m, st)
    for name, p in m.named_parameters():
        assert len(torch.unique(p.detach())) == cnt["per_tensor"][name]["free"], name
    assert cnt["dense"] == C.count_params(m)
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    assert S.ud_function_gap(m, x)["logit_rel_gap"] < 1e-12
    if d_minus == 0 and hidden_pairs == 1.0:
        assert 0.49 < cnt["free_frac"] < 0.51


def test_dense_structure_on_folded_model_is_exact():
    from mirror.m1 import random_head_structure, random_involution_fixing_ones
    from mirror.matching import random_pairing

    m = _random_model(4)
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        y0 = m(x)
    S.fold_norms_(m)
    with torch.no_grad():
        assert float((m(x) - y0).norm() / y0.norm()) < 1e-12                       # folding keeps the function
    assert all(bool((n.weight == 1).all()) and bool((n.bias == 0).all())
               for n in [m.norm] + [b.norm1 for b in m.blocks] + [b.norm2 for b in m.blocks])
    qk, v = [], []
    for b in range(3):
        hm = random_head_structure(3, 16, 1, b)
        qk.append(S.Op(dense=GR.head_rep(hm.sigma, hm.B)))
        v.append(S.Op(dense=GR.head_rep(hm.sigma, hm.C)))
    st = S.UDStructure("fitted", S.Op(dense=random_involution_fixing_ones(48, 30, 0)),
                       [S.Op(perm=random_pairing(192, 70, b)) for b in range(3)], qk, v, 8, 8, {})
    with torch.no_grad():
        m.blocks[0].norm2.weight.mul_(1.5)                                         # gamma becomes a scalar
    S.project_(m, st, "param")
    assert S.breaking_shares(m, st)["s_all"] < 1e-25
    assert S.ud_function_gap(m, x)["logit_rel_gap"] < 1e-12
    assert _aligned_token_gap(m, st, x) < 1e-12
    assert torch.allclose(m.blocks[0].norm2.weight, torch.full((48,), 1.5, dtype=torch.float64))


def test_gradient_projection_is_the_restricted_gradient():
    """Projecting the gradient equals the gradient of the loss restricted to the equivariant subspace: a step
    along the projected gradient stays equivariant and changes the loss to first order exactly as the full
    gradient does along that direction."""
    m = _random_model(6)
    st = S.canonical_structure(m)
    S.project_(m, st, "param")
    x = torch.randn(4, 3, 64, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    loss = m(x).pow(2).mean()
    loss.backward()
    full = {n: p.grad.clone() for n, p in m.named_parameters()}
    S.project_(m, st, "grad")
    dot_full = sum(float((full[n] * p.grad).sum()) for n, p in m.named_parameters())
    dot_proj = sum(float((p.grad * p.grad).sum()) for n, p in m.named_parameters())
    assert math.isclose(dot_full, dot_proj, rel_tol=1e-10)                      # self-adjoint projector
    with torch.no_grad():
        for p in m.parameters():
            p.sub_(1e-2 * p.grad)
    assert S.breaking_shares(m, st)["s_all"] == 0.0


def test_fast_projector_matches_generic_bit_for_bit():
    """PermProjector (one gather per tensor, float32) equals project_ (float64, rounded once), weights and
    gradients; for a dense structure make_projector falls back to project_."""
    a, b = _random_model(8, dtype=torch.float32, scale=1.0), _random_model(8, dtype=torch.float32, scale=1.0)
    st = S.canonical_structure(a, d_minus=10, hidden_pairs=0.7)
    assert S.is_permutation_structure(st)
    for m in (a, b):
        for p in m.parameters():
            p.grad = torch.randn_like(p)
    for p, q in zip(a.parameters(), b.parameters()):
        q.grad.copy_(p.grad)
    fast = S.make_projector(a, st)
    assert isinstance(fast, S.PermProjector)
    fast("param")
    fast("grad")
    S.project_(b, st, "param")
    S.project_(b, st, "grad")
    for (n, p), q in zip(a.named_parameters(), b.parameters()):
        assert torch.equal(p, q) and torch.equal(p.grad, q.grad), n
    from mirror.m1 import random_involution_fixing_ones

    dense = S.UDStructure("fitted", S.Op(dense=random_involution_fixing_ones(48, 30, 0)), st.hidden, st.qk, st.v,
                          8, 8, {})
    assert not S.is_permutation_structure(dense) and not isinstance(S.make_projector(a, dense), S.PermProjector)


def test_init_rescale_keeps_symmetry_and_restores_scale():
    torch.manual_seed(0)
    m = C.build_model("vit_test8", num_classes=10)
    w = m.blocks[0].mlp.fc1.weight.detach().clone()
    st = S.canonical_structure(m)
    S.project_(m, st, "param")
    proj = m.blocks[0].mlp.fc1.weight.detach().clone()
    S.restore_init_scale_(m, st)
    assert torch.allclose(m.blocks[0].mlp.fc1.weight, proj * math.sqrt(2.0))      # every fc1 entry is in a pair
    assert S.breaking_shares(m, st)["s_all"] < 1e-30
    ratio = float(m.blocks[0].mlp.fc1.weight.detach().std() / w.std())
    assert 0.85 < ratio < 1.15


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
def _cfg(tmp, mode, run, **kw):
    base = dict(exp="TEST", arch="vit_test8", condition="UD", mode=mode, seed=1, epochs=2, ckpt_epochs=(0, 1, 2),
                batch_size=16, lr=3.2e-2, warmup_epochs=1.0, device="cpu", amp="none", compile="off", eval_batch=16,
                save_every_min=0.0, log_every=0, cache_dir=str(tmp / "cache"), run_dir=str(tmp / run),
                results_root=str(tmp / "results"), fit_images=20, match_workers=1)
    base.update(kw)
    return S.SymTrainConfig(**base)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sym")
    C.write_synthetic_cache(tmp / "cache", num_classes=10, train_per_class=8, val_per_class=4, seed=0)
    out = {"tmp": tmp}
    out["sym"] = S.train_sym(_cfg(tmp, "sym_init", "sym"))
    out["tied"] = S.train_sym(_cfg(tmp, "tied", "tied"))
    out["stops"] = (S.train_sym(_cfg(tmp, "tied", "tied_b", stop_after_steps=3)),     # 5 steps per epoch
                    S.train_sym(_cfg(tmp, "tied", "tied_b", stop_after_steps=5)))
    out["tied_b"] = S.train_sym(_cfg(tmp, "tied", "tied_b"))
    src = C.TrainConfig(exp="CORE-01", arch="vit_test8", condition="UD", seed=1, epochs=2, ckpt_epochs=(0, 2),
                        batch_size=16, lr=3.2e-2, warmup_epochs=1.0, device="cpu", amp="none", compile="off",
                        eval_batch=16, save_every_min=0.0, log_every=0, cache_dir=str(tmp / "cache"),
                        run_dir=str(tmp / "src"))
    C.train(src)
    out["src_ckpt"] = tmp / "src" / "ckpt_e002.pt"
    out["refund"] = S.train_sym(_cfg(tmp, "refund_heal", "refund", epochs=1, ckpt_epochs=(0, 1), lr=3.2e-3,
                                     warmup_epochs=0.2, source_ckpt=str(out["src_ckpt"])))
    return out


def test_sym_init_starts_exact_and_training_breaks_it(runs):
    rows = _rows(runs["tmp"] / "sym" / "breaking_share.csv")
    assert [r["epoch"] for r in rows] == ["0", "1", "2"]
    assert float(rows[0]["s_all"]) == 0.0 and float(rows[0]["s_mlp_b1_3"]) < 1e-25
    assert float(rows[-1]["s_all"]) > 1e-8                     # unconstrained steps on single images break it
    c0 = C.load_checkpoint(runs["tmp"] / "sym" / "ckpt_e000.pt")
    m = C.build_model("vit_test8", 10)
    m.load_state_dict(c0["model"])
    assert S.breaking_shares(m, S.canonical_structure(m))["s_all"] == 0.0
    summ = json.loads((runs["tmp"] / "results" / "TEST" / "vit_test8_UD_sym_init_s1" / "summary.json").read_text())
    assert summ["mode"] == "sym_init" and summ["breaking_share"]["n_rows"] == 3
    assert summ["params_free"] < summ["params_dense"] == C.count_params(m)


def test_tied_keeps_breaking_share_zero(runs):
    rows = _rows(runs["tmp"] / "tied" / "breaking_share.csv")
    assert len(rows) == 3
    for r in rows:
        assert float(r["s_all"]) == 0.0
        assert all(float(r[f"s_{k}_b{b}"]) < 1e-25 for k in ("fc1", "fc2", "attn") for b in (1, 2, 3))
    last = torch.load(runs["tmp"] / "tied" / "last.pt", weights_only=True)
    init = C.load_checkpoint(runs["tmp"] / "tied" / "ckpt_e000.pt")["model"]
    assert last["step"] == 10
    assert not torch.equal(init["blocks.0.mlp.fc1.weight"], last["model"]["blocks.0.mlp.fc1.weight"])
    m = C.build_model("vit_test8", 10)
    m.load_state_dict(last["model"])
    st = S.canonical_structure(m)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    S.project_(m, st, "param")
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())     # already exactly equivariant
    m64 = m.double()
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert S.ud_function_gap(m64, x)["logit_rel_gap"] < 1e-12
    meta = json.loads((runs["tmp"] / "tied" / "meta.json").read_text())
    assert meta["extra"]["status"] == "complete" and meta["extra"]["sym_mode"] == "tied"
    assert meta["extra"]["param_counts"]["free"] < meta["extra"]["param_counts"]["dense"]


def test_resume_reproduces_uninterrupted_tied_run(runs):
    A, B = runs["tmp"] / "tied", runs["tmp"] / "tied_b"
    assert runs["stops"][0]["stopped"] and runs["stops"][0]["step"] == 3
    assert runs["stops"][1]["stopped"] and runs["stops"][1]["step"] == 5
    la = torch.load(A / "last.pt", weights_only=True)
    lb = torch.load(B / "last.pt", weights_only=True)
    assert la["step"] == lb["step"] == 10
    assert all(torch.equal(la["model"][k], lb["model"][k]) for k in la["model"])
    for i in la["optimizer"]["state"]:
        for k in la["optimizer"]["state"][i]:
            assert torch.equal(la["optimizer"]["state"][i][k], lb["optimizer"]["state"][i][k])
    cols = ["epoch", "step", "train_loss", "train_top1", "grad_norm_mean"]
    assert [[r[c] for c in cols] for r in _rows(A / "train_log.csv")] == \
        [[r[c] for c in cols] for r in _rows(B / "train_log.csv")]
    assert [r["epoch"] for r in _rows(B / "breaking_share.csv")] == ["0", "1", "2"]
    assert [r["resumed"] for r in _rows(B / "train_log.csv")] == ["1", "0"]
    with pytest.raises(ValueError, match="different settings"):
        S.train_sym(_cfg(runs["tmp"], "tied", "tied_b", d_minus=10))


def test_refund_heal_is_exact_and_stays_exact(runs):
    R = runs["tmp"] / "refund"
    st, extra = S.load_structure(R / "structure.npz")
    assert st.kind == "fitted" and not st.A.is_perm and extra["fit_images"] == 20
    assert extra["source_ckpt"] == str(Path(runs["src_ckpt"]).resolve())
    assert np.allclose(st.A.matrix() @ np.ones(48), np.ones(48), atol=1e-10)         # A* 1 = 1
    rows = _rows(R / "breaking_share.csv")
    assert [r["epoch"] for r in rows] == ["0", "1"]
    assert all(float(r["s_all"]) < 1e-10 for r in rows)
    x = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    for name in ("ckpt_e000.pt", "ckpt_e001.pt"):                    # zero-shot refund, then healed
        m, info = C.load_checkpoint_model(R / name)
        assert info["sym_mode"] == "refund_heal"
        assert S.ud_function_gap(m, x)["logit_rel_gap"] < 1e-5      # float32 weights
        assert bool((m.blocks[1].norm1.weight == m.blocks[1].norm1.weight[0]).all())  # gamma stays a scalar
    summ = json.loads((runs["tmp"] / "results" / "TEST" / "vit_test8_UD_refund_heal_s1" / "summary.json").read_text())
    assert summ["params_free"] < summ["params_dense"]
    assert summ["structure_source"]["source_val"]["top1_orig"] >= 0
    src = C.load_checkpoint(runs["src_ckpt"])["model"]
    healed = C.load_checkpoint(R / "ckpt_e001.pt")["model"]
    assert not torch.equal(src["blocks.0.mlp.fc1.weight"], healed["blocks.0.mlp.fc1.weight"])


def test_checkpoints_read_by_adapter_and_measure_groups(runs, tmp_path):
    ck = runs["tmp"] / "tied" / "ckpt_e002.pt"
    ad = C.load_checkpoint_adapter(ck, "cpu")
    assert ad.grid_hw == (8, 8) and ad.info["condition"] == "UD" and ad.info["sym_mode"] == "tied"
    mg = _load_script("measure_groups")
    import os

    old = os.environ.get("IN100_CACHE_DIR")
    os.environ["IN100_CACHE_DIR"] = str(runs["tmp"] / "cache")
    try:
        m = mg.main(["--checkpoint", str(ck), "--arch", "controlled", "--device", "cpu", "--groups", "UD",
                     "--fit-images", "20", "--null-draws", "1", "--twin-null-draws", "0", "--summary-ranges", "1-3",
                     "--no-attn", "--workers", "1", "--experiment", "UP-04-test", "--out", str(tmp_path / "mg")])
    finally:
        if old is None:
            os.environ.pop("IN100_CACHE_DIR")
        else:
            os.environ["IN100_CACHE_DIR"] = old
    r = m["groups"]["UD"]["ranges"]["1-3"]
    assert r["s_mlp_mean"] < 1e-10                       # the fitted UD structure finds the exact symmetry
    assert r["twin_fraction_t50"] > 0.99                 # every hidden unit has its exact twin


def test_script_tiny_and_summaries(tmp_path):
    mod = _load_script("train_controlled_sym")
    out = mod.main(["--tiny", "--mode", "sym_init", "--epochs", "1", "--out-root", str(tmp_path),
                    "--results-root", str(tmp_path / "res")])
    run = tmp_path / "UP-02" / "vit_test8_UD_sym_init_s0"
    assert not out["stopped"]
    for f in ("ckpt_e000.pt", "ckpt_e001.pt", "last.pt", "train_log.csv", "val_acc.csv", "breaking_share.csv",
              "meta.json", "last_done.flag"):
        assert (run / f).exists(), f
    res = tmp_path / "res" / "UP-02" / "vit_test8_UD_sym_init_s0"
    for f in ("summary.json", "breaking_share.csv", "val_acc.csv", "meta.json"):
        assert (res / f).exists(), f
    meta = json.loads((res / "meta.json").read_text())
    assert meta["experiment"] == "UP-02" and meta["git"]["git_hash"] and meta["prereg"]["sha256"]
    s = mod.main(["--tiny", "--summarise", "--out-root", str(tmp_path), "--results-root", str(tmp_path / "res")])
    assert "UP-02" in s and (tmp_path / "res" / "UP-02" / "summary.md").exists()
