"""MUST-09: structured trainable students (mirror/heal_structured.py), training-state resume (mirror/heal.py) and
scripts/must09_heal.py --tiny. CPU only: tiny random timm ViT, synthetic images. The compressor library
(mirror/budget.py, mirror.baselines.compress) is written separately; until it exists the compressed models here are
built by hand in each kind's form (lowrank, width, slice, merge), and the mirror family through
mirror.heal_vexact.build_vexact_student."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mirror.attention import attn_dims, fold_attn, write_absorbed_attn
from mirror.baselines import _shrink_mlp
from mirror.heal import HealConfig, HealData, LowRankLinear, LowRankMLP, heal
from mirror.heal_structured import (IndexedLinear, SlicedLinear, adapter_spec, dense_weight, distill_teacher,
                                    feature_gap, load_healed_model, make_trainable, verify_rebuild, write_spec)
from mirror.heal_vexact import build_vexact_student, span_exactness
from mirror.m1 import apply_attention_svd, feature_invariance, random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing
from mirror.surgery import apply_svd_baseline, fold_mlp, write_absorbed_mlp

ROOT = Path(__file__).resolve().parents[1]
KINDS = ["lowrank", "width", "slice", "merge", "mirror"]


def load_must09():
    spec = importlib.util.spec_from_file_location("must09_heal_test", ROOT / "scripts" / "must09_heal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def span_structure(adapter, seed=0):
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
    c = dict(blocks=list(blocks), seed=0, lr=1e-2, max_steps=3, batch_size=8, micro_batch=4, crop_size=56,
             warmup_steps=1, calib_images=8, log_every=0, prefetch=0, train_scope="all")
    c.update(kw)
    return HealConfig(**c)


def images(n, seed=0, dtype=torch.float64):
    return torch.randn(n, 3, 56, 56, dtype=dtype, generator=torch.Generator().manual_seed(seed))


def compressed(kind, adapter, seed=0):
    """(compressed model, info) in the compressor contract's form, built by hand for each kind."""
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    H, dh = attn_dims(adapter.model.blocks[0])
    D = H * dh
    blocks = list(range(L))
    rng = np.random.default_rng(seed)
    if kind == "mirror":
        A, perms, maps = span_structure(adapter, seed)
        m, _, binfo = build_vexact_student(adapter, A, L, perms, maps)
        return m, {"kind": "mirror", "method": "mirror_exact", "blocks": blocks, "components": ["mlp", "attn"],
                   "A_star": A, "perms": perms, "maps_after": binfo["maps_after"], "reg_perm": np.arange(0)}
    m = copy.deepcopy(adapter.model)
    entries, tensors = [], {}
    if kind == "lowrank":
        ranks = [(10 + b, 12) for b in blocks]
        att = [(8, 9, 6, 7 + b) for b in blocks]
        apply_svd_baseline(m, blocks, ranks)
        apply_attention_svd(m, blocks, att)
        for b in blocks:
            entries += [{"block": b, "name": n, "kind": "lowrank", "rank": r} for n, r in zip(("fc1", "fc2"), ranks[b])]
            entries += [{"block": b, "name": n, "kind": "lowrank", "rank": r}
                        for n, r in zip(("wq", "wk", "wv", "wo"), att[b])]
    elif kind == "width":
        for b in blocks:
            blk = m.blocks[b]
            f = fold_mlp(blk)
            kept = np.sort(rng.choice(h, h // 2, replace=False))
            drop = np.setdiff1d(np.arange(h), kept)
            if b % 2 == 0:                                  # full size, pruned fc2 columns zeroed
                W2 = f["W2"].clone()
                W2[:, torch.as_tensor(drop)] = 0
                write_absorbed_mlp(blk, f["W1"], f["b1"], W2, f["b2"])
            else:                                           # already shrunk (as baselines.apply_baseline prune_act)
                k = torch.as_tensor(kept)
                _shrink_mlp(blk, len(kept))
                write_absorbed_mlp(blk, f["W1"][k], f["b1"][k], f["W2"][:, k], f["b2"])
            entries += [{"block": b, "name": n, "kind": "width", "kept": kept.tolist()} for n in ("fc1", "fc2")]
            fa = fold_attn(blk)                             # remove one head (q, k, v rows and Wo columns)
            gone = np.arange(dh) + (b % H) * dh
            ch = np.setdiff1d(np.arange(D), gone)
            g = torch.as_tensor(gone)
            for key in ("Wq", "Wk", "Wv"):
                fa[key][g] = 0
            for key in ("bq", "bk", "bv"):
                fa[key][g] = 0
            fa["Wo"][:, g] = 0
            write_absorbed_attn(blk, **fa)
            entries += [{"block": b, "name": n, "kind": "width", "kept": ch.tolist()} for n in ("wq", "wk", "wv", "wo")]
    elif kind == "slice":
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        tensors["Q"] = Q
        n = 40
        P = torch.as_tensor(Q[:, :n] @ Q[:, :n].T)
        for b in blocks:
            blk = m.blocks[b]
            f = fold_mlp(blk)
            write_absorbed_mlp(blk, f["W1"] @ P, f["b1"], P @ f["W2"], P @ f["b2"])
            fa = fold_attn(blk)
            for key in ("Wq", "Wk", "Wv"):
                fa[key] = fa[key] @ P
            fa["Wo"], fa["bo"] = P @ fa["Wo"], P @ fa["bo"]
            write_absorbed_attn(blk, **fa)
            entries += [{"block": b, "name": "fc1", "kind": "slice", "d_in": n, "d_out": h, "basis_key": "Q"},
                        {"block": b, "name": "fc2", "kind": "slice", "d_in": h, "d_out": n, "basis_key": "Q"}]
            entries += [{"block": b, "name": nm, "kind": "slice", "d_in": n, "d_out": D, "basis_key": "Q"}
                        for nm in ("wq", "wk", "wv")]
            entries.append({"block": b, "name": "wo", "kind": "slice", "d_in": D, "d_out": n, "basis_key": "Q"})
    elif kind == "merge":
        groups = [[2 * i, 2 * i + 1] for i in range(32)] + [[j] for j in range(64, h)]
        for b in blocks:
            blk = m.blocks[b]
            f = fold_mlp(blk)
            W1, b1 = f["W1"].clone(), f["b1"].clone()
            for gr in groups:
                W1[gr] = W1[gr].mean(0)
                b1[gr] = b1[gr].mean()
            write_absorbed_mlp(blk, W1, b1, f["W2"], f["b2"])
            entries += [{"block": b, "name": n, "kind": "merge", "groups": groups} for n in ("fc1", "fc2")]
    else:
        raise ValueError(kind)
    return m, {"kind": "compressed", "method": kind, "blocks": blocks, "per_matrix": entries, "tensors": tensors}


def teacher_for(adapter, spec, info, targets="sym"):
    A = info["A_star"] if spec["family"] == "mirror" else span_structure(adapter)[0]
    return distill_teacher(adapter, spec, targets, A, None)


# ---------------------------------------------------------------- structure survives training
@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_stays_structured_after_optimiser_steps(tiny64, kind):
    comp, info = compressed(kind, tiny64)
    student, cons, spec = make_trainable(copy.deepcopy(comp), info, adapter=tiny64)
    x = images(3)
    assert feature_gap(tiny64, comp, student, x)["rel_max"] < 1e-10            # the same function as the compressor's
    verify_rebuild(tiny64, spec, student)
    before = {k: v.clone() for k, v in student.state_dict().items()}
    res = heal(teacher_for(tiny64, spec, info), student, heal_data(), cfg(range(tiny64.depth)), cons, None)
    assert res["steps_done"] == 3 and not res["diverged"]
    after = student.state_dict()
    names = set(res["trainable_names"])
    # what trains: the embeddings and the converted blocks' structured parameters; norms, LayerScale, head frozen
    assert {"patch_embed.proj.weight", "cls_token", "pos_embed"} <= names
    assert not any(n.startswith(("norm.", "head.")) or ".norm1." in n or ".norm2." in n or ".ls1." in n or ".ls2." in n
                   for n in names), sorted(names)
    for k, v in after.items():
        if k in names:
            continue
        assert torch.equal(v, before[k]), k                                    # frozen tensors never move
    assert not torch.equal(after["patch_embed.proj.weight"], before["patch_embed.proj.weight"])
    L = tiny64.depth
    if kind == "mirror":
        ex = span_exactness(tiny64, student, [x], info["A_star"], L, None)
        assert ex["max_rel_inside"] < 1e-10
        assert feature_invariance(tiny64, student, x)["rel_max"] < 1e-10
        return
    chk = cons[0].check(student)
    assert chk["ok"], chk
    for b in range(L):
        blk = student.blocks[b]
        if kind == "lowrank":
            mods = [blk.mlp.fc1, blk.mlp.fc2, blk.attn.qkv.q, blk.attn.qkv.k, blk.attn.qkv.v, blk.attn.proj]
            targets = [10 + b, 12, 8, 9, 6, 7 + b]
            assert all(isinstance(mm, LowRankLinear) for mm in mods)
            assert [mm.rank64() for mm in mods] == targets                    # float64 rank from the factors
            assert not torch.equal(after[f"blocks.{b}.mlp.fc1.U"], before[f"blocks.{b}.mlp.fc1.U"])
        elif kind == "width":
            assert blk.mlp.fc1.out_features == blk.mlp.fc2.in_features == tiny64.mlp_hidden // 2
            H, dh = attn_dims(tiny64.model.blocks[0])
            gone = np.arange(dh) + (b % H) * dh
            y = blk.attn.qkv(torch.randn(2, 5, tiny64.embed_dim, dtype=torch.float64))
            D = H * dh
            for i in range(3):                                                 # removed head: q, k, v exactly 0
                assert y[..., i * D + gone].abs().max().item() == 0.0
            z = torch.randn(2, 5, D, dtype=torch.float64)
            z2 = z.clone()
            z2[..., gone] = 7.0                                                # Wo ignores the removed head
            assert torch.equal(blk.attn.proj(z), blk.attn.proj(z2))
            assert isinstance(blk.attn.proj, IndexedLinear)
        elif kind == "slice":
            Q = torch.as_tensor(info["tensors"]["Q"][:, :40])
            Pc = torch.eye(tiny64.embed_dim, dtype=torch.float64) - Q @ Q.T
            W1, _ = dense_weight(blk.mlp.fc1)
            W2, b2 = dense_weight(blk.mlp.fc2)
            Wo, bo = dense_weight(blk.attn.proj)
            Wqkv, _ = dense_weight(blk.attn.qkv)
            assert (W1 @ Pc).abs().max().item() < 1e-12 and (Wqkv @ Pc).abs().max().item() < 1e-12
            assert (Pc @ W2).abs().max().item() < 1e-12 and (Pc @ b2).abs().max().item() < 1e-12
            assert (Pc @ Wo).abs().max().item() < 1e-12 and (Pc @ bo).abs().max().item() < 1e-12
            assert isinstance(blk.mlp.fc1, SlicedLinear) and torch.equal(blk.mlp.fc1.basis_in, Q)
        elif kind == "merge":
            assert blk.mlp.fc1.out_features == 32 + tiny64.mlp_hidden - 64


def test_conversions_have_teeth(tiny64):
    """A lowrank entry whose rank is below the compressed weight's rank, pruned units whose fc2 columns are not zero,
    and fc1 rows of a merge group that are not tied are all refused (they would change the compressed function)."""
    comp, info = compressed("lowrank", tiny64)
    bad = copy.deepcopy(info)
    bad["per_matrix"][0]["rank"] -= 3
    with pytest.raises(ValueError, match="differs from the compressed weight"):
        make_trainable(copy.deepcopy(comp), bad, adapter=tiny64)
    comp, info = compressed("width", tiny64)
    untouched = copy.deepcopy(tiny64.model)                                  # full size, fc2 columns not zeroed
    with pytest.raises(ValueError, match="not zero"):
        make_trainable(untouched, {**info, "per_matrix": [e for e in info["per_matrix"] if e["block"] == 0]},
                       adapter=tiny64)
    comp, info = compressed("merge", tiny64)
    with pytest.raises(ValueError, match="not tied"):
        make_trainable(copy.deepcopy(tiny64.model), info, adapter=tiny64)


def test_low_rank_mlp_check_reads_the_rank_in_float64():
    """The known E3 issue (results/STATUS.md): the rank check of a float32 student read full rank."""
    from mirror.adapters import TimmTinyAdapter

    ad = TimmTinyAdapter(img_size=56, seed=0)                                 # float32
    m = copy.deepcopy(ad.model)
    blocks, ranks = [0, 2], [(10, 12), (7, 9)]
    apply_svd_baseline(m, blocks, ranks)
    con = LowRankMLP(blocks, ranks)
    con.prepare(m)
    assert con.check(m)["numerical_ranks"] == [list(r) for r in ranks]
    full = int(torch.linalg.matrix_rank(m.blocks[0].mlp.fc1.weight.detach().double()))
    assert full > 10                                                          # the float32 product reads higher


# ---------------------------------------------------------------- resume
@pytest.mark.parametrize("kind", ["lowrank", "mirror"])
def test_resume_after_interruption_gives_the_uninterrupted_final_state(tiny64, tmp_path, kind):
    comp, info = compressed(kind, tiny64)

    def fresh():
        return make_trainable(copy.deepcopy(comp), info, adapter=tiny64)

    s_ref, cons_ref, spec = fresh()
    teacher = teacher_for(tiny64, spec, info)
    c = cfg(range(tiny64.depth), lr=5e-3, max_steps=6, warmup_steps=2)
    data = heal_data()
    ref = heal(teacher, s_ref, data, c, cons_ref, tmp_path / "ref")

    class Preempted(Exception):
        pass

    def hook(step, model):
        if step == 5:
            raise Preempted

    state = tmp_path / "train_state.pt"
    s1, cons1, _ = fresh()
    with pytest.raises(Preempted):
        heal(teacher, s1, data, c, cons1, tmp_path / "run", step_hook=hook, hook_steps=[5], state_path=state,
             state_every=2)
    assert state.exists()
    s2, cons2, _ = fresh()
    res = heal(teacher, s2, data, c, cons2, tmp_path / "run", state_path=state, state_every=2, resume=True)
    assert res["resumed_from_step"] == 4 and res["steps_done"] == 6
    for k, v in s_ref.state_dict().items():
        assert torch.equal(v, s2.state_dict()[k]), k                          # bit for bit
    assert res["heldout_final"] == ref["heldout_final"] and res["history"] == ref["history"]
    with open(tmp_path / "run" / "train_log.csv", newline="") as fh:
        steps = [int(r["step"]) for r in csv.DictReader(fh)]
    assert steps == [1, 2, 3, 4, 5, 6]
    # a state saved by another run (here: another learning rate) is refused
    from dataclasses import replace

    s4, cons4, _ = fresh()
    with pytest.raises(ValueError, match="another run"):
        heal(teacher, s4, data, replace(c, lr=1e-3), cons4, None, state_path=state, resume=True)


# ---------------------------------------------------------------- load_healed_model
@pytest.mark.parametrize("kind", KINDS)
def test_load_healed_model_rebuilds_identical_outputs(tiny64, tmp_path, kind):
    comp, info = compressed(kind, tiny64)
    student, cons, spec = make_trainable(copy.deepcopy(comp), info, adapter=tiny64)
    run = tmp_path / kind
    c = cfg(range(tiny64.depth), max_steps=2, checkpoint_path=str(run / "healed_state.pt"))
    heal(teacher_for(tiny64, spec, info), student, heal_data(), c, cons, run / "final")
    spec.update(adapter=adapter_spec("tiny", img_size=56, seed=0, dtype="float64"), state_file="healed_state.pt")
    struct = ({k: info[k] for k in ("A_star", "perms", "maps_after", "reg_perm")} if kind == "mirror" else None)
    write_spec(run, spec, struct)
    adapter, model, linfo = load_healed_model(run)                            # adapter rebuilt from the spec
    x = images(4, seed=5)
    assert torch.equal(adapter.cls_feature(x, model), tiny64.cls_feature(x, student))
    assert torch.equal(adapter.forward_tokens(x, model), tiny64.forward_tokens(x, student))
    for name, chk in linfo["checks"].items():
        if "max_rel_violation" in chk:
            assert chk["max_rel_violation"] < 1e-10, name
        if "ok" in chk:
            assert chk["ok"], name
    assert not any(p.requires_grad for p in model.parameters())
    # parameter counts for the Tier-2 tables
    assert linfo["model_params"] == sum(p.numel() for p in model.parameters())
    assert 0 < linfo["converted_params_kept"] < linfo["converted_params_dense"]
    assert linfo["model_params_structured"] == (linfo["model_params_original"] - linfo["converted_params_dense"]
                                                + linfo["converted_params_kept"])
    assert linfo["method"] == info["method"] and "seed" in linfo
    if kind == "mirror":
        assert np.array_equal(linfo["structure"]["A"], np.asarray(info["A_star"]))
        assert linfo["model_params"] > linfo["model_params_structured"]          # stored dense, counted free
    else:
        assert linfo["structure"] is None


# ---------------------------------------------------------------- scripts/must09_heal.py --tiny
def test_cli_tiny_mirror_curves_measures_and_rebuild(tmp_path):
    M = load_must09()
    runs, res = tmp_path / "runs", tmp_path / "results"
    m = M.main(["--tiny", "--method", "mirror_exact", "--mirror-builder", "shim", "--out-root", str(runs),
                "--results-root", str(res)])
    run = runs / "tiny" / "mirror_exact" / "s0"
    for f in ("spec.json", "structure.npz", "healed_state.pt", "curves.csv", "metrics.json", "meta.json",
              "lr_selection/lr_selection.json", "final/metrics.json", "final/train_log.csv", "knn/step0_fp32.npz",
              "knn/step4_fp32.npz"):
        assert (run / f).is_file(), f
    assert not (run / "train_state.pt").exists()                               # removed after a finished run
    with open(run / "curves.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["step"]) for r in rows] == [0, 1, 2, 4]                       # curve steps (tiny budget: 4 steps)
    assert [r["full_validation"] for r in rows] == ["True", "False", "False", "True"]
    assert [int(r["n_query"]) for r in rows] == [20, 10, 10, 20]                # --curve-queries subset in between
    assert m["complete"] and m["targets"] == "sym" and m["family"] == "mirror"
    assert m["knn_fp32"]["flip_consistency"] == 100.0 and m["knn_fp32"]["n_query"] == 20
    assert m["exactness"]["pass"] and m["conversion_check"]["passed"]
    assert m["cost"]["params_frac"] < 1 and m["cost"]["gflops_frac"] < 1
    meta = json.loads((run / "meta.json").read_text())
    assert meta["experiment"] == "MUST-09" and meta["seeds"]["heal"] == 0 and meta["extra"]["targets"] == "sym"
    with open(res / "summary.csv", newline="") as fh:
        srows = list(csv.DictReader(fh))
    assert len(srows) == 1 and srows[0]["method"] == "mirror_exact" and float(srows[0]["flip_consistency_fp32"]) == 100.0
    assert {"knn_top1_step0", "knn_top1_step4", "gflops", "params_model", "peak_train_memory_gb"} <= set(srows[0])
    # the rebuilt model gives the recorded fp32 k-NN and is exactly invariant
    adapter, model, info = load_healed_model(run)
    payload = torch.load(run / "healed_state.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(v, payload[k]) for k, v in model.state_dict().items())
    assert info["checks"]["0:MirrorMLPConstraint"]["max_rel_violation"] < 1e-6
    x = images(3, dtype=torch.float32)
    assert feature_invariance(adapter, model, x)["rel_max"] < 1e-5
    # a second call finds the finished run and does not heal again
    again = M.main(["--tiny", "--method", "mirror_exact", "--mirror-builder", "shim", "--out-root", str(runs),
                    "--results-root", str(res)])
    assert again["timing"] == m["timing"]
    # latency mode on the rebuilt model (CPU, eager)
    lat = M.main(["--tiny", "--method", "mirror_exact", "--latency-only", "--with-original", "--out-root", str(runs),
                  "--results-root", str(res)])
    assert {r["variant"] for r in lat["rows"]} == {"original", "healed_as_stored", "healed_even_odd"}
    assert lat["even_odd_check"]["rel_max"] < 1e-5 and (res / "latency.csv").is_file()
    # seed 1 reuses seed 0's learning rate (PREREG_NEXT_PHASE.md, MUST-09) and runs no trials of its own
    m1 = M.main(["--tiny", "--method", "mirror_exact", "--mirror-builder", "shim", "--seed", "1", "--lr-from", str(run),
                 "--out-root", str(runs), "--results-root", str(res)])
    assert m1["lr"] == m["lr"] and m1["lr_selection"]["reused_from"].endswith("metrics.json")
    assert not (runs / "tiny" / "mirror_exact" / "s1" / "lr_selection" / "lr_selection.json").exists()


def test_cli_resume_after_interruption(tmp_path, monkeypatch):
    """Interrupt the full run (after the training state at step 2, in the curve point at step 3), rerun the same
    command: finished learning-rate trials are reused, training continues from the saved state, and the healed state,
    the curve and the summary equal those of an uninterrupted run."""
    M = load_must09()
    base = ["--tiny", "--method", "svd", "--compressor-builder", "shim", "--state-every", "2", "--curve-steps", "0,3"]

    def args(root):
        return base + ["--out-root", str(root / "runs"), "--results-root", str(root / "results")]

    ref = M.main(args(tmp_path / "a"))
    orig = M.CurveLogger.__call__

    def preempt(self, step, model):
        if step == 3 and not self.done(3):
            raise KeyboardInterrupt
        return orig(self, step, model)

    monkeypatch.setattr(M.CurveLogger, "__call__", preempt)
    with pytest.raises(KeyboardInterrupt):
        M.main(args(tmp_path / "b"))
    run_b = tmp_path / "b" / "runs" / "tiny" / "svd" / "s0"
    assert (run_b / "train_state.pt").exists() and not (run_b / "metrics.json").exists()
    monkeypatch.setattr(M.CurveLogger, "__call__", orig)
    got = M.main(args(tmp_path / "b"))
    run_a = tmp_path / "a" / "runs" / "tiny" / "svd" / "s0"
    sa = torch.load(run_a / "healed_state.pt", weights_only=True)["state_dict"]
    sb = torch.load(run_b / "healed_state.pt", weights_only=True)["state_dict"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert [r["knn_top1"] for r in got["curve"]] == [r["knn_top1"] for r in ref["curve"]]
    assert got["knn_fp32"]["top1"] == ref["knn_fp32"]["top1"] and got["lr"] == ref["lr"]
    sel = json.loads((run_b / "lr_selection" / "lr_selection.json").read_text())
    assert all(t.get("reused") for t in sel["trials"])                         # trials finished before the crash
    fin = json.loads((run_b / "final" / "metrics.json").read_text())
    assert fin["resumed_from_step"] == 2
    # the SVD student: ranks exact in float64, loads back, structure checks pass
    adapter, model, info = load_healed_model(run_b)
    chk = info["checks"]["0:StructuredTrainable"]
    assert chk["ok"] and chk["numerical_ranks"] == chk["target_ranks"]
    assert got["compressor"]["source"].startswith("SHIM")


@pytest.mark.parametrize("method", ["mirror_exact", "random_structure"])
def test_mirror_family_builder_matches_the_e3_shim(method):
    """mirror.mirror_family.build_mirror_variant (written separately) and the E3 code path
    (mirror.heal_vexact.build_vexact_student) build the same student from the same structure and seed."""
    pytest.importorskip("mirror.mirror_family")
    M = load_must09()
    built = {}
    for builder in ("family", "shim"):
        cfg = M.setup(M.parse_args(["--tiny", "--method", method, "--mirror-builder", builder]))
        teacher, _ = M.build_teacher(cfg, torch.device("cpu"))
        model, info, _ = M.build_compressed(cfg, teacher, M.load_span(cfg, teacher, False), None, torch.device("cpu"))
        built[builder] = (model.state_dict(), info)
    (sa, ia), (sb, ib) = built["family"], built["shim"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert np.abs(np.asarray(ia["A_star"]) - np.asarray(ib["A_star"])).max() < 1e-12     # cleaned twice: rounding only
    assert all(np.array_equal(p, q) for p, q in zip(ia["perms"], ib["perms"]))


def test_library_compressor_path_when_available(tmp_path):
    """Runs only once mirror/budget.py and mirror.baselines.METHODS_V2 exist (written separately)."""
    budget = pytest.importorskip("mirror.budget")
    import mirror.baselines as B

    if not hasattr(B, "METHODS_V2") or not hasattr(B, "compress") or not hasattr(budget, "mirror_exact_budget"):
        pytest.skip("compressor library not complete yet")
    M = load_must09()
    m = M.main(["--tiny", "--method", "svd", "--compressor-builder", "library", "--lr", "1e-4",
                "--out-root", str(tmp_path / "runs"), "--results-root", str(tmp_path / "results")])
    assert m["complete"] and m["conversion_check"]["passed"] and m["compressor"]["source"] == "mirror.baselines.compress"


def test_from_gate_b_picks_method_and_hparams(tmp_path):
    M = load_must09()
    gate = {"complete": True, "missing_methods": [], "best": "svdllm", "second_best": "flap",
            "ranking": [{"rank": 2, "method": "flap", "hparams": {"allocation": "global"}},
                        {"rank": 1, "method": "svdllm", "hparams": {"damp": 1e-6}}]}
    path = tmp_path / "gate_b.json"
    path.write_text(json.dumps(gate))
    a = M.parse_args(["--from-gate-b", str(path)])
    assert a.method == "svdllm" and json.loads(a.hparams_json) == {"damp": 1e-6}
    a = M.parse_args(["--from-gate-b", str(path), "--gate-rank", "2", "--seed", "1"])
    assert a.method == "flap" and json.loads(a.hparams_json) == {"allocation": "global"}
    with pytest.raises(SystemExit):
        M.parse_args(["--from-gate-b", str(path), "--method", "svd"])          # disagrees with the gate
    path.write_text(json.dumps({**gate, "complete": False, "missing_methods": ["zipit"]}))
    with pytest.raises(SystemExit):
        M.parse_args(["--from-gate-b", str(path)])
    assert M.parse_args(["--from-gate-b", str(path), "--allow-incomplete-gate"]).method == "svdllm"
    with pytest.raises(SystemExit):
        M.parse_args([])                                                        # neither --method nor the gate
