"""Student registry (mirror/final/heal/students.py), load_student and the memory estimate (memory.py).

(g) every existing method builds on a tiny model with a fitted tiny SpanStructure (mirror.mirror_family.
fit_span_structure, as tests/test_must07.py), with MUST-09's targets; the registry's students and targets equal those
of scripts/must09_heal.py (build_compressed, make_trainable, distill_teacher) bit for bit; all of them train together
in one lockstep job and load back from their checkpoints. CPU, the float32 tiny ViT of must09_heal.py --tiny."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mirror import mirror_family as MF
from mirror.data import make_loader, synthetic_dataset
from mirror.heal import _Plan
from mirror.heal_structured import distill_teacher, make_trainable, write_spec
from mirror.variants import TinyProtocol

from mirror.final.heal import students as ST
from mirror.final.heal.memory import (MUST09_MEASURED_PEAK_GIB, PRESETS, estimate_heal_single, estimate_lockstep,
                                      estimate_table)
from mirror.final.heal.multistudent import LockstepStudent, train_lockstep
from mirror.final.heal.students import BuildContext, build_student, load_student, resolve_structure
from mirror.final.heal.targets import SharedTeacher

from test_fp_heal_common import cfg, heal_data, same_state

ROOT = Path(__file__).resolve().parents[1]
EXISTING = ("mirror_exact", "random_structure", "pair_random", "svd", "svdllm", "slicegpt", "flap", "original",
            "rot90_probe")


def load_must09():
    spec = importlib.util.spec_from_file_location("must09_heal_fp_test", ROOT / "scripts" / "must09_heal.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fp_students")
    M = load_must09()
    cfg0 = M.setup(M.parse_args(["--tiny", "--method", "mirror_exact"]))
    teacher, aspec = M.build_teacher(cfg0, torch.device("cpu"))      # TimmTinyAdapter(56, seed 0, float32)
    images, labels = synthetic_dataset(4, 4, 56, seed=3)
    st = MF.fit_span_structure(teacher, make_loader(images, labels, np.arange(16), 8), "cpu", model_name="tiny",
                               top_k=3, workers=1, pairings=False)
    path = MF.save_span_structure(tmp / "tiny_s0.npz", st)
    gate = ROOT / "results" / "MUST-07" / "gate_b.json"
    ctx = BuildContext(adapter=teacher, model_name="tiny", adapter_spec=aspec, device="cpu", structure=str(path),
                       gate_b=gate if gate.exists() else None, fit_loader=TinyProtocol().fit_loader)
    builds = {m: build_student(m, ctx) for m in EXISTING}
    return {"M": M, "teacher": teacher, "aspec": aspec, "path": path, "ctx": ctx, "builds": builds}


# ---------------------------------------------------------------- (g) the registry
def test_registry_builds_every_existing_method_with_must09_targets(env):
    b, L = env["builds"], env["teacher"].depth
    for m in ("mirror_exact", "random_structure", "pair_random"):
        t = b[m].targets
        assert (t.final, t.blocks, t.k) == ("symmetrised", "symmetrised", L), m
        assert b[m].spec["family"] == "mirror" and set(b[m].structure) >= {"A_star", "perms", "maps_after", "reg_perm"}
        assert np.array_equal(t.A, b[m].structure["A_star"])           # the student's own operator
    assert not np.allclose(b["random_structure"].targets.A, b["mirror_exact"].targets.A)
    assert np.array_equal(b["pair_random"].targets.A, b["mirror_exact"].targets.A)      # fitted operator ...
    assert not all(np.array_equal(p, q) for p, q in zip(b["pair_random"].structure["perms"],
                                                         b["mirror_exact"].structure["perms"]))   # ... random pairing
    for m in ("svd", "svdllm", "flap", "original"):
        assert (b[m].targets.final, b[m].targets.blocks) == ("symmetrised", "raw"), m
    assert b["slicegpt"].targets.blocks == "centred" and b["slicegpt"].spec["base"]["fold"]  # folded stream
    t = b["rot90_probe"].targets
    assert (t.final, t.blocks, t.input_transform) == ("raw", "raw", "rot90")
    for m in EXISTING:
        sb = b[m]
        assert sb.train_scope == "all" and sb.spec["fp_heal"]["method"] == m and sb.spec["adapter"] == env["aspec"]
        fresh, cons = sb.fresh()
        assert same_state(fresh, sb.model) and len(cons) == len(sb.constraints), m
        if m in ("original", "rot90_probe"):
            assert sb.params["model_params_frac"] == 1.0
        else:
            assert sb.info["conversion_check"]["passed"] and 0 < sb.params["model_params_frac"] < 1, m
    if env["ctx"].gate_b is not None:                                  # gate-B hyperparameters, as MUST-09
        assert b["svdllm"].spec["hparams"] == {"damp": 1e-06}
        assert b["slicegpt"].spec["hparams"] == {"basis": "per_block", "pca": "stream"}
    # trainable sets: MUST-09's for the original (norms and LayerScale frozen), every block parameter for rot90
    from mirror.heal import configure_trainable

    names = {}
    for m in ("original", "rot90_probe"):
        model, cons = b[m].fresh()
        names[m] = {n for n, _ in configure_trainable(model, list(range(L)), "all", cons)}
    assert not any(".norm1." in n or ".ls1." in n for n in names["original"])
    assert any(".norm1." in n for n in names["rot90_probe"]) and "patch_embed.proj.weight" in names["rot90_probe"]
    assert not any(n.startswith("norm.") or n.startswith("head.") for n in names["rot90_probe"])


@pytest.mark.parametrize("method", ["mirror_exact", "random_structure", "svdllm", "slicegpt"])
def test_registry_equals_must09_construction_and_targets(env, method):
    """The registry's student is scripts/must09_heal.py's (build_compressed + make_trainable) bit for bit, and its
    targets from the shared teacher pass are distill_teacher's (Gate P plumbing)."""
    M, teacher, L = env["M"], env["teacher"], env["teacher"].depth
    args = ["--tiny", "--method", method, "--span-structure", str(env["path"])]
    hp = env["ctx"].gate_b_hparams(method)
    if hp is not None:
        args += ["--hparams-json", json.dumps(hp)]
    c = M.setup(M.parse_args(args))
    fitted = M.load_span(c, teacher, False)
    comp, info, tstruct = M.build_compressed(c, teacher, fitted, hp, torch.device("cpu"))
    s0, cons0, spec = make_trainable(copy.deepcopy(comp), info, adapter=teacher,
                                     lowrank_factors=str(c.get("lowrank_factors", "balanced")))
    sb = env["builds"][method]
    assert same_state(s0, sb.model)
    assert [type(x).__name__ for x in cons0] == [type(x).__name__ for x in sb.constraints]
    vt = distill_teacher(teacher, spec, "sym", tstruct["A_star"], tstruct["reg_perm"])
    x = torch.randn(2, 3, 56, 56, generator=torch.Generator().manual_seed(3))
    cap = list(range(L))
    plan = _Plan(capture=frozenset(cap), first_grad=0, prefix_grad=True, share=False, checkpoint=False, n_prefix=1)
    want, _ = vt.distill_targets(x, plan)
    sh = SharedTeacher(teacher)
    got = sh.targets(sh.run(x, True, cap), sb.targets, cap)
    assert list(got) == list(want) and all(torch.equal(got[k], want[k]) for k in want)


def test_registry_students_train_together_and_load_back(env, tmp_path):
    teacher, L = env["teacher"], env["teacher"].depth
    data = heal_data()
    c = cfg(range(L), max_steps=2, lr=1e-3)
    studs = []
    for m, sb in env["builds"].items():
        model, cons = sb.fresh()
        rd = tmp_path / m
        sp = dict(sb.spec)
        sp.update(state_file="ckpt/step2.pt", seed=0)
        write_spec(rd, sp, sb.structure)
        studs.append(LockstepStudent(m, model, cons, sb.targets, 1e-3, run_dir=rd))
    res = train_lockstep(teacher, studs, data, c, hook_steps=[2])
    assert res["job"]["students"].keys() == set(EXISTING)
    x = torch.randn(3, 3, 56, 56, generator=torch.Generator().manual_seed(9))
    for s in studs:
        r = res["students"][s.name]
        assert r["steps_done"] == 2 and not r["diverged"] and all(np.isfinite(r["history"])), s.name
        ad2, model, info = load_student(tmp_path / s.name)
        assert info["step"] == 2 and info["method"] == s.name and not any(p.requires_grad for p in model.parameters())
        assert torch.equal(ad2.cls_feature(x, model), teacher.cls_feature(x, s.model)), s.name
        if s.name in ("mirror_exact", "random_structure", "pair_random"):
            assert info["structure"] is not None and np.array_equal(info["structure"]["A"],
                                                                    env["builds"][s.name].structure["A_star"])
    with pytest.raises(FileNotFoundError):
        load_student(tmp_path / "svd", step=1)


def test_external_methods_raise_a_clear_error_when_missing(env, monkeypatch):
    monkeypatch.setitem(ST.EXTERNAL_METHODS, "svdllm_full", "mirror.final.baselines.no_such_module_fp_test")
    with pytest.raises(ImportError, match="not importable"):
        build_student("svdllm_full", env["ctx"])
    with pytest.raises(ValueError, match="unknown method"):
        build_student("no_such_method", env["ctx"])


def test_logits_option_and_adopted_structure(env, tmp_path):
    sb = build_student("svd", env["ctx"], logits=True)
    assert sb.targets.logits and sb.spec["fp_heal"]["targets"]["logits"]
    with pytest.raises(ValueError, match="rot90"):
        build_student("rot90_probe", env["ctx"], logits=True)
    dec = tmp_path / "decision.json"
    dec.write_text(json.dumps({"adopted": True, "structure": str(env["path"])}))
    assert resolve_structure("adopted", dec) == Path(env["path"])
    choices = {"wspace": str(tmp_path / "w.npz"), "astar": str(env["path"])}
    for adopted, want in ((True, choices["wspace"]), (False, choices["astar"])):   # {"adopted": bool} (4.5a)
        dec.write_text(json.dumps({"adopted": adopted}))
        assert resolve_structure("adopted", dec, choices) == Path(want)
    with pytest.raises(ValueError, match="adopted_structures"):
        resolve_structure("adopted", dec)
    with pytest.raises(FileNotFoundError):
        resolve_structure("adopted", tmp_path / "missing.json")
    # a job's (or a student's) own choices win over the structure path named in the decision file, so DeiT-III,
    # DINOv2-L, the 9-block point and the A02 fit sets never pick up DINOv2-B's main structure
    other = {"wspace": str(tmp_path / "deit_w.npz"), "astar": str(tmp_path / "deit_a.npz")}
    dec.write_text(json.dumps({"adopted": True, "structure": str(env["path"])}))
    assert resolve_structure("adopted", dec, other) == Path(other["wspace"])
    assert resolve_structure({"adopted": other}, dec) == Path(other["wspace"])
    dec.write_text(json.dumps({"adopted": False, "structure": str(env["path"])}))
    assert resolve_structure({"adopted": other}, dec, choices) == Path(other["astar"])
    with pytest.raises(ValueError, match="gives no"):
        resolve_structure("adopted", dec, {"wspace": other["wspace"]})


# ---------------------------------------------------------------- memory estimate
def test_memory_estimate_matches_must09_and_scales():
    assert PRESETS["dinov2_vitb14_reg"].params() == 86_583_552                # design doc, Section 3
    assert PRESETS["dinov2_vitb14_reg"].tokens == 261 and PRESETS["deit3_b16"].tokens == 197
    est = estimate_heal_single("dinov2_vitb14_reg", 64)["total_gib"]
    assert abs(est - MUST09_MEASURED_PEAK_GIB["mirror_exact"]) / MUST09_MEASURED_PEAK_GIB["mirror_exact"] < 0.10
    a1, a2 = estimate_lockstep("dinov2_vitb14_reg", 64, 1), estimate_lockstep("dinov2_vitb14_reg", 64, 4)
    assert abs((a2["total_gib"] - a1["total_gib"]) - 3 * a1["per_student_persistent_gib"]) < 1e-9
    assert estimate_lockstep("dinov2_vitb14_reg", 128)["active_student_gib"] > 1.9 * a1["active_student_gib"]
    assert a1["estimate"] is True
    rows = estimate_table()
    assert {r["model"] for r in rows} == set(PRESETS) and {r["micro_batch"] for r in rows} == {64, 128}
