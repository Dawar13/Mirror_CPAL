"""Lockstep healing (mirror/final/heal/multistudent.py): divergence handling, (d) resume equal to an uninterrupted
run, hooks with checkpoints and evaluation requests, and (f) the learning-rate selection (equal to
mirror.heal.select_lr; lowest held-out loss, ties to the smaller rate). CPU, tiny random ViT in float64."""
import csv
import json

import numpy as np
import pytest
import torch

from mirror.heal import heal, select_lr

from mirror.final.heal.data_gpu import ConcatImages, LockstepData
from mirror.final.heal.multistudent import EvalQueue, choose_lr, select_lr_lockstep, train_lockstep

from test_fp_heal_common import Student, cfg, heal_data, mode_for, must09_teacher, same_state, tiny_adapter


@pytest.fixture(scope="module")
def ad():
    return tiny_adapter()


@pytest.fixture(scope="module")
def kinds(ad):
    return {k: Student(k, ad) for k in ("mirror", "lowrank")}


class Poison:
    """A constraint that writes NaN into the patch embedding on its n-th call (call 1 is the projection before
    training, call n + 1 follows optimiser step n), so the next loss is non-finite."""

    def __init__(self, after):
        self.n, self.after = 0, after

    def __call__(self, model):
        self.n += 1
        if self.n == self.after:
            with torch.no_grad():
                model.patch_embed.proj.bias.fill_(float("nan"))


# ---------------------------------------------------------------- data modes and the begin_step hook
def test_resident_streamed_and_split_data_give_the_same_run(ad, kinds):
    """The same job on the data resident on the device, streamed, and split over two arrays (ConcatImages)."""
    data = heal_data()
    c = cfg(range(ad.depth), max_steps=2)
    cut = 10
    split = ConcatImages([data.images[:cut], data.images[cut:]])
    rows = np.random.default_rng(5).permutation(len(data.images))[:7]
    assert np.array_equal(split[rows], data.images[rows]) and split.shape == data.images.shape
    models = []
    for mode, images in (("gpu", data.images), ("stream", data.images), ("stream", split), ("gpu", split)):
        s = kinds["lowrank"].lockstep("s", c.lr)
        ld = LockstepData(images, data.train_rows, "cpu", mode, heldout_rows=data.heldout_rows)
        assert ld.mode == mode and (ld.resident_bytes > 0) == (mode == "gpu")
        train_lockstep(ad, [s], ld, c)
        models.append(s.model)
    assert all(same_state(models[0], m) for m in models[1:])


def test_begin_step_hook_runs_before_every_step(ad, kinds):
    """The optional hook begin_step(step, total, model) (the full SVD-LLM factor schedule): called before every
    step with the 0-based step and the budget; a factor group it freezes does not move in those steps."""

    class Schedule:
        def __init__(self):
            self.calls = []

        def __call__(self, model):
            return None

        def begin_step(self, step, total, model):
            self.calls.append((step, total))
            on = step >= 2                                   # U frozen for steps 0 and 1, trained after
            for b in range(len(model.blocks)):
                p = model.blocks[b].mlp.fc1.U
                p.requires_grad_(on)
                if not on:
                    p.grad = None

    data = heal_data()
    c = cfg(range(ad.depth), max_steps=4)
    s = kinds["lowrank"].lockstep("s", 1e-2)
    sched = Schedule()
    s.constraints = list(s.constraints) + [sched]
    u0 = s.model.blocks[0].mlp.fc1.U.detach().clone()
    v0 = s.model.blocks[0].mlp.fc1.V.detach().clone()
    seen = []

    def hook(step, models):
        seen.append((step, models["s"].blocks[0].mlp.fc1.U.detach().clone()))

    train_lockstep(ad, [s], data, c, step_hook=hook)
    assert sched.calls == [(0, 4), (1, 4), (2, 4), (3, 4)]
    by_step = dict(seen)
    assert torch.equal(by_step[2], u0) and not torch.equal(by_step[3], u0)  # frozen in steps 0, 1; trained in 2
    assert not torch.equal(s.model.blocks[0].mlp.fc1.V, v0)


# ---------------------------------------------------------------- divergence
def test_divergence_flag_matches_the_host_check_and_others_continue(ad, kinds):
    """The student poisoned after its second step diverges at step 3 on every path, with heal()'s records; the
    device-side flag (fused AdamW's found_inf, no host synchronisation) gives the host check's result."""
    data = heal_data()
    c = cfg(range(ad.depth), max_steps=4)
    out = {}
    for name, kw in (("host", {}), ("fused_host", {"fused": True, "sync_each_step": True}),
                     ("fused_flag", {"fused": True, "sync_each_step": False})):
        good = kinds["mirror"].lockstep("good", 1e-2)
        bad = kinds["lowrank"].lockstep("bad", 1e-2)
        bad.constraints = list(bad.constraints) + [Poison(3)]
        out[name] = (train_lockstep(ad, [good, bad], data, c, **kw), good.model, bad.model)
    for name, (res, _, _) in out.items():
        b = res["students"]["bad"]
        assert b["diverged"] and b["steps_done"] == 2 and len(b["history"]) == 3, name
        assert not np.isfinite(b["history"][-1]) and b["heldout_final"] is None
        assert res["students"]["good"]["steps_done"] == 4 and res["students"]["good"]["heldout_final"] is not None
    (rh, gh, bh), (rf, gf, bf) = out["fused_host"], out["fused_flag"]
    assert rh["students"]["bad"]["history"][:2] == rf["students"]["bad"]["history"][:2]
    assert rh["students"]["good"]["history"] == rf["students"]["good"]["history"] and same_state(gh, gf)
    assert same_state(bh, bf)                     # the flagged student stayed at its state before the bad step
    # heal() records the same divergence for that student alone (the default, non-fused path on the CPU)
    m, cons, _ = kinds["lowrank"].fresh()
    ref = heal(must09_teacher(ad, kinds["lowrank"].spec, kinds["lowrank"].info), m, data, c, list(cons) + [Poison(3)])
    got = out["host"][0]["students"]["bad"]
    assert ref["diverged"] and ref["steps_done"] == 2 and ref["history"][:2] == got["history"][:2]
    assert not np.isfinite(ref["history"][2]) and same_state(m, out["host"][2])
    # and the healthy student equals its own single run
    ref_good, ref_g = kinds["mirror"].single(c, data)
    assert same_state(ref_good, out["host"][1]) and ref_g["history"] == out["host"][0]["students"]["good"]["history"]


# ---------------------------------------------------------------- (d) resume equals an uninterrupted run
def test_resume_equals_uninterrupted(ad, kinds, tmp_path):
    data = heal_data()
    c = cfg(range(ad.depth), max_steps=6, warmup_steps=2, lr=5e-3)

    def students(root):
        return [kinds["mirror"].lockstep("mirror", 5e-3, run_dir=root / "m"),
                kinds["lowrank"].lockstep("lowrank", 1e-2, run_dir=root / "l")]

    ref_s = students(tmp_path / "a")
    ref = train_lockstep(ad, ref_s, data, c, job_dir=tmp_path / "a" / "job")

    class Preempted(Exception):
        pass

    def hook(step, models):
        if step == 5:
            raise Preempted

    with pytest.raises(Preempted):
        train_lockstep(ad, students(tmp_path / "b"), data, c, job_dir=tmp_path / "b" / "job", state_every=2,
                       step_hook=hook)
    assert (tmp_path / "b" / "job" / "train_state.pt").exists()
    got_s = students(tmp_path / "b")
    got = train_lockstep(ad, got_s, data, c, job_dir=tmp_path / "b" / "job", state_every=2, resume=True)
    for a_, b_ in zip(ref_s, got_s):
        assert same_state(a_.model, b_.model), a_.name                        # bit for bit
        ra, rb = ref["students"][a_.name], got["students"][b_.name]
        assert ra["history"] == rb["history"] and ra["heldout_final"] == rb["heldout_final"]
        assert rb["resumed_from_step"] == 4
        with open(tmp_path / "b" / ("m" if a_.name == "mirror" else "l") / "train_log.csv", newline="") as fh:
            assert [int(r["step"]) for r in csv.DictReader(fh)] == [1, 2, 3, 4, 5, 6]
    # a state saved by another job (here another learning rate) is refused
    other = students(tmp_path / "b")
    other[0].lr = 1e-3
    with pytest.raises(ValueError, match="another job"):
        train_lockstep(ad, other, data, c, job_dir=tmp_path / "b" / "job", resume=True)


# ---------------------------------------------------------------- hooks, checkpoints and the evaluation queue
def test_hooks_write_checkpoints_and_queue_requests(ad, kinds, tmp_path):
    data = heal_data()
    c = cfg(range(ad.depth), max_steps=4)
    q = EvalQueue(tmp_path / "queue", n_query_full=20, n_query_partial=10)
    s = kinds["mirror"].lockstep("mirror", 1e-2, run_dir=tmp_path / "run", tags={"experiment": "T", "model": "tiny"})
    seen = []
    train_lockstep(ad, [s], data, c, hook_steps=[0, 2, 4], queue=q,
                   step_hook=lambda step, models: seen.append((step, sorted(models))))
    for step in (0, 2, 4):
        ck = tmp_path / "run" / "ckpt" / f"step{step}.pt"
        assert ck.is_file() and ck.with_suffix(".queued").is_file()
    payload = torch.load(tmp_path / "run" / "ckpt" / "step4.pt", weights_only=True)
    sd = s.model.state_dict()
    assert payload["step"] == 4 and all(torch.equal(v, sd[k]) for k, v in payload["state_dict"].items())
    step0 = torch.load(tmp_path / "run" / "ckpt" / "step0.pt", weights_only=True)["state_dict"]
    assert not all(torch.equal(v, payload["state_dict"][k]) for k, v in step0.items())   # step 0 is before training
    reqs = sorted((json.loads(p.read_text()) for p in (tmp_path / "queue").glob("*.json")), key=lambda r: r["step"])
    assert [r["step"] for r in reqs] == [0, 2, 4]
    assert [r["n_query"] for r in reqs] == [20, 10, 20] and reqs[-1]["passes"] == ["bf16", "fp32"]
    assert reqs[0]["passes"] == ["bf16"] and [r["priority"] for r in reqs] == [1, 0, 2]
    assert all(r["run_dir"] == str(tmp_path / "run") and {"run_dir", "step", "passes", "n_query", "priority"} <= set(r)
               for r in reqs)
    assert [st for st, _ in seen] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- (f) learning-rate selection
def test_choose_lr_lowest_loss_and_ties_to_smaller_rate():
    assert choose_lr([{"lr": 2e-4, "heldout_loss": 1.0}, {"lr": 1e-4, "heldout_loss": 2.0}]) == 2e-4
    assert choose_lr([{"lr": 2e-4, "heldout_loss": 1.0}, {"lr": 1e-4, "heldout_loss": 1.0}]) == 1e-4
    assert choose_lr([{"lr": 1e-4, "heldout_loss": 1.0, "diverged": True}, {"lr": 2e-4, "heldout_loss": 3.0}]) == 2e-4
    with pytest.raises(RuntimeError):
        choose_lr([{"lr": 1e-4, "heldout_loss": float("inf")}])


def test_lockstep_lr_selection_equals_select_lr(ad, kinds, tmp_path):
    data = heal_data()
    c = cfg(range(ad.depth))
    lrs = [1e-4, 1e-2, 3e-2]

    def maker(kind):
        def make():
            m, cons, _ = kinds[kind].fresh()
            return {"model": m, "constraints": cons, "mode": mode_for(kinds[kind].spec, kinds[kind].info, ad)}
        return make

    cands = {k: {"lrs": lrs[::-1], "make": maker(k), "out_dir": tmp_path / k} for k in ("mirror", "lowrank")}
    sel = select_lr_lockstep(ad, cands, data, c, budget_steps=3)
    for kind in ("mirror", "lowrank"):
        st = kinds[kind]

        def make_single():
            m, cons, _ = st.fresh()
            return m, cons

        ref = select_lr(must09_teacher(ad, st.spec, st.info), make_single, data, c, lrs=lrs, budget_steps=3)
        losses = [t["heldout_loss"] for t in sel[kind]["trials"]]
        assert sel[kind]["lrs"] == lrs and losses == [t["heldout_loss"] for t in ref["trials"]]
        assert sel[kind]["best_lr"] == ref["best_lr"] == lrs[int(np.argmin(losses))]
        assert len(set(losses)) == len(losses)                                # the rates really differ
        saved = json.loads((tmp_path / kind / "lr_selection.json").read_text())
        assert saved["best_lr"] == sel[kind]["best_lr"] and saved["budget_steps"] == 3 and saved["seed"] == 0
        assert (tmp_path / kind / "lr_1e-02" / "train_log.csv").is_file()
