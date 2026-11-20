#!/usr/bin/env python
"""Write the Phase G job files of the final phase from one place (FINAL_PHASE_PLAN.md v2, Sections 4, 8 and 9):

    jobs/heal/<job>.yaml          one lockstep healing job each (scripts/fp_heal.py --spec)
    jobs/a12/<model>_<arm>.json   A12 systems-table manifests (scripts/fp_a12_bench.py --manifest)
    jobs/final_phase.yaml         the runner manifest (scripts/fp_runner.py; format in jobs/final_phase.schema.md)

    python scripts/fp_make_jobs.py            # rewrite every file
    python scripts/fp_runner.py check jobs/final_phase.yaml

Arms and run folders follow docs/final_phase_design.md 4.5a. GPU memory per job comes from
mirror/final/heal/memory.py (micro-batch 128 for ViT-B-size models, 64 for DINOv2-L; images streamed from host
memory, so heal jobs do not hold the 20 GB heal set on the GPU), plus a margin. Heal-units follow the plan's
Section 9 (one DINOv2-B heal of 780 steps = 1; DINOv2-L = 3; an A11 extended heal = 3; a 200-step learning-rate trial
= 0.25 per rate; a lockstep job = the sum over its students). Cut ranks follow Section 9 (jobs/final_phase.schema.md).
Jobs whose structure depends on the A01 decision use `structure: adopted` with per-model choices
(mirror/final/heal/students.py resolve_structure) or scripts/fp_main_arm.py inside their command.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
B, DT, LM = "dinov2_vitb14_reg", "deit3_b16", "dinov2_vitl14_reg"
CANON = {B: f"results/MUST-07/structure/{B}_s0.npz", DT: f"results/MUST-07/structure/{DT}_s0.npz",
         LM: f"results/MUST-07/structure/{LM}_s0.npz"}
WSPACE = {B: f"results/A01/structure/{B}_wspace_matched.npz",
          DT: f"results/A01/{DT}/structure/{DT}_wspace_matched.npz",
          LM: f"results/A01/{LM}/structure/{LM}_wspace_matched.npz"}
K9_WSPACE = f"results/A01/k9/structure/{B}_wspace_matched_k9.npz"
A02_ROLES = ("fit1", "fit2", "coco")


def a02_fit(role):
    return f"results/A02/fits/{B}_{role}.npz"


def a02_wspace(role):
    return f"results/A02/wspace_{role}/structure/{B}_wspace_matched.npz"


LR1, LRSEL = 1.0e-4, {"lrs": [1.0e-4, 2.0e-4]}
TRIAL_UNITS = 2 * 0.25          # one 200-step trial per rate, two rates
GPU_GB = 141.0                  # H200 (the runner reads the real total from nvidia-smi)

BASE = {
    "model": B, "micro_batch": 128, "compile": False, "out_root": "runs/final", "queue_dir": "runs/final/_queue",
    "structure": CANON[B], "adopted_structures": {"wspace": WSPACE[B], "astar": CANON[B]},
    "gate_b": "results/MUST-07/gate_b.json", "random_seed": 1, "cache_dir": "cache",
    "data": {"mode": "stream"}, "state_every": 100,
    "eval": {"n_query_full": 50000, "n_query_partial": 10000, "fp32_last": False},
    "lr_selection": {"budget_steps": 200},
}
EXT_DATA = {"sources": [{"images": "cache/heal256_uint8.npy", "index": "cache/train_index.npz", "key": "heal_rows"},
                        {"images": "cache/heal_ext256_uint8.npy", "index": "cache/heal_ext_index.npz", "key": "rows"}],
            "heldout_key": "healval_rows", "heldout_source": 0, "mode": "stream"}


def per_student_gb(model):
    return {B: 1.29, DT: 1.29, LM: 4.54}[model]


def job_gb(model, n_students, micro_batch, lowrank=0, extra=0.0):
    """mirror/final/heal/memory.py estimate (shared teacher outputs + one active student + n students' parameters,
    gradients and AdamW states), plus about 1.4 GiB per low-rank student (MUST-09 measured), CUDA context and a
    margin of 15%."""
    active = {(B, 64): 6.86, (B, 128): 13.56, (DT, 64): 5.22, (DT, 128): 10.27, (LM, 64): 18.31, (LM, 128): 36.04}
    shared = {(B, 64): 2.31, (B, 128): 4.24, (DT, 64): 1.85, (DT, 128): 3.33, (LM, 64): 6.03, (LM, 128): 10.88}
    gb = shared[(model, micro_batch)] + active[(model, micro_batch)] + n_students * per_student_gb(model)
    gb += 1.4 * lowrank + extra + 1.0
    return round(gb * 1.15 + 2.0, 0)


def heal_spec(name, experiment, group, seed, students, model=B, hook_steps=(0, 100, 300), steps=None,
              micro_batch=None, data=None, extra=None, lowrank=0, extra_gb=0.0):
    spec = copy.deepcopy(BASE)
    spec.update(experiment=experiment, group=group, seed=int(seed), model=model, students=students,
                hook_steps=list(hook_steps), job=name)
    if model != B:
        spec["structure"] = CANON[model]
        spec["adopted_structures"] = {"wspace": WSPACE[model], "astar": CANON[model]}
    if steps is not None:
        spec["steps"] = int(steps)
    mb = int(micro_batch or (64 if model == LM else 128))
    spec["micro_batch"] = mb
    if data is not None:
        spec["data"] = data
    spec.update(extra or {})
    gb = job_gb(model, len(students), mb, lowrank, extra_gb)
    spec["gpu_frac"] = round(min(0.95, (gb + 6.0) / GPU_GB), 2)
    return spec, gb


def units_of(students, model=B, steps=780):
    per = {B: 1.0, DT: 1.0, LM: 3.0}[model] * steps / 780.0
    u = 0.0
    for s in students:
        u += per
        if "lr_select" in s:
            u += TRIAL_UNITS * {B: 1.0, DT: 1.0, LM: 3.0}[model]
    return round(u, 2)


def done_of(spec):
    """The DONE file of the lockstep job folder (scripts/fp_heal.py writes it last)."""
    return f"{spec['out_root']}/{spec['group']}/{spec['model']}/_jobs/{spec['job']}/s{spec['seed']}/DONE"


def run_dir(group, model, folder, seed):
    return f"runs/final/{group}/{model}/{folder}/s{seed}"


# ----------------------------------------------------------------------------------------------
# Heal jobs
# ----------------------------------------------------------------------------------------------
def heal_jobs():
    J = []   # (job id, wave, spec, gpu_gb, units, cut_rank, deps, when, priority, note)

    def add(jid, wave, spec, gb, units, cut="never_cut", deps=(), when=None, prio=50, after=(), note=""):
        J.append(dict(id=jid, wave=wave, spec=spec, gb=gb, units=units, cut=cut, deps=list(deps), when=when,
                      prio=prio, after=list(after), note=note))

    ready = ["w0_timing", "w0_data", "w0_models"]
    # W1: Gate P (current A*, SVD-LLM; 3 seeds) and the A*_w heals (seeds 0, 1), packed by seed
    # Gate P runs first and alone (PREREG_FINAL_PHASE.md 2.2: no other heal before it passes); it does not wait for
    # the CPU-bound A*_w fit. The float32 k-NN pass (about 200k float32 forwards) runs on seed 0 of each Mirror arm.
    for s in (0, 1, 2):
        st = [{"method": "mirror_exact", "variant": "astar", "lr": LR1, "fp32": s == 0}, {"method": "svdllm", "lr": LR1}]
        spec, gb = heal_spec(f"gate_s{s}", "A01", "main", s, st, hook_steps=(0, 100, 300) if s == 0 else (),
                             lowrank=1)
        add(f"heal_gate_s{s}", "W1", spec, gb, units_of(st), deps=ready, prio=80,
            note="Gate P (FINAL_PHASE_PLAN.md A01); never cut")
    # A01: the A*_w heals, seeds 0 and 1, once the fit is done and Gate P has passed (step 0 of seed 0 is A01's
    # pre-registered zero-shot value)
    for s in (0, 1):
        st = [{"method": "mirror_exact", "variant": "wspace", "structure": WSPACE[B], "lr": LR1, "fp32": s == 0}]
        spec, gb = heal_spec(f"a01_wspace_s{s}", "A01", "main", s, st, hook_steps=(0, 100, 300) if s == 0 else ())
        add(f"heal_a01_wspace_s{s}", "W1", spec, gb, units_of(st), deps=["a01_fit", "gate_p"], prio=85,
            note="A01 healed comparison; never cut")
    st = [{"method": "mirror_exact", "variant": "wspace", "structure": WSPACE[B], "lr": LR1}]
    spec, gb = heal_spec("a01_wspace_s2", "A01", "main", 2, st, hook_steps=())
    add("heal_a01_wspace_s2", "W2", spec, gb, units_of(st), deps=["a01_decide"],
        when={"file": "results/A01/decision.json", "key": "adopted", "equals": True}, prio=70,
        note="third A*_w seed, only if A*_w is adopted")

    # W2: the A05 matrix and the A04 rows, seed 0 in one lockstep process (5 new methods select their rate here)
    st0 = [{"method": "svdllm_full", "lr_select": LRSEL}, {"method": "iosvd", "lr_select": LRSEL},
           {"method": "flap", "lr_select": LRSEL}, {"method": "blockdrop", "lr_select": LRSEL},
           {"method": "original", "lr_select": LRSEL}, {"method": "slicegpt", "lr": LR1},
           {"method": "svd", "lr": LR1}, {"method": "random_structure", "lr": LR1}]
    spec, gb = heal_spec("main_s0", "A05", "main", 0, st0, lowrank=4, extra_gb=17.0,
                         extra={"blockdrop_score_fn": "mirror.final.heal.blockdrop_score:msel_scorer"})
    add("heal_main_s0", "W2", spec, gb, units_of(st0), deps=["gate_p"], prio=75,
        note="A04 and A05 seed 0 (learning-rate trials of the new methods first); the block-drop scorer holds the "
             "bank and method-selection images on the GPU while it builds (+17 GB)")
    lrf = lambda m: run_dir("main", B, m, 0)  # noqa: E731
    sel_file = lambda m: {"file": f"{lrf(m)}/lr_selection/lr_selection.json"}  # noqa: E731
    st = [{"method": "svdllm_full", "lr_from": lrf("svdllm_full")}, {"method": "iosvd", "lr_from": lrf("iosvd")}]
    spec, gb = heal_spec("main_s1", "A05", "main", 1, st, hook_steps=(), lowrank=2)
    add("heal_main_s1", "W2", spec, gb, units_of(st), deps=["gate_p"],
        when=[sel_file("svdllm_full"), sel_file("iosvd")], prio=60, note="A05 seed 1 of the new low-rank methods")
    st = [{"method": "slicegpt", "lr": LR1}]
    spec, gb = heal_spec("main_s1_slicegpt", "A05", "main", 1, st, hook_steps=())
    add("heal_main_s1_slicegpt", "W2", spec, gb, units_of(st), cut=6, deps=["gate_p"], prio=40,
        note="second SliceGPT seed (cut order 6)")
    st = [{"method": "svdllm_full", "lr_from": lrf("svdllm_full")}]
    spec, gb = heal_spec("main_s2", "A05", "main", 2, st, hook_steps=(), lowrank=1)
    add("heal_main_s2", "W2", spec, gb, units_of(st), deps=["gate_p"], when=[sel_file("svdllm_full")], prio=55,
        note="A05 seed 2 of full SVD-LLM")
    st = [{"method": "iosvd", "lr_from": lrf("iosvd")}]
    spec, gb = heal_spec("main_s2_iosvd", "A05", "main", 2, st, hook_steps=(), lowrank=1)
    add("heal_main_s2_iosvd", "W2", spec, gb, units_of(st), cut=6, deps=["gate_p"], when=[sel_file("iosvd")],
        prio=40, note="third IO-SVD seed (cut order 6)")

    # W2: A02 heals from fit sets 1, 2 and COCO (the adopted operator type fitted on each fit set)
    st = ([{"method": "mirror_exact", "variant": r, "lr": LR1,
            "structure": {"adopted": {"wspace": a02_wspace(r), "astar": a02_fit(r)}}} for r in A02_ROLES]
          + [{"method": "pair_random", "variant": "h1", "structure": "adopted", "lr": LR1}])
    spec, gb = heal_spec("a02_heals", "A02", "main", 0, st, hook_steps=(0,))
    add("heal_a02", "W2", spec, gb, units_of(st), deps=["a01_decide", "a02_fit_fit1", "a02_fit_fit2", "a02_fit_coco"],
        after=["a02_wspace_fit1", "a02_wspace_fit2", "a02_wspace_coco"], prio=65,
        note="A02-5, A02-6 and A03-2 (H1); never cut")

    # W3: A06 frontier and stacking, seed 0. The SVD-LLM points need only Gate P; Mirror + SVD-LLM and the 9-block
    # point use the adopted operator (they select their learning rate here)
    adopted_b = "adopted"
    st = [{"method": "svdllm", "variant": f"f{int(round(f * 100)):03d}", "frac": f, "lr": LR1} for f in (0.65, 0.50, 0.40)]
    spec, gb = heal_spec("a06_svd_s0", "A06", "a06", 0, st, hook_steps=(), lowrank=3)
    add("heal_a06_svd_s0", "W2", spec, gb, units_of(st), deps=["gate_p"], prio=60,
        note="A06 SVD-LLM frontier at 1 seed: never cut")
    st = ([{"method": "mirror_svdllm", "variant": f"f{int(round(f * 100)):03d}", "frac": f, "structure": adopted_b,
            "lr_select": LRSEL, **({"fp32": True} if f == 0.50 else {})} for f in (0.65, 0.50, 0.40)]
          + [{"method": "mirror_partial", "variant": "k9", "k": 9, "lr_select": LRSEL,
              "structure": {"adopted": {"wspace": K9_WSPACE, "astar": CANON[B]}}}])
    spec, gb = heal_spec("a06_mirror_s0", "A06", "a06", 0, st, hook_steps=(), lowrank=3)
    add("heal_a06_mirror_s0", "W3", spec, gb, units_of(st), deps=["a01_decide"], after=["a01_fit_k9"], prio=60,
        note="A06 stacking and the price of exactness at 1 seed: never cut")
    st = [{"method": "svdllm", "variant": "f050", "frac": 0.50, "lr": LR1},
          {"method": "mirror_svdllm", "variant": "f050", "frac": 0.50, "structure": adopted_b,
           "lr_from": run_dir("a06", B, "mirror_svdllm_f050", 0)}]
    spec, gb = heal_spec("a06_s1", "A06", "a06", 1, st, hook_steps=(), lowrank=2)
    add("heal_a06_s1", "W3", spec, gb, units_of(st), cut=5, deps=["a01_decide"],
        when=[{"file": f"{run_dir('a06', B, 'mirror_svdllm_f050', 0)}/lr_selection/lr_selection.json"}], prio=35,
        note="extra seeds at 0.50 (cut order 5)")

    # W4: DeiT-III-B (rates: the DINOv2-B choice against the other grid value, 200-step trials, seed 0)
    dl = {"logits": True}
    st = [{"method": "mirror_exact", "structure": "adopted", "lr_select": LRSEL, **dl},
          {"method": "svdllm", "lr_select": LRSEL, **dl}]
    spec, gb = heal_spec("deit_s0", "A08", "a08", 0, st, model=DT, lowrank=1)
    add("heal_deit_s0", "W4", spec, gb, units_of(st, DT), deps=["a01_decide"], after=["a01_fit_deit"], prio=55,
        note="A08-1 seed 0; never cut")
    for s, cut in ((1, "never_cut"), (2, 7)):
        st = [{"method": "mirror_exact", "structure": "adopted", "lr_from": run_dir("a08", DT, "mirror_exact", 0),
               **dl},
              {"method": "svdllm", "lr_from": run_dir("a08", DT, "svdllm", 0), **dl}]
        spec, gb = heal_spec(f"deit_s{s}", "A08", "a08", s, st, model=DT, hook_steps=(), lowrank=1)
        add(f"heal_deit_s{s}", "W4", spec, gb, units_of(st, DT), cut=cut, deps=["heal_deit_s0"], prio=50,
            note="A08-1 seed %d" % s + ("" if cut == "never_cut" else " (cut order 7: third DeiT seeds)"))
    st = [{"method": "random_structure", "lr_select": LRSEL, **dl}]
    spec, gb = heal_spec("deit_random", "A08", "a08", 0, st, model=DT, hook_steps=())
    add("heal_deit_random", "W4", spec, gb, units_of(st, DT), cut=7, deps=["a01_decide"], prio=30,
        note="DeiT random-structure control (cut order 7)")

    # W5: A11 extended healing, DINOv2-L, A15b
    for m, cut in (("mirror_exact", 4), ("svdllm", 3)):
        st = [{"method": m, "variant": "ext", "lr": LR1, **({"structure": "adopted"} if m == "mirror_exact" else {})}]
        spec, gb = heal_spec(f"a11_{m}", "A11", "a11", 0, st, steps=2340, hook_steps=(), data=EXT_DATA,
                             lowrank=int(m == "svdllm"))
        add(f"heal_a11_{m}", "W5", spec, gb, units_of(st, B, 2340), cut=cut,
            deps=(["a01_decide"] if m == "mirror_exact" else ["gate_p"]) + ["w0_roles_ext"], prio=45,
            note=f"A11 extended healing, {m} (cut order {cut})")
    st = [{"method": "mirror_exact", "structure": "adopted", "lr": LR1, "fp32": True}, {"method": "svdllm", "lr": LR1}]
    spec, gb = heal_spec("l_s0", "A08", "a08", 0, st, model=LM, hook_steps=(), lowrank=1)
    add("heal_l_s0", "W5", spec, gb, units_of(st, LM), cut=2, deps=["a01_decide"], after=["a01_fit_l"], prio=40,
        note="DINOv2-L, indicative (cut order 2)")
    st = [{"method": "rot90_probe", "lr_select": LRSEL}]
    spec, gb = heal_spec("rot90", "A15b", "a15", 0, st, hook_steps=())
    add("heal_a15b", "W5", spec, gb, units_of(st), cut=1, deps=["gate_p"], prio=10,
        note="A15b cause probe (cut first)")
    return J


# ----------------------------------------------------------------------------------------------
# The manifest
# ----------------------------------------------------------------------------------------------
def py(cmd):
    return f"python {cmd}"


def manifest(heals):
    jobs = []

    def job(**kw):
        kw = {k: v for k, v in kw.items() if v not in (None, [], "")}
        jobs.append(kw)

    w0env = {"TIMING_STEPS": "60", "FULL_STEPS": "780",
             "TIMING_CMD": "python scripts/fp_heal.py --spec jobs/heal/w0_timing_{model}.yaml --max-steps {steps} "
                           "--out-root {out} --queue-dir runs/final/_timing_queue",
             "SMOKE_HEAL_CMD": "python scripts/fp_heal.py --dry-run --device cuda --out-root runs/final/_smoke/heal "
                               "--queue-dir runs/final/_smoke/queue",
             "SMOKE_EVAL_CMD": "python scripts/fp_eval.py --dry-run --device cuda --out runs/final/_smoke/eval"}
    w0 = "logs/final/w0"
    # W0
    job(id="w0_gpu", wave="W0", kind="light", gpu_mem_gb=4, command="bash scripts/fp_vm_w0.sh gpu",
        done=f"{w0}/gpu.done", priority=100)
    job(id="w0_models", wave="W0", kind="cpu", command="bash scripts/fp_vm_w0.sh models", done=f"{w0}/models.done",
        priority=100)
    job(id="w0_data", wave="W0", kind="cpu", host_mem_gb=24, command="bash scripts/fp_vm_w0.sh data",
        done=f"{w0}/data.done", priority=100)
    job(id="w0_roles", wave="W0", kind="cpu", host_mem_gb=16, command="bash scripts/fp_vm_w0.sh roles",
        deps=["w0_data"], done=f"{w0}/roles.done", priority=95)
    job(id="w0_tests", wave="W0", kind="cpu", command="bash scripts/fp_vm_w0.sh tests", done=f"{w0}/tests.done",
        priority=90)
    job(id="w0_smoke", wave="W0", kind="light", gpu_mem_gb=20, command="bash scripts/fp_vm_w0.sh smoke",
        deps=["w0_gpu", "w0_models"], done=f"{w0}/smoke.done", priority=95, env=w0env)
    job(id="w0_timing", wave="W0", kind="light", gpu_mem_gb=60, exclusive=True, command="bash scripts/fp_vm_w0.sh timing",
        deps=["w0_smoke", "w0_data"], done="results/final/w0_timing.json", priority=95, env=w0env)
    job(id="w0_extra", wave="W1", kind="cpu", host_mem_gb=16, command="bash scripts/fp_vm_w0.sh extra",
        deps=["w0_data"], done=f"{w0}/extra.done", priority=30, note="ADE20K, transfer sets, ImageNet-V2, Sketch")
    job(id="w0_roles_ext", wave="W1", kind="cpu", host_mem_gb=24, command="bash scripts/fp_vm_w0.sh roles_ext",
        deps=["w0_roles"], done=f"{w0}/roles_ext.done", priority=35, cut_rank=4,
        note="heal_ext (200k images) for A11 only, so it is cut with A11")

    # W1 services and structure work
    job(id="eval_server", wave="W1", kind="eval", service=True, gpu_mem_gb=35, priority=90,
        command=py("scripts/fp_eval_server.py --gpu-frac 0.25"), deps=["w0_data", "w0_models"])
    job(id="a01_fit", wave="W1", kind="cpu", host_mem_gb=24, priority=99,
        command=py("scripts/fp_a01_weightspace.py --mode matched --skip-knn --workers 10"),
        deps=["w0_models", "w0_data"], done=WSPACE[B], note="A*_w, matched mode (pre-registered)")
    job(id="a01_fit_free", wave="W2", kind="cpu", host_mem_gb=16, priority=5,
        command=py("scripts/fp_a01_weightspace.py --mode free --skip-knn --workers 6 --out results/A01/free"),
        deps=["a01_fit"], done=f"results/A01/free/structure/{B}_wspace_free.npz",
        note="diagnostic only (free mode; never healed)")
    for k in (0, 1, 2):
        job(id=f"a02_fit_fit{k}", wave="W1", kind="cpu", host_mem_gb=16, priority=60,
            command=py(f"scripts/fp_structure_fit.py --role fit{k} --fit-seed {k} --workers 6 "
                       f"--operators runs/x2/perlayer_reg_s{k}/results/e1_operators.npz "
                       f"--reference results/MUST-07/structure/{B}_s{k}.npz"),
            deps=["w0_models", "w0_data"], done=a02_fit(f"fit{k}"))
    job(id="a02_fit_fit3", wave="W1", kind="cpu", host_mem_gb=16, priority=60,
        command=py("scripts/fp_structure_fit.py --role fit3 --fit-seed 3 --workers 6 --images cache/fit3_224_uint8.npy "
                   "--index cache/fit3_index.npz --rows-key rows"),
        deps=["w0_models", "w0_roles"], done=a02_fit("fit3"))
    job(id="a02_fit_coco", wave="W1", kind="cpu", host_mem_gb=16, priority=60,
        command=py("scripts/fp_structure_fit.py --role coco --fit-seed 100 --workers 6 --images cache/coco224_uint8.npy "
                   "--index cache/coco_index.npz --rows-key rows"),
        deps=["w0_models", "w0_roles"], done=a02_fit("coco"))
    job(id="a02_stability", wave="W2", kind="light", gpu_mem_gb=30, priority=50,
        command=py("scripts/fp_a02_stability.py --out results/A02"),
        deps=["a02_fit_fit0", "a02_fit_fit1", "a02_fit_fit2", "a02_fit_fit3", "a02_fit_coco"],
        done="results/A02/metrics.json")
    job(id="a03_controls", wave="W1", kind="cpu", host_mem_gb=24, priority=45,
        command=py(f"scripts/fp_a03_controls.py --workers 8 --reference-structure {CANON[B]}"),
        deps=["w0_models", "w0_data"], done="results/A03/metrics.json", note="A03-1 (split-half ratio) and C1 to C4")
    adopted = {"file": "results/A01/decision.json", "key": "adopted", "equals": True}
    for r in A02_ROLES:
        rows = {"fit1": "--fit-images cache/train224_uint8.npy --fit-index cache/train_index.npz --fit-rows-key fit1_rows",
                "fit2": "--fit-images cache/train224_uint8.npy --fit-index cache/train_index.npz --fit-rows-key fit2_rows",
                "coco": "--fit-images cache/coco224_uint8.npy --fit-index cache/coco_index.npz --fit-rows-key rows"}[r]
        job(id=f"a02_wspace_{r}", wave="W2", kind="cpu", host_mem_gb=16, priority=60, when=adopted,
            command=py(f"scripts/fp_a01_weightspace.py --mode matched --skip-knn --workers 6 --structure {a02_fit(r)} "
                       f"{rows} --out results/A02/wspace_{r}"),
            deps=["a01_decide", f"a02_fit_{r}"], done=a02_wspace(r),
            note="the adopted operator type on this fit set (only if A*_w is adopted)")
    job(id="a01_fit_k9", wave="W2", kind="cpu", host_mem_gb=16, priority=60, when=adopted,
        command=py("scripts/fp_a01_weightspace.py --mode matched --skip-knn --workers 6 --k 9"),
        deps=["a01_decide"], done=K9_WSPACE, note="A06 9-block point with the adopted operator type")
    job(id="a01_fit_deit", wave="W2", kind="cpu", host_mem_gb=16, priority=55, when=adopted,
        command=py(f"scripts/fp_a01_weightspace.py --model {DT} --structure {CANON[DT]} --mode matched --skip-knn "
                   f"--workers 6 --out results/A01/{DT}"),
        deps=["a01_decide"], done=WSPACE[DT])
    job(id="a01_fit_l", wave="W2", kind="cpu", host_mem_gb=24, priority=30, when=adopted, cut_rank=2,
        command=py(f"scripts/fp_a01_weightspace.py --model {LM} --structure {CANON[LM]} --mode matched --skip-knn "
                   f"--workers 6 --out results/A01/{LM}"),
        deps=["a01_decide"], done=WSPACE[LM])

    # heal jobs
    for h in heals:
        job(id=h["id"], wave=h["wave"], kind="heal", gpu_mem_gb=h["gb"], heal_units=h["units"],
            cut_rank=h["cut"], priority=h["prio"], deps=h["deps"], after=h["after"], when=h["when"],
            command=py(f"scripts/fp_heal.py --spec jobs/heal/{h['spec']['job']}.yaml --resume"),
            done=done_of(h["spec"]), note=h["note"])

    # decisions
    gate_runs = [run_dir("main", B, "mirror_exact_astar", s) for s in (0, 1, 2)] + \
                [run_dir("main", B, "svdllm", s) for s in (0, 1, 2)]
    wait = " ".join(f"--run-dir {r}" for r in gate_runs)
    job(id="gate_p", wave="W1", kind="decision", priority=98,
        command=f"python scripts/fp_wait_eval.py {wait} --passes bf16 && python scripts/fp_gate_p.py --decisions-log notes/decisions.md",
        deps=["heal_gate_s0", "heal_gate_s1", "heal_gate_s2"], done="results/A01/gate_p.json")
    wruns = [run_dir("main", B, "mirror_exact_wspace", s) for s in (0, 1)]
    job(id="a01_decide", wave="W1", kind="decision", priority=98,
        command=(f"python scripts/fp_wait_eval.py {' '.join('--run-dir ' + r for r in wruns)} --passes bf16 && "
                 f"python scripts/fp_wait_eval.py --run-dir {wruns[0]} --step 0 --passes bf16 && "
                 f"python scripts/fp_a01_decide.py --decisions-log notes/decisions.md"),
        deps=["gate_p", "heal_a01_wspace_s0", "heal_a01_wspace_s1"], done="results/A01/decision.json")

    arm = "$(python scripts/fp_main_arm.py)"
    arm_dir = lambda s: f"runs/final/main/{B}/{arm}/s{s}"  # noqa: E731
    # A09 exactness audit (main arm seed 0; DeiT-III Mirror seed 0)
    job(id="a09_audit_b", wave="W2", kind="light", gpu_mem_gb=30, priority=45,
        command=py(f"scripts/fp_a09_audit.py --run {arm_dir(0)} --out results/A09/{B} --knn-impl knn_fast"),
        deps=["a01_decide"], after=["heal_gate_s0"], done=f"results/A09/{B}/metrics.json")
    job(id="a09_audit_deit", wave="W4", kind="light", gpu_mem_gb=30, priority=40,
        command=py(f"scripts/fp_a09_audit.py --run {run_dir('a08', DT, 'mirror_exact', 0)} --model {DT} "
                   f"--out results/A09/{DT} --no-flip"),
        deps=["heal_deit_s0"], done=f"results/A09/{DT}/metrics.json",
        note="the classifier's flip consistency comes from scripts/fp_a08_top1.py")
    # A08 top-1 of DeiT-III (own classifier), every healed run and the original
    deit_runs = ([run_dir("a08", DT, m, s) for m in ("mirror_exact", "svdllm") for s in (0, 1, 2)]
                 + [run_dir("a08", DT, "random_structure", 0)])
    for i, r in enumerate(deit_runs):
        dep = {0: "heal_deit_s0", 1: "heal_deit_s1", 2: "heal_deit_s2"}[int(r[-1])] if "random" not in r else \
            "heal_deit_random"
        cut = {"heal_deit_s2": 7, "heal_deit_random": 7}.get(dep, "never_cut")
        passes = "bf16,fp32" if "mirror_exact" in r else "bf16"
        job(id=f"a08_top1_{Path(r).parent.name}_{Path(r).name}", wave="W4", kind="eval", gpu_mem_gb=12,
            priority=45, cut_rank=cut, command=py(f"scripts/fp_a08_top1.py --run-dir {r} --passes {passes}"),
            deps=[dep], done=f"{r}/eval/top1_step780.json")
    job(id="a08_params", wave="W4", kind="cpu", priority=45,
        command=py(f"scripts/fp_a08_params.py --run-dir {run_dir('a08', DT, 'mirror_exact', 0)}"),
        deps=["heal_deit_s0"], done=f"results/A08/{DT}/params_check.json",
        note="A08-1: stored parameters of the even/odd form equal the formula")
    job(id="a08_top1_original", wave="W4", kind="eval", gpu_mem_gb=12, priority=45,
        command=py(f"scripts/fp_a08_top1.py --original {DT} --passes bf16,fp32"), deps=["w0_models", "w0_data"],
        done=f"runs/final/a08/{DT}/_original/eval/top1_original.json")
    # A07: flip averaging at equal compute (evaluation only)
    a06r = [run_dir("a06", B, f"svdllm_f{x}", 0) for x in ("040", "050", "065")]
    job(id="a07_flipavg_svdllm", wave="W3", kind="eval", gpu_mem_gb=30, priority=40,
        command=py(" ".join(["scripts/fp_eval.py", *(f"--run-dir {r}" for r in a06r),
                             "--step 780 --flip-average --passes bf16 --probe --probe-flipavg --cost --experiment A07"])),
        deps=["heal_a06_svd_s0"], done=f"{a06r[-1]}/eval/step780_probe_flipavg.json")
    for m in (B, "dinov2_vits14_reg"):
        job(id=f"a07_flipavg_{m}", wave="W3", kind="eval", gpu_mem_gb=30, priority=40,
            command=py(f"scripts/fp_eval.py --original {m} --flip-average --passes bf16 --probe --probe-flipavg "
                       f"--cost --experiment A07 --out runs/final/a07/{m}"),
            deps=["w0_models", "w0_data"], done=f"runs/final/a07/{m}/eval/original_probe_flipavg.json")
    job(id="a07_probe_mirror", wave="W3", kind="eval", gpu_mem_gb=30, priority=40,
        command=(py(f"scripts/fp_eval.py --run-dir {arm_dir(0)} --run-dir {arm_dir(1)} --run-dir {arm_dir(2)} "
                    f"--step 780 --passes bf16 --probe --cost --experiment A07")
                 + " && mkdir -p runs/final/a07 && touch runs/final/a07/mirror_probe.done"),
        deps=["a01_decide"], after=["heal_gate_s2", "heal_a01_wspace_s2"],
        done="runs/final/a07/mirror_probe.done")
    # A12 systems table (exclusive GPU), DINOv2-B and DeiT-III
    job(id="a12_bench_b", wave="W3", kind="light", gpu_mem_gb=40, exclusive=True, priority=35,
        command=py(f"scripts/fp_a12_bench.py --manifest jobs/a12/{B}_$(python scripts/fp_main_arm.py --variant).json "
                   f"--out results/A12/{B}"),
        deps=["heal_main_s0", "heal_a06_svd_s0", "a01_decide"], done=f"results/A12/{B}/metrics.json")
    job(id="a12_bench_deit", wave="W4", kind="light", gpu_mem_gb=30, exclusive=True, priority=30,
        command=py(f"scripts/fp_a12_bench.py --manifest jobs/a12/{DT}.json --out results/A12/{DT}"),
        deps=["heal_deit_s0"], done=f"results/A12/{DT}/metrics.json")
    # A12b: the fused even/odd kernel (best effort). The check always exits 0 and records its result, so a kernel
    # that does not compile or does not match can never trip the runner's two-failure stop; the timing runs alone on
    # the GPU and only if the check passed (PREREG_FINAL_PHASE.md row A12b)
    chk = "results/A12b/correctness_only/metrics.json"
    job(id="a12b_check", wave="W1", kind="light", gpu_mem_gb=12, priority=60,
        command=(py("scripts/fp_a12b_kernel.py --correctness-only --out results/A12b/correctness_only")
                 + f"; [ -f {chk} ] || {{ mkdir -p results/A12b/correctness_only; "
                 + f"echo '{{\"correctness\": {{\"pass\": false, \"error\": \"no output\"}}}}' > {chk}; }}; "
                 + "mkdir -p logs/final && touch logs/final/a12b_check.done"),
        deps=["w0_smoke"], done="logs/final/a12b_check.done", note="Triton kernel against the reference (W0 smoke)")
    job(id="a12b_timing", wave="W3", kind="light", gpu_mem_gb=40, exclusive=True, priority=30,
        when={"file": chk, "key": "correctness.pass", "equals": True},
        command=(py(f"scripts/fp_a12b_kernel.py --model {B} --structure {CANON[B]} --out results/A12b")
                 + "; mkdir -p logs/final && touch logs/final/a12b_timing.done"),
        deps=["a12b_check", "w0_timing"], done="logs/final/a12b_timing.done",
        note="A12b timing, alone on the GPU; the bar is scored only on whole-model compiled timings")

    # A10: full bank and the DINOv2 linear protocol; downstream; P11.2
    fb = [f"--model original=original:{B}"] + [f"--model mirror_s{s}={arm_dir(s)}" for s in (0, 1, 2)] + \
         [f"--model svdllm_s{s}={run_dir('main', B, 'svdllm', s)}" for s in (0, 1, 2)] + \
         [f"--model svdllm_full_s0={run_dir('main', B, 'svdllm_full', 0)}",
          f"--model iosvd_s0={run_dir('main', B, 'iosvd', 0)}"]
    ext = (f'X=(); for m in mirror_exact_ext svdllm_ext; do d=runs/final/a11/{B}/$m/s0; '
           f'[ -f $d/DONE ] && X+=(--model a11_$m=$d); done; ')
    job(id="a10_fullbank", wave="W5", kind="eval", gpu_mem_gb=45, host_mem_gb=32, priority=35,
        command=(ext + "python scripts/fp_fullbank.py " + " ".join(fb) + ' "${X[@]}" '
                 "--knn-only svdllm_full_s0,iosvd_s0 --linear-tf32 --out runs/final/A10/fullbank"),
        deps=["a01_decide", "heal_main_s0", "heal_main_s1", "heal_main_s2"],
        after=["heal_gate_s2", "heal_a01_wspace_s2", "heal_a11_mirror_exact", "heal_a11_svdllm"],
        done="runs/final/A10/fullbank/summary.json", note="one stream over the 1.28M training images for every model")
    dl = ([f"--healed-run vexact_s{s}={arm_dir(s)}" for s in (0, 1, 2)]
          + [f"--healed-run best_svdllm_s{s}={run_dir('main', B, 'svdllm', s)}" for s in (0, 1, 2)])
    job(id="a10_downstream", wave="W5", kind="eval", gpu_mem_gb=25, priority=35,
        command=(f"python scripts/fp_eval_downstream.py --script tier2_linear -- --model {B} --variants original "
                 + " ".join(dl) + " --out runs/final/A10/t2 && "
                 f"python scripts/fp_eval_downstream.py --script core07_shift -- --model {B} --variants original "
                 + " ".join(dl) + " --probe-from runs/final/A10/t2 --out runs/final/A10/core07"),
        deps=["a01_decide", "w0_extra"], after=["heal_gate_s2", "heal_a01_wspace_s2"],
        done="runs/final/A10/core07/fp_eval_downstream_meta.json",
        note="linear probe, ImageNet-V2 and deduplicated Sketch for the new-pipeline Mirror and SVD-LLM seeds")
    job(id="a10_p112", wave="W5", kind="eval", gpu_mem_gb=25, priority=30,
        command=(f"python scripts/tier2_ade20k.py --model {B} --variants original --mask-eq --out runs/final/A10/p112 && "
                 f"python scripts/tier2_ade20k.py --model {B} --variants \"\" --healed-run "
                 f"vexact_s0=runs/MUST-09/{B}/mirror_exact/s0 --mask-eq --out runs/final/A10/p112 && "
                 f"touch runs/final/A10/p112/DONE"),
        deps=["w0_extra", "w0_models"], done="runs/final/A10/p112/DONE",
        note="P11.2 re-measured on the MUST-09 model it was measured on")
    # A15
    job(id="a15a_siglip", wave="W5", kind="light", gpu_mem_gb=30, priority=10, cut_rank=1,
        command=py("scripts/fp_a15a_siglip.py --out results/A15a"), deps=["w0_models", "w0_data"],
        done="results/A15a/metrics.json")
    job(id="a15b_measure", wave="W5", kind="light", gpu_mem_gb=20, priority=10, cut_rank=1,
        command=py(f"scripts/fp_a15b_measure.py --finetuned {run_dir('a15', B, 'rot90_probe', 0)} --out results/A15b"),
        deps=["heal_a15b"], done="results/A15b/metrics.json")
    # figures that need the VM (teaser) and the report
    job(id="fig_teaser", wave="W4", kind="light", gpu_mem_gb=10, priority=10,
        command=py("scripts/fp_fig_teaser.py") + " && touch logs/final/fig_teaser.done", deps=["w0_models", "w0_data"],
        done="logs/final/fig_teaser.done")
    job(id="fig_fourier", wave="W4", kind="cpu", priority=10,
        command=py("scripts/fp_fig_fourier.py") + " && touch logs/final/fig_fourier.done",
        deps=["w0_data"], done="logs/final/fig_fourier.done")
    job(id="report", wave="End", kind="report", priority=100, after=["W5"],
        command=py("scripts/collect_final_phase.py"), done="results/FINAL_PHASE_REPORT.md")
    for f in ("frontier", "stability", "controls", "curves"):        # F1 to F4 from the result files (CPU)
        job(id=f"fig_{f}", wave="End", kind="report", priority=90, after=["W5"],
            command=py(f"scripts/fp_fig_{f}.py") + f" && touch logs/final/fig_{f}.done",
            done=f"logs/final/fig_{f}.done")

    settings = {
        "env": {"MIRROR_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
        "stop_when": [{"file": "results/A01/gate_p.json", "key": "pass", "equals": False,
                       "reason": "Gate P failed (FINAL_PHASE_PLAN.md Section 6): stop and report"}],
        "barrier_waves": [], "max_heals": 3, "max_cpu": 6,
    }
    return {"version": 1, "settings": settings, "jobs": jobs}


def a12_manifests():
    out = {}
    for variant in ("astar", "wspace"):
        arm = f"mirror_exact_{variant}"
        out[f"{B}_{variant}"] = [
            {"name": "original", "kind": "original", "model": B},
            {"name": "mirror_dense", "kind": "run", "run_dir": run_dir("main", B, arm, 0), "form": "dense"},
            {"name": "mirror_even_odd", "kind": "run", "run_dir": run_dir("main", B, arm, 0), "form": "even_odd",
             "align": 8},
            {"name": "svdllm", "kind": "run", "run_dir": run_dir("main", B, "svdllm", 0)},
            {"name": "iosvd", "kind": "run", "run_dir": run_dir("main", B, "iosvd", 0)},
            {"name": "flap", "kind": "run", "run_dir": run_dir("main", B, "flap", 0)},
            {"name": "blockdrop", "kind": "run", "run_dir": run_dir("main", B, "blockdrop", 0)},
            {"name": "flipavg_original", "kind": "flip_average", "base": "original", "batched": False},
            # A07 candidates: the seed-0 SVD-LLM models of A06 and the released DINOv2-S/14 reg, each also timed as
            # two-pass flip averaging (PREREG_FINAL_PHASE.md Section 2.5: measured time per image at batch 256)
            *[{"name": f"svdllm_f{x}", "kind": "run", "run_dir": run_dir("a06", B, f"svdllm_f{x}", 0)}
              for x in ("040", "050", "065")],
            {"name": "dinov2_vits14_reg", "kind": "original", "model": "dinov2_vits14_reg"},
            *[{"name": f"flipavg_{n}", "kind": "flip_average", "base": n, "batched": False}
              for n in ("svdllm_f040", "svdllm_f050", "svdllm_f065", "dinov2_vits14_reg")],
        ]
    out[DT] = [
        {"name": "original", "kind": "original", "model": DT},
        {"name": "mirror_dense", "kind": "run", "run_dir": run_dir("a08", DT, "mirror_exact", 0), "form": "dense"},
        {"name": "mirror_even_odd", "kind": "run", "run_dir": run_dir("a08", DT, "mirror_exact", 0), "form": "even_odd",
         "align": 8},
        {"name": "svdllm", "kind": "run", "run_dir": run_dir("a08", DT, "svdllm", 0)},
    ]
    return out


def timing_specs():
    """W0 timing (scripts/fp_vm_w0.sh timing): one Mirror-exact student per model."""
    out = {}
    for m in (B, DT, LM):
        st = [{"method": "mirror_exact", "lr": LR1}]
        spec, _ = heal_spec(f"w0_timing_{m}", "W0", "w0", 0, st, model=m, hook_steps=())
        spec["eval"] = {"n_query_full": 50000, "n_query_partial": 10000, "fp32_last": False}
        out[f"w0_timing_{m}"] = spec
    return out


def main() -> int:
    heals = heal_jobs()
    hdir = ROOT / "jobs" / "heal"
    hdir.mkdir(parents=True, exist_ok=True)
    for h in heals:
        (hdir / f"{h['spec']['job']}.yaml").write_text(yaml.safe_dump(h["spec"], sort_keys=False), encoding="utf-8")
    for name, spec in timing_specs().items():
        (hdir / f"{name}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    adir = ROOT / "jobs" / "a12"
    adir.mkdir(parents=True, exist_ok=True)
    for name, entries in a12_manifests().items():
        (adir / f"{name}.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    man = manifest(heals)
    header = ("# Generated by scripts/fp_make_jobs.py (edit that script, not this file). FINAL_PHASE_PLAN.md v2,\n"
              "# Sections 4, 8 and 9; format in jobs/final_phase.schema.md; check with\n"
              "#   python scripts/fp_runner.py check jobs/final_phase.yaml\n")
    (ROOT / "jobs" / "final_phase.yaml").write_text(header + yaml.safe_dump(man, sort_keys=False, width=120),
                                                    encoding="utf-8")
    units = sum(h["units"] for h in heals)
    print(f"wrote {len(heals)} heal specs, {len(a12_manifests())} A12 manifests and {len(man['jobs'])} jobs; "
          f"{units:.1f} heal-units in total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
