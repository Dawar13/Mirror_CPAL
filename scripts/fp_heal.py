#!/usr/bin/env python
"""One lockstep healing job of the final phase: several (method, variant) students, one model, one seed
(docs/final_phase_design.md Section 5; FINAL_PHASE_PLAN.md Sections 4 and 8).

    python scripts/fp_heal.py --spec jobs/heal/a05_s0.yaml               # GPU job
    python scripts/fp_heal.py --spec jobs/heal/a05_s0.yaml --resume      # continue after an interruption
    python scripts/fp_heal.py --dry-run --out-root /tmp/fp               # CPU, tiny random ViT, synthetic images

Job spec (YAML or JSON; every key but `students` has a default):
    experiment: A05                 # recorded in meta.json and the evaluation requests
    group: main                     # runs/final/<group>/<model>/<method>[_<variant>]/s<seed>/ (design doc 4.5a: main,
                                    # a06, a08, a11, a15; default: the experiment); LR trials go to group "lr"
    model: dinov2_vitb14_reg        # dinov2_vitb14_reg | dinov2_vitl14_reg (torch.hub) | deit3_b16 (timm registry)
    seed: 0                         # healing seed: data order, crops, flips (and the learning-rate trials)
    steps: null                     # null: the recipe's 2 epochs (780 steps on the 100k heal set); 2340 extended
    micro_batch: 64                 # images per forward and backward (does not change results)
    compile: false                  # torch.compile of every student's blocks (keep off for Gate P)
    gpu_frac: 0.9                   # torch.cuda.set_per_process_memory_fraction
    out_root: runs/final
    queue_dir: runs/final/_queue    # evaluation requests (design doc 4.4)
    job: null                       # job folder name (default: experiment and a hash of the students)
    structure: results/MUST-07/structure/<model>_s0.npz   # canonical SpanStructure (fit seed 0)
    adopted_structures: {wspace: null, astar: null}       # for structure "adopted" with results/A01/decision.json's
                                                          # {"adopted": true|false} (astar default: `structure`)
    gate_b: results/MUST-07/gate_b.json                   # compressor hyperparameters (gate B)
    random_seed: 1                  # structure seed of random_structure and pair_random (configs/must09.yaml)
    cache_dir: cache                # mirror.protocol.Protocol: fit set 0 for the compressor statistics
    data: {images: cache/heal256_uint8.npy, index: cache/train_index.npz, train_key: heal_rows,
           heldout_key: healval_rows, heldout_images: null, heldout_index: null, max_train: null,
           mode: auto, max_resident_gb: null}
                                    # extended set (A11): sources: [{images, index, key}, ...] are read as one array
                                    # (e.g. heal256 + heal_ext256 with keys heal_rows and rows); the held-out rows come
                                    # from heldout_key in the index of sources[heldout_source] (default 0)
    hook_steps: [0, 100, 300]       # checkpoint + evaluation request; the last step is always added
    state_every: 100                # training state for --resume
    eval: {n_query_full: 50000, n_query_partial: 10000, fp32_last: true}
                                    # fp32_last: the float32 k-NN pass (bank, queries and mirrored queries) at the last
                                    # step; a student's own `fp32: true|false` overrides it
    lr_selection: {budget_steps: 200}
    heal: {}                        # HealConfig overrides on top of configs/must09.yaml's heal section
    logits_fn: null                 # "module:function" giving logits_fn(model, final_tokens) (DeiT-III); default:
                                    # timm forward_head on the final-norm tokens
    blockdrop_score_fn: null        # "module:function": factory(ctx) -> score_fn(model) -> float (higher is better;
                                    # the k-NN score on the method-selection set) for the blockdrop student
    students:
      - {method: mirror_exact, variant: astar, lr: 1.0e-4}
      - {method: svdllm, lr: 1.0e-4}
      - {method: svdllm_full, lr_select: {lrs: [1.0e-4, 2.0e-4]}}            # 200-step trials, lowest held-out loss
      - {method: mirror_exact, variant: wspace, structure: <A*_w file>, lr: 1.0e-4}
      - {method: original, lr_from: runs/final/main/dinov2_vitb14_reg/original/s0}
    Student keys: method, variant, fp32 (true|false: the float32 evaluation pass at the last step; default the job's
    eval.fp32_last), lr | lr_select {lrs, budget_steps, choose: true} | lr_from (a run folder or an
    lr_selection.json), and builder options (structure, hparams, random_seed, k, frac, n_drop, logits, trainable;
    mirror/final/heal/students.py build_student). With lr_select and choose: false the trials are recorded and `lr`
    is used.

The recipe is MUST-09's (configs/must09.yaml heal section; train_scope "all"; every block in the loss), so
mirror_exact and svdllm at lr 1e-4 reproduce MUST-09 (Gate P). The script calls
mirror.final.prereg.require_prereg() before touching the GPU (dry runs allow a missing record) and writes meta.json
with mirror.final.prereg.write_final_meta in every student folder and in the job folder.

Outputs per student (design doc 4.4): spec.json (+ structure.npz), ckpt/step<S>.pt, train_log.csv, heldout_log.csv,
heldout.json, metrics.json, meta.json, lr_selection/lr_selection.json when the rate was selected here, and DONE (written
last, for the manifest's done: files); the trials of a selection go to
runs/final/lr/<model>/<method>[_<variant>]_lr<rate>/s<seed>/ (design doc 4.5a). Per job, in
<out_root>/<group>/<model>/_jobs/<job>/s<seed>/: job.json, train_state.pt (removed after a finished job),
summary.json, meta.json and DONE.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.heal import HealConfig  # noqa: E402
from mirror.utils import REPO_ROOT, load_config, log, set_seed  # noqa: E402

DEFAULTS = {
    "experiment": "heal", "group": None, "adopted_structures": None, "model": "dinov2_vitb14_reg", "seed": 0,
    "steps": None, "micro_batch": 64,
    "compile": False, "gpu_frac": 0.9, "out_root": "runs/final", "queue_dir": "runs/final/_queue", "job": None,
    "structure": None, "gate_b": "results/MUST-07/gate_b.json", "random_seed": 1, "cache_dir": "cache",
    "data": {"images": "cache/heal256_uint8.npy", "index": "cache/train_index.npz", "train_key": "heal_rows",
             "heldout_key": "healval_rows", "heldout_images": None, "heldout_index": None, "max_train": None,
             "mode": "auto", "max_resident_gb": None},
    "hook_steps": [0, 100, 300], "state_every": 100, "eval": {"n_query_full": 50000, "n_query_partial": 10000},
    "lr_selection": {"budget_steps": 200}, "heal": {}, "logits_fn": None, "blockdrop_score_fn": None, "students": [],
}
DRY_RUN = {
    "experiment": "dryrun", "model": "tiny", "seed": 0, "steps": 4, "micro_batch": 4, "gpu_frac": None,
    "hook_steps": [0, 2], "state_every": 2, "eval": {"n_query_full": 20, "n_query_partial": 10},
    "lr_selection": {"budget_steps": 2},
    "heal": {"epochs": 1, "batch_size": 8, "crop_size": 56, "calib_images": 8, "eval_every": 2, "log_every": 2,
             "prefetch": 0},
    "students": [{"method": "mirror_exact", "lr": 1e-3}, {"method": "svdllm", "lr_select": {"lrs": [1e-4, 1e-3]}},
                 {"method": "original", "lr": 1e-3}],
}
BUILDER_KEYS = ("structure", "hparams", "random_seed", "lowrank_factors", "k", "frac", "n_drop", "max_candidates",
                "logits", "trainable")


# ----------------------------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------------------------
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_spec(path) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text) or {}


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=None, help="job spec (YAML or JSON)")
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny random ViT, synthetic uint8 images")
    p.add_argument("--resume", action="store_true", help="continue from the job's saved training state")
    p.add_argument("--fresh", action="store_true", help="discard a saved training state and start again")
    p.add_argument("--force", action="store_true", help="run again even if every student has finished")
    p.add_argument("--out-root", default=None)
    p.add_argument("--queue-dir", default=None)
    p.add_argument("--device", default=None, help="default: cuda (cpu for --dry-run)")
    p.add_argument("--max-steps", type=int, default=None, help="cap the budget (debugging only)")
    a = p.parse_args(argv)
    if not a.spec and not a.dry_run:
        p.error("--spec is required (or --dry-run)")
    return a


def build_spec(a) -> dict:
    spec = _merge(DEFAULTS, DRY_RUN) if a.dry_run else copy.deepcopy(DEFAULTS)
    if a.spec:
        user = load_spec(a.spec)
        if a.dry_run and "students" in user:
            spec["students"] = []
        spec = _merge(spec, user)
    if a.out_root:
        spec["out_root"] = a.out_root
    if a.queue_dir:
        spec["queue_dir"] = a.queue_dir
    elif a.out_root:
        spec["queue_dir"] = str(Path(a.out_root) / "_queue")
    if a.max_steps:
        spec["steps"] = int(a.max_steps)
    if not spec["students"]:
        raise SystemExit("the job spec lists no students")
    if spec["structure"] is None and spec["model"] != "tiny":
        spec["structure"] = f"results/MUST-07/structure/{spec['model']}_s0.npz"
    return spec


def student_key(e: dict) -> str:
    from mirror.final.heal.students import student_folder

    return student_folder(e["method"], e.get("variant"))


def job_name(spec: dict) -> str:
    if spec.get("job"):
        return str(spec["job"])
    h = hashlib.sha1(json.dumps([{k: e.get(k) for k in ("method", "variant", "lr", "lr_select", "lr_from")}
                                 for e in spec["students"]], sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{spec['experiment']}_{h}"


# ----------------------------------------------------------------------------------------------
# Teacher, data, structure
# ----------------------------------------------------------------------------------------------
def build_teacher(spec: dict, device):
    from mirror.heal_structured import adapter_spec, build_adapter_from_spec

    name = spec["model"]
    if name == "tiny":
        aspec = adapter_spec("tiny", img_size=56, seed=0, dtype="float32")
    elif name.startswith("dinov2"):
        aspec = adapter_spec("dinov2", hub_repo="facebookresearch/dinov2", name=name, img_size=224)
    else:
        aspec = adapter_spec("registry", name=name, img_size=224)
    return build_adapter_from_spec(aspec, device), aspec


def build_data(spec: dict, device, dry: bool):
    from mirror.final.heal.data_gpu import LockstepData

    d = spec["data"]
    if dry:
        rng = np.random.default_rng(0)
        images = rng.integers(0, 256, size=(40, 64, 64, 3), dtype=np.uint8)
        rows = rng.permutation(40)
        return LockstepData(images, rows[:32], device, "stream", heldout_rows=rows[32:], prefetch=0), \
            {"source": "synthetic uint8 64 px (dry run)", "n_train": 32, "n_heldout": 8}
    held_images = None
    if d.get("sources"):          # training rows from several caches, read as one array (ConcatImages)
        from mirror.final.heal.data_gpu import ConcatImages

        arrays, trains, held = [], [], None
        hsrc = int(d.get("heldout_source", 0))
        offset = 0
        for i, s in enumerate(d["sources"]):
            arr = np.load(resolve(s["images"]), mmap_mode="r")
            with np.load(resolve(s["index"])) as z:
                trains.append(np.asarray(z[s["key"]], dtype=np.int64) + offset)
                if i == hsrc and d.get("heldout_key") and not d.get("heldout_images"):
                    held = np.asarray(z[d["heldout_key"]], dtype=np.int64) + offset
            arrays.append(arr)
            offset += len(arr)
        images = ConcatImages(arrays)
        train = np.concatenate(trains)
    else:
        images = np.load(resolve(d["images"]), mmap_mode="r")
        with np.load(resolve(d["index"])) as z:
            train = np.asarray(z[d["train_key"]])
            held = np.asarray(z[d["heldout_key"]]) if (d.get("heldout_key") and not d.get("heldout_index")) else None
    if d.get("heldout_images"):
        held_images = np.load(resolve(d["heldout_images"]), mmap_mode="r")
        with np.load(resolve(d.get("heldout_index") or d["index"])) as z:
            held = np.asarray(z[d["heldout_key"]])
    if d.get("max_train"):
        train = train[:int(d["max_train"])]
    ld = LockstepData(images, train, device, d.get("mode", "auto"), heldout_rows=held, held_images=held_images,
                      max_resident_gb=d.get("max_resident_gb"), prefetch=int(spec["heal_cfg"].prefetch))
    log(f"data: {images.shape} {images.dtype}; {len(train)} train rows, {0 if held is None else len(held)} held-out "
        f"rows; mode {ld.mode} ({ld.resident_bytes / 2 ** 30:.2f} GiB resident)")
    return ld, {"images": d.get("images"), "index": d.get("index"), "train_key": d.get("train_key"),
                "sources": d.get("sources"), "heldout_key": d.get("heldout_key"), "n_train": int(len(train)),
                "n_heldout": 0 if held is None else int(len(held)), "mode": ld.mode}


def tiny_structure(adapter, seed: int = 0):
    """A tiny SpanStructure from random parts (dry runs; the canonical structures are fitted on the VM)."""
    from mirror.attention import attn_dims
    from mirror.m1 import random_head_structure, random_involution_fixing_ones
    from mirror.matching import random_pairing
    from mirror.mirror_family import SpanStructure

    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=seed + 3)
    perms = [np.asarray(random_pairing(h, h // 3, seed=seed + 10 + b), dtype=np.int64) for b in range(L)]
    heads = [random_head_structure(H, dh, (1, 2, 0, 1)[b % 4], seed=seed + 20 + b) for b in range(L)]
    return SpanStructure(model="tiny", fit_seed=0, A_layers=[A] * (L + 1), perms=perms, heads=heads,
                         A_star={k: A for k in range(1, L + 1)},
                         reg_perm={k: np.zeros(0, dtype=np.int64) for k in range(1, L + 1)},
                         meta={"source": "random parts (dry run)"})


def resolve_logits_fn(text):
    if not text:
        return None
    mod, _, fn = str(text).partition(":")
    return getattr(importlib.import_module(mod), fn)


# ----------------------------------------------------------------------------------------------
# Learning rates
# ----------------------------------------------------------------------------------------------
def lr_from(path) -> float:
    p = resolve(path)
    for cand in (p, p / "lr_selection" / "lr_selection.json", p / "lr_selection.json", p / "metrics.json"):
        if cand.is_file():
            d = json.loads(cand.read_text(encoding="utf-8"))
            if "best_lr" in d:
                return float(d["best_lr"])
            if "lr" in d:
                return float(d["lr"])
            if isinstance(d.get("config"), dict) and "lr" in d["config"]:
                return float(d["config"]["lr"])
    raise FileNotFoundError(f"lr_from: no learning rate found under {p}")


def finished_selection(path: Path, lrs, budget: int, seed: int) -> dict | None:
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if sorted(float(v) for v in d.get("lrs", [])) != sorted(float(v) for v in lrs):
        return None
    if int(d.get("budget_steps", -1)) != int(budget) or int(d.get("seed", -1)) != int(seed):
        return None
    return d


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def run(a) -> dict:
    t_start = time.time()
    spec = build_spec(a)
    dry = bool(a.dry_run)
    from mirror.final.prereg import require_prereg, write_final_meta

    prereg = require_prereg(allow_missing=True if dry else None)       # before anything touches the GPU
    dev_name = a.device or ("cpu" if dry else "cuda")
    device = torch.device(dev_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available; use --dry-run on a CPU machine")
        if spec.get("gpu_frac"):
            from mirror.utils import cuda_index
            torch.cuda.set_per_process_memory_fraction(float(spec["gpu_frac"]), cuda_index(device))
        torch.backends.cuda.matmul.allow_tf32 = False                  # configs/must09.yaml compute.tf32: false
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    from mirror.final.heal.multistudent import (EvalQueue, LockstepStudent, default_hook_steps, select_lr_lockstep,
                                                train_lockstep)
    from mirror.final.heal.students import BuildContext, build_student
    from mirror.heal_structured import write_spec

    seed = int(spec["seed"])
    must09 = load_config(REPO_ROOT / "configs" / "must09.yaml", tiny=dry)
    set_seed(seed)
    teacher, aspec = build_teacher(spec, device)
    L = int(teacher.depth)
    hdict = {**must09["heal"], **(spec.get("heal") or {})}
    hdict.update(blocks=list(range(L)), seed=seed, train_scope="all", micro_batch=int(spec["micro_batch"]))
    if spec.get("steps") is not None:
        hdict["max_steps"] = int(spec["steps"])
    hdict.pop("checkpoint_path", None)
    hcfg = HealConfig.from_dict(hdict)
    spec["heal_cfg"] = hcfg
    ldata, data_info = build_data(spec, device, dry)
    del spec["heal_cfg"]
    n_train = len(ldata.train_rows)
    total = int(hcfg.max_steps) if hcfg.max_steps is not None else int(hcfg.epochs * (n_train // hcfg.batch_size))
    model_name = spec["model"]
    out_root = resolve(spec["out_root"])
    group = str(spec.get("group") or spec["experiment"])
    exp_dir = out_root / group / model_name
    job_dir = exp_dir / "_jobs" / job_name(spec) / f"s{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    state_file = job_dir / "train_state.pt"
    if state_file.exists() and not (a.resume or a.fresh):
        raise SystemExit(f"{state_file} exists: pass --resume to continue the interrupted job or --fresh to restart")
    if a.fresh and state_file.exists():
        state_file.unlink()

    # probe images for the conversion checks: centre views of the first calibration rows (training images)
    from mirror.heal import centre_view

    probe_rows = np.sort(ldata.train_rows)[:8]
    probe = centre_view(next(ldata.chunks(probe_rows, 8, "train")), hcfg.crop_size, torch.float32)
    struct = tiny_structure(teacher) if dry else spec["structure"]
    fit_loader = None
    if dry:
        from mirror.variants import TinyProtocol

        proto = TinyProtocol()
        fit_loader = proto.fit_loader
    gate_b = resolve(spec["gate_b"]) if spec.get("gate_b") and resolve(spec["gate_b"]).exists() else None
    ctx = BuildContext(adapter=teacher, model_name=model_name, adapter_spec=aspec, device=device, structure=struct,
                       random_seed=int(spec["random_seed"]),
                       lowrank_factors=str(must09.get("lowrank_factors", "balanced")),
                       gate_b=gate_b, fit_seed=int(must09["compressor"].get("fit_seed", 0)), fit_loader=fit_loader,
                       cache_dir=spec["cache_dir"], probe_images=probe,
                       conversion_tol=float(must09["checks"].get("conversion_tol", 1e-3)),
                       adopted_structures={"astar": None if dry else spec["structure"],
                                           **(spec.get("adopted_structures") or {})})
    logits_fn = resolve_logits_fn(spec.get("logits_fn"))
    if spec.get("blockdrop_score_fn"):
        ctx.extra["blockdrop_score_fn"] = resolve_logits_fn(spec["blockdrop_score_fn"])(ctx)

    entries = [dict(e) for e in spec["students"]]
    keys = [student_key(e) for e in entries]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"two students share a folder: {keys}")
    run_dirs = {k: exp_dir / k / f"s{seed}" for k in keys}

    def finished(k) -> bool:
        m = run_dirs[k] / "metrics.json"
        if not m.is_file():
            return False
        d = json.loads(m.read_text(encoding="utf-8"))
        return (int(d.get("steps_done", -1)) == total and not d.get("diverged") and d.get("job") == job_name(spec)
                and (run_dirs[k] / "DONE").is_file())

    if not a.force and all(finished(k) for k in keys):
        log(f"fp_heal: every student of {job_name(spec)} has finished ({exp_dir}); --force runs again")
        return {"skipped": True, "job_dir": str(job_dir), "run_dirs": {k: str(v) for k, v in run_dirs.items()}}

    builds = {}
    for e, k in zip(entries, keys):
        opts = {kk: e[kk] for kk in BUILDER_KEYS if kk in e}
        log(f"fp_heal: building {k} ({e['method']})")
        builds[k] = build_student(e["method"], ctx, **opts)

    # learning rates: given, reused (lr_from) or selected here in one lockstep job of (method, rate) trials
    lrs, selections = {}, {}
    cands = {}
    for e, k in zip(entries, keys):
        sel = e.get("lr_select")
        if sel:
            budget = int(sel.get("budget_steps", spec["lr_selection"].get("budget_steps", 200)))
            out = run_dirs[k] / "lr_selection"
            done = None if a.force else finished_selection(out / "lr_selection.json", sel["lrs"], budget, seed)
            if done is not None:
                selections[k] = done
                log(f"fp_heal: {k}: learning-rate selection finished before ({out / 'lr_selection.json'})")
            else:
                sb = builds[k]

                def make(sb=sb):
                    m, cons = sb.fresh()
                    return {"model": m, "constraints": cons, "mode": copy.deepcopy(sb.targets), "blocks": sb.blocks,
                            "train_scope": sb.train_scope, "compile": bool(spec.get("compile"))}
                cands[k] = {"lrs": sel["lrs"], "make": make, "out_dir": out, "budget": budget,
                            "trial_dir": (lambda lr, k=k: out_root / "lr" / model_name / f"{k}_lr{float(lr):.0e}"
                                          / f"s{seed}"),
                            "tags": {"experiment": spec["experiment"], "group": "lr", "model": model_name,
                                     "method": e["method"], "variant": e.get("variant"), "seed": seed}}
        elif e.get("lr_from"):
            lrs[k] = lr_from(e["lr_from"])
            selections[k] = {"reused_from": str(e["lr_from"]), "best_lr": lrs[k]}
        elif e.get("lr") is not None:
            lrs[k] = float(e["lr"])
            selections[k] = {"given": True, "best_lr": lrs[k]}
        else:
            raise SystemExit(f"{k}: give lr, lr_select or lr_from")
    budgets = {c["budget"] for c in cands.values()}
    for budget in sorted(budgets):
        trials = {k: c for k, c in cands.items() if c["budget"] == budget}
        res = select_lr_lockstep(teacher, trials, ldata, hcfg, budget_steps=budget,
                                 job_dir=job_dir / f"lr_selection_{budget}", resume=a.resume,
                                 state_every=int(spec.get("state_every") or 0), logits_fn=logits_fn)
        selections.update(res)
        (job_dir / f"lr_selection_{budget}" / "train_state.pt").unlink(missing_ok=True)   # selections are written
    for e, k in zip(entries, keys):
        if k in selections and e.get("lr_select"):
            choose = e["lr_select"].get("choose", True)
            lrs[k] = float(selections[k]["best_lr"]) if choose else float(e["lr"])

    # the final job
    queue = EvalQueue(resolve(spec["queue_dir"]), int(spec["eval"].get("n_query_full", 50000)),
                      int(spec["eval"].get("n_query_partial", 10000)),
                      fp32_last=bool(spec["eval"].get("fp32_last", True)))
    students = []
    for e, k in zip(entries, keys):
        sb = builds[k]
        sb.fresh = None                                  # the trials are done: free the compressed copy
        rd = run_dirs[k]
        sp = dict(sb.spec)
        sp.update(experiment=spec["experiment"], seed=seed, state_file=f"ckpt/step{total}.pt")
        sp.setdefault("fp_heal", {}).update(variant=e.get("variant"), lr=lrs[k], job=job_name(spec),
                                            lr_selection=selections.get(k), params=sb.params)
        write_spec(rd, sp, sb.structure)
        st_ = sb.structure or {}
        if sp.get("family") != "mirror" and {"A_star", "perms", "maps_after", "reg_perm"} <= set(st_):
            from mirror.heal_structured import save_structure      # partial span, Mirror + SVD: structure.npz too

            save_structure(rd / "structure.npz", st_["A_star"], st_["perms"], st_["maps_after"], st_.get("reg_perm"),
                           st_.get("blocks", sp.get("blocks")))
        students.append(LockstepStudent(name=k, model=sb.model, constraints=sb.constraints, mode=sb.targets,
                                        lr=lrs[k], blocks=sb.blocks, run_dir=rd, train_scope=sb.train_scope,
                                        compile=bool(spec.get("compile")),
                                        tags={"experiment": spec["experiment"], "group": group, "model": model_name,
                                              "method": e["method"], "variant": e.get("variant"), "seed": seed,
                                              **({"fp32": bool(e["fp32"])} if "fp32" in e else {})}))
    (job_dir / "job.json").write_text(json.dumps({**spec, "job": job_name(spec), "total_steps": total,
                                                  "heal": asdict(hcfg), "lrs": lrs}, indent=2, default=str),
                                      encoding="utf-8")
    hooks = default_hook_steps(total, spec.get("hook_steps") or [])
    res = train_lockstep(teacher, students, ldata, hcfg, job_dir=job_dir, hook_steps=hooks, queue=queue,
                         state_every=int(spec.get("state_every") or 0), resume=a.resume, logits_fn=logits_fn,
                         ckpt_meta={"job": job_name(spec)}, log_prefix=f"heal {spec['experiment']} s{seed}")
    wall = time.time() - t_start
    for e, k in zip(entries, keys):
        rd = run_dirs[k]
        m = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
        m.update(job=job_name(spec), method=e["method"], variant=e.get("variant"), lr=lrs[k],
                 lr_selection=selections.get(k), params=builds[k].params, build=builds[k].info,
                 complete=not m.get("diverged"))
        (rd / "metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
        write_final_meta(rd, experiment=str(spec["experiment"]),
                         config={"job": {kk: v for kk, v in spec.items() if kk != "students"}, "student": e,
                                 "heal": asdict(hcfg), "args": vars(a)},
                         seeds={"heal": seed, "structure_fit": ctx.fit_seed,
                                "random_structure": int(e.get("random_seed", spec["random_seed"]))
                                if e["method"] in ("random_structure", "pair_random") else None},
                         models={model_name: aspec}, device=device, wall_seconds=wall,
                         peak_memory_gb=res["job"].get("peak_memory_gb"),
                         extra={"prereg": prereg, "job_dir": str(job_dir), "data": data_info, "run_dir": str(rd),
                                "dry_run": dry})
    write_final_meta(job_dir, experiment=str(spec["experiment"]), config={"job": spec, "heal": asdict(hcfg)},
                     seeds={"heal": seed}, models={model_name: aspec}, device=device, wall_seconds=wall,
                     peak_memory_gb=res["job"].get("peak_memory_gb"),
                     extra={"prereg": prereg, "data": data_info, "students": keys, "lrs": lrs, "dry_run": dry})
    if state_file.exists() and all(not res["students"][k]["diverged"] for k in keys):
        state_file.unlink()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for k in keys:                                  # written last: the manifest's done: files
        if not res["students"][k]["diverged"]:
            (run_dirs[k] / "DONE").write_text(stamp, encoding="utf-8")
    diverged = [k for k in keys if res["students"][k]["diverged"]]
    (job_dir / "DONE").write_text(json.dumps({"finished_utc": stamp, "diverged": diverged}), encoding="utf-8")
    log(f"fp_heal: job {job_name(spec)} s{seed} done in {wall / 60:.1f} min; outputs under {exp_dir}")
    return {"job_dir": str(job_dir), "run_dirs": {k: str(v) for k, v in run_dirs.items()}, "lrs": lrs,
            "selections": selections, "result": res}


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
