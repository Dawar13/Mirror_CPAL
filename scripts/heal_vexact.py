#!/usr/bin/env python
"""V-exact healing (PAPER_EXPERIMENTS.md, Stage 3 "V-exact full conversion, healed"; also the V-exact
arm of the M3 grid): the V-exact student (one operator A* over blocks 1..k, exactly mirror-equivariant
over the span; k = 12 is the full conversion, 100% flip consistency) healed by distillation from the
original model with the M3-pilot recipe.

    python scripts/heal_vexact.py --config configs/m3_vexact.yaml --k 12     # full conversion
    python scripts/heal_vexact.py --config configs/m3_vexact.yaml --k 6
    python scripts/heal_vexact.py --config configs/m3_vexact.yaml --tiny     # CPU smoke test

Steps:
 1. teacher = the original model; data = the M3 healing cache (100k train images; 5k held-out train
    images for the learning-rate choice, never validation);
 2. structure: A* and the register pairing for this k from the M1 structure file (structure.heads,
    runs/m1/m1_structure.npz; scripts/m1_attention.py fitted A* on stream layers 0..k, each layer
    normalised to equal energy), the neuron pairings from the X2 operator file (structure.operators)
    and the M1 head maps (same loaders as scripts/heal.py);
 3. student = mirror.heal_vexact.build_vexact_student: constraints re-project MLP and attention with
    A_in = A_out = A* after every optimiser step and, with vexact.train_embeddings (default), the
    embeddings too; the even head (k = depth) is a fixed buffer. Targets: mirror.heal_vexact.VExactTeacher
    (original model; block outputs centred and mirror-symmetrised with A*; at k = depth the final
    features mirror-averaged). See that module for the reasoning;
 4. learning rate chosen on the held-out train images, then the full budget (mirror.heal);
 5. Tier-1 k-NN with the paper protocol (scripts/heal.py evaluate_final_knn): original (cache),
    zero-shot V-exact and healed V-exact, bf16;
 6. exactness after healing (fp32, TF32 off, validation images): mirrored tokens against A* times the
    original tokens at every layer of the span (M1 probe, tolerance 1e-4 relative);
 7. k = depth only: fp32 flip check after healing (M1 protocol: 10 bank images per class, 5,000
    validation queries): must be 100%.
Writes <out_dir>/k<k>/: lr_selection/, final/ (train_log.csv, heldout_log.csv, metrics.json,
trained_blocks.pt), knn.json, exactness.json, flipcheck.json (k = depth), summary.json and SUMMARY.md.
Every output records the seed and the git hash. With vexact.train_embeddings the checkpoint holds the
whole student (train_scope "all"); load it into a fresh build_vexact_student with
mirror.heal.load_trained_blocks.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25)  # shared GPU: cap before anything is allocated; config sets the final cap

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.data import make_loader, make_splits, normalize_for, synthetic_dataset  # noqa: E402
from mirror.heal import DEFAULT_LRS, HealConfig, exact_involution, heal, heal_with_lr_selection  # noqa: E402
from mirror.heal_vexact import (FINAL_TARGETS, VExactTeacher, build_vexact_student, flip_check,  # noqa: E402
                                span_exactness, train_scope_for, vexact_param_counts)
from mirror.m1 import random_involution_fixing_ones, register_info  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, load_config, log, set_seed, setup_torch  # noqa: E402

METHOD = "vexact"


def load_heal_cli():
    """scripts/heal.py as a module: teacher, data, structure loaders and the final k-NN are the pilot's."""
    key = "_m3_heal_cli"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().with_name("heal.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


heal_cli = load_heal_cli()   # sets its own 0.2 import-time cap; setup_torch applies the config's cap in main()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_vexact.yaml"))
    p.add_argument("--k", type=int, default=None, help="convert blocks 1..k (overrides `k`; the depth = full conversion)")
    p.add_argument("--tiny", action="store_true", help="tiny random ViT + synthetic data on CPU")
    p.add_argument("--out-dir", default=None, help="overrides `out_dir`; results go to <out_dir>/k<k>/")
    p.add_argument("--lr", type=float, default=None, help="skip learning-rate selection and use this rate")
    p.add_argument("--max-steps", type=int, default=None, help="cap the full run (debugging only)")
    p.add_argument("--memory-fraction", type=float, default=None, help="overrides compute.gpu_memory_fraction")
    p.add_argument("--final-target", choices=FINAL_TARGETS, default=None, help="overrides vexact.final_target")
    p.add_argument("--freeze-embeddings", action="store_true",
                   help="keep the projected embeddings frozen (vexact.train_embeddings false)")
    return p.parse_args(argv)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# Structure and check data
# ----------------------------------------------------------------------------------------------
def load_vexact_structure(cfg: dict, adapter, k: int) -> dict:
    """A* and the register pairing for span 0..k (M1 structure file), neuron pairings and head maps
    (scripts/heal.py load_structure). Tiny: a random span operator and the tiny random structure."""
    base = heal_cli.load_structure(cfg, adapter)
    if base.get("heads") is None:
        raise ValueError("structure.attention must be true: V-exact converts MLP and attention")
    d = adapter.embed_dim
    R = register_info(adapter.model, adapter)["R"]
    out = {"perms": base["perms"], "heads": base["heads"], "operators_source": base["source"],
           "heads_source": base["heads_source"]}
    if cfg.get("tiny_mode"):
        A = random_involution_fixing_ones(d, d // 2 + 4, seed=int(cfg["seed"]) * 100 + 90 + int(k))
        out.update(A_star=A, reg_perm=np.arange(R, dtype=np.int64), span_source="tiny random span operator (smoke test)",
                   span_sha256=None)
        return out
    path = heal_cli._resolve(cfg["structure"]["heads"])
    with np.load(path) as z:
        if "A_star" not in z.files:
            raise ValueError(f"{path} has no span operators (A_star); rerun scripts/m1_attention.py")
        ks = [int(v) for v in z["ks"]]
        model_name = str(z["model"]) if "model" in z.files else None
        if k not in ks:
            raise ValueError(f"{path} holds span operators for k in {ks}, not {k}: rerun scripts/m1_attention.py "
                             f"with --ks including {k}")
        i = ks.index(k)
        A = exact_involution(z["A_star"][i])
        reg_perm = np.asarray(z["reg_perm"][i], dtype=np.int64)
    if model_name is not None and model_name != cfg["model"]["name"]:
        raise ValueError(f"{path} was fitted on {model_name}, not {cfg['model']['name']}")
    if len(reg_perm) != R:
        raise ValueError(f"{path}: register pairing has {len(reg_perm)} entries, the model has {R} registers")
    out.update(A_star=A, reg_perm=reg_perm, span_source=str(path), span_sha256=sha256_file(path))
    return out


def check_data(cfg: dict, device) -> dict:
    """Loaders for the checks after healing: flip-check bank (train images) and queries, and probe
    images for exactness (validation). Same subsets as scripts/m1_attention.py."""
    c = cfg.get("checks", {}) or {}
    bs, nw = int(c.get("batch_size", 128)), int(c.get("num_workers", 4))
    if cfg.get("tiny_mode"):
        n_cls = int(c.get("num_classes", 4))
        per = {"bank": int(c.get("bank_per_class", 8)), "query": int(c.get("query_per_class", 4)),
               "probe": int(c.get("probe_per_class", 2))}
        images, labels = synthetic_dataset(n_cls, sum(per.values()), int(cfg["heal"]["crop_size"]), seed=int(cfg["seed"]) + 2)
        sp = make_splits(labels, per)
        return {"bank": make_loader(images, labels, sp["bank"], bs), "query": make_loader(images, labels, sp["query"], bs),
                "probe": make_loader(images, labels, sp["probe"], bs), "num_classes": n_cls, "protocol": "tiny synthetic"}
    from mirror.protocol import Protocol

    pin = device.type == "cuda"
    proto = Protocol(heal_cli._resolve(c.get("cache_dir", "cache")), bs, 16, nw, pin)
    bank_rows = proto.index["bank_rows"]
    sub = make_splits(proto.train_labels[bank_rows], {"b": int(c.get("flip_bank_per_class", 10))})["b"]
    return {"bank": make_loader(proto.train, proto.train_labels, bank_rows[sub], bs, nw, pin),
            "query": proto.query_loader(int(c.get("flip_queries", 5000))), "probe": proto.heldout_loader(),
            "num_classes": 1000, "protocol": "v2"}


def probe_batches(adapter, loader, n: int, device) -> list:
    out, got = [], 0
    for imgs, _ in loader:
        x = normalize_for(adapter, imgs.to(device))
        out.append(x[: n - got])
        got += out[-1].shape[0]
        if got >= n:
            break
    return out


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def _f(x, nd: int = 2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:.{nd}f}" if math.isfinite(x) else "n/a"


def _e(x) -> str:
    try:
        return f"{float(x):.1e}"
    except (TypeError, ValueError):
        return "n/a"


def _dtype(name: str) -> str:
    return str(name).replace("torch.", "")


def write_summary_md(out: Path, s: dict) -> Path:
    k, L = s["k"], s["depth"]
    g = s["git_hash"]
    info, fin, sel = s["method_info"], s["final"], s["lr_selection"]
    hc = fin["config"]
    tiny = bool(s["config"].get("tiny_mode"))
    lines = [
        f"# V-exact healing, k = {k}" + (" (full conversion)" if k == L else "")
        + (" (TINY TEST, random weights and synthetic data)" if tiny else ""),
        "",
        f"Model `{s['config']['model']['name'] if not tiny else 'tiny random timm ViT'}` ({L} blocks); blocks 1 to {k} "
        f"converted with one operator A* (d+ {info['d_plus']}, d- {info['d_minus']}), MLP and attention, exactly "
        f"mirror-equivariant over the span" + (" with the even head (final features of an image and its mirror equal)"
                                              if k == L else "") + ". "
        f"A* and register pairing: `{s['structure']['span_source']}`; neuron pairings: `{s['structure']['operators_source']}`; "
        f"head maps: `{s['structure']['heads_source']}`. Seed {s['seed']}; git `{g[:10]}`"
        f"{' (uncommitted changes)' if s.get('git_dirty') else ''}. Written {s['date']}.",
        "",
        f"Recipe (as the M3 pilot): {fin['steps_done']} steps of {hc['batch_size']} images "
        f"({fin['images_seen']} images seen, {s['data']['n_train']} healing images, train split), AdamW, cosine schedule, "
        f"{'bf16 autocast' if fin['amp'] == 'bf16' else 'fp32'}; learning rate {sel['best_lr']:.0e} "
        + ("given on the command line" if sel.get("skipped") else
           f"chosen from {{{', '.join(f'{v:.0e}' for v in sel['lrs'])}}} on {s['data']['n_heldout']} held-out train images")
        + f". Targets: block outputs {s['teacher']['block_targets']}; final target {s['teacher']['final_target']}. "
        f"Embeddings: {s['embeddings']}.",
        "",
    ]
    kn = s.get("knn")
    if kn:
        prec = "bf16" if (kn["amp"] == "bf16" and s["env"]["device"].startswith("cuda")) else "fp32"
        lines += [f"| model | k-NN top-1 | drop (points) | flip consistency ({prec}) |", "|---|---|---|---|",
                  f"| original | {_f(kn['original']['top1'])} | | {_f(kn['original']['flip_consistency'])} |"]
        for key, label in (("zero_shot", "V-exact, zero-shot"), ("healed", "V-exact, healed")):
            if key in kn:
                r = kn[key]
                lines.append(f"| {label} | {_f(r['top1'])} | {_f(r['drop_points'])} | {_f(r['flip_consistency'])} |")
        lines += ["", f"k-NN: {kn['protocol']} protocol ({kn['healed']['n_bank']} bank, {kn['healed']['n_query']} queries), "
                      f"k = {kn['k']}, T = {kn['temperature']}, {prec} features.", ""]
    hi, hf = fin.get("heldout_initial") or {}, fin.get("heldout_final") or {}
    lines.append(f"Held-out distillation loss ({fin['heldout_images']} held-out train images, V-exact targets): "
                 f"before {_f(hi.get('total'), 4)}, after {_f(hf.get('total'), 4)}.")
    ex = s["exactness"]
    lines.append(f"Exactness after healing ({_dtype(ex['dtype'])}, {ex['n_images']} validation images): largest relative error "
                 f"inside the span {_e(ex['max_rel_inside'])} (tolerance {ex['tol']:g}): **{'PASS' if ex['pass'] else 'FAIL'}**.")
    fl = s.get("flipcheck")
    if fl:
        lines.append(f"Flip check after healing ({_dtype(fl['dtype'])}, TF32 {'on' if fl['tf32_matmul'] else 'off'}; {fl['n_query']} validation "
                     f"queries, bank {fl['n_bank']} train images): flip consistency {_f(fl['flip_consistency'])}%, "
                     f"**{'PASS' if fl['pass'] else 'FAIL'}** (must be 100%); largest relative CLS feature difference "
                     f"between an image and its mirror {_e(fl['cls_feature_rel_max'])}.")
    ch = fin.get("constraint_checks", {})
    parts = []
    for key, v in ch.items():
        if "max_rel_violation" in v:
            parts.append(f"{key.split(':', 1)[1]} {_e(v['max_rel_violation'])}")
        elif "max_share" in v:
            parts.append(f"{key.split(':', 1)[1]} embedding share {_e(v['max_share'])}")
    if parts:
        lines.append("Constraint checks at the end of training (0 = exact): " + "; ".join(parts) + ".")
    lines += [
        "",
        f"Cost: the converted blocks keep {info['converted_params_kept']:,} of {info['converted_params_dense']:,} matrix "
        f"parameters ({_f(info['converted_kept_frac'], 3)}); whole model {info['model_params']:,} parameters "
        f"({_f(info['model_params_frac'], 3)} of {info['model_params_original']:,}; biases and embeddings counted dense, as "
        f"in M1). Trainable while healing: {fin['n_trainable']:,} parameters. Final run {_f(fin['seconds_train'], 0)} s "
        f"of training" + (f", peak GPU memory {_f(fin['peak_memory_gb'], 2)} GB" if fin.get("peak_memory_gb") is not None else "")
        + f"; whole script {_f(s['timing']['total_s'], 0)} s.",
        "",
        "Notes:",
        "- One seed. Held-out losses use the V-exact targets (centred, symmetrised), so they are not comparable with the "
        "M3 pilot's.",
        "- In bf16 the mirrored computation rounds differently, so the bf16 flip consistency of the fully converted model "
        "can fall slightly below 100%; the exact check is the fp32 one.",
        "",
        "Files: summary.json, knn.json, exactness.json" + (", flipcheck.json" if fl else "")
        + ", final/ (metrics.json, train_log.csv, heldout_log.csv, trained_blocks.pt), lr_selection/.",
    ]
    text = "\n".join(lines) + "\n"
    assert "—" not in text
    path = out / "SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    cfg = load_config(a.config, tiny=a.tiny)
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    if a.memory_fraction is not None:
        cfg["compute"]["gpu_memory_fraction"] = float(a.memory_fraction)
    vx = dict(cfg.get("vexact", {}) or {})
    if a.final_target:
        vx["final_target"] = a.final_target
    if a.freeze_embeddings:
        vx["train_embeddings"] = False
    cfg["vexact"] = vx
    k = int(a.k if a.k is not None else cfg["k"])
    cfg["k"], cfg["method"] = k, METHOD
    device = setup_torch(cfg)  # GPU memory cap and TF32 off (config)
    seed = int(cfg["seed"])
    set_seed(seed)
    timer = StageTimer()
    out = heal_cli._resolve(cfg["out_dir"]) / f"k{k}"
    out.mkdir(parents=True, exist_ok=True)
    git = git_info()
    log(f"V-exact heal: k {k}, out {out}, git {git['git_hash'][:10]}{' (dirty)' if git.get('git_dirty') else ''}")

    with timer.stage("load"):
        teacher = heal_cli.build_teacher(cfg, device)
        L = teacher.depth
        if not 1 <= k <= L:
            raise ValueError(f"k must be in [1, {L}]")
        data = heal_cli.load_heal_data(cfg)
        st = load_vexact_structure(cfg, teacher, k)
    A, reg_perm = st["A_star"], st["reg_perm"]
    train_emb = bool(vx.get("train_embeddings", True))
    symmetrise = bool(vx.get("symmetrise_targets", True))
    final_target = str(vx.get("final_target", "symmetrised"))
    blocks = list(range(k))
    vteacher = VExactTeacher(teacher, A, k, reg_perm, symmetrise=symmetrise, final_target=final_target)

    def make_student():
        m, cons, _ = build_vexact_student(teacher, A, k, st["perms"], st["heads"], reg_perm, train_embeddings=train_emb)
        return m, cons

    info = vexact_param_counts(teacher, A, k, st["perms"], st["heads"])
    log(f"A* d+ {info['d_plus']} d- {info['d_minus']}; converted blocks keep {info['converted_kept_frac']:.3f} of their "
        f"matrix parameters; embeddings {'trained (re-projected)' if train_emb else 'frozen'}; targets: {vteacher.describe()}")
    hcfg = HealConfig.from_dict({**cfg.get("heal", {}), "blocks": blocks, "seed": seed,
                                 "train_scope": train_scope_for(train_emb)})
    if a.max_steps:
        hcfg = replace(hcfg, max_steps=int(a.max_steps))

    sel_cfg = cfg.get("lr_selection", {}) or {}
    with timer.stage("heal"):
        if a.lr is not None:
            student, cons = make_student()
            final = heal(vteacher, student, data, replace(hcfg, lr=float(a.lr)), cons, out / "final")
            selection = {"skipped": True, "best_lr": float(a.lr), "lrs": [float(a.lr)]}
        else:
            res = heal_with_lr_selection(vteacher, make_student, data, hcfg,
                                         lrs=[float(x) for x in sel_cfg.get("lrs", DEFAULT_LRS)],
                                         budget_steps=int(sel_cfg.get("budget_steps", 100)), out_dir=out)
            selection, final, student = res["lr_selection"], res["final"], res["student"]
    student.eval()

    knn_res = None
    if (cfg.get("knn", {}) or {}).get("enabled", True):
        with timer.stage("knn"):
            knn_res = heal_cli.evaluate_final_knn(cfg, teacher, student, make_student, device, out, git)

    ccfg = cfg.get("checks", {}) or {}
    tags = {"seed": seed, "git_hash": git["git_hash"]}
    with timer.stage("checks"):
        cd = check_data(cfg, device)
        xs = probe_batches(teacher, cd["probe"], int(ccfg.get("exact_images", 64)), device)
        exact = {**span_exactness(teacher, student, xs, A, k, reg_perm, tol=float(ccfg.get("exact_tol", 1e-4))), **tags}
        with open(out / "exactness.json", "w", encoding="utf-8") as fh:
            json.dump(exact, fh, indent=2)
        log(f"exactness after healing: max relative error inside the span {exact['max_rel_inside']:.2e} "
            f"({'PASS' if exact['pass'] else 'FAIL'})")
        flip = None
        if k == L and ccfg.get("flip_check", True):
            kc = cfg.get("knn", {}) or {}
            flip = {**flip_check(teacher, student, cd["bank"], cd["query"], xs, cd["num_classes"], device,
                                 k=int(kc.get("k", 20)), temperature=float(kc.get("temperature", 0.07))),
                    "protocol": cd["protocol"], **tags}
            with open(out / "flipcheck.json", "w", encoding="utf-8") as fh:
                json.dump(flip, fh, indent=2)
            log(f"fp32 flip check after healing: {flip['flip_consistency']:.2f}% ({'PASS' if flip['pass'] else 'FAIL'})")

    final = {kk: v for kk, v in final.items() if kk != "history"}
    summary = {
        "method": METHOD, "k": k, "depth": L, "blocks": blocks, "seed": seed, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method_info": info,
        "structure": {kk: st[kk] for kk in ("span_source", "span_sha256", "operators_source", "heads_source")},
        "reg_perm": [int(v) for v in reg_perm], "teacher": vteacher.describe(),
        "embeddings": "trained, re-projected every step" if train_emb else "frozen after projection",
        "data": {"n_train": int(len(data.train_rows)),
                 "n_heldout": int(len(data.heldout_rows)) if data.heldout_rows is not None else 0},
        "lr_selection": selection, "final": final, "knn": knn_res, "exactness": exact, "flipcheck": flip,
        "config": cfg, **git, "env": env_info(device), "timing": {"stages": timer.stages, "total_s": timer.total()},
    }
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    md = write_summary_md(out, summary)
    log(f"V-exact heal done: k {k}, lr {selection['best_lr']:.0e}"
        + (f", k-NN healed {knn_res['healed']['top1']:.2f}" if knn_res else "") + f"; wrote {md}")
    return summary


if __name__ == "__main__":
    main()
