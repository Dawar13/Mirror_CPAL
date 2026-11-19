#!/usr/bin/env python
"""Final phase: downstream evaluation (Tier-2 linear probe, ImageNet-V2, deduplicated Sketch) of healed runs in either
run layout (FINAL_PHASE_PLAN.md v2, A10: downstream for SVD-LLM seeds 1 and 2).

What exists: scripts/tier2_linear.py (--healed-run NAME=DIR) and scripts/core07_shift.py (--healed-run NAME=DIR, or
--must09-dir with --best-seeds 1,2 and --gate-b naming svdllm) already take MUST-09-style run directories, through
mirror.heal_structured.load_healed_model: it needs <run>/spec.json in the "must09-healed-v1" format and the state file
that spec names (healed_state.pt). So the SVD-LLM seeds of a MUST-09-style run (scripts/must09_heal.py, Gate P) need
no wrapper:
    python scripts/tier2_linear.py --model dinov2_vitb14_reg --variants "" \
        --healed-run best_svdllm_s1=<must09 run>/svdllm/s1 --healed-run best_svdllm_s2=<must09 run>/svdllm/s2 --out ...
    python scripts/core07_shift.py --model dinov2_vitb14_reg --must09-dir <must09 run> --mirror-seeds "" \
        --best-seeds 1,2 --probe-from <the tier2_linear --out>
A final-phase run directory (runs/final/<experiment>/<model>/<method>/s<seed>/ with ckpt/step<S>.pt,
docs/final_phase_design.md Section 4.4) keeps a MUST-09-format spec.json (mirror/final/heal/students.py) but no single
state file: its checkpoints are ckpt/step<S>.pt, so load_healed_model cannot choose the step. This wrapper replaces
mirror.heal_structured.load_healed_model, for this process only, by a bridge that calls
mirror.final.heal.students.load_student(run_dir, step, device) (--loader) for every directory with a ckpt/ folder and
keeps the original loader for every other (MUST-09-style) directory, then runs the named script with the arguments
after "--":

    python scripts/fp_eval_downstream.py --script tier2_linear --step 780 -- --model dinov2_vitb14_reg --variants "" \
        --healed-run best_svdllm_s1=runs/final/A05/dinov2_vitb14_reg/svdllm/s1 --out runs/final/A10/t2
    python scripts/fp_eval_downstream.py --script core07_shift --step 780 -- --model dinov2_vitb14_reg --variants "" \
        --healed-run best_svdllm_s1=runs/final/A05/dinov2_vitb14_reg/svdllm/s1 --probe-from runs/final/A10/t2 \
        --out runs/final/A10/core07

--step S is the checkpoint of every final-phase directory (default: the loader's own default); --run-step DIR=S sets it
per directory. If the loader returns no adapter, the bridge returns a stand-in with the attributes tier2_linear
compares (depth, embed_dim, n_prefix_tokens, patch_size) read from the model. The variant names decide the rows of
the downstream tables (core07 treats best_<method>_s<seed> as the best baseline, vexact_s<seed> as Mirror-exact).
The pre-registration is checked before the script runs on a GPU, and fp_eval_downstream_meta.json is written to the
script's --out.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.utils import REPO_ROOT, log  # noqa: E402

SCRIPTS = ("tier2_linear", "core07_shift")
MUST09_FORMAT = "must09-healed-v1"


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--script", required=True, choices=SCRIPTS)
    p.add_argument("--step", type=int, default=None, help="checkpoint step of every final-phase run directory")
    p.add_argument("--run-step", action="append", default=None, metavar="DIR=STEP")
    p.add_argument("--loader", default="mirror.final.heal.students:load_student")
    a = p.parse_args(argv)
    a.script_args = rest
    return a


def must09_style(run_dir) -> bool:
    """True if mirror.heal_structured.load_healed_model can read the directory (spec.json in the MUST-09 format and
    its state file present)."""
    run = Path(run_dir)
    try:
        spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return spec.get("format") == MUST09_FORMAT and bool(spec.get("state_file")) and (run / spec["state_file"]).is_file()


def standin_adapter(model):
    """The attributes scripts/tier2_linear.VariantFactory._build_healed_run compares, read from the model."""
    blocks = model.blocks
    if hasattr(model, "num_register_tokens"):
        n_prefix = 1 + int(model.num_register_tokens)
    else:
        n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    ps = getattr(model, "patch_size", None)
    if ps is None:
        ps = model.patch_embed.patch_size
    ps = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
    return SimpleNamespace(model=model, depth=len(blocks), embed_dim=int(blocks[0].mlp.fc1.in_features),
                           n_prefix_tokens=n_prefix, patch_size=ps, standin=True)


def final_phase_layout(run_dir) -> bool:
    """True for a final-phase run directory (it has ckpt/, where the trainer saves step<S>.pt); MUST-09 runs have
    healed_state.pt and no ckpt/."""
    return (Path(run_dir) / "ckpt").is_dir()


def make_bridge(original_loader, load_student, default_step=None, steps: dict | None = None):
    """load_healed_model(run_dir, device, adapter=None, check=True) for both layouts: final-phase directories
    (ckpt/) go to load_student at the chosen step, every other directory to the original loader."""
    steps = {str(Path(k).resolve()): int(v) for k, v in (steps or {}).items()}
    calls = []

    def load_healed_model(run_dir, device="cpu", adapter=None, check=True):
        if not final_phase_layout(run_dir):
            calls.append(("must09", str(run_dir), None))
            return original_loader(run_dir, device, adapter=adapter, check=check)
        step = steps.get(str(Path(run_dir).resolve()), default_step)
        ad, model, info = load_student(run_dir, step, device)
        calls.append(("load_student", str(run_dir), step))
        info = dict(info) if isinstance(info, dict) else {}
        info.setdefault("run_dir", str(run_dir))
        info["loaded_by"] = f"scripts/fp_eval_downstream.py bridge: load_student(run_dir, step={step})"
        if ad is None:
            ad = adapter if adapter is not None else standin_adapter(model)
        return ad, model, info

    load_healed_model.calls = calls
    return load_healed_model


def install_bridge(load_student, default_step=None, steps=None):
    """Replace mirror.heal_structured.load_healed_model by the bridge (this process only); returns the bridge."""
    import mirror.heal_structured as HS

    original = getattr(HS.load_healed_model, "original_loader", HS.load_healed_model)
    bridge = make_bridge(original, load_student, default_step, steps)
    bridge.original_loader = original
    HS.load_healed_model = bridge
    return bridge


def load_script(name: str):
    key = f"_fp_downstream_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


def _out_of(script_args: list):
    for i, s in enumerate(script_args):
        if s == "--out" and i + 1 < len(script_args):
            return script_args[i + 1]
        if s.startswith("--out="):
            return s.split("=", 1)[1]
    return None


def main(argv=None, load_student=None) -> dict:
    a = parse_args(argv)
    t0 = time.time()
    from mirror.final.eval.knn_fast import resolve_callable
    from mirror.final.prereg import require_prereg, write_final_meta

    tiny = "--tiny" in a.script_args
    st = require_prereg(allow_missing=True if tiny else None)
    steps = {}
    for spec in a.run_step or []:
        d, sep, s = spec.rpartition("=")
        if not sep or not s.isdigit():
            raise SystemExit(f"--run-step {spec!r}: expected DIR=STEP")
        steps[d] = int(s)
    bridge = install_bridge(load_student or resolve_callable(a.loader), a.step, steps)
    log(f"fp_eval_downstream: {a.script} with the run-layout bridge (step {a.step}, per-run steps {steps})")
    mod = load_script(a.script)
    res = mod.main(a.script_args)
    out = _out_of(a.script_args)
    if out:
        import torch

        dev = "cuda" if torch.cuda.is_available() and not tiny else "cpu"
        write_final_meta(out, name="fp_eval_downstream_meta.json", experiment=f"downstream via {a.script}",
                         config={"script": a.script, "script_args": a.script_args, "step": a.step,
                                 "run_steps": steps, "loader": a.loader}, device=dev,
                         wall_seconds=time.time() - t0,
                         extra={"bridge_calls": bridge.calls, "prereg_ok": st.get("ok"),
                                "prereg_allowed_without_check": bool(st.get("allowed_without_check"))})
    return {"script_result": res, "bridge_calls": list(bridge.calls)}


if __name__ == "__main__":
    main()
