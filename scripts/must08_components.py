#!/usr/bin/env python
"""MUST-08: matched component ablation of Mirror-exact (PROJECT_NEXT_PHASE.md; PREREG_NEXT_PHASE.md P8.1, P8.2).

Mirror-exact on blocks 1 to 6 and on the full model with MLP only, attention only, and both: parameters, analytic
FLOPs, k-NN top-1 and flip consistency (bf16; fp32 too for full-model rows). Same code path as MUST-07: this is
scripts/must07_compressors.py with --experiment MUST-08 (sizes 6mlp,6attn,6both,fullmlp,fullattn,full; method
mirror_exact; outputs in results/MUST-08). Rows that MUST-07 already scored with the same fitted structure are copied
with --reuse-from results/MUST-07 instead of being evaluated again.

    python scripts/must08_components.py --model dinov2_vitb14_reg --fit-seeds 0,1,2 --reuse-from results/MUST-07
    python scripts/must08_components.py --tiny --out tiny_out/must08          # CPU, synthetic data

Every option of scripts/must07_compressors.py is accepted.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _runner():
    spec = importlib.util.spec_from_file_location("must07_compressors", Path(__file__).resolve().parent / "must07_compressors.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod              # dataclasses in the script look their module up
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, s in enumerate(argv):
        if (s == "--experiment" and i + 1 < len(argv) and argv[i + 1] != "MUST-08") or \
                (s.startswith("--experiment=") and s.split("=", 1)[1] != "MUST-08"):
            raise SystemExit("scripts/must08_components.py runs MUST-08 only; use scripts/must07_compressors.py")
    if not any(s == "--experiment" or s.startswith("--experiment=") for s in argv):
        argv = ["--experiment", "MUST-08"] + argv
    return _runner().main(argv)


if __name__ == "__main__":
    main()
