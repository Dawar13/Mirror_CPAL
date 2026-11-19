#!/usr/bin/env python
"""A08-1, third part: "stored parameters equal the formula" (FINAL_PHASE_PLAN.md Section 6; PREREG_FINAL_PHASE.md
Section 2.6): the matrix parameters the even/odd form of a healed Mirror-exact model stores
(mirror.efficient.fast_vit_cost, matrix_params_converted, without alignment padding) against the equivariant-map
count formula of the run's structure (mirror.heal_structured.param_counts, converted_params_kept: (h d + tr P tr A)/2
per MLP matrix and the head-map counts for attention, as MUST-01 checked them). Writes
results/A08/<model>/params_check.json with params_stored, params_formula, max_abs_diff_pp (the difference as a
percentage of the formula count) and verdict (PASS when they are equal).

    python scripts/fp_a08_params.py --run-dir runs/final/a08/deit3_b16/mirror_exact/s0
    python scripts/fp_a08_params.py --dry-run --out /tmp/a08p        # CPU, a tiny DeiT-III with a fitted structure
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.utils import REPO_ROOT, log  # noqa: E402

EXPERIMENT = "A08"


def count_check(adapter, model, structure: dict, spec: dict, n_patches: int) -> dict:
    """Stored (even/odd form, align 1) against formula (param_counts) for one fully converted model."""
    from mirror.efficient import fast_vit_cost
    from mirror.final.audit.exactness import AuditStructure, build_even_odd
    from mirror.heal_structured import param_counts

    st = AuditStructure.from_info({"structure": structure})
    fv = build_even_odd(model, st, align=1)
    stored = int(fast_vit_cost(fv, adapter.n_tokens, n_patches)["matrix_params_converted"])
    formula = int(param_counts(adapter, model, spec, structure)["converted_params_kept"])
    diff_pp = 100.0 * abs(stored - formula) / max(formula, 1)
    fv8 = build_even_odd(model, st, align=8)
    stored8 = int(fast_vit_cost(fv8, adapter.n_tokens, n_patches)["matrix_params_converted"])
    return {"params_stored": stored, "params_formula": formula, "max_abs_diff_pp": diff_pp,
            "params_stored_align8": stored8, "verdict": "PASS" if stored == formula else "FAIL",
            "definition": "matrix parameters of the converted matrices: even/odd form (mirror.efficient, no padding) "
                          "against the equivariant-map count formula (mirror.heal_structured.param_counts)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", default=None, help="default results/A08/<model>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from mirror.final.prereg import require_prereg, write_final_meta

    t0 = time.time()
    if a.dry_run:
        from mirror import mirror_family as MF
        from mirror.data import make_loader, synthetic_dataset
        from mirror.final.adapters import tiny_deit3

        pre = require_prereg(allow_missing=True)
        adapter = tiny_deit3(dtype=torch.float64)
        images, labels = synthetic_dataset(3, 4, int(adapter.img_size), seed=2)
        ss = MF.fit_span_structure(adapter, make_loader(images, labels, np.arange(8), 4), "cpu", model_name="tiny",
                                   top_k=3, workers=1)
        L = adapter.depth
        model, info = MF.build_mirror_variant(adapter, "mirror_exact", range(L), ("mlp", "attn"), ss)
        structure = {"A_star": ss.A_star[L], "perms": info["structure"]["perms"],
                     "maps_after": info["structure"]["heads_after"], "reg_perm": ss.reg_perm[L],
                     "blocks": list(range(L))}
        spec = {"blocks": list(range(L)), "components": ["mlp", "attn"], "family": "mirror"}
        out = Path(a.out or (REPO_ROOT / "tiny_out" / "fp_a08_params"))
        name = "tiny_deit3"
    else:
        pre = require_prereg()
        from mirror.final.heal.students import load_student

        adapter, model, info = load_student(a.run_dir, a.step, device=a.device)
        if info.get("structure") is None:
            raise SystemExit(f"{a.run_dir} is not a mirror-family run (no structure)")
        structure, spec = info["structure"], info["spec"]
        name = spec.get("model", "model")
        out = Path(a.out or (REPO_ROOT / "results" / EXPERIMENT / name))
    gh, gw = adapter.grid_hw
    rec = count_check(adapter, model, structure, spec, int(gh) * int(gw))
    rec.update(run_dir=a.run_dir, model=name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params_check.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    write_final_meta(out, name="meta_params_check.json", experiment=EXPERIMENT, config=vars(a), seeds={},
                     models={name: info.get("spec", {}).get("adapter") if isinstance(info, dict) else None},
                     device=a.device, wall_seconds=time.time() - t0, extra={"prereg_check": pre})
    log(f"A08-1 parameters: stored {rec['params_stored']:,} against formula {rec['params_formula']:,} "
        f"({rec['max_abs_diff_pp']:.4f} pp): {rec['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
