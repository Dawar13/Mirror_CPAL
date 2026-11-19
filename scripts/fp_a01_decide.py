#!/usr/bin/env python
"""The A01 rule (PREREG_FINAL_PHASE.md Sections 1 and 2.2): is the weight-space operator A*_w adopted as the main
operator?

    python scripts/fp_a01_decide.py                     # waits up to 10 minutes for missing inputs, then decides
    python scripts/fp_a01_decide.py --wait-minutes 0    # decide now
    python scripts/fp_a01_decide.py --check             # print the status, write nothing

Rule (pre-registered): adopt A*_w when (1) the zero-shot k-NN of its full conversion, the step-0 evaluation of its
healing run (runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/eval/step0.json, passes.bf16.top1 on all 50,000
queries; seed 1's step 0 if seed 0 has none), is at least 58.84, and (2) the mean final k-NN of A*_w over healing seeds
0 and 1 is at least the mean of the current operator over the same seeds (mirror_exact_astar/s0, s1, the Gate P heals)
plus 0.15. Otherwise the current A* is kept and A*_w is reported as an ablation. The decision is made as soon as it is
determined: a failed part means "not adopted" even while the other part is still missing (so later runs need not wait),
and an A01 input that was cut or skipped also means "not adopted" (the condition can no longer be met). The computation
is mirror.final.collect.a01_record, the function the final collector scores A01 with.

Writes results/A01/decision.json only when decided: "adopted" (true or false), "status" (PASS = adopted, FAIL or NOT
RUN = not adopted), "main_arm" (mirror_exact_wspace or mirror_exact_astar), "main_arm_dir", "structure" (the
SpanStructure every later Mirror run of DINOv2-B must use: results/A01/structure/dinov2_vitb14_reg_wspace_matched.npz,
or the canonical results/MUST-07/structure/dinov2_vitb14_reg_s0.npz) with its sha256, the zero-shot and healed
numbers, the bars, the Gate P record (results/A01/gate_p.json) and every source file with its commit. A pending
decision is never written there: a file with "adopted": null would make the runner's when-conditions read false (and
mirror.final.heal.students.resolve_structure pick the current A*), so a pending status goes to
results/A01/decision_pending.json instead. In jobs/final_phase.yaml the job runs this script with no arguments after
scripts/fp_wait_eval.py has seen the A*_w evaluations, so the default wait (10 minutes) only covers files still being
written. Exit codes: 0 decided, 3 still pending after the wait, 2 the pre-registration check refused.
--decisions-log notes/decisions.md appends one row for the decision.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.final import collect as C  # noqa: E402

REL_OUT = Path("results") / "A01" / "decision.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(C.REPO_ROOT), help="tree holding runs/ and results/ (default: the repository)")
    p.add_argument("--out", default=None, help=f"default <root>/{REL_OUT.as_posix()}")
    p.add_argument("--wait-minutes", type=float, default=10.0,
                   help="how long to wait for missing evaluations before giving up (0: decide now)")
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--check", action="store_true", help="print the status and write nothing")
    p.add_argument("--decisions-log", default=None, help="append the decision to this table (notes/decisions.md)")
    p.add_argument("--n-val", type=int, default=C.N_VAL, help="validation queries a value must use (tests only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.out) if a.out else root / REL_OUT
    pending_path = out.with_name(out.stem + "_pending.json")
    try:
        from mirror.final.prereg import require_prereg

        st = require_prereg(root=root)
        pre = {k: st.get(k) for k in ("ok", "reason", "sha256", "recorded_commit", "allowed_without_check")}
    except Exception as e:  # noqa: BLE001  (PreregError: the file changed or is not recorded)
        C.log(f"A01: {e}")
        return 2
    deadline = time.time() + 60.0 * max(0.0, a.wait_minutes)
    while True:
        c = C.Final(root, n_val=a.n_val)
        rec = C.a01_record(c)
        if rec["adopted"] is not None or a.check or time.time() >= deadline:
            break
        C.log(f"A01 pending: {rec['reason']}; checking again in {a.poll_seconds:.0f} s")
        time.sleep(max(1.0, a.poll_seconds))
    gp = C.read_json(root / "results" / "A01" / "gate_p.json")
    rec.update(written_by="scripts/fp_a01_decide.py", date_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               code_git=C.git_self(), prereg_check=pre, notes=c.notes,
               gate_p={"file": "results/A01/gate_p.json", "pass": gp.get("pass") if isinstance(gp, dict) else None,
                       "status": gp.get("status") if isinstance(gp, dict) else "missing"})
    z, h = rec["zero_shot"], rec["healed"]
    C.log(f"A01 {rec['status']} (adopted {rec['adopted']}): zero-shot {C.fmt(z['value'], '.2f')} (bar 58.84, "
          f"{z['verdict']}); healed A*_w {C.fmt(h['wspace']['mean'], '.3f')} vs A* {C.fmt(h['astar']['mean'], '.3f')} "
          f"(difference {C.fmt(h['difference'], '+.3f')}, bar +0.15, {h['verdict']})")
    if a.check:
        return 0 if rec["adopted"] is not None else 3
    if rec["adopted"] is None:
        C.write_json_atomic(pending_path, rec)
        C.log(f"A01 still pending: wrote {pending_path}; {out} not written")
        return 3
    prev = C.read_json(out)
    if isinstance(prev, dict) and isinstance(prev.get("adopted"), bool) and prev["adopted"] != rec["adopted"]:
        rec["notes"] = list(rec["notes"]) + [f"WARNING: an earlier decision.json said adopted {prev['adopted']}"]
        C.log(f"WARNING: the decision changed from adopted {prev['adopted']} to {rec['adopted']}")
    C.write_json_atomic(out, rec)
    if pending_path.is_file():
        pending_path.unlink()
    if a.decisions_log:
        C.append_decision(Path(a.decisions_log) if Path(a.decisions_log).is_absolute() else root / a.decisions_log,
                          f"A01 {rec['status']}: A*_w {'adopted' if rec['adopted'] else 'not adopted'} (zero-shot "
                          f"{C.fmt(z['value'], '.2f')}, healed difference {C.fmt(h['difference'], '+.3f')})",
                          "Pre-registered rule (PREREG_FINAL_PHASE.md Section 2.2): zero-shot at least 58.84 and "
                          "healed mean of seeds 0 and 1 at least the current A*'s + 0.15",
                          f"Main Mirror arm {rec['main_arm']}; structure {rec['structure']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
