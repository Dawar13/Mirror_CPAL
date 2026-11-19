#!/usr/bin/env python
"""Gate P (PREREG_FINAL_PHASE.md Sections 1 and 2.2): does the new pipeline reproduce MUST-09?

    python scripts/fp_gate_p.py                         # waits up to 10 minutes for missing inputs, then decides
    python scripts/fp_gate_p.py --wait-minutes 0        # decide now (pending if an input is missing)
    python scripts/fp_gate_p.py --check                 # print the status, write nothing

Rule (pre-registered): the means over healing seeds 0, 1 and 2 of the final-step bf16 k-NN on all 50,000 validation
queries of Mirror-exact with the current operator (runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0..s2) and of
SVD-LLM, whitening only (runs/final/main/dinov2_vitb14_reg/svdllm/s0..s2) are within +-0.15 k-NN of 78.34 and 78.06
respectively; both must hold. Each value is passes.bf16.top1 of <run>/eval/step780.json (n_query 50,000), written by
the evaluation server (scripts/fp_eval_server.py) for a finished heal (metrics.json complete, 780 steps). The
computation is mirror.final.collect.gate_p_record, the same function the final collector scores Gate P with.

Writes results/A01/gate_p.json: "pass" true or false (decided) or null (still pending after the wait), "status" (PASS,
FAIL, PENDING, or NOT RUN when a Gate P run was cut or skipped, which counts as not reproduced: pass false), the value
per seed, the means, the bars and every source file with the commit recorded in its meta.json. The runner's stop_when
reads "pass": false ({file: results/A01/gate_p.json, key: pass, equals: false}).

In jobs/final_phase.yaml the job runs this script with no arguments after scripts/fp_wait_eval.py has seen the six bf16
passes, so the default wait (10 minutes) only covers files still being written (a heal's final metrics.json).
Exit codes: 0 decided (pass true or false); 3 still pending after the wait (the runner records a failure and tries
again, never a success); 2 the pre-registration check refused (mirror.final.prereg.require_prereg). While waiting
nothing is written, and a pending file left by an earlier attempt is removed first, so the done marker exists only
when an attempt has ended (a lost process cannot leave a pending file that counts as done). For a status read by hand
use --check: a pending gate_p.json written before the runner's first attempt would make the runner skip the job.
--decisions-log notes/decisions.md appends one row for the decision (once per status).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.final import collect as C  # noqa: E402

REL_OUT = Path("results") / "A01" / "gate_p.json"


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


def prereg_guard(root: Path) -> dict:
    """mirror.final.prereg.require_prereg for the tree (MIRROR_FINAL_ALLOW_NO_PREREG=1 lets tests run)."""
    from mirror.final.prereg import require_prereg

    st = require_prereg(root=root)
    return {k: st.get(k) for k in ("ok", "reason", "sha256", "recorded_commit", "allowed_without_check")}


def main(argv=None) -> int:
    a = parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.out) if a.out else root / REL_OUT
    try:
        pre = prereg_guard(root)
    except Exception as e:  # noqa: BLE001  (PreregError: the file changed or is not recorded)
        C.log(f"Gate P: {e}")
        return 2
    if not a.check and out.is_file():
        prev = C.read_json(out)
        if not isinstance(prev, dict) or prev.get("pass") is None:
            out.unlink()                       # a pending record of an earlier attempt: never leave it as a marker
    deadline = time.time() + 60.0 * max(0.0, a.wait_minutes)
    while True:
        c = C.Final(root, n_val=a.n_val)
        rec = C.gate_p_record(c)
        if rec["pass"] is not None or a.check or time.time() >= deadline:
            break
        C.log(f"Gate P pending: {rec['reason']}; checking again in {a.poll_seconds:.0f} s")
        time.sleep(max(1.0, a.poll_seconds))
    rec.update(written_by="scripts/fp_gate_p.py", date_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               code_git=C.git_self(), prereg_check=pre, notes=c.notes)
    m, s = rec["arms"]["mirror_exact_astar"], rec["arms"]["svdllm"]
    C.log(f"Gate P {rec['status']} (pass {rec['pass']}): Mirror (current A*) mean {C.fmt(m['mean'], '.3f')} "
          f"[{m['n']} seeds], SVD-LLM mean {C.fmt(s['mean'], '.3f')} [{s['n']} seeds]; {rec['reason']}")
    if not a.check:
        C.write_json_atomic(out, rec)
        if a.decisions_log and rec["pass"] is not None:
            C.append_decision(Path(a.decisions_log) if Path(a.decisions_log).is_absolute() else root / a.decisions_log,
                              f"Gate P {rec['status']}: Mirror (current A*) mean {C.fmt(m['mean'], '.3f')}, SVD-LLM "
                              f"mean {C.fmt(s['mean'], '.3f')}",
                              "Pre-registered rule (PREREG_FINAL_PHASE.md Section 2.2): both within 0.15 of 78.34 and "
                              "78.06", "Phase G continues" if rec["pass"] else "Phase G stops and reports")
    return 0 if rec["pass"] is not None else 3


if __name__ == "__main__":
    sys.exit(main())
