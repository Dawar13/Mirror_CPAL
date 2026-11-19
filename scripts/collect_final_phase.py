#!/usr/bin/env python
"""Final-phase collector: every rule of PREREG_FINAL_PHASE.md scored mechanically, with every number's source file
and commit (FINAL_PHASE_PLAN.md v2, Section 8 "End" and Section 10). The code is mirror/final/collect.py; its
docstring lists the input files of every rule and the scoring conventions.

    python scripts/collect_final_phase.py                     # reads runs/ and results/ of the repository
    python scripts/collect_final_phase.py --root <dir>        # another tree with the same layout (tests)

Outputs (paths under --root unless given):
  results/FINAL_PHASE_REPORT.md   every rule (value, bar, n, verdict PASS, FAIL, PENDING with the reason, NOT RUN when
                                  the runner cut it, REPORTED for report-only rules; sources with commits), the
                                  scoring details, the headline numbers, the deviations logged in notes/decisions.md
                                  after the pre-registration commit, the scoring conventions and collector notes
  results/final/numbers.json      key, value, unit, n, description, source(s) and commit of every number, the rules,
                                  the bootstrap results, and the sha256 of every input read
  results/final/tables/           T1 to T8 of FINAL_PHASE_PLAN.md Section 10 as CSV (with source columns) and markdown
  PAPER_NUMBERS.json              a "final_phase" section is added or replaced; every other key is kept unchanged
                                  (scripts/collect_next_phase.py rewrites the whole file: run this script after it)
CPU only. Intervals use mirror.final.stats.hbootstrap with the pre-registered 10,000 resamples and seed 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.final import collect as C  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(C.REPO_ROOT), help="tree holding runs/ and results/ (default: the repository)")
    p.add_argument("--report", default=None, help="default <root>/results/FINAL_PHASE_REPORT.md")
    p.add_argument("--numbers-json", default=None, help="default <root>/results/final/numbers.json")
    p.add_argument("--paper-numbers", default=None, help="default <root>/PAPER_NUMBERS.json")
    p.add_argument("--tables-dir", default=None, help="default <root>/results/final/tables")
    p.add_argument("--n-resamples", type=int, default=C.BOOT_RESAMPLES,
                   help="bootstrap resamples (pre-registered 10,000; change only for tests)")
    p.add_argument("--n-val", type=int, default=C.N_VAL, help="validation queries a value must use (tests only)")
    p.add_argument("--no-t1-intervals", action="store_true",
                   help="skip the per-baseline intervals of table T1 (A05-1's own interval is always computed)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    root = Path(a.root).resolve()
    if a.n_resamples != C.BOOT_RESAMPLES:
        C.log(f"WARNING: {a.n_resamples} bootstrap resamples, not the pre-registered {C.BOOT_RESAMPLES:,}")
    c = C.Final(root, n_val=a.n_val, n_resamples=a.n_resamples, t1_intervals=not a.no_t1_intervals)
    if a.n_resamples != C.BOOT_RESAMPLES:
        c.notes.append(f"bootstrap run with {a.n_resamples} resamples, not the pre-registered {C.BOOT_RESAMPLES:,}")
    if a.n_val != C.N_VAL:
        c.notes.append(f"values required on {a.n_val} validation queries, not the pre-registered {C.N_VAL:,}")
    C.log(f"collecting from {root}; pre-registration: {c.prereg.get('status')}")
    C.run_all(c)
    report = Path(a.report) if a.report else root / "results" / "FINAL_PHASE_REPORT.md"
    numbers = Path(a.numbers_json) if a.numbers_json else root / "results" / "final" / "numbers.json"
    paper = Path(a.paper_numbers) if a.paper_numbers else root / "PAPER_NUMBERS.json"
    tables = Path(a.tables_dir) if a.tables_dir else root / "results" / "final" / "tables"
    payload = C.write_outputs(c, report, numbers, paper, tables)
    n = payload["counts"]
    C.log(f"{n['scored']} scored rules: PASS {n[C.PASS]}, FAIL {n[C.FAIL]}, PENDING {n[C.PENDING]}, NOT RUN "
          f"{n[C.NOT_RUN]}; {n[C.REPORTED]} of {n['report_only']} report-only rules reported; {len(c.numbers)} numbers; "
          f"wrote {report}, {numbers}, {tables} and the final_phase section of {paper}")
    return payload


if __name__ == "__main__":
    main()
