#!/usr/bin/env python
"""Runner for the final phase (mirror/final/run/runner.py; manifest format: jobs/final_phase.schema.md).

    . ~/.mirror_env.sh
    setsid nohup python scripts/fp_runner.py run jobs/final_phase.yaml > logs/final/runner.out 2>&1 < /dev/null &
    python scripts/fp_runner.py status jobs/final_phase.yaml       # every job, its state and why (writes nothing)
    python scripts/fp_runner.py check jobs/final_phase.yaml        # validate; jobs, heal-units and cut ranks per wave
    python scripts/fp_runner.py project jobs/final_phase.yaml [--hours-per-unit H --concurrency C]
                                                                   # projection and the cuts it would make (no writes)
    python scripts/fp_runner.py clock jobs/final_phase.yaml [--set-start 2026-11-21T06:10:00Z] [--adjust-hours H]
    python scripts/fp_runner.py heartbeat jobs/final_phase.yaml    # keeps the VM clock while no runner runs
    python scripts/fp_runner.py reset jobs/final_phase.yaml <job>  # after a fix: clear the job's failures
    python scripts/fp_runner.py clear-stop jobs/final_phase.yaml   # resume launching after a stop event
    python scripts/fp_runner.py --dry-run [--out DIR]              # synthetic manifest, simulated GPU and jobs, CPU

`run` refuses to start unless mirror.final.prereg.require_prereg() passes (PREREG_FINAL_PHASE.md recorded and
unchanged). Stopping the runner (Ctrl-C, SIGTERM) leaves its jobs running; start it again to resume. To stop one job
for good: kill -TERM -<pid> (pid in logs/final/state/<id>.pid), then touch logs/final/state/<id>.skip.
Exit codes of `run`: 0 every job settled, 1 some job failed or was blocked, 2 a stop event, 3 the 15-hour ceiling,
130 the runner was stopped while jobs run.
"""
from __future__ import annotations

import argparse
import calendar
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.final.run import runner as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _runner(a, **kw) -> R.Runner:
    m = R.load_manifest(a.manifest)
    return R.Runner(m, a.root, **kw)


def cmd_check(a) -> int:
    try:
        m = R.load_manifest(a.manifest)
    except R.ManifestError as exc:
        print(exc)
        return 1
    print(f"{len(m.jobs)} jobs; manifest sha256 {m.sha256[:12] if m.sha256 else '-'}")
    for w in m.settings.waves:
        js = [j for j in m.jobs if j.wave == w]
        if js:
            units = sum(j.heal_units for j in js)
            kinds = {}
            for j in js:
                kinds[j.kind] = kinds.get(j.kind, 0) + 1
            print(f"  {w:4s} {len(js):3d} jobs, {units:6.2f} heal-units, " + ", ".join(
                f"{k} {v}" for k, v in sorted(kinds.items())))
    ranks = {}
    for j in m.jobs:
        ranks.setdefault(j.cut_rank, []).append(j)
    for r in sorted(k for k in ranks if k is not None):
        print(f"  cut rank {r}: {len(ranks[r])} jobs, {sum(j.heal_units for j in ranks[r]):g} heal-units: "
              + ", ".join(j.id for j in ranks[r]))
    nev = ranks.get(None, [])
    print(f"  never cut: {len(nev)} jobs, {sum(j.heal_units for j in nev):g} heal-units")
    print(f"  total heal-units {sum(j.heal_units for j in m.jobs):g}")
    return 0


def cmd_status(a) -> int:
    r = _runner(a, gpu=lambda: None)
    snap = r.snapshot()
    print(f"VM hours used: {snap['used_hours']:.2f}; stop events: {snap['stop'] or 'none'}")
    for jid in r.m.order:
        j = r.m.by_id[jid]
        print(f"  {jid:36s} {j.wave:4s} {j.kind:8s} {snap['states'][jid]:10s} {snap['reasons'][jid][:90]}")
    p = snap["projection"]
    if p.get("available"):
        print(f"projection {p['total_hours']:.2f} h (target {p['target_hours']}), {p['remaining_units']:.2f} heal-units "
              f"left at {p['hours_per_unit']:.3f} h per unit, concurrency {p['concurrency']:.2f} ({p['rate_source']})")
    else:
        print(f"projection not available: {p.get('why')}")
    return 0


def cmd_project(a) -> int:
    m = R.load_manifest(a.manifest)
    if a.hours_per_unit and a.concurrency:
        m.settings.projection.hours_per_unit = a.hours_per_unit
        m.settings.projection.concurrency = a.concurrency
        m.settings.projection.source = "manual"
    r = R.Runner(m, a.root, gpu=lambda: None)
    now = r.clock()
    used = r.clock_update(now, write=False)
    states, _, hist = r.states(now, used)
    p = r.projection(now, used, states, hist)
    print(json.dumps({k: v for k, v in p.items()}, indent=2, default=str))
    for c in r.plan_cuts(now, used, states, hist, p):
        print(f"would cut rank {c['rank']}: {', '.join(c['jobs'])} ({c['heal_units']:g} heal-units): "
              f"{c['before_hours']:.2f} -> {c['after_hours']:.2f} h")
    return 0


def _parse_utc(text: str) -> float:
    return float(calendar.timegm(time.strptime(text.rstrip("Z"), "%Y-%m-%dT%H:%M:%S")))


def cmd_clock(a) -> int:
    r = _runner(a, gpu=lambda: None)
    f = r.clock_file()
    if a.set_start or a.adjust_hours is not None:
        b = r._read_json(f, None) or {}
        now = time.time()
        if a.set_start:
            t0 = _parse_utc(a.set_start)
            b.update(w0_start_epoch=t0, w0_start_utc=R._utc(t0), counted_s=max(0.0, now - t0), last_seen_epoch=now,
                     boot_epoch=r.boot_time(), source="set by hand")
        if a.adjust_hours is not None:
            b["adjust_s"] = float(b.get("adjust_s") or 0.0) + 3600.0 * a.adjust_hours
        b.setdefault("pauses", [])
        r._write_json(f, b)
        r.event("clock_set", set_start=a.set_start, adjust_hours=a.adjust_hours)
    used = r.clock_update(time.time(), write=False)
    print(json.dumps({**(r._read_json(f, {}) or {}), "used_hours_now": round(used, 3)}, indent=2))
    return 0


def cmd_heartbeat(a) -> int:
    r = _runner(a, gpu=lambda: None)
    lock = R.RunnerLock(r.state / "heartbeat.lock")
    try:
        lock.acquire()
    except R.RunnerBusy:
        print("a heartbeat already runs for this state directory")
        return 0
    stop = r.state / "heartbeat.stop"
    stop.unlink(missing_ok=True)
    try:
        while not stop.exists():
            used = r.clock_update(time.time())
            if a.once:
                print(f"VM hours used {used:.3f}")
                return 0
            time.sleep(a.interval)
    finally:
        lock.release()
    return 0


def cmd_reset(a) -> int:
    r = _runner(a, gpu=lambda: None)
    h = r.reset(a.job)
    print(f"{a.job}: failures and losses cleared ({len(h['attempts'])} attempts kept in its history)")
    return 0


def cmd_clear_stop(a) -> int:
    _runner(a, gpu=lambda: None).clear_stop()
    print("stop events cleared; a job that still fails twice or a stop condition that still holds stops it again")
    return 0


def cmd_run(a) -> int:
    r = _runner(a)
    try:
        return r.run()
    except R.RunnerBusy as exc:
        print(exc)
        return 4


# ----------------------------------------------------------------------------------------------
# Dry run: a synthetic manifest with simulated jobs, GPU and clock
# ----------------------------------------------------------------------------------------------
def dry_manifest() -> dict:
    heal = lambda i, w, deps, units=1.0, rank=None, mem=30: {  # noqa: E731
        "id": i, "wave": w, "kind": "heal", "command": f"python scripts/fp_heal.py --dry-run --id {i}",
        "deps": deps, "gpu_mem_gb": mem, "heal_units": units, "cut_rank": rank, "done": f"runs/dry/{i}.done",
        "priority": 50}
    jobs = [
        {"id": "w0_env", "wave": "W0", "kind": "cpu", "command": "bash scripts/fp_vm_w0.sh gpu",
         "done": "runs/dry/w0_env.done", "priority": 100},
        heal("w0_timing", "W0", ["w0_env"], units=1.0),
        heal("gate_mirror_s0", "W1", [], units=1.0),
        heal("gate_mirror_s1", "W1", [], units=1.0),
        heal("gate_svdllm_s0", "W1", [], units=1.0),
        {"id": "eval_server", "wave": "W1", "kind": "eval", "command": "python scripts/fp_eval_server.py",
         "service": True, "gpu_mem_gb": 20, "priority": 90},
        {"id": "a09_audit", "wave": "W1", "kind": "light", "command": "python scripts/fp_a09_audit.py",
         "gpu_mem_gb": 8, "done": "runs/dry/a09.done", "priority": 10},
        {"id": "gate_p", "wave": "W1", "kind": "decision", "command": "python -c 'gate'",
         "deps": ["gate_mirror_s0", "gate_mirror_s1", "gate_svdllm_s0"], "done": "runs/dry/gate_p.json",
         "priority": 95},
        heal("a05_main_s0", "W2", ["gate_p"], units=4.0, mem=60),
        heal("a15b_probe", "W5", [], units=1.0, rank=1),
        {"id": "a15a_siglip", "wave": "W5", "kind": "light", "command": "python scripts/fp_a15a_siglip.py",
         "gpu_mem_gb": 12, "done": "runs/dry/a15a.done", "cut_rank": 1},
        heal("a08_l_mirror", "W5", [], units=3.0, rank=2, mem=60),
        {"id": "a08_l_eval", "wave": "W5", "kind": "eval", "command": "python scripts/fp_eval.py --model l",
         "deps": ["a08_l_mirror"], "gpu_mem_gb": 20, "done": "runs/dry/a08_l_eval.done", "cut_rank": 2},
        heal("a11_svdllm", "W5", [], units=3.0, rank=3, mem=40),
        {"id": "report", "wave": "End", "kind": "report", "command": "python scripts/collect_next_phase.py",
         "after": ["W5"], "done": "runs/dry/report.done"},
    ]
    return {"settings": {"poll_s": 60, "gpu_log_s": 60, "status_s": 300, "stagger_s": 30, "warmup_s": 120,
                         "max_heals": 3,
                         "stop_when": [{"file": "runs/dry/gate_p.json", "key": "pass", "equals": False,
                                        "reason": "Gate P failed"}],
                         "projection": {"source": "manual", "hours_per_unit": 1.2, "concurrency": 1.0,
                                        "target_hours": 13.5}},
            "jobs": jobs}


def dry_run(out: Path | None) -> int:
    tmp = None
    if out is None:
        tmp = tempfile.mkdtemp(prefix="fp_runner_dry_")
        out = Path(tmp)
    out.mkdir(parents=True, exist_ok=True)
    try:
        (out / "results").mkdir(exist_ok=True)
        (out / "notes").mkdir(exist_ok=True)
        (out / "results" / "STATUS.md").write_text("# STATUS (dry run)\n\nOlder notes stay below.\n")
        (out / "notes" / "decisions.md").write_text(
            "| Date | Decision | Reason | Effect | Approved by |\n|---|---|---|---|---|\n")
        t = [1_700_000_000.0]
        clock = lambda: t[0]  # noqa: E731
        m = R.load_manifest(dry_manifest())
        durations = {"w0_env": 300, "eval_server": 10 ** 9, "gate_p": 60, "a09_audit": 900, "report": 120}
        sim = R.SimulatedLauncher(clock, out, durations=durations, default_s=3000)

        def gpu():
            running = [x for x, p in sim.procs.items() if not p["done"] and clock() < p["end"]]
            used = sum(sim.procs[x]["job"].gpu_mem_gb for x in running) * 1024
            heals = sum(sim.procs[x]["job"].kind == "heal" for x in running)
            return {"util": 0.95 if heals else 0.2, "mem_util": 0.5, "used_mb": used, "free_mb": 143_000 - used,
                    "total_mb": 143_000, "power_w": 500.0, "temp_c": 60.0}

        r = R.Runner(m, out, clock=clock, gpu=gpu, launcher=sim, host_mem=lambda: 500.0, boot_time=lambda: None,
                     prereg=lambda: {"reason": "dry run"}, sleep=lambda s: t.__setitem__(0, t[0] + s),
                     log=lambda line: None)
        # gate decision: written when gate_p "finishes"; pass = true
        orig_finish = sim._finish

        def finish(pid, rc):
            orig_finish(pid, rc)
            job = sim.procs[pid]["job"]
            if job.id == "gate_p" and rc == 0:
                (out / "runs" / "dry" / "gate_p.json").write_text(json.dumps({"pass": True}))

        sim._finish = finish
        code = r.run(max_steps=5000)
        status = (out / "results" / "STATUS.md").read_text()
        decisions = (out / "notes" / "decisions.md").read_text()
        snap = r.snapshot()
        print(status.split(R.MARK_END)[0][-3000:])
        print("decisions.md:\n" + "\n".join(decisions.splitlines()[2:]))
        cut = [j for j, s in snap["states"].items() if s == "cut"]
        ok = code == 0 and bool(cut) and snap["states"]["report"] == "finished" and \
            "Older notes stay below." in status
        print(f"dry run {'OK' if ok else 'FAILED'}: exit code {code}; cut {cut}; "
              f"{snap['used_hours']:.2f} simulated VM-hours")
        return 0 if ok else 1
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None, help="dry run: keep its files here")
    p.add_argument("--root", default=str(ROOT), help="repository root (jobs run here; paths are relative to it)")
    sub = p.add_subparsers(dest="cmd")
    for name in ("run", "status", "check", "project", "clock", "heartbeat", "reset", "clear-stop"):
        s = sub.add_parser(name)
        s.add_argument("manifest")
        if name == "project":
            s.add_argument("--hours-per-unit", type=float, default=None)
            s.add_argument("--concurrency", type=float, default=None)
        if name == "clock":
            s.add_argument("--set-start", default=None, help="UTC, e.g. 2026-11-21T06:10:00Z (counts all time since)")
            s.add_argument("--adjust-hours", type=float, default=None, help="add (or with a minus, remove) hours")
        if name == "heartbeat":
            s.add_argument("--interval", type=float, default=60.0)
            s.add_argument("--once", action="store_true")
        if name == "reset":
            s.add_argument("job")
    a = p.parse_args(argv)
    if a.dry_run:
        return dry_run(Path(a.out) if a.out else None)
    handlers = {"run": cmd_run, "status": cmd_status, "check": cmd_check, "project": cmd_project,
                "clock": cmd_clock, "heartbeat": cmd_heartbeat, "reset": cmd_reset, "clear-stop": cmd_clear_stop}
    if a.cmd not in handlers:
        p.print_help()
        return 2
    return handlers[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
