#!/usr/bin/env python3
"""Decision core of scripts/h100_sched.sh (standard library only; tests in tests/test_h100_sched.py).

Job file (compatible with scripts/vm_sched.sh): one job per line, "<tokens> | <command>"; blank lines and
lines starting with "#" are skipped. The command is everything after the FIRST "|" (it may contain more
pipes). Tokens before it, separated by spaces:
    <path>          a file that must exist before the job starts (relative to the repo root)
    -               nothing needed
    @<MiB>          GPU memory the job needs (default $NEED_MB)
    %lock=<name>    at most one running job holds <name> at a time, across every job file and scheduler
                    (heals use %lock=heal: one healing job at a time, CLAUDE.md)
    %done=<path>    skip the job when <path> exists (several allowed: skip when all exist)

Per-job state lives in <state dir> (logs/<jobs name>/state/), keyed by a hash of the job line, so it
survives scheduler restarts and edits elsewhere in the file:
    <key>.start     written when the job is launched (JSON: index, attempt, epoch, line)
    <key>.cmd       the command, read by the runner
    <key>.pid       "pid starttime", written by the runner (starttime from /proc/<pid>/stat)
    <key>.rc        exit code, written by the runner when the command ends
    <key>.reported  the end (or loss) has been printed
    <key>.skip      written by hand: never start this job again
Lock holders: <locks dir>/<lock>/<jobs name>__<key> (JSON: state dir, key).

Job status: done (all %done files exist, or .skip), running (live pid, or launched less than --grace s
ago and no pid yet), finished (rc 0), failed (rc 1..127), interrupted (rc >= 128, killed by a signal),
lost (launched, process gone, no rc: the machine stopped or the runner was killed), pending.
Interrupted and lost jobs are started again (up to --max-attempts launches in all); failed jobs only with
--retry-failed. A job starts when its files exist, none of its locks is held by a running job, fewer
than --par jobs of this file run, and
    free MiB - reserved MiB - its MiB >= --headroom MiB,
where reserved is the declared MiB of this file's jobs launched less than --warmup s ago (their memory
may not be allocated yet). Jobs may start out of file order. One job starts per call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

LOCK_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class Job:
    index: int                 # 1-based position among job lines
    key: str                   # stable id from the line text (and its occurrence)
    line: str
    cmd: str
    needs: list = field(default_factory=list)
    mib: int = 0
    locks: list = field(default_factory=list)
    done: list = field(default_factory=list)


def parse_jobs(text: str, default_mb: int) -> list[Job]:
    jobs, seen = [], {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"job line without '|': {line!r}")
        head, cmd = line.split("|", 1)
        cmd = cmd.strip()
        if not cmd:
            raise ValueError(f"job line without a command: {line!r}")
        job = Job(index=len(jobs) + 1, key="", line=line, cmd=cmd, mib=int(default_mb))
        for tok in head.split():
            if tok == "-":
                continue
            if tok.startswith("@"):
                if not tok[1:].isdigit():
                    raise ValueError(f"bad memory token {tok!r} in {line!r}")
                job.mib = int(tok[1:])
            elif tok.startswith("%lock="):
                name = tok[len("%lock="):]
                if not LOCK_RE.match(name):
                    raise ValueError(f"bad lock name {name!r} in {line!r}")
                job.locks.append(name)
            elif tok.startswith("%done="):
                if not tok[len("%done="):]:
                    raise ValueError(f"empty %done in {line!r}")
                job.done.append(tok[len("%done="):])
            elif tok.startswith("%"):
                raise ValueError(f"unknown directive {tok!r} in {line!r}")
            else:
                job.needs.append(tok)
        n = seen.get(line, 0)
        seen[line] = n + 1
        job.key = hashlib.sha1(line.encode()).hexdigest()[:12] + (f"-{n + 1}" if n else "")
        jobs.append(job)
    return jobs


# ----------------------------------------------------------------------------------------------
# Processes
# ----------------------------------------------------------------------------------------------
def proc_starttime(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat (start time in clock ticks), or None if the process is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    rest = stat[stat.rfind(")") + 2:].split()
    return rest[19] if len(rest) > 19 else None


def pid_alive(pid: int, starttime: str | None) -> bool:
    """True if `pid` runs and is the same process (same start time; guards against pid reuse)."""
    if Path("/proc").is_dir():
        now = proc_starttime(pid)
        return now is not None and (starttime is None or now == starttime)
    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False
    return True


# ----------------------------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------------------------
def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def read_start(state: Path, key: str) -> dict | None:
    txt = _read(state / f"{key}.start")
    if txt is None:
        return None
    try:
        return json.loads(txt)
    except ValueError:
        return {"epoch": 0, "attempt": 1}


def job_status(job: Job, state: Path, root: Path, now: float, alive=pid_alive, grace: float = 60.0) -> str:
    """One of: done, running, finished, failed, interrupted, lost, pending (see the module docstring)."""
    start = read_start(state, job.key)
    rc_txt = _read(state / f"{job.key}.rc")
    pid_txt = _read(state / f"{job.key}.pid")
    running = False
    if start is not None and rc_txt is None:
        if pid_txt:
            parts = pid_txt.split()
            running = alive(int(parts[0]), parts[1] if len(parts) > 1 else None)
        else:
            running = now - float(start.get("epoch", 0)) < grace
    if running:
        return "running"
    if (state / f"{job.key}.skip").exists():
        return "done"
    outputs_exist = bool(job.done) and all((root / d).exists() for d in job.done)
    rc = None
    if start is not None and rc_txt is not None:
        try:
            rc = int(rc_txt)
        except ValueError:
            rc = None
    if rc == 0:
        return "finished"
    if rc is not None and rc < 128:
        return "failed"
    # never ran, lost (no exit code) or killed by a signal: %done decides whether it still has to run
    if outputs_exist:
        return "done"
    if start is None:
        return "pending"
    return "lost" if rc is None else "interrupted"


def lock_entries(locks_dir: Path, lock: str) -> list[Path]:
    d = locks_dir / lock
    return sorted(p for p in d.iterdir() if p.is_file()) if d.is_dir() else []


def lock_busy(locks_dir: Path, lock: str, now: float, alive=pid_alive, grace: float = 60.0,
              clean: bool = True) -> str | None:
    """Name of a running holder of `lock`, or None. Entries whose job no longer runs are removed."""
    for entry in lock_entries(locks_dir, lock):
        try:
            info = json.loads(entry.read_text())
            st, key = Path(info["state"]), info["key"]
        except (OSError, ValueError, KeyError):
            continue
        holder = Job(index=0, key=key, line="", cmd="")
        if job_status(holder, st, Path("/nonexistent"), now, alive, grace) == "running":
            return entry.name
        if clean:
            try:
                entry.unlink()
            except OSError:
                pass
    return None


# ----------------------------------------------------------------------------------------------
# One scheduling step
# ----------------------------------------------------------------------------------------------
@dataclass
class Config:
    par: int = 3
    headroom: int = 8192
    warmup: float = 240.0
    grace: float = 60.0
    max_attempts: int = 3
    retry_failed: bool = False


def step(jobs: list[Job], name: str, state: Path, locks_dir: Path, root: Path, free_mb: int, cfg: Config,
         now: float | None = None, alive=pid_alive, launch: bool = True) -> dict:
    """Evaluate every job, report newly ended or lost jobs once, and choose at most one job to start.

    With launch=True the chosen job's .start and .cmd files and its lock entries are written here (the
    caller holds the global mutex), so the next call already counts it as running, and ended or lost
    jobs are reported once (marked .reported). With launch=False (status) nothing is written."""
    now = time.time() if now is None else now
    state.mkdir(parents=True, exist_ok=True)
    statuses = {j.key: job_status(j, state, root, now, alive, cfg.grace) for j in jobs}
    events = []
    for j in jobs:                       # every launched job that ended (exit code) or vanished, once
        rep = state / f"{j.key}.reported"
        start = read_start(state, j.key)
        if not launch or start is None or rep.exists() or statuses[j.key] == "running":
            continue
        rc = _read(state / f"{j.key}.rc")
        if rc is None and (state / f"{j.key}.skip").exists():
            continue
        ended = (state / f"{j.key}.rc").stat().st_mtime if rc is not None else now
        events.append({"kind": "END" if rc is not None else "LOST", "index": j.index, "key": j.key,
                       "rc": rc, "seconds": int(ended - float(start.get("epoch", ended))),
                       "attempt": start.get("attempt", 1), "status": statuses[j.key], "cmd": j.cmd})
        rep.write_text(str(int(now)))
    running = [j for j in jobs if statuses[j.key] == "running"]
    reserved = 0
    for j in running:
        start = read_start(state, j.key) or {}
        if now - float(start.get("epoch", 0)) < cfg.warmup:
            reserved += j.mib
    held = {lk for j in running for lk in j.locks}
    n_run, booked = len(running), reserved      # include the job chosen in this step for later reasons
    reasons, chosen = {}, None
    for j in jobs:
        st = statuses[j.key]
        if st in ("done", "running", "finished"):
            continue
        attempts = int((read_start(state, j.key) or {}).get("attempt", 0))
        if st == "failed" and not cfg.retry_failed:
            continue
        if st in ("interrupted", "lost", "failed") and attempts >= cfg.max_attempts:
            reasons[j.index] = f"gave up after {attempts} attempts ({st})"
            continue
        missing = [f for f in j.needs if not (root / f).is_file()]
        if missing:
            reasons[j.index] = "waiting for " + " ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
            continue
        busy = [lk for lk in j.locks if lk in held]
        busy += [lk for lk in j.locks
                 if lk not in busy and lock_busy(locks_dir, lk, now, alive, cfg.grace, clean=launch)]
        if busy:
            reasons[j.index] = "lock " + ",".join(busy) + " held"
            continue
        if n_run >= cfg.par:
            reasons[j.index] = f"PAR {cfg.par} reached ({n_run} running)"
            continue
        spare = free_mb - booked - j.mib
        if spare < cfg.headroom:
            reasons[j.index] = (f"GPU memory: free {free_mb} - reserved {booked} - needs {j.mib} = {spare} "
                                f"< headroom {cfg.headroom} MiB")
            continue
        if chosen is None:
            chosen = j
            reasons[j.index] = "starting"
            held |= set(j.locks)
            n_run += 1
            booked += j.mib
        else:
            reasons[j.index] = "ready (one job starts per step)"
    if chosen is not None and launch:
        attempt = int((read_start(state, chosen.key) or {}).get("attempt", 0)) + 1
        for suffix in (".pid", ".rc", ".reported"):
            try:
                (state / f"{chosen.key}{suffix}").unlink()
            except OSError:
                pass
        (state / f"{chosen.key}.cmd").write_text(chosen.cmd + "\n")
        (state / f"{chosen.key}.start").write_text(json.dumps(
            {"index": chosen.index, "attempt": attempt, "epoch": now, "line": chosen.line, "jobs": name,
             "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))}))
        for lk in chosen.locks:
            (locks_dir / lk).mkdir(parents=True, exist_ok=True)
            (locks_dir / lk / f"{name}__{chosen.key}").write_text(
                json.dumps({"state": str(state.resolve()), "key": chosen.key, "index": chosen.index}))
        statuses[chosen.key] = "running"
    open_ = [j for j in jobs if statuses[j.key] in ("pending", "running", "lost", "interrupted")
             or (statuses[j.key] == "failed" and cfg.retry_failed)]
    open_ = [j for j in open_ if statuses[j.key] == "running" or not reasons.get(j.index, "").startswith("gave up")]
    counts = {}
    for st in statuses.values():
        counts[st] = counts.get(st, 0) + 1
    return {"events": events, "start": chosen, "statuses": statuses, "reasons": reasons,
            "counts": counts, "all_done": not open_, "reserved": reserved,
            "running": [j.index for j in jobs if statuses[j.key] == "running"]}


# ----------------------------------------------------------------------------------------------
# Command line (used by scripts/h100_sched.sh)
# ----------------------------------------------------------------------------------------------
def _args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["step", "status", "check"])
    p.add_argument("--jobs", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--state", default=None)
    p.add_argument("--locks", default=None)
    p.add_argument("--free-mb", type=int, default=0)
    p.add_argument("--default-mb", type=int, default=4000)
    p.add_argument("--par", type=int, default=3)
    p.add_argument("--headroom", type=int, default=8192)
    p.add_argument("--warmup", type=float, default=240.0)
    p.add_argument("--grace", type=float, default=60.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--retry-failed", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _args(argv)
    root = Path(a.root)
    name = Path(a.jobs).stem
    state = Path(a.state) if a.state else root / "logs" / name / "state"
    locks = Path(a.locks) if a.locks else root / "logs" / "sched_locks"
    jobs = parse_jobs(Path(a.jobs).read_text(), a.default_mb)
    if a.mode == "check":
        for j in jobs:
            extra = " ".join([f"@{j.mib}"] + [f"%lock={x}" for x in j.locks] + [f"%done={x}" for x in j.done])
            print(f"{j.index:3d} {j.key} {extra} needs={len(j.needs)} | {j.cmd}")
        return 0
    cfg = Config(a.par, a.headroom, a.warmup, a.grace, a.max_attempts, a.retry_failed)
    res = step(jobs, name, state, locks, root, a.free_mb, cfg, launch=(a.mode == "step"))
    if a.mode == "status":
        for j in jobs:
            st = res["statuses"][j.key]
            why = res["reasons"].get(j.index, "")
            why = "could start now" if why == "starting" else why
            print(f"   job {j.index:3d} {st:11s} {why}".rstrip() + f" :: {j.cmd[:90]}")
        print(f"   counts {json.dumps(res['counts'])}; reserved {res['reserved']} MiB")
        return 0
    for ev in res["events"]:
        if ev["kind"] == "END":
            print(f"END {ev['index']} rc={ev['rc']} {ev['seconds']}s attempt={ev['attempt']} {ev['cmd']}")
        else:
            then = ("its %done files exist, so it is not run again" if ev["status"] == "done"
                    else "it runs again while attempts remain")
            print(f"LOST {ev['index']} after {ev['seconds']}s attempt={ev['attempt']} (no exit code; {then}) {ev['cmd']}")
    if res["start"] is not None:
        print(f"START {res['start'].index} {res['start'].key}")
    print("COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(res["counts"].items())))
    if res["all_done"]:
        print("ALLDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
