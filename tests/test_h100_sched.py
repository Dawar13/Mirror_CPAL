"""scripts/h100_sched_core.py (decision core of scripts/h100_sched.sh): parsing compatible with
scripts/vm_sched.sh, locks, %done, the headroom rule, warm-up reservation and safe restarts. The last
test runs the bash scheduler end to end (Linux with flock and setsid only, so it runs on the VM)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_test_h100_sched_core", ROOT / "scripts" / "h100_sched_core.py")
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)

NOW = 1_000_000.0


def cfg(**kw):
    base = dict(par=3, headroom=8192, warmup=240.0, grace=60.0, max_attempts=3, retry_failed=False)
    base.update(kw)
    return S.Config(**base)


class World:
    """A repo root, a state dir and a lock dir in tmp_path; `alive` holds the pids that run."""

    def __init__(self, tmp_path, text, name="jobs", default_mb=4000):
        self.root = tmp_path
        self.name = name
        self.state = tmp_path / "logs" / name / "state"
        self.locks = tmp_path / "logs" / "sched_locks"
        self.jobs = S.parse_jobs(text, default_mb)
        self.live = set()

    def alive(self, pid, starttime):
        return pid in self.live

    def step(self, free=80000, now=NOW, launch=True, **kw):
        return S.step(self.jobs, self.name, self.state, self.locks, self.root, free, cfg(**kw), now=now,
                      alive=self.alive, launch=launch)

    def run_started(self, res, pid):
        """Pretend the runner of the job just chosen has written its pid."""
        key = res["start"].key
        (self.state / f"{key}.pid").write_text(f"{pid} 123")
        self.live.add(pid)
        return key

    def finish(self, key, pid, rc):
        (self.state / f"{key}.rc").write_text(str(rc))
        self.live.discard(pid)

    def touch(self, rel):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


# ----------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------
def test_parse_compatible_with_vm_sched():
    text = """# comment
   # indented comment

@5000 runs/a.npz cache/b.json | export X=1; python a.py --k 6 | tee out.txt
- | python b.py
runs/c.npz | python c.py
@12000 %lock=heal %done=runs/h/SUMMARY.md - | python scripts/heal.py --cfg x
"""
    jobs = S.parse_jobs(text, 4000)
    assert [j.index for j in jobs] == [1, 2, 3, 4]
    a, b, c, h = jobs
    assert a.mib == 5000 and a.needs == ["runs/a.npz", "cache/b.json"]
    assert a.cmd == "export X=1; python a.py --k 6 | tee out.txt"       # later pipes stay in the command
    assert b.needs == [] and b.mib == 4000 and c.needs == ["runs/c.npz"]
    assert h.locks == ["heal"] and h.done == ["runs/h/SUMMARY.md"] and h.mib == 12000 and h.needs == []


def test_parse_existing_job_files():
    """Every job file written for vm_sched.sh still parses (the format is a superset of its format; the
    older files for vm_queue.sh hold plain commands without "|")."""
    files = [f for f in sorted((ROOT / "jobs").glob("*.txt")) if "vm_sched" in f.read_text(encoding="utf-8")]
    assert len(files) >= 5
    for f in files:
        S.parse_jobs(f.read_text(encoding="utf-8"), 4000)


@pytest.mark.parametrize("line", ["python a.py", "@12k | python a.py", "%lock=a/b | x", "%foo=1 | x",
                                  "a.npz |   ", "%done= | x"])
def test_parse_rejects_bad_lines(line):
    with pytest.raises(ValueError):
        S.parse_jobs(line, 4000)


def test_keys_stable_and_unique():
    a = S.parse_jobs("- | python a.py\n- | python b.py\n- | python a.py\n", 4000)
    assert len({j.key for j in a}) == 3 and a[2].key.endswith("-2")
    b = S.parse_jobs("- | python new.py\n- | python a.py\n- | python b.py\n", 4000)
    assert b[1].key == a[0].key and b[2].key == a[1].key       # inserting a line keeps the other keys


# ----------------------------------------------------------------------------------------------
# Starting rules
# ----------------------------------------------------------------------------------------------
def test_files_and_done(tmp_path):
    w = World(tmp_path, "%done=runs/a/done.txt - | python a.py\nruns/in.npz | python b.py\n")
    w.touch("runs/a/done.txt")
    res = w.step(launch=False)
    assert res["statuses"][w.jobs[0].key] == "done"
    assert res["start"] is None and res["reasons"][2].startswith("waiting for runs/in.npz")
    w.touch("runs/in.npz")
    res = w.step()
    assert res["start"].index == 2 and (w.state / f"{w.jobs[1].key}.start").exists()


def test_headroom_rule(tmp_path):
    w = World(tmp_path, "@12000 - | python big.py\n")
    res = w.step(free=12000 + 8191, launch=False)
    assert res["start"] is None and "headroom" in res["reasons"][1]
    assert w.step(free=12000 + 8192)["start"].index == 1


def test_smaller_job_may_start_out_of_order(tmp_path):
    w = World(tmp_path, "@40000 - | python big.py\n@6000 - | python small.py\n")
    res = w.step(free=30000)
    assert res["start"].index == 2


def test_warmup_reservation_and_par(tmp_path):
    w = World(tmp_path, "@20000 - | python a.py\n@20000 - | python b.py\n@20000 - | python c.py\n")
    r1 = w.step(free=45000, now=NOW)
    w.run_started(r1, 101)
    # nvidia-smi still shows 45,000 MiB free: job 1 has not allocated yet, so its 20,000 are reserved
    r2 = w.step(free=45000, now=NOW + 30)
    assert r2["start"] is None and r2["reserved"] == 20000 and "headroom" in r2["reasons"][2]
    # after the warm-up the reading is trusted
    r3 = w.step(free=45000, now=NOW + 300)
    assert r3["start"].index == 2
    w.run_started(r3, 102)
    r4 = w.step(free=80000, now=NOW + 900, par=2)
    assert r4["start"] is None and "PAR 2" in r4["reasons"][3]


def test_lock_within_file(tmp_path):
    w = World(tmp_path, "%lock=heal - | python heal1.py\n%lock=heal - | python heal2.py\n- | python probe.py\n")
    r1 = w.step()
    k1 = w.run_started(r1, 201)
    r2 = w.step(now=NOW + 600)
    assert r2["start"].index == 3 and r2["reasons"][2] == "lock heal held"
    w.run_started(r2, 202)
    w.finish(k1, 201, 0)
    r3 = w.step(now=NOW + 1200)
    assert r3["start"].index == 2
    assert [e["index"] for e in r3["events"]] == [1] and r3["events"][0]["rc"] == "0"


def test_lock_across_files(tmp_path):
    a = World(tmp_path, "%lock=heal - | python heal_a.py\n", name="a")
    b = World(tmp_path, "%lock=heal - | python heal_b.py\n- | python other.py\n", name="b")
    shared = set()
    a.live = b.live = shared
    ra = a.step()
    ka = a.run_started(ra, 301)
    rb = b.step(now=NOW + 100)
    assert rb["start"].index == 2 and rb["reasons"][1] == "lock heal held"
    a.finish(ka, 301, 0)
    rb2 = b.step(now=NOW + 200)
    assert rb2["start"].index == 1
    assert [p.name for p in S.lock_entries(b.locks, "heal")] == [f"b__{b.jobs[0].key}"]   # stale entry gone


def test_lock_held_during_launch_grace(tmp_path):
    """A heal chosen by one scheduler holds the lock before its runner has written a pid."""
    a = World(tmp_path, "%lock=heal - | python heal_a.py\n", name="a")
    b = World(tmp_path, "%lock=heal - | python heal_b.py\n", name="b")
    a.step(now=NOW)
    assert b.step(now=NOW + 5)["start"] is None
    assert b.step(now=NOW + 120)["start"].index == 1          # a's runner never appeared: lost, lock free


# ----------------------------------------------------------------------------------------------
# Restarts and outcomes
# ----------------------------------------------------------------------------------------------
def test_restart_does_not_duplicate_running_jobs(tmp_path):
    text = "@8000 - | python a.py\n@8000 - | python b.py\n"
    w = World(tmp_path, text)
    k1 = w.run_started(w.step(now=NOW), 401)
    # a new scheduler instance (fresh parse of the same file) sees job 1 running and starts job 2 only
    w2 = World(tmp_path, text)
    w2.live = w.live
    r = w2.step(now=NOW + 3600)
    assert r["statuses"][k1] == "running" and r["start"].index == 2 and r["running"] == [1, 2]
    w2.run_started(r, 402)
    r = w2.step(now=NOW + 3700)
    assert r["start"] is None and not r["all_done"]


def test_lost_and_interrupted_jobs_rerun_with_attempt_limit(tmp_path):
    w = World(tmp_path, "- | python a.py\n")
    key = w.jobs[0].key
    for attempt in (1, 2, 3):
        r = w.step(now=NOW + attempt * 1000)
        assert r["start"] is not None
        assert json.loads((w.state / f"{key}.start").read_text())["attempt"] == attempt
        w.run_started(r, 500 + attempt)
        w.live.clear()                                       # machine stopped: no exit code
    r = w.step(now=NOW + 9000)
    assert r["start"] is None and r["reasons"][1].startswith("gave up after 3 attempts")
    assert r["all_done"] and r["events"][0]["kind"] == "LOST"
    w2 = World(tmp_path, "- | python b.py\n")
    k = w2.run_started(w2.step(), 601)
    w2.finish(k, 601, 137)                                   # killed by a signal
    r = w2.step(now=NOW + 100)
    assert r["events"][0]["rc"] == "137" and r["start"] is not None


def test_failed_jobs_not_rerun_unless_asked(tmp_path):
    w = World(tmp_path, "- | python a.py\n")
    k = w.run_started(w.step(), 701)
    w.finish(k, 701, 1)
    r = w.step(now=NOW + 100)
    assert r["statuses"][k] == "failed" and r["start"] is None and r["all_done"]
    assert r["events"][0]["kind"] == "END" and r["events"][0]["rc"] == "1"
    assert w.step(now=NOW + 200)["events"] == []             # reported once
    assert w.step(now=NOW + 300, retry_failed=True)["start"].index == 1


def test_finished_jobs_stay_finished_and_skip_file(tmp_path):
    w = World(tmp_path, "- | python a.py\n- | python b.py\n")
    k = w.run_started(w.step(), 801)
    w.finish(k, 801, 0)
    (w.state / f"{w.jobs[1].key}.skip").write_text("")
    r = w.step(now=NOW + 100)
    assert r["statuses"][k] == "finished" and r["statuses"][w.jobs[1].key] == "done"
    assert r["start"] is None and r["all_done"]


def test_status_mode_writes_nothing(tmp_path):
    w = World(tmp_path, "- | python a.py\n")
    k = w.run_started(w.step(), 901)
    w.finish(k, 901, 0)
    r = w.step(now=NOW + 10, launch=False)
    assert r["events"] == [] and not (w.state / f"{k}.reported").exists()
    w2 = World(tmp_path, "- | python b.py\n", name="other")
    r = w2.step(launch=False)
    assert r["start"].index == 1 and not list(w2.state.glob("*.start"))


def test_pid_reuse_is_not_alive():
    me = os.getpid()
    if Path("/proc").is_dir():
        st = S.proc_starttime(me)
        assert st is not None and S.pid_alive(me, st) and not S.pid_alive(me, str(int(st) + 1))
    else:
        assert S.pid_alive(me, None)


def test_cli_check_and_status(tmp_path, capsys):
    jf = tmp_path / "jobs" / "demo.txt"
    jf.parent.mkdir()
    jf.write_text("@6000 %lock=heal - | python heal.py\nruns/x.npz | python probe.py\n")
    assert S.main(["check", "--jobs", str(jf), "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "@6000 %lock=heal" in out and "| python probe.py" in out
    assert S.main(["status", "--jobs", str(jf), "--root", str(tmp_path), "--free-mb", "80000"]) == 0
    out = capsys.readouterr().out
    assert "could start now" in out and "waiting for runs/x.npz" in out
    assert not list((tmp_path / "logs" / "demo" / "state").glob("*.start"))


# ----------------------------------------------------------------------------------------------
# End to end (bash), VM only
# ----------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (sys.platform.startswith("linux") and shutil.which("flock") and shutil.which("setsid")),
                    reason="needs Linux with flock and setsid (runs on the VM)")
def test_bash_scheduler_end_to_end(tmp_path):
    (tmp_path / "runs").mkdir()
    jf = tmp_path / "jobs" / "e2e.txt"
    jf.parent.mkdir()
    jf.write_text("\n".join([
        "%lock=e2e_heal - | sleep 2; echo one > runs/one.txt",
        "%lock=e2e_heal - | sleep 1; echo two > runs/two.txt",
        "runs/one.txt | cat runs/one.txt > runs/three.txt; exit 3",
        "%done=runs/already.txt - | echo never > runs/never.txt",
    ]) + "\n")
    (tmp_path / "runs" / "already.txt").write_text("x")
    env = {**os.environ, "SCHED_ROOT": str(tmp_path), "SCHED_NO_ENV": "1", "SCHED_FAKE_FREE_MB": "80000",
           "SCHED_PYTHON": sys.executable, "POLL": "1", "STAGGER": "0", "GPU_LOG_S": "0", "SUMMARY_S": "3600"}
    script = str(ROOT / "scripts" / "h100_sched.sh")
    t0 = time.time()
    p = subprocess.run(["bash", script, str(jf)], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout
    assert "== QUEUE_DONE" in out
    assert (tmp_path / "runs" / "one.txt").read_text().strip() == "one"
    assert (tmp_path / "runs" / "three.txt").read_text().strip() == "one"
    assert not (tmp_path / "runs" / "never.txt").exists()
    assert "rc=3" in out and out.count("== start ") == 3 and out.count("== end ") == 3
    state = tmp_path / "logs" / "e2e" / "state"
    assert sorted(p.read_text().strip() for p in state.glob("*.rc")) == ["0", "0", "3"]
    # the two locked jobs never overlapped: the second started after the first ended
    logs = {i: (tmp_path / "logs" / "e2e" / f"job{i}.log").read_text() for i in (1, 2, 3)}
    assert all("== end rc=" in v for v in logs.values())

    def stamp(text, prefix):          # "== start <date> <time> key ..." / "== end rc=<rc> <date> <time>"
        words = next(x for x in text.splitlines() if x.startswith(prefix)).split()
        return words[2:4] if prefix == "== start" else words[3:5]

    assert stamp(logs[2], "== start") >= stamp(logs[1], "== end")
    assert (tmp_path / "logs" / "e2e" / "gpu.csv").read_text().startswith("utc,gpu_util_pct")
    assert time.time() - t0 < 120
    # a restart starts nothing again
    p2 = subprocess.run(["bash", script, str(jf)], env=env, capture_output=True, text=True, timeout=60)
    assert p2.returncode == 0 and "== QUEUE_DONE" in p2.stdout and "== start " not in p2.stdout


def test_done_job_that_ran_reports_its_end(tmp_path):
    """A job with %done that finishes (and so creates its %done file) still gets its end line."""
    w = World(tmp_path, "%done=runs/out.txt - | python make_out.py\n")
    k = w.run_started(w.step(), 1001)
    w.touch("runs/out.txt")
    w.finish(k, 1001, 0)
    r = w.step(now=NOW + 50)
    assert r["statuses"][k] == "finished" and r["start"] is None and r["all_done"]
    assert [(e["kind"], e["rc"]) for e in r["events"]] == [("END", "0")]


def test_lost_job_with_outputs_is_done_not_rerun(tmp_path):
    w = World(tmp_path, "%done=runs/out.txt - | python make_out.py\n")
    k = w.run_started(w.step(), 1101)
    w.touch("runs/out.txt")
    w.live.clear()                                            # stopped after writing its output, before the rc
    r = w.step(now=NOW + 500)
    assert r["statuses"][k] == "done" and r["start"] is None and r["all_done"]
    assert r["events"][0]["kind"] == "LOST" and r["events"][0]["status"] == "done"


def test_reasons_count_the_job_being_started(tmp_path):
    w = World(tmp_path, "%lock=heal @6000 - | python h1.py\n%lock=heal @6000 - | python h2.py\n"
                        "@6000 - | python p.py\n")
    r = w.step(free=80000, launch=False, par=2)
    assert r["reasons"][1] == "starting" and r["reasons"][2] == "lock heal held"
    assert r["reasons"][3] == "ready (one job starts per step)"
    r = w.step(free=80000, launch=False, par=1)
    assert r["reasons"][3] == "PAR 1 reached (1 running)"
