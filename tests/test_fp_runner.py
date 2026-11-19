"""mirror/final/run/runner.py and scripts/fp_runner.py: manifest, dependencies, GPU packing, restarts, stop events,
when-conditions, the budget governor, the VM clock, the cut order and the STATUS section, with a fake clock, GPU
and launcher (no processes). The last test starts real processes and runs on Linux only (the VM)."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from mirror.final.run import runner as R

ROOT = Path(__file__).resolve().parents[1]
T0 = 1_800_000_000.0
H = 3600.0


class FakeLauncher:
    """Records launches; a job 'runs' until the test finishes it, crashes it or signals it."""

    def __init__(self, write_pid: bool = True, die_on_signal: bool = True):
        self.write_pid, self.die_on_signal = write_pid, die_on_signal
        self.live, self.pids, self.launched, self.signals, self.env = {}, {}, [], [], {}
        self.next_pid, self.state = 100, None

    def launch(self, job, log_path, state_dir, cwd, env):
        pid, self.next_pid = self.next_pid, self.next_pid + 1
        self.state = Path(state_dir)
        if self.write_pid:
            (self.state / f"{job.id}.pid").write_text(f"{pid} st{pid}")
        self.live[pid] = f"st{pid}"
        self.pids[job.id] = pid
        self.launched.append(job.id)
        self.env[job.id] = env

    def alive(self, pid, starttime):
        return self.live.get(pid) == starttime

    def signal_group(self, pid, starttime, sig):
        if not self.alive(pid, starttime):
            return False
        self.signals.append((pid, sig))
        if self.die_on_signal:
            del self.live[pid]
            jid = next(k for k, v in self.pids.items() if v == pid)
            (self.state / f"{jid}.rc").write_text(str(128 + int(sig)))
        return True

    def reap(self):
        pass


class World:
    def __init__(self, tmp_path, jobs, settings=None, free_gb=120.0, total_gb=141.0, util=0.9, launcher=None):
        self.root = tmp_path
        self.t = [T0]
        self.free, self.total, self.util, self.gpu_on = free_gb, total_gb, util, True
        self.host, self.boot = 500.0, 1_000.0
        self.launcher = launcher or FakeLauncher()
        base = {"stagger_s": 0, "warmup_s": 240, "poll_s": 20, "gpu_log_s": 30, "status_s": 60}
        base.update(settings or {})
        self.manifest = {"settings": base, "jobs": jobs}
        self.m = R.load_manifest(self.manifest)
        self.logs = []
        self.r = self.new_runner()

    def new_runner(self):
        return R.Runner(R.load_manifest(self.manifest), self.root, clock=lambda: self.t[0], gpu=self.gpu,
                        launcher=self.launcher, host_mem=lambda: self.host, boot_time=lambda: self.boot,
                        prereg=lambda: {"reason": "test"}, log=self.logs.append, sleep=self.advance)

    def gpu(self):
        if not self.gpu_on:
            return None
        return {"util": self.util, "mem_util": 0.4, "used_mb": (self.total - self.free) * 1024,
                "free_mb": self.free * 1024, "total_mb": self.total * 1024, "power_w": 400.0, "temp_c": 55.0}

    def advance(self, dt):
        self.t[0] += dt

    def step(self, dt=0.0):
        self.t[0] += dt
        return self.r.step()

    def finish(self, jid, rc=0, done=True):
        job = self.m.by_id[jid]
        if done and rc == 0 and job.done:
            p = self.root / job.done
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("ok")
        (self.root / "logs" / "final" / "state" / f"{jid}.rc").write_text(str(rc))
        self.launcher.live.pop(self.launcher.pids[jid], None)

    def crash(self, jid):
        self.launcher.live.pop(self.launcher.pids[jid], None)

    def states(self):
        return self.r.snapshot()["states"]

    def write(self, rel, obj):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj))


def job(i, wave="W1", kind="cpu", deps=(), mem=0, units=0.0, rank=None, prio=0, **kw):
    d = {"id": i, "wave": wave, "kind": kind, "command": f"echo {i}", "done": f"runs/t/{i}.done",
         "deps": list(deps), "gpu_mem_gb": mem, "heal_units": units, "priority": prio}
    if rank is not None:
        d["cut_rank"] = rank
    d.update(kw)
    return d


def heal(i, wave="W1", mem=30, units=1.0, **kw):
    return job(i, wave, "heal", mem=mem, units=units, **kw)


# ----------------------------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------------------------
def test_manifest_parses_yaml_and_validates(tmp_path):
    text = """
settings:
  headroom_gb: 12
  stop_when:
    - {file: results/A01/gate_p.json, key: pass, equals: false, reason: Gate P failed}
  projection: {target_hours: 13.5, source: timing_file}
jobs:
  - {id: w0_env, wave: W0, kind: cpu, command: bash scripts/fp_vm_w0.sh env, done: logs/final/w0/env.done}
  - id: a01_heal_s0
    wave: W1
    kind: heal
    command: python scripts/fp_heal.py --seed 0
    deps: [w0_env]
    gpu_mem_gb: 30
    heal_units: 1
    priority: 50
    cut_rank: never_cut
    done: runs/final/A01/s0/metrics.json
    when: {file: results/A01/decision.json, key: adopted, equals: true}
"""
    f = tmp_path / "m.yaml"
    f.write_text(text)
    m = R.load_manifest(f)
    assert m.settings.headroom_gb == 12.0 and m.settings.stop_when[0].reason == "Gate P failed"
    assert m.settings.projection.source == "timing_file" and m.sha256 and len(m.jobs) == 2
    j = m.by_id["a01_heal_s0"]
    assert j.cut_rank is None and j.when[0].op == "equals" and j.when[0].value is True and j.wave_index == 1
    assert m.order == ["w0_env", "a01_heal_s0"]

    def bad(jobs, settings=None):
        with pytest.raises(R.ManifestError) as e:
            R.load_manifest({"settings": settings or {}, "jobs": jobs})
        return "\n".join(e.value.problems)

    assert "kind" in bad([job("a", kind="train")])
    assert "duplicate" in bad([job("a"), job("a")])
    assert "unknown dependency" in bad([job("a", deps=["zz"])])
    assert "cycle" in bad([job("a", deps=["b"]), job("b", deps=["a"])])
    assert "later wave" in bad([job("a", wave="W1", deps=["b"]), job("b", wave="W2")])
    assert "never cut but needs" in bad([job("a", rank=3), job("b", deps=["a"])])
    assert "heal_units > 0" in bad([job("h", kind="heal", mem=30)])
    assert "unknown keys" in bad([job("a", gpu_mem=3)])
    assert "done marker" in bad([{"id": "a", "wave": "W1", "kind": "cpu", "command": "x"}])
    assert "unknown key" in bad([job("a")], {"headroom": 3})
    assert "not in settings.waves" in bad([job("a", wave="W9")])
    ok = R.load_manifest({"jobs": [job("a", rank=2), job("b", after=["a"]), job("c", wave="End", after=["W1"])]})
    assert ok.by_id["b"].after == ["a"] and ok.by_id["a"].cut_rank == 2


# ----------------------------------------------------------------------------------------------
# Dependencies, done markers, failures
# ----------------------------------------------------------------------------------------------
def test_dependencies_and_done_markers(tmp_path):
    w = World(tmp_path, [job("a"), job("b", deps=["a"]), job("c", deps=["b"])])
    assert w.step()["launched"] == "a"
    assert w.states()["b"] == "waiting" and w.step(5)["launched"] is None
    w.finish("a", rc=0, done=False)                      # exit 0 but no done marker: a failed attempt
    res = w.step(5)
    assert res["events"][0]["outcome"] == "incomplete" and res["launched"] == "a"
    w.finish("a")
    assert w.step(5)["launched"] == "b"
    w.finish("b")
    assert w.step(5)["launched"] == "c"
    w.finish("c")
    res = w.step(5)
    assert res["finished"] and w.r.exit_code == 0
    assert w.launcher.launched == ["a", "a", "b", "c"]
    assert w.launcher.env["b"]["FP_JOB_ID"] == "b"


def test_done_marker_before_start_skips_job(tmp_path):
    w = World(tmp_path, [job("a"), job("b", deps=["a"])])
    (tmp_path / "runs" / "t").mkdir(parents=True)
    (tmp_path / "runs" / "t" / "a.done").write_text("from an earlier run")
    assert w.step()["launched"] == "b" and w.states()["a"] == "done"


# ----------------------------------------------------------------------------------------------
# GPU packing
# ----------------------------------------------------------------------------------------------
def test_memory_packing_with_warmup_reservation(tmp_path):
    jobs = [heal("h1", mem=30, prio=10), heal("h2", mem=25, prio=9), job("l1", kind="light", mem=5, prio=1)]
    w = World(tmp_path, jobs, {"headroom_gb": 10, "declared_budget": False}, free_gb=50, total_gb=80)
    assert w.step()["launched"] == "h1"                  # 50 - 30 = 20 >= 10
    res = w.step(10)                                     # h1 not allocated yet: its 30 GB are reserved
    assert res["launched"] == "l1"                       # h2: 50 - 30 - 25 < 10; l1: 50 - 30 - 5 = 15
    assert "headroom" in w.r.last_reasons["h2"] and "reserved 30.0" in w.r.last_reasons["h2"]
    w.free = 15                                          # after the warm-up the reading is trusted
    assert w.step(300)["launched"] is None and "free 15.0" in w.r.last_reasons["h2"]
    w.finish("h1")
    w.free = 45
    assert w.step(20)["launched"] == "h2"


def test_declared_budget_headroom_and_caps(tmp_path):
    jobs = [heal("h1", mem=40), heal("h2", mem=40), heal("h3", mem=40), heal("h4", mem=5)]
    w = World(tmp_path, jobs, {"headroom_gb": 10, "max_heals": 3}, free_gb=140, total_gb=141)
    assert [w.step(300)["launched"] for _ in range(3)] == ["h1", "h2", "h3"]
    w.step(300)
    assert w.r.last_reasons["h4"].startswith("max_heals 3")
    w2 = World(tmp_path / "b", [heal("a", mem=60), heal("b", mem=60)], {"headroom_gb": 10}, free_gb=140,
               total_gb=141)
    w2.step()
    w2.step(300)                                         # declared 60 + 60 = 120 <= 141 - 10: both start
    assert w2.launcher.launched == ["a", "b"]
    w3 = World(tmp_path / "c", [heal("a", mem=70), heal("b", mem=70)], {"headroom_gb": 10}, free_gb=140,
               total_gb=141)
    w3.step()
    w3.step(300)
    assert w3.launcher.launched == ["a"] and "declared GPU budget" in w3.r.last_reasons["b"]


def test_light_jobs_preferred_when_utilisation_is_low(tmp_path):
    jobs = [heal("h1", prio=10), heal("h2", prio=10), job("l1", kind="light", mem=5, prio=0)]
    w = World(tmp_path, jobs, {"util_window_s": 60}, util=0.5)
    assert w.step()["launched"] == "h1"                  # nothing runs yet: the heal goes first
    assert w.step(70)["launched"] == "l1"                # a heal runs, utilisation 50% < 85%: pack a light job
    w2 = World(tmp_path / "hi", jobs, {"util_window_s": 60}, util=0.95)
    w2.step()
    assert w2.step(70)["launched"] == "h2"               # utilisation above target: heals first


def test_locks_exclusive_and_host_memory(tmp_path):
    jobs = [job("a", kind="light", mem=5, locks=["bench"]), job("b", kind="light", mem=5, locks=["bench"]),
            job("x", kind="light", mem=5, exclusive=True), job("big", kind="cpu", host_mem_gb=100)]
    w = World(tmp_path, jobs)
    w.host = 110
    assert w.step()["launched"] == "a"
    w.step(1)
    assert w.r.last_reasons["b"] == "lock bench held" and w.r.last_reasons["x"].startswith("exclusive")
    assert w.r.last_reasons["big"].startswith("host memory")
    w.finish("a")
    assert w.step(1)["launched"] == "b"


# ----------------------------------------------------------------------------------------------
# Restarts
# ----------------------------------------------------------------------------------------------
def test_never_double_start_across_restarts(tmp_path):
    w = World(tmp_path, [heal("h1"), heal("h2")], launcher=FakeLauncher(write_pid=False))
    assert w.step()["launched"] == "h1"
    w.r = w.new_runner()                                 # runner restarted while h1's wrapper has not written its pid
    assert w.step(30)["launched"] == "h2" and w.states()["h1"] == "running"
    assert w.launcher.launched == ["h1", "h2"]
    for _ in range(3):
        w.r = w.new_runner()
        w.step(10)
    assert w.launcher.launched == ["h1", "h2"]           # inside the grace period nothing starts twice
    res = w.step(200)                                    # no pid ever appeared: lost, started again (attempt 2)
    assert {e["outcome"] for e in res["events"]} == {"lost"}
    start = json.loads((tmp_path / "logs/final/state/h1.start").read_text())
    assert start["attempt"] == 2 and w.launcher.launched.count("h1") == 2


def test_resume_after_crash(tmp_path):
    w = World(tmp_path, [heal("a"), heal("b"), heal("c"), job("d", deps=["a", "b", "c"])], {"max_heals": 2})
    w.step()
    w.step(300)
    assert w.launcher.launched == ["a", "b"]
    w.finish("a")
    w.crash("b")                                         # machine trouble: b vanished without an exit code
    used_before = w.r.snapshot()["used_hours"]
    w.r = w.new_runner()                                 # the runner itself crashed and is started again
    w.step(600)
    w.step(300)
    assert w.launcher.launched == ["a", "b", "b", "c"]   # a not rerun; b lost and started again; then c
    st = w.states()
    assert st["a"] == "finished" and st["b"] == "running" and st["c"] == "running" and st["d"] == "waiting"
    assert w.r.snapshot()["used_hours"] == pytest.approx(used_before + 900 / H)
    hist = json.loads((tmp_path / "logs/final/state/b.json").read_text())
    assert hist["lost"] == 1 and hist["failures"] == 0


# ----------------------------------------------------------------------------------------------
# Stop events and conditions
# ----------------------------------------------------------------------------------------------
def test_two_failures_stop_new_launches(tmp_path):
    w = World(tmp_path, [heal("f", mem=20), job("x", kind="cpu")], {"retry_mem_factor": 1.5})
    assert w.step()["launched"] == "f"
    w.finish("f", rc=1)
    res = w.step(10)
    assert res["launched"] in ("f", "x")
    w.step(10)
    assert sorted(w.launcher.launched) == ["f", "f", "x"]
    start = json.loads((tmp_path / "logs/final/state/f.start").read_text())
    assert start["attempt"] == 2 and start["gpu_mem_gb"] == pytest.approx(30.0)   # retry asks for 1.5 x memory
    w.finish("f", rc=1)
    res = w.step(10)
    assert res["stopped"] and w.states()["f"] == "failed"
    stop = json.loads((tmp_path / "logs/final/state/stop.json").read_text())
    assert "job f failed 2 times" in stop["reasons"][0]
    assert not res["finished"]                            # x still runs
    w.finish("x")
    res = w.step(10)
    assert res["finished"] and w.r.exit_code == 2
    status = (tmp_path / "results/STATUS.md").read_text()
    assert "job f failed 2 times" in status
    w.r.reset("f")
    w.r.clear_stop()
    assert w.step(10)["launched"] == "f"


def test_gate_p_failure_stops_launches(tmp_path):
    jobs = [heal("g"), job("gate", kind="decision", deps=["g"], done="results/A01/gate_p.json"),
            heal("later", wave="W2", deps=["gate"])]
    stop = [{"file": "results/A01/gate_p.json", "key": "pass", "equals": False, "reason": "Gate P failed"}]
    w = World(tmp_path, jobs, {"stop_when": stop})
    w.step()
    w.finish("g")
    assert w.step(10)["launched"] == "gate"
    w.write("results/A01/gate_p.json", {"pass": False})
    w.finish("gate", done=False)
    res = w.step(10)
    assert res["stopped"] and res["launched"] is None and res["finished"] and w.r.exit_code == 2
    assert w.states()["later"] == "ready" and "Gate P failed" in w.r.last_reasons["later"]


def test_when_conditions(tmp_path):
    jobs = [heal("aw3", when=[{"file": "results/A01/decision.json", "key": "adopted", "equals": True}]),
            job("aw3_eval", kind="eval", mem=5, deps=["aw3"]), job("other")]
    w = World(tmp_path, jobs)
    assert w.step()["launched"] == "other"
    st = w.states()
    assert st["aw3"] == "waiting" and "decision.json" in w.r.snapshot()["reasons"]["aw3"]
    w.write("results/A01/decision.json", {"adopted": False})
    st = w.states()
    assert st["aw3"] == "not_needed" and st["aw3_eval"] == "not_needed"
    w.write("results/A01/decision.json", {"adopted": True})
    assert w.step(10)["launched"] == "aw3"
    for d, exp in (({"a": {"b": 3}}, True), ({"a": {"b": 4}}, False), ({"a": {}}, None)):
        w.write("x.json", d)
        assert R.eval_condition(R.Condition("x.json", "a.b", "in", [1, 3]), tmp_path) is exp
    assert R.eval_condition(R.Condition("missing.json"), tmp_path) is False


def test_barrier_wave_and_after(tmp_path):
    jobs = [job("w0a", wave="W0"), job("w1a", wave="W1"), job("rep", wave="End", after=["W1"])]
    w = World(tmp_path, jobs)
    assert w.step()["launched"] == "w0a"
    w.step(5)
    assert "waiting for W0" in w.r.last_reasons["w1a"] and "after W1" in w.r.last_reasons["rep"]
    w.finish("w0a")
    assert w.step(5)["launched"] == "w1a"
    w.finish("w1a", rc=1)
    w.step(5)
    w.finish("w1a", rc=1)                                # settled (failed): 'after' only needs it settled
    res = w.step(5)
    assert res["stopped"]                                # but two failures stop launches anyway


# ----------------------------------------------------------------------------------------------
# Budget governor and VM clock
# ----------------------------------------------------------------------------------------------
def test_governor_heal_stop_and_ceiling(tmp_path):
    jobs = [heal("h_run", mem=20), job("e_run", kind="eval", mem=5), heal("h_new", mem=20, deps=["e_run"]),
            job("e_new", kind="eval", mem=5, deps=["e_run"]), job("l_new", kind="light", mem=5, deps=["e_run"]),
            job("rep", kind="report", deps=["e_run"])]
    w = World(tmp_path, jobs, {"term_grace_s": 60})
    w.step()
    w.step(5)
    assert w.launcher.launched == ["h_run", "e_run"]
    w.t[0] = T0 + 13.6 * H
    w.finish("e_run")
    res = w.step()
    assert res["governor"] == "heal_stop" and res["launched"] in ("e_new", "rep")
    w.step(5)
    st = w.states()
    assert st["h_new"] == "budget" and st["l_new"] == "budget" and sorted(w.launcher.launched[2:]) == ["e_new", "rep"]
    # an unrelated process: same pid number in a pid file but another start time: never signalled
    w.launcher.die_on_signal = False
    w.t[0] = T0 + 15.01 * H
    res = w.step()
    assert res["governor"] == "ceiling" and not res["finished"]
    term = {pid for pid, sig in w.launcher.signals if sig == R.SIGTERM}
    assert term == {w.launcher.pids[x] for x in ("h_run", "e_new", "rep")}
    assert not [s for s in w.launcher.signals if s[1] == R.SIGKILL]
    res = w.step(61)                                     # survivors after the grace period get SIGKILL
    assert {pid for pid, sig in w.launcher.signals if sig == R.SIGKILL} == term
    for x in ("h_run", "e_new", "rep"):
        w.launcher.live.pop(w.launcher.pids[x], None)
    res = w.step(5)
    assert res["finished"] and w.r.exit_code == 3
    assert w.states()["h_run"] == "terminated"
    assert "CEILING reached" in (tmp_path / "results/STATUS.md").read_text()


def test_governor_never_signals_foreign_processes(tmp_path):
    w = World(tmp_path, [heal("h")])
    w.step()
    pid = w.launcher.pids["h"]
    (tmp_path / "logs/final/state/h.pid").write_text(f"{pid} another-start-time")   # pid reused by someone else
    w.t[0] = T0 + 15.5 * H
    w.step()
    assert w.launcher.signals == []


def test_vm_clock_leaves_out_pauses(tmp_path):
    w = World(tmp_path, [job("a")])
    w.step()
    w.step(2 * H)
    assert w.r.snapshot()["used_hours"] == pytest.approx(2.0)
    # runner stopped for an hour on a running VM (same machine start): counted
    w.r = w.new_runner()
    w.step(1 * H)
    assert w.r.snapshot()["used_hours"] == pytest.approx(3.0)
    # VM paused for 10 hours: the container starts again (new PID 1 start), the gap is left out
    w.boot = w.t[0] + 10 * H
    w.t[0] = w.boot + 0.5 * H
    w.r = w.new_runner()
    w.step()
    b = json.loads((tmp_path / "logs/final/state/budget.json").read_text())
    assert b["pauses"][0]["hours"] == pytest.approx(10.0)
    assert w.r.snapshot()["used_hours"] == pytest.approx(3.0 + R.HEARTBEAT_SLACK_S / H + 0.5)
    # a budget file started by scripts/fp_vm_w0.sh (no machine start recorded) counts from the start of W0
    w2 = World(tmp_path / "w0", [job("a")])
    w2.write("logs/final/state/budget.json", {"w0_start_epoch": T0 - 1800, "counted_s": 0, "last_seen_epoch": T0 - 1800,
                                              "boot_epoch": None, "pauses": [], "adjust_s": 0})
    w2.step()
    assert w2.r.snapshot()["used_hours"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------------------------
# Projection and cut order
# ----------------------------------------------------------------------------------------------
CUT_JOBS = [job("w0a", wave="W0"), heal("main1", units=2.0), heal("main2", units=2.0, deps=["main1"]),
            heal("a15b", wave="W5", units=1.0, rank=1), job("a15a", wave="W5", kind="light", mem=5, rank=1),
            heal("l_heal", wave="W5", units=3.0, rank=2), job("l_eval", wave="W5", kind="eval", mem=5,
                                                                deps=["l_heal"], rank=2),
            heal("a11_svd", wave="W5", units=3.0, rank=3), heal("a11_mirror", wave="W5", units=3.0, rank=4),
            job("report", wave="End", kind="report", after=["W5"])]


def _cut_world(tmp_path, hpu):
    s = {"projection": {"source": "manual", "hours_per_unit": hpu, "concurrency": 1.0, "target_hours": 13.5,
                        "reserve_hours": 0.5}}
    w = World(tmp_path, CUT_JOBS, s)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/decisions.md").write_text("| Date | Decision | Reason | Effect | Approved by |\n|---|---|---|---|---|\n")
    return w


def test_cut_order_until_the_projection_fits(tmp_path):
    w = _cut_world(tmp_path, hpu=1.0)                    # 14 heal-units left: 1 + 14 + 0.5 = 15.5 h > 13.5
    w.step()
    w.finish("w0a")
    w.t[0] += 1 * H
    plan = w.r.plan_cuts(w.t[0], 1.0, *[w.r.states(w.t[0], 1.0)[k] for k in (0, 2)])
    assert [c["rank"] for c in plan] == [1, 2] and not (tmp_path / "logs/final/state/cuts.json").exists()
    res = w.step()                                       # the first W1 start triggers the projection
    assert res["launched"] == "main1"
    cuts = json.loads((tmp_path / "logs/final/state/cuts.json").read_text())
    assert [c["rank"] for c in cuts["cuts"]] == [1, 2]
    assert cuts["jobs"] == sorted(["a15b", "a15a", "l_heal", "l_eval"])
    assert cuts["cuts"][0]["before_hours"] == pytest.approx(15.5)
    assert cuts["cuts"][-1]["after_hours"] == pytest.approx(11.5) and cuts["cuts"][-1]["after_hours"] <= 13.5
    lines = (tmp_path / "notes/decisions.md").read_text().splitlines()
    assert len(lines) == 4 and "rank 1" in lines[2] and "rank 2" in lines[3] and "after W0" in lines[2]
    st = w.states()
    assert {k for k, v in st.items() if v == "cut"} == {"a15b", "a15a", "l_heal", "l_eval"}
    assert st["a11_svd"] != "cut" and st["main2"] == "waiting"
    w.r = w.new_runner()                                 # cuts persist across restarts and are not repeated
    w.step(10)
    assert len(json.loads((tmp_path / "logs/final/state/cuts.json").read_text())["cuts"]) == 2
    assert len((tmp_path / "notes/decisions.md").read_text().splitlines()) == 4


def test_no_cut_when_the_projection_fits(tmp_path):
    w = _cut_world(tmp_path, hpu=0.5)                    # 1 + 7 + 0.5 = 8.5 h
    w.step()
    w.finish("w0a")
    w.step(H)
    assert not (tmp_path / "logs/final/state/cuts.json").exists()
    proj = json.loads((tmp_path / "logs/final/state/projections.json").read_text())
    assert proj[0]["trigger"] == "W1" and proj[0]["projection"]["fits"]


def test_measured_rate_from_heal_jobs_and_timing_file(tmp_path):
    jobs = [heal(f"h{i}", units=1.0) for i in range(4)] + [heal("w0t", wave="W0", units=0.5)]
    w = World(tmp_path, jobs, {"projection": {"min_heal_jobs": 3}})
    w.step()                                             # W0 timing heal
    w.t[0] += 0.5 * H
    w.finish("w0t")
    w.write("results/final/w0_timing.json", {"hours_per_unit": 0.9, "concurrency": 3})
    w.step()
    snap = w.r.snapshot()
    assert snap["projection"]["rate_source"].startswith("timing file")
    assert snap["projection"]["hours_per_unit"] == pytest.approx(0.9)
    for _ in range(2):
        w.step(1)
    # h0..h2 ran together for one hour each: 1 h per unit at concurrency 3
    w.t[0] += H - 2
    for i in range(3):
        w.finish(f"h{i}")
    w.step(1)
    p = w.r.snapshot()["projection"]
    assert p["rate_source"] == "jobs" and p["concurrency"] == pytest.approx(3.0, rel=0.01)
    assert p["hours_per_unit"] == pytest.approx(1.0, rel=0.01)
    assert p["effective_hours_per_unit"] == pytest.approx(1 / 3, rel=0.02)


# ----------------------------------------------------------------------------------------------
# STATUS.md, GPU log, services, timeouts, lock, pre-registration
# ----------------------------------------------------------------------------------------------
def test_status_section_between_markers(tmp_path):
    (tmp_path / "results").mkdir()
    f = tmp_path / "results/STATUS.md"
    f.write_text("# STATUS (paper experiments)\n\n## Older section\n\nkept text\n")
    w = World(tmp_path, [heal("h1"), job("c1")])
    w.step()
    text = f.read_text()
    assert text.startswith("# STATUS (paper experiments)\n\n<!-- fp_runner:begin")
    assert text.count("fp_runner:begin") == 1 and "## Older section\n\nkept text" in text
    assert "| h1 | W1 | heal | running | 1 |" in text and "VM time since the start of W0" in text
    f.write_text(text.replace("kept text", "kept text, edited by hand"))
    w.finish("h1")
    w.step(100)
    text2 = f.read_text()
    assert text2.count("fp_runner:begin") == 1 and "edited by hand" in text2
    assert "| h1 | W1 | heal | finished | 1 |" in text2
    status = json.loads((tmp_path / "logs/final/state/status.json").read_text())
    assert status["states"]["h1"] == "finished"
    R.update_marked_section(tmp_path / "new.md", "body")
    assert (tmp_path / "new.md").read_text().startswith(R.MARK_BEGIN + "\nbody\n" + R.MARK_END)


def test_gpu_log_every_30_seconds(tmp_path):
    w = World(tmp_path, [heal("h1")])
    for dt in (0, 10, 21, 14, 17):                       # t = 0, 10, 31, 45, 62
        w.step(dt)
    rows = (tmp_path / "logs/final/gpu.csv").read_text().splitlines()
    assert rows[0].startswith("utc,vm_hours,gpu_util_pct") and len(rows) == 4
    assert rows[1].split(",")[2] == "90" and rows[1].split(",")[-1] == "1"    # the heal started in that step
    assert rows[3].split(",")[-2] == "1"
    assert w.r.next_wait(w.t[0]) == pytest.approx(20.0)


def test_service_runs_while_others_do_and_timeouts(tmp_path):
    jobs = [job("server", kind="eval", mem=10, service=True, done=None), heal("h", timeout_hours=0.01)]
    w = World(tmp_path, jobs, {"term_grace_s": 30})
    w.step()
    w.step(1)
    assert sorted(w.launcher.launched) == ["h", "server"]
    w.step(40)                                           # h passed its 36 s timeout: SIGTERM, ends with 143
    w.step(1)
    hist = json.loads((tmp_path / "logs/final/state/h.json").read_text())
    assert hist["attempts"][0]["outcome"] == "timeout" and hist["failures"] == 1
    assert w.launcher.launched.count("h") == 2           # retried
    w.finish("h")
    w.step(1)                                            # every other job settled: the service is stopped
    res = w.step(1)
    assert res["finished"] and w.r.exit_code == 0 and w.states()["server"] == "finished"


def test_stop_event_also_stops_the_service(tmp_path):
    """After a stop event the evaluation server is stopped once no other job runs, so the runner can exit."""
    jobs = [job("server", kind="eval", mem=10, service=True, done=None), heal("f"), heal("later", deps=["f"])]
    w = World(tmp_path, jobs)
    w.step()
    w.step(1)
    for _ in range(2):
        w.finish("f", rc=1)
        w.step(1)
    assert w.states()["f"] == "failed" and w.states()["later"] == "blocked"
    w.step(1)
    res = w.step(1)
    assert res["finished"] and w.r.exit_code == 2 and w.states()["server"] == "finished"


def test_runner_lock_and_prereg(tmp_path, monkeypatch):
    lock = R.RunnerLock(tmp_path / "state" / "runner.lock").acquire()
    with pytest.raises(R.RunnerBusy):
        R.RunnerLock(tmp_path / "state" / "runner.lock").acquire()
    lock.release()
    R.RunnerLock(tmp_path / "state" / "runner.lock").acquire().release()

    class NoPrereg(RuntimeError):
        pass

    def refuse():
        raise NoPrereg("PREREG_FINAL_PHASE.md is not recorded")

    w = World(tmp_path / "w", [heal("h")])
    w.r.prereg = refuse
    with pytest.raises(NoPrereg):
        w.r.run(max_steps=3)
    assert w.launcher.launched == []
    monkeypatch.setenv("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
    st = R.default_prereg()                               # the real check: allowed in tests, or recorded and unchanged
    assert st.get("ok") or st.get("allowed_without_check")
    w.r.prereg = lambda: {"reason": "ok"}
    assert w.r.run(max_steps=2) == 0 and w.launcher.launched == ["h"]


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------
def _fp_runner():
    key = "_test_fp_runner_script"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / "fp_runner.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def test_cli_check_project_status(tmp_path, capsys):
    FP = _fp_runner()
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"jobs": CUT_JOBS}))
    assert FP.main(["--root", str(tmp_path), "check", str(f)]) == 0
    out = capsys.readouterr().out
    assert "cut rank 2: 2 jobs, 3 heal-units" in out and "total heal-units 14" in out
    assert FP.main(["--root", str(tmp_path), "project", str(f), "--hours-per-unit", "1", "--concurrency", "1"]) == 0
    out = capsys.readouterr().out
    assert "would cut rank 1" in out and not (tmp_path / "logs/final/state/cuts.json").exists()
    assert FP.main(["--root", str(tmp_path), "status", str(f)]) == 0
    assert "w0a" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"jobs": [job("a", kind="nope")]}))
    assert FP.main(["--root", str(tmp_path), "check", str(bad)]) == 1


def test_cli_dry_run(capsys):
    assert _fp_runner().main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run OK" in out and "Section 9 cut, rank 1" in out


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real process groups: runs on the VM (Linux)")
def test_real_launcher_end_to_end(tmp_path):
    """Real bash jobs in their own sessions: exit codes, done markers, a failing job, and a process-group SIGTERM."""
    jobs = [{"id": "a", "wave": "W1", "kind": "cpu", "command": "sleep 1; mkdir -p runs; echo ok > runs/a.done",
             "done": "runs/a.done"},
            {"id": "b", "wave": "W1", "kind": "cpu", "deps": ["a"], "command": "exit 3", "done": "runs/b.done"},
            {"id": "long", "wave": "W1", "kind": "cpu", "command": "sleep 300 & sleep 300; wait",
             "done": "runs/long.done"}]
    m = R.load_manifest({"settings": {"poll_s": 0.5, "gpu_log_s": 0.5, "status_s": 0.5, "grace_s": 30,
                                      "term_grace_s": 5}, "jobs": jobs})
    launcher = R.PosixLauncher()
    r = R.Runner(m, tmp_path, gpu=lambda: None, launcher=launcher, boot_time=lambda: None,
                 prereg=lambda: {"reason": "test"}, log=lambda line: None)
    t0 = time.time()
    while time.time() - t0 < 60:
        r.step()
        st = r.snapshot()["states"]
        if st["a"] == "finished" and st["b"] == "failed" and st["long"] == "running":
            break
        time.sleep(0.3)
    assert st["a"] == "finished" and st["b"] == "failed", st
    assert (tmp_path / "logs/final/a.log").read_text().count("== end rc=0") == 1
    pid, start = r._pid("long")
    assert R.pid_alive(pid, start) and launcher.signal_group(pid, start, R.SIGTERM)
    t1 = time.time()
    while R.pid_alive(pid, start) and time.time() - t1 < 10:
        time.sleep(0.1)
    assert not R.pid_alive(pid, start)
    deadline = time.time() + 10
    while not (tmp_path / "logs/final/state/long.rc").exists() and time.time() < deadline:
        time.sleep(0.1)
    assert int((tmp_path / "logs/final/state/long.rc").read_text()) >= 128   # the wrapper survives to write it
    assert not launcher.signal_group(pid, start, R.SIGTERM)                   # gone: never signalled again
