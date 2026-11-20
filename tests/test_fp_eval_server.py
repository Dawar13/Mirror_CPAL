"""Final-phase evaluation queue server (scripts/fp_eval_server.py) and one-off CLI (scripts/fp_eval.py): CPU tests with
a stand-in loader that returns the tiny random ViT. Nothing here downloads data or weights."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

from mirror.final.eval import knn_fast as KF

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    key = f"{name}_fp_eval_test"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


S = load_script("fp_eval_server")


@pytest.fixture(scope="module")
def data():
    return KF.EvalData.synthetic(num_classes=4, bank_per_class=8, val_per_class=4, img_size=56, seed=0)


def _req(q, name, **kw):
    (q / f"{name}.json").write_text(json.dumps(kw), encoding="utf-8")


def _orig(name, device):
    return KF.tiny_original(name, device)


def test_priority_order_bad_requests_and_outputs(tmp_path, data):
    q = tmp_path / "_queue"
    S.queue_dirs(q)
    run = tmp_path / "runs" / "tiny" / "mirror_exact" / "s0"
    _req(q, "20261119T000001_a", run_dir=str(run), step=1, passes=["bf16"], n_query=8, priority=0)
    _req(q, "20261119T000002_b", run_dir=str(run), step=2, passes=["bf16", "fp32"], n_query=50000, priority=5)
    _req(q, "20261119T000004_c", run_dir=str(run), step=3, passes=["fp16"], priority=9)          # unknown pass
    bad = q / "20261119T000003_bad.json"
    bad.write_text("{not json", encoding="utf-8")
    old = time.time() - 120
    os.utime(bad, (old, old))
    young = q / "20261119T000005_young.json"
    young.write_text("{half writt", encoding="utf-8")                  # still being written: left alone
    order = []

    def loader(run_dir, step, device):
        order.append(step)
        return KF.tiny_loader(run_dir, step, device)

    ev = KF.KnnEvaluator(data, "cpu", 8)
    counts = S.serve(q, ev, loader, _orig, poll=0.01, settle=30.0, once=True)
    assert counts == {"done": 2, "failed": 2, "recovered": 0}
    assert order == [2, 1]                                        # priority 5 before 0; the bad ones never load
    assert sorted(p.name for p in (q / "done").glob("*.json")) == ["20261119T000001_a.json", "20261119T000002_b.json"]
    failed = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (q / "failed").glob("*.json")}
    assert set(failed) == {"20261119T000003_bad.json", "20261119T000004_c.json"}
    assert "unknown pass" in failed["20261119T000004_c.json"]["error"]
    assert failed["20261119T000003_bad.json"]["traceback"]
    assert young.exists() and not (q / "server.lock").exists() and not list((q / "running").glob("*"))
    j2 = json.loads((run / "eval" / "step2.json").read_text(encoding="utf-8"))
    assert set(j2["passes"]) == {"bf16", "fp32"} and (run / "knn" / "step2_bf16.npz").is_file()
    j1 = json.loads((run / "eval" / "step1.json").read_text(encoding="utf-8"))
    assert set(j1["passes"]) == {"bf16_q8"} and j1["passes"]["bf16_q8"]["n_query"] == 8
    done_b = json.loads((q / "done" / "20261119T000002_b.json").read_text(encoding="utf-8"))
    assert set(done_b["passes"]) == {"bf16", "fp32"} and done_b["eval_json"].endswith("step2.json")


def test_restart_recovers_running_requests_and_skips_finished_passes(tmp_path, data):
    q = tmp_path / "_queue"
    S.queue_dirs(q)
    run = tmp_path / "run"
    ev = KF.KnnEvaluator(data, "cpu", 8)
    KF.evaluate_request(ev, {"run_dir": str(run), "step": 4, "passes": ["bf16"], "with_original": False},
                        KF.tiny_loader, _orig)
    _req(q / "running", "left_by_a_crash", run_dir=str(run), step=4, passes=["bf16", "fp32"], with_original=False)
    ran = []
    real_run = ev.run

    def spy(adapter, model, pass_name, *a, **k):
        ran.append(pass_name)
        return real_run(adapter, model, pass_name, *a, **k)

    ev.run = spy
    counts = S.serve(q, ev, KF.tiny_loader, _orig, poll=0.01, once=True)
    assert counts["recovered"] == 1 and counts["done"] == 1 and ran == ["fp32"]
    j = json.loads((run / "eval" / "step4.json").read_text(encoding="utf-8"))
    assert set(j["passes"]) == {"bf16", "fp32"}


def test_stop_file_lock_and_stop_between_passes(tmp_path, data):
    q = tmp_path / "_queue"
    S.queue_dirs(q)
    ev = KF.KnnEvaluator(data, "cpu", 8)
    _req(q, "r1", run_dir=str(tmp_path / "run"), step=1, passes=["bf16", "fp32"], with_original=False)
    (q / "STOP").write_text("", encoding="utf-8")
    assert S.serve(q, ev, KF.tiny_loader, _orig, poll=0.01, once=True)["done"] == 0
    assert not (q / "STOP").exists() and (q / "r1.json").exists()
    (q / "server.lock").write_text("{}", encoding="utf-8")                     # another server, fresh heartbeat
    with pytest.raises(SystemExit):
        S.serve(q, ev, KF.tiny_loader, _orig, poll=0.01, once=True)
    real_run = ev.run

    def run_then_stop(*a, **k):                                                 # STOP arrives during the first pass
        r = real_run(*a, **k)
        (q / "STOP").write_text("", encoding="utf-8")
        return r

    ev.run = run_then_stop
    counts = S.serve(q, ev, KF.tiny_loader, _orig, poll=0.01, once=True, force_lock=True)
    assert counts["done"] == 0 and (q / "r1.json").exists()                     # unfinished: back in the queue
    j = json.loads((tmp_path / "run" / "eval" / "step1.json").read_text(encoding="utf-8"))
    assert set(j["passes"]) == {"bf16"}                                         # the finished pass is kept


def test_server_cli_dry_run(tmp_path):
    q = tmp_path / "_queue"
    counts = S.main(["--dry-run", "--queue", str(q), "--once", "--poll", "0.01"])
    assert counts["done"] == 2 and counts["failed"] == 0
    meta = json.loads((q / "server" / "meta.json").read_text(encoding="utf-8"))
    assert meta["experiment"] == "fp_eval_server" and meta["extra"]["prereg_allowed_without_check"] is not None
    run = tmp_path / "dry_run_runs" / "tiny" / "mirror_exact" / "s0"
    assert (run / "eval" / "step1.json").is_file() and (run / "knn" / "step0_bf16_q8.npz").is_file()


def test_fp_eval_cli_dry_run_with_probes(tmp_path):
    E = load_script("fp_eval")
    out = tmp_path / "o"
    res = E.main(["--dry-run", "--out", str(out), "--step", "1", "--flip-average", "--probe", "--probe-flipavg"])
    rec = res["results"][0]
    run = out / "dry_run" / "tiny" / "mirror_exact" / "s0"
    assert set(rec["passes"]) == {"bf16", "fp32", "bf16_flipavg", "fp32_flipavg"}
    assert rec["passes"]["fp32_flipavg"]["flip_consistency"] == 100.0
    pr, pf = rec["probe"], rec["probe_flipavg"]
    assert 0 <= pr["top1"] <= 100 and 0 <= pf["top1"] <= 100 and pf["flip_average"] and not pr["flip_average"]
    assert (run / "eval" / "step1_probe_flipavg.json").is_file() and (run / "knn" / "step1_probe_pred.npz").is_file()
    assert E.steps_of(run) == []
    res2 = E.main(["--dry-run", "--out", str(out), "--step", "1", "--flip-average", "--probe"])     # resumed
    assert res2["results"][0]["probe"]["top1"] == pr["top1"]


def test_failed_request_is_requeued_then_failed(tmp_path, data):
    """A passing error (not a bad request) sends the request back to the queue at most twice; a decision job waiting
    for it is then not left waiting for ever."""
    q = tmp_path / "_queue"
    S.queue_dirs(q)
    run = tmp_path / "runs" / "tiny" / "mirror_exact" / "s0"
    _req(q, "20261120T000001_flaky", run_dir=str(run), step=1, passes=["bf16"], n_query=8)
    _req(q, "20261120T000002_broken", run_dir=str(run / "x"), step=1, passes=["bf16"], n_query=8)
    calls = {"flaky": 0, "broken": 0}

    def loader(run_dir, step, device):
        if str(run_dir).endswith("x"):
            calls["broken"] += 1
            raise RuntimeError("disk went away")
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            raise RuntimeError("a passing error")
        return KF.tiny_loader(run_dir, step, device)

    ev = KF.KnnEvaluator(data, "cpu", 8)
    counts = S.serve(q, ev, loader, _orig, poll=0.01, settle=30.0, once=True)
    assert counts["done"] == 1 and counts["failed"] == 1 and counts["requeued"] == 3
    assert calls == {"flaky": 2, "broken": 3}
    failed = json.loads((q / "failed" / "20261120T000002_broken.json").read_text(encoding="utf-8"))
    assert failed["request"][S.RETRY_KEY] == 2 and "disk went away" in failed["error"]
    assert not list(q.glob("*.json"))
    assert S.sticky_cuda_error(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not S.sticky_cuda_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
