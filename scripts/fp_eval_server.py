#!/usr/bin/env python
"""Final phase: the k-NN evaluation queue server (docs/final_phase_design.md Section 4.4).

    python scripts/fp_eval_server.py --queue runs/final/_queue                    # serve until stopped
    python scripts/fp_eval_server.py --queue runs/final/_queue --once             # drain the queue, then exit
    python scripts/fp_eval_server.py --queue runs/final/_queue --max-idle 1800    # exit after 30 idle minutes
    python scripts/fp_eval_server.py --dry-run --queue tiny_out/_queue --once     # CPU, tiny model, synthetic data

The trainer saves a checkpoint and writes runs/final/_queue/<timestamp>_<id>.json:
    {"run_dir": ..., "step": S, "passes": ["bf16", "fp32"], "n_query": 50000 or 10000, "priority": int}
(optional fields: mirror.final.eval.knn_fast.normalise_request). The server
- takes pending requests in priority order: the HIGHEST priority first (--low-priority-first reverses it), ties by
  file name (the timestamp prefix, so the oldest first); a request file that cannot be parsed yet is left alone for
  --settle seconds (the writer may still be writing; writers should write a temporary file and rename it);
- claims a request by moving it to _queue/running/ (an atomic rename, so two readers never take the same file);
- loads the student with the loader (default mirror.final.heal.students:load_student(run_dir, step, device); any
  "module:function" with --loader), runs the passes that are not done yet (mirror.final.eval.knn_fast:
  evaluate_request), writes <run_dir>/eval/step<S>.json, <run_dir>/knn/step<S>_<pass>.npz and
  <run_dir>/eval/step<S>_meta.json, and moves the request to _queue/done/ with a short summary;
- keeps the bank and validation images on the GPU between requests (loaded once) and caches the original model's
  bank features and k-NN results per model (in memory and under --cache-root/knn_original/);
- survives a bad request: any error moves the request to _queue/failed/ with the error and traceback, and the server
  goes on (one retry with half the batch size after a CUDA out-of-memory error);
- stops cleanly on _queue/STOP (after the current request; the file is removed), on SIGTERM or Ctrl-C (the current
  request stays in running/ and its finished passes stay on disk), after --max-idle seconds without work, or with
  --once when the queue is empty. On start, requests left in running/ by a stopped server go back to the queue, so
  nothing is lost; finished passes are skipped on the rerun. A heartbeat lock (_queue/server.lock) keeps a second
  server off the same queue.
GPU: the memory cap (--gpu-frac, default: the images plus --reserve-gb over the device's memory) is set before
anything is allocated, and mirror.final.prereg.require_prereg() runs before the GPU is used. meta.json of the server:
<queue>/server/meta.json (mirror.final.prereg.write_final_meta); only request files sit at the top of the queue.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.final.eval import knn_fast as KF  # noqa: E402
from mirror.final.prereg import write_final_meta  # noqa: E402
from mirror.utils import log  # noqa: E402

LOCK = "server.lock"
STOP = "STOP"
SUBDIRS = ("running", "done", "failed", "server")
NOT_REQUESTS = {"server_meta.json", "meta.json"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queue", default="runs/final/_queue")
    p.add_argument("--cache-dir", default="cache", help="ImageNet caches (train224, train_index, validation)")
    p.add_argument("--cache-root", default="runs/final/_cache", help="original-model k-NN cache (knn_original/)")
    p.add_argument("--loader", default=KF.DEFAULT_LOADER, help="module:function (run_dir, step, device) -> "
                                                              "(adapter or None, model, info)")
    p.add_argument("--original-builder", default=None, help="module:function (model_name, device) -> adapter "
                                                            "(default mirror.adapters_timm.build_model_adapter)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--data-on", default="device", choices=["device", "host"],
                   help="images on the GPU (default) or in pinned host memory")
    p.add_argument("--poll", type=float, default=5.0, help="seconds between looks at an empty queue")
    p.add_argument("--settle", type=float, default=30.0, help="seconds an unreadable request is left for its writer")
    p.add_argument("--once", action="store_true", help="exit when the queue is empty")
    p.add_argument("--max-idle", type=float, default=None, help="exit after this many idle seconds")
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--low-priority-first", action="store_true", help="smallest priority value first")
    p.add_argument("--gpu-frac", type=float, default=None, help="per-process GPU memory cap (default: automatic)")
    p.add_argument("--reserve-gb", type=float, default=12.0, help="GPU memory beyond the images (automatic cap)")
    p.add_argument("--force-lock", action="store_true", help="take the queue even if another server's lock is fresh")
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny model, synthetic data, a stand-in loader")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Queue mechanics
# ----------------------------------------------------------------------------------------------
def queue_dirs(queue: Path) -> None:
    queue.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (queue / sub).mkdir(exist_ok=True)


def read_request(path: Path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def pending(queue: Path, high_first: bool = True, settle: float = 30.0) -> list:
    """[(path, request or None)] in the order the server takes them; an unparsable file younger than `settle`
    seconds is skipped (still being written), an older one is returned with None (it will fail)."""
    items = []
    now = time.time()
    for path in sorted(queue.glob("*.json")):
        if path.name.startswith(".") or path.name in NOT_REQUESTS:
            continue
        req = read_request(path)
        if req is None:
            try:
                age = now - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < settle:
                continue
            items.append((path, None, 0))
            continue
        try:
            prio = int(req.get("priority") or 0)
        except (TypeError, ValueError):
            prio = 0
        items.append((path, req, prio))
    items.sort(key=lambda t: ((-t[2] if high_first else t[2]), t[0].name))
    return [(p, r) for p, r, _ in items]


def claim(path: Path, queue: Path):
    """Move a pending request to running/ (atomic); None if another reader took it first."""
    dest = queue / "running" / path.name
    try:
        os.replace(path, dest)
    except FileNotFoundError:
        return None
    return dest


def recover(queue: Path) -> list:
    """Requests left in running/ by a stopped server go back to the queue."""
    back = []
    for p in sorted((queue / "running").glob("*.json")):
        os.replace(p, queue / p.name)
        back.append(p.name)
    return back


def _record(path: Path, req, extra: dict) -> dict:
    rec = {"request": req if req is not None else {"raw": _raw(path)}, **extra}
    return rec


def _raw(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return ""


def finish(claimed: Path, queue: Path, req: dict, rec: dict) -> Path:
    summary = {k: {f: v.get(f) for f in ("top1", "flip_consistency", "n_query", "n_bank", "seconds")}
               for k, v in (rec.get("passes") or {}).items()}
    out = queue / "done" / claimed.name
    KF.write_json_atomic(out, _record(claimed, req, {
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tag": rec.get("tag"),
        "eval_json": str(Path(rec.get("request", {}).get("out_dir") or rec.get("run_dir") or "")
                         / "eval" / f"{rec.get('tag')}.json"), "passes": summary}))
    claimed.unlink(missing_ok=True)
    return out


def fail(claimed: Path, queue: Path, req, err: BaseException) -> Path:
    out = queue / "failed" / claimed.name
    KF.write_json_atomic(out, _record(claimed, req, {
        "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error": f"{type(err).__name__}: {err}",
        "traceback": "".join(traceback.format_exception(type(err), err, err.__traceback__))[-8000:]}))
    claimed.unlink(missing_ok=True)
    return out


class QueueLock:
    """A heartbeat lock file: fresh while its server touches it every poll; stale after `stale` seconds."""

    def __init__(self, queue: Path, stale: float):
        self.path, self.stale = queue / LOCK, float(stale)

    def acquire(self, force: bool = False) -> None:
        if self.path.exists() and not force:
            age = time.time() - self.path.stat().st_mtime
            if age < self.stale:
                raise SystemExit(f"{self.path} is held by another server ({_raw(self.path).strip()}; touched "
                                 f"{age:.0f} s ago); stop it or pass --force-lock")
        self.path.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}),
                             encoding="utf-8")

    def beat(self) -> None:
        try:
            os.utime(self.path, None)
        except FileNotFoundError:
            pass

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def process(ev, req: dict, loader, build_original, adapters: dict, should_stop) -> dict:
    """evaluate_request, with one retry at half the batch size after a CUDA out-of-memory error."""
    try:
        return KF.evaluate_request(ev, req, loader, build_original, adapters, should_stop=should_stop)
    except torch.cuda.OutOfMemoryError:
        KF.free_memory(ev.device)
        old = ev.batch_size
        ev.batch_size = max(1, old // 2)
        log(f"out of memory: retrying the request at batch size {ev.batch_size}")
        try:
            return KF.evaluate_request(ev, req, loader, build_original, adapters, should_stop=should_stop)
        finally:
            ev.batch_size = old


def serve(queue, ev, loader, build_original=None, *, poll: float = 5.0, settle: float = 30.0, once: bool = False,
          max_idle: float | None = None, max_requests: int | None = None, high_first: bool = True,
          force_lock: bool = False) -> dict:
    """The server loop (module docstring). Returns the counts of done and failed requests."""
    queue = Path(queue)
    queue_dirs(queue)
    lock = QueueLock(queue, stale=max(3 * poll, 60.0))
    lock.acquire(force_lock)
    counts = {"done": 0, "failed": 0, "recovered": 0}
    adapters: dict = {}
    stop = {"flag": False}

    def should_stop():
        return stop["flag"] or (queue / STOP).exists()

    try:
        back = recover(queue)
        counts["recovered"] = len(back)
        if back:
            log(f"requeued {len(back)} request(s) left in running/: {back}")
        idle_since = time.time()
        while True:
            lock.beat()
            if (queue / STOP).exists():
                log(f"{queue / STOP} found: stopping")
                (queue / STOP).unlink(missing_ok=True)
                break
            items = pending(queue, high_first, settle)
            if not items:
                if once or (max_idle is not None and time.time() - idle_since >= max_idle):
                    break
                time.sleep(poll)
                continue
            path, req = items[0]
            claimed = claim(path, queue)
            if claimed is None:
                continue
            t0 = time.time()
            try:
                if req is None:
                    raise ValueError(f"request {path.name} is not a readable JSON object")
                log(f"request {path.name}: run_dir {req.get('run_dir')}, step {req.get('step')}, passes "
                    f"{req.get('passes')}, n_query {req.get('n_query')}, priority {req.get('priority')}")
                rec = process(ev, req, loader, build_original, adapters, should_stop)
                if should_stop() and any(k not in rec.get("passes", {}) for k in _wanted_keys(ev, req)):
                    os.replace(claimed, queue / claimed.name)        # unfinished: back to the queue
                    log(f"request {path.name}: stopped before the end; back in the queue")
                    (queue / STOP).unlink(missing_ok=True)
                    break
                finish(claimed, queue, req, rec)
                counts["done"] += 1
                del rec
                KF.free_memory(ev.device)                    # give cached blocks back on a shared GPU
                log(f"request {path.name}: done in {time.time() - t0:.0f} s")
            except Exception as err:                       # a bad request must not stop the server
                fail(claimed, queue, req, err)
                counts["failed"] += 1
                log(f"request {path.name}: FAILED ({type(err).__name__}: {err}); moved to failed/")
                KF.free_memory(ev.device)
            idle_since = time.time()
            if max_requests is not None and counts["done"] + counts["failed"] >= max_requests:
                break
    finally:
        lock.release()
    return counts


def _wanted_keys(ev, req: dict) -> list:
    try:
        r = KF.normalise_request(req)
    except Exception:
        return []
    keys = []
    for p in r["passes"]:
        fp32 = p.startswith("fp32")
        nq = r["fp32_n_query"] if fp32 and r["fp32_n_query"] is not None else r["n_query"]
        keys.append(ev.counts(p, nq, r["fp32_bank_per_class"] if fp32 else None)[2])
    return keys


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t0 = time.time()
    queue = Path(a.queue)
    queue_dirs(queue)
    ev, device, frac, st = KF.make_evaluator(dry_run=a.dry_run, cache_dir=a.cache_dir, device=a.device,
                                             batch_size=a.batch_size, data_on=a.data_on, gpu_frac=a.gpu_frac,
                                             reserve_gb=a.reserve_gb, cache_root=a.cache_root, log=log)
    if a.dry_run and not pending(queue, settle=0.0):          # a self-contained smoke test: two demo requests
        run = queue.parent / "dry_run_runs" / "tiny" / "mirror_exact" / "s0"
        for i, (step, passes, nq, prio) in enumerate(((1, ["bf16", "fp32"], None, 1), (0, ["bf16"], 8, 5))):
            KF.write_json_atomic(queue / f"dryrun_{i}.json", {"run_dir": str(run), "step": step, "passes": passes,
                                                               "n_query": nq, "priority": prio})
    loader = KF.tiny_loader if a.dry_run else a.loader
    build_original = (lambda name, dev: KF.tiny_original(name, dev)) if a.dry_run else a.original_builder
    if frac is not None:
        try:
            loader = KF.resolve_callable(loader)             # import now, then re-apply this process's cap
        except Exception as err:                             # requests will fail one by one (failed/), not the server
            log(f"WARNING: the loader {loader} cannot be imported yet ({type(err).__name__}: {err})")
        torch.cuda.set_per_process_memory_fraction(float(frac), device)
    write_final_meta(queue / "server", name="meta.json", experiment="fp_eval_server",
                     config={k: v for k, v in vars(a).items()}, seeds={"note": "no random parts"},
                     device=device, extra={"data": ev.data.source, "gpu_frac": frac, "prereg_ok": st.get("ok"),
                                           "prereg_allowed_without_check": bool(st.get("allowed_without_check"))})
    log(f"evaluation server: queue {queue}, device {device}, data {ev.data.source}, batch {a.batch_size}"
        + (f", GPU cap {frac:.3f}" if frac is not None else ""))

    def on_term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    prev_handler = None
    if hasattr(signal, "SIGTERM"):
        try:
            prev_handler = signal.signal(signal.SIGTERM, on_term)
        except ValueError:                       # not the main thread
            prev_handler = None
    try:
        counts = serve(queue, ev, loader, build_original, poll=a.poll, settle=a.settle, once=a.once,
                       max_idle=a.max_idle, max_requests=a.max_requests, high_first=not a.low_priority_first,
                       force_lock=a.force_lock)
    except KeyboardInterrupt as e:
        log(f"stopped ({e}); requests in running/ go back to the queue at the next start")
        counts = {"stopped": True}
    finally:
        if prev_handler is not None:
            signal.signal(signal.SIGTERM, prev_handler)
    counts["seconds"] = round(time.time() - t0, 1)
    log(f"evaluation server finished: {counts}")
    return counts


if __name__ == "__main__":
    main()
