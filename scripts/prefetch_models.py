#!/usr/bin/env python
"""Download every model weight the experiments need, once, under $HF_HOME (VM only).

    export HF_HOME=/workspace/hf
    python scripts/prefetch_models.py                     # all; -> results/data_check/models.json
    python scripts/prefetch_models.py --dry-run           # print the plan, download nothing (laptop safe)
    python scripts/prefetch_models.py --only deit3_b16,dinov2_vitg14_reg
    python scripts/prefetch_models.py --extra-timm convnext_base.fb_in1k,resnet50.a1_in1k

Models:
  - every entry of mirror.adapters_timm.MODEL_REGISTRY (what resolve_name knows), read at run time, so
    entries other people add are included: source "timm" -> timm.create_model(timm_name,
    pretrained=True) (weights go to $HF_HOME/hub); source "torch_hub" -> torch.hub.load as
    mirror.adapters.Dinov2Adapter does it (hub dir $HF_HOME/torch_hub unless TORCH_HOME is set);
    source "hf" (hf_id or repo_id) -> huggingface_hub.snapshot_download; anything else is reported as
    skipped;
  - the DINOv2 hub models S, B, L and g, plain and with registers (PROJECT_NEXT_PHASE.md, Section 4.2),
    also when the registry lacks them;
  - --extra-timm and --extra-hf names (for models not yet in the registry).
Each model is loaded once on the CPU (which checks that the weights load) and then freed. A failure
(for example a gated model without access) is recorded and the rest continue. The record
(results/data_check/models.json) holds per model the status, seconds, the timm pretrained_cfg hub id
and the revision the hub served, and the size and sha256 of every torch.hub checkpoint file.
Rerunning reuses the downloaded files. Exit code 1 if any model failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
DINOV2_HUB = "facebookresearch/dinov2"
DINOV2_NAMES = [f"dinov2_vit{s}14{r}" for r in ("", "_reg") for s in ("s", "b", "l", "g")]


def plan(registry: dict, extra_timm=(), extra_hf=(), only=None) -> list[dict]:
    """One action per model: {"name", "kind" (timm | torch_hub | hf | skip), "id", ...}; no duplicates."""
    items, seen = [], set()

    def add(item):
        key = (item["kind"], item["id"])
        if key in seen:
            return
        seen.add(key)
        items.append(item)

    for name, e in registry.items():
        src = e.get("source")
        if src == "timm" and e.get("timm_name"):
            add({"name": name, "kind": "timm", "id": e["timm_name"]})
        elif src == "torch_hub" and e.get("hub_name"):
            add({"name": name, "kind": "torch_hub", "id": e["hub_name"], "repo": e.get("hub_repo", DINOV2_HUB)})
        elif e.get("hf_id") or e.get("repo_id"):
            add({"name": name, "kind": "hf", "id": e.get("hf_id") or e.get("repo_id"), "revision": e.get("revision")})
        else:
            add({"name": name, "kind": "skip", "id": name, "why": f"unknown source {src!r}"})
    for hub_name in DINOV2_NAMES:
        add({"name": hub_name, "kind": "torch_hub", "id": hub_name, "repo": DINOV2_HUB})
    for t in extra_timm:
        add({"name": t, "kind": "timm", "id": t})
    for h in extra_hf:
        add({"name": h, "kind": "hf", "id": h, "revision": None})
    if only:
        items = [it for it in items if it["name"] in only or it["id"] in only]
    return items


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _hub_dir() -> str:
    import torch

    hf_home = os.environ.get("HF_HOME")
    if hf_home and "TORCH_HOME" not in os.environ:      # the same rule as mirror.adapters.Dinov2Adapter
        torch.hub.set_dir(os.path.join(hf_home, "torch_hub"))
    return torch.hub.get_dir()


def fetch(item: dict) -> dict:
    """Download and load one model on the CPU; return its record."""
    t0 = time.time()
    rec = {k: item[k] for k in item}
    if item["kind"] == "skip":
        rec["status"] = "skipped"
        return rec
    if item["kind"] == "timm":
        import timm

        model = timm.create_model(item["id"], pretrained=True)
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        rec["hf_hub_id"] = cfg.get("hf_hub_id")
        rec["params"] = int(sum(p.numel() for p in model.parameters()))
        del model
        if rec["hf_hub_id"]:
            try:
                from huggingface_hub import HfApi

                rec["hf_revision"] = HfApi().model_info(rec["hf_hub_id"]).sha
            except Exception as exc:        # the weights are on disk; the revision is a nicety
                rec["hf_revision"] = f"unavailable ({type(exc).__name__})"
    elif item["kind"] == "torch_hub":
        import torch

        hub = _hub_dir()
        local = os.path.join(hub, item["repo"].replace("/", "_") + "_main")
        if os.path.isfile(os.path.join(local, "hubconf.py")):
            model = torch.hub.load(local, item["id"], source="local")
        else:
            model = torch.hub.load(item["repo"], item["id"], trust_repo=True)
        rec["params"] = int(sum(p.numel() for p in model.parameters()))
        rec["hub_dir"] = hub
        del model
    elif item["kind"] == "hf":
        from huggingface_hub import snapshot_download

        rec["path"] = snapshot_download(item["id"], revision=item.get("revision"))
    rec["status"] = "ok"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default="", help="comma-separated names or ids")
    p.add_argument("--extra-timm", default="", help="comma-separated timm names (with pretrained tag)")
    p.add_argument("--extra-hf", default="", help="comma-separated Hugging Face model repo ids")
    p.add_argument("--no-dinov2-g", action="store_true", help="skip the two g/14 models (about 4.5 GB each)")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "data_check" / "models.json"))
    a = p.parse_args(argv)
    from mirror.adapters_timm import MODEL_REGISTRY

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    items = plan(MODEL_REGISTRY, split(a.extra_timm), split(a.extra_hf), set(split(a.only)) or None)
    if a.no_dinov2_g:
        items = [it for it in items if "vitg14" not in it["id"]]
    for it in items:
        print(f"{it['kind']:9s} {it['name']:24s} {it['id']}", flush=True)
    if a.dry_run:
        return 0
    if not os.environ.get("HF_HOME"):
        print("WARNING: HF_HOME is not set; weights go to the default caches, not $HF_HOME", flush=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # CPU only: never touch the shared GPU
    records = []
    for it in items:
        try:
            rec = fetch(it)
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the rest
            rec = {**it, "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        records.append(rec)
        print(f"[{time.strftime('%H:%M:%S')}] {rec['status']:7s} {it['name']} ({rec.get('seconds', '-')} s)"
              + (f": {rec['error']}" if rec.get("error") else ""), flush=True)
    ckpts = []
    if any(r["kind"] == "torch_hub" and r["status"] == "ok" for r in records):
        cdir = Path(_hub_dir()) / "checkpoints"
        for f in sorted(cdir.glob("*")) if cdir.exists() else []:
            if f.is_file():
                ckpts.append({"file": f.name, "bytes": f.stat().st_size, "sha256": _sha256(f)})
    from mirror.utils import git_info

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hf_home": os.environ.get("HF_HOME"),
              "models": records, "torch_hub_checkpoints": ckpts, "git": git_info(),
              "failed": [r["name"] for r in records if r["status"] == "failed"]}
    if out.exists():                       # keep earlier successes when rerunning a subset
        try:
            old = json.loads(out.read_text())
            names = {r["name"] for r in records}
            report["models"] = [r for r in old.get("models", []) if r["name"] not in names] + records
        except (ValueError, KeyError):
            pass
    out.write_text(json.dumps(report, indent=2))
    n_ok = sum(r["status"] == "ok" for r in records)
    print(f"{n_ok}/{len(records)} models ready; failed: {report['failed'] or 'none'}; record {out}", flush=True)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
