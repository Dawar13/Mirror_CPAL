"""Every student of every Phase G job spec (jobs/heal/*.yaml, written by scripts/fp_make_jobs.py) builds through the
registry (mirror/final/heal/students.py build_student) on a tiny model, with the A01 decision both ways; and the runner
manifest (jobs/final_phase.yaml) passes the runner's own check. The structure files of the specs are replaced by a
tiny fitted SpanStructure, span sizes are capped at the tiny depth, and the block-drop scorer uses its synthetic
fallback. This catches misspelt builder options, missing methods and broken "adopted" choices before the GPU run."""
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mirror import mirror_family as MF
from mirror.data import make_loader, synthetic_dataset
from mirror.variants import TinyProtocol

os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")

from mirror.final.heal.students import BuildContext, build_student  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUILDER_KEYS = ("structure", "hparams", "random_seed", "lowrank_factors", "k", "frac", "n_drop", "max_candidates",
                "logits", "trainable")
SPECS = sorted(p for p in (ROOT / "jobs" / "heal").glob("*.yaml") if not p.name.startswith("w0_timing"))


def _must09():
    spec = importlib.util.spec_from_file_location("must09_heal_jobs_test", ROOT / "scripts" / "must09_heal.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fp_jobs")
    M = _must09()
    cfg0 = M.setup(M.parse_args(["--tiny", "--method", "mirror_exact"]))
    teacher, aspec = M.build_teacher(cfg0, torch.device("cpu"))
    images, labels = synthetic_dataset(4, 4, 56, seed=3)
    st = MF.fit_span_structure(teacher, make_loader(images, labels, np.arange(16), 8), "cpu", model_name="tiny",
                               top_k=3, workers=1, pairings=False)
    path = str(MF.save_span_structure(tmp / "tiny_s0.npz", st))
    return {"teacher": teacher, "aspec": aspec, "path": path, "tmp": tmp}


def _tiny_options(e: dict, path: str, depth: int) -> dict:
    hp = {k: copy.deepcopy(e[k]) for k in BUILDER_KEYS if k in e}
    if "structure" in hp:
        s = hp["structure"]
        if isinstance(s, dict):
            hp["structure"] = {"adopted": {"wspace": path, "astar": path}}
        elif s != "adopted":
            hp["structure"] = path
    if "k" in hp:
        hp["k"] = min(int(hp["k"]), depth - 1)
    if e["method"] == "blockdrop":
        hp["n_drop"] = 1
    return hp


def test_specs_exist_and_name_their_jobs():
    assert len(SPECS) >= 15
    for p in SPECS:
        spec = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert spec["job"] == p.stem and spec["students"], p.name
        assert 0 < float(spec["gpu_frac"]) <= 0.95


@pytest.mark.parametrize("adopted", [False, True])
def test_every_student_builds(tiny, adopted, tmp_path):
    from mirror.final.heal import blockdrop_score

    dec = tmp_path / "decision.json"
    dec.write_text(json.dumps({"adopted": adopted}))
    seen = set()
    for p in SPECS:
        spec = yaml.safe_load(p.read_text(encoding="utf-8"))
        ctx = BuildContext(adapter=tiny["teacher"], model_name="tiny", adapter_spec=tiny["aspec"], device="cpu",
                           structure=tiny["path"], gate_b=None, fit_loader=TinyProtocol().fit_loader,
                           a01_decision=str(dec), adopted_structures={"wspace": tiny["path"], "astar": tiny["path"]},
                           cache_dir=str(tmp_path / "no_cache"))
        if spec.get("blockdrop_score_fn"):
            ctx.extra["blockdrop_score_fn"] = blockdrop_score.msel_scorer(ctx)
        for e in spec["students"]:
            key = (e["method"], json.dumps(_tiny_options(e, tiny["path"], tiny["teacher"].depth), sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            sb = build_student(e["method"], ctx, **_tiny_options(e, tiny["path"], tiny["teacher"].depth))
            assert sb.method == e["method"], (p.name, e)
            assert any(t.requires_grad for t in sb.model.parameters()) or sb.train_scope, (p.name, e)
    assert {"mirror_exact", "svdllm", "svdllm_full", "iosvd", "flap", "blockdrop", "original", "slicegpt", "svd",
            "random_structure", "pair_random", "mirror_svdllm", "mirror_partial", "rot90_probe"} <= {m for m, _ in seen}


def test_manifest_passes_the_runner_check():
    from mirror.final.run import runner as R

    man = ROOT / "jobs" / "final_phase.yaml"
    assert man.exists()
    loader = getattr(R, "load_manifest", None)
    assert loader is not None
    m = loader(man)
    ids = [j.id if hasattr(j, "id") else j["id"] for j in (m.jobs if hasattr(m, "jobs") else m["jobs"])]
    assert len(ids) == len(set(ids)) and "gate_p" in ids and "a01_decide" in ids and "report" in ids
