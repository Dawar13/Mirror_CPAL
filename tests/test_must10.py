"""MUST-10 (scripts/must10_flipavg.py): healed Mirror-exact against test-time flip averaging. CPU only: a tiny
MUST-09 run (scripts/must09_heal.py --tiny) is healed first, then compared with the flip-averaged tiny original."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mirror.adapters import TimmTinyAdapter
from mirror.data import mirror

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_m10_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def healed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("must09")
    M9 = load("must09_heal")
    M9.main(["--tiny", "--method", "mirror_exact", "--mirror-builder", "shim", "--lr", "1e-4",
             "--out-root", str(root / "runs"), "--results-root", str(root / "results")])
    return root / "runs" / "tiny" / "mirror_exact" / "s0"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flip_averaging_is_exactly_invariant(dtype):
    """The flip-averaged feature MUST-10 scores (flip_avg_baseline.FlipAveragedAdapter) is the same for an image and
    its mirror bit for bit; the timing form gives the same feature up to rounding."""
    M10 = load("must10_flipavg")
    ad = TimmTinyAdapter(img_size=56, seed=0, dtype=dtype)
    x = torch.randn(4, 3, 56, 56, dtype=dtype, generator=torch.Generator().manual_seed(1))
    fa = M10.FA.FlipAveragedAdapter(ad)
    with torch.no_grad():
        assert torch.equal(fa.cls_feature(x), fa.cls_feature(mirror(x)))
        assert not torch.equal(ad.cls_feature(x), ad.cls_feature(mirror(x)))       # the plain feature is not invariant
        for two in (False, True):
            t = M10.FlipAveragedFeature(ad, ad.model, two_passes=two)(x)
            assert (t.double() - fa.cls_feature(x).double()).abs().max().item() < 1e-5


def test_cli_tiny_end_to_end(healed_run, tmp_path):
    M10 = load("must10_flipavg")
    out, res = tmp_path / "runs", tmp_path / "results"
    args = ["--tiny", "--healed-run", str(healed_run), "--out-root", str(out), "--results-root", str(res),
            "--n-boot", "200"]
    r = M10.main(args + ["--latency"])
    rows = {(x["variant"], x["mode"]): x for x in r["rows"]}
    assert set(rows) == {("original", "plain"), ("original", "flipavg"), ("healed", "plain")}
    fa, hp, op = rows[("original", "flipavg")], rows[("healed", "plain")], rows[("original", "plain")]
    assert fa["flip_consistency_fp32"] == 100.0 and fa["feature_identical_pct_fp32"] == 100.0   # exact by construction
    assert hp["flip_consistency_fp32"] == 100.0                                              # exact by structure
    assert fa["forwards_per_image"] == 2 and fa["gflops_per_image"] == pytest.approx(2 * op["gflops_per_image"])
    assert hp["gflops_per_image"] < op["gflops_per_image"] and hp["params"] < op["params"] == fa["params"]
    for row in rows.values():
        assert row["probe_top1"] is not None and 0 <= row["knn_top1"] <= 100
    comp = json.loads((res / "comparison_tiny_s0.json").read_text())
    assert comp["gflops_ratio"] < 0.5 and comp["params_ratio"] < 1
    assert comp["knn_bootstrap"]["n"] == hp["n_query"] and "probe_bootstrap" in comp
    md = (res / "MUST10.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in md and "flip-averaged" in md
    meta = json.loads((res / "meta.json").read_text())
    assert meta["experiment"] == "MUST-10" and meta["extra"]["healed_run"] == str(healed_run)
    assert {x["variant"] for x in r["latency"]} == {"original_plain", "original_flipavg_batched",
                                                     "original_flipavg_two_passes", "healed_as_stored",
                                                     "healed_even_odd"}
    assert (res / "flipavg.csv").is_file() and (res / "latency.csv").is_file()
    # resumable: a rerun reuses every finished part
    calls = []
    orig_knn, orig_probe = M10.knn_part, M10.probe_part
    M10.knn_part = lambda *a, **k: calls.append("knn") or orig_knn(*a, **k)
    M10.probe_part = lambda *a, **k: calls.append("probe") or orig_probe(*a, **k)
    try:
        r2 = M10.main(args)
    finally:
        M10.knn_part, M10.probe_part = orig_knn, orig_probe
    assert calls == [] and [x["knn_top1"] for x in r2["rows"]] == [x["knn_top1"] for x in r["rows"]]


def test_refuses_a_run_that_is_not_full_mirror_exact(tmp_path):
    """MUST-10 compares exactly invariant models; a compressed baseline's run is refused."""
    M9 = load("must09_heal")
    M9.main(["--tiny", "--method", "svd", "--compressor-builder", "shim", "--lr", "1e-4", "--no-fp32-pass",
             "--out-root", str(tmp_path / "runs"), "--results-root", str(tmp_path / "results")])
    M10 = load("must10_flipavg")
    with pytest.raises(SystemExit, match="not a full-conversion Mirror-exact"):
        M10.main(["--tiny", "--healed-run", str(tmp_path / "runs" / "tiny" / "svd" / "s0"),
                  "--out-root", str(tmp_path / "o"), "--results-root", str(tmp_path / "r")])
