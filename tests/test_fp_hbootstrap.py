"""mirror/final/stats/hbootstrap.py (A13): analytic checks of the hierarchical paired bootstrap."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from mirror.bootstrap import paired_bootstrap
from mirror.final.stats import hbootstrap as HB


def _bern(rng, p, n):
    return (rng.random(n) < p).astype(np.float64)


# ----------------------------------------------------------------------------------------------
# One run per method: image resampling only
# ----------------------------------------------------------------------------------------------
def test_one_run_each_matches_analytic_se():
    """Independent Bernoulli outcomes: the bootstrap SE of the difference is sqrt(var(a - b) / N)."""
    rng = np.random.default_rng(1)
    n = 20_000
    a, b = _bern(rng, 0.78, n), _bern(rng, 0.76, n)
    res = HB.hierarchical_bootstrap([("A", 0, 0, a), ("B", 0, 0, b)], n_resamples=10_000, seed=3)
    pr = res["pairs"][0]
    se = 100.0 * np.sqrt(np.var(a - b) / n)
    assert pr["se"] == pytest.approx(se, rel=0.03)
    assert pr["diff"] == pytest.approx(100.0 * (a.mean() - b.mean()), abs=1e-12)
    assert pr["ci_low"] < pr["diff"] < pr["ci_high"]
    for m, x in (("A", a), ("B", b)):
        assert res["methods"][m]["boot_se"] == pytest.approx(100.0 * np.sqrt(np.var(x) / n), rel=0.03)
        assert res["methods"][m]["sd_runs"] is None and res["methods"][m]["n_runs"] == 1
    assert pr["effect_size"] is None and "undefined" in pr["effect_note"]


def test_equals_paired_bootstrap_with_one_run_per_method():
    """Same seed and chunk: the image draws are those of mirror.bootstrap.paired_bootstrap, draw for draw."""
    rng = np.random.default_rng(5)
    n = 5_000
    a, b = _bern(rng, 0.8, n), _bern(rng, 0.79, n)
    b[:2000] = a[:2000]                                         # correlated, as two models on the same images
    ref = paired_bootstrap(a, b, n_resamples=4_000, seed=11, chunk=100)
    res = HB.hierarchical_bootstrap([("A", "f0", "h0", a), ("B", "f0", "h0", b)], n_resamples=4_000, seed=11,
                                    chunk=100)
    pr = res["pairs"][0]
    assert pr["diff"] == pytest.approx(ref["diff"], abs=1e-9)
    assert pr["ci_low"] == pytest.approx(ref["ci_low"], abs=1e-9)
    assert pr["ci_high"] == pytest.approx(ref["ci_high"], abs=1e-9)
    assert pr["excludes_zero"] == ref["excludes_zero"]
    assert res["methods"]["A"]["mean"] == pytest.approx(ref["mean_a"], abs=1e-9)


def test_pairing_over_images_shrinks_the_interval():
    rng = np.random.default_rng(2)
    n = 10_000
    a = _bern(rng, 0.8, n)
    b = a.copy()
    flip = rng.choice(n, 300, replace=False)
    b[flip] = 1 - b[flip]
    res = HB.hierarchical_bootstrap([("A", 0, 0, a), ("B", 0, 0, b)], n_resamples=5_000, seed=0)
    pr = res["pairs"][0]
    assert pr["se"] == pytest.approx(100.0 * np.sqrt(np.var(a - b) / n), rel=0.04)
    unpaired = 100.0 * np.sqrt((np.var(a) + np.var(b)) / n)
    assert pr["se"] < 0.4 * unpaired


# ----------------------------------------------------------------------------------------------
# Two levels: fit seeds and healing seeds
# ----------------------------------------------------------------------------------------------
def test_two_level_known_between_seed_variance():
    """Per-run scores constant over images (no image variance): the bootstrap variance of a method's score is
    (sigma_fit^2 + mean within-fit variance / H) / F, with population variances of the fixed run accuracies."""
    q = np.array([[0.70, 0.74], [0.78, 0.80], [0.73, 0.69]])    # F = 3 fit seeds x H = 2 healing seeds
    n = 50
    runs = [("M", f, h, np.full(n, q[f, h])) for f in range(3) for h in range(2)]
    runs.append(("ref", 0, 0, np.full(n, 0.7)))
    res = HB.hierarchical_bootstrap(runs, n_resamples=10_000, seed=4, keep_draws=True)
    F, H = q.shape
    var = (q.mean(axis=1).var() + q.var(axis=1).mean() / H) / F
    draws = res["draws"]["M"]
    assert draws.var(ddof=1) == pytest.approx(var, rel=0.05)
    assert draws.mean() == pytest.approx(q.mean(), abs=4 * np.sqrt(var / 10_000))
    m = res["methods"]["M"]
    assert m["mean"] == pytest.approx(100 * q.mean())
    assert m["sd_runs"] == pytest.approx(100 * q.ravel().std(ddof=1))
    assert m["fit_seed_means"] == pytest.approx(list(100 * q.mean(axis=1)))
    assert m["n_fit_seeds"] == 3 and m["n_runs"] == 6
    assert np.allclose(res["draws"]["ref"], 0.7, rtol=0, atol=1e-12)   # a constant single run never varies


def _exact_bootstrap_moments(x_runs, sizes):
    """Mean and variance of the bootstrap score, by enumerating every seed draw (fit draws, then healing draws)
    and adding the multinomial image variance pop_var(y) / N of each drawn mixture y."""
    n = x_runs.shape[1]
    F = len(sizes)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    means, vars_, probs = [], [], []
    for fseq in itertools.product(range(F), repeat=F):
        heal_choices = [list(itertools.product(range(sizes[f]), repeat=sizes[f])) for f in fseq]
        p_f = (1.0 / F) ** F
        for hseq in itertools.product(*heal_choices):
            p = p_f * np.prod([1.0 / len(c) for c in heal_choices])
            w = np.zeros(x_runs.shape[0])
            for f, hs in zip(fseq, hseq):
                for h in hs:
                    w[offsets[f] + h] += 1
            w /= w.sum()
            y = w @ x_runs
            means.append(y.mean())
            vars_.append(y.var() / n)
            probs.append(p)
    probs, means, vars_ = map(np.asarray, (probs, means, vars_))
    assert probs.sum() == pytest.approx(1.0)
    mu = (probs * means).sum()
    return mu, (probs * vars_).sum() + (probs * (means - mu) ** 2).sum()


def test_two_level_with_binary_images_matches_exact_enumeration():
    """Unequal healing seeds per fit seed (2 and 3), binary outcomes on 400 images: the bootstrap mean and variance
    of the score equal the exact values obtained by enumerating every seed draw."""
    rng = np.random.default_rng(7)
    n = 400
    acc = [0.60, 0.66, 0.80, 0.84, 0.75]                          # fit seed 0: runs 0-1; fit seed 1: runs 2-4
    x = np.stack([_bern(rng, p, n) for p in acc])
    runs = [("M", 0, 0, x[0]), ("M", 0, 1, x[1]), ("M", 1, 0, x[2]), ("M", 1, 1, x[3]), ("M", 1, 2, x[4])]
    res = HB.hierarchical_bootstrap(runs, n_resamples=10_000, seed=9, keep_draws=True)
    mu, var = _exact_bootstrap_moments(x, [2, 3])
    d = res["draws"]["M"]
    assert d.var(ddof=1) == pytest.approx(var, rel=0.05)
    assert d.mean() == pytest.approx(mu, abs=4 * np.sqrt(var / 10_000))


def test_paired_seeds_share_draws():
    """Two methods with the same seed labels and identical runs: with paired seeds every resampled difference is
    exactly 0; drawn independently it is not."""
    rng = np.random.default_rng(3)
    n = 300
    xs = {(f, h): _bern(rng, 0.6 + 0.08 * f + 0.03 * h, n) for f in range(2) for h in range(3)}
    runs = [(m, f, h, x) for m in ("A", "B") for (f, h), x in xs.items()]
    paired = HB.hierarchical_bootstrap(runs, n_resamples=2_000, seed=1, paired_seeds=True, keep_draws=True)
    assert np.array_equal(paired["draws"]["A"], paired["draws"]["B"])
    assert paired["pairs"][0]["se"] == 0.0 and paired["pairs"][0]["ci_low"] == 0.0
    indep = HB.hierarchical_bootstrap(runs, n_resamples=2_000, seed=1, keep_draws=True)
    assert indep["pairs"][0]["se"] > 0.1


def test_effect_size_and_reference_pairs():
    n = 1000
    rng = np.random.default_rng(0)
    base = _bern(rng, 0.5, n)

    def run_with(acc):                                          # exact accuracy, same images
        x = np.zeros(n)
        x[: int(round(acc * n))] = 1
        return x

    a = [0.783, 0.785, 0.781]
    b = [0.760, 0.761, 0.759]
    runs = [("mirror", 0, h, run_with(v)) for h, v in enumerate(a)]
    runs += [("svdllm", 0, h, run_with(v)) for h, v in enumerate(b)]
    runs += [("svd", 0, 0, base)]
    res = HB.hierarchical_bootstrap(runs, n_resamples=1_000, seed=0, reference="mirror")
    assert [(p["a"], p["b"]) for p in res["pairs"]] == [("mirror", "svdllm"), ("mirror", "svd")]
    pr = res["pairs"][0]
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = 100 * np.sqrt((2 * sa ** 2 + 2 * sb ** 2) / 4)
    assert pr["diff"] == pytest.approx(100 * (np.mean(a) - np.mean(b)))
    assert pr["pooled_sd"] == pytest.approx(pooled)
    assert pr["effect_size"] == pytest.approx(pr["diff"] / pooled)
    one = res["pairs"][1]                                        # svd has one run: pooled sd is mirror's alone
    assert one["pooled_sd"] == pytest.approx(100 * sa)


def test_deterministic_and_validates_inputs():
    rng = np.random.default_rng(0)
    x = [_bern(rng, 0.7, 200) for _ in range(4)]
    runs = [("A", 0, 0, x[0]), ("A", 1, 0, x[1]), ("B", 0, 0, x[2]), ("B", 0, 1, x[3])]
    r1 = HB.hierarchical_bootstrap(runs, n_resamples=500, seed=2)
    r2 = HB.hierarchical_bootstrap(runs, n_resamples=500, seed=2)
    r1.pop("seconds"), r2.pop("seconds")
    assert r1 == r2
    with pytest.raises(ValueError, match="same N images"):
        HB.hierarchical_bootstrap([("A", 0, 0, x[0]), ("B", 0, 0, x[1][:100])], n_resamples=10)
    with pytest.raises(ValueError, match="twice"):
        HB.hierarchical_bootstrap([("A", 0, 0, x[0]), ("A", 0, 0, x[1])], n_resamples=10)
    with pytest.raises(ValueError, match="not aligned"):
        HB.hierarchical_bootstrap(runs[:2], n_resamples=10, labels=[np.arange(200), np.arange(200)[::-1]])
    with pytest.raises(ValueError, match="without runs"):
        HB.hierarchical_bootstrap(runs, n_resamples=10, pairs=[("A", "C")])


def test_cli_reads_knn_files(tmp_path):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 500)
    spec = []
    for m, p in (("mirror", 0.8), ("svdllm", 0.75)):
        for h in range(2):
            pred = np.where(rng.random(500) < p, labels, (labels + 1) % 10)
            f = tmp_path / f"{m}_s{h}.npz"
            np.savez(f, pred=pred, pred_flip=pred, labels=labels)
            spec.append({"method": m, "fit_seed": 0, "heal_seed": h, "npz": f.name})
    (tmp_path / "runs.json").write_text(json.dumps(spec))
    assert HB.main(["--runs", str(tmp_path / "runs.json"), "--out", str(tmp_path / "out"),
                    "--reference", "mirror", "--n-resamples", "300"]) == 0
    saved = json.loads((tmp_path / "out" / "hbootstrap.json").read_text())
    assert saved["pairs"][0]["a"] == "mirror" and saved["methods"]["mirror"]["n_runs"] == 2
    assert (tmp_path / "out" / "hbootstrap_pairs.csv").read_text().startswith("a,b,diff")
    with np.load(tmp_path / "mirror_s0.npz") as z:
        pred, lab = z["pred"], z["labels"]
    np.savez(tmp_path / "mirror_s0.npz", pred=pred, labels=lab[::-1])
    with pytest.raises(ValueError, match="not aligned"):
        HB.main(["--runs", str(tmp_path / "runs.json"), "--out", str(tmp_path / "out2"), "--n-resamples", "10"])
