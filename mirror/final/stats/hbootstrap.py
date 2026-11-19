"""A13: hierarchical paired bootstrap over fit seeds, healing seeds and validation images.

FINAL_PHASE_PLAN.md v2, A13 and Section 6 (A05-1 uses the lower end of the interval of a difference).

Data. Every method has one or more runs. A run is one healed model: a per-image score vector over the SAME N
validation images in the same order (usually k-NN correctness, `pred == labels`, 0 or 1; real values are accepted,
as in mirror/bootstrap.py), a fit-seed label (which fitted structure or statistics it came from) and a healing-seed
label. Labels are any hashable values (ints or strings).

One bootstrap resample:
  1. per method, draw its fit seeds with replacement (as many draws as the method has fit seeds);
  2. for every drawn fit seed, draw that fit seed's healing seeds with replacement (as many draws as it has);
  3. draw N images with replacement, ONCE per resample, and use the same draw for every method (so every
     comparison is paired over images). The draw is the multinomial count vector of N uniform indices;
  4. a method's score is the mean, over its drawn runs (with multiplicity), of the run's accuracy on the drawn images.
The image draws use their own generator, default_rng(seed), and the same chunked `integers(0, N, (chunk, N))` calls
as mirror.bootstrap.paired_bootstrap, so with one run per method the result equals that function's draw for draw.
Seed draws use a separate generator, default_rng([seed, 7919]).

By default each method's seeds are drawn independently (the methods' seed effects are treated as independent, as in
a two-sample comparison). With paired_seeds=True, methods whose (fit seed, healing seed) label sets are identical
share one seed draw per resample (pairing by seed label, for designs where seed s means the same data order for every
method). The pre-registration fixes which one is reported.

Reported (accuracies in percentage points, like mirror/bootstrap.py):
  per method: mean over runs, standard deviation over runs (ddof 1), per-fit-seed means, the bootstrap standard error
  and the 95% percentile interval of the method's score;
  per pair (a, b): difference mean_a - mean_b, its bootstrap standard error and 95% percentile interval, the share
  of resamples with a difference <= 0, and the effect size difference / pooled seed standard deviation, with
  pooled^2 = ((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2) over runs (None when undefined).

    python -m mirror.final.stats.hbootstrap --runs runs.yaml --out results/A13 --reference mirror_exact

runs.yaml (or .json): a list of {method, fit_seed, heal_seed, npz} (npz in the MUST-09 k-NN format with arrays pred
and labels; `key` picks another prediction array, for example pred_flip). All runs must share the label array.
No external sources were used; the procedure is the one written in the plan.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

SEED_STREAM = 7919


@dataclass
class Run:
    method: str
    fit_seed: object
    heal_seed: object
    correct: np.ndarray


def _as_runs(runs) -> list[Run]:
    out = []
    for r in runs:
        if isinstance(r, Run):
            out.append(r)
        elif isinstance(r, dict):
            out.append(Run(str(r["method"]), r.get("fit_seed", 0), r.get("heal_seed", 0), r["correct"]))
        else:
            method, fit_seed, heal_seed, correct = r
            out.append(Run(str(method), fit_seed, heal_seed, correct))
    return out


def _label_key(x):
    return (0, x) if isinstance(x, (int, np.integer)) and not isinstance(x, bool) else (1, str(x))


class _Method:
    """Runs of one method grouped by fit seed (labels sorted, so the structure is canonical)."""

    def __init__(self, name: str, runs: list[Run], col_of: dict):
        self.name = name
        fits: dict = {}
        for r in runs:
            fits.setdefault(r.fit_seed, []).append(r)
        self.fit_labels = sorted(fits, key=_label_key)
        self.cols, self.heal_labels, sizes = [], [], []
        for f in self.fit_labels:
            rs = sorted(fits[f], key=lambda r: _label_key(r.heal_seed))
            heals = [r.heal_seed for r in rs]
            if len(set(map(_label_key, heals))) != len(heals):
                raise ValueError(f"method {name!r}: fit seed {f!r} lists a healing seed twice")
            self.heal_labels.append(heals)
            self.cols += [col_of[id(r)] for r in rs]
            sizes.append(len(rs))
        self.sizes = np.asarray(sizes, dtype=np.int64)              # runs per fit seed
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64)
        self.n_runs = int(self.sizes.sum())
        self.signature = tuple((_label_key(f), tuple(map(_label_key, h)))
                               for f, h in zip(self.fit_labels, self.heal_labels))

    @property
    def trivial(self) -> bool:
        return self.n_runs == 1

    def draw_weights(self, rng: np.random.Generator, m: int) -> np.ndarray:
        """[m, n_runs] weights (each row sums to 1) of the runs drawn in m resamples."""
        if self.trivial:
            return np.ones((m, 1))
        F = len(self.sizes)
        fdraw = rng.integers(0, F, size=(m, F))                       # drawn fit seeds
        hmax = int(self.sizes.max())
        hsel = self.sizes[fdraw]                                      # [m, F] runs of each drawn fit seed
        u = rng.random(size=(m, F, hmax))
        hidx = np.minimum((u * hsel[..., None]).astype(np.int64), hsel[..., None] - 1)
        valid = np.arange(hmax)[None, None, :] < hsel[..., None]
        local = self.offsets[fdraw][..., None] + hidx                 # [m, F, hmax] run index within the method
        rows = np.broadcast_to(np.arange(m)[:, None, None], local.shape)
        flat = (rows * self.n_runs + local)[valid]
        counts = np.bincount(flat, minlength=m * self.n_runs).reshape(m, self.n_runs).astype(np.float64)
        return counts / counts.sum(axis=1, keepdims=True)


def _image_counts(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    """[m, n] multinomial counts of n uniform draws per resample (the paired_bootstrap index draws, counted)."""
    idx = rng.integers(0, n, size=(m, n))
    idx += (np.arange(m, dtype=np.int64) * n)[:, None]
    return np.bincount(idx.ravel(), minlength=m * n).reshape(m, n).astype(np.float64)


def _pooled_sd(sa, na, sb, nb):
    dof = (na - 1) + (nb - 1)
    if dof <= 0:
        return None
    va = 0.0 if na < 2 else sa ** 2
    vb = 0.0 if nb < 2 else sb ** 2
    return float(np.sqrt(((na - 1) * va + (nb - 1) * vb) / dof))


def hierarchical_bootstrap(runs, n_resamples: int = 10_000, seed: int = 0, alpha: float = 0.05, chunk: int = 100,
                           pairs=None, reference: str | None = None, paired_seeds: bool = False,
                           labels=None, keep_draws: bool = False) -> dict:
    """Hierarchical paired bootstrap (see the module docstring).

    runs: list of Run, dicts {method, fit_seed, heal_seed, correct} or tuples (method, fit_seed, heal_seed, correct).
    pairs: list of (a, b) method names; default: every pair (a before b in first-appearance order), or
    (reference, other) for every other method when `reference` is given.
    labels: optional list of label arrays, one per run, that must all be equal (alignment check).
    keep_draws: also return the raw resampled scores (arrays) under "draws"."""
    t0 = time.time()
    runs = _as_runs(runs)
    if not runs:
        raise ValueError("no runs")
    if labels is not None:
        ref = np.asarray(labels[0])
        for i, lab in enumerate(labels):
            if not np.array_equal(np.asarray(lab), ref):
                raise ValueError(f"run {i} has another label array: the runs are not aligned on the same images")
    sizes = sorted({np.asarray(r.correct).size for r in runs})
    if len(sizes) != 1:
        raise ValueError(f"every run needs a score vector over the same N images: lengths {sizes}")
    X = np.stack([np.asarray(r.correct, dtype=np.float64).ravel() for r in runs], axis=1)
    n = X.shape[0]
    if n == 0:
        raise ValueError("empty score vectors")
    col_of = {id(r): i for i, r in enumerate(runs)}
    order = []
    for r in runs:
        if r.method not in order:
            order.append(r.method)
    methods = {m: _Method(m, [r for r in runs if r.method == m], col_of) for m in order}
    if pairs is None:
        if reference is not None:
            if reference not in methods:
                raise ValueError(f"reference {reference!r} has no runs")
            pairs = [(reference, m) for m in order if m != reference]
        else:
            pairs = list(combinations(order, 2))
    for a, b in pairs:
        if a not in methods or b not in methods:
            raise ValueError(f"pair ({a!r}, {b!r}) names a method without runs")

    rng_img = np.random.default_rng(seed)
    rng_seed = np.random.default_rng([seed, SEED_STREAM])
    scores = {m: np.empty(n_resamples) for m in order}
    for s in range(0, n_resamples, chunk):
        m_ = min(chunk, n_resamples - s)
        acc = _image_counts(rng_img, n, m_) @ X / n                     # [m_, runs] accuracy on the drawn images
        shared = {}
        for name in order:
            meth = methods[name]
            if paired_seeds and not meth.trivial:
                if meth.signature not in shared:
                    shared[meth.signature] = meth.draw_weights(rng_seed, m_)
                w = shared[meth.signature]
            else:
                w = meth.draw_weights(rng_seed, m_)
            scores[name][s:s + m_] = (acc[:, meth.cols] * w).sum(axis=1)

    q = [alpha / 2, 1 - alpha / 2]
    out_methods = {}
    for name in order:
        meth = methods[name]
        run_acc = X[:, meth.cols].mean(axis=0)
        sd = float(run_acc.std(ddof=1)) if meth.n_runs > 1 else None
        fit_means = [float(X[:, meth.cols[o:o + k]].mean()) for o, k in zip(meth.offsets, meth.sizes)]
        lo, hi = np.quantile(scores[name], q)
        out_methods[name] = {
            "n_runs": meth.n_runs, "n_fit_seeds": len(meth.fit_labels),
            "fit_seeds": [str(f) for f in meth.fit_labels],
            "runs": [{"fit_seed": str(f), "heal_seed": str(h), "acc": 100.0 * float(X[:, c].mean())}
                     for f, hs, cs in zip(meth.fit_labels, meth.heal_labels,
                                          np.split(np.asarray(meth.cols), np.cumsum(meth.sizes)[:-1]))
                     for h, c in zip(hs, cs)],
            "mean": 100.0 * float(run_acc.mean()),
            "sd_runs": None if sd is None else 100.0 * sd,
            "fit_seed_means": [100.0 * v for v in fit_means],
            "sd_fit_seed_means": 100.0 * float(np.std(fit_means, ddof=1)) if len(fit_means) > 1 else None,
            "boot_se": 100.0 * float(scores[name].std(ddof=1)),
            "ci_low": 100.0 * float(lo), "ci_high": 100.0 * float(hi),
        }
    out_pairs = []
    for a, b in pairs:
        d = scores[a] - scores[b]
        lo, hi = np.quantile(d, q)
        ma, mb = out_methods[a], out_methods[b]
        diff = ma["mean"] - mb["mean"]
        pooled = _pooled_sd(ma["sd_runs"] or 0.0, ma["n_runs"], mb["sd_runs"] or 0.0, mb["n_runs"])
        effect = None if pooled is None or pooled == 0.0 else diff / pooled
        out_pairs.append({
            "a": a, "b": b, "diff": diff, "boot_mean_diff": 100.0 * float(d.mean()),
            "se": 100.0 * float(d.std(ddof=1)), "ci_low": 100.0 * float(lo), "ci_high": 100.0 * float(hi),
            "p_le_0": float(np.mean(d <= 0)), "excludes_zero": bool(lo > 0 or hi < 0),
            "pooled_sd": pooled, "effect_size": effect,
            "effect_note": None if effect is not None else
            ("pooled seed standard deviation undefined (one run per method)" if pooled is None
             else "pooled seed standard deviation is 0"),
        })
    res = {"n_images": int(n), "n_resamples": int(n_resamples), "seed": int(seed), "alpha": float(alpha),
           "chunk": int(chunk), "paired_seeds": bool(paired_seeds), "units": "percentage points",
           "methods": out_methods, "pairs": out_pairs, "seconds": round(time.time() - t0, 2)}
    if keep_draws:
        res["draws"] = scores
    return res


# ----------------------------------------------------------------------------------------------
# Loading k-NN prediction files and the command line
# ----------------------------------------------------------------------------------------------
def load_knn_correct(path, key: str = "pred") -> tuple[np.ndarray, np.ndarray]:
    """(correct [N] bool, labels [N] int64) from a k-NN prediction file (arrays `key` and labels)."""
    with np.load(path) as z:
        if key not in z.files or "labels" not in z.files:
            raise KeyError(f"{path}: needs arrays {key!r} and 'labels', has {z.files}")
        pred, lab = np.asarray(z[key]), np.asarray(z["labels"], dtype=np.int64)
    if pred.shape != lab.shape:
        raise ValueError(f"{path}: {key} has shape {pred.shape}, labels {lab.shape}")
    return pred == lab, lab


def runs_from_spec(spec: list, base: Path | None = None) -> tuple[list[Run], list[np.ndarray]]:
    runs, labels = [], []
    for e in spec:
        p = Path(e["npz"])
        if base is not None and not p.is_absolute():
            p = base / p
        correct, lab = load_knn_correct(p, e.get("key", "pred"))
        runs.append(Run(str(e["method"]), e.get("fit_seed", 0), e.get("heal_seed", 0), correct))
        labels.append(lab)
    return runs, labels


def write_outputs(res: dict, out_dir) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in res.items() if k != "draws"}
    (out / "hbootstrap.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    with open(out / "hbootstrap_methods.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_runs", "n_fit_seeds", "mean", "sd_runs", "sd_fit_seed_means", "boot_se",
                    "ci_low", "ci_high"])
        for m, r in res["methods"].items():
            w.writerow([m, r["n_runs"], r["n_fit_seeds"], r["mean"], r["sd_runs"], r["sd_fit_seed_means"],
                        r["boot_se"], r["ci_low"], r["ci_high"]])
    with open(out / "hbootstrap_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "diff", "se", "ci_low", "ci_high", "p_le_0", "effect_size", "pooled_sd"])
        for p in res["pairs"]:
            w.writerow([p["a"], p["b"], p["diff"], p["se"], p["ci_low"], p["ci_high"], p["p_le_0"],
                        p["effect_size"], p["pooled_sd"]])
    return [out / "hbootstrap.json", out / "hbootstrap_methods.csv", out / "hbootstrap_pairs.csv"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="YAML or JSON list of {method, fit_seed, heal_seed, npz[, key]}")
    p.add_argument("--out", required=True)
    p.add_argument("--reference", default=None, help="compare this method with every other one")
    p.add_argument("--n-resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--paired-seeds", action="store_true")
    a = p.parse_args(argv)
    path = Path(a.runs)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        spec = yaml.safe_load(text)
    else:
        spec = json.loads(text)
    runs, labels = runs_from_spec(spec, base=path.parent)
    res = hierarchical_bootstrap(runs, a.n_resamples, a.seed, a.alpha, a.chunk, reference=a.reference,
                                 paired_seeds=a.paired_seeds, labels=labels)
    res["runs_file"] = str(path)
    for f in write_outputs(res, a.out):
        print(f"wrote {f}")
    for pr in res["pairs"]:
        print(f"{pr['a']} - {pr['b']}: {pr['diff']:+.3f} [{pr['ci_low']:+.3f}, {pr['ci_high']:+.3f}] "
              f"effect {pr['effect_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
