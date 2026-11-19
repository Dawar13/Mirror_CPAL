"""A02: stability of the operator, the twins and the head maps across independent fits, from the structures and
sidecars written by scripts/fp_structure_fit.py (FINAL_PHASE_PLAN.md A02; PREREG_FINAL_PHASE.md Section 2.3; the
metrics are defined and coded in mirror/stability.py).

  python scripts/fp_a02_stability.py --model dinov2_vitb14_reg --fits-dir results/A02/fits --out results/A02
  python scripts/fp_a02_stability.py --dry-run --out <dir>      (CPU: five tiny fits on synthetic images)

Fits (default): roles fit0, fit1, fit2, fit3 (group "imagenet") and coco (group "coco"), each read from
<fits-dir>/<model>_<role>.npz and <model>_<role>_sidecar.npz. --fit name=structure.npz[,sidecar.npz[,group]] adds a
fit or replaces a default one (for example a canonical MUST-07 structure with the sidecar of its refit, when the refit
reproduced it exactly).

Outputs in --out:
  stability.csv   one row per fit: d+ and d- of A*_L, per-layer d-, pair counts, strong pairs in blocks 1 to 6, twin
                  fraction, head pairs, register pairs, zero-shot k-NN of the full conversion
  pairs.csv       one row per pair of fits: operator distance, rho both ways, energy-weighted agreement, unweighted
                  mean cos^2 of the principal angles, per-layer operator distances, twin agreement both ways, its mean
                  and the Jaccard index, head-assignment share, B and C relative distances
  spectra.npz     per fit: eigenvalues of the span matrix S, per-layer spectra, per-layer d- and energies
  summary.md, metrics.json (A02-1 to A02-4 over the ImageNet fits; A02-5 and A02-6 come from the heals), meta.json
  knn/<fit>.json, knn/<fit>.npz   zero-shot k-NN of each fit's full conversion (resumable)
"""
from __future__ import annotations

import sys

import torch


def _early_gpu_frac(argv, default: float = 0.3) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if __name__ == "__main__" and torch.cuda.is_available() and "--dry-run" not in sys.argv[1:]:
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))  # shared GPU: cap before allocating

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import fit_weightspace as FW  # noqa: E402
from mirror import mirror_family as MF  # noqa: E402
from mirror import stability as STB  # noqa: E402
from mirror import variants as V  # noqa: E402
from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.utils import REPO_ROOT, log, set_seed  # noqa: E402

DEFAULT_ROLES = "fit0,fit1,fit2,fit3,coco"
DEFAULT_IMAGENET = "fit0,fit1,fit2,fit3"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg")
    p.add_argument("--fits-dir", default=None, help="structures and sidecars (default results/A02/fits)")
    p.add_argument("--roles", default=DEFAULT_ROLES, help="comma list of roles read from --fits-dir")
    p.add_argument("--imagenet-roles", default=DEFAULT_IMAGENET, help="roles in the group of the A02 gates")
    p.add_argument("--ref", default="fit0", help="reference fit of A02-1")
    p.add_argument("--fit", action="append", default=[],
                   help="name=structure.npz[,sidecar.npz[,group]]; adds or replaces a fit")
    p.add_argument("--allow-missing", action="store_true", help="skip roles whose files are missing")
    p.add_argument("--out", default=None, help="output directory (default results/A02)")
    p.add_argument("--skip-knn", action="store_true")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--eval-batch", type=int, default=None, help="feature batch (default 512)")
    p.add_argument("--stats-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu-frac", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="recompute the k-NN even when results exist")
    p.add_argument("--dry-run", action="store_true", help="tiny random ViT, five synthetic fits, CPU")
    return p.parse_args(argv)


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _f(x, nd=3) -> str:
    try:
        return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def dry_run_fits(fits_dir: Path, roles: list) -> None:
    """Five tiny fits on synthetic images (each role its own images; "coco" from another seed)."""
    from mirror.data import make_loader, synthetic_dataset

    adapter = None
    for i, role in enumerate(roles):
        s_path, c_path = fits_dir / f"tiny_{role}.npz", fits_dir / f"tiny_{role}_sidecar.npz"
        if s_path.exists() and c_path.exists():
            continue
        if adapter is None:
            adapter = FW.dry_run_adapter(seed=0)
        seed = 2000 + i if role != "coco" else 4242
        images, labels = synthetic_dataset(4, 4, 56, seed=seed)
        loader = make_loader(images, labels, np.arange(len(labels)), 8)
        st, sc = STB.fit_with_sidecar(adapter, loader, "cpu", model_name="tiny", role=role, fit_seed=i, top_k=3,
                                      workers=1)
        MF.save_span_structure(s_path, st)
        STB.save_sidecar(c_path, sc)


def collect_fits(a, fits_dir: Path, model_name: str) -> list:
    imagenet = {r.strip() for r in a.imagenet_roles.split(",") if r.strip()}
    fits = {}
    for role in [r.strip() for r in a.roles.split(",") if r.strip()]:
        s_path, c_path = fits_dir / f"{model_name}_{role}.npz", fits_dir / f"{model_name}_{role}_sidecar.npz"
        if not s_path.exists():
            if a.allow_missing:
                log(f"missing {s_path.name}: skipped")
                continue
            raise SystemExit(f"{s_path} not found (run scripts/fp_structure_fit.py --role {role}, or --allow-missing)")
        fits[role] = {"name": role, "group": "imagenet" if role in imagenet else role, "structure": s_path,
                      "sidecar": c_path if c_path.exists() else None}
    for spec in a.fit:
        name, rest = spec.split("=", 1)
        parts = rest.split(",")
        group = parts[2] if len(parts) > 2 else ("imagenet" if name in imagenet else name)
        fits[name] = {"name": name, "group": group, "structure": _abs(parts[0]),
                      "sidecar": _abs(parts[1]) if len(parts) > 1 and parts[1] else None}
    return list(fits.values())


def run_knn(a, ctx, f: dict, st, out: Path) -> dict | None:
    sha = f["sha256"]
    jpath, npath = out / "knn" / f"{f['name']}.json", out / "knn" / f"{f['name']}.npz"
    if jpath.exists() and npath.exists() and not a.force:
        rec = json.loads(jpath.read_text(encoding="utf-8"))
        if rec.get("structure_sha256") == sha:
            log(f"k-NN {f['name']}: reused ({rec['top1']:.2f})")
            return rec
    if ctx.get("adapter") is None:
        return None
    t = time.time()
    amp = "bf16" if torch.device(ctx["device"]).type == "cuda" else None
    r = FW.full_conversion_knn(ctx["adapter"], st, ctx["proto"].bank_loader(), ctx["proto"].query_loader(),
                               ctx["device"], ctx["ncls"], knn_k=a.knn_k, temperature=a.knn_temperature, amp=amp)
    npath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npath, pred=r.pop("pred"), pred_flip=r.pop("pred_flip"), labels=r.pop("labels"))
    rec = {**r, "fit": f["name"], "structure_sha256": sha, "seconds": round(time.time() - t, 1)}
    jpath.write_text(json.dumps(FW._jsonable(rec), indent=1), encoding="utf-8")
    log(f"k-NN {f['name']}: top-1 {rec['top1']:.2f}, flip {rec['flip_consistency']:.2f}")
    return rec


def write_summary(out: Path, model: str, fit_rows: list, pair_rows: list, gates: dict) -> Path:
    def verdict(g):
        return "not decided (missing data)" if g.get("pass") is None else ("PASS" if g["pass"] else "FAIL")

    lines = [f"# A02: stability across fits ({model})", ""]
    for gate, what in (("A02-1", "d- within 15% of fit 0"), ("A02-2", "odd subspaces agree (energy weighted, >= 0.8)"),
                       ("A02-3", "twins agree (>= 0.70 of strong pairs, blocks 1 to 6)"),
                       ("A02-4", "head maps agree (>= 0.90 of assignments)")):
        g = gates.get(gate, {})
        extra = f", smallest {_f(g.get('min'))}" if "min" in g else ""
        lines.append(f"- **{gate}** ({what}): {verdict(g)}{extra}")
    lines += ["- A02-5 and A02-6 are decided from the healed runs.", "", "## Fits", "",
              "| fit | group | d+ | d- | pairs | strong pairs (blocks 1-6) | twin fraction (1-6) | head pairs | "
              "zero-shot top-1 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in fit_rows:
        lines.append(f"| {r['name']} | {r['group']} | {r['d_plus']} | {r['d_minus']} | {r['n_pairs_total']:,} "
                     f"| {r['n_strong_blocks_1_6']} | {_f(r['twin_fraction_blocks_1_6'])} | {r['n_head_pairs']} "
                     f"| {_f(r.get('knn_top1'), 2)} |")
    lines += ["", "## Pairs of fits", "",
              "| a | b | distance | rho a->b | rho b->a | weighted | cos^2 (unweighted) | twins a->b | twins b->a "
              "| twins mean | Jaccard | heads same | B rel. dist. | C rel. dist. |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pair_rows:
        lines.append(f"| {p['a']} | {p['b']} | {_f(p['operator_distance'], 4)} | {_f(p['rho_ab'])} | {_f(p['rho_ba'])} "
                     f"| {_f(p['weighted_agreement'])} | {_f(p['unweighted_cos2'])} | {_f(p['twin_a_to_b'])} "
                     f"| {_f(p['twin_b_to_a'])} | {_f(p['twin_mean'])} | {_f(p['twin_jaccard'])} "
                     f"| {_f(p['head_share_same'])} | {_f(p['B_rel_mean'])} | {_f(p['C_rel_mean'])} |")
    lines += ["", "Definitions: mirror/stability.py (PREREG_FINAL_PHASE.md Section 2.3). Distance is "
              "||A_a - A_b||_F^2 / (2 d) for the full-model operator A*_L; rho is the energy-weighted share of one fit's "
              "odd subspace inside the other's; a strong pair has a symmetrised twin correlation of at least 0.5."]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    prereg = require_prereg()
    if a.dry_run:
        device = torch.device("cpu")
    else:
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not a.skip_knn:
            if device.type != "cuda":
                raise SystemExit("no GPU visible; use --dry-run or --skip-knn")
            torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(0)
    model_name = "tiny" if a.dry_run else a.model
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "A02"
    out.mkdir(parents=True, exist_ok=True)
    fits_dir = Path(a.fits_dir) if a.fits_dir else (out / "fits" if a.dry_run else REPO_ROOT / "results" / "A02" / "fits")
    fits_dir.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        dry_run_fits(fits_dir, [r.strip() for r in a.roles.split(",") if r.strip()])
    fits = collect_fits(a, fits_dir, model_name)
    if len(fits) < 2:
        raise SystemExit("need at least two fits")
    sts, scs = {}, {}
    for f in fits:
        sts[f["name"]] = MF.load_span_structure(f["structure"])
        scs[f["name"]] = STB.load_sidecar(f["sidecar"]) if f["sidecar"] is not None else None
        f["sha256"] = FW.sha256_file(f["structure"])
    L = {st.depth for st in sts.values()}
    if len(L) != 1 or len({st.d for st in sts.values()}) != 1:
        raise SystemExit("the fits are not of the same model")
    L = L.pop()
    twin_blocks = [b for b in STB.TWIN_BLOCKS if b < L]
    ctx = {"device": device}
    if not a.skip_knn:
        if a.dry_run:
            proto = V.TinyProtocol()
            ctx.update(adapter=FW.dry_run_adapter(seed=0), proto=proto, ncls=proto.num_classes)
        else:
            adapter, proto, ncls = V.build_runtime(a.model, False, device, a.cache_dir, a.eval_batch or 512,
                                                   a.stats_batch_size, a.num_workers)
            adapter.model.float().eval()
            ctx.update(adapter=adapter, proto=proto, ncls=ncls)

    fit_rows = []
    for f in fits:
        st, sc = sts[f["name"]], scs[f["name"]]
        dp, dm = STB.signature(st.A_star[L])
        dml = [STB.signature(A)[1] for A in st.A_layers]
        n_strong = (sum(len(STB.strong_pairs(st.perms[b], sc["twin_score"][b])) for b in twin_blocks)
                    if sc is not None else None)
        row = {"name": f["name"], "group": f["group"], "fit_seed": int(st.fit_seed), "structure": str(f["structure"]),
               "structure_sha256": f["sha256"], "sidecar": None if f["sidecar"] is None else str(f["sidecar"]),
               "d_plus": dp, "d_minus": dm, "d_minus_layers": ";".join(map(str, dml)),
               "n_pairs_total": int(sum(st.n_pairs(b) for b in range(L))),
               "n_pairs_blocks_1_6": int(sum(st.n_pairs(b) for b in twin_blocks)),
               "n_strong_blocks_1_6": n_strong,
               "twin_fraction_blocks_1_6": (2.0 * n_strong / (len(twin_blocks) * st.h) if n_strong is not None
                                            else float("nan")),
               "n_head_pairs": int(sum(hm.n_pairs for hm in st.heads)),
               "register_pairs": FW.n_pairs_of(st.reg_perm[L]) if len(st.reg_perm[L]) else 0,
               "A_from_S_matches": None if sc is None else bool(sc["meta"].get("A_from_S_matches"))}
        if not a.skip_knn:
            kn = run_knn(a, ctx, f, st, out)
            row.update({"knn_top1": kn["top1"] if kn else float("nan"),
                        "flip_consistency_bf16": kn["flip_consistency"] if kn else float("nan"),
                        "params_model": kn["params_model"] if kn else None})
        fit_rows.append(row)

    pair_rows, per_block = [], {}
    for x, y in STB.all_pairs([f["name"] for f in fits]):
        c = STB.compare_fits(sts[x], scs[x], sts[y], scs[y], twin_blocks=twin_blocks)
        per_block[f"{x}-{y}"] = c.pop("twin_per_block")
        lay = c.pop("layer_operator_distance")
        gx = next(f["group"] for f in fits if f["name"] == x)
        gy = next(f["group"] for f in fits if f["name"] == y)
        pair_rows.append({"a": x, "b": y, "group_a": gx, "group_b": gy, "both_imagenet": gx == gy == "imagenet",
                          **c, "layer_operator_distance": ";".join(f"{v:.6g}" for v in lay)})
    gates = STB.a02_gates(fit_rows, pair_rows, ref=a.ref, group="imagenet")

    pd.DataFrame(fit_rows).to_csv(out / "stability.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "pairs.csv", index=False)
    names = [f["name"] for f in fits]
    with_sc = [n for n in names if scs[n] is not None]
    np.savez(out / "spectra.npz", names=np.array(with_sc),
             S_eig=np.stack([scs[n]["S_eig"] for n in with_sc]) if with_sc else np.zeros((0,)),
             layer_eig=np.stack([scs[n]["layer_eig"] for n in with_sc]) if with_sc else np.zeros((0,)),
             d_minus_layers=np.stack([scs[n]["d_minus_layers"] for n in with_sc]) if with_sc else np.zeros((0,)),
             energies=np.stack([scs[n]["energies"] for n in with_sc]) if with_sc else np.zeros((0,)))
    summary = write_summary(out, model_name, fit_rows, pair_rows, gates)
    metrics = {"experiment": "A02", "model": model_name, "fits": fit_rows, "pairs": pair_rows, "gates": gates,
               "twin_agreement_per_block": per_block, "twin_blocks": twin_blocks, "strong_threshold": STB.STRONG,
               "dry_run": bool(a.dry_run)}
    (out / "metrics.json").write_text(json.dumps(FW._jsonable(metrics), indent=1, default=str), encoding="utf-8")
    write_final_meta(out, experiment="A02", config=vars(a), seeds={"global": 0},
                     models={model_name: model_name}, device=device, wall_seconds=time.time() - t_start,
                     extra={"prereg_check": {key: prereg.get(key) for key in ("ok", "reason", "allowed_without_check")},
                            "fits": {f["name"]: {"structure": str(f["structure"]), "sha256": f["sha256"],
                                                 "sidecar": None if f["sidecar"] is None else str(f["sidecar"])}
                                     for f in fits}})
    log(f"A02 done: {summary}")
    return {"out": str(out), "fits": fit_rows, "pairs": pair_rows, "gates": gates}


if __name__ == "__main__":
    main()
