"""Fit a SpanStructure with the canonical pipeline on any image set and write its A02 sidecar
(mirror/stability.py fit_with_sidecar; FINAL_PHASE_PLAN.md A02; PREREG_FINAL_PHASE.md Section 2.3).

  # a new fit set or another domain (uint8 [N, 224, 224, 3] .npy, no labels needed):
  python scripts/fp_structure_fit.py --model dinov2_vitb14_reg --role fit3 --images <fit3 images .npy> \\
      --fit-seed 3 --out results/A02/fits
  python scripts/fp_structure_fit.py --model dinov2_vitb14_reg --role coco --images <COCO images .npy> \\
      --fit-seed 100 --out results/A02/fits
  # the existing ImageNet fit sets 0, 1 and 2 (rows of cache/train224_uint8.npy listed in cache/train_index.npz under
  # fit<k>_rows), checked against results/MUST-07/structure/<model>_s<k>.npz:
  python scripts/fp_structure_fit.py --model dinov2_vitb14_reg --role fit0 --out results/A02/fits
  python scripts/fp_structure_fit.py --dry-run --role fit0 --out <dir>    (CPU, tiny random ViT, synthetic images)

--fit-seed is the integer label stored in the structure (SpanStructure.fit_seed); for a role fit<k> it defaults to k.
Pass 1 is variants.fit_structure on the same images, exactly what mirror_family.fit_span_structure runs by itself;
--operators uses an existing run_e1.py operator file instead (as MUST-07's --operators-template did for fit sets
0 to 2, which is how the canonical structures were made). --control-pairings also fits the pair_actcorr and pair_wcos
pairings (slow; A02 does not need them).

Outputs in --out:
  <model>_<role>.npz            the SpanStructure (mirror_family.save_span_structure)
  <model>_<role>_sidecar.npz    per-neuron twin scores, span matrix S and its eigenvalues, per-layer spectra,
                                energies and d- (mirror/stability.py)
  <model>_<role>_fit.json       summary, and the reproduction check against --reference (bitwise equality of A_star,
                                pairings and head maps, largest differences, agreement numbers)
  <model>_<role>_meta.json      mirror.final.prereg.write_final_meta
Resume: an existing structure and sidecar are reused (the summary and the reproduction check are rewritten); --force
refits.
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
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import fit_weightspace as FW  # noqa: E402
from mirror import mirror_family as MF  # noqa: E402
from mirror import stability as STB  # noqa: E402
from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.utils import REPO_ROOT, log, set_seed  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--role", required=True, help="name of the fit set, e.g. fit0, fit3, coco")
    p.add_argument("--images", default=None,
                   help="uint8 [N, S, S, 3] .npy (default for role fit<k>: <cache-dir>/train224_uint8.npy)")
    p.add_argument("--index", default=None,
                   help="npz with row indices into --images (default for role fit<k>: <cache-dir>/train_index.npz)")
    p.add_argument("--rows-key", default=None, help="key in --index (default for role fit<k>: fit<k>_rows)")
    p.add_argument("--fit-seed", type=int, default=None, help="integer label of the fit (default k for role fit<k>)")
    p.add_argument("--out", default=None, help="output directory (default results/A02/fits)")
    p.add_argument("--reference", default=None,
                   help="structure to compare with (default for fit0..fit2: results/MUST-07/structure/<model>_s<k>.npz; "
                        "'none' to skip)")
    p.add_argument("--operators", default=None, help="run_e1.py operator file to use as pass 1")
    p.add_argument("--control-pairings", action="store_true", help="also fit pair_actcorr and pair_wcos (slow)")
    p.add_argument("--top-k", type=int, default=None, help="matching candidates per neuron (default 10; dry run 3)")
    p.add_argument("--workers", type=int, default=None, help="processes for the matchings (default 8; dry run 1)")
    p.add_argument("--register-margin", type=float, default=0.0)
    p.add_argument("--stats-batch-size", type=int, default=16, help="paired passes (images with their mirrors)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--gpu-frac", type=float, default=0.3, help="per-process GPU memory cap (set at start)")
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="refit even when the outputs exist")
    p.add_argument("--dry-run", action="store_true", help="tiny random ViT, synthetic images, CPU")
    return p.parse_args(argv)


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_inputs(a, model_name: str, out: Path) -> dict:
    """Images, rows, fit seed and reference for the role."""
    m = re.fullmatch(r"fit(\d+)", a.role)
    fit_seed = a.fit_seed if a.fit_seed is not None else (int(m.group(1)) if m else None)
    if fit_seed is None:
        raise SystemExit(f"--fit-seed is required for role {a.role!r}")
    if a.dry_run and a.images is None:
        from mirror.data import synthetic_dataset

        path = out / f"_dry_images_{a.role}.npy"
        if not path.exists():
            images, _ = synthetic_dataset(4, 4, 56, seed=1000 + int(fit_seed))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, images)
        images_path, index, key = path, None, None
    elif a.images is None:
        if not m:
            raise SystemExit(f"--images is required for role {a.role!r}")
        cache = Path(a.cache_dir)
        images_path, index, key = cache / "train224_uint8.npy", cache / "train_index.npz", f"{a.role}_rows"
    else:
        images_path, index, key = _abs(a.images), (_abs(a.index) if a.index else None), a.rows_key
    if a.reference and a.reference.lower() == "none":
        ref = None
    elif a.reference:
        ref = _abs(a.reference)
    elif m and not a.dry_run and int(m.group(1)) in (0, 1, 2):
        ref = REPO_ROOT / "results" / "MUST-07" / "structure" / f"{model_name}_s{int(m.group(1))}.npz"
        ref = ref if ref.exists() else None
    else:
        ref = None
    return {"images": Path(images_path), "index": None if index is None else Path(index), "rows_key": key,
            "fit_seed": int(fit_seed), "reference": ref}


def summarise(st, sc) -> dict:
    L = st.depth
    dp, dm = STB.signature(st.A_star[L])
    twin_blocks = [b for b in STB.TWIN_BLOCKS if b < L]
    n_strong = sum(len(STB.strong_pairs(st.perms[b], sc["twin_score"][b])) for b in twin_blocks)
    return {"model": st.model, "role": sc["role"], "fit_seed": int(st.fit_seed), "depth": L, "d": st.d, "h": st.h,
            "d_plus": dp, "d_minus": dm, "d_minus_span": [int(x) for x in sc["d_minus_span"]],
            "d_minus_layers": [int(x) for x in sc["d_minus_layers"]], "n_pairs": [st.n_pairs(b) for b in range(L)],
            "n_strong_pairs_blocks_1_6": int(n_strong),
            "twin_fraction_blocks_1_6": 2.0 * n_strong / (len(twin_blocks) * st.h) if twin_blocks else float("nan"),
            "n_head_pairs": [hm.n_pairs for hm in st.heads],
            "register_pairs": FW.n_pairs_of(st.reg_perm[L]) if len(st.reg_perm[L]) else 0,
            "A_from_S_matches": bool(sc["meta"].get("A_from_S_matches")),
            "A_from_S_max_abs_diff": sc["meta"].get("A_from_S_max_abs_diff"), "n_tokens": sc["meta"].get("n_tokens"),
            "pass1": st.meta.get("pass1"), "fit_seconds_total": st.meta.get("fit_seconds_total")}


def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    prereg = require_prereg()
    if a.dry_run:
        device = torch.device("cpu")
    else:
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise SystemExit("no GPU visible; use --dry-run for CPU runs")
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    torch.backends.cuda.matmul.allow_tf32 = False          # as the canonical fits (X2: tf32 false)
    torch.backends.cudnn.allow_tf32 = False
    set_seed(0)
    top_k = a.top_k if a.top_k is not None else (3 if a.dry_run else 10)
    workers = a.workers if a.workers is not None else (1 if a.dry_run else 8)
    model_name = "tiny" if a.dry_run else a.model
    out = Path(a.out) if a.out else REPO_ROOT / "results" / "A02" / "fits"
    out.mkdir(parents=True, exist_ok=True)
    inp = resolve_inputs(a, model_name, out)
    s_path = out / f"{model_name}_{a.role}.npz"
    c_path = out / f"{model_name}_{a.role}_sidecar.npz"
    images, rows = STB.load_image_rows(inp["images"], inp["index"], inp["rows_key"])
    if s_path.exists() and c_path.exists() and not a.force:
        log(f"reusing {s_path.name} and {c_path.name}")
        st, sc = MF.load_span_structure(s_path), STB.load_sidecar(c_path)
    else:
        if a.dry_run:
            adapter = FW.dry_run_adapter(seed=0)
        else:
            from mirror.adapters_timm import build_model_adapter

            adapter = build_model_adapter(a.model, device=device)
        adapter.model.float().eval()
        if images.shape[1] != adapter.img_size or images.shape[2] != adapter.img_size:
            raise SystemExit(f"images are {images.shape[1]} x {images.shape[2]}, the model expects {adapter.img_size}")
        loader = STB.image_loader(images, rows, a.stats_batch_size, 0 if a.dry_run else a.num_workers,
                                  device.type == "cuda")
        log(f"fitting {model_name} on role {a.role} ({len(rows)} images from {inp['images']}), fit seed "
            f"{inp['fit_seed']}, top_k {top_k}, workers {workers}")
        st, sc = STB.fit_with_sidecar(adapter, loader, device, model_name=model_name, role=a.role,
                                      fit_seed=inp["fit_seed"], top_k=top_k, workers=workers,
                                      register_margin=a.register_margin,
                                      operators=None if a.operators is None else _abs(a.operators),
                                      control_pairings=a.control_pairings)
        st.meta["images"] = {"path": str(inp["images"]), "index": None if inp["index"] is None else str(inp["index"]),
                             "rows_key": inp["rows_key"], "n_images": int(len(rows))}
        sc["meta"]["images"] = st.meta["images"]
        MF.save_span_structure(s_path, st)
        STB.save_sidecar(c_path, sc)
        st, sc = MF.load_span_structure(s_path), STB.load_sidecar(c_path)
        log(f"saved {s_path.name}: d- {STB.signature(st.A_star[st.depth])[1]}, pairs {st.meta['n_pairs']}")
    summary = summarise(st, sc)
    rep = None
    if inp["reference"] is not None:
        ref = MF.load_span_structure(inp["reference"])
        rep = STB.compare_structures(st, ref)
        tw = STB.twin_agreement(st.perms, sc["twin_score"], ref.perms, None,
                                [b for b in STB.TWIN_BLOCKS if b < st.depth])
        rep.update({"reference": str(inp["reference"]), "reference_sha256": FW.sha256_file(inp["reference"]),
                    "twin_strong_of_refit_matched_in_reference": tw["a_to_b"]})
        log(f"reference {inp['reference'].name}: reproduces exactly: {rep['reproduces_exactly']} (A_star "
            f"{rep['A_star_equal']}, pairings {rep['perms_equal']}, heads {rep['heads_equal']}; largest A_star "
            f"difference {rep['A_star_max_abs_diff']:.2e}, {rep['perm_entries_differing']} pairing entries differ)")
    result = {**summary, "structure": str(s_path), "structure_sha256": FW.sha256_file(s_path),
              "sidecar": str(c_path), "images": str(inp["images"]), "rows_key": inp["rows_key"],
              "n_images": int(len(rows)), "reproduce": rep, "dry_run": bool(a.dry_run)}
    (out / f"{model_name}_{a.role}_fit.json").write_text(json.dumps(FW._jsonable(result), indent=1, default=str),
                                                          encoding="utf-8")
    write_final_meta(out, name=f"{model_name}_{a.role}_meta.json", experiment="A02-structure-fit", config=vars(a),
                     seeds={"fit_seed": inp["fit_seed"], "global": 0}, models={model_name: model_name},
                     device=device, wall_seconds=time.time() - t_start,
                     extra={"prereg_check": {key: prereg.get(key) for key in ("ok", "reason", "allowed_without_check")},
                            "structure": str(s_path), "reproduce": rep})
    return result


if __name__ == "__main__":
    main()
