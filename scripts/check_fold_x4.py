import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.1)  # shared GPU: cap before anything is allocated

__doc__ = """X4 check: full LayerNorm folding (mirror/fold.py) on a DINOv2 hub model is exact in fp32.

Loads the hub model (downloads under $HF_HOME/torch_hub), folds a deep copy with
mirror.fold.fold_layernorm_full, runs both on N validation images (seeded choice from the cache,
ImageNet-normalised) and compares the final normalised tokens (CLS, registers and patches; the
returned final affine is applied to the folded model's output). TF32 is switched off.
PASS if the max abs difference over all tokens is under --tol (1e-4).

Also reported: the hub forward (CLS feature) difference, the folded stream against the centred
original stream, and the largest |channel mean| / rms of the folded residual stream per layer.

    python scripts/check_fold_x4.py --name dinov2_vitb14_reg [--cache-dir cache] [--n-images 256]

Writes results/x4_fold_<name>.json (or --out). Exit code 0 on pass.
"""
import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.adapters import Dinov2Adapter  # noqa: E402
from mirror.data import load_val_cache, normalize  # noqa: E402
from mirror.fold import apply_final, fold_layernorm_full  # noqa: E402
from mirror.utils import REPO_ROOT, env_info, git_info, log, set_seed  # noqa: E402


class _StreamMeans:
    """Max over tokens of |mean over channels| / rms at each block input and the last block output."""

    def __init__(self, blocks):
        self.vals = [0.0] * (len(blocks) + 1)
        self.handles = []
        for i, blk in enumerate(blocks):
            self.handles.append(blk.register_forward_pre_hook(self._pre(i)))
        self.handles.append(blocks[-1].register_forward_hook(self._post(len(blocks))))

    @staticmethod
    def _rel(x):
        x = x.double()
        return float((x.mean(-1).abs() / x.pow(2).mean(-1).sqrt().clamp_min(1e-30)).max())

    def _pre(self, i):
        def hook(mod, args):
            self.vals[i] = max(self.vals[i], self._rel(args[0]))
        return hook

    def _post(self, i):
        def hook(mod, args, out):
            self.vals[i] = max(self.vals[i], self._rel(out))
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="dinov2_vitb14_reg", help="torch.hub entry, e.g. dinov2_vitb14_reg")
    p.add_argument("--hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--n-images", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                   help="float64 separates rounding from a real folding error")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    t0 = time.time()

    set_seed(args.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    images, labels, splits, meta = load_val_cache(args.cache_dir, args.crop)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(images), size=args.n_images, replace=False)).astype(np.int64)
    log(f"{len(idx)} validation images (seed {args.seed}) from {args.cache_dir}")

    dt = getattr(torch, args.dtype)
    adapter = Dinov2Adapter(hub_repo=args.hub_repo, name=args.name, img_size=args.crop).to(device=device, dtype=dt)
    adapter.model.float().eval()
    folded = copy.deepcopy(adapter.model)
    info = fold_layernorm_full(folded, final_norm_affine=True)
    folded.eval()
    n_ln_left = sum(isinstance(m, torch.nn.LayerNorm) for m in folded.modules())
    log(f"folded {args.name}: {info['n_blocks']} blocks, {info['n_rmsnorm']} RMSNorm, "
        f"{n_ln_left} LayerNorm left, eps {info['eps']}")

    R = adapter.n_prefix_tokens - 1  # register tokens after CLS
    groups = {"cls": slice(0, 1), "registers": slice(1, 1 + R), "patches": slice(1 + R, None)}
    diff = {k: 0.0 for k in groups}
    ref_max = 0.0
    hub_cls_diff = 0.0
    stream_vs_centred = 0.0
    stream_ref_max = 0.0
    probe = _StreamMeans(list(folded.blocks))
    try:
        for s in range(0, len(idx), args.batch_size):
            b = idx[s:s + args.batch_size]
            x = torch.from_numpy(np.ascontiguousarray(images[b])).permute(0, 3, 1, 2).to(device)
            x = normalize(x, dt)
            pre_ref = adapter.forward_tokens(x)
            pre_new = adapter.forward_tokens(x, folded)
            ref = adapter.model.norm(pre_ref)
            new = apply_final(info, folded.norm(pre_new))
            d = (new - ref).abs()
            for k, sl in groups.items():
                if d[:, sl].numel():
                    diff[k] = max(diff[k], float(d[:, sl].max()))
            ref_max = max(ref_max, float(ref.abs().max()))
            centred = pre_ref - pre_ref.mean(-1, keepdim=True)
            stream_vs_centred = max(stream_vs_centred, float((pre_new - centred).abs().max()))
            stream_ref_max = max(stream_ref_max, float(pre_ref.abs().max()))
            hub_cls_diff = max(hub_cls_diff, float((apply_final(info, folded(x)) - adapter.model(x)).abs().max()))
    finally:
        probe.close()

    max_diff = max(diff.values())
    ok = max_diff < args.tol
    log(f"max abs diff of final normalised tokens: {max_diff:.3e} (cls {diff['cls']:.3e}, registers "
        f"{diff['registers']:.3e}, patches {diff['patches']:.3e}); max |ref| {ref_max:.3g}")
    log(f"hub forward CLS diff {hub_cls_diff:.3e}; folded stream vs centred original "
        f"{stream_vs_centred:.3e} (max |stream| {stream_ref_max:.3g}); "
        f"max relative channel mean of folded stream {max(probe.vals):.3e}")
    log(f"X4 FOLD CHECK {'PASS' if ok else 'FAIL'} (tol {args.tol:g}, {args.dtype})")

    out = {
        "experiment": "X4 full LayerNorm folding check",
        "model": {"hub_repo": args.hub_repo, "name": args.name, "n_prefix_tokens": adapter.n_prefix_tokens,
                  "depth": adapter.depth, "embed_dim": adapter.embed_dim},
        "config": {k: v for k, v in vars(args).items()},
        "seed": args.seed,
        "git": git_info(),
        "env": env_info(device),
        "tf32": False,
        "dtype": args.dtype,
        "image_indices": idx.tolist(),
        "cache_meta": {k: meta.get(k) for k in ("n_images", "n_classes", "resize", "crop", "interpolation")},
        "fold": {"family": info["family"], "n_blocks": info["n_blocks"], "n_rmsnorm": info["n_rmsnorm"],
                 "layernorm_left": n_ln_left, "eps": info["eps"], "centred": info["centred"]},
        "max_abs_diff": max_diff,
        "max_abs_diff_by_group": diff,
        "max_abs_ref": ref_max,
        "max_rel_diff": max_diff / ref_max if ref_max > 0 else None,
        "hub_forward_cls_max_abs_diff": hub_cls_diff,
        "stream_vs_centred_original_max_abs_diff": stream_vs_centred,
        "stream_max_abs": stream_ref_max,
        "stream_rel_channel_mean_per_layer": probe.vals,
        "tol": args.tol,
        "pass": bool(ok),
        "seconds": round(time.time() - t0, 1),
    }
    path = Path(args.out) if args.out else REPO_ROOT / "results" / f"x4_fold_{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log(f"wrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
