"""Shared builders for the tests/test_fp_heal*.py files (no tests here). Student construction follows
tests/test_must09.py: hand-built compressed models in each kind's form and the mirror family through
mirror.heal_vexact.build_vexact_student."""
import copy

import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer

from mirror.adapters import ViTAdapter, _randomise_affine
from mirror.attention import attn_dims, fold_attn, write_absorbed_attn
from mirror.heal import HealConfig, HealData, heal
from mirror.heal_structured import distill_teacher, make_trainable
from mirror.heal_vexact import build_vexact_student
from mirror.m1 import apply_attention_svd, random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing
from mirror.surgery import apply_svd_baseline, fold_mlp, write_absorbed_mlp

from mirror.final.heal.multistudent import LockstepStudent
from mirror.final.heal.targets import TargetMode


def span_structure(adapter, seed=0):
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=seed + 3)
    perms = [random_pairing(h, h // 3, seed=seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2, 0, 1)[b % 4], seed=seed + 20 + b) for b in range(L)]
    return A, perms, maps


def heal_data(n_train=16, n_held=8, seed=0):
    rng = np.random.default_rng(seed)
    imgs = rng.integers(0, 256, size=(n_train + n_held, 64, 64, 3), dtype=np.uint8)
    rows = rng.permutation(n_train + n_held)
    return HealData(imgs, train_rows=rows[:n_train], heldout_rows=rows[n_train:])


def cfg(blocks, **kw):
    c = dict(blocks=list(blocks), seed=0, lr=1e-2, max_steps=3, batch_size=8, micro_batch=4, crop_size=56,
             warmup_steps=1, calib_images=8, log_every=0, prefetch=0, train_scope="all")
    c.update(kw)
    return HealConfig(**c)


def compressed(kind, adapter, seed=0):
    """(compressed model, info) in the compressor contract's form (tests/test_must09.py)."""
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    H, dh = attn_dims(adapter.model.blocks[0])
    D = H * dh
    blocks = list(range(L))
    rng = np.random.default_rng(seed)
    if kind == "mirror":
        A, perms, maps = span_structure(adapter, seed)
        m, _, binfo = build_vexact_student(adapter, A, L, perms, maps)
        return m, {"kind": "mirror", "method": "mirror_exact", "blocks": blocks, "components": ["mlp", "attn"],
                   "A_star": A, "perms": perms, "maps_after": binfo["maps_after"], "reg_perm": np.arange(0)}
    m = copy.deepcopy(adapter.model)
    entries, tensors = [], {}
    if kind == "lowrank":
        ranks = [(10 + b, 12) for b in blocks]
        att = [(8, 9, 6, 7 + b) for b in blocks]
        apply_svd_baseline(m, blocks, ranks)
        apply_attention_svd(m, blocks, att)
        for b in blocks:
            entries += [{"block": b, "name": n, "kind": "lowrank", "rank": r} for n, r in zip(("fc1", "fc2"), ranks[b])]
            entries += [{"block": b, "name": n, "kind": "lowrank", "rank": r}
                        for n, r in zip(("wq", "wk", "wv", "wo"), att[b])]
    elif kind == "slice":
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        tensors["Q"] = Q
        n = 40
        P = torch.as_tensor(Q[:, :n] @ Q[:, :n].T)
        for b in blocks:
            blk = m.blocks[b]
            f = fold_mlp(blk)
            write_absorbed_mlp(blk, f["W1"] @ P, f["b1"], P @ f["W2"], P @ f["b2"])
            fa = fold_attn(blk)
            for key in ("Wq", "Wk", "Wv"):
                fa[key] = fa[key] @ P
            fa["Wo"], fa["bo"] = P @ fa["Wo"], P @ fa["bo"]
            write_absorbed_attn(blk, **fa)
            entries += [{"block": b, "name": "fc1", "kind": "slice", "d_in": n, "d_out": h, "basis_key": "Q"},
                        {"block": b, "name": "fc2", "kind": "slice", "d_in": h, "d_out": n, "basis_key": "Q"}]
            entries += [{"block": b, "name": nm, "kind": "slice", "d_in": n, "d_out": D, "basis_key": "Q"}
                        for nm in ("wq", "wk", "wv")]
            entries.append({"block": b, "name": "wo", "kind": "slice", "d_in": D, "d_out": n, "basis_key": "Q"})
    else:
        raise ValueError(kind)
    return m, {"kind": "compressed", "method": kind, "blocks": blocks, "per_matrix": entries, "tensors": tensors}


def mode_for(spec, info, adapter, targets="sym"):
    """The TargetMode that reproduces mirror.heal_structured.distill_teacher (MUST-09)."""
    sym = targets == "sym"
    if spec["family"] == "mirror":
        return TargetMode(final="symmetrised" if sym else "raw", blocks="symmetrised" if sym else "centred",
                          A=info["A_star"], k=adapter.depth)
    blocks = "centred" if spec["base"].get("fold") else "raw"
    return TargetMode(final="symmetrised" if sym else "raw", blocks=blocks)


def must09_teacher(adapter, spec, info, targets="sym"):
    A = info["A_star"] if spec["family"] == "mirror" else span_structure(adapter)[0]
    return distill_teacher(adapter, spec, targets, A, None)


class Student:
    """A compressed student's factory (fresh trainable copies, as must09_heal.make_student)."""

    def __init__(self, kind, adapter):
        self.kind, self.adapter = kind, adapter
        self.comp, self.info = compressed(kind, adapter)
        _, _, self.spec = self.fresh()

    def fresh(self):
        return make_trainable(copy.deepcopy(self.comp), self.info, adapter=self.adapter)

    def lockstep(self, name, lr, **kw):
        m, cons, _ = self.fresh()
        return LockstepStudent(name=name, model=m, constraints=cons, mode=mode_for(self.spec, self.info, self.adapter),
                               lr=lr, **kw)

    def single(self, c, data, out=None, **kw):
        m, cons, _ = self.fresh()
        res = heal(must09_teacher(self.adapter, self.spec, self.info), m, data, c, cons, out, **kw)
        return m, res


def same_state(a, b):
    """Bit for bit (NaN equal to NaN, for diverged students)."""
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(
        sa[k].dtype == sb[k].dtype and sa[k].shape == sb[k].shape
        and (torch.equal(sa[k], sb[k]) if not sa[k].is_floating_point()
             else bool(torch.all((sa[k] == sb[k]) | (torch.isnan(sa[k]) & torch.isnan(sb[k])))))
        for k in sa)


def tiny_adapter(dtype=torch.float64):
    from mirror.adapters import TimmTinyAdapter

    return TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=dtype)


class TinyRegAdapter(ViTAdapter):
    """A tiny timm ViT with 2 register tokens (DINOv2-like prefix: CLS, registers, patches)."""

    def __init__(self, dtype=torch.float64):
        torch.manual_seed(1)
        m = VisionTransformer(img_size=56, patch_size=14, embed_dim=32, depth=3, num_heads=2, reg_tokens=2,
                              init_values=1e-5)
        _randomise_affine(m, 1)
        super().__init__(m.to(dtype), img_size=56, patch_size=14, n_prefix_tokens=int(m.num_prefix_tokens))

    def layer0_tokens(self, x, model=None):
        m = self._m(model)
        return m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))
