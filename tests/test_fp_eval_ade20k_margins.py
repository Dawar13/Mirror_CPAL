"""P11.2: the top-two margin record of the mask-equivariance evaluation in scripts/tier2_ade20k.py (PREREG_NEXT_PHASE.md
Section 1 rounding allowance), CPU only with the tiny random ViT and synthetic segmentation images."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mirror.adapters import TimmTinyAdapter
from mirror.attention import attn_dims
from mirror.m1 import build_vexact, random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    key = f"{name}_fp_eval_margins_test"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


S = load_script("tier2_ade20k")
SIZES = ((56, 84), (42, 60))


def test_relative_top_two_margin_definition():
    L = torch.tensor([[3.0, -1.0, 2.0], [1.0, -1.00001, 2.0]])            # [C = 2, n = 3]
    m = S.rel_top2_margin(L)
    assert m[0] == pytest.approx(2.0 / 3.0) and m[1] == pytest.approx(1e-5, rel=1e-2) and m[2] == 0.0
    assert torch.isinf(S.rel_top2_margin(torch.ones(1, 4))).all()


def test_margin_record_of_crafted_differing_pixels():
    m = {"strict": 0.0, "one_copy": 0.0, "not_tie": 0, "hist": np.zeros(len(S.MARGIN_HIST_EDGES) - 1, np.int64),
         "list": [], "rec": {k: [] for k in ("image", "y", "x", "rel_margin", "rel_margin_mirrored", "class",
                                             "class_mirrored")}, "n_rec": 0}
    lo = torch.zeros(3, 2, 2)                                              # [C, h, w]: classes 0..2
    lm = torch.zeros(3, 2, 2)
    lo[:, 0, 0], lm[:, 0, 0] = torch.tensor([1.0, 1.0 - 5e-6, 0.0]), torch.tensor([1.0 - 5e-6, 1.0, 0.0])  # near-tie
    lo[:, 1, 1], lm[:, 1, 1] = torch.tensor([2.0, 1.0, 0.0]), torch.tensor([1.0, 2.0, 0.0])                # not
    differ = torch.tensor([[True, False], [False, True]])
    S._record_margins(m, 7, differ, lo, lm)
    f = S._margin_fields(m, 2)
    assert f["max_rel_margin_of_differing_pixels"] == pytest.approx(0.5)
    assert f["max_rel_margin_of_differing_pixels_one_copy"] == pytest.approx(0.5)
    assert f["n_differ_not_near_tie"] == 1 and sum(f["rel_margin_histogram"]["counts"]) == 2
    assert sorted(f["rel_margins_of_differing_pixels"])[0] == pytest.approx(5e-6, rel=1e-2)    # float32 rounding
    assert m["rec"]["image"][0].tolist() == [7, 7] and m["rec"]["x"][0].tolist() == [0, 1]
    assert m["rec"]["class"][0].tolist() == [0, 0] and m["rec"]["class_mirrored"][0].tolist() == [1, 1]
    none = S._margin_fields({**m, "strict": 0.0}, 0)
    assert none["max_rel_margin_of_differing_pixels"] is None


def _equivariant_model():
    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=8)
    perms = [random_pairing(h, h // 3, seed=30 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, 1 + b % 2, seed=40 + b) for b in range(L)]
    model, _ = build_vexact(ad, A, list(range(L)), perms, maps, even_head=True)
    return ad, model


def test_run_mask_eq_writes_margins_and_recomputes_results_without_them(tmp_path, monkeypatch):
    ad, model = _equivariant_model()
    imgs, labs = S.synthetic_seg_dataset(len(SIZES), 4, SIZES, seed=1)
    store, lut = S.RaggedStore.from_arrays(imgs, labs), S.label_lut(4)
    head = S.LinearSegHead(ad.embed_dim, 4, bn=True, init_std=0.5, seed=0).eval()
    a = SimpleNamespace(force=False, mask_eq_images=None, even_tokens=False, crop=56, eval_stride=42, eval_batch=4)
    v, factory, vdir = SimpleNamespace(name="vx"), SimpleNamespace(operators=None), tmp_path / "seg"
    vdir.mkdir()
    fields = S.run_mask_eq(a, v, ad, model, factory, head, {"val": store}, lut, 4, "cpu", vdir, {"run": 1})
    me = json.loads((vdir / "vx_maskeq.json").read_text(encoding="utf-8"))
    raw, sym = me["result"]["raw"], me["result"]["sym"]
    assert raw["n_differ"] > 0 and raw["max_rel_margin_of_differing_pixels"] > 0
    assert raw["max_rel_margin_of_differing_pixels"] >= raw["max_rel_margin_of_differing_pixels_one_copy"]
    assert sum(raw["rel_margin_histogram"]["counts"]) == raw["n_differ"] == raw["n_differ_recorded"]
    assert sym["n_differ_not_near_tie"] == 0                    # exact by construction: every difference a near-tie
    assert "margin_definition" in me["result"] and "_margin_records" not in me["result"]
    with np.load(vdir / me["result"]["margin_records_file"]) as z:
        assert len(z["raw_rel_margin"]) == raw["n_differ"] and len(z["sym_y"]) == sym["n_differ"]
    for key in ("mask_eq_raw_max_rel_margin", "mask_eq_sym_max_rel_margin", "mask_eq_raw_n_differ_not_near_tie",
                "mask_eq_sym_n_differ_not_near_tie"):
        assert key in fields and key in S.CSV_COLUMNS
    assert fields["mask_eq_raw"] == raw["share"] and fields["mask_eq_sym_n_differ"] == sym["n_differ"]

    def boom(*args, **kw):
        raise AssertionError("recomputed")

    monkeypatch.setattr(S, "mask_equivariance", boom)
    assert S.run_mask_eq(a, v, ad, model, factory, head, {"val": store}, lut, 4, "cpu", vdir, {"run": 1}) == fields
    old = dict(me)                                                # a result saved before the margins existed
    old["result"] = {k: ({kk: vv for kk, vv in val.items() if "rel_margin" not in kk and "near_tie" not in kk}
                         if k in ("raw", "sym") else val) for k, val in me["result"].items()}
    (vdir / "vx_maskeq.json").write_text(json.dumps(old), encoding="utf-8")
    assert not S.has_margins(old["result"]) and S.has_margins(me["result"])
    with pytest.raises(AssertionError, match="recomputed"):
        S.run_mask_eq(a, v, ad, model, factory, head, {"val": store}, lut, 4, "cpu", vdir, {"run": 1})
