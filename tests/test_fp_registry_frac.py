"""The student registry builds SVD-LLM at the A06 fractions (mirror/final/heal/students.py with frac, through
mirror.final.baselines.sizes.budget_for_fraction): each student is eligible (0 to +1% over its target size) and
smaller fractions give smaller students. Tiny float32 model and structure as tests/test_fp_heal_students.py."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from mirror import mirror_family as MF
from mirror.data import make_loader, synthetic_dataset
from mirror.variants import TinyProtocol

from mirror.final.heal.students import BuildContext, build_student

ROOT = Path(__file__).resolve().parents[1]


def test_svdllm_at_fractions(tmp_path):
    spec = importlib.util.spec_from_file_location("must09_heal_frac_test", ROOT / "scripts" / "must09_heal.py")
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    spec.loader.exec_module(M)
    cfg0 = M.setup(M.parse_args(["--tiny", "--method", "mirror_exact"]))
    teacher, aspec = M.build_teacher(cfg0, torch.device("cpu"))
    images, labels = synthetic_dataset(4, 4, 56, seed=3)
    st = MF.fit_span_structure(teacher, make_loader(images, labels, np.arange(16), 8), "cpu", model_name="tiny",
                               top_k=3, workers=1, pairings=False)
    path = MF.save_span_structure(tmp_path / "tiny_s0.npz", st)
    ctx = BuildContext(adapter=teacher, model_name="tiny", adapter_spec=aspec, device="cpu", structure=str(path),
                       gate_b=None, fit_loader=TinyProtocol().fit_loader)
    sizes = []
    for frac in (0.65, 0.50, 0.40):
        sb = build_student("svdllm", ctx, frac=frac)
        p = sb.params
        n = int(p.get("params_model") or sum(v.numel() for v in sb.model.parameters()))
        sizes.append(n)
        assert sb.spec["family"] == "compressed" and sb.method == "svdllm"
    assert sizes[0] > sizes[1] > sizes[2]
