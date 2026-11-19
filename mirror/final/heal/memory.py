"""GPU memory per student for lockstep healing: ESTIMATES from a formula, not measurements (FINAL_PHASE_PLAN.md
Section 7, "estimated GPU memory per student for each model"). Phase G measures the real numbers in W0.

Formula (bytes; P stored student parameters, P_t trainable ones, m micro-batch, T tokens, d width, h MLP width,
L blocks, H heads):
  per student, persistent      4 P (fp32 weights) + 4 P_t (gradients) + 8 P_t (AdamW exp_avg and exp_avg_sq)
  active student (one at a time, while its micro-batch forward and backward run)
      activations saved for the backward pass, per block and token, under bf16 autocast with PyTorch SDPA:
          4d   norm1 input (fp32; layer_norm runs in fp32 under CUDA autocast)
          2d   qkv input (the bf16 cast of the norm output)
          6d   q, k, v (the bf16 qkv output, kept by SDPA) + 2d SDPA output + 4H bytes log-sum-exp
          2d   proj input + 2d proj output (kept by the LayerScale multiply)
          4d   norm2 input (fp32) + 2d fc1 input
          2h   fc1 output (kept by GELU) + 2h GELU output (fc2 input)
          2d   fc2 output (kept by the LayerScale multiply)
        = 26 d + 4 h bytes (+ 4H), i.e. 42 d for h = 4 d;
      loss terms: 4 d per token for each of the L block terms and the CLS + patch term (the fp32 difference kept by
          the square), plus 4 d for the final-norm input;
      autocast weight copies: 2 P (bf16 casts of the student's weights, cached during its forward).
  shared by the job
      teacher weights 4 P_teacher (fp32);
      teacher pass kept for all students: (2 m with flip targets) T d 4 bytes for the L block streams and the final
      norm output; one target-mode group's derived targets m T d 4 (L + 1) (symmetrised or centred; raw ones are
      views); the micro-batch images (fp32 x and its mirror) and the uint8 step batch;
      resident data: N S^2 3 bytes (100k heal images at 256 px: 19.66 GB; LR-selection set 0.98 GB).
  A job with n students needs about shared + n x persistent + one active student.
Not included: the CUDA context and allocator slack (about 0.5 to 1 GB, outside torch.cuda.max_memory_allocated),
the teacher's transient block intermediates (freed before the students run), torch.compile workspaces. Low-rank
students keep more activations than the formula (each LowRankLinear also keeps its input and the rank-r
intermediate; SplitQKV keeps three input casts): MUST-09 measured +1.41 GiB for SVD-LLM over Mirror at micro-batch 64.
If the DINOv2 attention runs without SDPA or xFormers (an explicit softmax), add about 6 m H T^2 bytes per block
(verify which path the hub code takes on the VM).

Check against measurements: mirror.heal.heal() with one student (results/MUST-09/summary.csv, peak_train_memory_gb =
torch.cuda.max_memory_allocated / 2^30, DINOv2-B/14 reg, micro-batch 64, data streamed): Mirror-exact 9.223 GiB,
SVD-LLM 10.63 to 10.64 GiB, SliceGPT 9.631 GiB. `estimate_heal_single` (heal()'s layout: one student, derived
targets of B rows) gives about 9.2 GiB for the Mirror student (tests/test_fp_heal_students.py checks it stays within
10% of 9.223).

Architectures (parameter counts from the formulas below; verify against the loaded models in W0):
  dinov2_vitb14_reg  d 768, 12 blocks, 12 heads, h 3072, 224 px, patch 14, 4 registers, 261 tokens, 37 x 37 position
                     table: 86,583,552 parameters (the design doc quotes 86,583,552 for the original).
  deit3_b16          d 768, 12 blocks, 12 heads, h 3072, 224 px, patch 16, 197 tokens, 196 position rows (timm
                     no_embed_class), 1000-class head: 86,585,320 (verify).
  dinov2_vitl14_reg  d 1024, 24 blocks, 16 heads, h 4096, 224 px, patch 14, 4 registers, 261 tokens: 304,372,736
                     (verify).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

GiB = 2 ** 30

__all__ = ["ArchDims", "PRESETS", "MUST09_MEASURED_PEAK_GIB", "estimate_lockstep", "estimate_heal_single",
           "estimate_table", "format_table"]

MUST09_MEASURED_PEAK_GIB = {   # results/MUST-09/summary.csv (seed 0 rows), heal() with one student, micro-batch 64
    "mirror_exact": 9.223287582397461, "svdllm": 10.63, "slicegpt": 9.631, "random_structure": 9.223287582397461}


@dataclass(frozen=True)
class ArchDims:
    name: str
    d: int
    depth: int
    heads: int
    mlp_hidden: int
    img: int
    patch: int
    registers: int
    pos_rows: int          # rows of the stored position table
    cls: bool = True
    mask_token: bool = False
    layerscale: bool = True
    head_classes: int = 0

    @property
    def tokens(self) -> int:
        g = self.img // self.patch
        return int(self.cls) + self.registers + g * g

    def params(self) -> int:
        """Stored parameters of the original model (pre-norm ViT with biases)."""
        d, h = self.d, self.mlp_hidden
        block = (3 * d * d + 3 * d) + (d * d + d) + (h * d + h) + (h * d + d) + 4 * d
        block += 2 * d if self.layerscale else 0
        emb = 3 * self.patch * self.patch * d + d + self.pos_rows * d + (d if self.cls else 0) + self.registers * d
        emb += d if self.mask_token else 0
        head = self.head_classes * d + self.head_classes if self.head_classes else 0
        return int(self.depth * block + emb + 2 * d + head)


PRESETS = {
    "dinov2_vitb14_reg": ArchDims("dinov2_vitb14_reg", 768, 12, 12, 3072, 224, 14, 4, 1 + 37 * 37, mask_token=True),
    "deit3_b16": ArchDims("deit3_b16", 768, 12, 12, 3072, 224, 16, 0, 196, head_classes=1000),
    "dinov2_vitl14_reg": ArchDims("dinov2_vitl14_reg", 1024, 24, 16, 4096, 224, 14, 4, 1 + 37 * 37, mask_token=True),
}


def _activations(a: ArchDims, m: int, P: int, attention: str = "sdpa") -> dict:
    tok = m * a.tokens
    per_block = tok * (26 * a.d + 4 * a.mlp_hidden + 4 * a.heads)
    if attention != "sdpa":
        per_block += 6 * m * a.heads * a.tokens * a.tokens
    blocks = a.depth * per_block
    loss = tok * 4 * a.d * (a.depth + 1) + tok * 4 * a.d
    return {"blocks": blocks, "loss_terms": loss, "autocast_weights": 2 * P}


def estimate_lockstep(arch: str | ArchDims, micro_batch: int, n_students: int = 1, *, student_params: int | None = None,
                      trainable: int | None = None, flip: bool = True, data_images: int = 0, image_size: int = 256,
                      batch_size: int = 256, attention: str = "sdpa") -> dict:
    """Estimated GPU memory (GiB) of a lockstep job: per-student persistent state, the active student's
    activations and the shared part (module docstring)."""
    a = PRESETS[arch] if isinstance(arch, str) else arch
    m = int(micro_batch)
    P_t = int(a.params())
    P = int(student_params or P_t)
    Pt = int(trainable or P)
    persistent = 4 * P + 4 * Pt + 8 * Pt
    act = _activations(a, m, P, attention)
    active = sum(act.values())
    f = 2 if flip else 1
    tok = m * a.tokens
    shared = {"teacher_weights": 4 * P_t, "teacher_streams": f * tok * a.d * 4 * (a.depth + 1),
              "derived_targets": tok * a.d * 4 * (a.depth + 1),
              "batch_images": f * m * 3 * a.img * a.img * 4 + batch_size * image_size * image_size * 3,
              "resident_data": int(data_images) * image_size * image_size * 3}
    total = sum(shared.values()) + int(n_students) * persistent + active
    out = {"model": a.name, "micro_batch": m, "n_students": int(n_students), "tokens": a.tokens,
           "params_teacher": P_t, "params_student": P, "per_student_persistent_gib": persistent / GiB,
           "active_student_gib": active / GiB, "active_parts_gib": {k: v / GiB for k, v in act.items()},
           "shared_gib": sum(shared.values()) / GiB, "shared_parts_gib": {k: v / GiB for k, v in shared.items()},
           "total_gib": total / GiB, "attention": attention, "flip_targets": bool(flip), "estimate": True}
    return out


def estimate_heal_single(arch: str | ArchDims, micro_batch: int, *, student_params: int | None = None,
                         trainable: int | None = None, batch_size: int = 256, image_size: int = 256) -> dict:
    """mirror.heal.heal() with one student (the MUST-09 layout): the derived targets of B rows are kept, not the
    teacher streams of 2B rows; data streamed. For the check against the measured MUST-09 peaks."""
    a = PRESETS[arch] if isinstance(arch, str) else arch
    m = int(micro_batch)
    P_t = int(a.params())
    P = int(student_params or P_t)
    Pt = int(trainable or P)
    act = _activations(a, m, P)
    tok = m * a.tokens
    parts = {"persistent": 4 * P + 12 * Pt, "active": sum(act.values()), "teacher_weights": 4 * P_t,
             "derived_targets": tok * a.d * 4 * (a.depth + 1),
             "batch_images": m * 3 * a.img * a.img * 4 + batch_size * image_size * image_size * 3}
    return {"model": a.name, "micro_batch": m, "parts_gib": {k: v / GiB for k, v in parts.items()},
            "total_gib": sum(parts.values()) / GiB, "estimate": True}


def estimate_table(models=("dinov2_vitb14_reg", "deit3_b16", "dinov2_vitl14_reg"), micro_batches=(64, 128)) -> list:
    """Rows for the report: per-student persistent memory, the active student at each micro-batch, the shared
    part without and with the resident 100k heal set, and the total for 1 and 6 students."""
    rows = []
    for name in models:
        for mb in micro_batches:
            e = estimate_lockstep(name, mb, 1)
            e6 = estimate_lockstep(name, mb, 6, data_images=105_000)
            rows.append({"model": name, "micro_batch": mb, "params": PRESETS[name].params(),
                         "per_student_gib": round(e["per_student_persistent_gib"], 2),
                         "active_gib": round(e["active_student_gib"], 2), "shared_gib": round(e["shared_gib"], 2),
                         "one_student_total_gib": round(e["total_gib"], 2),
                         "six_students_resident_data_gib": round(e6["total_gib"], 2)})
    return rows


def format_table(rows: list) -> str:
    cols = ["model", "micro_batch", "params", "per_student_gib", "active_gib", "shared_gib", "one_student_total_gib",
            "six_students_resident_data_gib"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:,}" if isinstance(r[c], int) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    print("ESTIMATES (formula in mirror/final/heal/memory.py; not measurements)")
    print(format_table(estimate_table()))
    for k in ("dinov2_vitb14_reg",):
        e = estimate_heal_single(k, 64)
        print(f"heal() single student, {k}, micro-batch 64: {e['total_gib']:.2f} GiB estimated; MUST-09 measured "
              f"{MUST09_MEASURED_PEAK_GIB['mirror_exact']:.2f} GiB (Mirror-exact)")
    print(json_dims())


def json_dims() -> str:
    import json

    return json.dumps({k: {**asdict(v), "tokens": v.tokens, "params": v.params()} for k, v in PRESETS.items()},
                      indent=1)


if __name__ == "__main__":
    main()
