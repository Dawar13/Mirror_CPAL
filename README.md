# Mirror Surgery: E1 go/kill experiment

Does pretrained DINOv2 ViT-B contain one residual mirror operator and mirror-twin MLP neurons, so
that projecting early-block MLPs onto mirror-equivariant weights costs little accuracy and clearly
less than an equal-size SVD cut? See `PROJECT_CONTEXT.md` and `E1_INSTRUCTIONS.md`.

## Layout

```
mirror/adapters.py   one interface over DINOv2 (torch.hub) and a tiny random timm ViT (tests)
mirror/data.py       ImageNet-val cache (uint8 memmap), class-balanced splits, synthetic data
mirror/stats.py      streaming float64 statistics of each batch and its mirror (hooks, un-flip)
mirror/fit.py        closed-form involution fit, random-involution nulls, shared operator
mirror/matching.py   mirror-twin neuron matching (networkx max-weight matching, parallel)
mirror/surgery.py    folding, projection, breaking share, surgery, rank-matched SVD baseline
mirror/knn.py        weighted k-NN (k 20, T 0.07) and flip consistency
mirror/report.py     decision table, plots, E1_REPORT.md
scripts/build_cache.py   B1: download validation shards and build the cache
scripts/sanity_b2.py     B2: folding and identity-surgery checks on the real model
scripts/run_e1.py        B3: the experiment (add --tiny for the CPU pipeline test)
configs/e1.yaml          all settings, decision thresholds and the --tiny overrides
tests/                   pytest, CPU only (tiny random ViT, synthetic data)
```

## Phase B on the VM

Accept the ImageNet-1k terms first at https://huggingface.co/datasets/ILSVRC/imagenet-1k
(gated; approval can take a while).

### B0. Setup

```bash
git clone <private repo url> Mirror_CPAL && cd Mirror_CPAL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/path/on/vm/big_disk/hf        # every download goes under here (hub weights too)
python -c "from huggingface_hub import login; login()"   # paste a read token
nvidia-smi                                     # check what the CVPR job is using
pytest -q                                      # 25 tests, about 1 minute; all must pass
```

Every GPU script caps itself with `torch.cuda.set_per_process_memory_fraction(0.2)`.

### B1. Data cache (no GPU needed, about 7.5 GB in `cache/` plus 6.7 GB under `$HF_HOME`)

```bash
python scripts/build_cache.py --config configs/e1.yaml
```

Downloads only `data/validation-*.parquet` (14 shards) from `ILSVRC/imagenet-1k`, then writes
`cache/val224_uint8.npy` (resize 256 bicubic, centre crop 224), `cache/val_labels.npy`,
`cache/splits.npz` (fit 2,000 / held-out 1,000 / bank 39,000 / queries 8,000) and
`cache/meta.json`. The DINOv2 weights are fetched by torch.hub on first use, into
`$HF_HOME/torch_hub`.

### B2. Sanity on the real model

```bash
python scripts/sanity_b2.py --config configs/e1.yaml
```

Must print `B2 SANITY PASS`: block-0 folding max abs difference below 1e-4 in fp32, and
identity surgery (A = I, P = I on all blocks) gives identical k-NN top-1. Writes
`results/sanity_b2.json`.

### B3. Run E1

```bash
python scripts/run_e1.py --config configs/e1.yaml 2>&1 | tee results/e1_log.txt
```

Outputs: `results/e1_layers.csv`, `results/e1_mlp.csv`, `results/e1_surgery.csv`,
`results/e1_meta.json` (config, seed, git hash, timings), `results/e1_operators.npz`
(A per layer, shared A, neuron permutations), `plots/equivariance_vs_depth.png`,
`plots/breaking_share_vs_depth.png` and `results/E1_REPORT.md`, which applies the decision
table and prints GO, PIVOT, KILL or AMBER. Commit the report and plots.

## Notes and choices

- Layer l is the output of block l (layer 0 = tokens entering block 1). "Blocks 1 to 6" in the
  decision table means layers 1 to 6 for the residual checks and the first six MLPs.
- The decision table does not say how to combine blocks 1 to 6. The report uses the mean of the
  per-layer relative errors (rows 1 and 2) and the larger of the mean fc1 and fc2 breaking
  shares (row 4), and says so in the report. Thresholds are exactly those in `E1_INSTRUCTIONS.md`.
- The table can leave a case undecided (for example row 1 between 3x and 5x with no KILL cell).
  The report then prints AMBER and lists the open rows instead of forcing GO, PIVOT or KILL.
- Residual tokens are channel-centred before the fit; the resulting all-ones null direction
  is assigned +1, so the operator stays an exact involution and commutes with LayerNorm.
- Besides the specified shared operator (plain sum of C over layers 1 to 12), the report also
  shows the error of a scale-normalised fit as a diagnostic; decisions use the specified one.
- Neuron matching runs networkx's exact blossom algorithm per block in parallel processes.
  Measured on a laptop CPU at h = 3072: about 45 s with clear structure, about 5 min for pure noise.
- `--tiny` writes to `tiny_out/` (git-ignored) so it never overwrites real results.

## Laptop (Phase A) environment

This laptop had too little free disk for a fresh 2 GB CPU PyTorch install, so the venv reuses the
system packages and adds only timm:

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install --no-cache-dir timm
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/run_e1.py --tiny
```
