# E1 INSTRUCTIONS: Mirror Surgery go/kill experiment

This replaces the earlier E1_SPEC.md.

**Question:** does pretrained DINOv2 ViT-B contain one mirror operator, mirror-twin MLP neurons,
and enough of this structure that deleting the symmetry-breaking MLP weights in early blocks costs
little accuracy and clearly less than an equal-size SVD cut?

Work happens in two phases.
- **Phase A (now, laptop, no GPU, no downloads):** write all code and tests; run tests on CPU with a tiny random ViT and synthetic data.
- **Phase B (VM with H100):** set up, download, run sanity checks, run E1, write the report.

---

## Phase A: code everything offline

### A0. Environment (laptop, small)
- If local disk allows (about 2 GB): create a venv and install CPU-only PyTorch
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), plus
  `timm scipy networkx numpy pandas matplotlib pyyaml pytest tqdm`.
- If disk is too tight: write the code and tests without running them; Phase B starts by running the tests.
- Do not download DINOv2, ImageNet or any weights on the laptop.

### A1. Model adapter (`mirror/adapters.py`)
One interface used by all code, with two implementations:
- `Dinov2Adapter`: wraps `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`.
  Layer-0 tokens via `model.prepare_tokens_with_masks(x)`; blocks `model.blocks`; final `model.norm`;
  CLS feature = normalised CLS token. Interpolated 16 x 16 position embedding via `model.interpolate_pos_encoding`.
- `TimmTinyAdapter` (tests only): `timm.models.vision_transformer.VisionTransformer(img_size=56,
  patch_size=14, embed_dim=64, depth=4, num_heads=4, init_values=1e-5)`, random init. Same module
  names as DINOv2 (`blocks[i].norm1, attn.qkv, attn.proj, ls1.gamma, norm2, mlp.fc1, mlp.fc2, ls2.gamma`).
- Interface: `layer0_tokens(x)`, `blocks`, `grid_hw`, `n_prefix_tokens` (CLS and registers),
  `cls_feature(x, model=None)`, `pos_embed_grid()`.

### A2. Data (`mirror/data.py`)
- ImageNet-1k validation via Hugging Face `datasets` (`ILSVRC/imagenet-1k`, split `validation`), used on the VM only.
- Deterministic, class-balanced split by the order of appearance per class: first 2 -> fit (2,000),
  next 1 -> held-out (1,000), next 39 -> k-NN memory bank (39,000), next 8 -> k-NN queries (8,000).
- Preprocess once: resize 256, centre crop 224, store as a uint8 memmap `cache/val224_uint8.npy`
  plus labels and split indices (about 7.5 GB on the VM disk). Normalise on the fly.
- Mirror = `torch.flip(img, dims=[-1])` after cropping.
- A synthetic dataset class for tests (random images, random labels).

### A3. Statistics collection (`mirror/stats.py`)
Streaming, float64 accumulators on the device. Never store raw activations.
- Hooks capture: layer-0 tokens, each block's output (residual stream), each block's `mlp.fc1` output (pre-activation).
- `unflip_tokens(t)`: reshape patch tokens to [H, W, d], reverse the W axis, reshape back; prefix tokens unchanged.
- Centre every residual token vector (subtract its channel mean) before accumulating.
- Per layer: cross-moment `C_l = sum X^T X~` (d x d). For held-out evaluation: a second pass after A is known, accumulating ||X~ - X A||^2, ||X~ - X||^2, ||X~||^2 and the null errors.
- Per block, MLP: sums, squared sums and cross-moment between mirrored-un-flipped pre-activations and original pre-activations (h x h), to form correlations R[i, j] = corr(z~_i, z_j).

### A4. Fitting (`mirror/fit.py`)
- `fit_involution(C)`: S = (C + C^T)/2; eigh; A = V diag(sign(lam)) V^T; return A, d+, d-.
- `random_involution(d, d_plus, seed)`: random orthogonal V from QR of a Gaussian; same signs.
- Per-layer A_l, shared A from summed C over layers 1..12, agreement tr(A_l A_shared)/d.

### A5. Neuron matching (`mirror/matching.py`)
- Input: symmetric score S = (R + R^T)/2 (h x h).
- Best involution pi maximising sum_i S[i, pi(i)] = sum_i S_ii + sum over pairs (2 S_ij - S_ii - S_jj).
  Solve max-weight matching with edge weights w_ij = 2 S_ij - S_ii - S_jj, keeping only positive
  edges among each neuron's top-10 candidates; unmatched neurons are self-twins.
  Use `networkx.max_weight_matching`.
- Return permutation P (as an index array), pair list, self-twin list, per-neuron match score S[i, pi(i)].

### A6. Breaking share and surgery (`mirror/surgery.py`)
- Folding per block: W1' = W1 diag(gamma2), b1' = b1 + W1 beta2 (norm2 affine into fc1);
  W2' = diag(ls2) W2, b2' = ls2 * b2 (LayerScale into fc2). Absorbed form: set norm2 affine to
  identity and ls2 to ones; outputs must be unchanged.
- Projection: W1_eq = (W1' + P W1' A)/2, W2_eq = (W2' + A W2' P)/2, b1_eq = (b1' + P b1')/2, b2_eq = (b2' + A b2')/2.
- Breaking share s = ||W - W_eq||^2 / ||W||^2 for W1' and W2'. Null: random involution plus random pairing.
- Equivariant parameter count per matrix: (h d + tr(P) tr(A)) / 2.
- `apply_surgery(model, blocks, A, P_list)`: writes the projected, absorbed weights into the first k blocks.
- `apply_svd_baseline(model, blocks, ranks)`: rank-r truncation of fc1 and fc2, r = floor(param_count / (h + d)).

### A7. k-NN evaluation (`mirror/knn.py`)
- CLS features, L2-normalised, cosine similarity, k = 20, temperature 0.07, weighted votes exp(sim/T) (DINO protocol).
- Flip consistency: fraction of queries whose prediction equals the prediction for the mirrored query.

### A8. Entry point and report (`scripts/run_e1.py`, `mirror/report.py`)
- `run_e1.py --config configs/e1.yaml [--tiny]`: `--tiny` uses `TimmTinyAdapter` and synthetic data so the full pipeline runs end to end on CPU in under a minute.
- Outputs: `results/e1_layers.csv`, `results/e1_mlp.csv`, `results/e1_surgery.csv`,
  `plots/equivariance_vs_depth.png`, `plots/breaking_share_vs_depth.png`, `results/E1_REPORT.md`.
- The report applies the decision table automatically and prints GO, PIVOT or KILL with the numbers.

### A9. Tests (`tests/`, all CPU)
1. `fit_involution` recovers a planted involution: X random, X~ = X A_true + 0.01 noise; error of recovered A small; A symmetric and A^2 = I.
2. Matching recovers a planted pairing (synthetic correlations with known pairs plus self-twins).
3. Projection: an equivariant W (built as (W0 + P W0 A)/2) has breaking share 0; a random W has about 0.5.
4. Parameter formula matches the numerical rank of the projection operator on small sizes (d = 6, h = 8).
5. Folding: absorbed block gives identical outputs to the original block (max abs diff under 1e-5, float64).
6. Identity surgery: with A = I and P = I, surgery leaves the model output unchanged.
7. Un-flip: un-flipping the tokens of a flipped symmetric input reproduces the original token order.
8. End-to-end: `run_e1.py --tiny` runs and writes all output files.

### A10. Finish Phase A
- `requirements.txt` (VM: CUDA PyTorch, `timm datasets huggingface_hub scipy networkx numpy pandas matplotlib pyyaml pytest tqdm pillow`), `README.md` with the exact Phase B commands, `.gitignore` (results data, cache, weights).
- Commit and push to a private git repository.

---

## Phase B: run on the VM

### B0. Setup
```bash
git clone <private repo> && cd Mirror_CPAL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/path/on/vm/big_disk/hf
huggingface-cli login          # accept ImageNet-1k terms on Hugging Face first
nvidia-smi                     # check how much memory the CVPR job is using
pytest -q                      # all Phase A tests must pass on the VM too
```
Set `torch.cuda.set_per_process_memory_fraction(0.2)` at the top of every GPU script.

### B1. Data cache
Download the validation split (about 6.4 GB under `$HF_HOME`), build `cache/val224_uint8.npy` and the split files.

### B2. Sanity on the real model
- Folding test on DINOv2 block 0: absorbed vs original output, max abs diff under 1e-4 (fp32).
- Identity surgery on DINOv2: k-NN accuracy identical to the original.

### B3. Run E1
`python scripts/run_e1.py --config configs/e1.yaml`
1. Statistics on the fit set (2,000 images and their mirrors).
2. A_l per layer, shared A, d+ and d-; held-out errors e(A), e(I), e(null).
3. Position-embedding check: ||mirror(P) - P A|| / ||P|| on the centred 16 x 16 grid; e(A) at layer 0.
4. Neuron matching per block; breaking shares s1, s2 vs null; implied savings.
5. Zero-shot surgery for k in {3, 6, 9}: k-NN top-1 and flip consistency for original, surgery,
   rank-matched SVD, and random-involution surgery.

Expected time under 1.5 GPU-hours; VRAM about 10 GB.

### B4. Decision table (applied automatically in E1_REPORT.md)
| Check (blocks 1 to 6) | GO | KILL |
|---|---|---|
| e(A_shared) vs e(null) | at least 5x lower | less than 3x lower |
| e(A) vs e(I) | e(A) at most 0.5 e(I) | informative only (if e(I) is tiny, features are already flip-invariant; check d-) |
| d- / d | at least 0.2 | below 0.1 |
| MLP breaking share (null about 0.5) | at most 0.2 | above 0.3 |
| Zero-shot surgery at k = 6, k-NN drop | at most 10 points and at least 5 points better than rank-matched SVD | no better than SVD |

- **GO:** all GO cells. Next: E2 (attention twins, block-diagonal implementation), E3 (healing and baselines).
- **PIVOT:** structure checks pass, but the surgery drop is above 10 points. Run a 1-hour healing distillation check before deciding.
- **KILL:** any KILL cell in rows 1, 4 or 5.
