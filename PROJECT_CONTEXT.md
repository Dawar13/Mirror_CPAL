# PROJECT CONTEXT: Mirror Surgery (CPAL 2027)

## 1. Summary
**Title:** Mirror Surgery: Pretrained Vision Transformers Secretly Contain a Mirror-Symmetric Network, and Extracting It Cuts Their Compute

Vision transformers trained with random left-right flips (DINOv2, DeiT-III, MAE) appear to learn a hidden
mirror symmetry inside their weights: one mirror operator on the residual stream, mirror-twin MLP
neurons, and attention heads that either mirror themselves or swap in pairs. We (1) measure this
structure in real foundation models for the first time (a law: near-exact in early and middle blocks,
fading with depth), and (2) use it to convert released checkpoints into mirror-equivariant,
block-diagonal networks that need roughly half the multiply-adds in the converted blocks, with a
short healing distillation instead of training from scratch.

## 2. Venue and dates
- CPAL 2027 (Conference on Parsimony and Learning), Tokyo, 23 to 26 March 2027.
- Proceedings Track: 5 December 2026 AoE, PMLR, 9 pages, double-blind. Primary target.
- Recent Spotlight Track: 18 January 2027 (non-archival). Fallback.
- Internal milestones: E1 go/kill by 9 Nov; E2 and E3 by 22 Nov; writing 22 Nov to 3 Dec; submit 5 Dec.

## 3. Background in plain words
- **ViT.** Image cut into patches; each patch becomes a token; tokens pass through blocks.
  Each block: attention (tokens share information, split into heads) then an MLP (each token alone,
  middle layer of "neurons"). Every block adds its result to a shared running total, the
  **residual stream**.
- **Flip augmentation.** DINOv2, DeiT-III and MAE saw randomly mirrored images during training.
- **Equivariance.** Flipping the input changes internal numbers in a fixed, predictable way.
- **Two feature types.** Under a mirror, every feature either stays the same ("even", +1) or flips
  sign ("odd", -1).
- **Schur's lemma, practical form.** A mirror-equivariant linear map never mixes even and odd
  features, so in the right coordinates each weight matrix is block-diagonal: about half the
  multiplications.
- **Twins.** Neurons or heads that come in mirror pairs can be computed together with half the work
  (a "butterfly": two half-size products, then add and subtract).
- **Distillation.** The original model teaches the converted model for a short time.

## 4. The mathematics (load-bearing)
Notation: residual width d (768 for ViT-B), MLP width h (3072), heads H (12) of size d_h (64).
X = residual-stream activations (tokens x d) for images; X~ = activations for the mirrored images,
with patch tokens un-flipped so that rows correspond to the same image location.

**4.1 Mirror operator (closed form).** We seek A with A = A^T, A^2 = I (orthogonal involution)
minimising ||X~ - X A||_F. Since ||X A|| = ||X||, this maximises tr(A C), C = sym(X^T X~).
For symmetric A with eigenvalues +-1, tr(A C) <= sum |lambda_i(C)|, with equality at
**A* = V sign(Lambda) V^T** where C = V Lambda V^T. The +1 eigenspace V+ (dimension d+) holds
even features, the -1 eigenspace V- (dimension d-) holds odd features.

**4.2 One operator through depth.** Each block adds into the same residual stream. If every block is
equivariant, x_{l+1} = x_l + f_l(x_l) forces the same A at every depth (on the span of the stream).
Prediction: per-layer A_l agree with one shared A in early and middle blocks.

**4.3 MLP constraints.** With a neuron involution P (pairs plus self-twins, P^2 = I), equivariance
of fc1 then GELU then fc2 requires P W1 = W1 A, P b1 = b1, A W2 = W2 P, A b2 = b2.
GELU commutes with permutations (not with general rotations), which is why the hidden symmetry is a
permutation. A self-twin neuron must read only even features (GELU is not an odd function).

**4.4 Projection onto equivariant weights.** The map W -> P W A is an involution on matrix space.
W_eq = (W + P W A)/2 is the nearest equivariant matrix; the breaking share is
s = ||W - W_eq||^2 / ||W||^2 (random structure gives about 0.5). The equivariant subspace has
dimension (h d + tr(P) tr(A)) / 2, which gives the exact parameter count.

**4.5 Attention.** Under a mirror, heads can only map to themselves or swap in pairs
(Ong and Bokman, NeurIPS 2026). For a twin pair h1 <-> h2 with W^(h2) = W^(h1) A: split
W^(h1) = [W+ | W-] on (V+, V-). Then z^(h1) = u+ + u-, z^(h2) = u+ - u- with u+- = W+- x+-.
Both heads cost d_h * d multiply-adds instead of 2 d_h d. A self-mirrored head with its own
internal involution A_h becomes block-diagonal.

**4.6 Savings.** A square d x d map becomes blocks of d+^2 + d-^2: saving 2 d+ d- / d^2
(50% if d+ = d-, 32% if d- = 0.2 d). Token-to-token attention cost is unchanged; savings are in
projections and MLPs. Real latency gain is smaller than the FLOP gain (see Quick ViTs below).

## 5. Hypotheses
- **H1 (shared mirror operator):** one A fits all early and middle layers, far better than a random involution.
- **H2 (twins):** MLP neurons and attention heads form mirror pairs plus self-twins.
- **H3 (depth law):** symmetry-breaking share is low early, rises in late blocks.
- **H4 (useful):** zero-shot projection onto the equivariant weights in early blocks loses little
  accuracy and beats an equal-parameter SVD cut; after healing, it beats SliceGPT, SVD and width
  pruning at equal FLOPs.
- **H5 (secondary finding):** text-supervised encoders (SigLIP2) are less mirror-symmetric than
  self-supervised ones, because text cares about direction.

## 6. Prior work, verified facts and our delta
| Work | Verified fact | Our delta |
|---|---|---|
| Flopping for FLOPs, Bokman, Nordstrom, Kahl, ICML 2025, arXiv 2502.05169 | Mirror-equivariant ViTs trained from scratch; E(ViT-H) 84.4% vs 84.6% top-1 at 87.3 vs 168.0 GFLOPs | We extract the symmetry from released checkpoints; no from-scratch training |
| Quick ViTs, arXiv 2505.15441 | Octic-equivariant first half of blocks, from scratch; H8 ViT-H/14 85.0% vs 84.6% at 102.3 vs 167.8 GFLOPs; measured training throughput gain only 1.15 to 1.22x | Same; plus we test whether pretrained models already contain the structure |
| Identifiable Equivariant Networks are Layerwise Equivariant, Shahverdi, Marchetti, Bokman, Kohn, ICML 2026, arXiv 2601.21645 | Under identifiability, end-to-end equivariance implies layerwise equivariance; experiments only on small models trained from scratch; saw two heads swap in a small CIFAR-10 attention autoencoder; states "we have not covered skip connections" | First test on real foundation models with skip connections; first use for compression |
| No Equivariant Architecture Covers All Equivariant Attention, Ong and Bokman, NeurIPS 2026, arXiv 2608.30417 | Group acts on heads only by permuting them; any single architecture covers at most one component; "remains open for an entire transformer"; no pretrained models | We count self-heads and twin heads in real models and compile whatever the model chose |
| Investigating how ReLU networks encode symmetries, Bokman and Kahl, NeurIPS 2023, arXiv 2305.17017 | Small CNNs trained with flips tend to be layerwise equivariant via permutations | Transformers and foundation models; residual operator; compression |
| Gruver et al., Lie derivative, ICLR 2023, arXiv 2210.02984 | Learned equivariance measured in 410 pretrained models; larger, more accurate models are more equivariant | Weight-space structure and exploitation, not only measurement |
| SliceGPT, arXiv 2401.15024 | Exact folding of LayerNorm into neighbouring weights; PCA slicing | We use the folding trick; our basis comes from data symmetry, not variance |
| Canonicalisation / symmetrisation (Mondal et al. NeurIPS 2023; arXiv 2305.09900; arXiv 2306.02866) | Make pretrained models equivariant by input-side averaging | These add compute; we remove it |
| Mirror-symmetric viewpoint tuning, eLife (PMC11142642) | Mirror-symmetric tuning emerges in deep networks and in primate face patch AL | Biological link for the discussion |

**Open statement (to re-check on OpenReview ICLR 2027 before submission):** as of 7 November 2026,
under the logged queries, nobody has recovered the residual mirror operator, twin neurons and twin
heads of a released flip-trained ViT and converted it into an exactly mirror-equivariant,
block-diagonal network for compute savings.

## 7. Experiment plan (total about 12 H100-hours)
- **E1 (go/kill, under 1.5 h):** DINOv2 ViT-B. Mirror operator, twin neurons, breaking share, zero-shot surgery vs SVD. See E1_INSTRUCTIONS.md.
- **E2 (about 2 h):** attention twins and self-heads; real block-diagonal and butterfly implementation; exactness tests; measured GFLOPs, parameters, latency.
- **E3 (about 5 h):** healing distillation (block-output feature matching, 100k ImageNet images) for DINOv2 ViT-S and ViT-B; equal-FLOP baselines with the same healing budget: SliceGPT slicing, rank-matched SVD, MLP width pruning.
- **E4 (about 2.5 h):** the law across models: DINOv2 S/B/L, DeiT-III B, MAE B, SigLIP2 B (H5). Measurement only, plus zero-shot surgery.
- **E5 (optional):** class-conditional DiT-XL/2 or SiT-XL (flip-trained), FID-10k.

## 8. Metrics
- Structure: e(A) vs e(null) and e(I); agreement of A across layers; d+ and d-; twin pairs vs self-twins (neurons and heads); breaking share per layer (headline law plot).
- Quality: ImageNet k-NN and linear-probe top-1; ADE20K linear segmentation mIoU; flip consistency (100% by construction after conversion).
- Cost: GFLOPs, parameters, measured latency and throughput on H100.

## 9. Risks and mitigations
- d- much smaller than d/2: savings smaller; report honestly with the 2 d+ d- / d^2 formula.
- Learned position embeddings not mirror-consistent: measure; if needed treat position embedding as an explicitly flipped input.
- LayerNorm centring and A: fold LayerNorm (SliceGPT) and fit A on centred features.
- Late blocks break symmetry: convert only the blocks the law says are clean.
- Latency gain below FLOP gain: report measured numbers; stress parameters and exact flip consistency.
- Collision: re-run the novelty search on OpenReview (ICLR 2027) before submission.

## 10. Repository layout
```
mirror/            library code (adapters, data, stats, fit, matching, surgery, knn, report)
scripts/           run_e1.py and later experiment entry points
tests/             pytest tests (CPU, tiny random ViT, synthetic data)
configs/           yaml configs per experiment
results/           csv/json outputs (git-ignored except reports)
plots/             figures
```

## 11. Environment and resources
- Laptop: about 6 GB free. Never download models or datasets locally. Local work is code and CPU tests only.
- VM: one H100 80 GB, shared with a CVPR project. Mirror E1 needs about 10 GB VRAM; cap with
  `torch.cuda.set_per_process_memory_fraction(0.2)`. All downloads under `$HF_HOME` on the VM disk.
- Code moves laptop -> VM through a private git repository.

## 12. Rules for Claude Code
- Never invent numbers, citations or results. Mark anything not verified as (verify).
- Every number in a report must come from a script in this repo with a fixed seed; log config, seed and git commit hash with each result file.
- Do not change the E1 decision thresholds without asking.
- Store statistics, not raw activations, to keep memory and disk small.
- Write reports in British spelling, plain language, no em-dashes.
