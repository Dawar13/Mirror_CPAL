# PROJECT CONTEXT v2: The Symmetry Tax / Mirror Surgery (CPAL 2027)

Replaces PROJECT_CONTEXT.md v1. Updated 8 November 2026 after E1 and E1b.

## 1. Working title and thesis
**The Symmetry Tax: Vision Foundation Models Learn Their Features Twice, and Mirror Surgery Refunds It**

Flip augmentation leaves a measurable structure in pretrained ViTs. Each layer has a mirror operator
A_l (a symmetric orthogonal matrix with A_l^2 = I). The features split into even directions
(unchanged by a mirror) and odd directions (sign-flipped). Most MLP neurons come in mirror-twin
pairs whose weights satisfy w_j = A_l w_i. This duplication is a parameter tax. It can be refunded
by projecting the weights onto the exactly mirror-symmetric form and computing each twin pair once
(u + v and u - v). A short healing distillation follows. Because the redundancy is a symmetry and
not low rank, the refund works where low-rank compression fails, for example on DINOv2, whose
training spreads information evenly across directions.

## 2. Venue and dates
- CPAL 2027 Proceedings Track: 5 December 2026 AoE (PMLR, 9 pages). Primary target.
- Recent Spotlight Track: 18 January 2027. Fallback.
- Plan: experiments finish by 26 Nov; writing runs 22 Nov to 3 Dec; submit 5 Dec.

## 3. Status so far (all on GitHub)
| Run | Verdict | Key numbers |
|---|---|---|
| E1: one operator for all 12 layers (DINOv2 ViT-B) | KILL | Shared fit collapsed (8 of 768 odd directions); layers 9 to 11 matched no operator and dominated the fit. Spec flaw: the fit pooled unnormalised layers |
| E1 per-layer diagnostic (chosen after seeing data) | all rows pass | Same five checks as E1b; block 1 has 74% twin neurons (mean twin correlation 0.77) |
| **E1b: per-layer operators, pre-registered, fresh fit images** | **GO** | Operator fit 10.6x better than a random involution; e(A)/e(I) = 0.16; d-/d = 26%; MLP breaking share 0.194 (matched random 0.416); k-NN: original 77.4%; mirror surgery 75.7 / 73.7 / 71.7% at 3 / 6 / 9 blocks vs same-size SVD 76.1 / 64.4 / 34.8%; about 40% of MLP parameters removed in converted blocks; flip consistency after surgery 89.7% |
| E1b arm: DINOv2 with registers | AMBER (fc1 breaking share 0.223) | Late layers line up with registers, supporting artefact tokens as the cause of E1's failure; 9-block surgery keeps 75.8%; SVD at 8% looks suspicious (check) |
| E1b arm: one operator shared over layers 1 to 6 | AMBER | Beats SVD at 6 blocks (72.6 vs 66.0), fits worse than per-layer |

All E1b numbers come from one fit seed. The headline numbers need 3 seeds.

## 4. Claims the paper will make (each tied to experiments in PAPER_EXPERIMENTS.md)
- **C1, the symmetry tax:** early and middle blocks of flip-trained ViTs are near mirror-equivariant with per-layer operators; most MLP neurons are mirror twins. The tax is given by a closed-form count (Section 5.3).
- **C2, the operator drifts with depth:** one operator cannot serve all layers; nearby layers agree on their high-energy subspace.
- **C3, artefact tokens break the symmetry:** in plain DINOv2, the late-layer failure comes from high-norm artefact tokens; registers (or masking those tokens) restore it.
- **C4, symmetry is not low rank:** at equal parameters, mirror projection beats low-rank compression when the weight spectrum is flat, and a simple rule predicts which wins, per layer.
- **C5, Mirror Surgery works:** released checkpoints become mirror-symmetric, cheaper networks (MLP and attention). After equal-budget healing they beat SVD, SliceGPT-style slicing and width pruning, and beat a random-structure control (so the structure matters, not only the healing).
- **C6 (stretch), data-free readout:** the operator and twins can be read from the weights alone.
- **C7, generality:** the effect appears in DeiT-III (supervised) and MAE, and is weaker in SigLIP2 (trained with text, which cares about direction).

## 5. Mathematics (load-bearing)
**5.1 Operator fit.** C = sum over tokens of X^T X~ (X~ = mirrored image's tokens at mirrored positions).
A = V sign(Lambda) V^T, with (C + C^T)/2 = V Lambda V^T. This is the exact minimiser of
||X~ - X A||_F over symmetric orthogonal involutions.

**5.2 Twins and projection.** P is the neuron involution (pairs plus self-twins). Equivariance needs
P W1 = W1 A, A W2 = W2 P, P b1 = b1, A b2 = b2. Projection: W_eq = (W + P W A)/2. Breaking share
s = ||W - W_eq||^2 / ||W||^2. In even/odd coordinates a twin pair has weights [a, b] and [a, -b],
computed as u = a.x+, v = b.x-, outputs u + v and u - v. A self-twin has b = 0 (GELU is not odd).

**5.3 The tax.** The free parameters of an equivariant h x d map number (h d + tr(P) tr(A)) / 2, with
tr(P) = number of self-twins f and tr(A) = d+ - d-. So the redundant share is
1/2 - f (d+ - d-) / (2 h d).

**5.4 Attention.** Under a mirror, heads map to themselves or swap in pairs. For head h with partner
sigma(h): W_Q^{sigma(h)} A = B_h W_Q^h and W_K^{sigma(h)} A = B_h W_K^h (the same orthogonal B_h,
so attention logits are preserved); W_V^{sigma(h)} A = C_h W_V^h and A W_O^h = W_O^{sigma(h)} C_h.
For twin heads the head-internal basis is a free gauge, so B_h = C_h = I after alignment. Self-heads
have involutions B_h and C_h.

**5.5 Exactness versus accuracy.** In a residual network, exact end-to-end equivariance needs one
operator for the stream across the converted span. So there are two designs:
- **V-exact:** one operator A* for the whole converted span, fitted with each layer's statistics normalised to equal energy. Exactly equivariant over the span; no basis changes; cleaner speed-up.
- **V-local:** per-layer operators. More accurate, approximately equivariant; needs one d x d basis rotation per block boundary (counted in FLOPs and latency).

The paper reports both. Healing may close the accuracy gap of V-exact.

**5.6 Symmetry versus low rank (C4 rule).** At an equal parameter count, SVD keeps the top r singular
directions and discards energy E_svd = 1 - sum_{i<=r} sigma_i^2 / sum_i sigma_i^2, while mirror projection
discards energy s. Prediction: mirror surgery wins where s < E_svd. Use activation-weighted versions
(weights multiplied by input activation covariance to the power 1/2) as the primary measure; plain weight energy is secondary.

**5.7 Savings.** A square d x d map in even/odd form costs d+^2 + d-^2, a saving of 2 d+ d- / d^2.
Twin pairs halve their cost. Token-to-token attention cost is unchanged.

## 6. Prior work and verified facts
| Work | Fact | Our delta |
|---|---|---|
| Flopping for FLOPs, ICML 2025, arXiv 2502.05169 | Mirror-equivariant ViTs from scratch; E(ViT-H) 84.4% vs 84.6% at 87.3 vs 168.0 GFLOPs | Released checkpoints, no from-scratch training |
| Quick ViTs, arXiv 2505.15441 | First-half octic equivariance from scratch; 85.0% vs 84.6% at 102.3 vs 167.8 GFLOPs; measured throughput gain 1.15 to 1.22x | Same |
| Shahverdi, Marchetti, Bokman, Kohn, ICML 2026, arXiv 2601.21645 | Layerwise equivariance under identifiability; small models trained from scratch; skip connections not covered | Real foundation models with skip connections; compression |
| Ong and Bokman, NeurIPS 2026, arXiv 2608.30417 | Heads map to themselves or swap in pairs; "open for an entire transformer" | Measured in real models; compiled into cheaper layers |
| Bokman and Kahl, NeurIPS 2023, arXiv 2305.17017 | Permutation-type layerwise equivariance in small CNNs | Transformers, foundation models, compression |
| Gruver et al., ICLR 2023, arXiv 2210.02984 | Measures learned equivariance in 410 models | Weight structure and exploitation |
| Chiba et al., arXiv 2607.03108 | Data-free symmetry readout from weights for small MLPs on 2D shapes | ViTs; compression |
| Darcet et al., Vision Transformers Need Registers, arXiv 2309.16588 | High-norm artefact tokens; registers remove them | Link to symmetry breaking (C3) |
| SliceGPT, arXiv 2401.15024 | LayerNorm folding; PCA slicing | Folding reused; symmetry basis instead of a variance basis |
| eLife mirror-symmetric viewpoint tuning (PMC11142642) | Mirror-symmetric tuning in deep networks and primate face patch AL | Biological link for the discussion |

Re-run the novelty search on OpenReview (ICLR 2027 submissions) in the final week.

## 7. Resources and rules
- Main model from now on: DINOv2 ViT-B/14 with registers (`dinov2_vitb14_reg`). Contrast model: plain `dinov2_vitb14`.
- Data hygiene from now on: fit operators on ImageNet **train** images (unlabelled use) and evaluate on **val**. Healing data comes from train only, never from val.
- GPU is shared: cap memory per process, run small jobs concurrently through a queue, and run long jobs (healing) one at a time.
- Never invent numbers; every number comes from a script with a seed, config and git hash.
- Pre-register gated hypotheses in `PREREG_E2.md` and commit it before running those experiments.
- Reports: British spelling, plain language, no em-dashes.
