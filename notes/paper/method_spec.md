# Method and setup specification (what the code does for the final runs)

Written 17 November 2026 from a read-only pass over the code, configs and stored run files. Every item describes the
code path of the final H200 runs (MUST-07, MUST-09, MUST-10, MUST-11, CORE-06, CORE-07, MUST-04). Where the plan
(PROJECT_CONTEXT.md, PROJECT_NEXT_PHASE.md, PREREG_NEXT_PHASE.md) says something else, the code is reported and the
difference is listed in Section 10. Anything not checked against code or a stored file is marked (verify).
References are file:line in this repository. Blocks are 0-based in code; "block 1" in prose is code index 0.

Main model: `dinov2_vitb14_reg` (torch.hub `facebookresearch/dinov2`, hub code 7764ea0f; docs/model_recipes.md:9).
Constants at 224 px: d = 768, h = 3072, L = 12 blocks, H = 12 heads, dh = 64, patch 14, grid 16 x 16, 4 registers,
T = 1 + 4 + 256 = 261 tokens (runs/MUST-09/.../mirror_exact/s0/metrics.json, cost.arch). The hub stores a 37 x 37
position table (518 px); LayerNorm eps 1e-6; LayerScale in every block.

## 0. Data roles and preprocessing

- All ImageNet caches: resize 256 (bicubic), centre crop 224, uint8 (scripts/build_train_cache.py:16-19,
  mirror/data.py:124-128). Mirror of an image: `torch.flip(x, dims=[-1])` on the 224 crop (mirror/data.py:73-75).
- Train roles, chosen per class with seed 0, disjoint by construction (scripts/build_train_cache.py:5-9, 52-53):
  fit0, fit1, fit2 (2 per class, 2,000 each); bank (100 per class, 100k); healval (5 per class, 5k, LR selection);
  heal (100 per class, 100k, stored as resize 256 + centre crop 256 for random crops).
- Method-selection set "msel": 10 per class (10k) from train shards after the used ones, own generator (seed 1)
  (scripts/build_train_cache.py:21-28; mirror/protocol.py:37-48).
- Validation: all 50k ImageNet-val images, reporting only (mirror/protocol.py:59-61).
- Every structure of the final runs (A*, pairings, head maps, register pairing) uses fit seed 0 only (2,000 images).

## 1. Residual-stream mirror operator

**Statistics** (mirror/stats.py:20-30, 33-35, 83-124). Each fit batch x is run as one batch [x; mirror(x)] through
the original model. At every stream layer l (layer 0 = tokens entering block 1, layer l = output of block l) the
mirrored half is aligned by reversing the patch grid's column axis only: patch (r, c) <-> (r, 15 - c); CLS and the
4 register tokens keep their own index (no register swap in the statistics). All 261 tokens of every image are used.
Each token is channel-centred (x - mean_c x) and accumulated in float64:
C_l = sum X^T X~ and G_l = sum X^T X (d x d), with X the original and X~ the aligned mirrored tokens.

**Per-layer operator A_l** (mirror/fit.py:11-30; mirror/variants.py:142-152). C_l is projected onto the centred
subspace, C <- Pc C Pc with Pc = I - 11^T/d; S = (C + C^T)/2 = V diag(lam) V^T; A_l = V diag(sign lam) V^T, with
|lam| <= 1e-10 max|lam| set to +1 (so A 1 = 1). This is the exact minimiser of ||X~ - X A||_F over symmetric
orthogonal involutions. Fitted d- for the main model, fit seed 0: 368 (layer 0), 279 (layer 1), 128 (layer 6),
8 (layer 12) (runs/x2/perlayer_reg_s0/results/e1_layers.csv). These A_l, the C_l and G_l and the twin pairings come
from the X2 run (`runs/x2/perlayer_reg_s0/results/e1_operators.npz`, config configs/x2_perlayer_reg.yaml), loaded as
"pass 1" of the canonical structure (jobs/h100_A_phase1.txt:13, `--operators-template`).

**Span operator A* (Mirror-exact)** (mirror/m1.py:78-107; called at mirror/mirror_family.py:256-259).
A*_k = fit_involution( sum_{l=0..k} Pc C_l Pc / tr(G_l) ), i.e. every stream layer weighted to equal energy, then
cleaned to an exact involution (mirror/heal.py:303-319). The full-model operator uses k = 12: stream layers 0 to 12
(embedding output and all 12 block outputs). Stored A*_12 for the main model: d+ = 700, d- = 68 (trace 632)
(runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/structure.npz; metrics.json teacher.d_plus = 700). Each span size
k in MUST-07 has its own A*_k (fitted on layers 0..k).

**Mirror-exact** (mirror/m1.py:286-322 build_vexact; mirror/mirror_family.py:504-517). One operator A* for the whole
span; deep copy of the model; position grid materialised (Section 4); full LayerNorm folding (Section 4); embeddings
projected if the span starts at block 1 (always, in MUST-07 and MUST-09); MLP and attention of every span block
projected with A_in = A_out = A*; at full conversion the even head is installed. Exactly equivariant over the span:
for the mirrored image the aligned tokens equal the original tokens times A* at every layer (docs/exactness.md:29-46).
**The final MUST-09 Mirror-exact converts the full model**: all 12 blocks, MLP and attention, the patch embedding,
position grid, CLS and registers, plus the even head (configs/must09.yaml `components: [mlp, attn]`;
scripts/must09_heal.py:423-433; spec.json blocks 0..11).

**Mirror-local** (ablation; mirror/mirror_family.py:16-20, 486-503). Per-layer operators: fc1 projected with A_b,
fc2 with A_(b+1), attention with A_in = A_out = A_b (M1 had used A_(b+1) for the output side). The model is NOT
folded and the embeddings are NOT projected. One d x d basis change per converted block is counted (k d^2).

## 2. MLP neuron pairing, projection, even/odd form

**Pairing statistic** (mirror/stats.py:127-157). fc1 outputs before GELU, all tokens of the fit images, original
half z and aligned mirrored half z~: R[i, j] = Pearson corr(z~_i, z_j) over tokens; S = (R + R^T)/2.

**Matching** (mirror/matching.py:45-98). An involution pi maximising sum_i S[i, pi(i)], i.e. a maximum-weight
matching with edge gains w_ij = 2 S_ij - S_ii - S_jj. Candidates: each neuron's top_k = 10 partners by w_ij,
positive gains only; exact blossom matching (networkx) on integer weights round(1e6 w). Matched pairs are "twins";
unmatched neurons are "self-twins". Surgery uses every matched pair, with no correlation threshold (the 0.5 threshold
is only in the twin-fraction statistic, Section 8). Main model, fit seed 0, pairs per block 1 to 12:
1168, 1116, 1013, 963, 878, 672, 554, 406, 155, 93, 19, 17; self-twins 736, 840, 1046, 1146, 1316, 1728, 1964, 2260,
2762, 2886, 3034, 3038 (structure.npz above; same as e1_mlp.csv).

**Folding before projection** (mirror/surgery.py:31-49). W1' = W1 diag(gamma2), b1' = b1 + W1 beta2 (norm2 affine);
W2' = diag(ls2) W2, b2' = ls2 * b2 (LayerScale). In the Mirror-exact model the full fold of Section 4 is applied first.

**Projection** (mirror/surgery.py:83-93, 152-158). Equivariance of fc1 -> GELU -> fc2 requires
P W1 = W1 A_in, P b1 = b1, A_out W2 = W2 P, A_out b2 = b2 (P the neuron permutation). Orthogonal projection, float64:
W1 <- (W1 + P W1 A_in)/2, b1 <- (b1 + P b1)/2, W2 <- (W2 + A_out W2 P)/2, b2 <- (b2 + A_out b2)/2.
So twin rows satisfy w_j = A w_i (tied fc1 rows), twin biases are equal, self-twin rows lie in the even subspace,
fc2 columns are tied the same way and b2 is even. Breaking share s = ||W - W_eq||_F^2 / ||W||_F^2
(mirror/surgery.py:96-100).

**GELU and biases.** GELU acts unit by unit, so it commutes with P; it is not odd, so a self-twin cannot carry an odd
input component (its row has no odd part) and the fc1 bias is shared within a pair; b2 lies in the even subspace.

**Even/odd computation** (mirror/efficient.py:13-24). With Q = [Q+, Q-] the eigenbasis of A (x = [x+, x-]):
twin pair rows [a+, a-] and [a+, -a-], self-twin row [c+, 0];
u = a+ . x+ + b1_i, v = a- . x-, pre_i = u + v, pre_j = u - v, pre_s = c+ . x+ + b1_s, g = GELU(pre);
y+ = sum_pairs p+ (g_i + g_j) + sum_self q+ g_s + b2+, y- = sum_pairs p- (g_i - g_j).
Cost per matrix h+ d+ + h- d- (h+ = pairs + self-twins, h- = pairs) = the free-parameter count of Section 5.
MUST-01 checks that this tensor-shape count equals the formula (results/MUST-01/summary.md: 0.00 pp difference).

## 3. Attention

**Head statistics** (mirror/attention.py:144-209; collected in pass 2 of mirror/mirror_family.py:263-269 on fit0,
original model). Per block, uncentred float64 cross-moments of the qkv outputs between the aligned mirrored half and
the original half, per head pair: M_t[g, h] = sum q~_g q_h^T for t in q, k, v (dh x dh).

**Head pairing** (mirror/attention.py:345-366, 411-443). On the summed q and k moments M = M_q + M_k: diagonal score
S[h,h] = sum |eig(sym M[h,h])| / sqrt(E~_h E_h) (best self-involution fit); off-diagonal
S[h,g] = ||M[g,h] + M[h,g]^T||_* / sqrt((E~_g + E~_h)(E_h + E_g)) (best orthogonal map, both directions).
sigma = involution maximising sum_h S[h, sigma(h)], exact subset dynamic programme (H = 12 <= 16), gains
2 S_hg - S_hh - S_gg, a pair kept if its gain exceeds 1e-12. Main model: 7 twin-head pairs (blocks 2, 3, 5, 6, 7, 8,
10, one each) and 130 self-heads (spec.json n_head_pairs).

**Head maps** (mirror/attention.py:446-468). Twin pair (h, g): B[h] = polar(M_qk[g,h] + M_qk[h,g]^T),
C[h] = polar of the same on v; B[g] = B[h]^T, C[g] = C[h]^T. Self-head: B[h], C[h] = fit_involution of M_qk[h,h],
M_v[h,h] (symmetric dh x dh involutions). The same B serves q and k, so logits are preserved:
(B q)^T (B k) = q^T k; the softmax scale is unchanged.

**Constraint and projection** (mirror/attention.py:543-586; docs/exactness.md:77-80). With g = sigma(h):
Wq^g A_in = B_h Wq^h, Wk^g A_in = B_h Wk^h, bq^g = B_h bq^h, bk^g = B_h bk^h, Wv^g A_in = C_h Wv^h,
bv^g = C_h bv^h, A_out Wo^h = Wo^g C_h, A_out bo = bo. Orbit averaging: self-head W <- (W + B W A_in)/2 (C for v),
b <- (b + B b)/2, Wo^h <- (Wo^h + A_out Wo^h C)/2; twin pair W^h <- (W^h + B^T W^g A_in)/2, W^g <- B W^h A_in, and
Wo likewise; bo <- (bo + A_out bo)/2. Twin pairs are first gauged (mirror/attention.py:522-540: head g rotated by
B_h^T on q, k and C_h^T on v, Wo^g <- Wo^g C_h), so after conversion B = C = I on pairs: Wq^g = Wq^h A*.
Q, K, V come from the folded attn.qkv (norm1 affine folded in), O from the folded attn.proj (LayerScale folded in)
(mirror/attention.py:84-107).

**Even/odd form** (mirror/efficient.py:26-36). A twin pair costs one head (q_h = u + v, q_g = u - v); a self-head is
block-diagonal after rotating its head basis by the eigenbasis of B (q, k) and C (v; absorbed into Wo); token mixing
softmax(q k^T / sqrt(dh)) v is unchanged.

## 4. Folding, positions, CLS, registers, final norm, even head

**LayerNorm folding** (mirror/fold.py:1-29, 130-171, 203-230, 236-292). SliceGPT-style, exact: every write into the
stream is centred with C = I - 11^T/d (patch embedding, CLS, registers, position table, mask token, attn.proj, fc2);
every LayerNorm becomes an RMSNorm without affine, x / sqrt(mean(x^2) + eps), same eps; norm1 and norm2 gamma, beta
fold into qkv and fc1 (W diag(gamma), b + W beta); LayerScale gammas fold into proj and fc2 (W <- C diag(gamma) W,
b <- C (gamma b)), gamma set to 1. The final norm becomes Sequential(rms, affine) with the affine kept in the model
(`final_norm_affine=False`). Exactness then only needs A* orthogonal; A* 1 = 1 also holds (Section 1).
Models with q/k-norm, gated attention, SwiGLU, `norm_pre` or `fc_norm` are refused (mirror/fold.py:181-199, 225-229).

**Position embeddings** (mirror/m1.py:151-167, 192-257). DINOv2 interpolates its 37 x 37 table bicubically with
interpolate_offset 0.1, which is not flip-symmetric, so the 16 x 16 grid of the evaluation size (224 px) is first
written into the model (`materialise_pos_embed`: 1 + 256 rows). Projection on that grid:
pos(r, c) <- (pos(r, c) + pos(r, 15 - c) A*)/2. The converted model is tied to 224 px. At 448 px (ADE20K) the
healed grid is interpolated to 32 x 32 with the model's own interpolation and re-projected with A*
(scripts/tier2_ade20k.py:83-93).

**Patch embedding**: W <- (W + A* flip_w(W))/2 (flip of the kernel width), b <- (b + A* b)/2 (mirror/m1.py:207-218).

**CLS** token and its position row projected to the even subspace, c <- (c + c A*)/2 (mirror/m1.py:238-250).

**Registers** (mirror/m1.py:328-381; mirror/mirror_family.py:261, 271-272). Alignment scores
S[l, i, j] = sum <reg~_i, reg_j A*> (centred tokens), cosine-normalised, averaged over stream layers 0..12 and
symmetrised; involution by maximum-weight matching, margin 0; projection reg_j <- (reg_j + reg_pi(j) A*)/2.
For the main model the matching returned the identity (no register pairs; teacher.reg_perm = [0, 1, 2, 3]), so
all 4 registers are projected to the even subspace. DINOv2 registers carry no position embedding.
The DINOv2 mask token is centred by the fold but not projected; it is unused at evaluation.

**Final norm and even head** (mirror/m1.py:128-143, 260-268). model.norm = Sequential(rms, even, affine) with
even: y -> y P, P = (I + A*)/2, a fixed fp32 buffer run outside autocast. It acts on every token (CLS and patches).
k-NN feature = affine(rms(t_CLS) P) = gamma * (rms(t_CLS) P) + beta, exactly flip-invariant. Dense tasks bypass P
(Section 7). The even head is installed only at full conversion.

## 5. Parameter accounting

**Free parameters of an equivariant map** (mirror/surgery.py:108-115; general group: mirror/groups.py:672-702).
For an h x d matrix with neuron involution P (f = tr P self-twins) and stream involution A:
free = (h d + tr(P) tr(A))/2 = h+ d+ + h- d-, tr(A) = d+ - d-. General finite group:
free = (1/|G|) sum_g tr(rho(g)) fix(sigma(g)); tax = 1 - free/(h d). For LR the redundant share is
1/2 - f (d+ - d-)/(2 h d). Example, block 1 under A*: (3072 x 768 + 736 x 632)/2 = 1,412,224 per MLP matrix
(59.9% of dense). Attention (mirror/attention.py:618-649): a self-head adds (dh d + tr B tr A_in)/2 to Wq and Wk,
(dh d + tr C tr A_in)/2 to Wv and (d dh + tr A_out tr C)/2 to Wo; a twin pair adds dh d per matrix (half its dense
count); biases are not part of the budget.

**Parameter fraction 0.7646** (scripts/must09_heal.py:575-599 -> scripts/collect_results.py:709-745;
mirror/heal_structured.py:901-978). params_model = N_orig - N_conv_dense + N_conv_free + basis changes, with
N_orig = 86,583,552 (every parameter of the released hub model, including its 37 x 37 position table and LayerNorm
affines), N_conv_dense = 12 x (4 d^2 + 2 h d) = 84,934,656, N_conv_free = 64,554,464 (sum of the formula above over
fc1, fc2, Wq, Wk, Wv, Wo of all 12 blocks), basis changes 0 at full conversion. 66,203,360 / 86,583,552 = 0.76462
(0.7600 of the converted matrices). Biases, norms and embeddings count as dense; the embedding projection and the
smaller materialised grid are not credited. The stored healed model holds 86,281,728 entries (dense projected
weights plus the d x d even-head buffer), so the saving is realised only by the even/odd form.

**GFLOPs fraction 0.774** (scripts/collect_results.py:718-745). GFLOPs = 2 x MACs per image,
MACs = 256 x 3 x 14^2 x d (patch embedding) + sum over blocks [T x (matrix MACs at the kept count) + 2 T^2 d]
(+ T d^2 per basis change); norms, softmax, GELU and biases not counted. Dense 47.078 GFLOPs; Mirror-exact 36.440;
ratio 0.7740 (metrics.json cost). Flip averaging counts twice the dense cost (94.16; results/MUST-10/MUST10.md).

**Basis changes** (mirror/mirror_family.py:36-48, 389-392; mirror/budget.py:189-195). Mirror-exact: d^2 at each span
end that is not a model boundary (6- and 9-block spans: 589,824; full: 0). Mirror-local: d^2 per converted block.

**Budget matching** (mirror/budget.py:1-40, 359-412, 754-761; scripts/must07_compressors.py:676-686).
budget_total = Mirror-exact's free matrix count + its basis changes, over the same converted matrices. A baseline's
params_total = its own matrix count + its own basis changes. Eligible iff 0 <= (params_total - budget)/budget <= 1%
(or within one allocation step when that step exceeds 1%, which never happens at DINOv2-B sizes). Per-matrix targets
= ceil(mirror_count x total / sum mirror_counts); allocations round up and a greedy fill (`_fill`) adds units where
the deficit is largest until the total is reached, so methods overshoot by at most one step. Full-size gaps:
SVD, SVD-LLM, ASVD, FLAP, ZipIt +544 parameters (+0.0008%); SliceGPT +0.19% (gate_b.json). The A*-type controls
(pair_*, random_structure) have gap 0 by construction. Mirror-local is not budget-matched (full: +7.5%).

## 6. Healing (MUST-09)

Config configs/must09.yaml (copied from configs/m3_vexact.yaml); runner scripts/must09_heal.py; trainer
mirror/heal.py:939-1202. Jobs: jobs/h100_A_phase1.txt:25, 34-35; jobs/h100_A3_lanes.txt:17-18;
scripts/h100_heal_gate.sh:20. All runs used `--span-structure results/MUST-07/structure/dinov2_vitb14_reg_s0.npz`
and `--micro-batch 64` (meta.json args).

- **Teacher**: the original model, frozen, eval mode. **Student**: the compressed model, also kept in eval mode
  (no dropout or drop path).
- **Targets** (mirror/heal_vexact.py:111-228; mirror/heal_structured.py:879-898; `targets: sym` for every method).
  The teacher runs on [x; mirror(x)]. Final terms (all methods): CLS and patch tokens after the teacher's final norm,
  flip-averaged, t = (f(x) + align(f(mirror x)))/2 (patch grid un-flipped; no A* here). Block terms, all 12 blocks:
  Mirror-exact and random structure get the centred teacher block output symmetrised with A*,
  (c(x) + align(c(mirror x)) A*)/2; a folded baseline (SliceGPT) gets the centred teacher output; an unfolded
  baseline (SVD, SVD-LLM) gets the teacher's raw block output (metrics.json teacher.block_targets per run).
- **Loss** (mirror/heal.py:709-718, 1039-1046): sum over 14 terms (CLS, patch, 12 blocks), weight 1 each, of
  mean((s - t)^2) / n_t, with n_t the teacher target mean square measured once before training on 1,024 evenly
  spaced heal images (centre view) and then fixed.
- **Data**: the 100k heal set, uint8 256 x 256; per image RandomResizedCrop to 224 (scale 0.35 to 1, ratio 3/4 to
  4/3, torchvision sampling, bilinear grid_sample without antialiasing) and horizontal flip p = 0.5, on the GPU
  (mirror/heal.py:227-293). Data order: a fresh seeded permutation per epoch (mirror/heal.py:202-221).
- **Schedule**: 2 epochs, batch 256 (micro-batch 64, gradient accumulation), floor(100,000 / 256) = 390 steps per
  epoch, **780 steps**; linear warm-up round(0.05 x 780) = 39 steps, then cosine to 0 (mirror/heal.py:797-804,
  1010-1020). bf16 autocast; no gradient clipping.
- **Optimiser**: AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0.05 on tensors with ndim >= 2 only (this includes
  pos_embed, cls_token and register_tokens), none on biases (mirror/heal.py:1022-1027).
- **What trains** (mirror/heal_structured.py:635-717; mirror/heal_vexact.py:248-254, 283-332). For every method:
  the converted blocks' parameters and the embeddings (patch embedding, position grid, CLS, registers); frozen:
  absorbed norm and LayerScale parameters, final norm affine, mask token; the even head is a buffer.
  Mirror-exact and random structure: the matrices are stored dense and AdamW updates every entry; after every
  optimiser step MLP, attention and embeddings are re-projected with A* in float64 and rounded once
  (mirror/heal.py:346-407, 1131-1135; mirror/heal_attn.py MirrorAttnConstraint), so the model stays exactly
  equivariant (stored n_trainable 85,671,168). Baselines: structure kept by module form, nothing projected:
  low rank as LowRankLinear with balanced factors U_r S_r^(1/2), S_r^(1/2) V_r^T of the compressed weight's own SVD
  (the same for SVD and SVD-LLM; rank fixed), slices as SlicedLinear
  with fixed bases, pruned or merged MLPs as smaller Linear layers (mirror/heal_structured.py:1-37).
- **Learning rate** (scripts/must09_heal.py:681-721; mirror/heal.py:1233-1336). Grid {1e-5, 5e-5, 1e-4}; one 100-step
  trial per rate (own warm-up and cosine, same seed and data order) on the seed-0 run; score = the normalised
  distillation loss on the 5k healval images (centre view), not k-NN; lowest wins, ties to the smaller rate. When the
  winner is 1e-4, a 2e-4 trial is run and recorded but never chosen. Seeds 1 and 2 reuse the seed-0 rate (`--lr-from`).
  Every method chose 1e-4; for plain SVD the 2e-4 trial would have won (runs/MUST-09/.../svd/s0/metrics.json).
- **Seeds**: healing seed 0, 1, 2 (data order, crops, flips; augmentation generator seed + 7919). The structure is the
  same fit-seed-0 structure for every seed and method; the random-structure control uses structure seed 1.
  Runs: Mirror-exact 3 seeds; SVD-LLM (gate-B best) 3; SliceGPT (second) 1; plain SVD 1; random structure 1.
- **Logging**: validation k-NN at steps 0 and 780 on all 50k, at 100 and 300 on 10 images per class (10k, first in
  dataset order; mirror/data.py:31-53); bf16. Final fp32 pass for flip consistency; exactness probe on 64 val images.

## 7. Evaluation protocols

**k-NN** (mirror/knn.py:15-46; scripts/must07_compressors.py:382-470). Bank: 100k train images (100 per class).
Queries: 50k val (reporting) or 10k msel (selection). Feature: CLS after the final norm (Mirror-exact: after the even
head), L2-normalised; cosine similarity; k = 20; votes exp(sim / 0.07); argmax. Forward in bf16 autocast for the
headline top-1.

**Method selection and gate B** (scripts/must07_compressors.py:781-868, 950-975). Each compressor grid point is
built, checked against the budget (Section 5), scored by k-NN of the 10k msel queries against the bank; best wins,
ties to the first grid point; the chosen point is then scored on validation. Fit seeds 1 and 2 reuse the seed-0
choice. Gate B: methods 3 to 12 ranked by msel top-1 at the full size, fit seed 0 (results/MUST-07/gate_b.json:
1 SVD-LLM 77.36, 2 SliceGPT 69.53, 3 FLAP 67.00, 4 ASVD 59.70, ...).

**Flip consistency**: share of validation images whose k-NN prediction for x equals that for mirror(x) (bank fixed).
Two passes: bf16 (bank, queries and mirrored queries all in bf16 autocast) and fp32 (no autocast, TF32 off, bank
and all 50k queries; scripts/must09_heal.py:755-770). Exact-invariance claims use fp32 (PREREG Section 1, with a
1e-4 near-tie allowance). Healed Mirror-exact: 100.000% fp32, 99.71 to 99.74% bf16 (results/MUST-09/summary.csv).

**Linear probe** (scripts/tier2_linear.py:12-41, 135). Features [CLS, mean of patch tokens] after the final norm
(registers excluded), 1,536 dims, bf16 forward, stored float16. For Mirror-exact the default "view a" is
[even CLS, mean of even patch tokens] (exactly invariant); `_fullpatch` rows use the full patch tokens. Multinomial
logistic regression on standardised features, loss = cross-entropy + wd/2 ||W||^2, full-batch L-BFGS (up to 1,000
iterations). wd from {1e-7, ..., 1e-2} plus half-decade neighbours of the best, chosen on a class-stratified 10%
split of the bank; refit on the full bank; top-1 on 50k val.

**Transfer** (scripts/tier2_transfer.py:20-44). Same features and probe on CIFAR-100, Flowers-102, Oxford-IIIT Pets,
DTD (HF copies; split identity marked (verify) in the script); wd chosen on 10% of the training split; MUST-11 ran the
original, healed Mirror-exact seed 0 and healed SVD-LLM seed 0.

**ADE20K** (scripts/tier2_ade20k.py:24-51, 83-128). Frozen backbone at 448 px (32 x 32 grid; Mirror-exact rebuilt
with a re-projected 32 x 32 grid); last-layer patch tokens after the final norm; head BatchNorm2d + 1 x 1 conv to 150
classes ("follows DINOv2 BNHead (verify)"); 8,000 iterations, batch 16, random 448 crops + flip; AdamW lr 1e-3, wd 1e-4
on the conv weight, 300 warm-up then cosine; sliding-window evaluation (stride 298) on 2,000 val images; mIoU.
Mirror-exact uses the full tokens (even head bypassed: rms then affine).
**Mask equivariance**: share of pixels with mask(mirror x) = mirror(mask x), fp32, all 2,000 val images,
flip-symmetric window layout. Raw head, and the symmetrised head (head(z) + head(T z))/2, z = affine(n),
T z = affine(n A): A = A* (= 2P - I) for Mirror-exact, and for other models the per-layer layer-12 operator A_L of
the X2 file. Only the Mirror-exact symmetrised head is exact by construction (99.9999%, 422 differing pixels).

**CUB orientation (CORE-06)** (scripts/core06_orientation.py:11-49). CUB-200-2011 official split, 224 crops, fp32
forward. Facing = sign(x_beak - x_tail) (parts 2 and 14) when both visible and more than 10% of box width apart.
Original model: u = final normalisation without affine; even = gamma (u (I + A_L)/2) + beta,
odd = gamma (u (I - A_L)/2), with A_L = A_layers[12] (per-layer, d- = 8). Healed Mirror-exact: even = its own feature,
odd = gamma (rms(t) - rms(t) P). Parts of the CLS and of the mean patch token. Logistic probes (tier2 code), wd from
{1e-5, ..., 1} by 5-fold CV on CUB train; species (200) and facing (2); tests on natural test images and on a
mirror-augmented set (each image plus its mirror, facing label flipped).

**ImageNet-V2 and Sketch (CORE-07)** (scripts/core07_shift.py:7-36). The ImageNet linear probe above applied
unchanged to ImageNet-V2 matched frequency (10k) and ImageNet-Sketch; drop = val top-1 minus shifted top-1; original,
3 healed Mirror-exact seeds, healed SVD-LLM seed 0. Sketch also reported with detected flipped and rotated
near-duplicates removed (heuristic thumbnail match).

**Flip averaging (MUST-10)** (scripts/must10_flipavg.py:7-30). f_avg(x) = (f(x) + f(mirror x))/2 on the CLS after the
final norm, before L2 normalisation (probe: the whole feature averaged); two passes, all parameters; compared with
healed Mirror-exact seed 0; paired bootstrap over the 50k images (10,000 resamples).

**Latency** (scripts/must09_heal.py:868-941; scripts/m2_latency.py:403-467). bf16 weights and inputs (random
224 px tensors), torch.compile (mode default, dynamic False), 20 warm-up runs, median of 100 runs timed with CUDA
events, batch 1, 64, 256, peak memory; NVIDIA H200. Variants: original, healed model as stored (dense projected
weights, same architecture as the original) and its even/odd form (mirror/efficient.py, dims padded to multiples of
8). Measured even/odd form is slower than the original (batch 1: 3.80 against 2.04 ms; results/MUST-09/latency.csv).

**Exactness audit (MUST-02)** (docs/exactness.md:125-135; scripts/must02_exactness.py). 1,000 val images, fp32
(TF32 off) and float64; per component rel_tensor_max = max over images ||T out(x) - out(Mx)|| / ||out(Mx)||.

## 8. Structure measurements (MUST-04, CORE-04; scripts/measure_groups.py, mirror/groups.py, mirror/group_stats.py)

- **Groups and token maps** (mirror/groups.py:7-14): LR (r, c) -> (r, G-1-c); UD (r, c) -> (G-1-r, c);
  R90 (r, c) -> (G-1-c, r) (counter-clockwise, torch.rot90); R180 (r, c) -> (G-1-r, G-1-c). Prefix tokens keep their
  index. One paired pass [x; g(x)] per generator over fit0 (2,000 images).
- **Operators per layer**: LR, UD, R180 by the involution fit of Section 1; R90 by orthogonal Procrustes on the
  centred subspace, then rho^4 = I by rounding the real-Schur rotation angles to multiples of 90 degrees.
- **Hidden permutations**: mutual matching of Section 2 for involutions; Hungarian on R for R90, keeping 1-, 2- and
  4-cycles (others become fixed points). Heads: the Section 3 pairing.
- **Breaking share per block** (mirror/groups.py:652-665): on the folded original weights with per-layer operators
  (fc1 against rho_b, fc2 against rho_(b+1)); s_mlp = (s_fc1 + s_fc2)/2. Activation-weighted share
  ||(W - W_eq) M^(1/2)||^2 / ||W M^(1/2)||^2 with M the uncentred input second moment on fit0 (mirror/variants.py:851).
- **Null** (scripts/measure_groups.py:491-525; mirror/groups.py:743-770): 20 seeded draws of random operators with the
  fitted character per layer (same d+, A 1 = 1) and a random hidden permutation with the same cycle type per block
  (plus random head maps); per-block null mean, and 2.5 and 97.5 percentiles over draws of a range mean.
- **Ratio to null** (scripts/measure_groups.py:763-799): mean over the range's blocks of s_mlp divided by the mean over
  the same blocks of the null-mean s_mlp. Attention shares are reported, not tested.
- **Twin fraction** (mirror/groups.py:498-522): share of hidden units in a 2-cycle of the matched permutation whose
  symmetrised twin correlation S_ij is at least 0.5 (also 0.6, 0.7, 0.8); mean over blocks of the range.
- **Twin-fraction null** (scripts/measure_groups.py:605-640): the same pipeline with the token alignment replaced by a
  random permutation of the 256 patch positions with the action's cycle type (UD: a fixed-point-free involution),
  prefix tokens fixed; 20 draws, blocks 1 to 6, default for UD only.
- **R90**: share of units in 4-cycles. Klein: two commuting involutions by coordinate ascent.
- **Ranges** (scripts/measure_groups.py:731-742): "1-6", "7-12" and first_half = blocks 1..L/2, second_half =
  blocks L/2+1..L (code 0..L/2-1 and L/2..L-1; DINOv2-L: 1-12 and 13-24).
- **Group surgery** (MUST-04 P4.3): zero-shot projection of the MLPs of blocks 1 to 6 with per-layer operators.

## 9. Baselines as implemented (mirror/baselines.py:470-582; allocation mirror/budget.py:415-685)

All compress the folded weights of the converted matrices; statistics from one pass of the fit images (no mirrors)
through the folded original model, all tokens, float64 sums, every block from the original's activations.
Grids (mirror/baselines.py:569-577) are searched on the msel set; full-size choices from gate_b.json.
- **svd**: truncated SVD per matrix, rank from the matched budget; no grid. Stored as balanced factors for healing.
- **svdllm** (SVD-LLM whitening only): L = chol(M + lam I), lam = damp x mean diag M, M the uncentred input moment
  (fc1: norm2 output; fc2: GELU output; q, k, v: norm1 output; o: attention context); W' = (U_r S_r V_r^T) L^-1.
  Grid {damp 1e-6} (one point). No sequential low-rank update.
- **asvd_scale** (ASVD): S = diag((mean |x_i|)^alpha), W' = trunc(W S) S^-1; alpha in {0.25, 0.5, 1.0}; chose 1.0.
- **slicegpt**: SliceGPT on the folded stream; with MLP and attention the residual stream is sliced to one d_s at every
  point (attention input, MLP input, span exit; PCA of the RMSNorm output "normed" or of the raw stream "stream");
  "per_block" bases with two d_s x d_s shortcut matrices per block counted, or one "shared" span basis; grid
  {per_block/normed, shared/normed, per_block/stream}; chose per_block/stream, d_s = 525 (6,615,000 shortcut
  parameters). Statistics from the original model, not propagated through sliced layers.
- **flap**: fluctuation score var(x_j) ||W[:, j]||^2 on fc2 inputs (neurons) and wo inputs (summed per head);
  removes neurons (smaller Linear) and whole heads (zeroed); bias compensation b += W[:, removed] mean(x_removed);
  allocation "global" (standardised scores, one ranking) or "uniform"; chose uniform.
- **zipit** (within one model): greedy merging of the most correlated GELU outputs until the width fits (fc1 rows and
  b1 averaged, fc2 columns summed); within-head merging of q channels (wq averaged, wk summed) and v channels
  (wv averaged, wo summed); alpha in {0, 0.1, 0.5}; chose 0.5.
- **pair_actcorr**: Mirror-exact machinery with the MLP pairing replaced by a matching on same-image fc1 correlation
  (diagonal zeroed, top-10 candidates), cut or greedily filled to the fitted pair count; attention keeps the fitted
  head maps (mirror/mirror_family.py:21-24, 171-227, 277-282). No grid.
- **pair_wcos**: pairing by plain cosine of the folded fc1 rows (top-10, blossom, n strongest pairs kept), tied with A*.
- **pair_random**: random pairing with the fitted numbers of pairs and self-twins, tied with A*.
- **random_structure** (mirror/m1.py:437-524): random involution with d+ = 700 and A 1 = 1, random neuron pairing with
  the fitted pair counts, random head maps with the same twin pairs, self-heads and self-head traces, random register
  pairing with the fitted number of pairs (none); seed 1 (MUST-07 fit seed 0 and MUST-09). Exactly invariant too.
- **mirror_local**: Section 1, ablation only.

## 10. Docs against code, and points to state carefully in the paper

1. Fit data: the plan's "refund from about 100 images" (PROJECT_NEXT_PHASE.md:23, 32) is not what the final model
   used: every structure is fitted on 2,000 train images (fit0), and full conversion also needs healing on 100k train
   images (+5k for LR choice); zero-shot full Mirror-exact is 53.88 val top-1 (results/MUST-07/rows_*.csv).
2. A*_12 has only 68 odd directions (d+ 700), far fewer than layer 1 (279). The full-model saving comes mostly from
   twin neurons; registers ended unpaired, so all four are forced even.
3. LayerNorm: plan 4.5 says A* fixes 1 with tied biases; code folds every LayerNorm into RMSNorm on a centred stream
   (docs/exactness.md:139-142).
4. Even head projects every token (plan: even part for the CLS only); dense tasks bypass it (docs/exactness.md:154-168).
5. Healing "only in its own structured parameters" (plan 4.8): Mirror-exact trains dense entries with AdamW and
   re-projects after each step (projected optimisation, not a reparameterisation); embeddings train for every method.
6. Targets differ by method: flip-averaged final targets for all; A*-symmetrised block targets only for the mirror
   family; raw block outputs for SVD and SVD-LLM, centred for SliceGPT (PREREG Section 9 item 4).
7. LR selection uses held-out distillation loss on 5k train images, not k-NN; all methods chose 1e-4 (the top of the
   grid); the SVD 2e-4 check trial would have won but was not used.
8. "3 seeds" are healing seeds over one fixed fit-seed-0 structure and one compressed start; structure variance is
   not in the healed error bars.
9. configs/must09.yaml defaults (micro_batch 32, `span_structure: null` -> M1 head file runs/m1/m1_structure.npz) differ
   from the runs (`--micro-batch 64`, MUST-07 canonical structure with head maps refitted on fit0).
10. The real SVD-LLM grid has one point (the SHIM in scripts/must07_compressors.py:223 listed two); SVD-LLM therefore had
    no hyperparameter search. MUST-07 rows record compressor_api "real".
11. Parameter fraction counts the hub's 37 x 37 position table and LayerNorm affines as dense; embedding savings are
    not credited; the healed checkpoint stores dense weights (no memory saving unless the even/odd form is used).
12. Speed: plan 4.5 suggests a cleaner speed-up; measured even/odd form is slower than the original at every batch
    (H200); GFLOPs fraction 0.774 is analytic.
13. Mirror-local attention uses A_out = A_b (M1 used A_(b+1)); Mirror-local is not folded and its embeddings are not
    projected; zero-shot full Mirror-local (62.10) beats Mirror-exact (53.88) at +7.5% parameters.
14. Symmetrised segmentation head for non-Mirror models uses the per-layer A_L (d- = 8), not A*; for the original it
    lowers mask equivariance (95.03 raw, 94.74 sym).
15. Unclear or unverified in code comments: ADE20K head as DINOv2 BNHead and its stride (verify); transfer splits
    (verify); SliceGPT shortcut layout against the paper (verify).
