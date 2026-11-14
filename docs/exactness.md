# Exact mirror equivariance of Mirror-exact (MUST-02)

This note states what Mirror-exact guarantees, sketches why, and lists for every component the condition on
its weights, why the relation passes through it, and where the code enforces it. It describes what the code
does (`mirror/m1.py`, `mirror/fold.py`, `mirror/surgery.py`, `mirror/attention.py`, `mirror/heal_vexact.py`);
where the plan's wording in PROJECT_NEXT_PHASE.md Section 4.5 differs, the code's version is given and the
difference is listed at the end. The measurement that checks it is `scripts/must02_exactness.py`.

Block numbers follow the plan (block 1 is code index 0).

## 1. Notation

- An image x gives a token matrix S(x) with one row per token: CLS first, then R register tokens (if the model
  has them), then the G x G patch grid in row-major order. Each row has d channels.
- M is the left-right mirror of the image.
- A* is the span operator: a symmetric orthogonal d x d matrix with A*^2 = I (an exact involution,
  `mirror.heal.exact_involution`). It acts on token rows from the right, x -> x A*.
- Pi is the token permutation that aligns the mirrored image's tokens with the original's: patch (r, c) and
  patch (r, G-1-c) swap, register j and register pi(j) swap (pi is an involution on the registers: pairs swap,
  the rest stay), CLS stays. In code: `mirror.m1.align_mirror_tokens` (and `token_index` in the audit script).
- For a layer output Y, the mirror relation with operator A* is

      Y(Mx) = Pi Y(x) A*          (for every token and every image x),

  that is, the mirrored image's tokens at aligned positions equal the original tokens times A*. For the
  hidden units of an MLP the relation is Z(Mx) = Pi Z(x) P_n with P_n a permutation of units, and for the
  queries, keys and values of attention it is head-wise (Section 3).

## 2. The proposition

**Proposition.** Let F be the Mirror-exact model with the whole model converted: `mirror.m1.build_vexact`
applied to blocks 1 to L with one operator A*, the fitted neuron pairings, head maps and register pairing,
and `even_head=True` (or a student healed from it by `scripts/heal_vexact.py`, whose constraints keep the
same conditions after every step). Then, in exact arithmetic, for every input x:

1. every component maps the mirrored input's representation to the transformed representation of the
   original: the stream entering every block, every norm output, branch output and block output satisfy
   Y(Mx) = Pi Y(x) A*; the hidden units satisfy Z(Mx) = Pi Z(x) P_n; per-head queries, keys and values
   satisfy the head relations of Section 3 and attention probabilities satisfy
   P_g(Mx) = Pi P_sigma(g)(x) Pi^T;
2. so the final tokens satisfy f(Mx) = T f(x), with T = Pi (A* acts as the identity after the even
   projection): the final-norm token map of the mirrored image is the original's with the patch grid flipped
   and paired registers swapped;
3. so the final CLS feature, which is the even part of the final stream followed by the final affine, is
   exactly invariant, f_CLS(Mx) = f_CLS(x), and k-NN on it gives the same prediction for x and Mx:
   100% flip consistency.

**Proof sketch (induction over components).** The relation holds for the tokens entering block 1 (embedding
rows of the table: patch embedding, positions, CLS, registers). Assume it holds for the stream x entering a
block. Each component in the table below maps an input that satisfies its input relation to an output that
satisfies its output relation; the attention branch and the MLP branch each map the A*-relation of their
input to the A*-relation of their output, and the residual additions add two terms that carry the same
A*, so the block output satisfies the A*-relation again. After the last block the final RMSNorm keeps the
relation, the even projection E = (I + A*)/2 turns it into invariance because A* E = E, and the per-channel
final affine keeps invariance. The CLS token is fixed by Pi, so its feature is invariant. Identical features
give identical cosine similarities to the bank, identical top-k sets and votes, hence identical predictions.

Nothing else in the network mixes tokens or channels in eval mode: dropout, drop path and patch drop are
inactive, there is no relative position term inside attention, and `mirror/fold.py` refuses models with
q/k norms, gated attention, SwiGLU MLPs, `norm_pre` or `fc_norm` (it raises `NotImplementedError`), so the
argument covers exactly the models the code converts.

## 3. Per-component conditions

Conventions of the code: torch weights act on column vectors (W is [out, in]); A* is symmetric, so x A* for
a row and A* x for a column are the same map. sigma is the head involution, B_h and C_h (head dimension
square, orthogonal) are the maps of head h, P_n (index array `perm`) the neuron involution.

| Component | Condition on the converted weights | Why the relation passes through | Where in the code |
|---|---|---|---|
| Patch embedding (flipped kernels) | flip_w(W) = A* W on the output channels, A* b = b (W: conv weight [d, 3, p, p], flip_w reverses the kernel width) | a patch of Mx at grid column G-1-c is the width-flipped patch (r, c) of x; convolving it with W equals convolving the original patch with flip_w(W) = A* W | `m1.project_embeddings`: W <- (W + A* flip_w(W))/2, b <- (b + A* b)/2 (orthogonal projection onto the condition) |
| Position embeddings | pos(r, G-1-c) = pos(r, c) A* | the patch token at (r, G-1-c) of Mx is (embedding + position) of (r, c) of x times A* | `m1.project_embeddings`: grid <- (grid + flip(grid) A*)/2. For DINOv2 the bicubic-interpolated grid of the evaluation size is first written into the model (`m1.materialise_pos_embed`), because the interpolation with `interpolate_offset` 0.1 is not flip-symmetric; the converted model is tied to that input size |
| CLS (even) | c A* = c for the CLS token and its position row | Pi keeps CLS at index 0, so CLS must map to itself times A* | `m1.project_embeddings`: c <- (c + c A*)/2 for `cls_token` and position row 0 |
| Registers (matched among themselves) | r_pi(j) = r_j A* (pairs swap, unpaired registers even), also for register position rows if the model has them | Pi swaps paired registers; registers are never matched to CLS or patches | pairing `pi`: `m1.match_registers` on alignment scores averaged over the span (`m1.RegisterScores`, `scripts/m1_attention.py` structure stage); projection `m1.project_embeddings`: r <- (r + r[pi] A*)/2. DINOv2 registers carry no position embedding; timm `reg_token` models have position rows, projected the same way |
| LayerNorm | none on A* beyond orthogonality: every LayerNorm is replaced by an RMSNorm without affine; gamma and beta are folded into the next linear layer (W diag(gamma), b + W beta) and the resulting biases are tied by the projections below | RMSNorm(x A*) = RMSNorm(x) A* for any orthogonal A*, token by token (the row norm is unchanged), so it also commutes with Pi | `fold.fold_layernorm_full` (`_fold_norm_into`). Every write into the stream is centred (C = I - 11^T/d), so the folded stream is mean-zero and RMSNorm equals LayerNorm. A* 1 = 1 holds (`m1.fit_span_operator` centres the cross-moments and the all-ones direction gets eigenvalue +1), so the projected writes stay centred and the converted model's norms are still LayerNorms; exactness itself does not need it |
| LayerScale (folded) | gamma folded into attn.proj and mlp.fc2 (W <- C diag(gamma) W, b <- C (gamma b)), LayerScale gamma set to 1 | a per-channel scale does not commute with A*, so it is folded before projecting; the module is then the identity | `fold._fold_scale_and_centre` |
| Q and K (same per-head map) | Wq^g A* = B_h Wq^h, Wk^g A* = B_h Wk^h, bq^g = B_h bq^h, bk^g = B_h bk^h for g = sigma(h), B_h orthogonal; B_g = B_h^T on twin pairs, B_h a symmetric involution on self-heads | q and k of head g on Mx at aligned tokens are B_h times those of head h on x; logits (B q)^T (B k) = q^T k, and the scale is the same for every head, so the logits of head g on Mx are the logits of head h on x with both token axes permuted by Pi | `attention.project_attn` (orbit averaging) inside `attention.apply_attention_surgery`; maps from `attention.fit_head_maps` (polar factors on twin pairs, `fit.fit_involution` on self-heads). Twin pairs are gauged (`attention.gauge_twin_heads`, `gauge=True`), so after conversion B = C = I on pairs |
| Softmax | none | softmax acts on each row over the keys; permuting the rows and the entries of every row by Pi permutes the probabilities the same way: softmax(Pi L Pi^T) = Pi softmax(L) Pi^T | (no parameters) |
| V and O | Wv^g A* = C_h Wv^h, bv^g = C_h bv^h, A* Wo^h = Wo^g C_h, A* bo = bo | the attention-weighted values of head g on Mx are C_h times those of head h on x; the output projection then gives sum_h Wo^sigma(h) C_h o_h + bo = A* (sum_h Wo^h o_h + bo) | `attention.project_attn` (Wv, bv, Wo columns, bo <- (bo + A* bo)/2) |
| Head permutation | sigma is an involution (heads map to themselves or swap in pairs) | the output projection sums over heads, so relabelling heads by sigma changes nothing | `attention.match_heads` (exact maximum-weight matching on pairing scores); `HeadMaps.check` |
| MLP | P_n W1 = W1 A*, P_n b1 = b1 (tied rows and biases), A* W2 = W2 P_n, A* b2 = b2 (tied columns) | fc1(x A*) = P_n fc1(x); GELU acts unit by unit, so it commutes with P_n; fc2 maps P_n g to A* fc2(g) | `surgery.project_mlp_two` with A_in = A_out = A* (`surgery.apply_surgery_two`); pairings from `matching.match_many` on neuron correlations of the fit set |
| Residual addition | the same A* on both branches and on the stream | x A* + f(x) A* = (x + f(x)) A* | `m1.build_vexact` passes `[A*] * k` as A_in and A_out to every block; with per-layer operators (Mirror-local) this step fails, which is why Mirror-local needs a basis change per block |
| Final norm and even projection | final norm = RMSNorm, then E = (I + A*)/2, then the final affine (gamma, beta) | RMSNorm keeps the A*-relation; y A* E = y E because A* E = E, so the projected tokens are invariant (T = Pi); a per-channel affine applied to every token keeps invariance. CLS is fixed by Pi, so f_CLS(Mx) = f_CLS(x). E acts on every token, patch tokens included; dense tasks bypass it (Section 6) | `fold.fold_layernorm_full(final_norm_affine=False)` keeps `model.norm = Sequential(rms, affine)`; `m1.install_even_head` inserts `m1.EvenProject` (runs outside autocast in its buffer's dtype) between them |

During healing (`mirror/heal_vexact.py`) the MLP and attention are re-projected with A* after every optimiser
step (`heal.MirrorMLPConstraint`, `heal_attn.MirrorAttnConstraint` with the gauged head maps), the
embeddings are re-projected (`heal_vexact.VExactEmbeddings`), the absorbed norm and LayerScale parameters are
frozen, the even head is a fixed buffer and the final affine is frozen. Each projection is computed in float64
and rounded once to the parameter dtype, so the conditions hold after training to that rounding.

## 4. Numerical caveat

The proposition is exact in exact arithmetic. In floating point two things limit it:

1. **Stored weights.** The projections are computed in float64 and rounded once to the parameter dtype, so in
   fp32 the conditions hold to a relative error of about 6e-8 per weight (fp32 rounding), not exactly.
2. **Summation order.** The mirrored forward pass does the same operations on permuted data: attention sums
   over keys in another order, the patch convolution sees flipped patches, and matrix kernels tile the
   permuted rows differently. Each operation therefore rounds differently for x and Mx, by a few units of the
   last place, and the differences accumulate mildly with depth.

Consequences:

- **fp32** (TF32 off): relative errors of order 1e-7 to 1e-5 are expected. The earlier fp32 stream probes in
  this repository report the largest relative error (max absolute error over max absolute entry, 64
  validation images) at the last layer of the 12-block conversion as 7.6e-6 for the zero-shot model
  (`runs/m1/m1_exactness.csv`, k = 12, layer 12) and 8.7e-6 for the healed model
  (`runs/m3_vexact_5090/k12/exactness.json`). MUST-02 measures every component, per image and per token.
  Tokens with small norms (for example rows of attention logits) give larger per-token relative errors than
  whole tensors do.
- **TF32** must be off: its 10-bit mantissa gives relative errors near 1e-3. Every audit script sets
  `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False` (the patch
  embedding is a convolution).
- **bf16** (8-bit mantissa) leaves differences of order 1e-2 between the features of x and Mx, so k-NN
  predictions near a tie can differ and the bf16 flip consistency falls below 100%. The pre-registered check is
  therefore fp32 on all 50,000 validation images; the bf16 value is reported beside it.
- **float64**: a model built in float64 (`--dtype float64` in the audit script, zero-shot) satisfies the
  conditions to float64 rounding, and the audit errors fall to rounding level (about 1e-15 on the tiny test
  model in `tests/test_must02.py`). A healed model stored in fp32 and cast to float64 keeps the fp32 rounding
  of its weights, so its float64 errors stay near the fp32 weight rounding.
- **Ties.** In exact arithmetic identical features give identical k-NN votes. In fp32 the features of x and
  Mx differ only by rounding, far below typical vote margins, but the 100% is checked on the full validation
  set, not assumed.

## 5. What the audit measures

`scripts/must02_exactness.py` runs [x; Mx] through the model with forward hooks at the output of every
component listed above and, for each, applies the expected action to the original half in float64 and
compares it with the aligned mirrored half: rel_tensor_max is the largest over images of
||T out(x) - out(Mx)|| / ||out(Mx)|| over the whole tensor of an image (the primary number), rel_token_max the
same per token (norm over every other axis of one token), rel_maxabs the M1 convention (largest absolute
error over largest absolute entry). Attention logits and probabilities are recomputed from the hooked q and k
in the model's dtype, because the fused attention kernel never materialises them. For a healed model loaded
through `mirror.heal_structured.load_healed_model`, the register pairing, neuron pairings and head maps are
read from the weights (`infer_structure`), and the residuals of that reading are recorded.

## 6. Where the code differs from the plan's wording

- **LayerNorm.** The plan says A* fixes the all-ones vector and gamma and beta are folded with tied biases.
  The code folds every LayerNorm completely into an RMSNorm on a centred stream, so exactness needs only that
  A* is orthogonal. A* 1 = 1 does hold, and it keeps the converted stream centred, so the RMSNorms remain
  LayerNorms of the converted model.
- **Registers.** "Matched among themselves" is an involution pi on registers chosen by maximum-weight matching
  of alignment scores averaged over the span's layers (margin 0 by default); unmatched registers are
  projected to the even subspace.
- **Position embeddings.** The projection acts on the grid materialised for the evaluation size, so the
  converted DINOv2 model is tied to that size. At another size (ADE20K at 448 px) the grid is materialised
  and projected again (`scripts/tier2_ade20k.py`, `scripts/tier2_linear.py`).
- **Head maps.** Twin pairs are gauged to B = C = I; self-heads keep their involutions.
- **Final affine.** It stays inside the model after the even projection. It is per channel, so it keeps
  invariance, but it would break A*-equivariance. The A*-relation of the full final tokens (all channels,
  before the even projection) therefore holds at the output of the final RMSNorm, which the audit reports
  as `final_rms`.
- **The even head projects every token.** `m1.install_even_head` puts the even projection between the final
  RMSNorm and the affine, so it acts on the CLS and on every patch token, and removes their odd
  (orientation) components. Section 4.5 of the plan keeps the even part for invariant use (the CLS) and
  the full tokens for equivariant use. The Tier-2 scripts therefore treat a model with an even head as
  follows:
  - ADE20K (`scripts/tier2_ade20k.py`) uses the full tokens by default: final RMSNorm, then affine, with the
    even stage bypassed (`--even-tokens` restores the projected tokens). Before the affine these tokens are
    exactly A*-equivariant, so the symmetrised head (head(z) + head(T z))/2 with T z = affine(n A*),
    A* = 2 P - I, is exactly mask-equivariant and the raw head is not.
  - The linear and transfer probes keep [even CLS, mean of even patch tokens] by default (view a, exactly
    invariant). `--full-patch-view` also reports [even CLS, mean of full patch tokens] (view b, rows
    `<variant>_fullpatch`). The odd part of that patch mean changes sign under a flip, so view b is not
    invariant.
  - The healed model's full tokens are meaningful: its distillation loss covers every block output with
    symmetrised targets, which keep odd components.
