# Novelty check, November 2026 (Symmetry Tax paper, CPAL 2027)

Run on 14 November 2026. Companion files: `docs/references_check.md` (citation check) and `docs/refs.bib`.

## Flags first: works that change how we must word the paper

No work found does what the paper does as a whole. Three works force more careful wording. None of them forces a claim to be dropped.

1. **Bökman and Kahl, NeurIPS 2023 (arXiv 2305.17017)** already merged flip-trained CNNs with their own mirrored, channel-permuted copy, including torchvision ResNet-50 on ImageNet (merged network 75.9% against 80.3% with REPAIR; Table 3). That merge is a zero-shot projection towards a flip-equivariant CNN, and their Table 1 notes that the equivariant VGG11 has half the parameters (4.61M against 9.23M), though they do not analyse this as a saving.
   - **Do not write:** "we are the first to find mirrored filters in ImageNet-trained networks" or "the first to project a trained network onto a mirror-equivariant form".
   - **Write instead:** "Bökman and Kahl (2023) showed that flip-trained CNNs are close to group-equivariant CNNs whose channels permute under a flip. We show that the same holds in vision transformers, where the residual stream needs a learned orthogonal operator with odd directions rather than a channel permutation; we count the duplicated parameters exactly, and we turn the structure into an exactly equivariant, smaller model."
2. **Rahman et al., "Tunable Soft Equivariance with Guarantees", CVPR 2026 (arXiv 2603.26657)** projects the weights of pretrained ViT-B/16, DINOv2-Base and ResNet-50 into a designed equivariant subspace and fine-tunes them, improving ImageNet accuracy and lowering equivariance error.
   - **Do not write:** "the first method that makes a pretrained foundation model equivariant by projecting its weights".
   - **Write instead:** our refund is (a) for the mirror group, with the latent operator and neuron pairing estimated per layer from about 100 images rather than fixed by the pixel grid, (b) exact rather than soft, (c) a parameter saving, and (d) compared against compressors.
3. **Shahverdi, Marchetti, Bökman, Kohn, ICML 2026 (arXiv 2601.21645)** gives a theorem explaining why equivariant structure appears in trained weights. In toy experiments (depth-4 MLPs and a one-layer attention autoencoder on CIFAR-10) it trains with left-right mirroring, fits a per-layer linear map by least squares, and reports lower layerwise equivariance error for left-right than for up-down mirroring. It also shows one pair of attention heads that swap.
   - **Effect on Claim 3:** the idea that the trained symmetry, and not another one, shows up layerwise has a toy precedent. Our controlled study is still new: ViTs at a real training scale, a dedicated up-down training arm, a parameter-level measure, and released models as the natural experiment. Cite it in the Claim 3 section, not only in related work.

One more point to watch: the Bökman group (Flopping for FLOPs, Quick ViTs, Platonic Transformers, Shahverdi et al., Ong and Bökman) is very active on exactly this topic, and Ong and Bökman write that the characterisation "remains open for an entire transformer". A paper from that group converting released ViTs to mirror-equivariant form is the most likely source of a scoop. Re-check their arXiv listings in the final week.

## How the search was done

- **OpenReview** (API search, all venues, about 40 queries): ICLR 2027 submissions (posted September 2026, visible as "ICLR 2027 Conference Submission"), ICML 2026, NeurIPS 2025 and 2026 (main and workshops), ICLR 2026, CVPR 2026, TMLR. Each query returned up to 25 forums; anything from 2024 onwards was screened by title, and likely matches were read from their abstracts.
- **arXiv** (export API, newest first, about 25 queries on abstracts and titles) covering 2024 to 13 November 2026, plus direct reads of the full HTML of the closest papers.
- **General web search** (several queries, including exact-phrase searches for "symmetry tax"). This stood in for Google Scholar, which was not queried directly.
- **Semantic Scholar** API: refused every request (HTTP 429), so it was **not** searched. DBLP refused scripted access too. Both gaps should be closed in the final-week re-run (Semantic Scholar with an API key, or by hand in a browser).
- Query topics: mirror-symmetric or twin neurons; learned and emergent equivariance in ViTs and foundation models; augmentation and weight-space symmetry; post hoc symmetrisation, equivariant fine-tuning and canonicalisation of pretrained models; equivariant or symmetry-aware pruning and compression; weight tying; "symmetry tax", "cost of invariance", "redundancy due to augmentation"; counting equivariant parameters with characters; test-time flip averaging against built-in equivariance; symmetry against low rank.

**Exact-phrase results.** "Symmetry tax": no machine-learning paper found. "Twin neurons" or "mirror twins": no machine-learning hits on arXiv (only physics). No paper found that counts free parameters of trained layers with a character formula. The terms are free to use.

## The six claims (from PROJECT_NEXT_PHASE.md Section 2)

- **C1 Real:** flip-trained ViTs store early features as working mirror twins, specific to the trained symmetry.
- **C2 Exact size:** a character formula counts the free parameters; trained networks sit near that subspace only for their trained group.
- **C3 Cause:** the training distribution's symmetry creates the tax (up-down training gives up-down twins).
- **C4 Invisible to compressors:** twin redundancy is not low rank or same-image correlation, and it stacks with low rank.
- **C5 Refund:** closed-form surgery on released models, exact equivariance, fewer parameters, fair comparison.
- **C6 Scope:** artefact tokens, model size, training recipes.

## Overlap summary

H = high, M = medium, L = low. A dash means no meaningful overlap.

| Work | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| Bökman and Kahl 2023 (NeurIPS) | H | L | L | - | M | L |
| Shahverdi et al. 2026 (ICML) | M | M | M | - | L | L |
| Rahman et al. 2026 (CVPR), soft equivariance by weight projection | L | L | - | - | M | - |
| Ong and Bökman 2026 (NeurIPS workshop, arXiv) | L | M | - | - | L | - |
| Equi-tuning, lambda-equitune (Basu et al. 2023); canonicalisation of large pretrained models (Mondal et al. 2023) | - | - | - | - | M | - |
| Flopping for FLOPs 2025, Quick ViTs 2025, Platonic Transformers 2026 (from scratch) | L | L | - | - | M | - |
| Nordenfors et al. (TMLR 2025); Gerken and Kessel 2024; Nordenfors and Flinth 2024 | - | L | M | - | - | - |
| Olah et al. 2020 (Distill) | M | - | - | - | - | - |
| Matsuoka 2026, effective dimension under symmetry augmentation (ICML workshop) | - | L | L | M | - | - |
| Layerwise and exotic equivariance in shallow networks (ICLR 2027 submission) | - | M | - | - | L | - |
| Berndt and Stühmer 2026 (ICML), projection-based approximate equivariance | - | L | - | - | L | - |
| Schlesinger et al. 2026, measuring learned equivariance (ICLR 2027 submission) | L | - | L | - | - | - |
| SEIS (ICLR 2027 submission) | L | - | L | - | - | - |
| Zhou, Foit, Hogg, Villar 2026; Foit, Hogg, Villar 2026 (post hoc group averaging) | - | - | - | - | L | - |
| Aiad 2026, low-rank networks and symmetry (ICML workshop) | L | - | - | L | - | - |
| Ziyin 2024 (ICML); Zhao, Walters, Yu 2025 (TMLR survey), parameter-space symmetry | - | - | - | L | - | - |
| Shang et al. 2016 (ICML), CReLU paired filters | L | - | - | - | L | - |
| Farzmahdi et al. 2024 (eLife) | L | - | L | - | - | L |
| Lawrence et al. 2026 (ICLR), distributional symmetry breaking | - | - | L | - | - | - |
| Santos-Escriche et al. 2026 (NeurIPS) | - | - | L | - | - | - |
| Gruver et al. 2023; Bruintjes et al. 2023; Bouchacourt et al. 2021; Lenc and Vedaldi 2015 | L | - | L | - | - | L |
| Flash EQ-Linear (ICLR 2027 submission); test-time symmetrisation (ICLR 2027 submission); post-training augmentation invariance (TMLR) | - | - | - | - | L | - |

No work scores High on C2, C3, C4 or C6. The only High is Bökman and Kahl on C1, and it is limited to CNNs.

---

## Close works in detail

### 1. Bökman and Kahl 2023, "Investigating how ReLU-networks encode symmetries" (NeurIPS 2023)
- **Link.** https://arxiv.org/abs/2305.17017
- **What it does.** Asks whether network equivariance implies layerwise equivariance. Conjectures that CNNs trained to be flip-invariant are layerwise equivariant with channel permutations, and relates this to the permutation conjecture for model merging. Measures a "GCNN barrier": the accuracy drop at the midpoint between a network and its flipped, channel-permuted copy. Flip-trained VGG11 (CIFAR-10) and ResNet-50 (ImageNet, two torchvision recipes) have small barriers; an invariance loss removes the barrier; self-supervised ResNet-50s (BYOL 0.118, DINO 0.166) have larger barriers.
- **Overlap.** C1 high (same phenomenon, CNNs only, measured through merge accuracy, no per-neuron count). C5 medium (zero-shot merge towards a GCNN, with REPAIR; no exact equivariance claimed, no parameter accounting, no compression comparison, no healing). C3 low (augmentation against invariance loss, but no wrong-group control). C6 low (self-supervised recipes are further from GCNN structure).
- **How we differ.** Transformers and foundation models; a residual-stream operator that is a symmetric orthogonal involution with odd directions (not a permutation); attention heads that map to themselves or swap; an exact parameter count from the character formula; up-down controls and controlled training; invisibility to rank compressors and stacking; exact conversion of released checkpoints with parameter savings, fair baselines and downstream tests.

### 2. Shahverdi, Marchetti, Bökman, Kohn 2026, "Identifiable Equivariant Networks are Layerwise Equivariant" (ICML 2026)
- **Link.** https://arxiv.org/abs/2601.21645
- **What it does.** Proves that an end-to-end equivariant network with identifiable parameters has an equivalent parameter choice whose layers are each equivariant for some latent action. Skip connections are explicitly not covered. Experiments: small MLPs (tanh and GELU) and a one-layer, 8-head attention autoencoder on CIFAR-10, trained with left-right mirroring augmentation plus a soft equivariance loss in the second half of training. A least-squares linear map is fitted per layer. Left-right equivariance error is lower than up-down (for example tanh f1 autoencoder 0.029 against 0.15). First-layer filters fall into mirrored copies, negated copies and symmetric filters. One example shows two heads swapping.
- **Overlap.** C1 medium (same structure at toy scale). C2 medium (trained networks are near the equivariant set for the trained group more than for another, at toy scale; no count). C3 medium (up-down used as a reference, not as a training arm; the equivariance loss is an extra push that our study does not use).
- **How we differ.** Real ViTs with skip connections and LayerNorm (outside their theory); our operator fit is restricted to symmetric orthogonal involutions; a dedicated up-down training arm and released models as controls; a parameter count; compression and refund.

### 3. Rahman, Lim Jun Hao, Jiang, Lim, Yeh 2026, "Tunable Soft Equivariance with Guarantees" (CVPR 2026)
- **Links.** https://arxiv.org/abs/2603.26657 ; https://openreview.net/forum?id=8iNWzp0kVZ
- **What it does.** Builds a per-layer subspace from the SVD of a Lie-algebra constraint matrix (the singular-value cut-off is the softness knob) and reparameterises the weights in that subspace. It applies to operations on structured grids (convolutions, patch embedding, positional encoding, flattened-grid fully connected layers) of pretrained ViT-B/16, DINOv2-Base and ResNet-50, which are then fine-tuned. The main group is 2D rotation (roto-reflections on MNIST). On ImageNet: ViT-B/16 81.67 to 82.28; DINOv2-Base 84.27 to 85.31; ResNet-50 77.91 to 77.96, with lower invariance error. It does not report the fully equivariant end point on ImageNet and does not claim parameter or FLOP savings.
- **Overlap.** C5 medium (pretrained weights projected into an equivariant subspace). Low elsewhere.
- **How we differ.** Exact mirror equivariance of the whole network rather than a softness knob on grid-structured layers; a latent operator and neuron pairing estimated from data per layer (they fix the representation from the pixel grid); a parameter refund; closed form before any healing; comparison with compressors. Their fine-tuned accuracy gain is a useful contrast: we report accuracy at matched or lower parameter counts.

### 4. Ông and Bökman 2026, multi-head self-attention under symmetry (arXiv 2608.30417; NeurIPS 2026 workshop)
- **Links.** https://arxiv.org/abs/2608.30417 ; https://neurips.cc/virtual/2026/157314
- **What it does.** Characterises equivariant multi-head self-attention: the group permutes clusters of heads, with matching constraints on QK and OV. For a group of order 2, a head is fixed or swapped with a partner. Any fixed exactly equivariant architecture covers at most one of very many components of the equivariant set. Artificial tasks show the predicted head structures arising under equivariance pressure.
- **Overlap.** C2 medium (the attention part of our count relies on this structure). C1 and C5 low.
- **How we differ.** We measure the head pairing in released models (130 of 144 heads map to themselves and 7 pairs swap in DINOv2-B, per PROJECT_NEXT_PHASE.md Section 3) and compile it into cheaper layers. Their expressivity result supports reading the pairing from data rather than fixing an architecture in advance; cite it that way.

### 5. Converting pretrained models by averaging or canonicalisation
- **Equi-tuning.** Basu et al., "Equi-Tuning: Group Equivariant Fine-Tuning of Pretrained Models", AAAI 2023. https://arxiv.org/abs/2210.06475. Averages features over group-transformed inputs and fine-tunes, giving an equivariant model close to the pretrained one.
- **Lambda-equitune.** Basu et al., "Efficient Equivariant Transfer Learning from Pretrained Models", NeurIPS 2023. https://arxiv.org/abs/2305.09900. Learns importance weights for the group-averaged features because plain averaging hurts zero-shot performance.
- **Canonicalisation.** Mondal, Panigrahi, Kaba, Rajeswar, Ravanbakhsh, "Equivariant Adaptation of Large Pretrained Models", NeurIPS 2023. https://arxiv.org/abs/2310.01647. Puts a canonicalisation network in front of a large pretrained model and addresses misalignment of canonical orientations with the training distribution.
- **Overlap.** C5 medium. These make a pretrained model equivariant at its output, at the cost of extra forward passes or an extra network. MUST-10 (test-time flip averaging) is the two-element case of equi-tuning's averaging and should be described as such.
- **How we differ.** We change the weights instead of the input or output path: one forward pass, fewer parameters, and equivariance holds for every intermediate feature (dense outputs included), not only at the output.

### 6. Mirror- and octic-equivariant ViTs trained from scratch
- Flopping for FLOPs (ICML 2025, https://arxiv.org/abs/2502.05169), Quick ViTs (arXiv 2505.15441) and Platonic Transformers (ICML 2026 per OpenReview, https://arxiv.org/abs/2510.03511).
- **What they do.** Build reflection-equivariant (or octic, or Platonic-group) transformers whose layers are block-diagonal in the irreducible components, roughly halving parameters and FLOPs for the mirror group, and train them from scratch on ImageNet-1K (DeiT III, and DINOv2 for Quick ViTs).
- **Overlap.** C5 medium (same algebra, same savings). C1 and C2 low.
- **How we differ.** We start from released checkpoints and need about 100 images plus a short heal, not full pre-training. Our even and odd split is the same block structure, so cite Flopping for FLOPs as the architecture we compile into.

### 7. Theory of augmentation and equivariance
- **Nordenfors, Ohlsson, Flinth, TMLR 2025** (https://openreview.net/forum?id=PTTa3U29NR). Under a compatibility condition and partial unitarity, the set of equivariant layers is invariant under the gradient flow of augmented training, and equivariant stationary points coincide with those of equivariant models. Equivariant points can be unstable under augmentation.
- **Gerken and Kessel, ICML 2024** (https://arxiv.org/abs/2403.03103). Deep ensembles trained with full augmentation are equivariant in the infinite-width limit at all training times; individual members are not.
- **Nordenfors and Flinth, arXiv 2410.01452.** Extends the ensemble result beyond the NTK limit.
- **"Equivariance and augmentation for Bayesian neural networks"** (anonymous ICLR 2027 submission, https://openreview.net/forum?id=PsqBNvnWwe; an arXiv version 2606.26273 appeared in web search, verify). Conditions for exact equivariance under variational inference, and symmetrisation techniques.
- **Overlap.** C3 medium: they explain why augmentation can produce equivariance, but either at the ensemble level or under idealised dynamics. None measures parameter-level structure in trained ViTs or runs a wrong-group control.

### 8. Measuring learned equivariance in representations (recent)
- **Schlesinger, Gatterbauer, Walters, "Measuring Learned Equivariance: An Information Theoretic Approach"**, arXiv 2610.17269 (https://arxiv.org/abs/2610.17269); also an anonymous ICLR 2027 submission (https://openreview.net/forum?id=G25kCPopTN) and a 2026 workshop version (TAG-DS). Uses mutual information and CKA to test whether original and transformed representations determine each other; augmentation usually increases consistency; on CIFAR-10 a CNN and a ViT learn C4-equivariance, which can weaken later in training, and show some equivariance without being trained for it.
- **SEIS, "Subspace-based Equivariance and Invariance Scores for Neural Representations"**, anonymous ICLR 2027 submission (https://openreview.net/forum?id=sJhKpdAEJl); arXiv 2602.04054 according to web search (verify). Label-free subspace scores separating equivariance from invariance; augmentation strengthens deeper layers; predicts accuracy retention under affine transformations.
- **Santos-Escriche, Engelmayer, Lin, Jegelka, "How Do Transformers Learn to Represent Symmetries?"**, NeurIPS 2026 (https://arxiv.org/abs/2610.10305). Vanilla transformers on point clouds with finite augmentation; ranks symmetry groups by learnability and finds interpretable mechanisms for invariance in trained models.
- **Overlap.** C1 and C3 low. These measure representations, not parameters, and none counts or removes duplicated parameters. They are the main body of "learned equivariance" work a CPAL reviewer will know from ICLR 2027; cite one or two to show we measure something different.

### 9. Symmetry, effective dimension and rank
- **Matsuoka 2026, "Effective Dimension Ratios under Symmetry Augmentation"**, HiLD workshop at ICML 2026 (https://openreview.net/forum?id=ceP3LrlVlz; author listed on OpenReview as "HikaruMatsuoka"). Using a Reynolds decomposition of the representation covariance, predicts the ratio of effective dimensions with and without symmetry augmentation for Z2, D2 and D4 on CIFAR-10 (median error 5.6% to 10%).
- **Aiad 2026, "Low-Rank Networks Recover Weight and Functional Symmetry Better"**, ICML 2026 workshop (https://openreview.net/forum?id=jC7nGjmgQ0). In toy low-rank random-feature networks, finds "mirror-paired first-layer atoms with matched outgoing weights".
- **Overlap.** C4 medium for Matsuoka: a reviewer could use it to argue that symmetry does show up in spectra. Our point is narrower: MUST-06 tests whether projection onto the symmetric form leaves the weight spectrum's effective rank nearly unchanged and whether twins are uncorrelated on the same image (both are pre-registered predictions, not yet results). Cite it in the C4 section and say how the two fit (representation covariance under augmentation against weight-level structure of a fixed model). Aiad is a toy precedent for mirror-paired weights; low overlap.

### 10. Equivariance constraints and projections at training time
- **"Layerwise and Exotic Equivariance Constraints in Shallow Neural Networks"**, anonymous ICLR 2027 submission (https://openreview.net/forum?id=DmheI1vMDA). A canonical form for shallow networks with GELU, SiLU and similar activations; characterises all equivariant shallow networks, including "exotic" ones that are not layerwise equivariant; tests the choice of structure in the MLP blocks of equivariant ViTs.
- **Berndt and Stühmer, "Approximate Equivariance via Projection-Based Regularisation"**, ICML 2026 (https://openreview.net/forum?id=Dly6gCtE4V). Splits each layer orthogonally into equivariant and non-equivariant parts and penalises the latter exactly.
- **Overlap.** C2 medium for the first: our twin and self-twin form is layerwise, and its result on exotic constraints bears on whether our count is the only way to be equivariant. Cite it if it is public by the deadline (it is anonymous now). Berndt and Stühmer's non-equivariant component is the same quantity as our breaking share, used as a training regulariser; cite it next to the breaking-share definition.

### 11. Post hoc group averaging
- **Zhou, Foit, Hogg, Villar, "Enforcing Symmetries on Pre-trained ML Models for Physics"**, arXiv 2611.08992 (6 November 2026). Group averaging (the Reynolds operator) as a post hoc projection of trained predictors, with extensions to non-compact groups; finds that models trained on invariant features match or beat projections of unconstrained ones.
- **Foit, Hogg, Villar, "Group Averaging for Physics Applications: Accuracy Improvements at Zero Training Cost"**, ML4PS 2026 workshop (https://openreview.net/forum?id=s7Uv5xTbV3).
- **Overlap.** C5 low. Useful for MUST-10: test-time flip averaging is the Reynolds projection of the predictor, with known non-increasing-risk guarantees under assumptions. Our weight-space projection gives the same guarantee without the extra pass.

### 12. Parameter-space symmetry (a different meaning of "symmetry")
- **Ziyin 2024, "Symmetry Induces Structure and Constraint of Learning"**, ICML 2024 (https://openreview.net/forum?id=7AF0AMI4AE). Mirror-reflection symmetries of the loss in parameter space lead to constraints such as low rank and sparsity.
- **Zhao, Walters, Yu, "Symmetry in Neural Network Parameter Spaces"**, TMLR survey (https://openreview.net/forum?id=jLpWq5QY6I).
- **Overlap.** C4 low, but a reviewer may confuse the two notions, since Ziyin links a symmetry to low rank. Add one sentence: our symmetry acts on the input data and induces structure in the weights; parameter-space symmetries leave the function unchanged and are a different object.

### 13. Other related work worth a line
- **Shang, Sohn, Almeida, Lee, ICML 2016 (CReLU)**, https://arxiv.org/abs/1603.05201. Lower-layer CNN filters form opposite-phase pairs, and CReLU exploits this with fewer parameters. An early case of pair redundancy refunded by architecture.
- **Farzmahdi et al., eLife 2024.** Reflection-equivariant intermediate features in CNNs pooled into mirror-symmetric viewpoint tuning; the biological link.
- **Lawrence et al., ICLR 2026**, https://arxiv.org/abs/2510.01349. A two-sample classifier test for how symmetric a dataset's distribution is. It could quantify our premise that natural images are roughly left-right symmetric but not up-down symmetric (CORE-01 rationale).
- **Brehmer, Behrends, de Haan, Cohen, "Does equivariance matter at scale?"**, TMLR, https://arxiv.org/abs/2410.23179. Augmentation closes the data-efficiency gap with enough epochs, while equivariant models win at each compute budget. Good framing for "learning a symmetry from data has a price".
- **Flash EQ-Linear** (anonymous ICLR 2027 submission, https://openreview.net/forum?id=x3dUiKEsSw). Exact acceleration of equivariant linear layers with group-wise DFTs and CUDA kernels; relevant to UP-05 (speed attempt).
- **Test-time symmetrisation scaling laws** (anonymous ICLR 2027 submission, https://openreview.net/forum?id=YXYps4MPae). Language models; only O(log|G|) rewritings needed. Not relevant for a two-element group.
- **Post-training augmentation invariance** (Eikenberry, Liu, Lee, TMLR; https://openreview.net/forum?id=Z4uUwU6zRe). Adapter on frozen DINO features to add invariance; output-level, no weight change.
- **Chiba et al. 2026** (arXiv 2607.03108). Symmetry readout from weights depends on positional encoding; relevant to UP-07 (pairing from weights).

## Overall risk assessment

**Risk: medium-low.** The combination is new: no work found measures mirror twins in vision transformer foundation models, counts them with a character formula, separates cause with an up-down training arm, shows that rank-based compressors miss the structure and that it stacks with low rank, and converts released ViTs into exactly mirror-equivariant, smaller models with fair baselines.

The realistic reviewer challenges are:
- "This is Bökman and Kahl (2023) for transformers, plus Flopping for FLOPs applied after training." The answer is the operator on the residual stream (not a permutation), the law, the cause, the invisibility and stacking result, and the exact conversion of released foundation models. These must be visible in the introduction, not only in related work.
- "Weight projection of pretrained models into equivariant subspaces exists (Rahman et al., CVPR 2026)." Answer: soft rotation constraints on grid-structured layers with fine-tuning, no parameter refund, versus exact mirror equivariance with a learned latent operator and a parameter refund.
- "Learned layerwise equivariance is explained by Shahverdi et al. (2026)." Answer: their theory excludes skip connections and their experiments are toy-scale; we supply the large-scale measurement and the controlled cause.

The scoop risk before 5 December is real but modest: the Bökman group has every ingredient. Check their arXiv listings and the OpenReview ICLR 2027 pool again in the week of 1 December.

## Works the paper must cite and discuss

In the introduction or in the section of the claim they touch:
1. Bökman and Kahl 2023 (C1, C5): the closest precedent.
2. Shahverdi et al. 2026 (C1 to C3): theory, and the toy left-right against up-down result.
3. Olah et al. 2020 (C1): naturally occurring mirrored features.
4. Rahman et al. 2026, CVPR (C5): weight projection of pretrained models.
5. Flopping for FLOPs 2025 and Quick ViTs 2025 (C5): the architecture we compile into, and the from-scratch alternative.
6. Equi-tuning (Basu et al. 2023) and Mondal et al. 2023 (C5, MUST-10): output-level conversion; flip averaging as the two-element case.
7. Ong and Bökman 2026 (C2, attention): head clusters, and that full transformers are open.
8. Nordenfors et al. (TMLR 2025) and Gerken and Kessel 2024 (C3, UP-02), with their exact conditions.
9. Matsuoka 2026 (C4): symmetry and effective dimension of representations.

In related work:
10. Gruver et al. 2023; Bruintjes et al. 2023; Bouchacourt et al. 2021; Lenc and Vedaldi 2015 (learned equivariance measured on activations).
11. Schlesinger et al. 2026 or SEIS (recent representation-level measures).
12. Platonic Transformers 2026 (from-scratch, weight sharing).
13. Berndt and Stühmer 2026 (equivariant and non-equivariant split of layers).
14. Zhou, Foit, Hogg, Villar 2026 (post hoc group averaging).
15. Ziyin 2024 or the Zhao, Walters, Yu survey (to separate data symmetry from parameter-space symmetry).
16. Shang et al. 2016 (paired filters refunded by architecture).
17. Farzmahdi et al. 2024 (biology).
18. The compression baselines (SliceGPT, SVD-LLM, ASVD, FLAP, ZipIt!, ToMe) and Darcet et al. (registers).

## Re-run before submission

- OpenReview: re-run the same queries limited to "ICLR 2027 Conference Submission" and check for withdrawals or public author names on the anonymous ones above.
- arXiv: new listings from 14 November to 4 December 2026 for "equivariant" with "pretrained", "mirror", "flip", "reflection", "compression" or "pruning"; the Bökman, Kahl, Nordström and Bekkers author pages.
- Semantic Scholar and Google Scholar by hand (both untested here).
