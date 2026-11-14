# Reference check for the Symmetry Tax paper (CPAL 2027)

Checked on 14 November 2026. BibTeX for every verified entry is in `docs/refs.bib`.

## How this was checked

- Every entry below was checked against a source opened in this session: the arXiv export API (title, full author list, dates, comments and journal-ref fields, abstract), arXiv abstract or HTML pages, ar5iv renderings of the full text, the OpenReview API (venue and decision fields), PMLR proceedings pages, CVF Open Access pages, OpenAlex (DOIs and page ranges), Distill, PubMed Central, Zenodo, the Caltech dataset page and the CMC GitHub repository.
- DBLP and the Semantic Scholar API refused scripted access (bot check and HTTP 429), so neither was used. Where a page range or DOI comes only from OpenAlex, this is said.
- "Matches" means our description in PROJECT_NEXT_PHASE.md or PROJECT_CONTEXT.md agrees with the source. Anything not confirmed is marked (verify).

## Mismatches that need action (summary)

1. **Ong and Bökman (arXiv 2608.30417) is not a NeurIPS 2026 main-track paper.** It was presented as a poster at the NeurIPS 2026 workshop "Symmetry and Geometry in Neural Representations", under its v1 title "No Equivariant Architecture Covers All Equivariant Attention". The current arXiv title (v2, 15 October 2026) is "Inherent Trade-offs when Enforcing Correct Data Symmetries in Multi-head Self-attention". PROJECT_CONTEXT.md Section 6 says "NeurIPS 2026". Fix to "NeurIPS 2026 workshop (NeurReps)" or cite as arXiv.
2. **Quick ViTs (arXiv 2505.15441) has no venue.** An earlier version, "Octic Vision Transformers: Quicker ViTs Through Equivariance", was submitted to ICLR 2026 and is listed on OpenReview as a rejected submission. Cite as an arXiv preprint. Its 1.15x to 1.22x figure is **training** throughput (Table 6), not inference.
3. **Gerken and Kessel (2024) does not say that symmetric parameters stay symmetric.** Their theorem is about the **ensemble mean output** in the infinite-width (neural tangent kernel) limit under full augmentation; individual members are generically not equivariant. PROJECT_NEXT_PHASE.md UP-02 cites it for "training on symmetric data keeps symmetric parameters symmetric". Drop it from that sentence or reword.
4. **"Nordenfors et al. 2024" is ambiguous and the year is off.** The statement we want ("the set of equivariant layers is invariant under the gradient flow for augmented models") is Theorem 3.7 of Nordenfors, Ohlsson and Flinth, "Optimization Dynamics of Equivariant and Augmented Neural Networks", published in **TMLR in January 2025** (arXiv 2303.13458, versions 3 to 5 in 2024). It holds only under a compatibility condition and partial unitarity, for gradient flow (and full-batch gradient descent) with augmentation averaged over the whole group; the paper says it does not extend to SGD. The other 2024 Nordenfors paper (with Flinth, arXiv 2410.01452) is about ensembles, not individual parameters.
5. **Bökman and Kahl (NeurIPS 2023) covers more than "small CNNs".** It also measures ResNet-50 on ImageNet (supervised torchvision recipes and self-supervised models including DINO and BYOL), and it merges each network with its mirrored, channel-permuted copy. That merge is a zero-shot projection towards a flip-equivariant CNN. Our delta in PROJECT_CONTEXT.md Section 6 understates their reach; see the novelty check.
6. **Lenc and Vedaldi (2015) do not report mirrored neuron copies.** They fit equivariance maps between layers and report how many AlexNet channels are approximately invariant to horizontal flips per layer. For HOG the flip map is an exact permutation. Reviewer question 1 in PROJECT_NEXT_PHASE.md Section 10 groups them with "mirrored neuron copies are known"; cite them for measuring equivariance with learned maps instead.
7. **Bruintjes et al. (2023): "smaller models learn more equivariance" is narrower than written.** It holds for **translation** equivariance when the width of WideResNet-40 is scaled on CIFAR-10, and the caption adds that the final amount is similar across widths. Horizontal flips were tried as an augmentation and had no significant effect on translation equivariance. Venue is a CVPR 2023 workshop (L3D-IVU), not the main conference. CORE-04 and the fallback table should say "for translation equivariance in CNNs".
8. **Bouchacourt et al. (2021) did not measure flip invariance.** "Invariance stems from the data" was shown for translation, scale, rotation, shear and appearance changes, for ResNet-18 and ViT. Flips appear only inside the standard augmentation. CORE-01's argument that LR twins may appear without augmentation uses this paper by analogy only; say so.
9. **Shahverdi et al. (ICML 2026) is closer to us than the table says.** Besides the theorem, its experiments train small MLPs and a one-layer attention autoencoder on CIFAR-10 with left-right mirroring, fit a per-layer linear map by least squares, report lower layerwise equivariance error for left-right than for up-down mirroring (up-down used as a reference), and show one pair of attention heads that swap under mirroring. See the novelty check.
10. **ASVD has no venue.** It was submitted to ICLR 2025 and is listed on OpenReview as rejected. Cite as an arXiv preprint.
11. **CMC (Tian et al.) was published at ECCV 2020**, although the repository still cites it as an arXiv 2019 preprint. Either is acceptable; the ECCV version is the archival one.

Everything else matches. Details follow.

---

## A. Learned and emergent equivariance

### Olah et al. 2020, Naturally Occurring Equivariance in Neural Networks
- **Reference.** Chris Olah, Nick Cammarata, Chelsea Voss, Ludwig Schubert, Gabriel Goh. "Naturally Occurring Equivariance in Neural Networks". *Distill*, 8 December 2020. DOI 10.23915/distill.00024.004. https://distill.pub/2020/circuits/equivariance/
- **What it shows.** In InceptionV1 trained on ImageNet, many neurons are transformed copies of one feature (rotated curve detectors; "horizontally flipped pairs become quite prevalent" in middle layers), and the article states that this reflects "a deeper symmetry that exists in the weights". It also shows left- and right-facing dog-head detectors being combined into a pose-invariant one.
- **Match.** Matches our use (mirrored neuron copies known qualitatively, in a CNN). It is qualitative and has no count, no cause test and no compression.

### Lenc and Vedaldi 2015, Understanding image representations by measuring their equivariance and equivalence
- **Reference.** Karel Lenc, Andrea Vedaldi. "Understanding Image Representations by Measuring Their Equivariance and Equivalence". CVPR 2015, pp. 991-999. DOI 10.1109/CVPR.2015.7298701. arXiv 1411.5908. https://openaccess.thecvf.com/content_cvpr_2015/html/Lenc_Understanding_Image_Representations_2015_CVPR_paper.html (extended version: IJCV 127(5):456-476, 2019, DOI 10.1007/s11263-018-1098-y, from OpenAlex).
- **What it shows.** Proposes learning equivariance maps (and stitching layers) between representations and applies them to HOG and AlexNet. For HOG, a horizontal flip acts as an exact permutation. For AlexNet, learning a flip map does not beat leaving features unchanged, and the share of approximately flip-invariant channels rises from about 54% (conv1) to 100% (conv5).
- **Mismatch.** It does not discuss mirrored filter pairs. See summary item 6.

### Bökman and Kahl 2023, Investigating how ReLU-networks encode symmetries
- **Reference.** Georg Bökman, Fredrik Kahl. "Investigating how ReLU-networks encode symmetries". NeurIPS 2023 (poster). arXiv 2305.17017. https://arxiv.org/abs/2305.17017 ; https://openreview.net/forum?id=8lbFwpebeu
- **What it shows.** Studies whether network equivariance implies layerwise equivariance, and conjectures that CNNs trained to be flip-invariant are layerwise equivariant with channel permutations. Defines a "GCNN barrier": the accuracy drop at the midpoint between a network and its flipped, channel-permuted copy. Flip-trained VGG11 (CIFAR-10) and ResNet-50 (ImageNet) are close to GCNN structure; self-supervised ResNet-50s (BYOL, DINO) are further away. For torchvision ResNet-50 (new recipe), the merged network reaches 75.9% against 80.3% with REPAIR (Table 3).
- **Mismatch.** See summary item 5. "Small CNNs" understates it.

### Shahverdi, Marchetti, Bökman, Kohn 2026, Identifiable Equivariant Networks are Layerwise Equivariant
- **Reference.** Vahid Shahverdi, Giovanni Luca Marchetti, Georg Bökman, Kathlén Kohn. "Identifiable Equivariant Networks are Layerwise Equivariant". ICML 2026 (poster; arXiv comment "Accepted at ICML 2026"; OpenReview venue "ICML 2026 regular"). arXiv 2601.21645. https://arxiv.org/abs/2601.21645 ; https://icml.cc/virtual/2026/poster/66515
- **What it shows.** If a network's end-to-end function is equivariant and its parameters are identifiable, there is a parameter choice with the same function whose layers are each equivariant under some action on the latent spaces. The authors present this as an explanation for equivariant structure appearing in trained weights. The paper states that it has "not covered skip connections". Experiments: depth-4 MLPs and a one-layer, 8-head attention autoencoder trained on CIFAR-10.
- **Match.** Venue, "small models trained from scratch" and "skip connections not covered" all match. See summary item 9 for what the description misses.

### Ong and Bökman 2026, multi-head self-attention under symmetry
- **Reference.** Tīkun Ông, Georg Bökman. "Inherent Trade-offs when Enforcing Correct Data Symmetries in Multi-head Self-attention". arXiv 2608.30417 (v1 31 August 2026 titled "No Equivariant Architecture Covers All Equivariant Attention"; v2 15 October 2026). Poster at the NeurIPS 2026 workshop "Symmetry and Geometry in Neural Representations" (https://neurips.cc/virtual/2026/157314). https://arxiv.org/abs/2608.30417
- **What it shows.** Complete characterisation of equivariant multi-head self-attention: the group can only act by permuting clusters of heads, with QK and OV matrices satisfying a matching constraint. For the two-element group, a pair of heads is either fixed or swapped. Any fixed architecture that enforces exact equivariance covers at most one of very many components of the equivariant set. Experiments are on artificial tasks only.
- **Match.** The quote is correct: "The same characterization question remains open for an entire transformer, or even just for a single transformer block." "Heads map to themselves or swap in pairs" is correct for a group of order 2. Venue is wrong; see summary item 1. The workshop page lists only "Tikun Ong" as author; the arXiv record lists both authors (verify the author line on the workshop page before citing that version).

### Chiba et al. 2026, symmetry readout from weights
- **Reference.** Naoya Chiba, Satoshi Sugiyama, Yuki Uranishi. "Observable- and Positional-Encoding-Dependent Symmetry Readout from Neural Network Weights". arXiv 2607.03108 (3 July 2026; no venue). https://arxiv.org/abs/2607.03108
- **What it shows.** For neural fields with positional encodings, the symmetry readable from weights is set by the trained parameters, the positional encoding and the readout observable, not by the true symmetry group. Tested on MLPs fitted to 2D signed distance functions.
- **Match.** Broadly matches "symmetry readout from weights for small MLPs on 2D shapes". The paper does not use the phrase "data-free"; say "post hoc readout from weights".

### Gruver et al. 2023, The Lie Derivative for Measuring Learned Equivariance
- **Reference.** Nate Gruver, Marc Finzi, Micah Goldblum, Andrew Gordon Wilson. "The Lie Derivative for Measuring Learned Equivariance". ICLR 2023. arXiv 2210.02984. https://arxiv.org/abs/2210.02984
- **What it shows.** Introduces the Lie derivative as an equivariance measure and applies it to 410 pretrained classifiers ("we evaluate 410 classification models"). Larger, more accurate models tend to be more equivariant, and much equivariance error comes from aliasing in pointwise non-linearities.
- **Match.** "410 models" is correct. Note that it covers continuous transformations (translation, rotation, shear and others) and measures activations, not weights; flips are not studied.

### Bruintjes et al. 2023, What Affects Learned Equivariance in Deep Image Recognition Models?
- **Reference.** Robert-Jan Bruintjes, Tomasz Motyka, Jan van Gemert. "What Affects Learned Equivariance in Deep Image Recognition Models?". CVPR 2023 Workshops (L3D-IVU), pp. 4839-4847. arXiv 2304.02628. https://openaccess.thecvf.com/content/CVPR2023W/L3D-IVU/html/Bruintjes_What_Affects_Learned_Equivariance_in_Deep_Image_Recognition_Models_CVPRW_2023_paper.html
- **What it shows.** Proposes a measure of learned equivariance in intermediate features, finds that learned translation equivariance correlates with ImageNet accuracy, and finds that augmentation, reduced capacity and convolution raise learned equivariance. CNNs show more translation and rotation equivariance than ViT and MLP-Mixer.
- **Mismatch.** See summary item 7.

### Bouchacourt et al. 2021, Grounding inductive biases in natural images
- **Reference.** Diane Bouchacourt, Mark Ibrahim, Ari S. Morcos. "Grounding inductive biases in natural images: invariance stems from variations in data". NeurIPS 2021. arXiv 2106.05121. https://arxiv.org/abs/2106.05121
- **What it shows.** "The training data itself is the main source of invariance, and data augmentation only further increases the learned invariances", shown for ResNet-18 and ViT on ImageNet.
- **Mismatch.** See summary item 8.

### Nordenfors, Ohlsson, Flinth (TMLR 2025), Optimization Dynamics of Equivariant and Augmented Neural Networks
- **Reference.** Oskar Nordenfors, Fredrik Ohlsson, Axel Flinth. "Optimization Dynamics of Equivariant and Augmented Neural Networks". *Transactions on Machine Learning Research*, 2025 (OpenReview publication date 16 January 2025, J2C certification). arXiv 2303.13458 (v5, 18 October 2024). https://openreview.net/forum?id=PTTa3U29NR
- **Exact statement.** Theorem 3.7, part 2: "If the embedding operators L and E are partially unitary, then ℰ is an invariant set of the gradient flow of Aug." Here ℰ is the set of equivariant layers and Aug is training with augmentation averaged over the group (Haar measure). Conditions: the compatibility condition (the orthogonal projections onto admissible and equivariant layers commute), partial unitarity, a compact group with unitary representations, equivariant non-linearities and an invariant loss. Remark 3.8 extends it to full-batch gradient descent but not to SGD. The paper also shows that equivariant stationary points can be unstable under augmented training while stable for equivariant models, and it gives no result that non-equivariant starting points converge to equivariant ones.
- **Mismatch.** See summary item 4. For UP-02 (symmetric start), write: "under idealised conditions (full group averaging, gradient flow), augmented training keeps equivariant parameters equivariant (Nordenfors et al., 2025); our setting (AdamW, minibatches, LayerNorm) is outside those conditions, so UP-02 tests it empirically."
- **Related.** Oskar Nordenfors, Axel Flinth. "Ensembles provably learn equivariance through data augmentation". arXiv 2410.01452 (v2 18 December 2025; no venue). Extends Gerken and Kessel beyond the NTK limit; results are about the ensemble, not individual networks.

### Gerken and Kessel 2024, Emergent Equivariance in Deep Ensembles
- **Reference.** Jan E. Gerken, Pan Kessel. "Emergent Equivariance in Deep Ensembles". ICML 2024 (oral), PMLR 235:15438-15465. arXiv 2403.03103. https://proceedings.mlr.press/v235/gerken24a.html
- **Exact statement.** "Deep ensembles become equivariant for all inputs and at all training times by simply using data augmentation." Theorem 5.3 holds in the infinite-width (NTK) limit for a finite group, with full augmentation over the orbit and MSE loss; the ensemble mean satisfies f̄ₜ(ρX(g)x) = ρY(g) f̄ₜ(x). Individual members "will generically not be equivariant". Finite-width corrections are left out of scope.
- **Mismatch.** See summary item 3.

### Farzmahdi et al. 2024, mirror-symmetric viewpoint tuning (eLife, PMC11142642)
- **Reference.** Amirhossein Farzmahdi, Wilbert Zarco, Winrich A. Freiwald, Nikolaus Kriegeskorte, Tal Golan. "Emergence of brain-like mirror-symmetric viewpoint tuning in convolutional neural networks". *eLife* 13:e90256, 2024. DOI 10.7554/eLife.90256. https://pmc.ncbi.nlm.nih.gov/articles/PMC11142642/
- **What it shows.** Mirror-symmetric viewpoint tuning, as in macaque face patch AL, appears in the fully connected and pooling layers of CNNs trained on object recognition, even without faces in training. The proposed mechanism: discriminating bilaterally symmetric objects produces reflection-equivariant intermediate features, which are then pooled into reflection-invariant ones.
- **Match.** Matches. Two notes for the discussion: flip augmentation is not named as a cause (the CIFAR-10 networks were trained without augmentation), and ViT was added to the analyses during review; one reading of the text suggests ViT does not show the same layer profile (verify before quoting).

## B. Equivariant ViTs trained from scratch

### Flopping for FLOPs (Bökman, Nordström, Kahl, ICML 2025)
- **Reference.** Georg Bökman, David Nordström, Fredrik Kahl. "Flopping for FLOPs: Leveraging Equivariance for Computational Efficiency". ICML 2025 (spotlight poster), PMLR 267:4823-4838. arXiv 2502.05169. https://proceedings.mlr.press/v267/bokman25a.html
- **What it shows.** Mirror-equivariant networks whose features split into mirror-symmetric and mirror-antisymmetric parts, so linear layers become block-diagonal with about half the FLOPs. ViTs are trained on ImageNet-1K with the DeiT III recipe; no pretrained models are converted.
- **Numbers.** Table 1 matches PROJECT_CONTEXT.md exactly: E(ViT-H) 84.4% against ViT-H 84.6%, at 87.3 against 168.0 GFLOPs (316.1M against 632.1M parameters). Also in Table 1: the hybrid H(ViT-H), with the first half of the blocks equivariant, reaches 85.0% at 127.6 GFLOPs.

### Quick ViTs (Nordström, Edstedt, Kahl, Bökman)
- **Reference.** David Nordström, Johan Edstedt, Fredrik Kahl, Georg Bökman. "Quick ViTs: Speeding up Vision Transformers through Equivariance". arXiv 2505.15441 (v5, 6 July 2026; earlier titles "Stronger ViTs With Octic Equivariance" and "Octic Vision Transformers: Quicker ViTs Through Equivariance"). https://arxiv.org/abs/2505.15441
- **What it shows.** Octic (D8: 90-degree rotations and reflections) equivariant linear layers with 5.33x fewer FLOPs than ordinary linear layers; hybrid models with the first half of the blocks equivariant (k = l/2) trained with DeiT III and DINOv2 objectives on ImageNet-1K.
- **Numbers.** Match: H8(ViT-H/14) 85.0% against 84.6% at 102.3 against 167.8 GFLOPs (Table 3). The 1.15x and 1.22x speed-ups are training throughput (Table 6: I8(ViT-H) on DeiT III, H8(ViT-H) on DINOv2). Inference throughput in Table 3 is 657 to 660 against 569 images per second (about 1.15x).
- **Mismatch.** No venue; see summary item 2.

## C. Compression baselines

### SliceGPT
- **Reference.** Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman. "SliceGPT: Compress Large Language Models by Deleting Rows and Columns". ICLR 2024. arXiv 2401.15024. https://arxiv.org/abs/2401.15024
- **What it shows.** Post-training compression that deletes rows and columns after an orthogonal change of basis, using "computational invariance" of transformers; removes up to 25% of parameters of large LLMs.
- **Match.** Section 3.2: LayerNorm is converted to RMSNorm by absorbing the mean subtraction into the preceding output matrices and folding the LayerNorm scale into the following input matrices. The slicing basis is the eigenvectors of the uncentred per-block covariance of calibration activations, computed sequentially on the already-transformed network. Section 3.2 does not say how the LayerNorm bias is handled (verify in the appendix or code before implementing).

### SVD-LLM
- **Reference.** Xin Wang, Yu Zheng, Zhongwei Wan, Mi Zhang. "SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression". ICLR 2025. arXiv 2403.07378. https://arxiv.org/abs/2403.07378
- **What it shows.** Truncation-aware whitening so that each truncated singular value maps directly to compression loss, plus a sequential low-rank update of the compressed weights.
- **Match.** The whitening matrix is the Cholesky factor of XXᵀ for the layer's input activations (Section 3.1, Algorithm 2), as MUST-07 says. Default calibration: 256 WikiText-2 samples.

### ASVD
- **Reference.** Zhihang Yuan, Yuzhang Shang, Yue Song, Dawei Yang, Qiang Wu, Yan Yan, Guangyu Sun. "ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models". arXiv 2312.05821 (v5, 28 August 2025). https://arxiv.org/abs/2312.05821
- **What it shows.** Scales the weight matrix by the activation distribution before SVD so that activation outliers are absorbed, with per-layer rank calibration; training-free.
- **Mismatch.** No venue; see summary item 10.

### FLAP
- **Reference.** Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang. "Fluctuation-based Adaptive Structured Pruning for Large Language Models". AAAI 2024, 38(10):10865-10873. DOI 10.1609/aaai.v38i10.28960. arXiv 2312.11983. https://arxiv.org/abs/2312.11983
- **What it shows.** Retraining-free structured pruning with a fluctuation-based importance metric, standardised scores for a global structure search, and bias terms that compensate for removed channels.
- **Match.** Matches "activation-fluctuation importance with bias compensation". (OpenAlex lists the second and third authors differently; the names above are from arXiv.)

### ZipIt!
- **Reference.** George Stoica, Daniel Bolya, Jakob Bjorner, Pratik Ramesh, Taylor Hearn, Judy Hoffman. "ZipIt! Merging Models from Different Tasks without Training". ICLR 2024 (poster). arXiv 2305.03053. https://openreview.net/forum?id=LEYUkvdUhq
- **What it shows.** Merges models trained on different tasks by allowing features to be merged within each model as well as across models (the "zip" operation), using feature correlations.
- **Match.** Matches its use as the source of within-model correlation merging.

### ToMe, Token Merging
- **Reference.** Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman. "Token Merging: Your ViT But Faster". ICLR 2023 (oral). arXiv 2210.09461. https://arxiv.org/abs/2210.09461
- **What it shows.** Merges similar tokens inside a ViT with a fast matching algorithm, about 2x throughput at a 0.2 to 0.3 point accuracy drop on large models, without training.
- **Match.** Matches its use in UP-06.

## D. Models, datasets and tools

### Darcet et al., Vision Transformers Need Registers
- **Reference.** Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. "Vision Transformers Need Registers". ICLR 2024 (oral). arXiv 2309.16588. https://openreview.net/forum?id=2dnO3LLiJ1
- **What it shows.** High-norm artefact tokens appear in low-information background regions of supervised and self-supervised ViTs and are repurposed for internal computation; extra register tokens remove them and smooth feature maps.
- **Match.** Matches.

### Tian et al., Contrastive Multiview Coding (ImageNet-100)
- **Reference.** Yonglong Tian, Dilip Krishnan, Phillip Isola. "Contrastive Multiview Coding". ECCV 2020, LNCS, pp. 776-794, DOI 10.1007/978-3-030-58621-8_45 (from OpenAlex). arXiv 1906.05849 (2019). https://arxiv.org/abs/1906.05849
- **ImageNet-100 class list.** The list is `imagenet100.txt` in https://github.com/HobbitLong/CMC (README: "May 22, 2020 - ImageNet-100 list uploaded"). The file has 100 unique WordNet IDs, first entries n02869837, n01749939, n02488291. SHA-256 of the copy fetched today: 79f3f1552ccc478ce24bd934b4573625db584ddc1f4643ae23c994a94cb11397. Use this file for the controlled-training data and log its hash.

### DINOv2
- **Reference.** Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, and 21 others. "DINOv2: Learning Robust Visual Features without Supervision". *Transactions on Machine Learning Research*, 2024. arXiv 2304.07193. https://openreview.net/forum?id=a68SUt6zFt

### DeiT III
- **Reference.** Hugo Touvron, Matthieu Cord, Hervé Jégou. "DeiT III: Revenge of the ViT". ECCV 2022, LNCS 13684, pp. 516-533. DOI 10.1007/978-3-031-20053-3_30. arXiv 2204.07118.

### MAE
- **Reference.** Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick. "Masked Autoencoders Are Scalable Vision Learners". CVPR 2022. DOI 10.1109/CVPR52688.2022.01553. arXiv 2111.06377.

### CLIP
- **Reference.** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. "Learning Transferable Visual Models From Natural Language Supervision". ICML 2021, PMLR 139:8748-8763. arXiv 2103.00020. https://proceedings.mlr.press/v139/radford21a.html
- **Recipe sentence (for Section 4.2).** Section 2.3: "A random square crop from resized images is the only data augmentation used during training." This confirms the note in PROJECT_NEXT_PHASE.md Section 4.2.

### SigLIP 2
- **Reference.** Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, Xiaohua Zhai. "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features". arXiv 2502.14786, 2025 (no venue found).

### DINO v1
- **Reference.** Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin. "Emerging Properties in Self-Supervised Vision Transformers". ICCV 2021. DOI 10.1109/ICCV48922.2021.00951. arXiv 2104.14294.

### EVA-02
- **Reference.** Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, Yue Cao. "EVA-02: A Visual Representation for Neon Genesis". *Image and Vision Computing* 149:105171, 2024. DOI 10.1016/j.imavis.2024.105171. arXiv 2303.11331.

### ConvNeXt
- **Reference.** Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie. "A ConvNet for the 2020s". CVPR 2022. arXiv 2201.03545.

### ResNet
- **Reference.** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition". CVPR 2016, pp. 770-778. DOI 10.1109/CVPR.2016.90. arXiv 1512.03385.

Training recipes (whether each model used flips) were not checked here; that is the Section 4.2 task in `configs/models/*.yaml`. Only the CLIP sentence above was checked.

### Roy and Vetterli 2007, effective rank
- **Reference.** Olivier Roy, Martin Vetterli. "The Effective Rank: A Measure of Effective Dimensionality". 15th European Signal Processing Conference (EUSIPCO), Poznań, Poland, September 2007, pp. 606-610. https://zenodo.org/records/40328 (DOI 10.5281/zenodo.40328).
- **What it shows.** Defines effective rank, a real-valued extension of rank computed from the entropy of the singular value spectrum (the usual form is the exponential of the Shannon entropy of the normalised singular values; check the normalisation against the PDF before writing the formula in the paper).
- **Match.** Matches its use in MUST-06 and CORE-05 ("effective rank (entropy)"). The page range comes from OpenAlex and the EURASIP listing.

### Recht et al. 2019, ImageNet-V2
- **Reference.** Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, Vaishaal Shankar. "Do ImageNet Classifiers Generalize to ImageNet?". ICML 2019, PMLR 97:5389-5400. arXiv 1902.10811. https://proceedings.mlr.press/v97/recht19a.html

### Wang et al. 2019, ImageNet-Sketch
- **Reference.** Haohan Wang, Songwei Ge, Eric P. Xing, Zachary C. Lipton. "Learning Robust Global Representations by Penalizing Local Predictive Power". NeurIPS 2019. arXiv 1905.13549 (journal-ref "NeurIPS 2019"). This paper introduces ImageNet-Sketch.

### CUB-200-2011
- **Reference.** Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, Serge Belongie. "The Caltech-UCSD Birds-200-2011 Dataset". Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. https://www.vision.caltech.edu/datasets/cub_200_2011/
- **Facts.** 200 classes, 11,788 images, 15 part locations per image. The dataset page does not list part names; read the beak and tail part IDs from the dataset's own parts file before CORE-06 (as the plan already says).

### ADE20K
- **Reference.** Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, Antonio Torralba. "Scene Parsing Through ADE20K Dataset". CVPR 2017, pp. 633-641 (CVF Open Access pages). https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Scene_Parsing_Through_CVPR_2017_paper.html
- **Journal version.** Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, Antonio Torralba. "Semantic Understanding of Scenes Through the ADE20K Dataset". IJCV 127(3):302-321, 2019. DOI 10.1007/s11263-018-1140-0. arXiv 1608.05442.

## E. Status table

| Entry | Exists | Venue confirmed | Description matches |
|---|---|---|---|
| Olah et al. 2020 | Yes | Distill | Yes |
| Lenc and Vedaldi 2015 | Yes | CVPR 2015 | Partly (item 6) |
| Bökman and Kahl 2023 | Yes | NeurIPS 2023 | Understated (item 5) |
| Bruintjes et al. 2023 | Yes | CVPR 2023 Workshops | Partly (item 7) |
| Bouchacourt et al. 2021 | Yes | NeurIPS 2021 | Partly (item 8) |
| Nordenfors et al. | Yes | TMLR 2025 | Conditional (item 4) |
| Gerken and Kessel 2024 | Yes | ICML 2024 | No for UP-02 (item 3) |
| Flopping for FLOPs | Yes | ICML 2025 | Yes, numbers exact |
| Quick ViTs | Yes | None (arXiv) | Numbers yes; speed-up is training (item 2) |
| Shahverdi et al. 2026 | Yes | ICML 2026 | Yes, but closer to us than stated (item 9) |
| Ong and Bökman 2026 | Yes | NeurIPS 2026 workshop | Venue wrong (item 1) |
| Chiba et al. 2026 | Yes | None (arXiv) | Yes |
| Gruver et al. 2023 | Yes | ICLR 2023 | Yes (410 models) |
| Darcet et al. | Yes | ICLR 2024 | Yes |
| SliceGPT | Yes | ICLR 2024 | Yes |
| SVD-LLM | Yes | ICLR 2025 | Yes |
| ASVD | Yes | None (arXiv) | Yes (item 10) |
| FLAP | Yes | AAAI 2024 | Yes |
| ZipIt! | Yes | ICLR 2024 | Yes |
| ToMe | Yes | ICLR 2023 | Yes |
| CMC (ImageNet-100) | Yes | ECCV 2020 | Yes (item 11) |
| DINOv2, DeiT III, MAE, CLIP, SigLIP 2, DINO, EVA-02, ConvNeXt, ResNet | Yes | See above | Not described in the plan beyond names |
| eLife mirror-symmetric tuning | Yes | eLife 2024 | Yes |
| Roy and Vetterli 2007 | Yes | EUSIPCO 2007 | Yes |
| ImageNet-V2, ImageNet-Sketch | Yes | ICML 2019, NeurIPS 2019 | Yes |
| CUB-200-2011 | Yes | Caltech tech report | Yes |
| ADE20K | Yes | CVPR 2017 (IJCV 2019) | Yes |
