# Model training recipes: horizontal flips

Checked on 2026-11-14 for PROJECT_NEXT_PHASE.md Section 4.2 and CORE-04. Each row was read from the linked paper, code or model card in this session; nothing is from memory. The full records (architecture, other augmentations, certainty, notes) are in `configs/models/<short_name>.yaml`. Revisions are Hugging Face commit hashes from the HF API unless marked "code"; for torch.hub models the weights are fixed by URL and the hash pins the hub code.

## Summary table

| Model | ID | Revision | Flips | Quote | Source |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 + registers | torch.hub `facebookresearch/dinov2`, `dinov2_vitb14_reg` | code `7764ea0f` | yes | `transforms.RandomHorizontalFlip(p=0.5),` | [augmentations.py L43-L60](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/data/augmentations.py#L43-L60) |
| DINOv2 ViT-B/14 plain | `dinov2_vitb14` | code `7764ea0f` | yes | same line | same file |
| DINOv2 ViT-S/14 + registers | `dinov2_vits14_reg` | code `7764ea0f` | yes | same line (distilled with "the same training loop with a few exceptions") | same file; arXiv 2304.07193 Section 5 |
| DINOv2 ViT-L/14 + registers | `dinov2_vitl14_reg` | code `7764ea0f` | yes | same line | same file |
| DINOv2 ViT-g/14 + registers | `dinov2_vitg14_reg` (not yet in MODEL_REGISTRY) | code `7764ea0f` | yes | same line | same file |
| DeiT-III ViT-B/16 | timm `deit3_base_patch16_224.fb_in1k` | `19c7b54d` | yes | "In addition to these 3 aumgnentations choices, we include the common color-jitter and horizontal flip." | [arXiv 2204.07118](https://arxiv.org/abs/2204.07118) Section 3.2; [augment.py L104-L108](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/augment.py#L104-L108) |
| MAE ViT-B/16 | timm `vit_base_patch16_224.mae` | `a64dffd1` | yes | "Our MAE works well using cropping-only augmentation, either fixed-size or random-size (both having random horizontal flipping)." | [arXiv 2111.06377](https://arxiv.org/abs/2111.06377) Section 4.1; [main_pretrain.py L123-L127](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py#L123-L127) |
| SigLIP 2 ViT-B/16 | timm `vit_base_patch16_siglip_224.v2_webli` | `4c3661e5` | unknown | "We use the original image for computing the SigLIP and LocCa losses from the previous section and apply the additional losses on additional augmented views." | [arXiv 2502.14786](https://arxiv.org/abs/2502.14786) Section 2.3 (flips never mentioned; SILC, which it follows, describes crops only) |
| OpenAI CLIP ViT-B/16 | timm `vit_base_patch16_clip_224.openai` | `977e3dd0` | no | "A random square crop from resized images is the only data augmentation used during training." | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) Section 2.3 |
| DINO v1 ViT-B/16 | timm `vit_base_patch16_224.dino` | `4735e392` | yes | `transforms.RandomHorizontalFlip(p=0.5),` | [main_dino.py L419-L460](https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py#L419-L460) |
| DINOv3 ViT-B/16 (gated) | HF `facebook/dinov3-vitb16-pretrain-lvd1689m`; timm `vit_base_patch16_dinov3.lvd1689m` | `5931719e`; timm `c6a5fb7d` | no (probable) | `horizontal_flips: false` | [ViT-L distillation config L211](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/configs/train/dinov3_vitl16_lvd1689m_distilled.yaml#L211), also all three 7B teacher configs |
| EVA-02 ViT-B/14 (MIM, IN-21K) | timm `eva02_base_patch14_224.mim_in22k` | `0ecfe3fa` | yes | `transforms.RandomHorizontalFlip(p=0.5),` | [EVA-02/asuka/datasets.py L34-L53](https://github.com/baaivision/EVA/blob/5623d08bb08a8268c7bc18e1d9b8b4bd651fe4d9/EVA-02/asuka/datasets.py#L34-L53) |
| ConvNeXt-B (ImageNet-1k) | timm `convnext_base.fb_in1k` | `28a94f63` | yes (implicit) | `hflip=0.5,` (timm 0.3.2 default, not overridden) | [ConvNeXt datasets.py L50-L73](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/datasets.py#L50-L73); timm 0.3.2 wheel `transforms_factory.py` L48, L73-L74 |
| ResNet-50 (ImageNet-1k, torchvision V1) | timm `resnet50.tv_in1k` = torchvision `IMAGENET1K_V1` | `78f3ecfd` | yes | `transforms.append(T.RandomHorizontalFlip(hflip_prob))` | [presets.py L17-L48](https://github.com/pytorch/vision/blob/4260301135e34180d865426765487113f84e7707/references/classification/presets.py#L17-L48) |

Full hashes are in the YAML files.

## Which models give a clean "no flips" control for CORE-04

CORE-04 predicts that "models whose recipe is verified to have no flips have a lower LR tax than every flip-trained model of the same size". At ViT-B size there are two candidates and no clean one.

OpenAI CLIP ViT-B/16 has the strongest written evidence: the paper says a random square crop is the only augmentation. Certainty about the recipe is medium to high, because the training code was never released and the sentence cannot be checked against code. DINOv3 ViT-B/16 has the strongest code evidence: every released DINOv3 config for the 7B teacher (pre-training, Gram anchoring, high-resolution adaptation) and for the distilled ViT-L sets `horizontal_flips: false`, against a code default of `true`. Its own ViT-B distillation config is not public, so "no flips" for ViT-B is inferred from its teacher and its sibling student (medium to high). DINOv3 against DINOv2 ViT-B is the closest same-family contrast available, but it still changes patch size (16 against 14), position encoding (RoPE), data (LVD-1689M), Gram anchoring and distillation from a 7B teacher. CLIP changes data, objective and activation. SigLIP 2 is unknown and should sit outside both groups.

Two caveats limit how strongly either control can be read. First, "no flip augmentation" does not mean "no mirror symmetry in the data": web and curated photos show objects facing both ways, so the training distribution may be close to mirror symmetric anyway, and a lower tax would then point to the augmentation itself rather than to the data. Second, with one or two no-flip models of different families, a lower LR tax cannot be attributed to the recipe alone. The clean test of the recipe claim is CORE-01's controlled training; CORE-04 can only show that released models agree with it. In the paper, CLIP should be called "no flips (paper statement)" and DINOv3 "no flips (released configs of its teacher and sibling)", and the prediction should be pre-registered with SigLIP 2 excluded.

## Differences from PROJECT_NEXT_PHASE.md Section 4.2

- **DINOv3:** listed as "Verify" and optional; the released configs show no flips, which makes it the best same-family no-flip control, not just an optional extra. Access: request form or gated HF repo; the timm mirror is ungated but under the DINOv3 licence, which asks papers to "acknowledge the use of DINO Materials".
- **SigLIP 2:** "Verify" resolves to unknown, not yes or no.
- **EVA-02 ViT-B/14:** "Verify" resolves to yes (hard-coded flip in MIM pre-training). The only 224 px B/14 weights are MIM-only (IN-21K, no fine-tuning); the fine-tuned B/14 checkpoints run at 448 px.
- **ConvNeXt-B:** yes, but only through timm 0.3.2's default; neither the paper nor ConvNeXt's own code names the flip.
- **DeiT-III:** yes in both listed stages for the 224 ImageNet-1k model (3-Augment pre-training at 192 px, then a 224 px fine-tune with RandAugment, which adds rotations up to 30 degrees).
- **DINOv2 g/14 and EVA-02** use SwiGLU MLPs, not GELU MLPs; the twin and operator code on MLP hidden units needs a rule for the gated product. `dinov2_vitg14_reg` is not yet in `MODEL_REGISTRY`, and neither are CLIP, DINO v1, DINOv3, EVA-02, ConvNeXt or ResNet-50 (the CNNs need a non-ViT path).
- **CLIP in timm:** the `.openai` tag runs GELU, while the original model uses QuickGELU (timm's own note); load the weights into `vit_base_patch16_clip_quickgelu_224` (verify the mechanism on the VM).
- **ResNet-50:** chose torchvision V1 because V2's TrivialAugmentWide rotates by up to 135 degrees, which would confound the R90 and UD controls.

Related data notes (CORE-06, CORE-07): CUB images overlap with ImageNet (Caltech's own warning), and ImageNet-Sketch pads some classes with flipped and rotated copies. See `configs/data/cub_parts.yaml` and `configs/data/shift_sets.yaml`.
