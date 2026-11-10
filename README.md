# Mirror Surgery

Vision transformers trained with left-right flip augmentation tend to learn many of their features
twice, once for each mirror orientation. This project measures that duplication (the *symmetry tax*)
and removes it from released checkpoints without retraining, by tying each pair of mirror-twin
neurons together so that the pair is stored and computed once.

> Status: research code for a paper in preparation. Experiments are still running, so code and
> results may change.

## The idea in brief

1. Pass images and their mirror images through a pretrained model and fit, for each layer, a
   *mirror operator*: the exact way the internal features change when the image is flipped.
2. Find *mirror-twin* neurons: pairs where one neuron responds to the mirrored image the way its
   twin responds to the original.
3. Project the weights onto an exactly mirror-symmetric form. Twins become exact mirror images of
   each other, so each pair is stored once. Nothing is switched off: both neurons still fire, they
   just share their weights.
4. Optionally, use one operator across the whole network. The result gives exactly the same output
   for an image and its mirror, by construction; a short distillation then recovers accuracy.

## Early findings (DINOv2 ViT-B, ImageNet)

- In the first block, roughly three quarters of the MLP neurons form mirror-twin pairs. The effect
  fades with depth.
- Across the first half of the network, around 40% of the MLP weights are redundant mirror copies.
- Converting the MLPs of the first half with no retraining keeps k-NN accuracy within about 2
  points of the original, while low-rank (SVD) compression at the same parameter count collapses.
- Converting every block gives an exactly mirror-invariant model with about three quarters of the
  parameters. After a short distillation it is within about 1 point of the original.
- The same structure appears in other flip-trained ViTs (DeiT-III, MAE) and is much weaker in
  SigLIP2.

Pass and fail thresholds for each gated experiment were committed to this repository before the
runs.

## Layout

```
mirror/    library: data, feature statistics, operator fitting, twin matching, surgery,
           attention, baselines, distillation, evaluation
scripts/   experiment entry points and results collection
configs/   settings for each experiment (YAML)
jobs/      GPU job lists for the scheduler in scripts/vm_sched.sh
tests/     CPU-only tests (tiny random ViT, synthetic data)
runs/      raw outputs of each experiment
results/   reports, tables and figures
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/path/to/big_disk/hf   # every download (datasets, model weights) goes here
pytest -q                             # CPU only
```

ImageNet-1k on Hugging Face is gated, so accept its terms first. The image caches are built with
`scripts/build_cache.py` (validation) and `scripts/build_train_cache.py` (training subsets). The
newest PyTorch wheel may need a newer GPU driver than you have; `requirements.txt` shows the
CUDA 12.8 install.

## Conventions

- Operators are fitted on ImageNet training images, evaluation uses the validation set, and
  distillation uses training images only.
- Every run logs its config, seed and git commit.
- GPU scripts cap their memory per process so that several jobs can share one GPU.
