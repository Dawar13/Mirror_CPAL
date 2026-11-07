#!/usr/bin/env bash
# B1 -> B2 -> B3 in one go on the VM; stops at the first failure.
#   HF_HOME=/home/gpunet/hf nohup bash scripts/vm_run_all.sh > ~/mirror_run_all.log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
: "${HF_HOME:?set HF_HOME first}"
mkdir -p results

echo "== access check $(date)"
python - <<'EOF' || { echo "NO ACCESS to ILSVRC/imagenet-1k: put a token in $HF_HOME/token and accept the dataset terms"; exit 10; }
from huggingface_hub import HfApi
files = HfApi().list_repo_files("ILSVRC/imagenet-1k", repo_type="dataset")
print("access ok,", sum(f.startswith("data/validation-") for f in files), "validation shards")
EOF

echo "== B0 tests $(date)"
python -m pytest -q 2>&1 | tail -3 | tee results/b0_pytest.txt
grep -q " passed" results/b0_pytest.txt && ! grep -q "failed" results/b0_pytest.txt || exit 9

echo "== B1 cache $(date)"
if [ -f cache/meta.json ] && grep -q '"complete": true' cache/meta.json; then
  echo "cache already complete, skipping"
else
  python scripts/build_cache.py --config configs/e1.yaml 2>&1 | tee results/b1_log.txt || exit 1
fi

echo "== B2 sanity $(date)"
python scripts/sanity_b2.py --config configs/e1.yaml 2>&1 | tee results/b2_log.txt || exit 2

echo "== B3 E1 $(date)"
python scripts/run_e1.py --config configs/e1.yaml 2>&1 | tee results/e1_log.txt || exit 3

echo "== extra plots $(date)"
python scripts/plot_e1_extra.py 2>&1 | tee results/extra_plots_log.txt || exit 4
cp cache/meta.json results/b1_cache_meta.json
nvidia-smi > results/gpu_info.txt 2>&1

echo "== ALL_DONE $(date)"
