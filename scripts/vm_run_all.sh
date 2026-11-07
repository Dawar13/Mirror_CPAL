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

echo "== ALL_DONE $(date)"
