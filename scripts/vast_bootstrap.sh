#!/usr/bin/env bash
# Fresh rented VM (vast.ai, one RTX 5090, 2026-11-09): Python environment, memory-cap scaling, HF_HOME, data caches.
# Run from the repo root after the code is unpacked and the HF token is in $HF_HOME/token:
#   bash scripts/vast_bootstrap.sh env      # venv, torch 2.11.0+cu128 (same build as the A100 runs), dependencies
#   bash scripts/vast_bootstrap.sh caches   # ImageNet validation and train caches (same seeds and selection as the A100)
# Everything downloaded goes under $HF_HOME (default ~/hf); caches go to cache/.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/hf}"
mkdir -p "$HF_HOME" logs cache

step="${1:-env}"
if [ "$step" = "env" ]; then
  python3 -m venv .venv || { pip install -q virtualenv && python3 -m virtualenv .venv; }
  . .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
  grep -vE "^\s*(#|$|torch|torchvision)" requirements.txt > /tmp/req_no_torch.txt
  pip install -q -r /tmp/req_no_torch.txt
  # The per-process GPU memory caps in the scripts are fractions of the card, chosen for a shared 80 GB A100.
  # With MIRROR_MEM_SCALE set (80 / 32 = 2.5 on a 32 GB card), every cap is multiplied by it (at most 0.95), so a
  # job keeps the same absolute limit in GB. The caps never change results. Without the variable nothing happens.
  site=$(python -c "import site; print(site.getsitepackages()[0])")
  cat > "$site/mirror_memscale.py" <<'PY'
import os
import torch

_scale = float(os.environ["MIRROR_MEM_SCALE"])
_orig = torch.cuda.set_per_process_memory_fraction


def _scaled(fraction, device=None):
    return _orig(min(0.95, float(fraction) * _scale), device)


torch.cuda.set_per_process_memory_fraction = _scaled
torch.cuda.memory.set_per_process_memory_fraction = _scaled
PY
  echo 'import os; os.environ.get("MIRROR_MEM_SCALE") and __import__("mirror_memscale")' > "$site/mirror_memscale.pth"
  python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)
PY
  MIRROR_MEM_SCALE=2.5 python -c "import torch; torch.cuda.set_per_process_memory_fraction(0.08); print('memory-cap scaling active')"
elif [ "$step" = "caches" ]; then
  . .venv/bin/activate
  test -s "$HF_HOME/token" || { echo "missing $HF_HOME/token (ImageNet is gated)"; exit 1; }
  python scripts/build_cache.py 2>&1 | tee logs/build_cache.log
  # nproc shows the host's CPUs inside a container; use the cgroup quota (cpu.max) when there is one.
  ncpu=$(awk '$1 != "max" {printf "%d", $1 / $2}' /sys/fs/cgroup/cpu.max 2>/dev/null); ncpu=${ncpu:-$(nproc)}
  [ "$ncpu" -ge 1 ] 2>/dev/null || ncpu=$(nproc)
  python scripts/build_train_cache.py --workers "$(( ncpu > 3 ? ncpu - 2 : 1 ))" 2>&1 | tee logs/build_train_cache.log
  ls -la cache/
else
  echo "usage: bash scripts/vast_bootstrap.sh env|caches"; exit 2
fi
