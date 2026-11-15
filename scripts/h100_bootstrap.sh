#!/usr/bin/env bash
# Fresh H100 container (Ubuntu, root; Jarvis Labs keeps only /home across pause and resume, so everything lives under /home): machine checks, system packages, .venv with a
# torch build that matches the driver, requirements, HF_HOME and the Hugging Face login, a GPU smoke test and the
# CPU test suite. Idempotent: every step checks before it acts, so rerun it after a failure or after copying the
# machine (it also repairs a .venv whose interpreter links broke in a copy).
#
#   cd /home/Mirror_CPAL
#   read -rs HF_TOKEN && export HF_TOKEN       # optional; typed into the ssh session, never written to the repo
#   bash scripts/h100_bootstrap.sh [--hf-home /home/hf] [--skip-tests] [--skip-apt] [--with-git]
#
# Settings (environment, defaults in brackets): TORCH_VERSION [2.11.0], TORCHVISION_VERSION [0.26.0], TORCH_CU
# [from the driver: cu128 when it supports CUDA 12.8 or newer, the build of the A100 and 5090 runs; cu126 for 12.6
# and 12.7; below 12.6, torch 2.6.0 with cu124 and a warning], MIN_DISK_GB [350].
# Writes .venv/, ~/.mirror_env.sh (HF_HOME, PATH, MIRROR_REPO; sourced from ~/.bashrc; never a token),
# $HF_HOME/token (only when HF_TOKEN is set; outside the repo), logs/h100_bootstrap.log and
# logs/h100_bootstrap_pytest.log. Exit code 1 if a required check failed (listed at the end).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"
HF_HOME_SET="${HF_HOME:-/home/hf}"; SKIP_TESTS=0; SKIP_APT=0; WITH_GIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --hf-home) HF_HOME_SET="$2"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-apt) SKIP_APT=1; shift ;;
    --with-git) WITH_GIT=1; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"; TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
MIN_DISK_GB="${MIN_DISK_GB:-350}"
mkdir -p logs
exec > >(tee -a logs/h100_bootstrap.log) 2>&1
FAILS=(); WARNS=()
step() { echo; echo "== $* ($(date -u '+%F %T'))"; }
warn() { echo "WARNING: $*"; WARNS+=("$*"); }
fail() { echo "FAILED: $*"; FAILS+=("$*"); }
ver_ge() { awk -v a="$1" -v b="$2" 'BEGIN { split(a, x, "."); split(b, y, "."); exit !((x[1] + 0 > y[1] + 0) || (x[1] + 0 == y[1] + 0 && x[2] + 0 >= y[2] + 0)) }'; }

# ---------------------------------------------------------------------------------------------------
step "1. machine"
GPU_NAME="none"; GPU_MB=0; DRV=""; DRV_CUDA="0.0"
if command -v nvidia-smi > /dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  DRV_CUDA=$(nvidia-smi | grep -o 'CUDA Version: *[0-9.]*' | grep -o '[0-9.]*$' | head -1)
  echo "GPU: $(nvidia-smi -L | wc -l) x $GPU_NAME, $GPU_MB MiB; driver $DRV (CUDA $DRV_CUDA)"
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
  echo "processes on the GPU (a container may not see them):"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  case "$GPU_NAME" in *H100*) ;; *) warn "expected an H100, found $GPU_NAME" ;; esac
else
  fail "nvidia-smi not found (no GPU driver in this container)"
fi
ncpu=$(awk '$1 != "max" {printf "%d", $1 / $2}' /sys/fs/cgroup/cpu.max 2>/dev/null); ncpu=${ncpu:-0}
[ "$ncpu" -ge 1 ] 2>/dev/null || ncpu=$(nproc)
echo "CPUs: $ncpu usable (nproc $(nproc))"
free -g | sed -n '1,2p'
RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
[ "${RAM_GB:-0}" -ge 64 ] || warn "only ${RAM_GB} GB RAM; the cache builds and healing data loaders want 64 GB or more"
mkdir -p "$HF_HOME_SET"
df -h "$REPO" "$HF_HOME_SET" | awk '!seen[$0]++'
DISK_AVAIL=$(df --output=avail -BG "$HF_HOME_SET" | tail -1 | tr -dc '0-9')
[ "${DISK_AVAIL:-0}" -ge "$MIN_DISK_GB" ] || warn "only ${DISK_AVAIL} GB free under $HF_HOME_SET; scripts/h100_data.sh needs about 300 GB (estimate)"

# ---------------------------------------------------------------------------------------------------
step "2. system packages"
command -v python3 > /dev/null || NEED_PY=1
PKGS=()
[ "${NEED_PY:-0}" = 1 ] && PKGS+=(python3)
if command -v python3 > /dev/null; then
  PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  python3 -c "import ensurepip" 2> /dev/null || PKGS+=("python${PYV}-venv")
  [ -f "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')/Python.h" ] || PKGS+=("python${PYV}-dev")
fi
command -v gcc > /dev/null || PKGS+=(build-essential)          # torch.compile (Triton) builds small C helpers
for c in curl rsync tmux unzip; do command -v "$c" > /dev/null || PKGS+=("$c"); done
{ command -v flock && command -v setsid; } > /dev/null || PKGS+=(util-linux)
[ "$WITH_GIT" = 1 ] && ! command -v git > /dev/null && PKGS+=(git)
if [ ${#PKGS[@]} -eq 0 ]; then
  echo "all present"
elif [ "$SKIP_APT" = 1 ]; then
  warn "missing packages (not installed, --skip-apt): ${PKGS[*]}"
elif command -v apt-get > /dev/null && { [ "$(id -u)" = 0 ] || sudo -n true 2> /dev/null; }; then
  SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -n"     # VMs log in as a normal user with passwordless sudo
  echo "installing: ${PKGS[*]}"
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -q > /dev/null && \
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends "${PKGS[@]}" > /dev/null \
    || warn "apt-get install failed for: ${PKGS[*]}"
else
  warn "cannot install ${PKGS[*]} (no apt-get or not root)"
fi
command -v python3 > /dev/null || { fail "python3 missing"; exit 1; }
PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
ver_ge "$PYV" 3.10 || warn "python $PYV; the code is tested with 3.12"
command -v git > /dev/null && echo "git present: run metadata will record whether tracked files differ from HEAD"

# ---------------------------------------------------------------------------------------------------
step "3. .venv"
venv_ok() { [ -x .venv/bin/python ] && .venv/bin/python -c "import sys" 2> /dev/null; }
if [ -d .venv ] && ! venv_ok; then
  # A copied machine can drop the interpreter links (.venv/bin/python3 -> /usr/bin/python3); relink them.
  home=$(sed -n 's/^home *= *//p' .venv/pyvenv.cfg 2> /dev/null | head -1)
  vv=$(sed -n 's/^version\(_info\)\? *= *\([0-9]*\.[0-9]*\).*/\2/p' .venv/pyvenv.cfg 2> /dev/null | head -1)
  target=""
  for cand in "$home/python$vv" "$(command -v "python$vv" 2> /dev/null)" "$home/python3"; do
    [ -n "$cand" ] && [ -x "$cand" ] && { target="$cand"; break; }
  done
  if [ -n "$target" ] && [ -n "$vv" ]; then
    echo "relinking .venv to $target (python $vv)"
    ln -sfn "$target" ".venv/bin/python$vv"; ln -sfn "python$vv" .venv/bin/python3; ln -sfn "python$vv" .venv/bin/python
  fi
  if ! venv_ok; then
    old=".venv.broken.$(date +%s)"; mv .venv "$old"; warn "moved a broken .venv to $old (python $vv not found)"
  fi
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv 2> /dev/null || { rm -rf .venv; python3 -m venv --without-pip .venv; } \
    || { fail "could not create .venv"; exit 1; }
fi
PY="$REPO/.venv/bin/python"
if ! "$PY" -m pip --version > /dev/null 2>&1; then     # a venv made without ensurepip has no pip
  "$PY" -m ensurepip --upgrade > /dev/null 2>&1 || curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$PY" - -q \
    || { fail "no pip in .venv (install python$PYV-venv, delete .venv, rerun)"; exit 1; }
fi
"$PY" -m pip install -q --upgrade pip > /dev/null || warn "pip upgrade failed"
echo "venv python: $("$PY" -V 2>&1)"

# ---------------------------------------------------------------------------------------------------
step "4. torch for driver CUDA $DRV_CUDA"
if [ -z "${TORCH_CU:-}" ]; then
  if ver_ge "$DRV_CUDA" 12.8; then TORCH_CU=cu128
  elif ver_ge "$DRV_CUDA" 12.6; then TORCH_CU=cu126
  else
    TORCH_CU=cu124; TORCH_VERSION=2.6.0; TORCHVISION_VERSION=0.21.0
    warn "the driver supports CUDA $DRV_CUDA only: installing torch 2.6.0+cu124 (earlier runs used 2.11.0+cu128)"
  fi
fi
WANT="$TORCH_VERSION+$TORCH_CU"
have_torch() { "$PY" -c "import torch, torchvision; print(torch.__version__)" 2> /dev/null; }
if [ "$(have_torch)" = "$WANT" ]; then
  echo "torch $WANT already installed"
else
  echo "installing torch $WANT and torchvision $TORCHVISION_VERSION"
  "$PY" -m pip install -q "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CU" || fail "torch install failed"
fi

# ---------------------------------------------------------------------------------------------------
step "5. requirements"
REQ=$(mktemp)
grep -vE '^\s*(#|$)' requirements.txt | grep -vE '^(torch|torchvision)([<>=!~ ;]|$)' > "$REQ"
"$PY" -m pip install -q -r "$REQ" || fail "pip install -r requirements.txt failed"
rm -f "$REQ"
if [ "$(have_torch)" != "$WANT" ]; then      # a dependency pulled another torch: put the matching one back
  warn "requirements changed torch to $(have_torch); reinstalling $WANT"
  "$PY" -m pip install -q --force-reinstall --no-deps "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CU" || fail "torch reinstall failed"
fi
# Memory caps in the scripts are fractions of an 80 GB card. With MIRROR_MEM_SCALE set, every cap is multiplied
# by it (at most 0.95), so a job keeps the same limit in GB on another card. Inactive unless the variable is set.
SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE/mirror_memscale.py" << 'PYEOF'
import os
import torch

_scale = float(os.environ["MIRROR_MEM_SCALE"])
_orig = torch.cuda.set_per_process_memory_fraction


def _scaled(fraction, device=None):
    return _orig(min(0.95, float(fraction) * _scale), device)


torch.cuda.set_per_process_memory_fraction = _scaled
torch.cuda.memory.set_per_process_memory_fraction = _scaled
PYEOF
echo 'import os; os.environ.get("MIRROR_MEM_SCALE") and __import__("mirror_memscale")' > "$SITE/mirror_memscale.pth"
# Thread cap for every venv process (BLAS, OpenMP, numexpr; MIRROR_THREADS, default 4): many CPU-bound jobs share the
# cores, and 28-thread pools per process made small matrix work hundreds of times slower (15 November). The name
# starts with 00 so it runs before mirror_memscale.pth, which imports torch and would start its pools uncapped.
echo 'import os; [os.environ.setdefault(k, os.environ.get("MIRROR_THREADS", "4")) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")]' > "$SITE/00_mirror_threads.pth"
MEMSCALE=""
if [ "${GPU_MB:-0}" -gt 0 ]; then
  MEMSCALE=$(awk -v g="$GPU_MB" 'BEGIN { s = 81920 / g; if (s < 0.95 || s > 1.05) printf "%.3f", s }')
fi
[ -n "$MEMSCALE" ] && warn "GPU has $GPU_MB MiB, not 80 GB: MIRROR_MEM_SCALE=$MEMSCALE goes into ~/.mirror_env.sh"

# ---------------------------------------------------------------------------------------------------
step "6. HF_HOME and environment file"
HF_REAL=$(readlink -f "$HF_HOME_SET"); REPO_REAL=$(readlink -f "$REPO")
case "$HF_REAL/" in "$REPO_REAL/"*) fail "HF_HOME ($HF_REAL) is inside the repo; choose a path outside it"; exit 1 ;; esac
export HF_HOME="$HF_REAL"
ENVF="$HOME/.mirror_env.sh"
{
  echo "# Written by scripts/h100_bootstrap.sh on $(date -u '+%F %T'). No secrets in this file."
  echo "export HF_HOME=\"$HF_REAL\""
  echo "export MIRROR_REPO=\"$REPO_REAL\""
  echo "case \":\$PATH:\" in *\":$REPO_REAL/.venv/bin:\"*) ;; *) export PATH=\"$REPO_REAL/.venv/bin:\$PATH\" ;; esac"
  [ -n "$MEMSCALE" ] && echo "export MIRROR_MEM_SCALE=$MEMSCALE"
} > "$ENVF"
grep -q 'mirror_env.sh' "$HOME/.bashrc" 2> /dev/null || echo '[ -f "$HOME/.mirror_env.sh" ] && . "$HOME/.mirror_env.sh"' >> "$HOME/.bashrc"
. "$ENVF"
echo "HF_HOME=$HF_HOME (env file $ENVF, sourced from ~/.bashrc)"

# ---------------------------------------------------------------------------------------------------
step "7. Hugging Face"
if [ -n "${HF_TOKEN:-}" ]; then
  "$PY" - << 'PYEOF' || fail "Hugging Face login failed"
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)   # saved to $HF_HOME/token, outside the repo
PYEOF
fi
HF_STATUS=$("$PY" - << 'PYEOF'
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

try:
    who = HfApi().whoami().get("name", "?")
except Exception as exc:
    print(f"not logged in ({type(exc).__name__}); set HF_TOKEN and rerun")
    raise SystemExit(0)
try:
    get_hf_file_metadata(hf_hub_url("ILSVRC/imagenet-1k", "data/validation-00000-of-00014.parquet", repo_type="dataset"))
    print(f"ok: logged in as {who}; ImageNet-1k (gated) is accessible")
except Exception as exc:
    print(f"logged in as {who}, but no access to ILSVRC/imagenet-1k ({type(exc).__name__}); accept its terms on the hub")
PYEOF
)
echo "$HF_STATUS"
case "$HF_STATUS" in ok:*) ;; *) fail "Hugging Face: $HF_STATUS" ;; esac

# ---------------------------------------------------------------------------------------------------
step "8. GPU smoke test"
SMOKE="skipped"
if [ "$GPU_NAME" != none ]; then
  if "$PY" - << 'PYEOF'
import time

import torch

print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch cannot see the GPU"
torch.cuda.set_per_process_memory_fraction(0.10)            # shared GPU etiquette (CLAUDE.md)
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
dev = torch.device("cuda")
a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
for _ in range(3):
    a @ b
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
dt = time.time() - t0
ref = (a[:256].float() @ b[:, :256].float())
err = ((c[:256, :256].float() - ref).abs().max() / ref.abs().max()).item()
print(f"bf16 matmul 8192^3: {2 * 8192 ** 3 * 20 / dt / 1e12:.0f} TFLOP/s, max relative error {err:.1e}")
assert err < 2e-2
torch.backends.cuda.matmul.allow_tf32 = False
m = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)).to(dev)
x = torch.randn(64, 256, device=dev)
eager = m(x)
cm = torch.compile(m)
t0 = time.time()
out = cm(x)
out.sum().backward()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out16 = cm(x)
torch.cuda.synchronize()
diff = (out - eager).abs().max().item()
print(f"torch.compile: compiled in {time.time() - t0:.1f} s, max difference from eager {diff:.1e}, bf16 autocast ok "
      f"({out16.dtype}); peak memory {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
assert diff < 1e-4
PYEOF
  then SMOKE="ok"; else SMOKE="failed"; fail "GPU smoke test (if torch.compile failed: apt-get install build-essential python3-dev)"; fi
fi

# ---------------------------------------------------------------------------------------------------
step "9. CPU tests"
TESTS="skipped (--skip-tests)"
if [ "$SKIP_TESTS" = 0 ]; then
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q > logs/h100_bootstrap_pytest.log 2>&1; trc=$?
  TESTS="$(tail -1 logs/h100_bootstrap_pytest.log) ($(( $(date +%s) - t0 )) s; logs/h100_bootstrap_pytest.log)"
  [ $trc -eq 0 ] || fail "CPU tests: $TESTS"
fi

# ---------------------------------------------------------------------------------------------------
step "summary"
COMMIT=$("$PY" -c "from mirror.utils import git_info; print(git_info()['git_hash'][:12])" 2> /dev/null)
echo "repo        $REPO_REAL (commit ${COMMIT:-unknown})"
echo "GPU         $GPU_NAME, $GPU_MB MiB, driver $DRV, CUDA $DRV_CUDA"
echo "CPU and RAM $ncpu CPUs, ${RAM_GB} GB RAM"
echo "disk        $(df -h --output=avail "$HF_HOME" | tail -1 | tr -d ' ') free under $HF_HOME"
echo "python      $("$PY" -V 2>&1); torch $(have_torch) ($("$PY" -c 'import torch; print(torch.version.cuda)' 2> /dev/null))"
echo "HF_HOME     $HF_HOME; $HF_STATUS"
echo "smoke test  $SMOKE"
echo "CPU tests   $TESTS"
for w in "${WARNS[@]}"; do echo "warning     $w"; done
if [ ${#FAILS[@]} -gt 0 ]; then
  for f in "${FAILS[@]}"; do echo "FAILED      $f"; done
  exit 1
fi
echo "ready. Next: . ~/.mirror_env.sh; setsid nohup bash scripts/h100_data.sh > logs/h100_data.log 2>&1 < /dev/null &"
