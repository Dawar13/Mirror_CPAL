#!/usr/bin/env bash
# Data caches, dataset downloads and model weights for a fresh H100, after scripts/h100_bootstrap.sh.
# Staged and resumable: each stage skips when its output exists, so rerun the same command after a failure or a
# preemption. Independent lanes run in parallel. Downloads go under $HF_HOME, decoded caches to cache/. Disk use is
# logged after every stage (logs/h100_data/disk.tsv); a stage is refused when HF_HOME + cache would pass BUDGET_GB.
#
#   . ~/.mirror_env.sh
#   setsid nohup bash scripts/h100_data.sh --optional > logs/h100_data.log 2>&1 < /dev/null &
#   bash scripts/h100_data.sh --status                   # which stages are done
#   bash scripts/h100_data.sh --only msel,check          # selected stages (comma-separated)
#   options: --optional (= --cub --imagenet-v2 --sketch), --skip <stages>, --serial, --budget-gb N, --workers N
#
# Stages, with ESTIMATES (not measured on an H100) of the disk they add and their wall time:
#   lane imagenet (needs the Hugging Face token; ImageNet-1k is gated)
#     val     validation cache, scripts/build_cache.py           6.7 GB shards + 7.5 GB cache            10-20 min
#     train   fit0..2, bank, healval, heal (build_train_cache)   43 GB (86 shards) + 37 GB cache        30-45 min
#             (measured: 30 min on the A100, 43 min on the 5090)
#     msel    method-selection set (build_train_cache --msel)    about 2 GB shards + 1.5 GB cache        5-10 min
#     check   scripts/check_disjoint.py -> results/data_check/disjoint.json                            1 min
#   lane in1k
#     in1k    every ImageNet-1k train shard (294, 146 GB in all; about 103 GB beyond train's)          15-60 min
#     in100   scripts/build_imagenet100.py (cache/in100_*; extra arguments from $IN100_ARGS), 2 GB   20-40 min
#   lane tier2
#     tier2   CIFAR-100, Flowers-102, Pets, DTD (cache/transfer/) and ADE20K (cache/ade20k/), built by the
#             tier2 scripts' own cache functions               5 GB parquet + about 37 GB decoded       20-40 min
#   lane models
#     models  scripts/prefetch_models.py (arguments from $PREFETCH_ARGS)      about 20 GB              10-20 min
#   lane extra (only when asked)
#     cub     CUB-200-2011 with part annotations, Caltech DATA record 65de6-vp158 (verify), md5 checked,
#             extracted to $HF_HOME/cub/CUB_200_2011            1.2 GB + 1.2 GB extracted                5 min
#     v2      ImageNet-V2 matched frequency, hub dataset vaishaal/ImageNetV2 (verify), extracted to
#             $HF_HOME/extracted/imagenetv2-matched-frequency-format-val       1.3 GB + 1.3 GB          5 min
#     sketch  ImageNet-Sketch, hub dataset songweig/imagenet_sketch (verify), kept zipped at
#             $HF_HOME/extracted/ImageNet-Sketch.zip (link into the hub cache)  7.6 GB                5-10 min
# Total with everything about 290 GB (estimate), leaving about 200 GB of a 500 GB disk for runs. Wall time with
# the lanes in parallel: about 1 to 1.5 hours (estimate; the train cache and ImageNet-100 are the long poles).
# Outputs: logs/h100_data/<stage>.log, status.tsv, disk.tsv; results/data_check/data_manifest.json.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
[ -f "$HOME/.mirror_env.sh" ] && . "$HOME/.mirror_env.sh"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi
export HF_HOME="${HF_HOME:-/workspace/hf}"
export CUDA_VISIBLE_DEVICES=""              # data work never touches the shared GPU
PY="$(command -v python || command -v python3)"

ONLY=""; SKIP=""; CUB=0; V2=0; SKETCH=0; SERIAL=0; STATUS=0; BUDGET_GB="${BUDGET_GB:-400}"; WORKERS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --cub) CUB=1; shift ;;
    --imagenet-v2) V2=1; shift ;;
    --sketch) SKETCH=1; shift ;;
    --optional) CUB=1; V2=1; SKETCH=1; shift ;;
    --serial) SERIAL=1; shift ;;
    --budget-gb) BUDGET_GB="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --status) STATUS=1; shift ;;
    -h|--help) sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done

LOGDIR="$ROOT/logs/h100_data"; MARK_HF="$HF_HOME/mirror_markers"
mkdir -p "$LOGDIR" "$ROOT/cache" "$ROOT/results/data_check" "$HF_HOME" "$MARK_HF"
RUN_ID="$(date -u +%Y%m%dT%H%M%S)"
ncpu=$(awk '$1 != "max" {printf "%d", $1 / $2}' /sys/fs/cgroup/cpu.max 2>/dev/null); ncpu=${ncpu:-0}
[ "$ncpu" -ge 1 ] 2>/dev/null || ncpu=$(nproc)
if [ -n "$WORKERS" ]; then W_MAIN=$WORKERS; W_SIDE=$WORKERS
elif [ "$SERIAL" = 1 ]; then W_MAIN=$(( ncpu > 3 ? ncpu - 2 : 1 )); W_SIDE=$W_MAIN
else W_MAIN=$(( ncpu * 2 / 3 > 2 ? ncpu * 2 / 3 : 2 )); W_SIDE=$(( ncpu / 4 > 2 ? ncpu / 4 : 2 )); fi
IN1K_REV=$(sed -n 's/^REVISION = "\([0-9a-f]*\)".*/\1/p' scripts/build_train_cache.py)
CUB_URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_MD5="97eceeb196236b17998738112f37df78"      # from the Caltech DATA record's file listing (2026-11-14)
V2_REPO="vaishaal/ImageNetV2"; V2_REV="d626240be2538720e83103a0e1178d24aca8b12c"; V2_FILE="imagenetv2-matched-frequency.tar.gz"
SK_REPO="songweig/imagenet_sketch"; SK_REV="0d8b8b5380efe50baae075842505918ad43c88fe"; SK_FILE="data/ImageNet-Sketch.zip"

note() { echo "[$(date -u +%H:%M:%S)] $*"; }
record() { printf "%s\t%s\t%s\t%s\t%s\n" "$RUN_ID" "$(date -u '+%F %T')" "$1" "$2" "$3" >> "$LOGDIR/status.tsv"; }
du_gb() { [ -e "$1" ] && du -s --block-size=1G "$1" 2>/dev/null | cut -f1 || echo 0; }
avail_gb() { df --output=avail -BG "$HF_HOME" | tail -1 | tr -dc '0-9'; }
disk_log() {
  local f="$LOGDIR/disk.tsv"
  [ -f "$f" ] || printf "utc\tafter_stage\tfree_gb\thf_home_gb\tcache_gb\truns_gb\n" > "$f"
  local line
  line=$(printf "%s\t%s\t%s\t%s\t%s\t%s" "$(date -u '+%F %T')" "$1" "$(avail_gb)" "$(du_gb "$HF_HOME")" \
         "$(du_gb "$ROOT/cache")" "$(du_gb "$ROOT/runs")")
  echo "$line" >> "$f"
  note "disk after $1: free $(cut -f3 <<< "$line") GB; HF_HOME $(cut -f4 <<< "$line") GB; cache $(cut -f5 <<< "$line") GB"
}
budget_ok() {   # $1 stage, $2 estimated GB it adds
  local used avail
  used=$(( $(du_gb "$HF_HOME") + $(du_gb "$ROOT/cache") )); avail=$(avail_gb)
  if [ $((used + $2)) -gt "$BUDGET_GB" ]; then
    note "$1: refused: HF_HOME + cache use $used GB; about $2 GB more would pass BUDGET_GB=$BUDGET_GB"; return 1
  fi
  if [ "$avail" -lt $(($2 + 10)) ]; then note "$1: refused: $avail GB free, needs about $2 GB + 10"; return 1; fi
}
want() {
  if [ -n "$ONLY" ]; then case ",$ONLY," in *",$1,"*) ;; *) return 1 ;; esac; fi
  case ",$SKIP," in *",$1,"*) return 1 ;; esac
  return 0
}
meta_complete() { "$PY" -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('complete') else 1)" "$1" 2>/dev/null; }
hf_token_ok() { "$PY" -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)" 2>/dev/null; }

# ---------------------------------------------------------------------------------------------------
# Done tests (a stage is skipped when its test passes)
# ---------------------------------------------------------------------------------------------------
done_val() { meta_complete cache/meta.json && [ -f cache/val224_uint8.npy ]; }
done_train() { meta_complete cache/train_meta.json && [ -f cache/train224_uint8.npy ] && [ -f cache/heal256_uint8.npy ] && [ -f cache/train_index.npz ]; }
done_msel() { meta_complete cache/msel_meta.json && [ -f cache/msel224_uint8.npy ] && [ -f cache/msel_index.npz ]; }
done_check() {
  local f=results/data_check/disjoint.json
  [ -f "$f" ] && [ "$f" -nt cache/train_index.npz ] && [ "$f" -nt cache/msel_index.npz ] && \
    "$PY" -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['ok'] else 1)" "$f" 2>/dev/null
}
done_in100() {   # scripts/build_imagenet100.py sets extra.complete in its mirror.meta record when all is written
  [ -f cache/in100_train72_uint8.npy ] && [ -f cache/in100_val72_uint8.npy ] && "$PY" -c "import json, sys; m = json.load(open(sys.argv[1])); sys.exit(0 if (m.get('extra') or {}).get('complete') or m.get('complete') else 1)" cache/in100_meta.json 2>/dev/null
}
done_in1k() { [ -f "$MARK_HF/imagenet1k_train_all.done" ] || done_in100; }
done_tier2() {
  local d
  for d in cifar100 flowers102 pets dtd; do meta_complete "cache/transfer/$d/meta.json" || return 1; done
  meta_complete cache/ade20k/meta.json
}
done_models() {
  "$PY" - << 'PYEOF' 2>/dev/null
import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, ".")
rec = Path("results/data_check/models.json")
if not rec.exists():
    sys.exit(1)
ok = {r["name"] for r in json.loads(rec.read_text()).get("models", []) if r.get("status") in ("ok", "skipped")}
spec = importlib.util.spec_from_file_location("_pm", "scripts/prefetch_models.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)
from mirror.adapters_timm import MODEL_REGISTRY
sys.exit(0 if {it["name"] for it in pm.plan(MODEL_REGISTRY)} <= ok else 1)
PYEOF
}
done_cub() { [ -f "$HF_HOME/cub/CUB_200_2011/parts/part_locs.txt" ]; }
done_v2() { [ -f "$MARK_HF/imagenetv2_matched_frequency.done" ]; }
done_sketch() { [ -s "$HF_HOME/extracted/ImageNet-Sketch.zip" ]; }

run_stage() {   # $1 stage, $2 estimated GB it adds, then the command
  local name=$1 est=$2 t0 rc dt
  shift 2
  want "$name" || return 0
  if "done_$name"; then note "$name: done already, skipping"; record "$name" skipped 0; return 0; fi
  budget_ok "$name" "$est" || { record "$name" "refused (disk)" 0; return 1; }
  note "$name: start (adds about $est GB, estimate); log logs/h100_data/$name.log"
  t0=$(date +%s)
  echo "== $name $(date -u '+%F %T') run $RUN_ID" >> "$LOGDIR/$name.log"
  "$@" >> "$LOGDIR/$name.log" 2>&1; rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && "done_$name"; then record "$name" ok "$dt"; note "$name: ok in $dt s"
  elif [ $rc -eq 0 ]; then rc=1; record "$name" "ran but output incomplete" "$dt"; note "$name: ran, output incomplete (see its log)"
  else record "$name" "failed rc=$rc" "$dt"; note "$name: FAILED rc=$rc after $dt s (see logs/h100_data/$name.log)"; fi
  disk_log "$name"
  return $rc
}

# ---------------------------------------------------------------------------------------------------
# Stage commands
# ---------------------------------------------------------------------------------------------------
cmd_in1k() {
  "$PY" - "$IN1K_REV" "$W_SIDE" << 'PYEOF' && touch "$MARK_HF/imagenet1k_train_all.done"
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
rev, workers = sys.argv[1] or None, int(sys.argv[2])
repo = "ILSVRC/imagenet-1k"
want = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
              if f.startswith("data/train-") and f.endswith(".parquet"))
local = Path(snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=["data/train-*.parquet"],
                               max_workers=max(4, min(workers, 16))))
have = sorted(p.relative_to(local).as_posix() for p in local.glob("data/train-*.parquet"))
print(f"{len(have)} of {len(want)} train shards in {local}")
sys.exit(0 if have == want else 1)
PYEOF
}

cmd_in100() {
  # shellcheck disable=SC2086
  "$PY" scripts/build_imagenet100.py --workers "$W_MAIN" ${IN100_ARGS:-}
}

cmd_tier2() {
  "$PY" - "$W_SIDE" << 'PYEOF'
import importlib.util, sys, time, traceback
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root))
workers = int(sys.argv[1])


def load(name):
    spec = importlib.util.spec_from_file_location(f"_h100_{name}", root / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


failed = []
T = load("tier2_transfer")
for d in ("cifar100", "flowers102", "pets", "dtd"):
    t0 = time.time()
    try:
        r = T.ensure_dataset_cache(d, T.DATASETS[d], root / "cache" / "transfer", "parquet", workers=workers)
        print(f"{d}: {r['meta']['counts']} in {r['dir']} ({time.time() - t0:.0f} s)", flush=True)
    except Exception:
        traceback.print_exc()
        failed.append(d)
A = load("tier2_ade20k")
t0 = time.time()
try:
    r = A.ensure_seg_cache(A.ADE20K, root / "cache", A.DEFAULTS["crop"][0], workers=workers)
    print(f"ade20k: {r['meta']['counts']} in {r['dir']} ({time.time() - t0:.0f} s)", flush=True)
except Exception:
    traceback.print_exc()
    failed.append("ade20k")
print("failed:", failed or "none")
sys.exit(1 if failed else 0)
PYEOF
}

cmd_cub() {
  local d="$HF_HOME/cub" f="$HF_HOME/cub/CUB_200_2011.tgz"
  mkdir -p "$d"
  if ! echo "$CUB_MD5  $f" | md5sum -c --status - 2>/dev/null; then
    curl -L --fail --retry 5 --retry-delay 10 -C - -o "$f" "$CUB_URL" || curl -L --fail --retry 5 -o "$f" "$CUB_URL" || return 1
    echo "$CUB_MD5  $f" | md5sum -c - || return 1
  fi
  tar -xzf "$f" -C "$d" --no-same-owner && ls "$d/CUB_200_2011"
}

cmd_hub_file() {   # $1 repo, $2 revision, $3 file, $4 extract to (tar.gz) or link as (anything else), $5 marker
  "$PY" - "$@" << 'PYEOF'
import os, sys, tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, rev, fname, dest, marker = sys.argv[1:6]
path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
print("downloaded", path, os.path.getsize(path), "bytes")
dest = Path(dest)
if fname.endswith(".tar.gz"):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as tf:
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(dest, **kw)
    print("extracted to", dest, sorted(os.listdir(dest))[:5])
else:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(os.path.realpath(path))
    print("linked", dest, "->", os.path.realpath(path))
if marker != "-":
    Path(marker).write_text(f"{repo}@{rev}/{fname}\n")
PYEOF
}

write_manifest() {
  "$PY" - "$HF_HOME" "$RUN_ID" << 'PYEOF'
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, ".")
hf, run = sys.argv[1], sys.argv[2]


def size(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) if p.is_dir() else None


def meta(p):
    try:
        m = json.loads(Path(p).read_text())
        return {k: m[k] for k in ("complete", "n_images", "n224", "n256", "n", "counts", "selection_sha256",
                                  "revision", "seed") if k in m}
    except (OSError, ValueError):
        return None


caches = {}
for f in ("val224_uint8.npy", "train224_uint8.npy", "heal256_uint8.npy", "msel224_uint8.npy"):
    caches[f] = size(Path("cache") / f)
for d in ("transfer/cifar100", "transfer/flowers102", "transfer/pets", "transfer/dtd", "ade20k"):
    caches[d] = {"bytes": size(Path("cache") / d), "meta": meta(Path("cache") / d / "meta.json")}
ext = {"cub": f"{hf}/cub/CUB_200_2011", "imagenet_v2_matched_frequency":
       f"{hf}/extracted/imagenetv2-matched-frequency-format-val", "imagenet_sketch_zip": f"{hf}/extracted/ImageNet-Sketch.zip"}
status = [line.rstrip("\n").split("\t") for line in open("logs/h100_data/status.tsv")] if os.path.exists("logs/h100_data/status.tsv") else []
try:
    from mirror.utils import git_info
    git = git_info()
except Exception as exc:  # noqa: BLE001 - provenance is a nicety here
    git = {"error": type(exc).__name__}
out = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run": run, "hf_home": hf,
       "metas": {k: meta(Path("cache") / k) for k in ("meta.json", "train_meta.json", "msel_meta.json")},
       "cache_bytes": caches, "external": {k: {"path": v, "present": os.path.exists(v)} for k, v in ext.items()},
       "stages_this_run": [s for s in status if s and s[0] == run], "git": git}
Path("results/data_check/data_manifest.json").write_text(json.dumps(out, indent=2))
print("wrote results/data_check/data_manifest.json")
PYEOF
}

# ---------------------------------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------------------------------
lane_imagenet() {
  if ! { want val || want train || want msel || want check; }; then return 0; fi
  if ! hf_token_ok && ! { done_val && done_train && done_msel; }; then
    note "imagenet: no Hugging Face token (ImageNet-1k is gated); run the bootstrap with HF_TOKEN set"
    record imagenet "failed (no token)" 0; return 1
  fi
  local rc=0
  run_stage val 15 "$PY" scripts/build_cache.py || rc=1
  run_stage train 80 "$PY" scripts/build_train_cache.py --workers "$W_MAIN" || rc=1
  run_stage msel 4 "$PY" scripts/build_train_cache.py --msel --workers "$W_MAIN" || rc=1
  run_stage check 0 "$PY" scripts/check_disjoint.py || rc=1
  return $rc
}

lane_in1k() {
  if ! { want in1k || want in100; }; then return 0; fi
  if ! hf_token_ok && ! done_in1k; then record in1k "failed (no token)" 0; return 1; fi
  local rc=0
  run_stage in1k 105 cmd_in1k || rc=1
  if want in100 && ! done_in100; then
    if [ -f scripts/build_imagenet100.py ]; then
      run_stage in100 5 cmd_in100 || rc=1
    else
      note "in100: scripts/build_imagenet100.py does not exist yet; rerun with --only in100 once it does"
      record in100 "waiting for scripts/build_imagenet100.py" 0
    fi
  fi
  return $rc
}

lane_tier2() { run_stage tier2 45 cmd_tier2; }
lane_models() {
  # shellcheck disable=SC2086
  run_stage models 20 "$PY" scripts/prefetch_models.py ${PREFETCH_ARGS:-}
}
asked() { [ "$1" = 1 ] || { [ -n "$ONLY" ] && want "$2"; }; }   # optional stage: its flag, or named in --only
lane_extra() {
  local rc=0
  if asked "$CUB" cub; then run_stage cub 3 cmd_cub || rc=1; fi
  if asked "$V2" v2; then
    local v2_marker="$MARK_HF/imagenetv2_matched_frequency.done"
    run_stage v2 3 cmd_hub_file "$V2_REPO" "$V2_REV" "$V2_FILE" "$HF_HOME/extracted" "$v2_marker" || rc=1
  fi
  if asked "$SKETCH" sketch; then
    run_stage sketch 8 cmd_hub_file "$SK_REPO" "$SK_REV" "$SK_FILE" "$HF_HOME/extracted/ImageNet-Sketch.zip" - || rc=1
  fi
  return $rc
}

if [ "$STATUS" = 1 ]; then
  for s in val train msel check in1k in100 tier2 models cub v2 sketch; do
    if "done_$s"; then echo "  $s: done"; else echo "  $s: not done"; fi
  done
  echo "  free $(avail_gb) GB; HF_HOME $(du_gb "$HF_HOME") GB; cache $(du_gb "$ROOT/cache") GB"
  exit 0
fi

note "h100_data run $RUN_ID: HF_HOME=$HF_HOME, $ncpu CPUs (workers $W_MAIN main, $W_SIDE side), budget $BUDGET_GB GB, optional: cub=$CUB v2=$V2 sketch=$SKETCH${ONLY:+, only $ONLY}${SKIP:+, skip $SKIP}"
disk_log start
T0=$(date +%s)
LANES=(lane_imagenet lane_in1k lane_tier2 lane_models lane_extra)
declare -A LRC
if [ "$SERIAL" = 1 ]; then
  for l in "${LANES[@]}"; do "$l"; LRC[$l]=$?; done
else
  declare -A PIDS
  for l in "${LANES[@]}"; do "$l" & PIDS[$l]=$!; done
  for l in "${LANES[@]}"; do wait "${PIDS[$l]}"; LRC[$l]=$?; done
fi
write_manifest > /dev/null 2>&1 || note "manifest: could not write results/data_check/data_manifest.json"
disk_log end
note "== summary of run $RUN_ID ($(( ($(date +%s) - T0) / 60 )) min)"
awk -F'\t' -v r="$RUN_ID" '$1 == r {printf "   %-8s %-40s %6s s\n", $3, $4, $5}' "$LOGDIR/status.tsv"
bad=0
for l in "${LANES[@]}"; do [ "${LRC[$l]}" = 0 ] || { note "$l: failed (rc ${LRC[$l]})"; bad=1; }; done
note "free $(avail_gb) GB; HF_HOME $(du_gb "$HF_HOME") GB; cache $(du_gb "$ROOT/cache") GB"
[ $bad = 0 ] && note "DATA_READY" || note "DATA_INCOMPLETE: rerun the same command to resume"
exit $bad
