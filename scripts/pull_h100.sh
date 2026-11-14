#!/usr/bin/env bash
# Bring results back from the H100 to the laptop (Git Bash; rsync is used when both ends have it, else tar over ssh).
# Pulled: results/ (any file type) and the small files of runs/ and logs/ (csv, json, jsonl, md, png, pdf, svg, log,
# txt, tsv, yaml, html). Skipped and listed: files above PULL_MAX_MB, .pt .pth .ckpt .safetensors .npy files, other
# types under runs/ and logs/, and results/STATUS.md (kept on the laptop). A local file that is newer than (or as
# old as) the VM's copy is kept, so laptop edits survive.
#
#   H100_HOST=1.2.3.4 H100_PORT=41234 bash scripts/pull_h100.sh
#   DRY_RUN=1 H100_HOST=... bash scripts/pull_h100.sh        # list only
#
# Settings: H100_HOST (required), H100_PORT [22], H100_USER [root], H100_KEY [~/.ssh/vast_ed25519],
# H100_DIR [/home/Mirror_CPAL], PULL_MAX_MB [20], PULL_DIRS ["results runs logs"], PULL_DEST [the repo],
# PULL_EXCLUDE [an extended regex on the relative path; default ^results/STATUS\.md$], DRY_RUN [0].
# The skipped list goes to <PULL_DEST>/logs/pull_h100/skipped_<time>.tsv (size in bytes, reason, path).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
: "${H100_HOST:?set H100_HOST and H100_PORT (vast.ai Connect dialog)}"
PORT="${H100_PORT:-22}"; RUSER="${H100_USER:-root}"; KEY="${H100_KEY:-$HOME/.ssh/vast_ed25519}"
DIR="${H100_DIR:-/home/Mirror_CPAL}"; MAX_MB="${PULL_MAX_MB:-20}"; DIRS="${PULL_DIRS:-results runs logs}"
DEST="${PULL_DEST:-$REPO}"; EXCL="${PULL_EXCLUDE:-^results/STATUS\.md$}"; DRY="${DRY_RUN:-0}"
OPTS=(-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
SSH=(ssh -p "$PORT" -i "$KEY" "${OPTS[@]}" "$RUSER@$H100_HOST")
STAMP=$(date +%Y%m%d_%H%M%S)
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

# One remote call: the VM's commit, then "<size><tab><path>" for every file under PULL_DIRS.
"${SSH[@]}" "bash -s -- '$DIR' $DIRS" > "$W/remote.txt" << 'REMOTE'
cd "$1" || exit 1
shift
h=$(cat .git/HEAD 2> /dev/null)
case "$h" in
  "ref: "*) r=${h#ref: }; c=$(cat ".git/$r" 2> /dev/null || grep " $r\$" .git/packed-refs | cut -d' ' -f1) ;;
  *) c=$h ;;
esac
echo "COMMIT ${c:-unknown}"
for d in "$@"; do [ -d "$d" ] && find "$d" -type f -printf '%s\t%p\n'; done
true
REMOTE
echo "VM $RUSER@$H100_HOST:$DIR at commit $(sed -n 's/^COMMIT //p' "$W/remote.txt" | cut -c1-12)"
grep -v '^COMMIT ' "$W/remote.txt" > "$W/all.tsv" || true
EXCL="$EXCL" awk -F'\t' -v max=$((MAX_MB * 1024 * 1024)) -v pull="$W/pull.txt" -v skip="$W/skip.tsv" '
  BEGIN { small = "\\.(csv|json|jsonl|md|png|pdf|svg|log|txt|tsv|yaml|yml|html)$"; excl = ENVIRON["EXCL"] }
  {
    size = $1; path = $2; why = ""
    if (path ~ /\.(pt|pth|ckpt|safetensors|npy)$/) why = "checkpoint or array"
    else if (size + 0 > max) why = "above size limit"
    else if (excl != "" && path ~ excl) why = "excluded"
    else if (path !~ /^results\// && tolower(path) !~ small) why = "file type"
    if (why == "") { print path > pull; n++; bytes += size }
    else { printf "%s\t%s\t%s\n", size, why, path > skip; ns++; sbytes += size }
  }
  END { printf "pull %d files (%.1f MB); skip %d files (%.1f MB)\n", n, bytes / 1048576, ns, sbytes / 1048576 }
' "$W/all.tsv"
touch "$W/pull.txt" "$W/skip.tsv"
mkdir -p "$DEST/logs/pull_h100"
cp "$W/skip.tsv" "$DEST/logs/pull_h100/skipped_$STAMP.tsv"
if [ -s "$W/skip.tsv" ]; then
  echo "skipped, by reason:"
  awk -F'\t' '{ c[$2]++; b[$2] += $1 } END { for (r in c) printf "  %-20s %6d files %10.1f MB\n", r, c[r], b[r] / 1048576 }' "$W/skip.tsv"
  echo "largest skipped:"
  sort -t$'\t' -k1,1nr "$W/skip.tsv" | head -10 | awk -F'\t' '{ printf "  %9.1f MB  %s  (%s)\n", $1 / 1048576, $3, $2 }'
  echo "full list: $DEST/logs/pull_h100/skipped_$STAMP.tsv"
fi
if [ "$DRY" = 1 ]; then echo "DRY_RUN=1: nothing copied. First files:"; head -20 "$W/pull.txt"; exit 0; fi
[ -s "$W/pull.txt" ] || { echo "nothing to pull"; exit 0; }
mkdir -p "$DEST"
if command -v rsync > /dev/null && "${SSH[@]}" "command -v rsync" > /dev/null 2>&1; then
  echo "copying with rsync (newer local files are kept)"
  rsync -a --update --files-from="$W/pull.txt" -e "ssh -p $PORT -i $KEY ${OPTS[*]}" "$RUSER@$H100_HOST:$DIR/" "$DEST/"
else
  echo "copying with tar over ssh (newer local files are kept)"
  "${SSH[@]}" "cd '$DIR' && tar -czf - -T -" < "$W/pull.txt" \
    | tar -xzf - -C "$DEST" --keep-newer-files --warning=no-ignore-newer || {
      rc=$?; [ $rc -eq 1 ] && echo "tar reported differences (files changed while reading); check them" || exit $rc; }
fi
echo "pulled $(wc -l < "$W/pull.txt") files into $DEST"
