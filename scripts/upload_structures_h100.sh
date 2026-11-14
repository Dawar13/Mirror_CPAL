#!/usr/bin/env bash
# Upload the earlier runs' operator and structure files (not in git; laptop copies in artifacts/a100/) to the H100,
# at the paths the job files expect (PREREG_NEXT_PHASE.md Section 9, item 10). About 385 MB in all. Each file is
# copied to <path>.part and renamed only when complete, so a job waiting for it never reads a partial file. Files
# already present on the VM with the same sha256 are skipped. Run from the laptop (Git Bash) after the first ship:
#   H100_HOST=1.2.3.4 H100_PORT=41234 bash scripts/upload_structures_h100.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${H100_HOST:?set H100_HOST and H100_PORT}"
PORT="${H100_PORT:-22}"; RUSER="${H100_USER:-root}"; KEY="${H100_KEY:-$HOME/.ssh/vast_ed25519}"
DIR="${H100_DIR:-/workspace/Mirror_CPAL}"
OPTS=(-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
SSH=(ssh -p "$PORT" -i "$KEY" "${OPTS[@]}" "$RUSER@$H100_HOST")
SCP=(scp -P "$PORT" -i "$KEY" "${OPTS[@]}")
FILES=(
  runs/x2/perlayer_reg_s0/results/e1_operators.npz
  runs/x2/perlayer_reg_s1/results/e1_operators.npz
  runs/x2/perlayer_reg_s2/results/e1_operators.npz
  runs/g/deit3_b16_s0/results/e1_operators.npz
  runs/m1/m1_structure.npz
)
for f in "${FILES[@]}"; do
  src="artifacts/a100/$f"
  [ -f "$src" ] || { echo "missing on the laptop: $src"; exit 1; }
  sha=$(sha256sum "$src" | cut -d' ' -f1)
  remote=$("${SSH[@]}" "sha256sum '$DIR/$f' 2>/dev/null | cut -d' ' -f1" || true)
  if [ "$remote" = "$sha" ]; then echo "present: $f"; continue; fi
  echo "upload: $f ($(du -h "$src" | cut -f1), sha256 ${sha:0:12})"
  "${SSH[@]}" "mkdir -p '$DIR/$(dirname "$f")'"
  "${SCP[@]}" "$src" "$RUSER@$H100_HOST:$DIR/$f.part"
  "${SSH[@]}" "cd '$DIR' && [ \"\$(sha256sum '$f.part' | cut -d' ' -f1)\" = '$sha' ] && mv -f '$f.part' '$f'"
done
echo "done: ${#FILES[@]} files on $H100_HOST:$DIR"
