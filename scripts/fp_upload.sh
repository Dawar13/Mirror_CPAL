#!/usr/bin/env bash
# Upload the git-ignored inputs of the final phase from the laptop (Git Bash) to the H200 VM, at the paths the jobs of
# jobs/final_phase.yaml expect (about 1.7 GB in all). Order: what Gate P needs first, then A02, then the other models,
# then P11.2's checkpoint. Each file goes to <path>.part and is renamed only when complete, so a job waiting for it
# never reads a partial file; a file already on the VM with the same sha256 is skipped (rerun after a broken link).
#
#   H100_HOST=1.2.3.4 H100_PORT=22 H100_USER=ubuntu bash scripts/fp_upload.sh
#
# Settings as scripts/ship_h100.sh: H100_HOST (required), H100_PORT [22], H100_USER [root], H100_KEY
# [~/.ssh/vast_ed25519], H100_DIR [/home/Mirror_CPAL]. Ship the code first (FIRST=1 bash scripts/ship_h100.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${H100_HOST:?set H100_HOST (and H100_PORT, H100_USER)}"
PORT="${H100_PORT:-22}"; RUSER="${H100_USER:-root}"; KEY="${H100_KEY:-$HOME/.ssh/vast_ed25519}"
DIR="${H100_DIR:-/home/Mirror_CPAL}"
OPTS=(-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
SSH=(ssh -p "$PORT" -i "$KEY" "${OPTS[@]}" "$RUSER@$H100_HOST")
SCP=(scp -P "$PORT" -i "$KEY" "${OPTS[@]}")
FILES=(
  results/MUST-07/structure/dinov2_vitb14_reg_s0.npz                      # Gate P, A01, the main arms
  runs/x2/perlayer_reg_s0/results/e1_operators.npz                        # A02: exact refits of fit sets 0 to 2
  runs/x2/perlayer_reg_s1/results/e1_operators.npz
  runs/x2/perlayer_reg_s2/results/e1_operators.npz
  results/MUST-07/structure/dinov2_vitb14_reg_s1.npz                      # A02 references
  results/MUST-07/structure/dinov2_vitb14_reg_s2.npz
  results/MUST-07/structure/deit3_b16_s0.npz                              # A08 DeiT-III-B
  results/MUST-07/structure/siglip2_b16_s0.npz                            # A15a
  results/MUST-07/structure/dinov2_vitl14_reg_s0.npz                      # A08 DINOv2-L (cut order 2)
  runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/structure.npz            # P11.2 on the MUST-09 model
  runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/healed_state.pt
  results/MUST-05/dinov2_vitb14_reg/teaser.npz                           # A14 teaser: the earlier pairs and images
)
total=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing on the laptop: $f"; exit 1; }
  total=$(( total + $(stat -c %s "$f") ))
done
echo "uploading up to $(( total / 1024 / 1024 )) MB to $RUSER@$H100_HOST:$DIR"
for f in "${FILES[@]}"; do
  sha=$(sha256sum "$f" | cut -d' ' -f1)
  remote=$("${SSH[@]}" "sha256sum '$DIR/$f' 2>/dev/null | cut -d' ' -f1" || true)
  if [ "$remote" = "$sha" ]; then echo "present: $f"; continue; fi
  echo "upload: $f ($(du -h "$f" | cut -f1), sha256 ${sha:0:12})"
  "${SSH[@]}" "mkdir -p '$DIR/$(dirname "$f")'"
  "${SCP[@]}" "$f" "$RUSER@$H100_HOST:$DIR/$f.part"
  got=$("${SSH[@]}" "sha256sum '$DIR/$f.part' | cut -d' ' -f1")
  if [ "$got" != "$sha" ]; then echo "checksum mismatch for $f (try again)"; exit 1; fi
  "${SSH[@]}" "mv '$DIR/$f.part' '$DIR/$f'"
done
echo "all ${#FILES[@]} files present on the VM with matching sha256"
