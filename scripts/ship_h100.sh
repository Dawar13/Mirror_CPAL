#!/usr/bin/env bash
# Ship the committed code (HEAD) plus .git from the laptop (Git Bash) to the H100. No git push or pull.
# Uncommitted changes are not shipped (they are listed). On the VM, code files are replaced, but the top-level
# runs/, results/ and logs/ are never overwritten: files the VM lacks there are added, existing ones are kept.
# FIRST=1 (a fresh machine) extracts everything. Excludes are anchored at the top level, so committed test
# fixtures such as tests/fixtures/collect_results/e3_tiny/runs/ do ship.
#
#   H100_HOST=1.2.3.4 H100_PORT=41234 FIRST=1 bash scripts/ship_h100.sh     # first time
#   H100_HOST=1.2.3.4 H100_PORT=41234 bash scripts/ship_h100.sh             # later
#
# Settings: H100_HOST (required), H100_PORT [22], H100_USER [root], H100_KEY [~/.ssh/vast_ed25519],
# H100_DIR [/home/Mirror_CPAL], FIRST [0].
# Jobs already running keep the code they loaded; jobs started later use the shipped code (they are listed).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${H100_HOST:?set H100_HOST and H100_PORT (vast.ai Connect dialog)}"
PORT="${H100_PORT:-22}"; RUSER="${H100_USER:-root}"; KEY="${H100_KEY:-$HOME/.ssh/vast_ed25519}"
DIR="${H100_DIR:-/home/Mirror_CPAL}"; FIRST="${FIRST:-0}"
OPTS=(-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
SSH=(ssh -p "$PORT" -i "$KEY" "${OPTS[@]}" "$RUSER@$H100_HOST")
SCP=(scp -P "$PORT" -i "$KEY" "${OPTS[@]}")

COMMIT=$(git rev-parse HEAD)
echo "shipping $(git log -1 --format='%h %s (%ci)') to $RUSER@$H100_HOST:$DIR (FIRST=$FIRST)"
DIRTY=$(git status --porcelain --untracked-files=no)
if [ -n "$DIRTY" ]; then
  echo "warning: uncommitted changes are NOT shipped:"
  echo "$DIRTY" | head -20
fi
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
git -c core.autocrlf=false archive --format=tar.gz -o "$S/mirror_code_ship.tgz" HEAD
# Exit code 1 means a file changed while it was read (an editor refreshing .git/index); objects never change.
tar --force-local --warning=no-file-changed -czf "$S/mirror_git_ship.tgz" .git || [ $? -eq 1 ]
echo "archives: code $(du -h "$S/mirror_code_ship.tgz" | cut -f1), .git $(du -h "$S/mirror_git_ship.tgz" | cut -f1)"
PARENT=$(dirname "$DIR")
"${SSH[@]}" "mkdir -p '$PARENT'"
"${SCP[@]}" "$S/mirror_code_ship.tgz" "$S/mirror_git_ship.tgz" "$RUSER@$H100_HOST:$PARENT/"

"${SSH[@]}" "bash -s -- '$DIR' '$FIRST' '$COMMIT'" << 'REMOTE'
set -euo pipefail
DIR=$1; FIRST=$2; COMMIT=$3; P=$(dirname "$DIR")
CODE="$P/mirror_code_ship.tgz"; GITT="$P/mirror_git_ship.tgz"
mkdir -p "$DIR"; cd "$DIR"
if [ "$FIRST" = 1 ]; then
  tar -xzf "$CODE" --no-same-owner
else
  tar -xzf "$CODE" --no-same-owner --anchored --exclude=runs --exclude=results --exclude=logs
  tops=$(tar -tzf "$CODE" | awk -F/ '{print $1}' | sort -u | grep -xE 'runs|results|logs' || true)
  # shellcheck disable=SC2086
  [ -z "$tops" ] || tar -xzf "$CODE" --no-same-owner --skip-old-files $tops
fi
rm -rf .git.ship && mkdir .git.ship && tar -xzf "$GITT" -C .git.ship --no-same-owner
rm -rf .git && mv .git.ship/.git .git && rmdir .git.ship
rm -f "$CODE" "$GITT"
head=$(cat .git/HEAD)
case "$head" in
  "ref: "*) ref=${head#ref: }
            if [ -f ".git/$ref" ]; then got=$(cat ".git/$ref"); else got=$(grep " $ref\$" .git/packed-refs | cut -d' ' -f1); fi ;;
  *) got=$head ;;
esac
if command -v git > /dev/null; then
  git config --global --get-all safe.directory | grep -qx "$DIR" || git config --global --add safe.directory "$DIR"
  # The index comes from Windows (other stat data, maybe staged changes): rebuild it from HEAD so that the
  # run metadata's git_dirty compares the files with HEAD. The files themselves are not touched.
  rm -f .git/index && git reset -q
fi
echo "VM now at commit $got"
[ "$got" = "$COMMIT" ] || { echo "ERROR: expected $COMMIT"; exit 1; }
running=$(ps -eo pid,etime,args | grep -E 'python[0-9.]* .*scripts/|h100_sched|vm_sched' | grep -v grep || true)
[ -z "$running" ] || { echo "running jobs keep the code they loaded:"; echo "$running" | cut -c1-200; }
REMOTE
echo "shipped $COMMIT"
