#!/usr/bin/env bash
# Run after `git clone` to make the tracked Stage-2 checkpoint available
# at the path the experiment scripts expect.
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$THIS_DIR/multitask_fk_best.pt"
TARGET_DIR="$(dirname "$THIS_DIR")/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7"
TARGET="$TARGET_DIR/multitask_fk_best.pt"

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found. Did you forget to git lfs pull or git pull?"
    exit 1
fi

mkdir -p "$TARGET_DIR"
ln -sf "$SRC" "$TARGET"
echo "[OK] Symlinked $TARGET -> $SRC"
echo "Now you can run: bash tier4_runs/expD_seeds_4to10_7dof.sh"
