#!/usr/bin/env bash
# Make the tracked Stage-2 shared checkpoint available at the path the
# experiment scripts expect (runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt).
#
# Run this once after `git clone` on any machine.
#
# Behaviour:
#   - If the checkpoint is already at the expected path: nothing to do.
#   - If model_tracking/multitask_fk_best.pt exists: symlink it to the runs/ path.
#   - Otherwise: explain how to fix.
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$THIS_DIR/multitask_fk_best.pt"
TRAIN_DIR="$(dirname "$THIS_DIR")"
TARGET_DIR="$TRAIN_DIR/runs/multitask/seperate_weight/1/fk_iiwa_5_6_7"
TARGET="$TARGET_DIR/multitask_fk_best.pt"

# Case 1: already in place as a real file (e.g., the original machine where Stage-2 was trained)
if [ -f "$TARGET" ] && [ ! -L "$TARGET" ]; then
    echo "[OK] $TARGET already exists (not a symlink). Nothing to do."
    exit 0
fi

# Case 2: symlink already in place and points to a valid file
if [ -L "$TARGET" ] && [ -e "$TARGET" ]; then
    echo "[OK] $TARGET is already a symlink to $(readlink "$TARGET"). Nothing to do."
    exit 0
fi

# Case 3: tracked checkpoint exists; create the symlink
if [ -f "$SRC" ]; then
    mkdir -p "$TARGET_DIR"
    ln -sf "$SRC" "$TARGET"
    echo "[OK] Created symlink:"
    echo "     $TARGET"
    echo "  -> $SRC"
    echo ""
    echo "Now you can run any tier4_runs experiment, e.g.:"
    echo "     cd $TRAIN_DIR"
    echo "     bash tier4_runs/expD_seeds_4to10_7dof.sh"
    exit 0
fi

# Case 4: missing on disk — explain
cat <<EOF
[ERROR] Stage-2 checkpoint not found at:
        $SRC

This usually means the file wasn't pulled with the clone. Possible fixes:

  1. Standard pull:
        cd $(dirname "$(dirname "$(dirname "$(dirname "$THIS_DIR")")")")
        git pull origin main

  2. If you cloned with --depth 1 (shallow), unshallow first:
        git fetch --unshallow
        git pull origin main

  3. If the file is genuinely missing from the repo, retrain Stage 2:
        cd "$TRAIN_DIR"
        bash train5_6.sh         # Stage 1 for 5- and 6-DoF
        bash run_7_only_1.sh     # Stage 1 for 7-DoF
        # Then run your existing Stage 2 trainer; output is at:
        # runs/multitask/seperate_weight/1/fk_iiwa_5_6_7/multitask_fk_best.pt

After fixing, re-run this script.
EOF
exit 1
