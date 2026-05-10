#!/usr/bin/env bash
# sync_checkpoints_to_hf.sh — upload all best.pt files from a tier4 run dir
# to the shared HF model repo for cross-machine access.
#
# Usage:
#   ./sync_checkpoints_to_hf.sh <tier4_run_dir> [machine_label]
#
# Example:
#   ./sync_checkpoints_to_hf.sh expL_singletask_regularized_20260510_215030 machine-A
#   ./sync_checkpoints_to_hf.sh expL_5_6dof_regularized_20260510_xxxxxx     machine-B
#
# Each best.pt is uploaded with a descriptive filename:
#   <experiment>_<machine>_<dof_seed>_<timestamp>.pt
# so multiple machines can push without overwriting each other.
#
# Sister command (download): hf download Chissanupong/kuka-iiwa-meta-kin-checkpoints \
#                              --include "expL_*" --local-dir ./fetched_ckpts/
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <tier4_run_dir> [machine_label]" >&2
    echo "Example: $0 expL_singletask_regularized_20260510_215030 machine-A" >&2
    exit 1
fi

RUN_DIR="$1"
MACHINE="${2:-$(hostname -s)}"
HF_REPO="Chissanupong/kuka-iiwa-meta-kin-checkpoints"

ROOT=/home/ubuntu/wish/kuka_kinematic_learner/original/ext_script/scripts/train
TIER4="$ROOT/tier4_runs"
SRC="$TIER4/$RUN_DIR"

if [ ! -d "$SRC" ]; then
    echo "Error: $SRC not found." >&2
    exit 1
fi

# Pull the timestamp out of the run dir name (last token after the last underscore-pair)
TS=$(echo "$RUN_DIR" | grep -oE '[0-9]{8}_[0-9]{6}' | tail -1)
EXP=$(echo "$RUN_DIR" | sed -E 's/_[0-9]{8}_[0-9]{6}$//')

echo "=== Sync checkpoints to $HF_REPO ==="
echo "  source dir: $SRC"
echo "  experiment: $EXP"
echo "  timestamp:  $TS"
echo "  machine:    $MACHINE"
echo

found=0
for ckpt in "$SRC"/*/tb/fk/fk_pose_best.pt; do
    if [ ! -f "$ckpt" ]; then continue; fi
    found=$((found + 1))
    # Pull the dof/seed tag from the path (e.g., dof5_seed42 or dof7_seed42_part000)
    tag=$(basename "$(dirname "$(dirname "$(dirname "$ckpt")")")")
    remote_name="${EXP}_${MACHINE}_${tag}_${TS}.pt"
    echo "[$found] $tag -> $remote_name"
    hf upload "$HF_REPO" "$ckpt" "$remote_name" --quiet 2>&1 | tail -3
    echo
done

if [ "$found" -eq 0 ]; then
    echo "[!] No best.pt files found under $SRC. Was training completed?"
    exit 1
fi

echo "=== Uploaded $found checkpoints ==="
echo "Pull from the other machine with:"
echo "  hf download $HF_REPO --include '${EXP}_*' --local-dir ./fetched_ckpts/"
