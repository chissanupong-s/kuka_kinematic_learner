#!/usr/bin/env bash
# Phase 2 chain: run expH (7-DoF adapt n=3, ~4h) -> aggregate -> expG
# (single-task n=3 at 15M, ~31h) -> aggregate. Total ~35h.
#
# Each step is gated on the previous one succeeding. If anything fails the
# chain stops and a banner is written to phase2_chain.log so it's easy to
# spot on return.
#
# Launch from tmux so it survives SSH disconnects:
#   tmux new -d -s phase2_chain "cd .../tier4_runs && ./phase2_chain.sh"
#
# Detach: Ctrl-B then D. Re-attach: `tmux attach -t phase2_chain`.
set -uo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAIN_LOG="$THIS_DIR/phase2_chain.log"
exec > >(tee -a "$CHAIN_LOG") 2>&1

banner () {
    echo
    echo "============================================================"
    echo "[$(date)] $*"
    echo "============================================================"
}

banner "Phase 2 chain starting"

# --- Step 1: expH (7-DoF adapt, n=3, part000) ---
banner "Step 1/4: expH_adapt_7dof_n3_part000.sh"
if ! "$THIS_DIR/expH_adapt_7dof_n3_part000.sh"; then
    banner "FAIL: expH errored. Aborting chain."
    exit 1
fi

# Find the most recent expH output dir
EXPH_OUT="$(ls -dt "$THIS_DIR"/expH_adapt_7dof_n3_part000_*/ 2>/dev/null | head -1)"
EXPH_OUT="${EXPH_OUT%/}"
if [ -z "$EXPH_OUT" ] || [ ! -d "$EXPH_OUT" ]; then
    banner "FAIL: could not locate expH output dir."
    exit 1
fi

banner "Step 2/4: aggregate expH ($EXPH_OUT)"
python3 "$THIS_DIR/aggregate_adapt_7dof_n3.py" "$EXPH_OUT" || true  # don't fail chain on aggregator error

# --- Step 3: expG (single-task n=3 at 15M) ---
banner "Step 3/4: expG_singletask_multiseed_15M.sh"
if ! "$THIS_DIR/expG_singletask_multiseed_15M.sh"; then
    banner "FAIL: expG errored. Aggregator skipped."
    exit 1
fi

# Find the most recent expG output dir
EXPG_OUT="$(ls -dt "$THIS_DIR"/expG_singletask_multiseed_15M_*/ 2>/dev/null | head -1)"
EXPG_OUT="${EXPG_OUT%/}"
if [ -z "$EXPG_OUT" ] || [ ! -d "$EXPG_OUT" ]; then
    banner "FAIL: could not locate expG output dir."
    exit 1
fi

banner "Step 4/4: aggregate expG ($EXPG_OUT)"
python3 "$THIS_DIR/aggregate_singletask_multiseed.py" "$EXPG_OUT" || true

banner "Phase 2 chain complete"
echo "  expH output: $EXPH_OUT"
echo "  expG output: $EXPG_OUT"
echo "  Full log:    $CHAIN_LOG"
