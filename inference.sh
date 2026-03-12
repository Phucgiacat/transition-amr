#!/bin/bash
# =============================================================================
# inference.sh  -  Run AMR batch inference with resume support
# =============================================================================
# Edit the variables in the "Configuration" section to match your files,
# then run:
#
#   bash inference.sh
#
# If the run is interrupted (e.g. Colab GPU disconnect), simply run the same
# command again - it will automatically resume from where it stopped.
# =============================================================================

set -euo pipefail

# Configuration
# Paths are relative to the repo root (where this script lives).
INPUT_FILE="train.en(2).jsonl"
OUTPUT_FILE="train-amr.txt"
PROCESS_FILE="process.json"      # Set to "" to run without checkpointing
MODEL="AMR3-structbart-L"        # or AMR2-structbart-L
BATCH_SIZE=8

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${REPO_DIR}/.venv/bin/python"

echo "============================================================"
echo "  IBM Transition AMR Parser - Batch Inference"
echo "============================================================"
echo "  Repo    : ${REPO_DIR}"
echo "  Input   : ${INPUT_FILE}"
echo "  Output  : ${OUTPUT_FILE}"
echo "  Process : ${PROCESS_FILE:-<none>}"
echo "  Model   : ${MODEL}"
echo "  Batch   : ${BATCH_SIZE}"
echo "============================================================"

# Verify the virtual environment exists
if [ ! -f "${PYTHON}" ]; then
    echo "[ERROR] Virtual environment not found: ${PYTHON}"
    echo "  Run setup first:  bash ${REPO_DIR}/setup.sh"
    exit 1
fi

cd "${REPO_DIR}"

# Build the argument list dynamically
ARGS=(
    --input  "${INPUT_FILE}"
    --output "${OUTPUT_FILE}"
    --model  "${MODEL}"
    --batch-size "${BATCH_SIZE}"
)

# Only pass --process if PROCESS_FILE is non-empty
if [ -n "${PROCESS_FILE}" ]; then
    ARGS+=(--process "${PROCESS_FILE}")
fi

# Run the parser
"${PYTHON}" run.py "${ARGS[@]}"

echo ""
echo "============================================================"
echo "  Done."
echo "  Output : ${REPO_DIR}/${OUTPUT_FILE}"
if [ -n "${PROCESS_FILE}" ]; then
    echo "  To check progress: ${PYTHON} progress_utils.py --show ${PROCESS_FILE}"
fi
echo "============================================================"
