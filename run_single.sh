#!/bin/bash
# =============================================================================
# run_single.sh
# SLURM job script for a single XDash experiment.
# Called by submit_all.sh with arguments — do not run directly.
#
# Arguments:
#   $1  TASK      (1–6)
#   $2  PARADIGM  (1–4)
#   $3  MODEL     (hmm | cnn | rnn | transformer)
#
# Usage (manual, for testing):
#   sbatch --time=06:00:00 run_single.sh 1 1 rnn
# =============================================================================

# ── Static SBATCH directives (NO executable code between these) ───────────────
# All resource flags (--partition, --gres, --mem, --time, --job-name,
# --output, --error) are passed by submit_all.sh on the sbatch command line.
# Only truly static defaults live here as fallbacks for manual runs.
#SBATCH --account=a.sathyanarayana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL

# =============================================================================
# Script starts here
# =============================================================================

# Validate arguments
if [ "$#" -ne 3 ]; then
    echo "ERROR: Expected 3 arguments: TASK PARADIGM MODEL"
    echo "Usage: sbatch run_single.sh <task> <paradigm> <model>"
    exit 1
fi

TASK=$1
PARADIGM=$2
MODEL=$3

PROJECT_ROOT="/home/singh.vishwa/xdash2"
ENV_NAME="xdash"
LOG_DIR="${PROJECT_ROOT}/logs"

echo "============================================================"
echo " Job:      ${MODEL^^}_T${TASK}_P${PARADIGM}"
echo " Task:     ${TASK}"
echo " Paradigm: ${PARADIGM}"
echo " Model:    ${MODEL}"
echo " Node:     $(hostname)"
echo " Start:    $(date)"
echo "============================================================"

# ── Verify conda env exists ────────────────────────────────────────────────────
# SLURM does not source .bashrc, so 'conda activate' is unavailable.
# We use 'conda run' instead — it activates the env for a single command
# without needing the shell integration to be initialised.
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "ERROR: conda env '${ENV_NAME}' not found. Run setup_env.sh first."
    exit 1
fi

# ── Confirm GPU visibility ─────────────────────────────────────────────────────
echo ""
echo "[INFO] CUDA devices:"
conda run -n "${ENV_NAME}" python -c "
import torch
print(f'  Available: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ── Move to project root so relative imports resolve ──────────────────────────
cd "${PROJECT_ROOT}"

# ── Build python command ───────────────────────────────────────────────────────
ARGS="--task ${TASK} \
    --paradigm ${PARADIGM} \
    --model ${MODEL} \
    --method truncate \
    --patience 15 \
    --min-delta 1e-4 \
    --save-checkpoints"

# Diagnostics for RNN, CNN and Transformer only
if [ "${MODEL}" != "hmm" ]; then
    ARGS="${ARGS} --diagnostics "
fi

if [ "${MODEL}" == "hmm" ]; then
    ARGS="${ARGS} --hmm-csv-dir data/events/ "

echo ""
echo "[INFO] Command: conda run -n ${ENV_NAME} python main.py ${ARGS}"
echo ""

# ── Run via conda run (handles env activation without shell integration) ───────
conda run -n "${ENV_NAME}" python main.py ${ARGS}
EXIT_CODE=$?

echo ""
echo "============================================================"
echo " End: $(date)"
echo " Exit code: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
