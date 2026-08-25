#!/bin/bash
# =============================================================================
# run_ablation.sh
# SLURM job script for a single XDash ABLATION experiment.
# Called by submit_ablations.sh — do not run directly unless testing.
#
# Arguments (positional, all required):
#   $1  TASK             (1–6)
#   $2  PARADIGM         (1–4)
#   $3  MODEL            (hmm | cnn | rnn | transformer)
#   $4  METHOD           (truncate | padding | sliding_window |
#                         dtw_embedding | downsample_truncate |
#                         variable_length | phase_shift)
#   $5  ABLATION_GROUP   short label used in job name & log file
#                        e.g. "freq25", "pad", "dtw_mds10", "shift25"
#   $6  EXTRA_ARGS       any additional python flags as a single quoted string
#                        e.g. "--target-rate 25 --original-rate 50"
#                        pass "" if none
#   $7  DATA_SOURCE      standard | event_window  (default: standard)
#
# Manual test examples:
#   sbatch --time=06:00:00 run_ablation.sh 1 1 rnn padding           pad       ""        
#   sbatch --time=06:00:00 run_ablation.sh 1 1 rnn sliding_window    win300    "--window-size 300 --overlap 0.3" 
#   sbatch --time=06:00:00 run_ablation.sh 1 1 rnn phase_shift       shift25   "--shift-fraction 0.25" 
#   sbatch --time=48:00:00 run_ablation.sh 1 1 hmm variable_length   vl        ""        event_window
#   sbatch --time=48:00:00 run_ablation.sh 1 1 hmm dtw_embedding     dtw_mds10 "--n-components 10 --dtw-method mds" 
# =============================================================================

# ── Static SBATCH directives ──────────────────────────────────────────────────
#SBATCH --account=a.sathyanarayana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL

# =============================================================================
# Script starts here
# =============================================================================

if [ "$#" -lt 6 ] || [ "$#" -gt 7 ]; then
    echo "ERROR: Expected 6 or 7 arguments: TASK PARADIGM MODEL METHOD ABLATION_GROUP EXTRA_ARGS [DATA_SOURCE]"
    echo "Usage: sbatch run_ablation.sh <task> <paradigm> <model> <method> <ablation_group> \"<extra_args>\" [standard|event_window]"
    exit 1
fi

TASK=$1
PARADIGM=$2
MODEL=$3
METHOD=$4
ABLATION_GROUP=$5
EXTRA_ARGS=$6
DATA_SOURCE=${7:-standard}

# Short tag embedded in job/experiment names to distinguish data sources
if [ "${DATA_SOURCE}" = "event_window" ]; then DS_TAG="EW"; else DS_TAG="SBJ"; fi

PROJECT_ROOT="/home/singh.vishwa/xdash2"
ENV_NAME="xdash"
LOG_DIR="${PROJECT_ROOT}/logs/ablations"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " Ablation Job: ABL_${DS_TAG}_${ABLATION_GROUP^^}_${MODEL^^}_T${TASK}_P${PARADIGM}"
echo " Task:         ${TASK}"
echo " Paradigm:     ${PARADIGM}"
echo " Model:        ${MODEL}"
echo " Method:       ${METHOD}"
echo " Ablation:     ${ABLATION_GROUP}"
echo " Extra args:   ${EXTRA_ARGS}"
echo " Data source:  ${DATA_SOURCE}"
echo " Node:         $(hostname)"
echo " Start:        $(date)"
echo "============================================================"

# ── Verify conda env ──────────────────────────────────────────────────────────
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "ERROR: conda env '${ENV_NAME}' not found. Run setup_env.sh first."
    exit 1
fi

# ── GPU visibility ────────────────────────────────────────────────────────────
if [ "${MODEL}" != "hmm" ]; then
    echo ""
    echo "[INFO] CUDA devices:"
    conda run -n "${ENV_NAME}" python -c "
import torch
print(f'  Available: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
fi

cd "${PROJECT_ROOT}"

# ── Build python command ───────────────────────────────────────────────────────
BASE_ARGS="--task ${TASK} \
    --paradigm ${PARADIGM} \
    --model ${MODEL} \
    --method ${METHOD} \
    --data-source ${DATA_SOURCE} \
    --save-checkpoints \
    --diagnostics \
    --experiment-name ABL_${DS_TAG}_${ABLATION_GROUP}_T${TASK}_P${PARADIGM}"

# HMM-specific additions
if [ "${MODEL}" == "hmm" ]; then
    BASE_ARGS="${BASE_ARGS} --hmm-csv-dir storage/raw/xdash/events/"
else
    BASE_ARGS="${BASE_ARGS} --patience 15 --min-delta 1e-4"
fi

# Append method-specific hyperparameter flags
FULL_ARGS="${BASE_ARGS} ${EXTRA_ARGS}"

echo ""
echo "[INFO] Command: conda run -n ${ENV_NAME} python main.py ${FULL_ARGS}"
echo ""

conda run -n "${ENV_NAME}" python main.py ${FULL_ARGS}
EXIT_CODE=$?

echo ""
echo "============================================================"
echo " End:       $(date)"
echo " Exit code: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}