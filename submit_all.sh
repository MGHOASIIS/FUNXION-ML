#!/bin/bash
# =============================================================================
# submit_all.sh
# Submits all 96 experiments (6 tasks × 4 paradigms × 4 models) as independent
# SLURM jobs. Each job runs in parallel — all are queued at once.
#
# Usage:
#   bash submit_all.sh             # submit all 96
#   bash submit_all.sh --dry-run   # print commands without submitting
#   bash submit_all.sh --model rnn # submit only RNN jobs (24 jobs)
#   bash submit_all.sh --task 1    # submit only task 1 jobs (16 jobs)
#
# Options:
#   --dry-run       Print sbatch commands without actually submitting
#   --model MODEL   Only submit jobs for one model (hmm | cnn | rnn | transformer)
#   --task TASK     Only submit jobs for one task (1–6)
#   --paradigm P    Only submit jobs for one paradigm (1–4)
# =============================================================================

# NOTE: set -e intentionally omitted here — a single failed sbatch submission
# should not abort the entire loop. Failures are caught and reported per-job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/run_single.sh"
PROJECT_ROOT="/home/singh.vishwa/xdash2"
LOG_DIR="${PROJECT_ROOT}/logs"

# ── Parse optional filters ────────────────────────────────────────────────────
DRY_RUN=false
FILTER_MODEL=""
FILTER_TASK=""
FILTER_PARADIGM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true;          shift ;;
        --model)      FILTER_MODEL="$2";     shift 2 ;;
        --task)       FILTER_TASK="$2";      shift 2 ;;
        --paradigm)   FILTER_PARADIGM="$2";  shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash submit_all.sh [--dry-run] [--model MODEL] [--task TASK] [--paradigm P]"
            exit 1 ;;
    esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "ERROR: Job script not found at ${JOB_SCRIPT}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

# ── Define experiment space ───────────────────────────────────────────────────
TASKS=(1 2 3 4 5 6)
PARADIGMS=(1 2 3 4)
MODELS=(hmm cnn rnn transformer)

TASK_NAMES=([1]="jar_opening" [2]="key_turning" [3]="cleaning" [4]="back_washing" [5]="cutting" [6]="hammering")
PARADIGM_NAMES=([1]="patients_vs_controls" [2]="rct_vs_controls" [3]="other_vs_controls" [4]="rct_vs_other")

# ── Submission loop ───────────────────────────────────────────────────────────
SUBMITTED=0
FAILED=0

echo "============================================================"
echo " XDash HPC Job Submission"
echo " HPC System: $(hostname)"
echo " User: $(whoami)"
echo " Project: ${PROJECT_ROOT}"
if [ "${DRY_RUN}" = true ]; then
    echo " Mode: DRY RUN (no jobs will be submitted)"
else
    echo " Mode: LIVE SUBMISSION"
fi
[ -n "${FILTER_MODEL}" ]    && echo " Filter: model     = ${FILTER_MODEL}"
[ -n "${FILTER_TASK}" ]     && echo " Filter: task      = ${FILTER_TASK}"
[ -n "${FILTER_PARADIGM}" ] && echo " Filter: paradigm  = ${FILTER_PARADIGM}"
echo "============================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    if [ -n "${FILTER_MODEL}" ] && [ "${MODEL}" != "${FILTER_MODEL}" ]; then
        continue
    fi

    for TASK in "${TASKS[@]}"; do
        if [ -n "${FILTER_TASK}" ] && [ "${TASK}" != "${FILTER_TASK}" ]; then
            continue
        fi

        for PARADIGM in "${PARADIGMS[@]}"; do
            if [ -n "${FILTER_PARADIGM}" ] && [ "${PARADIGM}" != "${FILTER_PARADIGM}" ]; then
                continue
            fi

            JOB_NAME="${MODEL^^}_T${TASK}_P${PARADIGM}"
            TASK_NAME="${TASK_NAMES[$TASK]}"
            PARADIGM_NAME="${PARADIGM_NAMES[$PARADIGM]}"

            case "${MODEL}" in
                rnn)
                    TIME_LIMIT="08:00:00"
                    MEMORY="64G"
                    PARTITION="gpu"
                    ;;
                cnn)
                    TIME_LIMIT="06:00:00"
                    MEMORY="64G"
                    PARTITION="gpu"
                    ;;
                hmm)
                    TIME_LIMIT="04:00:00"
                    MEMORY="32G"
                    PARTITION="short"
                    ;;
                transformer)
                    TIME_LIMIT="08:00:00"
                    MEMORY="64G"
                    PARTITION="gpu"
                    ;;
            esac

            # Build sbatch command as an array — safe with spaces and special
            # characters, avoids the eval word-splitting pitfall.
            SBATCH_CMD=(sbatch
                --account=a.sathyanarayana
                --partition="${PARTITION}"
                --job-name="${JOB_NAME}"
                --time="${TIME_LIMIT}"
                --nodes=1
                --ntasks=1
                --cpus-per-task=8
                --mem="${MEMORY}"
                --output="${LOG_DIR}/${JOB_NAME}_%j.out"
                --error="${LOG_DIR}/${JOB_NAME}_%j.err"
                --mail-type=FAIL
                --mail-user=singh.vishwa@northeastern.edu
            )

            # GPU models get a GPU allocation; HMM is CPU-only
            if [ "${MODEL}" != "hmm" ]; then
                SBATCH_CMD+=(--gres=gpu:1)
            fi

            SBATCH_CMD+=("${JOB_SCRIPT}" "${TASK}" "${PARADIGM}" "${MODEL}")

            if [ "${DRY_RUN}" = true ]; then
                echo "[DRY RUN] ${JOB_NAME}"
                echo "  Task: ${TASK} (${TASK_NAME})"
                echo "  Paradigm: ${PARADIGM} (${PARADIGM_NAME})"
                echo "  Model: ${MODEL} | Time: ${TIME_LIMIT} | Memory: ${MEMORY}"
                echo "  Command: ${SBATCH_CMD[*]}"
                echo ""
                SUBMITTED=$((SUBMITTED + 1))
            else
                echo -n "[SUBMITTING] ${JOB_NAME} (${TASK_NAME}, ${PARADIGM_NAME})... "

                if JOB_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1); then
                    JOB_ID=$(echo "${JOB_OUTPUT}" | awk '{print $NF}')
                    echo "✅ Job ID: ${JOB_ID}"
                    SUBMITTED=$((SUBMITTED + 1))
                else
                    echo "❌ FAILED"
                    echo "  Error: ${JOB_OUTPUT}"
                    FAILED=$((FAILED + 1))
                fi

                sleep 0.3
            fi
        done
    done
done

echo ""
echo "============================================================"
if [ "${DRY_RUN}" = true ]; then
    echo " DRY RUN COMPLETE"
    echo " Would submit: ${SUBMITTED} jobs"
    echo ""
    echo " To actually submit:"
    echo "   bash submit_all.sh"
    echo ""
    echo " To submit specific subsets:"
    echo "   bash submit_all.sh --model rnn"
    echo "   bash submit_all.sh --task 1 --paradigm 1"
else
    echo " SUBMISSION COMPLETE"
    echo " Successfully submitted: ${SUBMITTED} jobs"
    if [ "${FAILED}" -gt 0 ]; then
        echo " Failed submissions:     ${FAILED} jobs"
    fi
    echo ""
    echo " MONITORING:"
    echo "   squeue -u singh.vishwa"
    echo "   squeue -u singh.vishwa --format='%.8i %.12P %.20j %.8T %.10M %.6D %R'"
    echo ""
    echo " MANAGEMENT:"
    echo "   scancel -u singh.vishwa              # Cancel all your jobs"
    echo "   scancel -u singh.vishwa --state=PENDING  # Cancel only pending"
    echo ""
    echo " LOGS:"
    echo "   ls ${LOG_DIR}/"
    echo "   tail -f ${LOG_DIR}/RNN_T1_P1_*.out"
    echo "   find ${LOG_DIR} -name '*.err' -size +0"
fi
echo "============================================================"