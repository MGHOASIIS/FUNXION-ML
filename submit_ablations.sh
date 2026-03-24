#!/bin/bash
# =============================================================================
# submit_ablations.sh
# Queues ablation experiments as independent SLURM jobs.
#
# Flags (all optional, combinable):
#   --method METHOD       truncate | padding | variable_length | sliding_window |
#                         phase_shift | dtw_embedding
#                         Can be passed multiple times for a sweep.
#                         Default: variable_length
#
#   --freq HZ             Sampling frequency. Can be passed multiple times.
#                         If HZ < 50, sequences are resampled first.
#                         Default: 50 (no resampling)
#
#   --window-size W       Window size for sliding_window. Multiple values allowed.
#                         Default: 300
#   --overlap O           Overlap fraction for sliding_window. Multiple values allowed.
#                         Default: 0.30
#
#   --shift-fraction S    Shift fraction for phase_shift. Multiple values allowed.
#                         Default: 0.1
#
#   --dtw-components N    n_components for dtw_embedding. Multiple values allowed.
#                         Default: 10
#   --dtw-method M        mds | isomap | tsne. Multiple values allowed.
#                         Default: mds
#
#   --model MODEL         hmm | cnn | rnn | transformer (filter, default: all)
#   --task TASK           1–6 (filter, default: all)
#   --paradigm P          1–4 (filter, default: all)
#   --data-source S       subject | event_window (default: subject)
#   --dry-run             Preview without submitting
#
# Examples:
#   # variable_length at 50 Hz — all models, tasks, paradigms
#   bash submit_ablations.sh
#
#   # truncate at 20 Hz
#   bash submit_ablations.sh --method truncate --freq 20
#
#   # freq sweep: variable_length at 50, 30, 20, 10 Hz
#   bash submit_ablations.sh --freq 50 --freq 30 --freq 20 --freq 10
#
#   # method sweep: truncate and padding at 30 Hz
#   bash submit_ablations.sh --method truncate --method padding --freq 30
#
#   # DTW with tsne at 20 Hz
#   bash submit_ablations.sh --method dtw_embedding --freq 20 \
#       --dtw-components 3 --dtw-method tsne
#
#   # sliding window combinations
#   bash submit_ablations.sh --method sliding_window \
#       --window-size 150 --window-size 300 --overlap 0.30 --overlap 0.50
#
#   # phase shift sweep
#   bash submit_ablations.sh --method phase_shift \
#       --shift-fraction 0.0 --shift-fraction 0.1 --shift-fraction 0.25
#
#   # everything combined, single model, dry-run first
#   bash submit_ablations.sh --method truncate --method padding \
#       --freq 50 --freq 30 --freq 20 --freq 10 \
#       --model rnn --task 1 --dry-run
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/run_ablation.sh"
PROJECT_ROOT="/home/singh.vishwa/xdash2"
LOG_DIR="${PROJECT_ROOT}/logs/ablations"

# ── Parse CLI ─────────────────────────────────────────────────────────────────
DRY_RUN=false
FILTER_MODEL=""
FILTER_TASK=""
FILTER_PARADIGM=""
DATA_SOURCE="subject"

METHODS=()
FREQS=()
WINDOW_SIZES=()
OVERLAPS=()
SHIFT_FRACTIONS=()
DTW_COMPONENTS_LIST=()
DTW_METHODS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)         DRY_RUN=true;                    shift ;;
        --model)           FILTER_MODEL="$2";               shift 2 ;;
        --task)            FILTER_TASK="$2";                shift 2 ;;
        --paradigm)        FILTER_PARADIGM="$2";            shift 2 ;;
        --data-source)     DATA_SOURCE="$2";                shift 2 ;;
        --method)          METHODS+=("$2");                 shift 2 ;;
        --freq)            FREQS+=("$2");                   shift 2 ;;
        --window-size)     WINDOW_SIZES+=("$2");            shift 2 ;;
        --overlap)         OVERLAPS+=("$2");                shift 2 ;;
        --shift-fraction)  SHIFT_FRACTIONS+=("$2");         shift 2 ;;
        --dtw-components)  DTW_COMPONENTS_LIST+=("$2");     shift 2 ;;
        --dtw-method)      DTW_METHODS+=("$2");             shift 2 ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Usage: bash submit_ablations.sh [--method M] [--freq HZ] [--window-size W]"
            echo "       [--overlap O] [--shift-fraction S] [--dtw-components N] [--dtw-method M]"
            echo "       [--model MODEL] [--task T] [--paradigm P] [--dry-run]"
            exit 1 ;;
    esac
done

# Apply defaults when nothing was specified
[ ${#METHODS[@]}             -eq 0 ] && METHODS=("variable_length")
[ ${#FREQS[@]}               -eq 0 ] && FREQS=(50)
[ ${#WINDOW_SIZES[@]}        -eq 0 ] && WINDOW_SIZES=(300)
[ ${#OVERLAPS[@]}            -eq 0 ] && OVERLAPS=(0.30)
[ ${#SHIFT_FRACTIONS[@]}     -eq 0 ] && SHIFT_FRACTIONS=(0.1)
[ ${#DTW_COMPONENTS_LIST[@]} -eq 0 ] && DTW_COMPONENTS_LIST=(10)
[ ${#DTW_METHODS[@]}         -eq 0 ] && DTW_METHODS=("mds")

# Validate method names
VALID_METHODS="truncate padding variable_length sliding_window phase_shift dtw_embedding"
for M in "${METHODS[@]}"; do
    if [[ ! " ${VALID_METHODS} " =~ " ${M} " ]]; then
        echo "ERROR: Unknown method '${M}'"
        echo "  Valid methods: ${VALID_METHODS}"
        exit 1
    fi
done

if [[ ! " subject event_window " =~ " ${DATA_SOURCE} " ]]; then
    echo "ERROR: Unknown data source '${DATA_SOURCE}'"
    echo "  Valid options: subject event_window"
    exit 1
fi

if [ ! -f "${JOB_SCRIPT}" ]; then
    echo "ERROR: Job script not found at ${JOB_SCRIPT}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

# ── Experiment space ──────────────────────────────────────────────────────────
ALL_TASKS=(1 2 3 4 5 6)
ALL_PARADIGMS=(1 2 3 4)
ALL_MODELS=(hmm cnn rnn transformer)

SUBMITTED=0
FAILED=0

# ── submit_job ────────────────────────────────────────────────────────────────
submit_job() {
    local TASK=$1
    local PARADIGM=$2
    local MODEL=$3
    local METHOD=$4
    local JOB_TAG=$5
    local EXTRA_ARGS=$6
    local SRC=${7:-$DATA_SOURCE}

    [ -n "${FILTER_MODEL}"    ] && [ "${MODEL}"    != "${FILTER_MODEL}"    ] && return
    [ -n "${FILTER_TASK}"     ] && [ "${TASK}"     != "${FILTER_TASK}"     ] && return
    [ -n "${FILTER_PARADIGM}" ] && [ "${PARADIGM}" != "${FILTER_PARADIGM}" ] && return

    # DTW embedding produces fixed-size vectors, not sequences — incompatible with HMM
    if [ "${METHOD}" = "dtw_embedding" ] && [ "${MODEL}" = "hmm" ]; then
        [ "${DRY_RUN}" = true ] && echo "[SKIPPED] dtw_embedding incompatible with HMM — ${MODEL^^}_T${TASK}_P${PARADIGM}"
        return
    fi

    # Padding causes NaN in HMM parameters (startprob_, transmat_) — incompatible
    if [ "${METHOD}" = "padding" ] && [ "${MODEL}" = "hmm" ]; then
        [ "${DRY_RUN}" = true ] && echo "[SKIPPED] padding incompatible with HMM — ${MODEL^^}_T${TASK}_P${PARADIGM}"
        return
    fi

    local TIME_LIMIT="08:00:00"
    local MEMORY="64G"
    local PARTITION="gpu"
    local USE_GPU="true"

    if [ "${MODEL}" = "hmm" ]; then
        PARTITION="short"
        USE_GPU="false"
        MEMORY="32G"
        TIME_LIMIT="48:00:00"
    fi

    local DS_TAG="SBJ"; [ "${SRC}" = "event_window" ] && DS_TAG="EW"
    local JOB_NAME="ABL_${DS_TAG}_${JOB_TAG^^}_${MODEL^^}_T${TASK}_P${PARADIGM}"

    local SBATCH_CMD=(sbatch
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

    [ "${USE_GPU}" = "true" ] && SBATCH_CMD+=(--gres=gpu:1)

    SBATCH_CMD+=("${JOB_SCRIPT}" "${TASK}" "${PARADIGM}" "${MODEL}" \
                 "${METHOD}" "${JOB_TAG}" "${EXTRA_ARGS}" "${SRC}")

    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY RUN] ${JOB_NAME}"
        echo "  method=${METHOD}  data_source=${SRC}  extra='${EXTRA_ARGS}'"
        echo "  time=${TIME_LIMIT}  mem=${MEMORY}  partition=${PARTITION}  gpu=${USE_GPU}"
        echo ""
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo -n "[SUBMITTING] ${JOB_NAME}... "
        if JOB_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1); then
            JOB_ID=$(echo "${JOB_OUTPUT}" | awk '{print $NF}')
            echo "✅ ${JOB_ID}"
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "❌ FAILED — ${JOB_OUTPUT}"
            FAILED=$((FAILED + 1))
        fi
        sleep 0.3
    fi
}

# ── sweep: submit for every task × paradigm × model ──────────────────────────
sweep() {
    local METHOD=$1
    local JOB_TAG=$2
    local EXTRA_ARGS=$3

    for MODEL in "${ALL_MODELS[@]}"; do
        for TASK in "${ALL_TASKS[@]}"; do
            for PARADIGM in "${ALL_PARADIGMS[@]}"; do
                submit_job "${TASK}" "${PARADIGM}" "${MODEL}" \
                           "${METHOD}" "${JOB_TAG}" "${EXTRA_ARGS}"
            done
        done
    done
}

# ── Build extra python args ───────────────────────────────────────────────────
build_extra_args() {
    local METHOD=$1 FREQ=$2 WS=$3 OV=$4 SF=$5 DN=$6 DM=$7
    local ARGS=""
    [ "${FREQ}" -lt 50 ] && ARGS="${ARGS} --freq ${FREQ}"
    case "${METHOD}" in
        sliding_window) ARGS="${ARGS} --window-size ${WS} --overlap ${OV}" ;;
        phase_shift)    ARGS="${ARGS} --shift-fraction ${SF}" ;;
        dtw_embedding)  ARGS="${ARGS} --n-components ${DN} --dtw-method ${DM}" ;;
    esac
    echo "${ARGS# }"
}

# ── Build short job tag ───────────────────────────────────────────────────────
build_tag() {
    local METHOD=$1 FREQ=$2 WS=$3 OV=$4 SF=$5 DN=$6 DM=$7
    local TAG="${METHOD}_f${FREQ}"
    case "${METHOD}" in
        sliding_window) TAG="${TAG}_w${WS}_o${OV/./}" ;;
        phase_shift)    TAG="${TAG}_s${SF//./}" ;;   # strips ALL dots: 0.25 → 025, 0 → 0
        dtw_embedding)  TAG="${TAG}_${DM}${DN}" ;;
    esac
    echo "${TAG}"
}

# ── Header ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " XDash Ablation Submission"
echo " HPC:      $(hostname)"
echo " User:     $(whoami)"
echo " Methods:  ${METHODS[*]}"
echo " Freqs:    ${FREQS[*]} Hz"
[ -n "${FILTER_MODEL}"    ] && echo " Model:    ${FILTER_MODEL}"
[ -n "${FILTER_TASK}"     ] && echo " Task:     ${FILTER_TASK}"
[ -n "${FILTER_PARADIGM}" ] && echo " Paradigm: ${FILTER_PARADIGM}"
echo " Data src: ${DATA_SOURCE}"
[ "${DRY_RUN}" = true ]      && echo " Mode:     DRY RUN"
echo "============================================================"
echo ""

# ── Main sweep loop ───────────────────────────────────────────────────────────
for METHOD in "${METHODS[@]}"; do
    for FREQ in "${FREQS[@]}"; do
        case "${METHOD}" in
            sliding_window)
                for WS in "${WINDOW_SIZES[@]}"; do
                    for OV in "${OVERLAPS[@]}"; do
                        TAG=$(build_tag  "${METHOD}" "${FREQ}" "${WS}" "${OV}" "" "" "")
                        EXTRA=$(build_extra_args "${METHOD}" "${FREQ}" "${WS}" "${OV}" "" "" "")
                        sweep "${METHOD}" "${TAG}" "${EXTRA}"
                    done
                done ;;
            phase_shift)
                for SF in "${SHIFT_FRACTIONS[@]}"; do
                    TAG=$(build_tag  "${METHOD}" "${FREQ}" "" "" "${SF}" "" "")
                    EXTRA=$(build_extra_args "${METHOD}" "${FREQ}" "" "" "${SF}" "" "")
                    sweep "${METHOD}" "${TAG}" "${EXTRA}"
                done ;;
            dtw_embedding)
                for DN in "${DTW_COMPONENTS_LIST[@]}"; do
                    for DM in "${DTW_METHODS[@]}"; do
                        TAG=$(build_tag  "${METHOD}" "${FREQ}" "" "" "" "${DN}" "${DM}")
                        EXTRA=$(build_extra_args "${METHOD}" "${FREQ}" "" "" "" "${DN}" "${DM}")
                        sweep "${METHOD}" "${TAG}" "${EXTRA}"
                    done
                done ;;
            *)
                TAG=$(build_tag  "${METHOD}" "${FREQ}" "" "" "" "" "")
                EXTRA=$(build_extra_args "${METHOD}" "${FREQ}" "" "" "" "" "")
                sweep "${METHOD}" "${TAG}" "${EXTRA}" ;;
        esac
    done
done

# ── Footer ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
if [ "${DRY_RUN}" = true ]; then
    echo " DRY RUN COMPLETE — would submit: ${SUBMITTED} jobs"
    echo " Remove --dry-run to submit for real."
else
    echo " SUBMISSION COMPLETE"
    echo " Submitted: ${SUBMITTED} jobs"
    [ "${FAILED}" -gt 0 ] && echo " Failed:    ${FAILED} jobs"
fi
echo ""
echo " MONITORING:"
echo "   squeue -u singh.vishwa | grep ABL"
echo "   squeue -u singh.vishwa --format='%.8i %.12P %.24j %.8T %.10M %R' | grep ABL"
echo ""
echo " LOGS:"
echo "   ls ${LOG_DIR}/"
echo "   find ${LOG_DIR} -name '*.err' -size +0"
echo "============================================================"