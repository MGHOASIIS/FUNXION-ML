#!/bin/bash
# =============================================================================
# setup_env.sh
# Creates the xdash conda environment from environment.yml.
# Run this ONCE on the login node before submitting any jobs.
#
# Usage:
#   bash setup_env.sh
# =============================================================================

set -e

PROJECT_ROOT="/home/singh.vishwa/xdash2"
ENV_YML="${PROJECT_ROOT}/environment.yml"
ENV_NAME="xdash"

echo "============================================================"
echo " XDash HPC Environment Setup"
echo " Project root : ${PROJECT_ROOT}"
echo " Environment  : ${ENV_NAME}"
echo "============================================================"

# ── Verify conda is available ─────────────────────────────────────────────────
# conda is initialised via .bashrc on this cluster (shell function, not a
# file path). This script must be run on the login node where .bashrc is
# sourced automatically. Do NOT use 'source ~/miniconda3/...' here.
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found."
    echo "       Run this script on the login node (not inside a job)."
    exit 1
fi
echo "[INFO] conda found: $(conda --version)"

# ── Check if env already exists ───────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "[INFO] Environment '${ENV_NAME}' already exists."
    echo "       To rebuild from scratch, run:"
    echo "         conda env remove -n ${ENV_NAME}"
    echo "       then re-run this script."
    echo ""
else
    echo ""
    echo "[INFO] Creating environment from ${ENV_YML} ..."
    echo ""

    # Strip the local 'prefix:' line so conda uses the cluster's default path
    TMPYML="/tmp/environment_hpc.yml"
    grep -v "^prefix:" "${ENV_YML}" > "${TMPYML}"

    conda env create -f "${TMPYML}"
    rm -f "${TMPYML}"

    echo ""
    echo "[INFO] Environment '${ENV_NAME}' created successfully."
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "[INFO] Verifying key packages..."
conda run -n "${ENV_NAME}" python -c "
import torch, sklearn, numpy, hmmlearn, tslearn
print(f'  torch      : {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'  numpy      : {numpy.__version__}')
print(f'  sklearn    : {sklearn.__version__}')
print(f'  hmmlearn   : {hmmlearn.__version__}')
print(f'  tslearn    : {tslearn.__version__}')
"

# ── Verify project structure ───────────────────────────────────────────────────
echo ""
echo "[INFO] Verifying project structure..."
for path in \
    "${PROJECT_ROOT}/main.py" \
    "${PROJECT_ROOT}/data/xdash_px_details.xlsx" \
    "${PROJECT_ROOT}/data/pickled_datasets"; do
    if [ -e "${path}" ]; then
        echo "  [OK]      ${path}"
    else
        echo "  [MISSING] ${path}"
    fi
done

echo ""
echo "============================================================"
echo " Setup complete. You can now run:"
echo "   bash submit_all.sh --dry-run"
echo "   bash submit_all.sh"
echo "============================================================"
