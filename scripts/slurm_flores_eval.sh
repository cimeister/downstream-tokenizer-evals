#!/bin/bash
#SBATCH --job-name=flores-eval
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/flores_%j.out
#SBATCH --error=logs/flores_%j.err
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default

# ============================================================
# FLORES-200 BPB evaluation for a single model checkpoint.
# Usage: sbatch scripts/slurm_flores_eval.sh <run_name>
# ============================================================

set -euo pipefail

RUN_NAME="${1:?Usage: sbatch slurm_flores_eval.sh <run_name>}"

REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${HOME}/tokenizer_lm_exps_env"
CHECKPOINT_BASE="/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/checkpoints"

export PATH="${VENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/results/flores"

PYTHON="${VENV_DIR}/bin/python"

echo "=== FLORES Evaluation ==="
echo "Job: ${SLURM_JOB_ID}"
echo "Run: ${RUN_NAME}"
echo "Python: $(${PYTHON} --version 2>&1)"

${PYTHON} -u ${REPO_DIR}/scripts/batch_flores_eval.py \
    --checkpoint-base "${CHECKPOINT_BASE}" \
    --output-dir "${REPO_DIR}/results/flores" \
    --device cuda \
    --only "${RUN_NAME}"
