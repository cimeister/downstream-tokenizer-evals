#!/bin/bash
#SBATCH --job-name=tok-lm-eval
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --uenv=pytorch/v2.8.0:/user-environment
#SBATCH --view=default

# ============================================================
# Evaluation: run benchmarks on a trained checkpoint
# Usage: sbatch scripts/slurm_eval.sh /path/to/checkpoint_dir /path/to/tokenizer [tasks]
# ============================================================

set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: sbatch slurm_eval.sh /path/to/checkpoint_dir /path/to/tokenizer [tasks]}"
TOKENIZER="${2:?Missing tokenizer path}"
TASK_GROUPS="${3:-all}"

REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${REPO_DIR}/.venv"

RUN_NAME="$(basename ${CHECKPOINT_DIR})"
OUTPUT="${REPO_DIR}/results/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).json"

if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
fi

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1

mkdir -p "${REPO_DIR}/results" "${REPO_DIR}/logs"

echo "=== Evaluation ==="
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Tasks: ${TASK_GROUPS}"
echo "Output: ${OUTPUT}"

python ${REPO_DIR}/evaluate.py \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --tokenizer ${TOKENIZER} \
    --output ${OUTPUT} \
    --tasks ${TASK_GROUPS} \
    --batch-size 16

echo "Results: ${OUTPUT}"
