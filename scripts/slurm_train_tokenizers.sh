#!/bin/bash
#SBATCH --job-name=train-tok
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_tok_%j.out
#SBATCH --error=logs/train_tok_%j.err
#SBATCH --gpus-per-task=0

# ============================================================
# Train custom tokenizers (CPU-only, needs high memory for BPE)
# Usage: sbatch scripts/slurm_train_tokenizers.sh [--only name1 name2 ...]
# ============================================================

set -euo pipefail

REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${HOME}/tokenizer_lm_exps_env"
OUTPUT_DIR="/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/tokenizers"

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"

mkdir -p "${REPO_DIR}/logs"

echo "=== Tokenizer Training ==="
echo "Job: ${SLURM_JOB_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Output: ${OUTPUT_DIR}"
echo "Python: $(which python)"
echo "Args: $@"

python -u ${REPO_DIR}/scripts/train_custom_tokenizers.py \
    --output-dir "${OUTPUT_DIR}" \
    --target-bytes 10e9 \
    --vocab-size 128256 \
    "$@"
