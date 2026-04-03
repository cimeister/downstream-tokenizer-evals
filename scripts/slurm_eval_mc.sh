#!/bin/bash
#SBATCH --job-name=eval-mc
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_mc_%j.out
#SBATCH --error=logs/eval_mc_%j.err
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default

# MC-Evaluation: GSM8K-MC, MATH-MC, PythonIO-MC (5-shot, 4 options)
# Usage: sbatch scripts/slurm_eval_mc.sh <run_name>

set -euo pipefail

RUN_NAME="${1:?Usage: sbatch slurm_eval_mc.sh <run_name>}"

REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${HOME}/tokenizer_lm_exps_env"
CHECKPOINT_BASE="/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/checkpoints"
CKPT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"

PYTHON="${VENV_DIR}/bin/python"

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${REPO_DIR}/mc_evaluation:${PYTHONPATH:-}"

META_FILE=$(ls "${CKPT_DIR}"/meta_*.json 2>/dev/null | sort | tail -1)
if [[ -z "${META_FILE}" ]]; then
    echo "ERROR: No meta file found in ${CKPT_DIR}"
    exit 1
fi
TOKENIZER=$(${PYTHON} -c "import json; print(json.load(open('${META_FILE}'))['tokenizer_path'])")

mkdir -p "${REPO_DIR}/results/mc_eval"
OUTPUT="${REPO_DIR}/results/mc_eval/${RUN_NAME}.json"

echo "=== MC Evaluation ==="
echo "Job: ${SLURM_JOB_ID} | Run: ${RUN_NAME}"
echo "Checkpoint: ${CKPT_DIR}"
echo "Tokenizer: ${TOKENIZER}"
echo "Output: ${OUTPUT}"

srun -ul --export=ALL \
    env PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${REPO_DIR}/mc_evaluation:${PYTHONPATH:-}" \
    ${PYTHON} -u ${REPO_DIR}/eval_mc.py \
    --checkpoint-dir "${CKPT_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --output "${OUTPUT}" \
    --datasets gsm8k math pythonio \
    --k 5 --options 4 \
    --device cuda
