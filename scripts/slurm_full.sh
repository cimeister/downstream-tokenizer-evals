#!/bin/bash
#SBATCH --job-name=tok-lm-full
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#SBATCH --uenv=pytorch/v2.8.0:/user-environment
#SBATCH --view=default
#SBATCH -C nvidia_vboost_enabled

# ============================================================
# Full-scale: 4 nodes (16 GPUs)
# Usage: sbatch scripts/slurm_full.sh <config> <tokenizer> <data_dir> [run_name]
# Example: sbatch scripts/slurm_full.sh configs/full_128k.yaml /path/to/tok /path/to/data
# ============================================================

set -euo pipefail

CONFIG="${1:?Usage: sbatch slurm_full.sh <config.yaml> <tokenizer> <data_dir> [run_name]}"
TOKENIZER="${2:?Missing tokenizer path}"
DATA_DIR="${3:?Missing data directory}"
RUN_NAME="${4:-full-$(basename ${TOKENIZER})}"

REPO_DIR="${HOME}/tokenizer-lm"
VENV_DIR="${REPO_DIR}/.venv"

if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
fi

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-tokenizer-lm}"
export WANDB_MODE="${WANDB_MODE:-online}"

export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
export CUDA_CACHE_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_HOME="/dev/shm/"

mkdir -p "${REPO_DIR}/logs"

echo "=== Full-Scale Training ==="
echo "Job: ${SLURM_JOB_ID} | Nodes: ${SLURM_JOB_NUM_NODES} | GPUs: $((SLURM_JOB_NUM_NODES * 4))"
echo "Config: ${CONFIG}"
echo "Tokenizer: ${TOKENIZER}"
echo "Data: ${DATA_DIR}"
echo "Run: ${RUN_NAME}"

srun -ul python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --node_rank=${SLURM_PROCID} \
    --master_addr=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1) \
    --master_port=29500 \
    ${REPO_DIR}/train.py \
    --config ${CONFIG} \
    --tokenizer ${TOKENIZER} \
    --data-dir ${DATA_DIR} \
    --run-name ${RUN_NAME}
