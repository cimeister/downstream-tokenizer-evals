#!/bin/bash
# Launch MC evaluation for all 1B models.
# Usage: bash scripts/launch_mc_eval.sh

set -euo pipefail

REPO_DIR="${HOME}/tokenizer-lm"
CKPT_BASE="/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/checkpoints"

cd "${REPO_DIR}"
mkdir -p logs

MODELS=(
    full-128k-apertus
    full-128k-claude-balanced-bpe
    full-128k-claude-balanced-nfc-bpe
    full-128k-claude-balanced-unigram
    full-128k-claude-english-bpe
    full-128k-gpt4o-balanced-bpe
    full-128k-gpt4o-balanced-nfc-bpe
    full-128k-gpt4o-balanced-unigram
    full-128k-gpt4o-code-bpe
    full-128k-gpt4o-english-bpe
    full-128k-llama3
    full-128k-pabpe
    full-128k-punct-balanced-bpe
    full-128k-punct-english-bpe
    full-128k-rightalign-balanced-bpe
    full-128k-rightalign-balanced-nfc-bpe
    full-128k-rightalign-balanced-unigram
    full-128k-superbpe
)

submitted=0
skipped=0

for model in "${MODELS[@]}"; do
    output="${REPO_DIR}/results/mc_eval/${model}.json"
    if [[ -f "${output}" ]]; then
        echo "SKIP ${model} (results already exist)"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ ! -d "${CKPT_BASE}/${model}" ]]; then
        echo "WARN: checkpoint dir not found for ${model}, skipping"
        skipped=$((skipped + 1))
        continue
    fi

    jobid=$(sbatch --parsable scripts/slurm_eval_mc.sh "${model}")
    echo "SUBMITTED ${model} -> job ${jobid}"
    submitted=$((submitted + 1))
done

echo ""
echo "=== Done: ${submitted} submitted, ${skipped} skipped ==="
