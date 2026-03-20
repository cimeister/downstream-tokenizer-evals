#!/bin/bash
# Create the training data mixture from datasets available on Clariden.
#
# Usage: bash scripts/create_mixture.sh [total_docs]
#   total_docs: number of documents in the mixture (default: 5000000)
#
# Data sources (all on capstor):
#   English:      FineWeb-Edu (quality-filtered Common Crawl)
#   Multilingual: Filtered FineWeb2 (top 33% quality, robot-filtered)
#   Math:         FineMath-4plus (math score >= 4)
#   Code:         StarCoderData (quality-tiered, threshold_0 = highest)
#
# Mixture: 45% English, 20% multilingual, 15% math, 15% code, 5% additional langs
#
# Output: /capstor/scratch/cscs/$USER/data/tokenizer-lm-mix

set -euo pipefail

TOTAL_DOCS="${1:-5000000}"
REPO_DIR="${HOME}/tokenizer-lm"
OUTPUT="/capstor/scratch/cscs/${USER}/data/tokenizer-lm-mix"

# --- Data roots ---
FINEWEB_EDU="/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/fineweb-edu/data"
FINEWEB2="/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_33-filterrobots/data/output"
FINEMATH="/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceTB/finemath/finemath-4plus"
STARCODERDATA="/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/starcoderdata/thresholds"

echo "=== Creating training mixture ==="
echo "Total docs: ${TOTAL_DOCS}"
echo "Output: ${OUTPUT}"
echo ""

# Verify sources exist
for src in "$FINEWEB_EDU" "$FINEWEB2" "$FINEMATH" "$STARCODERDATA"; do
    if [ ! -d "$src" ]; then
        echo "ERROR: Source not found: $src"
        exit 1
    fi
done

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/nanochat:${PYTHONPATH:-}"

python ${REPO_DIR}/data/prepare.py mix \
    --sources \
        "${FINEWEB_EDU}:0.45:text" \
        "${FINEWEB2}/fra_Latn:0.025:text" \
        "${FINEWEB2}/deu_Latn:0.025:text" \
        "${FINEWEB2}/spa_Latn:0.025:text" \
        "${FINEWEB2}/rus_Cyrl:0.025:text" \
        "${FINEWEB2}/jpn_Jpan:0.025:text" \
        "${FINEWEB2}/cmn_Hani:0.025:text" \
        "${FINEWEB2}/arb_Arab:0.02:text" \
        "${FINEWEB2}/hin_Deva:0.01:text" \
        "${FINEWEB2}/kor_Hang:0.01:text" \
        "${FINEWEB2}/tha_Thai:0.005:text" \
        "${FINEWEB2}/vie_Latn:0.005:text" \
        "${FINEMATH}:0.15:text" \
        "${STARCODERDATA}/python/threshold_0:0.075:content" \
        "${STARCODERDATA}/javascript/threshold_0:0.025:content" \
        "${STARCODERDATA}/java/threshold_0:0.025:content" \
        "${STARCODERDATA}/cpp/threshold_0:0.025:content" \
    --output "${OUTPUT}" \
    --total-docs "${TOTAL_DOCS}" \
    --val-docs 10000

echo ""
echo "=== Mixture created ==="
echo "To train: pass --data-dir ${OUTPUT} to train.py"
