# tokenizer-lm

A controlled experimental framework for measuring how **tokenization affects language model quality**. Train identical architectures with different tokenizers on the same data, compare on downstream benchmarks. All models trained from scratch.

## Motivation

Tokenization is a foundational design choice in language model training, yet its impact on downstream performance is poorly understood in isolation. Most tokenizer comparisons are confounded by differences in model architecture, training data, or compute budget. This repo isolates the tokenizer as the **sole experimental variable** by holding everything else constant.

Key experimental properties:
- **Data is tokenizer-independent**: raw text stored as parquet files, tokenized on-the-fly during training
- **Same architecture** for all tokenizers within a vocab-size bucket
- **Same training data** (identical documents in identical order) across all runs
- **Primary metric is BPB** (bits-per-byte): a tokenizer-independent measure of model quality
- **Per-vocab-size configs**: tokenizers are compared within the same vocab size (50K, 64K, 128K, 256K), never across

## Design choices

### Architecture

Built on [nanochat](https://github.com/karpathy/nanochat) (git submodule) with all its architectural innovations. These interact with tokenization in known ways, documented here for transparency:

| Feature | Tokenization interaction |
|---------|------------------------|
| **RoPE** (rotary positional embeddings) | None — position-based, not token-based |
| **QK-Norm** (RMSNorm on Q,K) | None |
| **ReLU²** activation | None |
| **Muon optimizer** (Polar Express) | Used for transformer matrices only; embeddings use AdamW. Optimizer behavior is identical across tokenizers within the same vocab size |
| **Value embeddings** (ResFormer-style) | Per-token-ID lookup tables. Dimension capped at `ve_dim=128` with learned projection to prevent embedding-dominated parameter budgets. Same total VE params for same vocab size |
| **Sliding window attention** (`SSSL` pattern) | Window size is in tokens. A more efficient tokenizer covers more bytes per window — this is a real effect, not an artifact |
| **Smear gate** (adjacent token mixing) | Token boundaries differ by tokenizer, so "adjacent" means different things. Effect is small (gate is learned and initialized near zero) |
| **Logit soft-capping** (`15·tanh(logits/15)`) | Applied uniformly regardless of vocab size |
| **Parameterless RMSNorm** | None |
| **Norm after embedding** | None |

### Training budget

Token budget is computed as `target_param_data_ratio × (transformer_matrices + lm_head)` with ratio 10.5 (Chinchilla-like). This means:
- Transformer matrix params are identical across tokenizers within the same vocab size
- `lm_head` params scale with vocab size, so the token budget scales proportionally
- Since comparisons are within the same vocab size, all runs get the same budget

Batch sizes are fixed explicitly in each config (not auto-computed) to eliminate a source of fragility. All schedules (warmup, warmdown, momentum) are identical across runs.

### Hyperparameter selection

Three measures reduce hyperparameter sensitivity as a confounding factor:

**1. µP (Maximal Update Parameterization).** Embedding and lm_head learning rates scale as `∝ 1/width` relative to a 768-dim reference model (`width_lr_exponent: -1.0`). This makes hyperparameters transfer correctly across model widths by construction. Muon's matrix LR is width-invariant (spectral normalization handles the scaling internally).

**2. Fixed batch sizes.** Each config specifies an explicit `total_batch_size` rather than auto-computing via Power Lines scaling. This removes a fragile formula (referencing a d12 model) as a source of variation and makes the setup fully transparent.

**3. LR sweep validation.** The default `matrix_lr=0.02` was validated via a 5-point sweep (500 steps each, GPT-2 tokenizer, pilot-50k config):

| matrix_lr | embedding_lr | unembedding_lr | val BPB (500 steps) |
|-----------|-------------|----------------|-------------------|
| 0.005 | 0.075 | 0.002 | 1.1246 |
| 0.010 | 0.150 | 0.004 | 1.0982 |
| **0.020** | **0.300** | **0.008** | **1.0390** |
| 0.040 | 0.600 | 0.016 | 1.0531 |
| 0.080 | 1.200 | 0.032 | 1.0744 |

The sweep confirms `matrix_lr=0.02` is optimal — a clean U-shape centered on the default value. Embedding and lm_head LRs scale proportionally (15× and 0.4× matrix_lr respectively), following nanochat's tuned ratios.

To run your own sweep:
```bash
PYTHONPATH=".:nanochat" python scripts/lr_sweep.py \
    --config configs/pilot_50k.yaml \
    --tokenizer openai-community/gpt2 \
    --data-dir /path/to/data \
    --lr-values 0.005 0.01 0.02 0.04 0.08
```

### Evaluation metrics

- **BPB (bits-per-byte)**: Primary metric for both training validation and FLORES-200 multilingual evaluation. Computed as `sum(nats) / (ln(2) × sum(bytes))`, weighting each token's loss by its UTF-8 byte length. Special tokens are excluded. This is inherently tokenizer-independent.
- **Downstream tasks** (GSM8K, MGSM, HumanEval, MBPP, BLiMP): These have natural tokenizer sensitivity — measuring that sensitivity is part of the experimental goal.

### What is NOT controlled

These effects are inherent to tokenizer comparison and are part of what is being measured:
- **Packing efficiency**: Different tokenizers produce different token counts for the same text. A more efficient tokenizer sees more unique text per training step (~35% of tokens are cropped in the BOS-aligned best-fit packing).
- **Effective context length**: A 2048-token sliding window covers different amounts of text depending on tokenizer fertility.
- **Token boundary effects**: Smear gate and attention patterns operate on token boundaries, which differ by tokenizer.

## Data

Raw text in parquet files, sourced from datasets on CSCS Clariden. Data preparation is a one-time operation shared across all tokenizer experiments. The same data mixture is used for both LM training and tokenizer training.

### Sources

| Domain | Dataset | Path on Clariden | Mix weight |
|--------|---------|-----------------|-----------|
| English web | FineWeb-Edu | `.../HuggingFaceFW/fineweb-edu/data` | 35% |
| Multilingual | Filtered FineWeb2 (top 33% quality) | `.../swiss-ai/fineweb-2_0_1-quality_33-filterrobots/data/output/{lang}` | 30% |
| Math | FineMath-4plus (score ≥ 4) | `.../HuggingFaceTB/finemath/finemath-4plus` | 15% |
| Code | StarCoderData (highest quality tier) | `.../swiss-ai/starcoderdata/thresholds/{lang}/threshold_0` | 15% |

### Multilingual language selection and weighting

30 languages across 8 scripts and 11+ language families. Within the 30% multilingual budget, languages are weighted **proportionally to their estimated character counts** in the source data (filtered FineWeb2). Character counts are approximated by sampling average characters per document from the first parquet file of each language, then multiplying by total rows × number of files.

| Language | Script | Est. chars | Mix weight |
|----------|--------|-----------|-----------|
| Russian | Cyrillic | 2,261B | 10.10% |
| Spanish | Latin | 520B | 2.32% |
| German | Latin | 505B | 2.26% |
| French | Latin | 438B | 1.96% |
| Chinese | CJK | 407B | 1.82% |
| Japanese | CJK | 269B | 1.20% |
| Italian | Latin | 254B | 1.14% |
| Portuguese | Latin | 253B | 1.13% |
| Turkish | Latin | 200B | 0.89% |
| Indonesian | Latin | 194B | 0.87% |
| Polish | Latin | 187B | 0.84% |
| Ukrainian | Cyrillic | 144B | 0.64% |
| Dutch | Latin | 138B | 0.62% |
| Romanian | Latin | 123B | 0.55% |
| Arabic | Arabic | 115B | 0.51% |
| Hungarian | Latin | 108B | 0.48% |
| Vietnamese | Latin | 92B | 0.41% |
| Czech | Latin | 91B | 0.41% |
| Greek | Greek | 75B | 0.34% |
| Finnish | Latin | 50B | 0.22% |
| Thai | Thai | 49B | 0.22% |
| Slovak | Latin | 42B | 0.19% |
| Bulgarian | Cyrillic | 41B | 0.18% |
| Croatian | Latin | 36B | 0.16% |
| Korean | Hangul | 35B | 0.16% |
| Catalan | Latin | 22B | 0.10% |
| Hindi | Devanagari | 21B | 0.09% |
| Hebrew | Hebrew | 20B | 0.09% |
| Bengali | Bengali | 15B | 0.07% |
| Tamil | Tamil | 8B | 0.04% |

This character-proportional weighting means low-resource languages naturally receive less training data — creating the data scarcity gradient needed for studying how tokenizer quality interacts with resource level.

### Creating the mixture

```bash
bash scripts/create_mixture.sh          # 5M docs default
bash scripts/create_mixture.sh 10000000 # 10M docs
```

Or use `data/prepare.py` directly for custom mixtures:
```bash
python data/prepare.py mix \
    --sources /path/to/source1:0.5:text /path/to/source2:0.3:content \
    --output /capstor/scratch/cscs/$USER/data/my-mix \
    --total-docs 5000000
```

Sources are `path:weight[:text_field]`. Parquet files are discovered recursively. The `text_field` defaults to `text` but can be overridden (e.g., `content` for StarCoderData).

### Custom tokenizer training

14 custom tokenizers are trained at 128K vocab with different pretokenization strategies (Punctuation, GPT-4o, Claude, right-aligned digits), training data compositions (English-only, balanced multilingual, code-heavy), normalization (with/without NFC), and algorithms (BPE, UnigramLM). Tokenizer training reads directly from the source parquet files — no intermediate text file extraction.

```bash
python scripts/train_custom_tokenizers.py --list                    # list all 14
python scripts/train_custom_tokenizers.py --output-dir /path/to/out # train all
python scripts/train_custom_tokenizers.py --output-dir /path/to/out --only gpt4o-balanced-bpe
```

Tokenizer and LM training use the **same data sources and the same character-proportional weighting**, ensuring tokenizer-model data alignment.

## Configs

Per-vocab-size configs with properly balanced architectures. Compare tokenizers **only within the same vocab-size bucket**.

### Pilot scale (single node, 4× GH200 GPUs)

| Config | Vocab | n_embd | n_layer | Batch size | Transformer params | Total params |
|--------|-------|--------|---------|-----------|-------------------|-------------|
| `pilot_50k.yaml` | ~50K | 1024 | 16 | 524,288 | ~300M | ~400M |
| `pilot_64k.yaml` | ~64K | 1024 | 16 | 524,288 | ~300M | ~500M |
| `pilot_128k.yaml` | ~128K | 1024 | 16 | 1,048,576 | ~300M | ~700M |
| `pilot_256k.yaml` | ~256K | 1280 | 16 | 1,048,576 | ~470M | ~1.4B |

### Full scale (4 nodes, 16× GH200 GPUs)

| Config | Vocab | n_embd | n_layer | Batch size | Transformer params | Total params |
|--------|-------|--------|---------|-----------|-------------------|-------------|
| `full_64k.yaml` | ~64K | 2048 | 24 | 1,048,576 | ~1.2B | ~1.6B |
| `full_128k.yaml` | ~128K | 2048 | 24 | 1,048,576 | ~1.2B | ~1.9B |
| `full_256k.yaml` | ~256K | 2048 | 24 | 2,097,152 | ~1.2B | ~2.5B |

All configs use `ve_dim=128` (capped value embedding dimension), `width_lr_exponent=-1.0` (µP scaling), and explicit `total_batch_size`.

## Setup on CSCS Clariden

```bash
ssh clariden.alps.cscs.ch

git clone --recurse-submodules <repo-url> tokenizer-lm
cd tokenizer-lm

# Python environment
uenv start pytorch/v2.8.0:v1 --view=default
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install datasets pyarrow wandb lm-eval evaluate sentencepiece tiktoken rustbpe

export PYTHONPATH="$HOME/tokenizer-lm:$HOME/tokenizer-lm/nanochat"
export WANDB_API_KEY="your-key"
export WANDB_PROJECT="tokenizer-lm"
```

## Running experiments

### 1. Prepare data (once)

```bash
bash scripts/create_mixture.sh
```

### 2. Train

```bash
# Interactive (on a node with GPUs)
torchrun --nproc_per_node=4 train.py \
    --config configs/pilot_128k.yaml \
    --tokenizer swiss-ai/Apertus-70B-2509 \
    --data-dir /capstor/scratch/cscs/$USER/data/tokenizer-lm-mix-pilot \
    --run-name pilot-128k-apertus

# Via SLURM
sbatch scripts/slurm_pilot.sh configs/pilot_128k.yaml \
    swiss-ai/Apertus-70B-2509 \
    /capstor/scratch/cscs/$USER/data/tokenizer-lm-mix-pilot \
    pilot-128k-apertus
```

### 3. Evaluate

```bash
sbatch scripts/slurm_eval.sh \
    /capstor/scratch/cscs/$USER/tokenizer-lm/checkpoints/pilot-128k-apertus \
    swiss-ai/Apertus-70B-2509 \
    all
```

### 4. Compare

Results are saved as structured JSON. The primary comparison metric is **val BPB** (logged to W&B as `val/bpb`). FLORES-200 evaluation reports per-language and per-language-family BPB.

### Batch experiments

```bash
bash scripts/run_pilot_experiments.sh        # all buckets
bash scripts/run_pilot_experiments.sh 50k    # 50K bucket only
bash scripts/run_pilot_experiments.sh 128k   # 128K bucket only
```

## Evaluation benchmarks

| Group | Tasks | What it measures |
|-------|-------|-----------------|
| Math | GSM8K (8-shot CoT), MGSM (11 languages) | Math reasoning, cross-lingual math × tokenization |
| Code | HumanEval, MBPP | Code generation quality |
| Multilingual | FLORES-200 BPB (per-language, per-family) | **Tokenization fairness across languages** |
| Linguistic | BLiMP (67 subtasks) | Grammatical knowledge |

## Estimated MFU

Target: **60–70% on GH200 GPUs** at full scale.

- GH200 uses the H100 Hopper die: 989 TFLOPS bf16 peak
- `torch.compile` + Flash Attention 3 + ReLU² + bf16 throughout
- Pilot scale (~300M transformer): ~25-30% MFU (model too small to saturate GPU)
- Full scale (~1.2B transformer): ~60-70% MFU expected

## Project structure

```
tokenizer-lm/
├── nanochat/                 # Git submodule (model, optimizer, dataloader)
├── configs/
│   ├── pilot_{50k,64k,128k,256k}.yaml
│   ├── full_{64k,128k,256k}.yaml
│   └── sweep_lr.yaml        # LR sweep meta-config
├── scripts/
│   ├── slurm_pilot.sh        # SLURM: single-node training
│   ├── slurm_full.sh         # SLURM: multi-node training
│   ├── slurm_eval.sh         # SLURM: evaluation
│   ├── create_mixture.sh     # Build data mixture from Clariden datasets
│   ├── run_pilot_experiments.sh  # Launch tokenizer comparison experiments
│   └── lr_sweep.py           # Hyperparameter validation sweep
├── train.py                  # Training entry point
├── evaluate.py               # LM Eval Harness + FLORES-200 BPB
├── data/
│   └── prepare.py            # Build data mixtures from parquet sources
├── tokenizer_lm/
│   ├── __init__.py
│   └── tokenizer.py          # Tokenizer wrapper (robust BOS detection, token_bytes)
├── environment.yml
├── README.md
└── EXPERIMENTS.md            # Lab notebook template
```
