# Experiment Log

Copy a block for each new experiment. Primary comparison metric: **val BPB** (bits-per-byte) — tokenizer-independent.

---

## Experiment: [name]

**Date:** YYYY-MM-DD
**Status:** planned | running | completed | failed

### Setup

- **Tokenizer:** [path, vocab size, description]
- **Config:** `configs/pilot_125m.yaml` | `configs/full_1b.yaml`
- **Data:** [mixture, total bytes]
- **Nodes/GPUs:** [e.g., 1 node / 4 GPUs]
- **SLURM Job ID:**
- **W&B Run:** [link]

### Training results

| Metric | Value |
|--------|-------|
| Final val BPB | |
| Min val BPB | |
| Final train loss | |
| MFU (avg %) | |
| Tokens/sec | |
| Bytes consumed | |
| Wall time | |

### Math (GSM8K 8-shot CoT, MGSM direct)

| Task | Score |
|------|-------|
| GSM8K | |
| MGSM-en | |
| MGSM-de | |
| MGSM-fr | |
| MGSM-ja | |
| MGSM-zh | |

### Code

| Task | Score |
|------|-------|
| HumanEval (pass@1) | |
| MBPP (pass@1) | |

### FLORES-200 perplexity (lower = better)

| Language Family | Mean PPL | Langs |
|----------------|----------|-------|
| Germanic | | |
| Romance | | |
| Slavic | | |
| Indo-Aryan | | |
| Dravidian | | |
| Sino-Tibetan | | |
| Niger-Congo | | |

### BLiMP

| Aggregate accuracy | |

### Notes

[Observations, tokenizer-specific effects, comparison with other variants]

---
