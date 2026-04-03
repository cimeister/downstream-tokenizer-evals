#!/usr/bin/env python3
"""
Train custom tokenizers for controlled tokenizer comparison experiments.

Trains BPE and UnigramLM tokenizers at 128K vocab with different
pretokenization strategies and training data compositions.
No GPU required — runs on CPU.

Usage:
    python scripts/train_custom_tokenizers.py --output-dir /path/to/output
    python scripts/train_custom_tokenizers.py --output-dir /path/to/output --only gpt4o-balanced
    python scripts/train_custom_tokenizers.py --output-dir /path/to/output --list

Training data (~10B bytes) is sampled from the same sources used for LM
training, ensuring tokenizer-model data alignment.
"""

import argparse
import json
import os
import sys
import tempfile
import time

import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Pretokenizer regex definitions
# ---------------------------------------------------------------------------

# GPT-4o (o200k_base): CamelCase split, \p{N}{1,3} digit groups, contractions
REGEX_GPT4O = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# GPT-4o with right-aligned 3-digit parsing for place-value alignment
# "123456" -> ["123", "456"] instead of potential ["12", "345", "6"]
REGEX_GPT4O_RIGHTALIGN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}(?=(?:\p{N}{3})*(?:\P{N}|$))"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Claude pretokenizer (reverse-engineered via count_tokens API, ~7000 measurements).
# Differs from GPT-4o in:
#   - [ ]? prefix (space only, not [^\r\n\p{L}\p{N}]? — punct doesn't attach to words)
#   - No trailing [\r\n/]* on punct rule
#   - Whitespace split by type ([ ]+ | [\t]+ | [\n]+ | [\r]+), not \s+
# Shares with GPT-4o:
#   - CamelCase-aware letter grouping (Lu*Ll+ | Lu+Ll*)
#   - \p{N}{1,3} digit groups
#   - Contraction handling ('s|'t|'re|'ve|'m|'ll|'d) — both standalone and suffix
#   - Both U+0027 and U+2019 apostrophes trigger contractions
REGEX_CLAUDE = (
    r"(?i:[''\u2019]s|[''\u2019]t|[''\u2019]re|[''\u2019]ve|[''\u2019]m|[''\u2019]ll|[''\u2019]d)"
    r"|[ ]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"(?i:[''\u2019]s|[''\u2019]t|[''\u2019]re|[''\u2019]ve|[''\u2019]m|[''\u2019]ll|[''\u2019]d)?"
    r"|[ ]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"(?i:[''\u2019]s|[''\u2019]t|[''\u2019]re|[''\u2019]ve|[''\u2019]m|[''\u2019]ll|[''\u2019]d)?"
    r"|\p{N}{1,3}"
    r"|[ ]?[^\s\p{L}\p{N}]+"
    r"|[ ]+"
    r"|[\t]+"
    r"|[\n]+"
    r"|[\r]+"
    r"|\s+"
    r"|\S"
)

# ---------------------------------------------------------------------------
# HuggingFace tokenizer config builders
# ---------------------------------------------------------------------------

def _bytelevel_pretok(use_regex=False):
    return {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": use_regex}

def _bytelevel_decoder():
    return {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}

def _regex_split(regex):
    return {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": False}

def _regex_bytelevel_config(regex, normalizer=None):
    config = {
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [_regex_split(regex), _bytelevel_pretok(use_regex=False)],
        },
        "decoder": _bytelevel_decoder(),
    }
    if normalizer:
        config["normalizer"] = normalizer
    return config

def _punctuation_bytelevel_config(normalizer=None):
    config = {
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Punctuation", "behavior": "Isolated"},
                _bytelevel_pretok(use_regex=True),
            ],
        },
        "decoder": _bytelevel_decoder(),
    }
    if normalizer:
        config["normalizer"] = normalizer
    return config

def _claude_bytelevel_config(normalizer=None):
    """Claude pretokenizer via Rust-native Split(Regex(...)).

    Reverse-engineered from ~7000 count_tokens API measurements.
    vs GPT-4o: [ ]? prefix (not [^LN]?), no trailing [\\r\\n/]* on punct,
    whitespace split by type. Shares: CamelCase grouping, contractions, \\p{N}{1,3}.
    """
    return _regex_bytelevel_config(REGEX_CLAUDE, normalizer=normalizer)

# ---------------------------------------------------------------------------
# Data paths on Clariden
# ---------------------------------------------------------------------------

DATA_ROOTS = {
    "fineweb_edu": "/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/fineweb-edu/data",
    "fineweb2": "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_33-filterrobots/data/output",
    "finemath": "/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceTB/finemath/finemath-4plus",
    "starcoderdata": "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/starcoderdata/thresholds",
}

# Training data compositions
# Each maps to a list of (source_path, text_column, weight, max_files)
DATA_COMPOSITIONS = {
    # D1: English-only
    "english": [
        (DATA_ROOTS["fineweb_edu"], "text", 1.0, 20),
    ],
    # D2: Balanced multilingual — 30 languages + math + code
    # Domain split: 35% English, 30% multilingual, 15% math, 15% code, 5% buffer
    # Multilingual weights are proportional to estimated character counts in
    # the filtered FineWeb2 source (quality_33, robot-filtered).
    # Character counts estimated by sampling avg chars/doc × total rows × files.
    "balanced": [
        (DATA_ROOTS["fineweb_edu"], "text", 0.35, 10),
        # --- 30 languages, character-proportional within 30% multilingual budget ---
        # Weights = (est_chars / total_ml_chars) × 0.30
        (DATA_ROOTS["fineweb2"] + "/rus_Cyrl", "text", 0.10104, 5),
        (DATA_ROOTS["fineweb2"] + "/spa_Latn", "text", 0.02324, 3),
        (DATA_ROOTS["fineweb2"] + "/deu_Latn", "text", 0.02257, 3),
        (DATA_ROOTS["fineweb2"] + "/fra_Latn", "text", 0.01957, 3),
        (DATA_ROOTS["fineweb2"] + "/cmn_Hani", "text", 0.01819, 3),
        (DATA_ROOTS["fineweb2"] + "/jpn_Jpan", "text", 0.01202, 3),
        (DATA_ROOTS["fineweb2"] + "/ita_Latn", "text", 0.01135, 3),
        (DATA_ROOTS["fineweb2"] + "/por_Latn", "text", 0.01131, 3),
        (DATA_ROOTS["fineweb2"] + "/tur_Latn", "text", 0.00894, 3),
        (DATA_ROOTS["fineweb2"] + "/ind_Latn", "text", 0.00867, 3),
        (DATA_ROOTS["fineweb2"] + "/pol_Latn", "text", 0.00836, 3),
        (DATA_ROOTS["fineweb2"] + "/ukr_Cyrl", "text", 0.00644, 2),
        (DATA_ROOTS["fineweb2"] + "/nld_Latn", "text", 0.00617, 2),
        (DATA_ROOTS["fineweb2"] + "/ron_Latn", "text", 0.00550, 2),
        (DATA_ROOTS["fineweb2"] + "/arb_Arab", "text", 0.00514, 2),
        (DATA_ROOTS["fineweb2"] + "/hun_Latn", "text", 0.00483, 2),
        (DATA_ROOTS["fineweb2"] + "/vie_Latn", "text", 0.00411, 2),
        (DATA_ROOTS["fineweb2"] + "/ces_Latn", "text", 0.00407, 2),
        (DATA_ROOTS["fineweb2"] + "/ell_Grek", "text", 0.00335, 2),
        (DATA_ROOTS["fineweb2"] + "/fin_Latn", "text", 0.00223, 2),
        (DATA_ROOTS["fineweb2"] + "/tha_Thai", "text", 0.00219, 2),
        (DATA_ROOTS["fineweb2"] + "/slk_Latn", "text", 0.00188, 2),
        (DATA_ROOTS["fineweb2"] + "/bul_Cyrl", "text", 0.00183, 2),
        (DATA_ROOTS["fineweb2"] + "/hrv_Latn", "text", 0.00161, 2),
        (DATA_ROOTS["fineweb2"] + "/kor_Hang", "text", 0.00156, 2),
        (DATA_ROOTS["fineweb2"] + "/cat_Latn", "text", 0.00098, 2),
        (DATA_ROOTS["fineweb2"] + "/hin_Deva", "text", 0.00094, 2),
        (DATA_ROOTS["fineweb2"] + "/heb_Hebr", "text", 0.00089, 2),
        (DATA_ROOTS["fineweb2"] + "/ben_Beng", "text", 0.00067, 2),
        (DATA_ROOTS["fineweb2"] + "/tam_Taml", "text", 0.00036, 2),
        # --- Math + Code (30%) ---
        (DATA_ROOTS["finemath"], "text", 0.15, 5),
        (DATA_ROOTS["starcoderdata"] + "/python/threshold_0", "content", 0.10, 5),
        (DATA_ROOTS["starcoderdata"] + "/javascript/threshold_0", "content", 0.05, 3),
    ],
    # D3: Code-heavy
    "code": [
        (DATA_ROOTS["fineweb_edu"], "text", 0.50, 10),
        (DATA_ROOTS["starcoderdata"] + "/python/threshold_0", "content", 0.20, 10),
        (DATA_ROOTS["starcoderdata"] + "/javascript/threshold_0", "content", 0.10, 5),
        (DATA_ROOTS["starcoderdata"] + "/java/threshold_0", "content", 0.10, 5),
        (DATA_ROOTS["starcoderdata"] + "/cpp/threshold_0", "content", 0.10, 5),
    ],
}

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# Each entry: (name, algorithm, pretok_config, data_composition, description)
# algorithm: "bpe" or "unigram"

EXPERIMENTS = [
    # --- BPE variants ---
    ("punct-balanced-bpe", "bpe",
     _punctuation_bytelevel_config(), "balanced",
     "Punctuation + ByteLevel pretok, balanced data"),

    ("gpt4o-english-bpe", "bpe",
     _regex_bytelevel_config(REGEX_GPT4O), "english",
     "GPT-4o regex, English-only data"),

    ("gpt4o-balanced-bpe", "bpe",
     _regex_bytelevel_config(REGEX_GPT4O), "balanced",
     "GPT-4o regex, balanced multilingual data"),

    ("gpt4o-balanced-nfc-bpe", "bpe",
     _regex_bytelevel_config(REGEX_GPT4O, normalizer={"type": "NFC"}), "balanced",
     "GPT-4o regex, balanced data, NFC normalization"),

    ("gpt4o-code-bpe", "bpe",
     _regex_bytelevel_config(REGEX_GPT4O), "code",
     "GPT-4o regex, code-heavy data"),

    ("claude-balanced-bpe", "bpe",
     _claude_bytelevel_config(), "balanced",
     "Claude pretok (case-level splits), balanced data"),

    ("claude-balanced-nfc-bpe", "bpe",
     _claude_bytelevel_config(normalizer={"type": "NFC"}), "balanced",
     "Claude pretok, balanced data, NFC normalization"),

    ("rightalign-balanced-bpe", "bpe",
     _regex_bytelevel_config(REGEX_GPT4O_RIGHTALIGN), "balanced",
     "GPT-4o regex with right-aligned 3-digit parsing, balanced data"),

    ("rightalign-balanced-nfc-bpe", "bpe",
     _regex_bytelevel_config(REGEX_GPT4O_RIGHTALIGN, normalizer={"type": "NFC"}), "balanced",
     "Right-aligned digits, balanced data, NFC normalization"),

    ("punct-english-bpe", "bpe",
     _punctuation_bytelevel_config(), "english",
     "Punctuation + ByteLevel pretok, English-only data"),

    ("claude-english-bpe", "bpe",
     _claude_bytelevel_config(), "english",
     "Claude pretok, English-only data"),

    # --- UnigramLM variants (subset of pretok × data combos) ---
    ("gpt4o-balanced-unigram", "unigram",
     _regex_bytelevel_config(REGEX_GPT4O), "balanced",
     "GPT-4o regex, balanced data, UnigramLM algorithm"),

    ("claude-balanced-unigram", "unigram",
     _claude_bytelevel_config(), "balanced",
     "Claude pretok, balanced data, UnigramLM algorithm"),

    ("rightalign-balanced-unigram", "unigram",
     _regex_bytelevel_config(REGEX_GPT4O_RIGHTALIGN), "balanced",
     "Right-aligned digits, balanced data, UnigramLM algorithm"),
]

# ---------------------------------------------------------------------------
# Data iteration (reads parquet directly, no intermediate text files)
# ---------------------------------------------------------------------------

import random

def find_parquet_files(root, max_files=None):
    """Find parquet files recursively under root."""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in sorted(filenames):
            if f.endswith(".parquet"):
                files.append(os.path.join(dirpath, f))
    if max_files:
        files = files[:max_files]
    return files


# UnigramTrainer has a 100K character limit per input string.
# Split longer documents to stay within this limit.
MAX_CHARS_PER_DOC = 90_000  # leave some margin below 100K


def _iter_source(source_path, text_column, max_files):
    """Iterate over text documents from a parquet source.

    Long documents are split at whitespace boundaries to stay under
    the UnigramTrainer's 100K character limit.
    """
    parquet_files = find_parquet_files(source_path, max_files=max_files)
    for pq_path in parquet_files:
        try:
            pf = pq.ParquetFile(pq_path)
            for batch in pf.iter_batches(batch_size=4096, columns=[text_column]):
                for value in batch.column(text_column):
                    text = value.as_py()
                    if not text or len(text.strip()) < 20:
                        continue
                    if len(text) <= MAX_CHARS_PER_DOC:
                        yield text
                    else:
                        # Split at whitespace boundaries
                        for chunk in _split_long_text(text, MAX_CHARS_PER_DOC):
                            yield chunk
        except Exception as e:
            print(f"    Error reading {pq_path}: {e}")


def _split_long_text(text, max_chars):
    """Split text into chunks of at most max_chars, breaking at whitespace."""
    while len(text) > max_chars:
        # Find last whitespace before the limit
        split_pos = text.rfind(" ", 0, max_chars)
        if split_pos == -1:
            split_pos = max_chars  # no whitespace found, hard split
        chunk = text[:split_pos]
        if len(chunk.strip()) >= 20:
            yield chunk
        text = text[split_pos:].lstrip()
    if len(text.strip()) >= 20:
        yield text


def training_data_iterator(data_composition, target_bytes=10_000_000_000, seed=42):
    """
    Yield text documents from a data composition, sampled proportionally
    to weights. Reads directly from parquet — no intermediate files.

    Each document is yielded as a single string. The tokenizers library's
    train_from_iterator() handles the rest.

    The same iterator instance can be reused across tokenizers with the
    same data composition — documents come in the same order (deterministic).
    """
    random.seed(seed)
    composition = DATA_COMPOSITIONS[data_composition]
    total_weight = sum(w for _, _, w, _ in composition)

    # Build per-source iterators
    sources = []
    for source_path, text_column, weight, max_files in composition:
        norm_weight = weight / total_weight
        it = _iter_source(source_path, text_column, max_files)
        name = os.path.basename(source_path.rstrip("/"))
        sources.append({"iter": it, "weight": norm_weight, "name": name, "exhausted": False})

    total_bytes = 0
    doc_count = 0
    active_sources = [s for s in sources if not s["exhausted"]]

    while active_sources and total_bytes < target_bytes:
        # Weighted random source selection
        r = random.random()
        cumulative = 0.0
        chosen = active_sources[-1]
        for s in active_sources:
            cumulative += s["weight"]
            if r < cumulative:
                chosen = s
                break

        try:
            text = next(chosen["iter"])
            total_bytes += len(text.encode("utf-8"))
            doc_count += 1
            if doc_count % 500000 == 0:
                print(f"    {doc_count:,} docs, {total_bytes/1e9:.2f}GB")
            yield text
        except StopIteration:
            chosen["exhausted"] = True
            active_sources = [s for s in sources if not s["exhausted"]]
            if active_sources:
                # Re-normalize weights
                total_w = sum(s["weight"] for s in active_sources)
                for s in active_sources:
                    s["weight"] /= total_w
            continue

    print(f"    Done: {doc_count:,} docs, {total_bytes/1e9:.2f}GB")


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128256  # Match LLaMA-3's vocab size
SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>"]


def _save_tokenizer(tokenizer, tokenizer_config, output_path):
    """Save tokenizer to JSON."""
    tokenizer.save(output_path)
    print(f"  Saved to {output_path} (vocab={tokenizer.get_vocab_size()})")


def train_bpe(data_iterator, tokenizer_config, output_path, vocab_size=VOCAB_SIZE):
    """Train a BPE tokenizer from an iterator of text strings."""
    from tokenizers.trainers import BpeTrainer

    tokenizer = _build_tokenizer_from_config(tokenizer_config, model_type="bpe")

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        max_token_length=64,
    )

    print(f"  Training BPE (vocab_size={vocab_size})...")
    tokenizer.train_from_iterator(data_iterator, trainer=trainer)
    _save_tokenizer(tokenizer, tokenizer_config, output_path)
    return tokenizer


def train_unigram(data_iterator, tokenizer_config, output_path, vocab_size=VOCAB_SIZE):
    """Train a UnigramLM tokenizer from an iterator of text strings."""
    from tokenizers.trainers import UnigramTrainer

    tokenizer = _build_tokenizer_from_config(tokenizer_config, model_type="unigram")

    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        unk_token="<unk>",
    )

    print(f"  Training UnigramLM (vocab_size={vocab_size})...")
    tokenizer.train_from_iterator(data_iterator, trainer=trainer)
    _save_tokenizer(tokenizer, tokenizer_config, output_path)
    return tokenizer


def _build_tokenizer_from_config(config, model_type="bpe"):
    """Build a HuggingFace Tokenizer from a component config dict.

    All pretokenizer configs (including Claude) use the standard JSON
    serialization path via Tokenizer.from_str(). The Split pretokenizer
    with {"Regex": pattern} correctly handles all Unicode properties
    including \\p{Lu} and \\p{Ll}.
    """
    from tokenizers import Tokenizer

    if model_type == "bpe":
        skeleton = {
            "version": "1.0",
            "model": {"type": "BPE", "vocab": {}, "merges": []},
        }
    else:
        skeleton = {
            "version": "1.0",
            "model": {"type": "Unigram", "vocab": []},
        }
    for key in ("pre_tokenizer", "decoder", "normalizer", "post_processor"):
        if key in config and config[key] is not None:
            skeleton[key] = config[key]

    tokenizer = Tokenizer.from_str(json.dumps(skeleton))
    return tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train custom tokenizers for tokenizer-lm experiments"
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save tokenizers (required unless --list)")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Train only these tokenizers (by name)")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit")
    parser.add_argument("--target-bytes", type=float, default=10e9,
                        help="Target training data size in bytes (default: 10GB)")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help=f"Vocabulary size (default: {VOCAB_SIZE})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data sampling (default: 42)")
    args = parser.parse_args()

    if args.list:
        print(f"{'Name':<35} {'Algo':<8} {'Data':<10} Description")
        print("-" * 90)
        for name, algo, _, data, desc in EXPERIMENTS:
            print(f"{name:<35} {algo:<8} {data:<10} {desc}")
        print(f"\nTotal: {len(EXPERIMENTS)} experiments")
        return

    if args.output_dir is None:
        parser.error("--output-dir is required unless --list is used")

    vocab_size = args.vocab_size

    os.makedirs(args.output_dir, exist_ok=True)

    # Filter experiments
    experiments = EXPERIMENTS
    if args.only:
        experiments = [(n, a, c, d, desc) for n, a, c, d, desc in EXPERIMENTS if n in args.only]
        if not experiments:
            print(f"No experiments match: {args.only}")
            print(f"Available: {[n for n, _, _, _, _ in EXPERIMENTS]}")
            return

    # Group experiments by data composition so we can share iterators.
    # Tokenizers using the same composition read the same data in the
    # same order (deterministic via seed), ensuring fair comparison.
    by_composition = {}
    for name, algo, config, data_comp, desc in experiments:
        by_composition.setdefault(data_comp, []).append((name, algo, config, desc))

    print(f"\n{'='*60}")
    print(f"Training {len(experiments)} tokenizers at {vocab_size:,} vocab")
    print(f"Data compositions: {list(by_composition.keys())}")
    print(f"Target bytes per composition: {args.target_bytes/1e9:.1f}GB")
    print(f"{'='*60}\n")

    results = []

    for data_comp, comp_experiments in by_composition.items():
        print(f"\n=== Data composition: {data_comp} ({len(comp_experiments)} tokenizers) ===\n")

        for name, algo, config, desc in comp_experiments:
            output_path = os.path.join(args.output_dir, name, "tokenizer.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if os.path.exists(output_path):
                print(f"SKIP {name} (already exists)")
                results.append((name, "skipped"))
                continue

            print(f"\n--- {name} ---")
            print(f"  Algorithm: {algo}")
            print(f"  Data: {data_comp}")
            print(f"  Description: {desc}")

            # Create a fresh iterator for each tokenizer (same seed = same data)
            data_iter = training_data_iterator(
                data_comp,
                target_bytes=int(args.target_bytes),
                seed=args.seed,
            )
            t0 = time.time()

            try:
                if algo == "bpe":
                    tokenizer = train_bpe(data_iter, config, output_path, vocab_size=vocab_size)
                elif algo == "unigram":
                    tokenizer = train_unigram(data_iter, config, output_path, vocab_size=vocab_size)
                else:
                    raise ValueError(f"Unknown algorithm: {algo}")

                # Sanity check
                test = tokenizer.encode("Hello world! 123456 def foo(): pass")
                print(f"  Sanity: {test.tokens[:10]}...")
                print(f"  Time: {time.time() - t0:.0f}s")
                results.append((name, "ok"))
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, f"failed: {e}"))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, status in results:
        print(f"  {name:<35} {status}")
    print(f"\nTokenizers saved to: {args.output_dir}")
    print(f"To use with tokenizer-lm: --tokenizer {args.output_dir}/<name>")


if __name__ == "__main__":
    main()
