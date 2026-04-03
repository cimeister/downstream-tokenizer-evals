"""
Data preparation: build training mixtures from existing parquet datasets on Clariden.

This is tokenizer-INDEPENDENT. Raw text is stored in parquet format with a `text`
column. Tokenization happens on-the-fly during training via nanochat's dataloader.
Prepare data ONCE, train with ANY tokenizer.

Commands:
    mix     Create a weighted mixture from existing parquet directories
    download  Stream a HuggingFace dataset and write to parquet (for datasets not on disk)

Example — build a mixture from Clariden datasets:

    python data/prepare.py mix \
        --sources \
            /capstor/store/.../fineweb-edu/data:0.45:text \
            /capstor/store/.../fineweb-2_0_1-.../data/output/fra_Latn:0.05:text \
            /capstor/store/.../fineweb-2_0_1-.../data/output/deu_Latn:0.05:text \
            /capstor/store/.../finemath/finemath-4plus:0.15:text \
            /capstor/store/.../starcoderdata/thresholds/python/threshold_0:0.15:content \
        --output /capstor/scratch/cscs/$USER/data/mixed \
        --total-docs 5000000
"""

import argparse
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq


DOCS_PER_ROW_GROUP = 1000


def _find_parquet_files(path, recursive=True):
    """Find all parquet files under a path, recursively if needed."""
    if not os.path.isdir(path):
        raise ValueError(f"Path does not exist: {path}")
    if recursive:
        result = []
        for root, dirs, files in os.walk(path):
            # Skip _removed directories (FineWeb2 convention)
            dirs[:] = [d for d in dirs if not d.endswith("_removed")]
            for f in sorted(files):
                if f.endswith(".parquet"):
                    result.append(os.path.join(root, f))
        return result
    else:
        return sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(".parquet")
        ])


def _iter_parquet_texts(parquet_files, text_field="text", max_docs=None):
    """Iterate over text documents from a list of parquet files."""
    count = 0
    for filepath in parquet_files:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=[text_field])
            for text in rg.column(text_field).to_pylist():
                if text and len(str(text).strip()) >= 20:
                    yield str(text)
                    count += 1
                    if max_docs and count >= max_docs:
                        return


def mix_datasets(sources: list, output_dir: str, total_docs: int = None,
                 val_docs: int = 10000, seed: int = 42):
    """
    Create a data mixture by interleaving multiple parquet sources.

    Each source is "path:weight" or "path:weight:text_field".
    Parquet files are discovered recursively under each path.
    The last output file is the validation shard (nanochat convention).
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Parse sources: "path:weight:text_field" or "path:weight"
    source_configs = []
    total_weight = 0.0
    for s in sources:
        parts = s.rsplit(":", 2)
        if len(parts) == 3:
            path, weight, text_field = parts[0], float(parts[1]), parts[2]
        elif len(parts) == 2:
            # Could be "path:weight" or "path:text_field" — try float parse
            try:
                path, weight, text_field = parts[0], float(parts[1]), "text"
            except ValueError:
                path, weight, text_field = parts[0], 1.0, parts[1]
        else:
            path, weight, text_field = parts[0], 1.0, "text"
        total_weight += weight
        source_configs.append({"path": path, "weight": weight, "text_field": text_field})

    # Normalize weights
    for sc in source_configs:
        sc["weight"] /= total_weight

    # Discover parquet files for each source
    print("Mixture sources:")
    source_iterators = []
    for sc in source_configs:
        files = _find_parquet_files(sc["path"])
        print(f"  {sc['path']}")
        print(f"    weight={sc['weight']:.1%}, text_field={sc['text_field']}, "
              f"{len(files)} parquet files")
        if not files:
            print(f"    WARNING: no parquet files found, skipping")
            continue
        it = _iter_parquet_texts(files, text_field=sc["text_field"])
        source_iterators.append((it, sc["weight"], sc["path"]))

    if not source_iterators:
        raise ValueError("No valid sources with parquet files")

    # Re-normalize after dropping empty sources
    total_w = sum(w for _, w, _ in source_iterators)
    source_iterators = [(it, w / total_w, p) for it, w, p in source_iterators]

    # Interleave documents, tracking per-source text byte counts
    doc_buffer = []
    file_idx = 0
    total = 0
    total_bytes = 0
    docs_per_file = 50000
    per_source_stats = {}  # path -> {"docs": int, "text_bytes": int}

    while source_iterators:
        if total_docs and total >= total_docs:
            break

        # Weighted sampling
        r = random.random()
        cumulative = 0.0
        chosen_idx = len(source_iterators) - 1
        for i, (it, weight, path) in enumerate(source_iterators):
            cumulative += weight
            if r < cumulative:
                chosen_idx = i
                break

        try:
            text = next(source_iterators[chosen_idx][0])
        except StopIteration:
            exhausted_path = source_iterators[chosen_idx][2]
            print(f"  Source exhausted: {exhausted_path} ({total:,} docs so far)")
            source_iterators.pop(chosen_idx)
            if not source_iterators:
                break
            total_w = sum(w for _, w, _ in source_iterators)
            source_iterators = [(it, w / total_w, p) for it, w, p in source_iterators]
            continue

        doc_buffer.append(text)
        total += 1
        text_byte_len = len(text.encode("utf-8"))
        total_bytes += text_byte_len

        # Track per-source stats
        src_path = source_iterators[chosen_idx][2]
        if src_path not in per_source_stats:
            per_source_stats[src_path] = {"docs": 0, "text_bytes": 0}
        per_source_stats[src_path]["docs"] += 1
        per_source_stats[src_path]["text_bytes"] += text_byte_len

        if len(doc_buffer) >= docs_per_file:
            filepath = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
            table = pa.table({"text": doc_buffer})
            pq.write_table(table, filepath, row_group_size=DOCS_PER_ROW_GROUP)
            doc_buffer = []
            file_idx += 1
            if total % 500000 == 0:
                print(f"  {total:,} docs, {total_bytes/1e9:.2f} GB, {file_idx} files")

    # Flush remaining as train + val
    if doc_buffer:
        if len(doc_buffer) > val_docs:
            train_part = doc_buffer[:-val_docs]
            val_part = doc_buffer[-val_docs:]

            filepath = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
            table = pa.table({"text": train_part})
            pq.write_table(table, filepath, row_group_size=DOCS_PER_ROW_GROUP)
            file_idx += 1

            # Val shard: small row groups for multi-GPU eval
            filepath = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
            table = pa.table({"text": val_part})
            pq.write_table(table, filepath, row_group_size=100)
        else:
            filepath = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
            table = pa.table({"text": doc_buffer})
            pq.write_table(table, filepath, row_group_size=100)

    # Build per-source breakdown for metadata
    source_breakdown = []
    for sc in source_configs:
        stats = per_source_stats.get(sc["path"], {"docs": 0, "text_bytes": 0})
        source_breakdown.append({
            "path": sc["path"],
            "input_weight": sc["weight"],
            "text_field": sc["text_field"],
            "docs": stats["docs"],
            "text_bytes": stats["text_bytes"],
            "actual_byte_fraction": stats["text_bytes"] / total_bytes if total_bytes > 0 else 0,
        })

    metadata = {
        "sources": source_breakdown,
        "total_docs": total,
        "total_text_bytes": total_bytes,
        "num_files": file_idx + 1,
        "val_docs": min(val_docs, len(doc_buffer) if len(doc_buffer) <= val_docs else val_docs),
        "seed": seed,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone: {total:,} docs, {total_bytes/1e9:.2f} GB text, {file_idx+1} files")
    print(f"Output: {output_dir}")

    # Print per-source breakdown
    print(f"\nPer-source breakdown (text bytes, not parquet bytes):")
    print(f"  {'Source':<60} {'Docs':>10} {'Text GB':>10} {'Weight':>8} {'Actual':>8}")
    print(f"  {'-'*60} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for sb in sorted(source_breakdown, key=lambda x: -x["text_bytes"]):
        name = os.path.basename(sb["path"].rstrip("/"))
        print(f"  {name:<60} {sb['docs']:>10,} {sb['text_bytes']/1e9:>10.3f} "
              f"{sb['input_weight']:>7.1%} {sb['actual_byte_fraction']:>7.1%}")


def download_dataset(
    dataset_name: str,
    output_dir: str,
    dataset_config: str = None,
    text_field: str = "text",
    split: str = "train",
    max_docs: int = None,
    val_docs: int = 10000,
):
    """
    Stream a HuggingFace dataset and write to parquet files with a `text` column.
    Use this for datasets that aren't already on disk.
    """
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    ds_kwargs = {"streaming": True, "split": split, "trust_remote_code": True}
    if dataset_config:
        ds_kwargs["name"] = dataset_config

    print(f"Streaming: {dataset_name} (config={dataset_config}, split={split})")
    dataset = load_dataset(dataset_name, **ds_kwargs)

    doc_buffer = []
    file_idx = 0
    total_docs = 0
    total_bytes = 0
    docs_per_file = 50000

    def flush_buffer(filepath):
        nonlocal doc_buffer
        if not doc_buffer:
            return
        table = pa.table({"text": doc_buffer})
        pq.write_table(table, filepath, row_group_size=DOCS_PER_ROW_GROUP)
        doc_buffer = []

    for sample in dataset:
        text = sample.get(text_field, "")
        if not text or len(text.strip()) < 20:
            continue

        doc_buffer.append(text)
        total_docs += 1
        total_bytes += len(text.encode("utf-8"))

        if len(doc_buffer) >= docs_per_file:
            filepath = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
            flush_buffer(filepath)
            file_idx += 1
            if total_docs % 100000 == 0:
                print(f"  {total_docs:,} docs, {total_bytes/1e9:.2f} GB, {file_idx} files")

        if max_docs and total_docs >= max_docs:
            break

    # Flush remaining
    if doc_buffer:
        filepath = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
        flush_buffer(filepath)
        file_idx += 1

    # Split off validation from last shard
    all_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".parquet")])
    if len(all_files) >= 1 and total_docs > val_docs:
        last_train = os.path.join(output_dir, all_files[-1])
        table = pq.read_table(last_train)
        texts = table.column("text").to_pylist()
        actual_val = min(val_docs, len(texts) // 2)
        if actual_val > 0:
            train_texts = texts[:-actual_val]
            val_texts = texts[-actual_val:]
            if train_texts:
                pq.write_table(pa.table({"text": train_texts}), last_train,
                               row_group_size=DOCS_PER_ROW_GROUP)
            else:
                os.remove(last_train)
            val_path = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
            pq.write_table(pa.table({"text": val_texts}), val_path,
                           row_group_size=100)
            print(f"Validation shard: {val_path} ({actual_val} docs)")

    metadata = {
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "text_field": text_field,
        "total_docs": total_docs,
        "total_bytes": total_bytes,
        "num_files": file_idx + 1,
        "val_docs": min(val_docs, total_docs),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone: {total_docs:,} docs, {total_bytes/1e9:.2f} GB, {file_idx+1} files")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for tokenizer-lm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mx = subparsers.add_parser("mix", help="Create a weighted data mixture from existing parquets")
    mx.add_argument("--sources", nargs="+", required=True,
                     help="path:weight[:text_field] — text_field defaults to 'text'")
    mx.add_argument("--output", type=str, required=True)
    mx.add_argument("--total-docs", type=int, default=None)
    mx.add_argument("--val-docs", type=int, default=10000)
    mx.add_argument("--seed", type=int, default=42)

    dl = subparsers.add_parser("download", help="Download HF dataset to parquet")
    dl.add_argument("--dataset", type=str, required=True)
    dl.add_argument("--dataset-config", type=str, default=None)
    dl.add_argument("--output", type=str, required=True)
    dl.add_argument("--text-field", type=str, default="text")
    dl.add_argument("--split", type=str, default="train")
    dl.add_argument("--max-docs", type=int, default=None)
    dl.add_argument("--val-docs", type=int, default=10000)

    args = parser.parse_args()

    if args.command == "mix":
        mix_datasets(args.sources, args.output, args.total_docs, args.val_docs, args.seed)
    elif args.command == "download":
        download_dataset(
            args.dataset, args.output,
            dataset_config=args.dataset_config,
            text_field=args.text_field,
            split=args.split,
            max_docs=args.max_docs,
            val_docs=args.val_docs,
        )


if __name__ == "__main__":
    main()
