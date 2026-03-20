"""
Evaluation script for tokenizer-lm.

Runs LM Evaluation Harness benchmarks and custom FLORES-200 perplexity evaluation.
Produces a structured JSON results file for cross-tokenizer comparison.

Usage:
    python evaluate.py \
        --checkpoint-dir /path/to/checkpoints/run_name \
        --step 5000 \
        --tokenizer /path/to/tokenizer \
        --output results/run_name.json \
        --tasks all
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import yaml
import lm_eval
import lm_eval.api.model

from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint, find_last_step
from nanochat.common import print0

from tokenizer_lm.tokenizer import load_tokenizer, generate_token_bytes


MGSM_LANGUAGES = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

LANGUAGE_FAMILIES = {
    "Germanic": ["deu_Latn", "nld_Latn", "swe_Latn", "dan_Latn", "nob_Latn", "eng_Latn", "afr_Latn"],
    "Romance": ["fra_Latn", "spa_Latn", "por_Latn", "ita_Latn", "ron_Latn", "cat_Latn"],
    "Slavic": ["rus_Cyrl", "pol_Latn", "ces_Latn", "ukr_Cyrl", "bul_Cyrl", "hrv_Latn", "slk_Latn"],
    "Indo-Aryan": ["hin_Deva", "ben_Beng", "urd_Arab", "mar_Deva", "guj_Gujr", "npi_Deva"],
    "Dravidian": ["tam_Taml", "tel_Telu", "kan_Knda", "mal_Mlym"],
    "Sino-Tibetan": ["zho_Hans", "zho_Hant", "mya_Mymr"],
    "Japonic": ["jpn_Jpan"],
    "Koreanic": ["kor_Hang"],
    "Turkic": ["tur_Latn", "aze_Latn", "uzb_Latn", "kaz_Cyrl"],
    "Afro-Asiatic": ["ara_Arab", "heb_Hebr", "amh_Ethi", "hau_Latn", "som_Latn"],
    "Niger-Congo": ["swa_Latn", "yor_Latn", "ibo_Latn", "zul_Latn", "xho_Latn", "lin_Latn"],
    "Austronesian": ["ind_Latn", "msa_Latn", "tgl_Latn", "jav_Latn", "ceb_Latn"],
    "Tai-Kadai": ["tha_Thai", "lao_Laoo"],
    "Austroasiatic": ["vie_Latn", "khm_Khmr"],
    "Uralic": ["fin_Latn", "hun_Latn", "est_Latn"],
    "Hellenic": ["ell_Grek"],
}


def load_model_for_eval(checkpoint_dir, step, tokenizer_path, device="cuda"):
    """Load a nanochat model checkpoint for evaluation."""
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)

    # Handle torch.compile key prefix
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Reconstruct model
    model_config_kwargs = meta_data["model_config"]
    # Patch missing keys for older checkpoints
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
    config = GPTConfig(**model_config_kwargs)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    # Patch missing state dict keys
    n_layer = config.n_layer
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)

    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    tokenizer, vocab_size, bos_id = load_tokenizer(tokenizer_path)
    assert vocab_size == config.vocab_size, (
        f"Tokenizer vocab ({vocab_size}) != model vocab ({config.vocab_size})"
    )

    return model, tokenizer, meta_data


class NanochatLM(lm_eval.api.model.LM):
    """
    lm-eval-harness wrapper for nanochat's GPT model.

    HFLM expects a HuggingFace model API which nanochat doesn't match.
    This implements the LM interface directly: loglikelihood and generate_until.
    """

    def __init__(self, model, tokenizer, batch_size=8, device="cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self.vocab_size = tokenizer.get_vocab_size()

    @property
    def eot_id(self):
        return self.tokenizer.get_bos_token_id()

    @property
    def max_length(self):
        return self.model.config.sequence_len

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        ids = self.tokenizer.encode(string)
        if left_truncate_len:
            ids = ids[-left_truncate_len:]
        return ids

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            logits = self.model(inps)  # (B, T, vocab_size)
        return logits

    def _model_generate(self, context, max_length, stop, **kwargs):
        # Greedy generation with text-level stop-string matching
        ids = context
        ctx_len = context.shape[1]
        for _ in range(max_length - ctx_len):
            logits = self._model_call(ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if stop:
                generated_text = self.tok_decode(ids[0, ctx_len:].tolist())
                if any(s in generated_text for s in stop):
                    break
        return ids

    def loglikelihood(self, requests, disable_tqdm=False):
        from lm_eval.api.instance import Instance
        from tqdm import tqdm

        results = []
        for chunk_start in range(0, len(requests), self._batch_size):
            chunk = requests[chunk_start:chunk_start + self._batch_size]

            for req in chunk:
                ctx, cont = req.args
                ctx_ids = self.tok_encode(ctx)
                cont_ids = self.tok_encode(cont)
                all_ids = (ctx_ids + cont_ids)[-self.max_length:]
                input_tensor = torch.tensor([all_ids], dtype=torch.long, device=self._device)

                with torch.no_grad():
                    logits = self.model(input_tensor)

                # Get log probs for continuation tokens
                cont_len = len(cont_ids)
                logits_cont = logits[0, -cont_len-1:-1, :]
                target_cont = torch.tensor(all_ids[-cont_len:], device=self._device)
                log_probs = F.log_softmax(logits_cont, dim=-1)
                token_log_probs = log_probs.gather(1, target_cont.unsqueeze(1)).squeeze(1)
                total_ll = token_log_probs.sum().item()
                is_greedy = (logits_cont.argmax(dim=-1) == target_cont).all().item()
                results.append((total_ll, is_greedy))

        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            (text,) = req.args
            ids = self.tok_encode(text)[-self.max_length:]
            input_tensor = torch.tensor([ids], dtype=torch.long, device=self._device)

            with torch.no_grad():
                logits = self.model(input_tensor)

            log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
            targets = torch.tensor(ids[1:], device=self._device)
            token_lls = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            results.append((token_lls.sum().item(),))
        return results

    def generate_until(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            ctx, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ctx_ids = self.tok_encode(ctx)[-self.max_length:]
            input_tensor = torch.tensor([ctx_ids], dtype=torch.long, device=self._device)

            generated = self._model_generate(input_tensor, min(len(ctx_ids) + max_gen, self.max_length), until)
            gen_ids = generated[0, len(ctx_ids):].tolist()
            gen_text = self.tok_decode(gen_ids)

            # Truncate at stop strings
            for stop in until:
                idx = gen_text.find(stop)
                if idx >= 0:
                    gen_text = gen_text[:idx]
                    break
            results.append(gen_text)
        return results


def run_lm_eval_tasks(model, tokenizer, task_names, batch_size=8, device="cuda"):
    """Run LM Evaluation Harness tasks using custom LM wrapper."""
    lm = NanochatLM(model, tokenizer, batch_size=batch_size, device=device)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_names,
        confirm_run_unsafe_code=True,
    )
    return results


def compute_flores_perplexity(model, tokenizer, token_bytes, device="cuda", max_samples=200):
    """
    Compute per-language BPB (bits-per-byte) on FLORES-200 devtest split.

    Uses BPB instead of per-token perplexity so results are comparable across
    tokenizers with different vocab sizes and fertility rates.
    """
    from datasets import load_dataset

    results = {"per_language": {}, "per_family": {}}

    try:
        flores = load_dataset("openlanguagedata/flores_plus", split="devtest", trust_remote_code=True)
    except Exception:
        try:
            flores = load_dataset("facebook/flores", "all", split="devtest", trust_remote_code=True)
        except Exception as e:
            print(f"WARNING: Could not load FLORES-200: {e}")
            return results

    # Find language columns
    lang_cols = [c for c in flores.column_names if c.startswith("sentence_") or
                 (len(c) >= 7 and "_" in c and c not in ("id", "URL", "domain", "topic", "has_image", "has_hyperlink"))]

    print(f"FLORES-200: found {len(lang_cols)} language columns")
    family_scores = {}

    for col in lang_cols:
        lang_code = col.replace("sentence_", "") if col.startswith("sentence_") else col
        try:
            texts = [row[col] for row in flores.select(range(min(max_samples, len(flores))))]
            texts = [t for t in texts if t and len(t.strip()) > 0]
            if len(texts) < 10:
                continue

            bpb = _compute_bpb(model, tokenizer, token_bytes, texts, device)
            results["per_language"][lang_code] = {"bpb": bpb, "num_samples": len(texts)}

            for family, langs in LANGUAGE_FAMILIES.items():
                if lang_code in langs:
                    family_scores.setdefault(family, []).append(bpb)
                    break
        except Exception as e:
            print(f"  Skipping {lang_code}: {e}")

    for family, bpbs in family_scores.items():
        results["per_family"][family] = {
            "mean_bpb": sum(bpbs) / len(bpbs),
            "num_languages": len(bpbs),
        }

    return results


@torch.no_grad()
def _compute_bpb(model, tokenizer, token_bytes, texts, device, max_length=512):
    """
    Compute bits-per-byte (BPB) on a list of text strings.

    BPB = total_nats / (ln(2) * total_bytes), where each target token's
    NLL is weighted by its byte length. Special tokens (byte length 0) are
    excluded. This is the same metric as nanochat's evaluate_bpb and is
    directly comparable across tokenizers.
    """
    total_nats = 0.0
    total_bytes = 0
    bos_id = tokenizer.get_bos_token_id()

    for text in texts:
        ids = tokenizer.encode(text, prepend=bos_id)
        if len(ids) < 2:
            continue
        ids = ids[:max_length]
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([ids[1:]], dtype=torch.long, device=device)

        logits = model(input_ids)
        # Per-token NLL (no reduction)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")

        # Weight by byte length of each target token; exclude special tokens
        target_bytes = token_bytes[targets.view(-1)]  # byte length per target token
        valid = target_bytes > 0
        total_nats += (nll * valid.float()).sum().item()
        total_bytes += target_bytes.sum().item()

    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained tokenizer-lm model")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--tasks", nargs="+", default=["all"],
                        choices=["all", "math", "code", "flores", "blimp"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    step = args.step or find_last_step(args.checkpoint_dir)
    print(f"Loading checkpoint: {args.checkpoint_dir} step {step}")

    model, tokenizer, meta_data = load_model_for_eval(
        args.checkpoint_dir, step, args.tokenizer, args.device
    )

    task_groups = set(args.tasks)
    if "all" in task_groups:
        task_groups = {"math", "code", "flores", "blimp"}

    all_results = {
        "metadata": {
            "checkpoint_dir": args.checkpoint_dir,
            "step": step,
            "tokenizer_path": args.tokenizer,
            "run_name": meta_data.get("training_config", {}).get("name", "unknown"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "val_bpb": meta_data.get("val_bpb"),
            "total_bytes_consumed": meta_data.get("loop_state", {}).get("total_bytes_consumed"),
        },
        "scores": {},
    }

    # LM Eval Harness tasks
    harness_tasks = []
    if "math" in task_groups:
        harness_tasks.append("gsm8k_cot")
        harness_tasks.extend([f"mgsm_direct_{lang}" for lang in MGSM_LANGUAGES])
    if "code" in task_groups:
        harness_tasks.extend(["humaneval", "mbpp"])
    if "blimp" in task_groups:
        harness_tasks.append("blimp")

    if harness_tasks:
        print(f"\nRunning LM Eval Harness: {harness_tasks}")
        harness_results = run_lm_eval_tasks(
            model, tokenizer, harness_tasks, args.batch_size, args.device
        )
        for task_name, metrics in harness_results.get("results", {}).items():
            all_results["scores"][task_name] = {
                k: v for k, v in metrics.items()
                if k not in ("alias", "name") and not k.startswith("_")
            }
        all_results["metadata"]["n_shot"] = harness_results.get("n-shot", {})

    if "flores" in task_groups:
        print("\nRunning FLORES-200 BPB evaluation...")
        token_bytes = generate_token_bytes(tokenizer, device=args.device)
        flores_results = compute_flores_perplexity(model, tokenizer, token_bytes, args.device)
        all_results["scores"]["flores200"] = flores_results

    # Save
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for task_name, metrics in all_results["scores"].items():
        if task_name == "flores200":
            n_langs = len(metrics.get("per_language", {}))
            n_fam = len(metrics.get("per_family", {}))
            print(f"  FLORES-200: {n_langs} languages, {n_fam} families")
            for family, fdata in sorted(metrics.get("per_family", {}).items()):
                print(f"    {family}: BPB={fdata['mean_bpb']:.3f}")
        else:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and "stderr" not in k:
                    print(f"  {task_name}: {k}={v:.4f}")
                    break


if __name__ == "__main__":
    main()
