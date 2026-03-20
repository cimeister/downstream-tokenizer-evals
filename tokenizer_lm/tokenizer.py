"""
Tokenizer wrapper for arbitrary HuggingFace tokenizers.

Provides a uniform interface compatible with nanochat's dataloader, which expects:
- .encode(text_or_list, prepend=bos_id, num_threads=N) -> list[list[int]]
- .get_bos_token_id() -> int
- .get_vocab_size() -> int

The key design constraint: swapping tokenizers requires changing exactly one value
(the tokenizer path). This module handles the mapping from an arbitrary HF tokenizer
to nanochat's expected interface.
"""

import os
import torch
from nanochat.tokenizer import HuggingFaceTokenizer


# BOS token conventions across different tokenizer families
_BOS_TOKEN_CANDIDATES = [
    "<|bos|>",              # nanochat's convention
    "<|endoftext|>",        # GPT-2, GPT-J, Falcon, Pythia
    "<s>",                  # LLaMA 1/2, Mistral, SentencePiece models
    "<|begin_of_text|>",    # LLaMA 3
    "<bos>",                # Gemma
    "[BOS]",                # some custom tokenizers
    "<|end_of_text|>",      # Granite (uses end_of_text as document delimiter)
    "<｜begin▁of▁sentence｜>",  # DeepSeek (fullwidth Unicode)
]


class TokenizerWrapper(HuggingFaceTokenizer):
    """
    Extends nanochat's HuggingFaceTokenizer with robust BOS token detection
    for arbitrary HuggingFace tokenizers.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._bos_id = self._find_bos_token_id()

    @classmethod
    def from_path(cls, tokenizer_path: str):
        """
        Load a tokenizer from a path. Tries:
        1. Local directory containing tokenizer.json
        2. HuggingFace hub model name (e.g., "gpt2", "meta-llama/Llama-3-8B")
        """
        tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
        if os.path.isdir(tokenizer_path) and os.path.exists(tokenizer_json):
            return cls.from_directory(tokenizer_path)
        else:
            return cls.from_pretrained(tokenizer_path)

    @classmethod
    def from_pretrained(cls, hf_path):
        from tokenizers import Tokenizer as HFTokenizer
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        from tokenizers import Tokenizer as HFTokenizer
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    def _find_bos_token_id(self) -> int:
        """Try multiple BOS token conventions and return the first match."""
        for candidate in _BOS_TOKEN_CANDIDATES:
            token_id = self.encode_special(candidate)
            if token_id is not None:
                return token_id
        raise ValueError(
            f"No standard BOS token found in tokenizer. "
            f"Tried: {_BOS_TOKEN_CANDIDATES}. "
            f"Your tokenizer must have one of these special tokens. "
            f"Add one via tokenizer.add_special_tokens() before saving."
        )

    def get_bos_token_id(self) -> int:
        return self._bos_id

    def get_bos_token_name(self) -> str:
        """Return the name of the BOS token for logging."""
        for candidate in _BOS_TOKEN_CANDIDATES:
            if self.encode_special(candidate) == self._bos_id:
                return candidate
        return f"token_id={self._bos_id}"


def generate_token_bytes(tokenizer, device="cpu") -> torch.Tensor:
    """
    Generate a token_bytes tensor for BPB (bits-per-byte) evaluation.

    Maps each token ID to its UTF-8 byte length. Special tokens map to 0
    (excluded from BPB computation). This enables tokenizer-independent
    evaluation via nanochat's evaluate_bpb().

    For byte-level BPE tokenizers (GPT-2 style), single tokens may not
    decode cleanly in isolation. We handle this by decoding pairs of
    tokens and computing the marginal byte contribution.
    """
    vocab_size = tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int32, device=device)
    special_tokens = tokenizer.get_special_tokens()

    # First pass: try direct decode for each token
    failed_ids = []
    for token_id in range(vocab_size):
        token_str = tokenizer.id_to_token(token_id)
        if token_str is None or token_str in special_tokens:
            continue
        try:
            text = tokenizer.decode([token_id])
            byte_len = len(text.encode("utf-8"))
            # Detect replacement characters (U+FFFD) — sign of broken decode
            if "\ufffd" in text:
                failed_ids.append(token_id)
                continue
            if byte_len > 0:
                token_bytes[token_id] = byte_len
            else:
                failed_ids.append(token_id)
        except Exception:
            failed_ids.append(token_id)

    # Second pass: for tokens that failed solo decode (common with byte-level BPE),
    # estimate byte length by decoding [reference_token, target_token] and subtracting
    # the reference token's contribution.
    if failed_ids:
        # Find a reference token that decodes cleanly (e.g., "a" or first working token)
        ref_id = None
        ref_text = None
        for tid in range(vocab_size):
            if token_bytes[tid] > 0:
                ref_id = tid
                ref_text = tokenizer.decode([tid])
                break

        if ref_id is not None:
            ref_byte_len = len(ref_text.encode("utf-8"))
            recovered = 0
            for token_id in failed_ids:
                try:
                    pair_text = tokenizer.decode([ref_id, token_id])
                    pair_bytes = len(pair_text.encode("utf-8"))
                    marginal = pair_bytes - ref_byte_len
                    if marginal > 0 and "\ufffd" not in pair_text:
                        token_bytes[token_id] = marginal
                        recovered += 1
                except Exception:
                    pass
            if recovered < len(failed_ids):
                n_still_failed = len(failed_ids) - recovered
                print(f"WARNING: {n_still_failed}/{vocab_size} tokens could not be "
                      f"mapped to byte lengths (will be excluded from BPB)")

    return token_bytes


def load_tokenizer(tokenizer_path: str):
    """
    Load a tokenizer from path. Single entry point for tokenizer loading.
    Returns (tokenizer, vocab_size, bos_token_id).
    """
    tokenizer = TokenizerWrapper.from_path(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    bos_id = tokenizer.get_bos_token_id()
    return tokenizer, vocab_size, bos_id
