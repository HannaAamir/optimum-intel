#!/usr/bin/env python

"""
shrink_tokenizer.py
by Hanna Aamir, 
December 6, 2025,
aamir.hanna@gmail.com

Builds a tiny tokenizer for MiniCPM by doing the following:
- Keeping special tokens
- Keeping the lowest-ID (most frequent) regular tokens
- Limiting the vocab size to TARGET_VOCAB_SIZE tokens
"""

from pathlib import Path 

from transformers import AutoTokenizer, PreTrainedTokenizerFast # this is the original tokenizer    
from tokenizers import Tokenizer
from tokenizers.models import WordLevel # this is the new tokenizer
from tokenizers.pre_tokenizers import Whitespace # this is the pre-tokenizer

BASE_MODEL_ID = "openbmb/MiniCPM-o-2_6" # this is the original model
OUTPUT_DIR = Path("tiny-minicpm-tokenizer")
TARGET_VOCAB_SIZE = 1000  # chose not to go smaller than this as tradeoff between performance and file size was disadvantageous


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load the original MiniCPM tokenizer
    orig = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    orig_vocab = orig.get_vocab()  # this is a dictionary: token -> id
    special_tokens = orig.all_special_tokens
    special_ids = set(orig.all_special_ids)

    # Step 2: Decide core special tokens
    unk = orig.unk_token or "[UNK]"
    pad = orig.pad_token or "[PAD]"
    bos = getattr(orig, "bos_token", None)
    eos = getattr(orig, "eos_token", None)

    # Step 3: Build a new small vocab mapping token -> new_id
    new_vocab: dict[str, int] = {}

    def add_token(tok: str):
        if tok is None:
            return
        if tok not in new_vocab:
            new_vocab[tok] = len(new_vocab)

    # Step 3a: Ensure unk/pad/bos/eos exist
    add_token(unk)
    add_token(pad)
    add_token(bos)
    add_token(eos)

    # Step 3b: Ensure all special tokens exist
    for tok in special_tokens:
        add_token(tok)

    # Step 3c: Add most "frequent" normal tokens (low original IDs first)
    # NOTE: for many tokenizers, lower ids roughly correspond to more frequent tokens
    sorted_items = sorted(orig_vocab.items(), key=lambda kv: kv[1])

    for tok, tok_id in sorted_items:
        if tok in new_vocab:
            continue  # already added as special token
        if len(new_vocab) >= TARGET_VOCAB_SIZE:
            break
        add_token(tok)

    # Step 4: Build a WordLevel tokenizer with this small vocab
    word_model = WordLevel(vocab=new_vocab, unk_token=unk)
    tk = Tokenizer(word_model)
    tk.pre_tokenizer = Whitespace() # this is the pre-tokenizer

    # Step 5: Wrap in a Hugging Face PreTrainedTokenizerFast
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token=unk,
        pad_token=pad,
        bos_token=bos,
        eos_token=eos,
    )

    # Keep any extra special tokens that aren't unk/pad/bos/eos (a.k.a. not core special tokens)
    additional_specials = [
        t
        for t in special_tokens
        if t not in {unk, pad, bos, eos} and t is not None
    ]
    if additional_specials:
        fast_tok.add_special_tokens(
            {"additional_special_tokens": additional_specials}
        )

    # Step 6: Save the tiny tokenizer
    fast_tok.save_pretrained(OUTPUT_DIR)

    print(f"Shrunken tokenizer saved to: {OUTPUT_DIR.resolve()}")
    print(f"New vocab size: {fast_tok.vocab_size} tokens")


if __name__ == "__main__":
    main()
