#!/usr/bin/env python

"""
validate_tiny_minicpm.py
by Hanna Aamir, 
December 6, 2025,
aamir.hanna@gmail.com

Rebuilds the same tiny MiniCPM-o model from the base config from generate_tiny_minicpm.py
and runs a minimal forward pass on the underlying Qwen2 LLM
to verify the following:
- that the tiny config is valid
- that the text backbone can run a forward pass
"""

import torch
from pathlib import Path
from transformers import AutoConfig, AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()

BASE_MODEL_ID = "openbmb/MiniCPM-o-2_6"  # original repo on HuggingFace
TOKENIZER_DIR = Path("tiny-minicpm-tokenizer")  # this is the tiny tokenizer


def build_tiny_config(vocab_size: int):
    """
    Builds the same tiny config as in generate_tiny_minicpm.py,
    but with the vocab_size wired in. This is so that we can use the same tiny tokenizer.
    """
    cfg = AutoConfig.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    # shrink LLM for text backbone: must match generate_tiny_minicpm.py 
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2

    # Qwen2-based configs require that the layer_types length == num_hidden_layers
    if hasattr(cfg, "layer_types") and isinstance(cfg.layer_types, list):
        cfg.layer_types = cfg.layer_types[: cfg.num_hidden_layers]

    cfg.max_position_embeddings = 64

    # shrink vision backbone
    if hasattr(cfg, "vision_config"):
        vcfg = cfg.vision_config
        vcfg.hidden_size = 8
        vcfg.intermediate_size = 16
        vcfg.num_hidden_layers = 1
        vcfg.num_attention_heads = 1
        if hasattr(vcfg, "image_size"):
            vcfg.image_size = 64

    # shrink audio backbone
    if hasattr(cfg, "audio_config"):
        acfg = cfg.audio_config
        acfg.encoder_layers = 1
        acfg.decoder_layers = 1
        acfg.decoder_ffn_dim = 128
        if hasattr(acfg, "d_model"):
            acfg.d_model = 16

    # shrink TTS backbone
    if hasattr(cfg, "tts_config"):
        tcfg = cfg.tts_config
        tcfg.llm_dim = cfg.hidden_size
        tcfg.hidden_size = 8
        tcfg.intermediate_size = 16
        tcfg.num_layers = 1
        tcfg.num_hidden_layers = 1
        tcfg.num_heads = 1
        tcfg.num_attention_heads = 1
        tcfg.num_mel_bins = 16
        tcfg.num_text_tokens = 32
        tcfg.num_audio_tokens = 16

    # setting the vocab size to match the tiny tokenizer
    cfg.vocab_size = vocab_size

    return cfg


def main():
    # Step 1: Load the tiny tokenizer
    tiny_tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        trust_remote_code=False,
    )
    vocab_size = tiny_tokenizer.vocab_size

    # Step 2: Build the tiny config with this vocab size
    cfg = build_tiny_config(vocab_size)

    # Step 3: Build the tiny MiniCPMO model with random weights
    model = AutoModel.from_config(cfg, trust_remote_code=True)
    model.eval()

    # Step 4: Tokenize a simple string using the tiny tokenizer
    text = "hello tiny MiniCPM"
    inputs = tiny_tokenizer(text, return_tensors="pt")

    # Step 5: Run a forward pass on the underlying LLM
    with torch.no_grad():
        outputs = model.llm(**inputs)

    # Step 6: Print the results
    print("Input IDs shape:", inputs["input_ids"].shape)
    if hasattr(outputs, "logits"):
        print("Logits shape:", outputs.logits.shape)
    elif hasattr(outputs, "last_hidden_state"):
        print("Last hidden state shape:", outputs.last_hidden_state.shape)
    else:
        print("Output type:", type(outputs))


if __name__ == "__main__":
    main()
