#!/usr/bin/env python

"""
generate_tiny_minicpm.py 
by Hanna Aamir, 
December 6, 2025,
aamir.hanna@gmail.com


Creates a very tiny random MiniCPM-o-2.6-style model for testing.

- Uses the MiniCPMO architecture from openbmb/MiniCPM-o-2_6
- Shrinks text, vision, audio, and TTS submodules
- To further shrink, it also uses a tiny tokenizer (1000 tokens) from tiny-minicpm-tokenizer/
- Saves a HuggingFace-compatible folder on disk
"""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, logging

logging.set_verbosity_error()  # this keep logs quiet

BASE_MODEL_ID = "openbmb/MiniCPM-o-2_6"
OUTPUT_DIR = Path("tiny-random-MiniCPM-o-2_6-tiny")
TOKENIZER_DIR = Path("tiny-minicpm-tokenizer")  # this is the tiny tokenizer
DEVICE = "cpu"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load the original MiniCPM-o-2_6 config (no weights)
    cfg = AutoConfig.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    # LLM part/shrink the text backbone
    # hidden_size must be >= 128 so that the vision resampler has at least 1 head 
    cfg.hidden_size = 128              # 128 // 128 = 1 resampler head
    cfg.intermediate_size = 256        # still small
    cfg.num_hidden_layers = 1          # just 1 transformer block
    cfg.num_attention_heads = 4        # 128 / 4 = 32 dims per head
    cfg.num_key_value_heads = 2        # value is valid as long as <= num_attention_heads

    # Qwen2-based configs require that the layer_types length == num_hidden_layers
    if hasattr(cfg, "layer_types") and isinstance(cfg.layer_types, list):
        cfg.layer_types = cfg.layer_types[: cfg.num_hidden_layers]

    # Optional: could also shrink context length
    cfg.max_position_embeddings = 64

    # now, we shrink the vision backbone
    if hasattr(cfg, "vision_config"):
        vcfg = cfg.vision_config
        vcfg.hidden_size = 8
        vcfg.intermediate_size = 16
        vcfg.num_hidden_layers = 1
        vcfg.num_attention_heads = 1
        if hasattr(vcfg, "image_size"):
            vcfg.image_size = 64

    # then, we shrink the audio backbone
    if hasattr(cfg, "audio_config"):
        acfg = cfg.audio_config
        acfg.encoder_layers = 1
        acfg.decoder_layers = 1
        acfg.decoder_ffn_dim = 128
        if hasattr(acfg, "d_model"):
            acfg.d_model = 16

    # lastly, we shrink the TTS backbone
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

    # use the tiny tokenizer and set the vocab_size
    tiny_tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        trust_remote_code=False,  # a standard fast tokenizer
    )
    cfg.vocab_size = tiny_tokenizer.vocab_size

    # Step 2: Build model from this tiny config which is randomly initialized
    model = AutoModel.from_config(cfg, trust_remote_code=True)

    # Step 3: Cast to bf16 for asmaller file size
    model = model.to(dtype=torch.bfloat16, device=DEVICE)

    # Step 4: Save the tiny model + config
    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
    )
    cfg.save_pretrained(OUTPUT_DIR)

    # Step 5: Save the tiny tokenizer into the same folder
    tiny_tokenizer.save_pretrained(OUTPUT_DIR)

    # Step 6: Save the processor, if available
    if hasattr(model, "processor") and model.processor is not None:
        model.processor.save_pretrained(OUTPUT_DIR)

    print(f"Tiny MiniCPM-o model saved to: {OUTPUT_DIR.resolve()}")
    print(f"Vocab size used: {cfg.vocab_size}")


if __name__ == "__main__":
    main()
