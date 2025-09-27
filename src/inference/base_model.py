import os
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_config import ModelConfig


def download_base_model() -> None:
    # Check if the model has already been downloaded
    if os.path.exists(ModelConfig.MODEL_DIR):
        print(f"{ModelConfig.MODEL_NAME} has already been downloaded. Skipping...")
        return

    # Download tokenizer
    print(f"Downloading tokenizer for {ModelConfig.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        ModelConfig.MODEL_NAME, trust_remote_code=True
    )
    tokenizer.save_pretrained(ModelConfig.MODEL_DIR)

    # Download model
    print(f"Downloading {ModelConfig.MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        ModelConfig.MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.save_pretrained(ModelConfig.MODEL_DIR)
    print(f"✓ Base model saved to: {ModelConfig.MODEL_DIR}")

    # Save model-info.json
    print(f"Saving model meta data to {ModelConfig.MODEL_META_DATA}")
    with open(ModelConfig.MODEL_META_DATA, "w") as f:
        model_info_data = {
            "model-name": ModelConfig.MODEL_NAME,
            "model-dir": ModelConfig.MODEL_DIR,
            "model-tokenizer": ModelConfig.MODEL_TOKENIZER,
            "model-dtype": "float16",
            "model-adapter": None,
            "model-device": "auto",
        }
        json.dump(obj=model_info_data, fp=f, indent=2)


if __name__ == "__main__":
    download_base_model()
