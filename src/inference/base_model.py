import os
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)

from .model_config import ModelConfig


# Base model and path to model file
model_name = ModelConfig.MODEL_NAME
model_dir = ModelConfig.MODEL_DIR
model_tokenizer = ModelConfig.MODEL_TOKENIZER
model_info = os.path.join(ModelConfig.MODEL_DIR, "model.json")


def download_base_model() -> None:
    # Check if the model has already been downloaded
    if os.path.exists(model_dir):
        print(f"{model_name} has already been downloaded. Skipping...")
        return

    # Download tokenizer
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(model_dir)

    # Download model
    print(f"Downloading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.save_pretrained(model_dir)
    print(f"Base model saved to: {model_dir}")

    # Save model-info.json
    print(f"Saving model meta data to {model_info}")
    with open(model_info, "w") as f:
        model_info_data = {
            "model-name": model_name,
            "model-dir": model_dir,
            "model-tokenizer": model_tokenizer,
        }
        json.dump(obj=model_info_data, fp=f, indent=2)


if __name__ == "__main__":
    download_base_model()
