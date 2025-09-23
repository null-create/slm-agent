import os
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__file__)


# Base model and path to model file
model_name = "microsoft/Phi-3.5-mini-instruct"
model_info = "model.json"
model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model")
model_tokenizer = os.path.join(model_dir, "tokenizer")
model_path = os.path.join(model_dir, "phi-3.5-mini-base")


def download_base_model() -> None:
    # Check if the model has already been downloaded
    if os.path.exists(model_path):
        print(f"{model_name} has already been downloaded. Skipping...")
        return

    # Download tokenizer
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(model_path)

    # Download model
    print(f"Downloading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.save_pretrained(model_path)
    print(f"Base model saved to: {model_path}")

    # Save model-info.json
    meta_data_file = os.path.join(model_dir, model_info)
    print(f"Saving model meta data to {meta_data_file}")
    with open(meta_data_file, "w") as f:
        model_meta_data = {
            model_name: {
                "model-dir": model_dir,
                "model-file": model_path,
            }
        }
        json.dump(obj=model_meta_data, fp=f, indent=2)


if __name__ == "__main__":
    download_base_model()
