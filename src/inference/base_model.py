import os
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_configs import ModelConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


# Base model and path to model file
model_name = ModelConfig.MODEL_NAME
model_dir = ModelConfig.MODEL_DIR
model_tokenizer = ModelConfig.MODEL_TOKENIZER
model_path = ModelConfig.MODEL_PATH
mcp_server_addr = ModelConfig.MCP_SERVER


def download_base_model() -> bool:
    # Check if the model has already been downloaded
    if os.path.exists(model_path):
        print(f"{model_name} has already been downloaded. Skipping...")
        return True

    # Download tokenizer and base model
    os.makedirs(model_path, exist_ok=True)

    try:
        print(f"Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(model_path)

        # Download model
        print(f"Downloading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, dtype=torch.float16
        )
        model.save_pretrained(model_path)
        print(f"Base model saved to: {model_path}")
    except Exception as e:
        print(f"Error downloading model {model_name}: {e}")
        return False

    # Save model-info.json
    meta_data_file = ModelConfig.MODEL_META_DATA
    print(f"Saving model meta data to {meta_data_file}")
    with open(meta_data_file, "w") as f:
        model_meta_data = {
            "base_model": model_name,
            "model": model_name,
            "adapter": None,
            "dtype": "float16",
            "device": "auto",
            "tokenizer": os.path.join(model_path, "tokenizer.json"),
            "model-dir": model_dir,
            "model-file": model_path,
            "mcp-server": mcp_server_addr,
        }
        json.dump(obj=model_meta_data, fp=f, indent=2)

    return True


if __name__ == "__main__":
    download_base_model()
