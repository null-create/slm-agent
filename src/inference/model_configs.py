import os

from dotenv import load_dotenv

load_dotenv()

# Default base model to use if none is specified
#
# This model was chosen because it is relatively small and can run on a single GPU is
# also an instruction-tuned model, which makes it more suitable for chat applications.
#
# See https://huggingface.co/microsoft/Phi-3.5-mini-instruct
DEFAULT_MODEL = "microsoft/Phi-3.5-mini-instruct"


class ModelConfig:
    MODEL_NAME = os.getenv("SLM_MODEL_NAME", DEFAULT_MODEL)
    MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model")
    MODEL_PATH = os.getenv("SLM_MODEL_PATH", os.path.join(MODEL_DIR, MODEL_NAME))
    MODEL_TOKENIZER = os.getenv(
        "SLM_MODEL_TOKENIZER", os.path.join(MODEL_PATH, "tokenizer.json")
    )
    MODEL_META_DATA = os.getenv(
        "SLM_MODEL_META_DATA", os.path.join(MODEL_PATH, "model.json")
    )
    MCP_SERVER = os.getenv("SLM_MCP_SERVER", "http://localhost:9000/mcp")
