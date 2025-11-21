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
    MODEL_NAME: str = os.getenv("SLM_MODEL_NAME", DEFAULT_MODEL)
    MODEL_DIR: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model")
    MODEL_HOST: str = os.path.join("SLM_MODEL_HOST", "0.0.0.0")
    MODEL_PORT: int = int(os.path.join("SLM_MODEL_PORT", "9999"))
    MODEL_PATH: str = os.getenv("SLM_MODEL_PATH", os.path.join(MODEL_DIR, MODEL_NAME))
    MODEL_TOKENIZER: str = os.getenv(
        "SLM_MODEL_TOKENIZER", os.path.join(MODEL_PATH, "tokenizer.json")
    )
    MODEL_META_DATA: str = os.getenv(
        "SLM_MODEL_META_DATA", os.path.join(MODEL_PATH, "model.json")
    )
    MCP_SERVER: str = os.getenv("SLM_MCP_SERVER", "http://localhost:9000/mcp")
