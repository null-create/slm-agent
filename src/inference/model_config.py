import os


class ModelConfig:
    MODEL_NAME = os.getenv("SLM_MODEL_NAME", "microsoft/Phi-3.5-mini-instruct")
    MODEL_DIR = os.getenv(
        "SLM_MODEL_PATH",
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "model", MODEL_NAME),
    )
    MODEL_TOKENIZER = os.getenv(
        "SLM_MODEL_TOKENIZER", os.path.join(MODEL_DIR, "tokenizer.json")
    )
    MODEL_META_DATA = os.getenv(
        "SLM_MODEL_META_DATA", os.path.join(MODEL_DIR, "model.json")
    )
