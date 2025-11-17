import asyncio
import argparse

from src.inference.model_configs import ModelConfig
from src.inference.model_handler import setup_model_handler

model_config = ModelConfig()
model = setup_model_handler(model_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SLM Chat session")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Whether to stream the models responses (defaults to False)",
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.streaming:
        asyncio.run(model.streaming_chat())
    else:
        asyncio.run(model.chat())
