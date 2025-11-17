import asyncio

from src.inference.model_configs import ModelConfig
from src.inference.model_handler import setup_model_handler

model_config = ModelConfig()
model = setup_model_handler(model_config)


if __name__ == "__main__":
    asyncio.run(model.chat())
