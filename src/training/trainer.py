"""
Fine-tuning trainer with LoRA for agentic use cases.
"""

import torch
import yaml
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
import wandb

# Set up logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


@dataclass
class ModelTrainingConfig:
    """Configuration for model training."""

    model_name: str
    output_dir: str
    train_dataset: str
    eval_dataset: str
    config_path: str


class AgentTrainer:
    def __init__(self, config_path: str):
        """Initialize the trainer with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _load_config(self) -> dict[str, Any]:
        """Load training configuration from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with LoRA configuration."""
        log.info("Loading model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=self.config["model"]["trust_remote_code"],
            padding_side="right",
        )

        # Add special tokens for tool usage
        special_tokens = {
            "additional_special_tokens": [
                "<tool_use>",
                "</tool_use>",
                "<tool_name>",
                "</tool_name>",
                "<parameters>",
                "</parameters>",
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=self.config["model"]["trust_remote_code"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=self.config["quantization"]["load_in_4bit"],
        )

        # Resize token embeddings for new special tokens
        model.resize_token_embeddings(len(self.tokenizer))

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Setup LoRA configuration
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias=self.config["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to model
        self.model = get_peft_model(model, lora_config)
        self.model.log.info_trainable_parameters()

    def prepare_datasets(self) -> None:
        """Load and prepare training and evaluation datasets."""
        log.info("Loading datasets...")

        # Load datasets
        train_dataset = load_dataset(
            "json", data_files=self.config["data"]["train_dataset"]
        )["train"]
        eval_dataset = load_dataset(
            "json", data_files=self.config["data"]["eval_dataset"]
        )["train"]

        # Tokenize datasets
        def tokenize_function(examples):
            # Create full text by combining instruction, input, and output
            texts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                input_text = examples["input"][i] if examples["input"][i] else ""
                output = examples["output"][i]

                # Format as instruction-following conversation
                if input_text:
                    full_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    full_text = (
                        f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    )

                texts.append(full_text)

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config["data"]["max_seq_length"],
                return_tensors=None,
            )

            # Set labels (copy of input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Apply tokenization
        self.train_dataset = train_dataset.map(
            tokenize_function, batched=True, remove_columns=train_dataset.column_names
        )

        self.eval_dataset = eval_dataset.map(
            tokenize_function, batched=True, remove_columns=eval_dataset.column_names
        )

        log.info(f"Train dataset size: {len(self.train_dataset)}")
        log.info(f"Eval dataset size: {len(self.eval_dataset)}")

    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments."""
        training_config = self.config["training"]

        return TrainingArguments(
            output_dir=training_config["output_dir"],
            num_train_epochs=training_config["num_train_epochs"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            optim=training_config["optim"],
            save_steps=training_config["save_steps"],
            logging_steps=training_config["logging_steps"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            fp16=training_config["fp16"],
            bf16=training_config["bf16"],
            max_grad_norm=training_config["max_grad_norm"],
            max_steps=training_config["max_steps"],
            warmup_ratio=training_config["warmup_ratio"],
            group_by_length=training_config["group_by_length"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            report_to=training_config["report_to"],
            # Evaluation settings
            evaluation_strategy=self.config["evaluation"]["evaluation_strategy"],
            eval_steps=self.config["evaluation"]["eval_steps"],
            save_strategy=self.config["evaluation"]["save_strategy"],
            metric_for_best_model=self.config["evaluation"]["metric_for_best_model"],
            greater_is_better=self.config["evaluation"]["greater_is_better"],
            load_best_model_at_end=self.config["evaluation"]["load_best_model_at_end"],
            # Additional settings
            remove_unused_columns=False,
            run_name=f"phi3-agent-{wandb.util.generate_id()}",
            seed=self.config["data"]["seed"],
        )

    def train(self) -> Trainer:
        """Execute the training process."""
        log.info("Starting training...")

        # Setup model and datasets
        self.setup_model_and_tokenizer()
        self.prepare_datasets()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )

        # Training arguments
        training_args = self.setup_training_arguments()

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Start training
        self.trainer.train()

        # Save final model
        self.trainer.save_model()
        log.info(f"Training completed. Model saved to {training_args.output_dir}")

        return self.trainer

    def evaluate_model(self) -> dict[str, float]:
        """Evaluate the trained model."""
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")

        log.info("Evaluating model...")
        eval_results = self.trainer.evaluate()

        log.info("Evaluation Results:")
        for key, value in eval_results.items():
            print(f"{key}: {value}")

        return eval_results

    def save_model_for_inference(self, save_path: str) -> None:
        """Save the final model in a format suitable for inference."""
        log.info(f"Saving model for inference to {save_path}")

        # Save the adapter
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save configuration
        config_save_path = Path(save_path) / "training_config.yaml"
        with open(config_save_path, "w") as f:
            yaml.dump(self.config, f)

        log.info(f"Model and configuration saved to {save_path}")


def main() -> None:
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training config file"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="phi3-agent-finetuning",
        help="Wandb project name",
    )
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project=args.wandb_project)

    # Train model
    trainer = AgentTrainer(args.config)
    trainer.train()
    trainer.evaluate_model()

    # Save for inference
    trainer.save_model_for_inference("./models/phi3-agent-final")

    wandb.finish()


if __name__ == "__main__":
    main()
