"""
Training script for PHI-3.5 Agent fine-tuning.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.trainer import AgentTrainer
from data.dataset_builder import AgenticDatasetBuilder
import wandb


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )


def prepare_training_data(data_dir: Path, num_samples: int = 5000) -> None:
    """Prepare training data if it doesn't exist."""
    train_file = data_dir / "processed" / "train_dataset.json"
    eval_file = data_dir / "processed" / "eval_dataset.json"

    if not train_file.exists() or not eval_file.exists():
        print("Generating training dataset...")

        builder = AgenticDatasetBuilder(str(data_dir / "processed"))

        # Generate full dataset
        full_dataset = builder.generate_dataset(num_samples)

        # Split into train/eval
        split_idx = int(len(full_dataset) * 0.9)
        train_dataset = full_dataset[:split_idx]
        eval_dataset = full_dataset[split_idx:]

        # Save datasets
        builder.save_dataset(train_dataset, "train_dataset.json")
        builder.save_dataset(eval_dataset, "eval_dataset.json")

        print(
            f"Created {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples"
        )
    else:
        print("Training data already exists, skipping generation.")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PHI-3.5 Agent Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--data-samples",
        type=int,
        default=5000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="phi3-agent-finetuning",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/phi3-agent-final",
        help="Output directory for final model",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--skip-data-prep", action="store_true", help="Skip data preparation step"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting PHI-3.5 Agent Training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data if needed
        if not args.skip_data_prep:
            data_dir = Path("./data")
            prepare_training_data(data_dir, args.data_samples)

        # Initialize Weights & Biases
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "data_samples": args.data_samples,
                "config_file": args.config,
                "output_dir": args.output_dir,
            },
        )

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = AgentTrainer(args.config)

        # Start training
        logger.info("Starting training process...")
        trainer.train()

        # Evaluate model
        logger.info("Evaluating trained model...")
        eval_results = trainer.evaluate_model()

        # Log final metrics to wandb
        wandb.log({"final_" + k: v for k, v in eval_results.items()})

        # Save final model
        logger.info(f"Saving final model to {args.output_dir}")
        trainer.save_model_for_inference(args.output_dir)

        logger.info("Training completed successfully!")

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model saved to: {args.output_dir}")
        print(f"Final evaluation loss: {eval_results.get('eval_loss', 'N/A')}")
        print(f"Training samples: {args.data_samples}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
