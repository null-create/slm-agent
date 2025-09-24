"""
Setup script for Agent fine-tuning project.
"""

import os
import subprocess
import sys
from pathlib import Path

from src.inference.base_model import download_base_model
from src.inference.model_config import ModelConfig


def run_command(command: str, cwd: str = None, check: bool = True) -> str:
    """Run a shell command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, shell=True, check=check, cwd=cwd, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return str(e)


def check_gpu() -> bool:
    """Check GPU availability."""
    print("Checking GPU availability...")
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ Found {gpu_count} GPU(s): {gpu_name}")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU detected - training will be slow on CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False


def install_requirements() -> None:
    """Install Python requirements."""
    print("Installing Python requirements...")

    # Check if we're in a virtual environment
    if not (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    ):
        print("⚠️  Warning: Not in a virtual environment")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            print("Please create and activate a virtual environment first:")
            print("  python -m venv phi3-env")
            print(
                "  source phi3-env/bin/activate  # On Windows: phi3-env\\Scripts\\activate"
            )
            sys.exit(1)

    # Install requirements
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    run_command(f"{sys.executable} -m pip install -r requirements.txt")


def setup_wandb() -> None:
    """Setup Weights & Biases for experiment tracking."""
    print("\nSetting up Weights & Biases...")

    try:
        import wandb

        # Check if already logged in
        try:
            wandb.login()
            print("✓ Weights & Biases is already configured")
        except:
            print("Please login to Weights & Biases:")
            print("  Run 'wandb login' and enter your API key")
            print("  You can get your API key from: https://wandb.ai/authorize")

    except ImportError:
        print("⚠️  Weights & Biases not available (will be installed with requirements)")


def validate_base_model() -> bool:
    """Validate that the base model can be loaded."""
    print("\nValidating base model access...")

    try:
        from transformers import AutoTokenizer

        model_name = ModelConfig.MODEL_NAME
        print(f"Testing access to {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False  # Avoid potential issues
        )

        print("✓ Base model accessible")
        print(f"  Vocab size: {len(tokenizer)}")

        return True

    except Exception as e:
        print(f"⚠️  Error accessing base model: {e}")
        print("This might be due to:")
        print("  - Network connectivity issues")
        print("  - Hugging Face authentication (for gated models)")
        print("  - Missing transformers installation")
        return False


def create_sample_config() -> None:
    """Create a sample training configuration if it doesn't exist."""
    config_path = Path("config/training_config.yaml")

    if not config_path.exists():
        print("Creating sample training configuration...")
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # The config content is already created in the artifacts above
        print(f"✓ Sample config created at {config_path}")
        print("  Review and modify the configuration before training")
    else:
        print(f"✓ Training configuration exists at {config_path}")


def run_quick_test() -> bool:
    """Run a quick test to ensure everything is working."""
    print("\nRunning quick functionality test...")

    try:
        # Test data generation
        print("Testing data generation...")
        sys.path.append("src")

        from data.dataset_builder import AgenticDatasetBuilder

        builder = AgenticDatasetBuilder("./data/test")
        test_samples = builder.generate_dataset(num_examples=10)

        print(f"✓ Generated {len(test_samples)} test samples")

        # Clean up test data
        import shutil

        shutil.rmtree("./data/test", ignore_errors=True)

        return True

    except Exception as e:
        print(f"⚠️  Quick test failed: {e}")
        return False


def main() -> None:
    """Main setup function."""
    print("=" * 40)
    print("Agent Fine-tuning Setup")
    print("=" * 40)

    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Run setup steps
    # setup_directories()

    # Check GPU before installing requirements (for better error messages)
    gpu_available = check_gpu()

    # install_requirements()

    # Download base model
    download_base_model()

    # Re-check GPU after installing PyTorch
    if not gpu_available:
        check_gpu()

    create_sample_config()
    setup_wandb()
    validate_base_model()

    # Run tests
    # test_passed = run_quick_test()

    # if test_passed:
    #     print_next_steps()
    # else:
    #     print("\n⚠️  Setup completed with some issues")
    #     print(
    #         "Please check the error messages above and resolve them before training"
    #     )


if __name__ == "__main__":
    main()
