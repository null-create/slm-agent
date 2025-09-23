# PHI-3.5 Agent Fine-tuning for MCP Integration

This project provides a complete pipeline for fine-tuning Microsoft's PHI-3.5-Mini-Instruct model for agentic use cases with Model Context Protocol (MCP) server integration and external tool usage.

## 🎯 Project Overview

Fine-tune PHI-3.5-Mini-Instruct to create an intelligent agent capable of:

- **Tool Selection**: Automatically choose appropriate tools for tasks
- **Parameter Extraction**: Extract correct parameters for tool calls
- **Multi-step Reasoning**: Chain multiple tool calls to complete complex tasks
- **MCP Integration**: Seamlessly work with external MCP servers
- **Error Handling**: Gracefully handle tool failures and edge cases

## 🏗️ Architecture

```
User Request → PHI-3.5 Agent → Tool Selection → MCP Client → External Tools
                     ↑                                              ↓
              Final Response ← Response Generation ← Tool Results ←
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to project
git clone <repository>
cd phi3-agent-finetuning

# Create virtual environment
python -m venv phi3-env
source phi3-env/bin/activate  # Windows: phi3-env\Scripts\activate

# Run setup script
python scripts/setup.py
```

### 2. Configuration

Review and customize `config/training_config.yaml`:

- Adjust batch sizes based on your GPU memory
- Modify LoRA parameters for your use case
- Set training epochs and learning rate

### 3. Training

```bash
# Train with default settings
python scripts/train_model.py --config config/training_config.yaml

# Custom training
python scripts/train_model.py \
  --config config/training_config.yaml \
  --data-samples 10000 \
  --wandb-project my-phi3-agent
```

### 4. Evaluation

```bash
# Comprehensive evaluation
python scripts/evaluate_model.py \
  --model-path ./models/phi3-agent-final \
  --run-benchmarks

# Generate evaluation dataset
python scripts/evaluate_model.py \
  --model-path ./models/phi3-agent-final \
  --generate-eval-data \
  --eval-samples 500
```

### 5. Inference Demo

```bash
# Interactive demo
python scripts/inference_demo.py \
  --model-path ./models/phi3-agent-final \
  --mode interactive

# Run benchmarks
python scripts/inference_demo.py \
  --model-path ./models/phi3-agent-final \
  --mode benchmark
```

## 📊 Success Metrics

The model is evaluated on multiple dimensions:

### Core Metrics

- **Tool Selection Accuracy**: >85% (correct tool choice)
- **Parameter Extraction**: >90% (accurate parameter parsing)
- **Task Completion Rate**: >80% (successful end-to-end execution)
- **Hallucination Rate**: <10% (factual accuracy)
- **Response Time**: <5s average (performance)

### Evaluation Categories

- **Single Tool Usage**: Simple, direct tool calls
- **Multi-step Tasks**: Complex workflows requiring multiple tools
- **Error Handling**: Graceful failure recovery
- **Context Maintenance**: Coherence across conversation turns

## 🔧 Customization

### Adding New Tools

1. **Update MCP Client** (`src/inference/mcp_client.py`):

```python
self.available_tools["new_tool"] = {
    "server": "tool_server",
    "endpoint": "/new_endpoint",
    "description": "Tool description",
    "parameters": {
        "param1": {"type": "string", "required": True}
    }
}
```

2. **Update Dataset Builder** (`src/data/dataset_builder.py`):

```python
# Add tool scenarios
tool_scenarios["new_tool"] = ["scenario1", "scenario2"]
```

3. **Regenerate Training Data**:

```bash
python scripts/train_model.py --config config/training_config.yaml
```

### Fine-tuning Hyperparameters

Key parameters in `config/training_config.yaml`:

```yaml
lora:
  r: 16 # LoRA rank (8-64)
  lora_alpha: 32 # LoRA scaling (16-64)
  lora_dropout: 0.1 # Dropout rate (0.05-0.2)

training:
  learning_rate: 2.0e-4 # Learning rate (1e-4 to 5e-4)
  num_train_epochs: 3 # Training epochs (2-5)
  per_device_train_batch_size: 4 # Batch size (2-8)
```

## 📁 Project Structure

```
phi3-agent-finetuning/
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── config/
│   └── training_config.yaml      # Training configuration
├── src/
│   ├── data/                     # Data processing modules
│   │   ├── dataset_builder.py    # Training data generation
│   │   └── data_formatter.py     # Data formatting utilities
│   ├── training/                 # Training modules
│   │   ├── trainer.py            # Main training logic
│   │   └── evaluation.py         # Model evaluation
│   └── inference/                # Inference modules
│       ├── model_handler.py      # Model inference handler
│       └── mcp_client.py         # MCP client implementation
├── scripts/
│   ├── setup.py                  # Environment setup
│   ├── train_model.py           # Training script
│   ├── evaluate_model.py        # Evaluation script
│   └── inference_demo.py        # Demo script
└── data/                        # Data directories
    ├── raw/                     # Raw data files
    ├── processed/               # Processed datasets
    └── evaluation/              # Evaluation datasets
```

## 🛠️ Development Workflow

### 1. Data Preparation Phase

```bash
# Generate custom training data
python -c "
from src.data.dataset_builder import AgenticDatasetBuilder
builder = AgenticDatasetBuilder()
dataset = builder.generate_dataset(5000)
builder.save_dataset(dataset, 'custom_dataset.json')
"
```

### 2. Experimental Training

```bash
# Quick training run for testing
python scripts/train_model.py \
  --config config/training_config.yaml \
  --data-samples 1000 \
  --wandb-project phi3-experiment
```

### 3. Model Evaluation

```bash
# Detailed evaluation with custom metrics
python scripts/evaluate_model.py \
  --model-path ./results/checkpoint-1000 \
  --eval-dataset ./data/custom/eval.json \
  --run-benchmarks
```

### 4. Production Deployment

```bash
# Export optimized model
python -c "
from src.training.trainer import PHI3AgentTrainer
trainer = PHI3AgentTrainer('config/training_config.yaml')
trainer.save_model_for_inference('./models/production')
"
```

## 🔍 Troubleshooting

### Common Issues

**GPU Memory Errors**:

```yaml
# Reduce batch sizes in config/training_config.yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

**Slow Training**:

```yaml
# Enable mixed precision
bf16: true
fp16: false # Use bf16 instead of fp16 for better stability
```

**Tool Call Parsing Issues**:

- Check tool usage format in training data
- Validate JSON parameter formatting
- Ensure consistent tool naming

**Model Not Learning**:

- Increase learning rate to 3e-4
- Add more diverse training examples
- Check data quality and formatting

### Performance Optimization

**Memory Usage**:

- Use gradient checkpointing: `gradient_checkpointing: true`
- Enable 4-bit quantization: `load_in_4bit: true`
- Reduce sequence length: `max_seq_length: 1024`

**Training Speed**:

- Increase batch size if memory allows
- Use multiple GPUs with `--multi_gpu`
- Enable compilation: `torch_compile: true` (PyTorch 2.0+)

## 📈 Monitoring

### Weights & Biases Integration

The project includes comprehensive W&B logging:

- Training/validation loss curves
- Tool usage accuracy metrics
- Parameter extraction success rates
- Response quality scores
- Hardware utilization

Access your runs at: `https://wandb.ai/<username>/<project>`

### Local Monitoring

Check training progress:

```bash
# View training logs
tail -f training.log

# Monitor GPU usage
nvidia-smi -l 1

# Check disk space
df -h
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-capability`
3. **Add tests**: Ensure new features have appropriate test coverage
4. **Submit PR**: Include detailed description and test results

### Adding New Evaluation Metrics

```python
# In src/training/evaluation.py
def _evaluate_custom_metric(self, sample, result):
    """Add your custom evaluation logic."""
    return score

# Update evaluate_full_model to include new metric
```

## 📚 References

- [PHI-3.5 Model Paper](https://arxiv.org/abs/2404.14219)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

For questions and support:

- Open an issue in the repository
- Check the troubleshooting section above
- Review the evaluation metrics for model performance
