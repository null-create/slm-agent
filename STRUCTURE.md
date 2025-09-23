```
slm-agent/
├── requirements.txt
├── config/
│   ├── training_config.yaml
│   └── model_config.json
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_builder.py
│   │   ├── data_formatter.py
│   │   └── data_validator.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── lora_config.py
│   │   └── evaluation.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_handler.py
│   │   └── mcp_client.py
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py
│       └── metrics.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── inference_demo.py
└── tests/
    ├── test_data_processing.py
    ├── test_training.py
    └── test_inference.py
```
