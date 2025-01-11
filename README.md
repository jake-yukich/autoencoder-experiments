# autoencoder-experiments
Project structure:
```text
autoencoder-experiments/
├── README.md                   # Project overview, setup instructions, results
├── requirements.txt           
├── setup.py
├── configs/
│   ├── model_configs/         # YAML files for model architectures
│   │   ├── linear_ae.yaml
│   │   ├── conv_ae.yaml
│   │   └── gan.yaml
│   └── training_configs/      # Training hyperparameters
│       ├── mnist_config.yaml
│       └── cifar_config.yaml
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base classes
│   │   ├── linear_ae.py      # Linear autoencoder
│   │   ├── conv_ae.py        # Convolutional autoencoder
│   │   └── gan.py            # GAN implementation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py       # Dataset classes
│   │   └── transforms.py     # Custom transforms (noise, masking)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop logic
│   │   └── metrics.py        # Performance metrics
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py   # Plotting utilities
│       └── logging.py        # Logging utilities
├── notebooks/
│   ├── exploration.ipynb     # Original notebook
│   └── results_analysis.ipynb # Results visualization
├── scripts/
│   ├── train.py             # Training entry point
│   └── evaluate.py          # Evaluation script
├── tests/
│   ├── test_models.py
│   ├── test_datasets.py
│   └── test_transforms.py
└── docs/
    ├── architecture.md       # Model architecture details
    ├── experiments.md        # Experiment results
    └── api/                  # API documentation
```

Key organizational principles:
1. Separation of Concerns
  * Models separate from training logic
  * Config files separate from code
  * Clear distinction between library code and scripts
2. Reproducibility
  * Configs version controlled
  * Requirements clearly specified
  * Documentation of results and methods
3. Testing & Documentation
  * Unit tests for core functionality
  * Clear documentation of design decisions
  * API documentation for reuse
4. Flexibility
  * Easy to add new models
  * Easy to modify training procedures
  * Easy to run different experiments
