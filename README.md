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

---



Example of how to use configs:

```yaml:configs/model_configs/conv_ae.yaml
# Model architecture configuration
model_name: "ConvolutionalAutoencoder"
input_channels: 1  # MNIST=1, CIFAR=3
architecture:
  encoder:
    - layer1:
        type: "Conv2d"
        out_channels: 16
        kernel_size: 5
        stride: 2
        padding: 2
        batch_norm: true
    - layer2:
        type: "Conv2d"
        out_channels: 32
        kernel_size: 5
        stride: 2
        padding: 2
        batch_norm: true
    - layer3:
        type: "Conv2d"
        out_channels: 64
        kernel_size: 7
        padding: 3
        batch_norm: true
  decoder:
    - layer1:
        type: "ConvTranspose2d"
        out_channels: 32
        kernel_size: 7
        padding: 3
        batch_norm: true
    - layer2:
        type: "ConvTranspose2d"
        out_channels: 16
        kernel_size: 5
        stride: 2
        padding: 2
        output_padding: 1
        batch_norm: true
    - layer3:
        type: "ConvTranspose2d"
        out_channels: 1  # Match input_channels
        kernel_size: 5
        stride: 2
        padding: 2
        output_padding: 1
        activation: "sigmoid"
```

```yaml:configs/training_configs/mnist_config.yaml
# Training configuration
dataset:
  name: "MNIST"
  batch_size: 128
  num_workers: 4

training:
  epochs: 10
  learning_rate: 0.001
  optimizer:
    type: "Adam"
    betas: [0.9, 0.999]
  scheduler:
    type: "ReduceLROnPlateau"
    patience: 3
    factor: 0.1

noise:
  type: "gaussian"
  intensity: 0.2

logging:
  log_interval: 100
  save_interval: 1000
```

Then create a model factory that uses these configs:

```python:src/models/factory.py
import yaml
import torch.nn as nn

class ModelFactory:
    @staticmethod
    def create_model(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if config['model_name'] == 'ConvolutionalAutoencoder':
            return ConvolutionalAutoencoder(config)

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Build encoder
        encoder_layers = []
        for layer_config in config['architecture']['encoder']:
            encoder_layers.extend(self._create_layer(layer_config))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        for layer_config in config['architecture']['decoder']:
            decoder_layers.extend(self._create_layer(layer_config))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _create_layer(self, layer_config):
        layers = []
        layer = getattr(nn, layer_config['type'])(
            **{k: v for k, v in layer_config.items() 
               if k not in ['type', 'batch_norm', 'activation']}
        )
        layers.append(layer)
        
        if layer_config.get('batch_norm', False):
            layers.append(nn.BatchNorm2d(layer_config['out_channels']))
            
        if 'activation' in layer_config:
            layers.append(getattr(nn, layer_config['activation'])())
        else:
            layers.append(nn.ReLU())
            
        return layers
```

And use it in the training script:

```python:scripts/train.py
import yaml
from src.models.factory import ModelFactory
from src.training.trainer import Trainer

def main():
    # Load configs
    with open('configs/model_configs/conv_ae.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    with open('configs/training_configs/mnist_config.yaml', 'r') as f:
        training_config = yaml.safe_load(f)
    
    # Create model
    model = ModelFactory.create_model(model_config)
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```

Benefits of this approach:
1. **Experiment Tracking**: Easy to version control different model configurations
2. **Reproducibility**: All parameters are documented
3. **Flexibility**: Easy to modify architecture without changing code
4. **Clarity**: Clear separation between model definition and hyperparameters
5. **Reusability**: Can reuse same configs across different experiments

(Could also extend to use tools like Hydra.)

