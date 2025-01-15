import yaml
import argparse
from pathlib import Path
from src.models.factory import ModelFactory
from src.data.datasets import DatasetFactory
from src.training.trainer import Trainer

def load_configs(training_config_path, model_config_path=None):
    """Load separate config files"""
    # Load training config
    with open(training_config_path) as f:
        train_config = yaml.safe_load(f)
    
    # Load model config from either specified path or training config
    model_config_path = model_config_path or train_config.get('model_config')
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    
    return train_config, model_config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train autoencoder model')
    parser.add_argument('--train-config', type=str, required=True,
                      help='Path to training config YAML file')
    parser.add_argument('--model-config', type=str, required=False,
                      help='Optional path to model config YAML file. If not provided, will use path from training config.')
    args = parser.parse_args()
    
    # Load configs
    train_config, model_config = load_configs(args.train_config, args.model_config)

    # Create model
    model = ModelFactory.create_model(model_config)

    # Create datasets
    train_dataset = DatasetFactory.create_dataset(train_config, split='train')
    val_dataset = DatasetFactory.create_dataset(train_config, split='val')
    
    train_loader = DatasetFactory.create_dataloader(train_dataset, train_config)
    val_loader = DatasetFactory.create_dataloader(val_dataset, train_config)

    # Train
    trainer = Trainer(model, train_config)
    train_losses, val_losses = trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()

"""
from src.models.factory import ModelFactory
from src.data.datasets import DatasetFactory
from src.training.trainer import Trainer

def main():
    # Load config
    config = {
        'model': {...},
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'optimizer': {
                'type': 'Adam',
                'params': {'betas': (0.9, 0.999)}
            }
        },
        'dataset': {...}
    }
    
    # Create model and datasets
    model = ModelFactory.create_model(config['model'])
    train_dataset = DatasetFactory.create_dataset(config['dataset'], split='train')
    val_dataset = DatasetFactory.create_dataset(config['dataset'], split='val')
    
    train_loader = DatasetFactory.create_dataloader(train_dataset, config)
    val_loader = DatasetFactory.create_dataloader(val_dataset, config)
    
    # Create trainer and train
    trainer = Trainer(model, config)
    train_losses, val_losses = trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
"""