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