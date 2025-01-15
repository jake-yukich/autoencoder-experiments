from src.data.datasets import DatasetFactory

# class Trainer:
#     def __init__(self, model, config):
#         self.model = model
#         self.config = config
        
#         # Create datasets and dataloaders
#         train_dataset = DatasetFactory.create_dataset(config, split='train')
#         test_dataset = DatasetFactory.create_dataset(config, split='test')
        
#         self.train_loader = DatasetFactory.create_dataloader(train_dataset, config)
#         self.test_loader = DatasetFactory.create_dataloader(test_dataset, config)
        
#         # Rest of trainer initialization...

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import wandb  # Optional, for logging

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup from config
        self.setup_training()
        
    def setup_training(self):
        """Initialize optimizer, criterion, etc. from config"""
        optimizer_config = self.config['training']['optimizer']
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            **optimizer_config.get('params', {})
        )
        
        self.criterion = nn.MSELoss()
        self.epochs = self.config['training']['epochs']
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstructed = self.model(data)
                loss = self.criterion(reconstructed, data)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                reconstructed = self.model(data)
                loss = self.criterion(reconstructed, data)
                val_loss += loss.item()
                
        return val_loss / len(val_loader)
    
    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                
                print(f'Epoch {epoch+1}/{self.epochs} - '
                      f'Train Loss: {train_loss:.4f} - '
                      f'Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{self.epochs} - '
                      f'Train Loss: {train_loss:.4f}')
            
            # Optional: Log to wandb
            if self.config.get('logging', {}).get('use_wandb', False):
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss if val_loader else None,
                    'epoch': epoch
                })
        
        return train_losses, val_losses
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])