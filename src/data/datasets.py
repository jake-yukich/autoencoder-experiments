from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent

class DatasetFactory:
    """Factory class for creating datasets based on config"""
    
    @staticmethod
    def create_dataset(config, split='train'):
        """
        Create dataset based on config specifications
        
        Args:
            config: Dictionary containing dataset configuration
            split: 'train' or 'test'
        """
        dataset_name = config['dataset']['name'].lower()
        
        if dataset_name == 'mnist':
            return MNISTDataset(config, split)
        elif dataset_name == 'fashion_mnist':
            return FashionMNISTDataset(config, split)
        elif dataset_name == 'cifar10':
            return CIFAR10Dataset(config, split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    @staticmethod
    def create_dataloader(dataset, config):
        """Create dataloader with config specifications"""
        return DataLoader(
            dataset,
            batch_size=config['dataset']['batch_size'],
            num_workers=config['dataset']['num_workers'],
            shuffle=(dataset.split == 'train'),
            pin_memory=True
        )

class BaseDataset:
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.data_dir = PROJECT_ROOT / Path(config['dataset'].get('data_dir', 'src/data/dumps'))
        self.transforms = self._get_transforms()
        
    def _get_transforms(self):
        """Get transforms based on dataset config"""
        transform_list = []
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(**self.config['dataset'].get('normalize', 
                {'mean': [0.5], 'std': [0.5]}))
        ])
        
        # Add noise if specified in config
        # if 'noise' in self.config:
        #     transform_list.append(NoiseTransform(self.config['noise']['intensity']))
        
        return transforms.Compose(transform_list)

class MNISTDataset(BaseDataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        
        try:
            with tqdm(desc=f"Loading MNIST ({split})") as pbar:
                self.dataset = datasets.MNIST(
                    root=self.data_dir,
                    train=(split == 'train'),
                    download=True,
                    transform=self.transforms
                )
                pbar.update()
        except Exception as e:
            raise RuntimeError(f"Failed to load/download MNIST dataset: {str(e)}")
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class FashionMNISTDataset(BaseDataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        
        try:
            with tqdm(desc=f"Loading Fashion-MNIST ({split})") as pbar:
                self.dataset = datasets.FashionMNIST(
                    root=self.data_dir,
                    train=(split == 'train'),
                    download=True,
                    transform=self.transforms
                )
                pbar.update()
        except Exception as e:
            raise RuntimeError(f"Failed to load/download Fashion-MNIST dataset: {str(e)}")
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class CIFAR10Dataset(BaseDataset):
    def __init__(self, config, split):
        super().__init__(config, split)
        
        try:
            with tqdm(desc=f"Loading CIFAR10 ({split})") as pbar:
                self.dataset = datasets.CIFAR10(
                    root=self.data_dir,
                    train=(split == 'train'),
                    download=True,
                    transform=self.transforms
                )
                pbar.update()
        except Exception as e:
            raise RuntimeError(f"Failed to load/download CIFAR10 dataset: {str(e)}")
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]