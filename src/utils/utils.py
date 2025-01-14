import torch
import yaml

# Noise:
# -------------------------------------------------------------------------------------------------

def add_noise(data, noise_level=0.2):
    """Add noise to the input data"""
    noise = torch.randn_like(data).to(data.device) * noise_level
    return data + noise

# Pickleable Noise Transform:
# class NoiseTransform:
#     """Transform class for adding Gaussian noise to images"""
#     def __init__(self, intensity):
#         self.intensity = intensity
    
#     def __call__(self, x):
#         noise = torch.randn_like(x) * self.intensity
#         return torch.clamp(x + noise, 0, 1)

# LOAD CONFIG:
# -------------------------------------------------------------------------------------------------

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

# Masking:
# -------------------------------------------------------------------------------------------------

class Mask:
    def __init__(self, size=10, location=None):
        """Initialize a mask configuration
        
        Args:
            size: Size of the square mask
            location: Optional tuple (x,y) for mask position. If None, a random location will be used.
        """
        self.size = size
        self.location = location

    def apply(self, images):
        """Apply the mask to input images
        
        Args:
            images: Input tensor of shape (B, C, H, W)
        Returns:
            Masked version of input images
        """
        masked = images.clone()
        _, _, height, width = images.shape
        
        if self.location is None:
            x = torch.randint(0, height - self.size + 1, (1,)).item()
            y = torch.randint(0, width - self.size + 1, (1,)).item()
        else:
            x, y = self.location
            
        x = min(max(x, 0), height - self.size)
        y = min(max(y, 0), width - self.size)
        
        # Store actual location used (e.g. for visualization)
        self.applied_location = (x, y)
        
        masked[:, :, x:x+self.size, y:y+self.size] = 0
        return masked
    
    @property
    def bounds(self):
        """Returns the bounds of the last applied mask as (x, y, width, height)"""
        if hasattr(self, 'applied_location'):
            x, y = self.applied_location
            return (x, y, x + self.size, y + self.size)
        return None

def create_masked_images(images, mask_size=10, mask_location=None):
    """Create masked versions of input images by adding a black square at random location
    
    Args:
        images: Input tensor of shape (B, C, H, W)
        mask_size: Size of the square mask
        mask_location: Optional tuple (x,y) for mask position. If None, random location is used.
    """
    mask = Mask(size=mask_size, location=mask_location)
    return mask.apply(images)
