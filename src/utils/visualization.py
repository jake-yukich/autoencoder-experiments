import matplotlib.pyplot as plt

def plot_masked_image(image, mask):
    plt.imshow(image)
    if mask.bounds:
        x, y, w, h = mask.bounds
        rect = plt.Rectangle((y, x), w, h, fill=False, color='red')
        plt.gca().add_patch(rect)
    plt.show()

def plot_reconstructions(model, data):
    """Plot original vs reconstructed images"""
    pass

def plot_training_curves(losses):
    """Plot training/validation losses"""
    pass
