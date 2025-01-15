from setuptools import setup, find_packages

setup(
    name="autoencoder-experiments",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "tqdm",
        "matplotlib",
    ],
)
