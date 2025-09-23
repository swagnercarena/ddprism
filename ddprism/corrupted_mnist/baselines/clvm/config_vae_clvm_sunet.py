"""Basic configuration for the cLVM with cVAE setup."""
from ddprism.corrupted_mnist.baselines.clvm import config_vae_clvm
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = config_vae_clvm.get_config()

    config.vae = ConfigDict({
        # UNET-specific parameters (used when architecture is 'unet')
        'hid_channels': [32], # Number of channels per level
        'hid_blocks': [2], # Number of blocks per level
        'heads': None,
    })

    return config
