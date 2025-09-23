"""Basic configuration for the cLVM with cVAE setup."""
from ddprism.corrupted_mnist.baselines.clvm import config_vae_clvm
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = config_vae_clvm.get_config()

    config.vae['hidden_channels'] = [32]
    config.vae['hidden_blocks'] = [2]
    config.vae['heads'] = None

    return config
