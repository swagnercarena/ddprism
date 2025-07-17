"""Basic configuration for PCPCA with gradient descent."""
from ml_collections import ConfigDict
from ddprism.corrupted_mnist import config_base_grass, config_base_mnist


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # Configuration files for grass and MNIST digits
    config.config_grass = config_base_grass.get_config()
    config.config_mnist = config_base_mnist.get_config()
    
    # PCPCA parameters
    config.gamma = 0.15
    config.latent_dim = 2

    # Wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'pcpca-mnist', 'mode': 'online', 'run_name': None}
    )
    
    return config