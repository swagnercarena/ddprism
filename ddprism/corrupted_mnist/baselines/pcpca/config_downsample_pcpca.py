"""Basic configuration for PCPCA with gradient descent."""
from ml_collections import ConfigDict
from ddprism.corrupted_mnist import config_downsample_grass
from ddprism.corrupted_mnist import config_downsample_mnist


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # Configuration files for grass and MNIST digits
    config.config_grass = config_downsample_grass.get_config()
    config.config_mnist = config_downsample_mnist.get_config()

    # PCPCA parameters
    config.gamma = 0.39
    config.latent_dim = 5

    # Wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'pcpca-mnist-down', 'mode': 'online', 'run_name': None}
    )

    return config
