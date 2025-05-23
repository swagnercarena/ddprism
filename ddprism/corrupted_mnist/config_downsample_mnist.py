"""Basic configuration for the variable mixing matrix problem."""
import config_base_mnist


def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base_mnist.get_config()

    config.downsampling_ratios = {1: 1.0/3.0, 2: 1.0/3.0, 4: 1.0/3.0}

    return config
