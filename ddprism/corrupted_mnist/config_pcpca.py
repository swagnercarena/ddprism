"""Basic configuration for PCPCA with gradient descent."""
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()
    
    config.latent_dim = 2
    config.gamma = 0.385
    
    # Optimization hyperparameters
    config.n_iter = 100
    config.learning_rate = 1e-3
    config.lr_schedule = 'linear'

    return config