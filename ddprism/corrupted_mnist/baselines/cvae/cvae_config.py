"""Basic configuration for the variable mixing matrix problem."""
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # RNG key
    config.rng_key = 0

    config.encoder = 'mlp'
    config.decoder = 'mlp'

    config.latent_features = 2
    
    config.learning_rate = 1e-3
    config.beta = 1.
    config.epochs = 100
    config.batch_size = 128

    config.run_name=None
    

    return config
