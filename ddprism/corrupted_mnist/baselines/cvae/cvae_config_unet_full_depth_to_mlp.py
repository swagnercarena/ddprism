"""Basic configuration for the variable mixing matrix problem."""
import cvae_config


def get_config():
    """Get the default hyperparameter configuration."""
    config = cvae_config.get_config()

    config.encoder = 'unet_full_depth'
    config.decoder = 'mlp'
    config.epochs = 1000

    config.learning_rate = 1e-4
    
    return config