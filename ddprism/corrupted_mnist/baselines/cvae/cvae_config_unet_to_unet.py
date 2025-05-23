"""Basic configuration for the variable mixing matrix problem."""
import cvae_config


def get_config():
    """Get the default hyperparameter configuration."""
    config = cvae_config.get_config()

    config.encoder = 'unet'
    config.decoder = 'unet'
    config.epochs = 1000
    
    config.learning_rate = 1e-5
    
    return config