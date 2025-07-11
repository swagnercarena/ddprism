"""Configuration file for optimizing over PCPCA hyperparameters for the contrastive problem. """
import config_base
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base.get_config()

    # Parameters for PCPCA 
    config.n_trials = 100

    # Run the PCPCA algorithm
    # Parameters for PCPCA
    config.gamma_min = 1e-2
    config.gamma_max = 1.
    config.latent_dim_min = 1
    config.latent_dim_max = 5
    
    # Optimization hyperparameters
    config.n_iter_min = 10
    config.n_iter_max = 300
    config.n_iter_step = 10

    config.lr_min = 1e-4
    config.lr_max = 5e-2

    return config
