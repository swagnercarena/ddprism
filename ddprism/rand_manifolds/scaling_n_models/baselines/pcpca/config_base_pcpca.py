"""Basic configuration for the contrastive problem with Gibbs sampling."""
from ddprism.rand_manifolds.scaling_n_models import config_base
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base.get_config()

    # Parameters for PCPCA
    config.gamma = 0.3
    config.latent_dim = 5


    # Optimization hyperparameters
    config.n_iter = 10
    config.learning_rate = 1e-3
    config.lr_schedule = 'linear'

    config.log_figure = True

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'pcpca_rand_manifolds', 'mode': 'online', 'run_name': None}
    )

    return config
