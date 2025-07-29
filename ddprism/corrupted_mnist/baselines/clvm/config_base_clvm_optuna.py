"""Configuration file for optimizing over CLVM hyperparameters."""
import config_base_clvm
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()
    config.clvm_config = config_base_clvm.get_config()

    # Parameters for CLVM.
    config.n_trials = 100

    # Parameters for CLVM trials.
    config.latent_dim_min = 1
    config.latent_dim_max = 20

    # Optimization hyperparameters
    config.lr_min = 1e-5
    config.lr_max = 1e-3

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {
            'project': 'clvm_optuna_linear_mnist_cont', 'mode': 'online',
            'run_name': None
        }
    )

    return config
