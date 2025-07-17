"""Configuration file for optimizing over PCPCA hyperparameters."""
import config_base_pcpca
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()
    config.pcpca_config = config_base_pcpca.get_config()

    # Parameters for PCPCA
    config.n_trials = 100

    # Parameters for PCPCA
    config.gamma_min = 1e-2
    config.gamma_max = 0.4
    config.latent_dim_min = 2
    config.latent_dim_max = 20

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {
            'project': 'pcpca-mnist-optuna', 'mode': 'online',
            'run_name': None
        }
    )
    
    return config
