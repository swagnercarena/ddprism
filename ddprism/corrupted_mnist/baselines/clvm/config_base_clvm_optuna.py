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
    config.latent_t_dim_min = 1
    config.latent_t_dim_max = 16
    config.latent_z_dim_min = 1
    config.latent_z_dim_max = 512

    # Optimization hyperparameters
    config.lr_min = 1e-5
    config.lr_max = 1e-3

    # VAE hyperparameters (only used when model_type == "vae")
    config.vae_hid_features_min = 32
    config.vae_hid_features_max = 512
    config.vae_normalize_choices = [True, False]
    config.vae_activation_choices = ['silu', 'tanh']

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {
            'project': 'clvm_optuna_linear_mnist_cont', 'mode': 'online',
            'run_name': None
        }
    )

    return config
