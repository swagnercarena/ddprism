"""Basic configuration for the cLVM with cVAE setup."""
from ddprism.rand_manifolds.scaling_n_models import config_base
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base.get_config()

    # Linear model parameters
    config.latent_dim_z = 5
    config.latent_dim_t = 5

    # Training parameters.
    config.lr_init_val = 1e-4
    config.lr_end_val = 1e-6
    config.epochs = 1024
    config.batch_size = 1024

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'cvlm_linear_rand_cont', 'mode': 'online', 'run_name': None}
    )

    return config