"""Basic configuration for the cLVM with cVAE setup."""
from ddprism.corrupted_mnist.baselines.clvm import config_base_clvm

def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base_clvm.get_config()

    # Set to best VAE parameters from search.
    config.model_type = "vae"
    config.batch_size = 256
    config.lr_init_val = 2e-5
    config.latent_dim_z = 507
    config.latent_dim_t = 8

    # wandb parameters
    config.wandb_kwargs['project'] = 'cvlm_vae_mnist_cont'

    return config
