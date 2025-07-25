"""Basic configuration for the cLVM with cVAE setup."""
from ddprism.rand_manifolds.scaling_n_models import config_base
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base.get_config()

    # VAE parameters
    config.hidden_layers = 3
    config.hidden_feats_per_layer = 256
    config.latent_dim_tg = 5
    config.latent_dim_bkg = 5

    # Optimization hyperparameters
    config.lr_init_val = 1e-3
    config.lr_end_val = 1e-6
    config.epochs = 10
    config.batch_size = 1024
    config.lr_schedule = 'cosine'

    config.log_figure = True
    config.log_figure_bkg = True

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'cvae_test_pcpca_rand_manifolds', 'mode': 'online', 'run_name': None}
    )

    return config