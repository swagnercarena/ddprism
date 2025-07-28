"""Basic configuration for the cLVM with cVAE setup."""
from ddprism.rand_manifolds.scaling_n_models import config_base
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base.get_config()

    # Model type: "linear" or "vae"
    config.model_type = "linear"

    # Linear model parameters
    config.latent_dim_z = 5
    config.latent_dim_t = 5

    # VAE model parameters (only used when model_type == "vae")
    config.vae = ConfigDict({
        # Hidden features for encoders/decoders
        'signal_encoder_hid_features': (256, 256, 256),
        'bkg_encoder_hid_features': (256, 256, 256),
        'signal_decoder_hid_features': (256, 256, 256),
        'bkg_decoder_hid_features': (256, 256, 256),
        'activation': 'silu',
        'normalize': True,
        'dropout_rate': 0.1,
    })

    # Training parameters.
    config.lr_init_val = 1e-4
    config.lr_end_val = 1e-6
    config.lr_schedule = 'cosine'
    config.epochs = 1024
    config.batch_size = 1024

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'cvlm_linear_rand_cont', 'mode': 'online', 'run_name': None}
    )

    return config