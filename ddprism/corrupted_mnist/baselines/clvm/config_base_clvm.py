"""Basic configuration for the cLVM with cVAE setup."""
from ddprism.corrupted_mnist import config_base_grass, config_base_mnist
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # Configuration files for grass and MNIST digits
    config.config_grass = config_base_grass.get_config()
    config.config_mnist = config_base_mnist.get_config()

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
    config.lr_schedule = ConfigDict({
        'type': 'cosine',
        'warmup_steps': 0,
        'min_lr_ratio': 0.0
    })
    config.epochs = 16
    config.steps_per_epoch = 1024
    config.batch_size = 1920

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'cvlm_linear_mnist_cont', 'mode': 'online', 'run_name': None}
    )

    return config
