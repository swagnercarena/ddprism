"""Basic configuration for the cLVM with cVAE setup."""
from ddprism.corrupted_mnist import config_base_mnist
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # Configuration files for grass and MNIST digits
    config.config_mnist = config_base_mnist.get_config()

    # Model type: "linear" or "vae"
    config.model_type = "linear"

    # Linear model parameters
    config.latent_dim_z = 380
    config.latent_dim_t = 15

    # VAE model parameters (only used when model_type == "vae")
    config.vae = ConfigDict({
        # Architecture types - can be 'mlp' or 'unet' (defaults to 'mlp')
        'encoder_architecture': 'mlp',
        'decoder_architecture': 'mlp',

        # MLP-specific parameters (used when architecture is 'mlp')
        'hid_features': (70, 70, 70), # Hidden features for MLP.
        'normalize': False,                # Normalization for MLP.

        # UNET-specific parameters (used when architecture is 'unet')
        'hid_channels': [32, 64, 128], # Number of channels per level
        'hid_blocks': [2, 2, 2], # Number of blocks per level
        'heads': {'1': 2, '2': 4}, # Attention heads for levels 1 and 2

        # Common parameters
        'activation': 'silu',
        'dropout_rate': 0.1,
    })

    # Training parameters.
    config.lr_init_val = 2e-4
    config.lr_schedule = ConfigDict({
        'type': 'cosine',
        'warmup_steps': 0,
        'min_lr_ratio': 0.0
    })
    config.epochs = 16
    config.steps_per_epoch = 1024
    config.batch_size = 256
    config.sample_batch_size = 256

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'cvlm_linear_mnist_cont', 'mode': 'online', 'run_name': None}
    )

    return config
