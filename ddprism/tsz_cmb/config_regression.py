"""Configuration for regression-based SZ denoising."""
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # Random seed for reproducibility
    config.rng_key = 0

    # Data parameters
    config.n_train = 61_440  # Number of training samples
    config.n_val = 1024  # Number of validation samples
    config.map_norm = 2000.0
    config.data_max = 1.0

    # Model architecture parameters
    config.emb_features = 512
    config.n_blocks = 8
    config.dropout_rate_block = [0.1] * config.n_blocks
    config.heads = 8
    config.patch_size_list = [64 ** 2, 32 ** 2, 16 ** 2, 8 ** 2]
    config.n_average_layers = 4

    # Training parameters
    config.epochs = 128
    config.batch_size = 128
    config.lr_init_val = 1e-3
    config.grad_clip_norm = 1.0
    config.optimizer = ConfigDict({
        'type': 'adam',
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.0,
        'eps': 1e-8
    })
    config.lr_schedule = ConfigDict({
        'type': 'cosine',  # 'cosine', 'exponential', 'constant'
        'warmup_steps': 0,
        'min_lr_ratio': 0.0
    })
    config.ema_decay = 0.995  # EMA decay rate

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'mvss-sz-regression', 'mode': 'online', 'run_name': None}
    )

    return config
