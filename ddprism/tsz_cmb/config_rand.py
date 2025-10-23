"""Basic configuration for the variable mixing matrix problem."""
from ml_collections import ConfigDict

def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    config.rng_key = 0

    # Parameters for dataset generation
    config.n_train = 32_768
    config.map_norm = 2000.0
    config.data_max = 1.0

    # Parameters for the Denoisers.
    config.sde = ConfigDict({'a': 1e-4, 'b': 1e2})
    config.emb_features = 512
    config.n_blocks = 8
    config.dropout_rate_block = [0.1] * config.n_blocks
    config.heads = 8
    config.patch_size_list = [64 ** 2, 32 ** 2, 16 ** 2, 8 ** 2]
    config.time_emb_features = 128
    config.n_average_layers = 4

    # Posterior parameters.
    config.post_rtol = 1e-3
    config.post_maxiter = 1
    config.post_use_dplr = True
    config.post_safe_divide = 1e-6
    config.post_regularization = 1e-6
    config.post_error_threshold = 0.05

    # Training parameters.
    config.lr_init_val = 1e-3
    config.epochs = 16_384
    config.use_dynamic = False
    config.em_laps = 64
    config.gaussian_em_laps = 4
    config.batch_size = 128
    config.ema_decay = 0.995
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
    config.time_sampling = ConfigDict({
        'distribution': 'beta',  # 'beta', 'uniform', 'log_normal'
        'beta_a': 3.0,
        'beta_b': 3.0
    })

    # Sampling arguments
    config.sampling_kwargs = ConfigDict(
        {
            'steps': 256, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-2,
            'clip_method': 'none'
        }
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {
            'steps': 16, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-2,
            'clip_method': 'none'
        }
    )
    config.sample_batch_size = 64

    # wandb parameters.
    config.wandb_kwargs = ConfigDict(
        {'project': 'mvss-sz', 'mode': 'online', 'run_name': None}
    )

    return config
