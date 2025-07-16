"""Basic configuration for the variable mixing matrix problem."""
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # RNG key
    config.rng_key = 0

    # Parameters for dataset generation
    config.dataset_size = 4096
    config.data_max = 2.0
    config.arcsinh_scaling = 0.1
    config.data_norm = 0.2

    # Parameters for the Denoisers.
    config.sde = ConfigDict({'a': 1e-2, 'b': 1e1}) # Remaining std is 5e-2.
    config.hid_channels = (64, 128, 256, 256, 512)
    config.hid_blocks = (2, 2, 2, 2, 2)
    config.kernel_size = (3, 3)
    config.emb_features = 64
    config.heads = {'2': 4, '3': 8, '4': 16}
    config.dropout_rate = 0.1

    # Posterior parameters.
    config.post_rtol = 1e-6
    config.post_maxiter = 1
    config.post_use_dplr = True
    config.post_safe_divide = 1e-32
    config.post_regularization = 0.0

    # Training parameters.
    config.lr_init_val = 1e-5
    config.epochs = 4096
    config.em_laps = 32
    config.gaussian_em_laps = 4
    config.batch_size = 32
    config.ema_decay = 0.999
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
            'steps': 64, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-2,
            'clip_method': 'value', 'clip_adaptive': True,
            'clip_value': config.data_max, 'clip_early_strength': 0.5,
            'clip_late_strength': 1.0
        }
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {
            'steps': 64, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-2,
            'clip_method': 'value', 'clip_adaptive': True,
            'clip_value': config.data_max, 'clip_early_strength': 0.5,
            'clip_late_strength': 1.0
        }
    )
    config.sample_batch_size = 16

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'mvss-cosmos', 'mode': 'online', 'run_name': None}
    )
    config.eval_samples = 128

    return config
