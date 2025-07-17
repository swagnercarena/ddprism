"""Basic configuration for the variable mixing matrix problem."""
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # RNG key
    config.rng_key = 0

    # Parameters for dataset generation
    config.dataset_size = 32768
    config.sigma_y = 0.01
    config.downsampling_ratios = {1: 1.0}

    # Parameters for the Denoisers.
    config.sde = ConfigDict({'a': 1e-4, 'b': 1e2})
    config.hid_channels = (32, 64, 128)
    config.hid_blocks = (2, 2, 2)
    config.kernel_size = (3, 3)
    config.emb_features = 64
    config.heads = {'2': 4}
    config.dropout_rate = 0.1

    # Posterior parameters.
    config.post_rtol = 1e-3
    config.post_maxiter = 1
    config.post_use_dplr = True

    # Training parameters.
    config.lr_init_val = 1e-3
    config.epochs = 4096
    config.em_laps = 192
    config.gaussian_em_laps = 32
    config.batch_size = 1920
    config.ema_decay = 0.9999
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
        {'steps': 256, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-2}
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {'steps': 16, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-2}
    )
    config.sample_batch_size = 128
    config.pq_mass_samples = 8192
    config.psnr_samples = 8192
    config.MAX_SPREAD = 1

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'mvss-mnist', 'mode': 'online', 'run_name': None}
    )

    return config
