"""Basic configuration for the variable mixing matrix problem."""
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # RNG key
    config.rng_key = 0

    # Parameters for dataset generation
    config.dataset_size = 4096
    config.data_max = 1e1
    config.arcsinh_scaling = 1.0
    config.data_norm = 1e1

    # Parameters for the Denoisers.
    config.sde = ConfigDict({'a': 1e-2, 'b': 1e2})
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

    # Training parameters.
    config.lr_init_val = 1e-5
    config.epochs = 4096
    config.em_laps = 32
    config.gaussian_em_laps = 4
    config.batch_size = 32
    config.ema_decay = 0.999

    # Sampling arguments
    config.sampling_kwargs = ConfigDict(
        {'steps': 64, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-3}
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {'steps': 16, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-3}
    )
    config.sample_batch_size = 16

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'mvss-cosmos', 'mode': 'online', 'run_name': None}
    )

    return config
