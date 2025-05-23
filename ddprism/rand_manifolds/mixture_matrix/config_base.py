"""Basic configuration for the variable mixing matrix problem."""
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # RNG key
    config.rng_key = 0

    # Parameters for dataset generation
    config.n_sources = 2
    config.sample_size = 65536
    config.feat_dim = 5
    config.obs_dim = 3
    config.alpha_list = (3.0, 4.0)
    config.mix_frac = 0.0
    config.sigma_y = 0.01
    config.source_mix_varies = False

    # Parameters for the Denoisers.
    config.sde = ConfigDict({'a': 1e-3, 'b': 1e1})
    config.hidden_features = (256, 256, 256)
    config.time_mlp_normalize = True
    config.emb_features = 64

    # Posterior parameters.
    config.post_rtol = 1e-3
    config.post_maxiter = 1
    config.post_use_dplr = True
    config.post_safe_divide = 1e-4

    # Training parameters.
    config.lr_init_val = 1e-3
    config.lr_end_val = 1e-6
    config.epochs = 65536
    config.batch_size = 1024
    config.gaussian_em_laps = 8192
    config.diffusion_em_laps = 128

    # Sampling arguments
    config.sampling_kwargs = ConfigDict(
        {'steps': 16384, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1}
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {'steps': 32, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1}
    )
    config.sinkhorn_samples = 16384
    config.sampling_mask = True
    config.sampling_strategy = 'joint'
    config.gaussian_dplr_rank = 2

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'mvss-dif', 'mode': 'online', 'run_name': None}
    )

    return config
