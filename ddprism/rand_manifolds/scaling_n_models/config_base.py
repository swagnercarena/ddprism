"""Basic configuration for the contrastive problem."""
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ConfigDict()

    # RNG key
    config.rng_key = 0

    # Parameters for dataset generation
    config.n_sources = 3
    config.sample_size = 65536
    config.feat_dim = 5
    config.obs_dim = 3
    config.alpha = [3.0, 4.0, 5.0]
    config.phase = None
    config.sigma_y = 0.01

    # Parameters for the Denoisers.
    config.sde = ConfigDict({'a': 1e-3, 'b': 1e1})
    config.hidden_features = (256, 256, 256)
    config.time_mlp_normalize = True
    config.time_conditioning = 'concat'
    config.dropout_rate = 0.0
    config.emb_features = 64

    # Posterior parameters.
    config.post_rtol = 1e-6
    config.post_maxiter = 1
    config.post_use_dplr = True
    config.post_safe_divide = 0.0
    config.post_regularization = 0.0
    config.post_error_threshold = 0.0

    # Training parameters.
    config.lr_init_val = 1e-3
    config.lr_end_val = 1e-6
    config.epochs = 65_536
    config.batch_size = 1024
    config.gaussian_em_laps = 16
    config.diffusion_em_laps = [16, 32, 64]
    config.gaussian_dplr_rank = 2

    # Sampling arguments
    config.sampling_kwargs = ConfigDict(
        {
            'steps': 16384, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'clip_method': 'none'
        }
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {
            'steps': 16384, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'clip_method': 'none'
        }
    )

    config.sinkhorn_samples = 16384
    config.pqmass_samples = 1024
    config.psnr_samples = 16384

    config.sampling_mask = True
    config.sampling_strategy = 'joint'

    # wandb parameters
    config.wandb_kwargs = ConfigDict(
        {'project': 'rand-manifolds-cont', 'mode': 'online', 'run_name': None}
    )
    config.log_figure = False

    return config
