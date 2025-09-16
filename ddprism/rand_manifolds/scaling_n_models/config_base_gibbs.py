"""Basic configuration for the contrastive problem with Gibbs sampling."""
import config_base
from ml_collections import ConfigDict


def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base.get_config()
    config.diffusion_em_laps = [16, 96, 192]

    # Sampling arguments. Default is to have the number of model evaluations
    # per sampling step be the same for gibbs and joint (therefore fewer
    # steps for gibbs).
    config.sampling_strategy = 'gibbs'
    config.sampling_kwargs = ConfigDict(
        {
            'steps': 256, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'gibbs_rounds': 64, 'clip_method': 'none'
        }
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {
            'steps': 256, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'gibbs_rounds': 64, 'clip_method': 'none'
        }
    )

    return config
