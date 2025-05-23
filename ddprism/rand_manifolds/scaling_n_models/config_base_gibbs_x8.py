"""Basic configuration for the contrastive problem with Gibbs sampling."""
import config_base_gibbs


def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base_gibbs.get_config()
    config.sampling_kwargs = ConfigDict(
        {
            'steps': 2048, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'gibbs_rounds': 64
        }
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {
            'steps': 2048, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'gibbs_rounds': 64
        }
    )
    return config