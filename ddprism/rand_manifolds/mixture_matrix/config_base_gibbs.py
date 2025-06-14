"""Basic configuration for the variable mixing matrix problem."""
from ml_collections import ConfigDict
import config_base


def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base.get_config()

    # Training parameters.
    config.gaussian_em_laps = 16

    # Sampling arguments. Default is to have the number of model evaluations
    # per sampling step be the same for gibbs and joint (therefore fewer
    # steps for gibbs).
    config.sampling_kwargs = ConfigDict(
        {
            'steps': 32, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'gibbs_rounds': 512
        }
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {
            'steps': 32, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'gibbs_rounds': 512
        }
    )
    config.sampling_strategy = 'gibbs'

    return config
