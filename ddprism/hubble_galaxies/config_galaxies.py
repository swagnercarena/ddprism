"""Basic configuration for the variable mixing matrix problem."""
from ml_collections import ConfigDict
import config_randoms


def get_config():
    """Get the default hyperparameter configuration."""
    config = config_randoms.get_config()

    # Smaller dataset batch to give the model a fighting chance to sample.
    config.dataset_size = 2048

    # Sampling arguments
    config.sampling_kwargs = ConfigDict(
        {'steps': 64, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-3}
    )

    return config
