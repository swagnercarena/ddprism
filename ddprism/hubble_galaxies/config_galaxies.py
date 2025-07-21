"""Basic configuration for the variable mixing matrix problem."""
import config_randoms


def get_config():
    """Get the default hyperparameter configuration."""
    config = config_randoms.get_config()

    # Smaller dataset batch to give the model a fighting chance to sample.
    config.dataset_size = 2048

    # Posterior parameters.
    config.post_safe_divide = 1e-2
    config.post_regularization = 1e-2
    config.post_error_threshold = 1e1

    # Sampling arguments
    config.sampling_kwargs = ConfigDict(
        {
            'steps': 64, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'clip_method': 'none'
        }
    )
    config.gaussian_sampling_kwargs = ConfigDict(
        {
            'steps': 64, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-1,
            'clip_method': 'none'
        }
    )

    return config
