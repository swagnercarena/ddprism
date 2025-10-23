"""Basic configuration for the variable mixing matrix problem."""
import config_randoms

def get_config():
    """Get the default hyperparameter configuration."""
    config = config_randoms.get_config()

    # Posterior parameters.
    config.post_rtol = 1e-3
    config.post_maxiter = 1
    config.post_use_dplr = True
    config.post_safe_divide = 1e-6
    config.post_regularization = 1e-6
    config.post_error_threshold = 0.05

    # Sampling arguments
    config.sampling_kwargs = ConfigDict(
        {
            'steps': 256, 'sampler': 'pc', 'corrections': 1, 'tau': 1e-2,
            'clip_method': 'none'
        }
    )

    # wandb parameters.
    config.wandb_kwargs = ConfigDict(
        {'project': 'mvss-sz', 'mode': 'online', 'run_name': None}
    )

    return config
