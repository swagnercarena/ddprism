"""Basic configuration for the variable mixing matrix problem."""
from ddprism.corrupted_mnist import config_base_grass


def get_config():
    """Get the default hyperparameter configuration."""
    config = config_base_grass.get_config()

    # RNG key
    config.rng_key = 3
    config.rng_key_val = 4

    # Parameters for dataset generation
    config.dataset_size = 13_824
    config.mnist_amp = 0.5
    config.gaussian_em_laps = 32

    config.sampling_kwargs.steps = 256

    # Model type: "mlp" or "unet"
    config.model_type = 'mlp'

    # Parameters for the Denoisers.
    config.hidden_features = (256, 256, 256)
    config.time_mlp_normalize = True
    config.time_conditioning = 'film'
    config.dropout_rate = 0.1
    config.emb_features = 128

    return config
