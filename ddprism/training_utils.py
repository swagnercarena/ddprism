"""Utility functions for training."""

from flax.core import FrozenDict
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from ddprism import diffusion
from ddprism import embedding_models


def apply_model(state, x, rng, pmap=False):
    """Computes gradients and loss for a single batch."""

    # diffusion loss
    def loss_fn(params):
        # draw random values of noise and time
        rng_z, rng_t, rng_drop = jax.random.split(rng, 3)
        z = jax.random.normal(rng_z, shape=x.shape)
        t = jax.random.beta(rng_t, a=3, b=3, shape=x.shape[:1])

        # evolve x samples forward in time to get noisy realizations
        sigma_t = state.apply_fn({'params': params}, t, method='sde_sigma')
        lmbda_t = 1 / sigma_t**2 + 1
        x_t = state.apply_fn({'params': params}, x, z, t, method='sde_x_t')

        # compute expected denoised values
        x_expected = state.apply_fn(
            {'params': params}, x_t, t, rngs={'dropout': rng_drop}
        )

        # compute error
        error = x_expected - x
        loss = jnp.mean(lmbda_t * jnp.mean(error**2, axis=-1))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # If being used in pmap, make sure you apply pmean.
    if pmap:
        loss = jax.lax.pmean(loss, axis_name='batch')
        grads = jax.lax.pmean(grads, axis_name='batch')

    return grads, loss


def update_model(state, grads):
    """Update model with gradients."""
    return state.apply_gradients(grads=grads)


def get_optimizer(config):
    """Get the optimizer specified by the config."""
    optimizer_name = config.get('optimizer', 'adam')

    if optimizer_name == 'adam':
        return optax.adam

    raise ValueError(f'Optimizer {optimizer_name} has not been implemented.')


def create_denoiser_timemlp(config):
    """Create the TimeMLP Denoiser."""
    # SDE
    sde = diffusion.VESDE(config.sde.a, config.sde.b)

    # TimeMLP
    time_mlp = embedding_models.TimeMLP(
        features=config.feat_dim, hid_features=config.hidden_features,
        normalize=config.time_mlp_normalize
    )

    # Denoiser
    denoiser = diffusion.Denoiser(
        sde, time_mlp, emb_features=config.emb_features
    )

    return denoiser


def create_denoiser_gaussian(config):
    """Create the TimeMLP Denoiser."""
    # Denoiser
    denoiser = diffusion.GaussianDenoiserDPLR(
        diffusion.VESDE(config.sde.a, config.sde.b)
    )

    return denoiser


def create_denoiser_unet(config, image_shape):
    """Create the FlatUNet Denoiser."""
    # SDE
    sde = diffusion.VESDE(config.sde['a'], config.sde['b'])

    # TimeMLP
    unet = embedding_models.FlatUNet(
        hid_channels=config.hid_channels, hid_blocks=config.hid_blocks,
        kernel_size=config.kernel_size, emb_features=config.emb_features,
        heads=config.heads, dropout_rate=config.dropout_rate,
        image_shape=image_shape
    )

    return diffusion.Denoiser(sde, unet, emb_features=config.emb_features)


def create_train_state_timemlp(rng, config, learning_rate_fn, params=None):
    """Creates initial TrainState for timemlp"""
    # Denoiser
    denoiser = create_denoiser_timemlp(config)

    # Initialize new parameters if the old parameters are not passed in.
    if params is None:
        params = denoiser.init(
            rng, jnp.ones((1, config.feat_dim)) , jnp.ones((1,))
        )

    adam = get_optimizer(config)(learning_rate_fn)
    tx = optax.chain(optax.clip_by_global_norm(1.0), adam)

    return train_state.TrainState.create(
        apply_fn=denoiser.apply, params=params['params'], tx=tx
    )


def create_train_state_gaussian(rng, config, learning_rate_fn, params=None):
    """Creates initial TrainState for GaussianDPLR"""
    # Denoiser
    denoiser = create_denoiser_gaussian(config)

    # Initialize new parameters if the old parameters are not passed in.
    if params is None:
        params = denoiser.init(
            rng, jnp.ones((1, config.feat_dim)) , jnp.ones((1,))
        )
        # Start with zero mean and wide covariance
        params['params']['mu_x'] *= 0.0
        params['params']['cov_x'] *= 4.0

    adam = get_optimizer(config)(learning_rate_fn)
    tx = optax.chain(optax.clip_by_global_norm(1.0), adam)

    return train_state.TrainState.create(
        apply_fn=denoiser.apply, params=params['params'], tx=tx
    )


def create_train_state_unet(
    rng, config, learning_rate_fn, image_shape, params=None
):
    """Creates initial TrainSate for FlatUNet."""
    # Denoiser
    denoiser = create_denoiser_unet(config, image_shape)

    if params is None:
        # Pass an example to get the parameters.
        image_features = denoiser.score_model.feat_dim
        params = denoiser.init(
            rng, jnp.ones((1, image_features)), jnp.ones((1, ))
        )


    adam = get_optimizer(config)(learning_rate_fn)
    tx = optax.chain(optax.clip_by_global_norm(1.0), adam)

    return train_state.TrainState.create(
        apply_fn=denoiser.apply, params=params['params'], tx=tx
    )


class EMA:
    """Exponential moving average of state parametters.

    Args:
        params: Initial parameter values.
    """
    def __init__(self, params: FrozenDict):
        self.params = params

    def update(self, new_params, decay):
        """Update EMA parameters with given decay."""
        updated_params = jax.tree_map(
            lambda ema, new: decay * ema + (1. - decay) * new,
            self.params,
            new_params
        )
        return EMA(params=updated_params)
