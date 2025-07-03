"""Utility functions for training."""

from flax.core import FrozenDict
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from ddprism import diffusion
from ddprism import embedding_models


def get_time_sampling_fn(config=None):
    """Get the time sampling function based on config. Default to beta dist."""
    if config is None:
        return lambda rng, shape: jax.random.beta(rng, a=3, b=3, shape=shape)

    time_config = config.get(
        'time_sampling', {'distribution': 'beta', 'beta_a': 3.0, 'beta_b': 3.0}
    )

    if time_config['distribution'] == 'beta':
        return lambda rng, shape: jax.random.beta(
            rng, a=time_config['beta_a'], b=time_config['beta_b'], shape=shape
        )
    elif time_config['distribution'] == 'uniform':
        return lambda rng, shape: jax.random.uniform(rng, shape=shape)
    else:
        raise ValueError(
            f"Unknown time distribution: {time_config['distribution']}"
        )


def apply_model(state, x, rng, config=None, pmap=False):
    """Computes gradients and loss for a single batch."""

    # Get time sampling function and otherwise use default beta distribution.
    sample_time = get_time_sampling_fn(config)

    # Diffusion loss
    def loss_fn(params):
        # Draw random values of noise and time
        rng_z, rng_t, rng_drop = jax.random.split(rng, 3)
        z = jax.random.normal(rng_z, shape=x.shape)
        t = sample_time(rng_t, shape=x.shape[:1])

        # Evolve x samples forward in time to get noisy realizations
        sigma_t = state.apply_fn({'params': params}, t, method='sde_sigma')
        lmbda_t = 1 / sigma_t**2 + 1
        x_t = state.apply_fn({'params': params}, x, z, t, method='sde_x_t')

        # Compute expected denoised values
        x_expected = state.apply_fn(
            {'params': params}, x_t, t, rngs={'dropout': rng_drop}
        )

        # Compute error
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
    optimizer_config = config.get('optimizer', {'type': 'adam'})

    if optimizer_config['type'] == 'adam':
        return lambda lr: optax.adam(
            learning_rate=lr,
            b1=optimizer_config.get('beta1', 0.9),
            b2=optimizer_config.get('beta2', 0.999),
            eps=optimizer_config.get('eps', 1e-8)
        )
    elif optimizer_config['type'] == 'adamw':
        return lambda lr: optax.adamw(
            learning_rate=lr,
            b1=optimizer_config.get('beta1', 0.9),
            b2=optimizer_config.get('beta2', 0.999),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(
            f'Optimizer {optimizer_config["type"]} has not been implemented.'
        )


def get_learning_rate_schedule(config, base_lr, total_steps):
    """Get learning rate schedule based on config."""
    # Default to cosine schedule.
    lr_config = config.get('lr_schedule', {'type': 'cosine'})

    if lr_config['type'] == 'cosine':
        warmup_steps = lr_config.get('warmup_steps', 0)
        min_lr_ratio = lr_config.get('min_lr_ratio', 0.1)
        warmup_fn = optax.linear_schedule(
            init_value=0.0, end_value=base_lr, transition_steps=warmup_steps
        )
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_lr, decay_steps=total_steps - warmup_steps,
            alpha=min_lr_ratio
        )
        return optax.join_schedules(
            schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps]
        )
    elif lr_config['type'] == 'exponential':
        decay_rate = lr_config.get('decay_rate', 0.96)
        decay_steps = lr_config.get('decay_steps', 1000)
        return optax.exponential_decay(
            init_value=base_lr, transition_steps=decay_steps,
            decay_rate=decay_rate
        )
    elif lr_config['type'] == 'constant':
        return lambda step: base_lr
    else:
        raise ValueError(
            f"Unknown learning rate schedule: {lr_config['type']}"
        )


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
    """Create the Gaussian Denoiser."""
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

    optimizer = get_optimizer(config)(learning_rate_fn)
    grad_clip_norm = config.get('grad_clip_norm', 1.0)
    tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)

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

    optimizer = get_optimizer(config)(learning_rate_fn)
    grad_clip_norm = config.get('grad_clip_norm', 1.0)
    tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)

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

    optimizer = get_optimizer(config)(learning_rate_fn)
    grad_clip_norm = config.get('grad_clip_norm', 1.0)
    tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)

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
        updated_params = jax.tree_util.tree_map(
            lambda ema, new: decay * ema + (1. - decay) * new,
            self.params,
            new_params
        )
        return EMA(params=updated_params)
