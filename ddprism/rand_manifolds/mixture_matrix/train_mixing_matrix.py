"Train a diffusion model with variable mixing matrix and n_models."
import os
import functools

from absl import app, flags
from flax.training import orbax_utils, train_state
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointer
import optax
from tqdm import tqdm
import wandb

from ddprism import diffusion
from ddprism import linalg
from ddprism import training_utils
from ddprism import utils
from ddprism.rand_manifolds import random_manifolds

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)

def get_dataset(rng, config):
    """Return the dataset specified in the config."""

    # Rng per source.
    rng_x, rng_a, rng_y = jax.random.split(rng, 3)

    # One view per source.
    n_views = config.n_sources

    # Set up alpha for each x value.
    if 'alpha_list' in config:
        assert len(config.alpha_list) == config.n_sources
        alpha_list = config.alpha_list
    else:
        alpha_list = [config.alpha] * config.n_sources

    # Generate our sources.
    rng_x = jax.random.split(rng_x, config.n_sources)
    x_all = [
        random_manifolds.generate_x(
            rng, n_views * config.sample_size, man_dim=1,
            feat_dim=config.feat_dim, alpha=alpha, normalize=True
        ) for rng, alpha in zip(rng_x, alpha_list)
    ]
    x_all = jnp.stack(x_all, axis=1)

    # Generate our A matrices.
    if config.source_mix_varies:
        A_all = [
            random_manifolds.generate_A(
                rng, n_views * config.sample_size, obs_dim=config.obs_dim,
                feat_dim=config.feat_dim
            ) for rng in jax.random.split(rng_a, config.n_sources)
        ]
        A_all = jnp.stack(A_all, axis=1).reshape(
            (-1, n_views, config.n_sources, config.obs_dim, config.feat_dim)
        )
    else:
        A_all = random_manifolds.generate_A(
            rng_a, config.sample_size, obs_dim=config.obs_dim,
            feat_dim=config.feat_dim
        )[:, None, None]
        A_all = jnp.tile(A_all, (1, n_views, config.n_sources, 1, 1))

    # Scale the A matrices depending on the view.
    mixing_scaling = (
        config.mix_frac * jnp.ones((n_views, config.n_sources)) +
        (1 - config.mix_frac) * jnp.eye(n_views)
    )[None, :, :, None, None]
    A_all *= mixing_scaling
    A_all = jnp.reshape(
        A_all, (-1, config.n_sources, config.obs_dim, config.feat_dim)
    )

    # Generate our observations
    y, cov_y = random_manifolds.generate_y(
        rng_y, A_all, x_all, noise=config.sigma_y
    )

    # Put the covariance in the DPLR representation.
    cov_y = linalg.DPLR(diagonal=jnp.tile(cov_y[None], (y.shape[0], 1)))

    return x_all, A_all, y, cov_y

def create_train_state_list(
    rng, config, gaussian_indices=None, previous_list = None
):
    """Create train state list with gaussian indices"""
    learning_rate_fn = optax.linear_schedule(
        init_value=config.lr_init_val, end_value=config.lr_end_val,
        transition_steps=config.epochs
    )

    if gaussian_indices is None:
        gaussian_indices = []

    state_list = []
    for i in range(config.n_sources):
        # Use pre-loaded params if available.
        rng_state, rng = jax.random.split(rng)
        params = None
        if previous_list is not None:
            params = {'params': previous_list[i].params}

        if i in gaussian_indices:
            state_list.append(
                training_utils.create_train_state_gaussian(
                    rng_state, config, learning_rate_fn, params=params
                )
            )
        else:
            state_list.append(
                training_utils.create_train_state_timemlp(
                    rng_state, config, learning_rate_fn, params=params
                )
            )

    return state_list


def create_posterior_train_state_joint(rng, config, gaussian_indices=None):
    "Create joint posterior denoiser."
    learning_rate_fn = optax.linear_schedule(
        init_value=config.lr_init_val, end_value=config.lr_end_val,
        transition_steps=config.epochs
    )
    if gaussian_indices is None:
        gaussian_indices = []

    denoiser_models = []
    for i in range(config.n_sources):
        if i in gaussian_indices:
            denoiser_models.append(
                training_utils.create_denoiser_gaussian(config)
            )
        else:
            denoiser_models.append(
                training_utils.create_denoiser_timemlp(config)
            )

    # Joint Denoiser
    posterior_denoiser = diffusion.PosteriorDenoiserJoint(
        denoiser_models=denoiser_models, y_features=config.obs_dim,
        rtol=config.post_rtol, maxiter=config.post_maxiter,
        use_dplr=config.post_use_dplr, safe_divide=config.post_safe_divide
    )

    # Initialize posterior denoiser.
    params = posterior_denoiser.init(
        rng, jnp.ones((1, (config.n_sources) * config.feat_dim)),
        jnp.ones((1,))
    )

    tx = optax.adam(learning_rate_fn)

    return train_state.TrainState.create(
        apply_fn=posterior_denoiser.apply, params=params['params'], tx=tx
    )


# Create jitted functions for this script.
sample = jax.jit( # pylint: disable=invalid-name
    utils.sample,
    static_argnames=[
        'sample_shape', 'feature_shape', 'steps','sampler','corrections'
    ],
)


sample_gibbs = jax.jit( # pylint: disable=invalid-name
    utils.sample_gibbs,
    static_argnames=['steps','sampler','corrections','gibbs_rounds'],
)


def apply_model_with_config(config):
    """Create apply_model function with config."""
    return jax.jit(functools.partial(training_utils.apply_model, config=config))

apply_model = jax.jit(training_utils.apply_model) # pylint: disable=invalid-name


update_model = jax.jit(training_utils.update_model) # pylint: disable=invalid-name


def _sample_wrapper_joint(
    rng, post_state, state_list, variables, config, gaussian
):
    """Wrapper for sampling operation."""
    params = {
        f'denoiser_models_{i}': state_list[i].params
        for i in range(len(state_list))
    }

    sampling_kwargs = config.sampling_kwargs
    # If we're sampling with Gaussian distributions, check for the Gaussian
    # sampling kwargs.
    if gaussian:
        sampling_kwargs = config.get(
            'gaussian_sampling_kwargs', config.sampling_kwargs
        )

    # Sample given the current posterior
    x_post = sample( # pylint: disable=not-callable
        rng, post_state, {'params': params,'variables': variables},
        sample_shape = variables['y'].shape[:-1],
        feature_shape = config.feat_dim * config.n_sources,
        **sampling_kwargs
    )

    # Mask out large values if requested.
    if config.get('sampling_mask', True):
        x_post = x_post[
            jnp.all(jnp.logical_and(x_post > -4, x_post < 4), axis=-1)
        ]

    # Split on different sources.
    x_post = jnp.split(x_post, config.n_sources, axis=-1)

    return x_post


def _sample_wrapper_gibbs(
    rng, x_prev, post_state, state_list, variables, config, gaussian
):
    """Wrapper for sampling operation."""
    params = {
        f'denoiser_models_{i}': state_list[i].params
        for i in range(len(state_list))
    }

    # Reverse the jnp.split call.
    x_prev = jnp.concat(x_prev, axis=-1)

    sampling_kwargs = config.sampling_kwargs
    # If we're sampling with Gaussian distributions, check for the Gaussian
    # sampling kwargs.
    if gaussian:
        sampling_kwargs = config.get(
            'gaussian_sampling_kwargs', config.sampling_kwargs
        )

    # Sample given the current posterior
    x_post = sample_gibbs( # pylint: disable=not-callable
        rng, post_state, {'params': params, 'variables': variables},
        x_prev, **sampling_kwargs
    )

    # Mask out large values if requested. Must still have values for all
    # samples in gibbs, so set to previous value.
    if config.get('sampling_mask', True):
        mask_region = jnp.any(jnp.logical_or(x_post < -4, x_post > 4), axis=-1)
        x_post = x_post.at[mask_region].set(x_prev[mask_region])

    # Split on different sources.
    x_post = jnp.split(x_post, config.n_sources, axis=-1)

    return x_post


def _sample_wrapper(
    rng, x_post, post_state, state_list, variables, config, gaussian=False
):
    sampling_strategy = config.get('sampling_strategy', 'joint')
    if sampling_strategy == 'gibbs':
        return _sample_wrapper_gibbs(
            rng, x_post, post_state, state_list, variables, config,
            gaussian
        )
    elif sampling_strategy == 'joint':
        return _sample_wrapper_joint(
            rng, post_state, state_list, variables, config, gaussian
        )
    else:
        raise ValueError(f'Invalid sampling strategy {sampling_strategy}.')


def main(_):
    """Train a joint posterior denoiser."""
    config = FLAGS.config
    workdir = FLAGS.workdir
    rng = jax.random.PRNGKey(config.rng_key)
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    # Set up wandb logging and checkpointing.
    wandb.init(
        config=config.copy_and_resolve_references(),
        project=config.wandb_kwargs.get('project', None),
        name=config.wandb_kwargs.get('run_name', None),
        mode=config.wandb_kwargs.get('mode', 'disabled')
    )
    checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(
        os.path.join(workdir, 'checkpoints'), checkpointer
    )

    # Generate our dataset.
    rng, rng_data = jax.random.split(rng)
    x_all, A_all, y, cov_y = get_dataset(rng_data, config)

    # Save the dataset to disk.
    np.save(os.path.join(workdir, 'x_all.npy'), x_all)
    np.save(os.path.join(workdir, 'A_all.npy'), A_all)
    np.save(os.path.join(workdir, 'y.npy'), y)

    variables = {'y': y, 'A': A_all, 'cov_y': cov_y}

    # Start by fitting a set of Gaussian Denoisers.
    gaussian_indices = list(jnp.arange(config.n_sources))
    rng_state, rng_post, rng = jax.random.split(rng, 3)
    state_list = create_train_state_list(rng_state, config, gaussian_indices)
    post_state = create_posterior_train_state_joint(
        rng_post, config, gaussian_indices
    )

    # Generate the initial samples.
    if config.get('sampling_strategy', 'joint') == 'gibbs':
        x_post = jnp.zeros((config.n_sources, y.shape[0], config.feat_dim))
    else:
        x_post = None

    print(f'Fitting {config.n_sources} Gaussian denoisers to observations.')
    for _ in tqdm(range(config.gaussian_em_laps), desc='Gaussian denoiser EM'):
        # Set up rng and sample
        rng_sample, rng = jax.random.split(rng)
        x_post = _sample_wrapper(
            rng_sample, x_post, post_state, state_list, variables, config,
            gaussian=True
        )

        # Fit a new Gaussian for each sampled distribution.
        for i, x_single in enumerate(x_post):
            rng_ppca, rng = jax.random.split(rng)
            # Full rank covariance since we're operating in low dimensions.
            mu_x, cov_x = utils.ppca(
                rng_ppca, x_single, rank=config.get('gaussian_dplr_rank', 2)
            )
            state_list[i] = state_list[i].replace(
                params={'mu_x': mu_x, 'cov_x': cov_x}
            )

    # Final sample with best Gaussian fit.
    rng_sample, rng = jax.random.split(rng)
    x_post = _sample_wrapper(
        rng_sample, x_post, post_state, state_list, variables, config,
        gaussian=True
    )

    # Record the initial divergence.
    divergence_dict = {}
    for i, x_single in enumerate(x_post):
        divergence = utils.sinkhorn_divergence(
            x_single[:config.sinkhorn_samples],
            x_all[:config.sinkhorn_samples, i]
        )
        divergence_dict[f'divergence_x_{i}'] = divergence
    wandb.log(divergence_dict, step=1)

    # Save the state list and samples.
    ckpt = {
        'state_list': state_list, 'x_post': x_post,
        'divergence': divergence_dict, 'config': config.to_dict()
    }
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(0, ckpt, save_kwargs={'save_args': save_args})

    # Create our non-Gaussian state list and posterior state.
    rng_state, rng_post, rng = jax.random.split(rng, 3)
    state_list = create_train_state_list(rng_state, config)
    post_state = create_posterior_train_state_joint(rng_post, config)

    print(f'Beginning EM laps for {config.n_sources} diffusion models fitting.')
    for step in tqdm(
        range(1, config.diffusion_em_laps + 1), desc='Diffusion EM'
    ):

        # Training laps between samples.
        pbar = tqdm(range(config.epochs), desc='Train epoch', leave=False)
        for epoch in pbar:
            rng_epoch, rng = jax.random.split(rng, 2)
            batch_i = jax.random.randint(
                rng_epoch, shape=(config.batch_size,), minval=0,
                maxval=len(x_post[0])
            )

            for i, state in enumerate(state_list):
                grads, loss = apply_model( # pylint: disable=not-callable
                    state, x_post[i][batch_i], rng_epoch
                )
                state = update_model( # pylint: disable=not-callable
                    state, grads
                )
                state_list[i] = state
                wandb.log(
                    {f'loss_state_{i}': loss},
                    step=(step * config.epochs + epoch)
                )

        # Sample the new posterior.
        rng_sample, rng = jax.random.split(rng)
        x_post = _sample_wrapper(
            rng_sample, x_post, post_state, state_list, variables, config
        )

        # Log the divergence.
        divergence_dict = {}
        for i, x_single in enumerate(x_post):
            divergence = utils.sinkhorn_divergence(
                x_single[:config.sinkhorn_samples],
                x_all[:config.sinkhorn_samples, i]
            )
            divergence_dict[f'divergence_x_{i}'] = divergence
        wandb.log(divergence_dict, step=(step + 1) * config.epochs)

        # Save the state list and samples.
        ckpt = {
            'state_list': state_list, 'x_post': x_post,
            'divergence': divergence_dict, 'config': config.to_dict()
        }
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            step, ckpt, save_kwargs={'save_args': save_args}
        )

        # Make new states.
        rng_state, rng = jax.random.split(rng)
        state_list = create_train_state_list(
            rng_state, config, previous_list=state_list
        )


if __name__ == '__main__':
    app.run(main)
