"Train a diffusion model for contrastive MVSS with n_sources."
import os
import functools

from absl import app, flags
from flax.training import orbax_utils, train_state
import jax
import jax.numpy as jnp
from ml_collections import config_flags, ConfigDict
import numpy as np
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointer
import optax
from tqdm import tqdm
import wandb

from ddprism import diffusion
from ddprism import training_utils
from ddprism import utils
from ddprism import plotting_utils
from ddprism.metrics import metrics
from ddprism.rand_manifolds.random_manifolds import MAX_SPREAD

import load_dataset

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)


def update_config_with_sweep(config):
    """Update config with wandb sweep parameters if available."""
    if hasattr(wandb, 'config') and wandb.config:
        print(f"Updating config with sweep parameters: {dict(wandb.config)}")
        # Create a mutable copy of the config
        config_dict = config.to_dict()

        # Update with sweep parameters
        for key, value in wandb.config.items():
            if '.' in key:
                # Handle nested keys. Will fail if the key is not in the config.
                parts = key.split('.')
                config_dict[parts[0]][parts[1]] = value
            else:
                config_dict[key] = value

        # Convert back to ConfigDict
        return ConfigDict(config_dict)

    return config


def create_train_state_list(
    rng, config, n_sources, gaussian_indices=None, previous_list=None
):
    """Create train state list with gaussian indices"""
    learning_rate_fn = optax.linear_schedule(
        init_value=config.lr_init_val, end_value=config.lr_end_val,
        transition_steps=config.epochs
    )

    if gaussian_indices is None:
        gaussian_indices = []
    if previous_list is None:
        previous_list = []

    state_list = []
    for i in range(n_sources):
        rng_state, rng = jax.random.split(rng)
        # Use pre-loaded params if available.
        params = None
        if i < len(previous_list):
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


def create_posterior_train_state_joint(
    rng, config, n_sources, gaussian_indices=None
):
    "Create joint posterior denoiser."
    learning_rate_fn = optax.linear_schedule(
        init_value=config.lr_init_val, end_value=config.lr_end_val,
        transition_steps=config.epochs
    )
    if gaussian_indices is None:
        gaussian_indices = []

    denoiser_models = []
    for i in range(n_sources):
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
        rng, jnp.ones((1, n_sources * config.feat_dim)),
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
        'sample_shape', 'feature_shape', 'steps', 'sampler', 'corrections',
        'clip_method'
    ],
)


sample_gibbs = jax.jit( # pylint: disable=invalid-name
    utils.sample_gibbs,
    static_argnames=[
        'steps', 'sampler', 'corrections', 'gibbs_rounds', 'clip_method'
    ],
)


def apply_model_with_config(config):
    """Create apply_model function with config."""
    return jax.jit(functools.partial(training_utils.apply_model, config=config))

apply_model = jax.jit(training_utils.apply_model) # pylint: disable=invalid-name


update_model = jax.jit(training_utils.update_model) # pylint: disable=invalid-name


def _sample_wrapper_joint(
    rng, post_state, state_list, variables, config, n_sources, gaussian
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

    # For the Gibbs sampling case:
    # When training the model for the 1st source, use equivalent number of
    # sampling steps.
    if 'gibbs_rounds' in sampling_kwargs:
        sampling_kwargs = sampling_kwargs.to_dict().copy()
        sampling_kwargs['steps'] = (
            sampling_kwargs['steps'] * sampling_kwargs['gibbs_rounds']
        )


    # Sample given the current posterior.
    x_post = sample( # pylint: disable=not-callable
        rng, post_state, {'params': params,'variables': variables},
        sample_shape = variables['y'].shape[:-1],
        feature_shape = config.feat_dim * n_sources,
        **sampling_kwargs
    )

    # Mask out large values if requested.
    if config.get('sampling_mask', True):
        x_post = x_post[
            jnp.all(jnp.logical_and(x_post > -4, x_post < 4), axis=-1)
        ]

    # Split on different sources.
    x_post = jnp.split(x_post, n_sources, axis=-1)

    return x_post


def _sample_wrapper_gibbs(
    rng, x_prev, post_state, state_list, variables, config, n_sources, gaussian
):
    """Wrapper for sampling operation."""
    params = {
        f'denoiser_models_{i}': state_list[i].params
        for i in range(len(state_list))
    }

    # Reshape for sampling.
    x_prev = jnp.concat(x_prev, axis=-1)

    sampling_kwargs = config.sampling_kwargs
    # If we're sampling with Gaussian distributions, check for the Gaussian
    # sampling kwargs.
    if gaussian:
        sampling_kwargs = config.get(
            'gaussian_sampling_kwargs', config.sampling_kwargs
        )

    # Sample given the current posterior.
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
    x_post = jnp.split(x_post, n_sources, axis=-1)

    return x_post


def _sample_wrapper(
    rng, x_post, post_state, state_list, variables, config, n_sources,
    gaussian=False
):
    # When fitting only one source, use MMPS sampling strategy.
    if n_sources == 1:
        return _sample_wrapper_joint(
                rng, post_state, state_list, variables, config, n_sources,
                gaussian
            )
    else:
        sampling_strategy = config.get('sampling_strategy', 'joint')
        if sampling_strategy == 'gibbs':
            return _sample_wrapper_gibbs(
                rng, x_post, post_state, state_list, variables, config,
                n_sources, gaussian
            )
        elif sampling_strategy == 'joint':
            return _sample_wrapper_joint(
                rng, post_state, state_list, variables, config, n_sources,
                gaussian
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

    # Set up wandb logging and checkpointing. Use sweep if SWEEP_ID is set.
    if os.environ.get('WANDB_SWEEP_ID') is not None:
        # Don't pass config to wandb.init() if running as part of a sweep
        # to avoid overwriting sweep parameters
        print('Running as part of a sweep.')
        wandb.init()
        config = update_config_with_sweep(config)
    else:
        # Normal run - pass our config to wandb
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
    x_all, A_all, y_all, cov_y_all = load_dataset.get_dataset(rng_data, config)

    # Save the dataset to disk.
    np.save(os.path.join(workdir, 'x_all.npy'), x_all)
    np.save(os.path.join(workdir, 'A_all.npy'), A_all)
    np.save(os.path.join(workdir, 'y_all.npy'), y_all)

    # initially set the state list to an empty list
    state_list = None
    step_offset = 0

    if isinstance(config.gaussian_em_laps, list):
        assert len(config.gaussian_em_laps) == config.n_sources
        gaussian_em_laps = config.gaussian_em_laps
    else:
        gaussian_em_laps = [
            config.gaussian_em_laps for _ in range(config.n_sources)
        ]

    if isinstance(config.diffusion_em_laps, list):
        assert len(config.diffusion_em_laps) == config.n_sources
        diffusion_em_laps = config.diffusion_em_laps
    else:
        diffusion_em_laps = [
            config.diffusion_em_laps for _ in range(config.n_sources)
        ]

    for source_index in range(config.n_sources):
        n_sources = source_index + 1
        variables = {
            'y': y_all[:, source_index], 'A': A_all[:, :n_sources],
            'cov_y': cov_y_all[source_index]
        }

        # Start by fitting a set of Gaussian Denoisers to n-th source distribution.
        rng_state, rng_post, rng = jax.random.split(rng, 3)
        gaussian_indices = [source_index]
        state_list = create_train_state_list(
            rng_state, config, n_sources, gaussian_indices,
            previous_list=state_list
        )
        post_state = create_posterior_train_state_joint(
            rng_post, config, n_sources, gaussian_indices
        )

        # Generate the initial samples.
        if config.get('sampling_strategy', 'joint') == 'gibbs':
            x_post = jnp.zeros(
                (n_sources, y_all.shape[0], config.feat_dim)
            )
        else:
            x_post = None

        print(
            f'Fitting source {source_index+1} with a Gaussian denoiser ' +
            'to observations.'
        )
        for _ in tqdm(
            range(gaussian_em_laps[source_index]), desc='Gaussian denoiser EM'
        ):
            # Set up rng and sample
            rng_sample, rng = jax.random.split(rng)
            x_post = _sample_wrapper(
                rng_sample, x_post, post_state, state_list, variables, config,
                n_sources, gaussian=True
            )

            # Fit a new Gaussian for the n-th source distribution.
            rng_ppca, rng = jax.random.split(rng)
            # Full rank covariance since we're operating in low dimensions.
            mu_x, cov_x = utils.ppca(
                rng_ppca, x_post[source_index],
                rank=config.get('gaussian_dplr_rank', 2)
            )
            state_list[source_index] = state_list[source_index].replace(
                params={'mu_x': mu_x, 'cov_x': cov_x}
            )

        # Final sample with best Gaussian fit.
        rng_sample, rng = jax.random.split(rng)
        x_post = _sample_wrapper(
            rng_sample, x_post, post_state, state_list, variables, config,
            n_sources,
            gaussian=True
        )

        # Log a figure with the initial posterior samples.
        if config.log_figure:
            fig = plotting_utils.show_corner(jnp.concat(x_post, axis=1))._figure
            wandb.log(
                    {'posterior samples': wandb.Image(fig)}, commit=False
                )

        # Log the initial divergence, pqmass, and psnr.
        metrics_dict = {}
        for i, x_single in enumerate(x_post):
            divergence = metrics.sinkhorn_divergence(
                x_single[:config.sinkhorn_samples],
                x_all[:config.sinkhorn_samples, i]
            )
            metrics_dict[f'divergence_x_{i}'] = divergence
            pqmass = metrics.pq_mass(
                x_single[:config.pqmass_samples],
                x_all[:config.pqmass_samples, i]
            )
            metrics_dict[f'pqmass_x_{i}'] = pqmass
            psnr = metrics.psnr(
                x_single[:config.psnr_samples],
                x_all[:config.psnr_samples, i],
                max_spread=MAX_SPREAD
            )
            metrics_dict[f'psnr_x_{i}'] = psnr
        wandb.log(metrics_dict, commit=False)

        # Save the state list and samples.
        ckpt = {
            'state_list': state_list, 'x_post': x_post,
            'metrics': metrics_dict, 'config': config.to_dict()
        }
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            step_offset, ckpt, save_kwargs={'save_args': save_args}
        )

        # Create our non-Gaussian state list and posterior state.
        rng_state, rng_post, rng = jax.random.split(rng, 3)
        # The state list contains all previously trained diffusion models.
        state_list = create_train_state_list(
            rng_state, config, n_sources, previous_list=state_list[:-1]
        )
        post_state = create_posterior_train_state_joint(
            rng_post, config, n_sources
        )

        print(f'Beginning EM laps for diffusion model {n_sources} fitting.')
        for step in tqdm(
            range(1, diffusion_em_laps[source_index] + 1), desc='Diffusion EM'
        ):

            # Training laps between samples.
            pbar = tqdm(
                range(1, config.epochs + 1), desc='Train epoch', leave=False
            )
            for _ in pbar:
                rng_epoch, rng = jax.random.split(rng, 2)
                batch_i = jax.random.randint(
                    rng_epoch, shape=(config.batch_size,), minval=0,
                    maxval=len(x_post[source_index])
                )

                grads, loss = apply_model( # pylint: disable=not-callable
                    state_list[source_index], x_post[source_index][batch_i],
                    rng_epoch
                )
                state_list[source_index] = update_model( # pylint: disable=not-callable
                    state_list[source_index], grads
                )
                wandb.log({f'loss_state_{source_index}': loss})


            # Sample the new posterior.
            rng_sample, rng = jax.random.split(rng)
            x_post = _sample_wrapper(
                rng_sample, x_post, post_state, state_list, variables, config,
                n_sources
            )

            # Log a figure with new posterior samples.
            if config.log_figure:
                fig = plotting_utils.show_corner(
                    jnp.concat(x_post, axis=1)
                )._figure
                wandb.log(
                    {'posterior samples': wandb.Image(fig)}, commit=False
                )

            # Log the divergence, pqmass, and psnr.
            metrics_dict = {}
            for i, x_single in enumerate(x_post):
                divergence = metrics.sinkhorn_divergence(
                    x_single[:config.sinkhorn_samples],
                    x_all[:config.sinkhorn_samples, i]
                )
                metrics_dict[f'divergence_x_{i}'] = divergence
                pqmass = metrics.pq_mass(
                    x_single[:config.pqmass_samples],
                    x_all[:config.pqmass_samples, i]
                )
                metrics_dict[f'pqmass_x_{i}'] = pqmass
                psnr = metrics.psnr(
                    x_single[:config.psnr_samples],
                    x_all[:config.psnr_samples, i],
                    max_spread=MAX_SPREAD
                )
                metrics_dict[f'psnr_x_{i}'] = psnr
            wandb.log(
                metrics_dict, commit=False
            )

            # Save the state list and samples.
            ckpt = {
                'state_list': state_list, 'x_post': x_post,
                'metrics': metrics_dict, 'config': config.to_dict()
            }
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(
                step + step_offset, ckpt, save_kwargs={'save_args': save_args}
            )

            # Make new states.
            rng_state, rng = jax.random.split(rng)
            state_list = create_train_state_list(
                rng_state, config, n_sources, previous_list=state_list
            )

        # Need to offset the steps for wandb and checkpoint logging.
        step_offset += diffusion_em_laps[source_index] + 1

    if 'divergence_x_2' in metrics_dict:
        wandb.log({'final_divergence_x_2': metrics_dict['divergence_x_2']})
    else:
        print('No divergence_x_2 in metrics_dict')

    wandb.finish()


if __name__ == '__main__':
    # Run a sweep if WANDB_SWEEP_ID is set.
    if os.environ.get('WANDB_SWEEP_ID') is not None:
        wandb.agent(os.environ.get('WANDB_SWEEP_ID'), main, count=1)
    else:
        app.run(main)
