"Train a diffusion model on the random dataset."
import functools
import gc
import os

import numpy as np

from absl import app, flags
from einops import rearrange
from flax import jax_utils
from flax.training import orbax_utils, train_state
import jax
import jax.numpy as jnp
from ml_collections import config_flags, ConfigDict
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint import PyTreeCheckpointer
import optax
from tqdm import tqdm
import wandb

from ddprism import diffusion, training_utils, utils
from ddprism.tsz_cmb import load_datasets, training_utils_healpix

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
flags.DEFINE_string('sz_path', None, 'path to sz dataset.')
flags.DEFINE_string(
    'sz_no_noise_path', None, 'path to sz dataset without noise.'
)
flags.DEFINE_string(
    'randoms_workdir', None,
    'working directory from which randoms model will be loaded.'
)
flags.DEFINE_integer(
    'randoms_lap', -1,
    'Checkpoint number corresponding to which lap of the randoms model to load.'
)
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)


def match_filered_rmse(x_post, sz_no_noise):
    """Match filter the x_post and compute the rmse."""
    filters = sz_no_noise / (
        jnp.sqrt(jnp.sum(jnp.square(sz_no_noise), axis=-1, keepdims=True))
    )
    x_post_matched = jnp.sum(x_post * filters, axis=-1, keepdims=True) * filters
    rmse_matched = jnp.sqrt(jnp.mean(jnp.square(x_post_matched - sz_no_noise)))
    return rmse_matched


def compute_metrics_for_samples(x_post, sz_no_noise):
    """Compute metrics for randoms samples."""
    rmse = jnp.sqrt(jnp.mean(jnp.square(x_post - sz_no_noise)))
    return {
        'rmse_sz': rmse,
        'rmse_matched': match_filered_rmse(x_post, sz_no_noise)
    }


def create_posterior_train_state(
    rng, config, config_randoms, healpix_shapes, feat_dim,  mu_x=None,
    cov_x=None, gaussian=False
):
    "Create joint posterior denoiser."
    # Learning rate is irrelevant for the posterior denoiser because we don't
    # optimize its parameters directly.
    learning_rate_fn = training_utils.get_learning_rate_schedule(
        config, config.lr_init_val, config.epochs
    )

    denoiser_models = [
        training_utils_healpix.create_denoiser_transformer(
            config_randoms, hp_shape
        )
        for hp_shape in healpix_shapes[:-1]
    ]
    if gaussian:
        denoiser_models.append(
            training_utils.create_denoiser_gaussian(config)
        )
    else:
        denoiser_models.append(
            training_utils_healpix.create_denoiser_transformer(
                config, healpix_shapes[-1]
            )
        )

    # Joint Denoiser
    x_features = [
        hp_shape[0] * hp_shape[1] for hp_shape in healpix_shapes
    ]
    posterior_denoiser = diffusion.PosteriorDenoiserJointDiagonal(
        denoiser_models=denoiser_models, y_features=feat_dim,
        rtol=config.post_rtol, maxiter=config.post_maxiter,
        use_dplr=config.post_use_dplr,
        safe_divide=config.get('post_safe_divide', 1e-32),
        regularization=config.get('post_regularization', 0.0),
        error_threshold=config.get('post_error_threshold', None),
    )


    # Initialize posterior denoiser.
    total_x_dim = sum(x_features)
    params = posterior_denoiser.init(
        rng, jnp.ones((1, total_x_dim)), jnp.ones((1,))
    )
    if mu_x is not None:
        model_i = len(healpix_shapes) - 1
        params['params'][f'denoiser_models_{model_i}']['mu_x'] = mu_x
    if cov_x is not None:
        model_i = len(healpix_shapes) - 1
        params['params'][f'denoiser_models_{model_i}']['cov_x'] = cov_x

    # Use the new configurable optimizer
    optimizer = training_utils.get_optimizer(config)(learning_rate_fn)
    grad_clip_norm = config.get('grad_clip_norm', 1.0)
    tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)

    return train_state.TrainState.create(
        apply_fn=posterior_denoiser.apply, params=params['params'], tx=tx
    )


def main(_):
    """Train a joint posterior denoiser."""
    config = FLAGS.config
    workdir = FLAGS.workdir
    sz_path = FLAGS.sz_path
    sz_no_noise_path = FLAGS.sz_no_noise_path
    randoms_workdir = FLAGS.randoms_workdir
    randoms_lap = FLAGS.randoms_lap
    rng = jax.random.PRNGKey(config.rng_key)
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.local_devices()}')
    print(f'Working directory: {workdir}')

    # Set up wandb logging and checkpointing.
    wandb.init(
        config=config.copy_and_resolve_references(),
        project=config.wandb_kwargs.get('project', None),
        name=config.wandb_kwargs.get('run_name', None),
        mode=config.wandb_kwargs.get('mode', 'disabled')
    )
    checkpointer = PyTreeCheckpointer()
    checkpoint_options = CheckpointManagerOptions(
        enable_async_checkpointing=False
    )

    # Load the relevant information for the randoms model.
    checkpoint_manager = CheckpointManager(
        os.path.join(randoms_workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )
    if randoms_lap == -1:
        randoms_lap = checkpoint_manager.latest_step()
    print(
        f'Loading randoms model from {randoms_workdir} at '
        f'checkpoint {randoms_lap}.'
    )
    restore = checkpoint_manager.restore(randoms_lap)
    randoms_params = restore['ema_params']
    config_randoms = ConfigDict(restore['config'])
    checkpoint_manager.close()

    # Delete checkpoint restore dict to free memory
    del restore
    gc.collect()

    # Switch to the working directory for the sz model.
    checkpoint_manager = CheckpointManager(
        os.path.join(workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )

    # Get our observations, mixing matrix, and covariance, with the
    # later only having batch size of the sampling_batch_size.
    sz_obs, vec_map, A_mat, cov_y = load_datasets.load_sz(
        config, sz_path
    )
    vec_map_flat = rearrange(vec_map, 'B P S N V -> (B P S) N V')

    sz_no_noise, _, _, _ = load_datasets.load_sz(
        config, sz_no_noise_path
    )
    sz_no_noise = rearrange(sz_no_noise, 'B P S (NC) -> (B P S) (NC)')

    # Delete old copy of sz_no_noise to free memory
    gc.collect()

    # Set dimension for posterior sampling.
    # TODO: Hardcoded!
    healpix_shapes = [(sz_obs.shape[-1] // 3, 3)] * 2
    feat_dim = sz_obs.shape[-1]
    total_x_dim = sum(
        [hp_shape[0] * hp_shape[1] for hp_shape in healpix_shapes]
    )

    # Initialize our Gaussian state. Override mu_x with zeros.
    rng_state, rng = jax.random.split(rng)
    post_state_transformer = create_posterior_train_state(
        rng_state, config, config_randoms, healpix_shapes, feat_dim,
        mu_x=jnp.zeros(feat_dim),
        gaussian=True,
    )
    # Store params (will be modified in loop)
    post_state_params = post_state_transformer.params

    # Prepare post_state_gauss for pmap.
    post_state_transformer = jax_utils.replicate(post_state_transformer)

    # Create our sampling function. We want to pmap it, but we also have to
    # batch to avoid memory issues. Start with the pmapped call to sample.
    def sample(
        batch, rng, state_local, params_local, A_local, cov_local,
        vec_map_local, total_x_features, sample_batch_size, sampling_kwargs
    ):
        vec_map_dict = {
            f'denoiser_models_{i}': {'vec_map': vec_map_local}
            for i in range(A_local.shape[-2])
        }
        return utils.sample(
            rng, state_local,
            {
                'params': params_local,
                'variables': (
                    {'y': batch, 'cov_y': cov_local, 'A': A_local} | vec_map_dict
                )
            },
            sample_shape=(sample_batch_size,),
            feature_shape=total_x_features,
            **sampling_kwargs
        )

    sample_pmap = jax.pmap(
        functools.partial(
            sample, total_x_features=total_x_dim,
            sample_batch_size=config.sample_batch_size,
            sampling_kwargs=config.gaussian_sampling_kwargs
        ),
        axis_name='batch'
    )

    print('Initial EM Gaussian fit.')
    for lap in tqdm(range(config.gaussian_em_laps), desc='EM Lap'):
        rng_samp, rng = jax.random.split(rng)
        rng_samp = jax.random.split(
            rng_samp, (sz_obs.shape[0], jax.device_count())
        )
        # Loop over the sampling batches, saving the outputs to cpu to avoid
        # memory issues.
        x_post = []
        params = {
            'denoiser_models_0': randoms_params,
            'denoiser_models_1': post_state_params['denoiser_models_1']
        }
        params = jax_utils.replicate(params)

        for sz_batch, vec_batch, rng_pmap in zip(sz_obs, vec_map, rng_samp):
            x_post.append(
                jax.device_put(
                    sample_pmap(
                        sz_batch, rng_pmap, post_state_transformer,
                        params, A_mat, cov_y, vec_batch
                    ),
                    jax.local_devices(backend="cpu")[0]
                )
            )

        # Stack and process on CPU device
        x_post = rearrange(
            jnp.stack(x_post, axis=0), 'K M N ... -> (K M N) ...'
        )
        x_post = jnp.clip(x_post, -config.data_max, config.data_max)
        x_post = jnp.split(x_post, 2, axis=-1)

        # Get the statistics of the separate grass sample.
        rng_ppca, rng = jax.random.split(rng)
        sz_mean, sz_cov = utils.ppca(rng_ppca, x_post[1], rank=2)
        post_state_params['denoiser_models_1']['mu_x'] = sz_mean
        post_state_params['denoiser_models_1']['cov_x'] = sz_cov

    metrics_dict = compute_metrics_for_samples(x_post[1], sz_no_noise)
    wandb.log(metrics_dict, commit=False)

    # Save our initial samples.
    ckpt = {'x_post': x_post, 'config': config.to_dict()}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(0, ckpt, save_kwargs={'save_args': save_args})
    checkpoint_manager.wait_until_finished()

    # Initialize our state and posterior state.
    rng_state, rng = jax.random.split(rng, 2)
    learning_rate_fn = training_utils.get_learning_rate_schedule(
        config, config.lr_init_val, config.epochs
    )
    state_transformer = training_utils_healpix.create_train_state_transformer(
        rng_state, config, learning_rate_fn, healpix_shapes[1]
    )
    state_transformer = jax_utils.replicate(state_transformer)

    post_state_transformer = create_posterior_train_state(
        rng_state, config, config_randoms, healpix_shapes, feat_dim,
    )
    post_state_transformer = jax_utils.replicate(post_state_transformer)
    # EMA params live on-device, replicated, so the per-step update fuses
    # with the gradient/optimizer step inside `train_step`.
    ema_params = jax.tree_util.tree_map(jnp.copy, state_transformer.params)

    train_step_pmap = jax.pmap(
        functools.partial(
            training_utils_healpix.train_step, config=config, pmap=True,
        ),
        axis_name='batch', donate_argnums=(0, 1),
    )

    # Change the sampling parameters to those for the diffusion model.
    sample_pmap = jax.pmap(
        functools.partial(
            sample, total_x_features=total_x_dim,
            sample_batch_size=config.sample_batch_size,
            sampling_kwargs=config.sampling_kwargs
        ),
        axis_name='batch'
    )

    print('Beginning EM laps for diffusion model fitting.')
    x_post_np = np.array(x_post[1])
    vec_map_np = np.array(vec_map_flat)
    for lap in tqdm(range(config.em_laps), desc='EM Lap'):
        # Training laps between samples.
        pbar = tqdm(range(config.epochs), desc='Epoch', leave=False)
        for epoch in pbar:
            # Grab a batch of the data for this training step.
            rng_data, rng_apply, rng = jax.random.split(rng, 3)
            batch_i = jax.random.randint(
                rng_data, shape=(jax.local_device_count(), config.batch_size),
                minval=0, maxval=len(x_post[1])
            )

            rng_apply = jax.random.split(rng_apply, jax.local_device_count())
            batch_i_np = np.array(batch_i)
            batch_x = jax.device_put(x_post_np[batch_i_np])
            batch_vec = jax.device_put(vec_map_np[batch_i_np])
            decay = jax_utils.replicate(jnp.float32(
                config.ema_decay ** (
                    config.em_laps * config.epochs /
                    (lap * config.epochs + epoch + 1)
                )
            ))
            # pylint: disable=not-callable
            state_transformer, ema_params, loss = train_step_pmap(
                state_transformer, ema_params, batch_x, batch_vec, rng_apply,
                decay=decay,
            )
            wandb.log(
                {'loss_state_sz': jax_utils.unreplicate(loss)}
            )

        # Generate new posterior samples with our model.
        rng_samp, rng = jax.random.split(rng)
        rng_samp = jax.random.split(
            rng_samp, (sz_obs.shape[0], jax.device_count())
        )
        x_post = []
        post_state_params = {
            'denoiser_models_0': jax_utils.replicate(randoms_params),
            'denoiser_models_1': ema_params,
        }

        for sz_batch, vec_batch, rng_pmap in zip(
            sz_obs, vec_map, rng_samp
        ):
            x_post.append(
                jax.device_put(
                    sample_pmap(
                        sz_batch, rng_pmap, post_state_transformer,
                        post_state_params, A_mat, cov_y, vec_batch
                    ),
                    jax.local_devices(backend="cpu")[0]
                )
            )

        # Stack and process on CPU device
        x_post = rearrange(
            jnp.stack(x_post, axis=0), 'K M N ... -> (K M N) ...'
        )
        x_post = jnp.clip(x_post, -config.data_max, config.data_max)
        x_post = jnp.split(x_post, 2, axis=-1)

        x_post_np = np.array(x_post[1])

        # Calculate and log metrics for our posterior sample.
        metrics_dict_post = compute_metrics_for_samples(x_post[1], sz_no_noise)
        wandb.log(metrics_dict_post, commit=False)

        # Save the state, ema, and some samples.
        ckpt = {
            'state': jax.device_get(jax_utils.unreplicate(state_transformer)),
            'x_post': x_post,
            'ema_params': jax.device_get(jax_utils.unreplicate(ema_params)),
            'config': config.to_dict(),
            'metrics_post': jax.device_get(metrics_dict_post),
        }
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            lap + 1, ckpt, save_kwargs={'save_args': save_args}
        )

        # Initialize our next state with the current parameters.
        state_transformer = (
            training_utils_healpix.create_train_state_transformer(
                rng_state, config, learning_rate_fn, healpix_shapes[1],
                params={'params': jax_utils.unreplicate(ema_params)}
            )
        )
        state_transformer = jax_utils.replicate(state_transformer)
        ema_params = jax.tree_util.tree_map(
            lambda x: x, state_transformer.params
        )


if __name__ == '__main__':
    app.run(main)
