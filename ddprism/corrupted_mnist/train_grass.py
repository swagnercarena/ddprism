"Train a diffusion model on the grass dataset."
import functools
import os

from absl import app, flags
from einops import rearrange
from flax import jax_utils
from flax.training import orbax_utils, train_state
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint import PyTreeCheckpointer
import optax
from tqdm import tqdm
import wandb

from ddprism import diffusion
from ddprism import training_utils
from ddprism import utils

import datasets
from ddprism.metrics import image_metrics as metrics

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
flags.DEFINE_string('imagenet_path', None, 'path to imagenet grass dataset.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)


def apply_model_with_config(config):
    """Create apply_model function with config."""
    return jax.pmap( # pylint: disable=invalid-name
        functools.partial(training_utils.apply_model, config=config, pmap=True),
        axis_name='batch'
    )


update_model = jax.pmap( # pylint: disable=invalid-name
    training_utils.update_model, axis_name='batch'
)


def create_posterior_train_state(
    rng, config, image_shape, mu_x=None, cov_x=None, gaussian=False
):
    "Create joint posterior denoiser."
    # Learning rate is irrelevant for the posterior denoiser because we don't
    # optimize its parameters directly.
    learning_rate_fn = training_utils.get_learning_rate_schedule(
        config, config.lr_init_val, config.epochs
    )

    if gaussian:
        denoiser_models = [
            training_utils.create_denoiser_gaussian(config)
            for _ in range(1)
        ]
    else:
        denoiser_models = [
            training_utils.create_denoiser_unet(config, image_shape)
            for _ in range(1)
        ]

    # Joint Denoiser
    feat_dim = image_shape[0] * image_shape[1] * image_shape[2]
    posterior_denoiser = diffusion.PosteriorDenoiserJoint(
        denoiser_models=denoiser_models, y_features=feat_dim,
        rtol=config.post_rtol, maxiter=config.post_maxiter,
        use_dplr=config.post_use_dplr
    )

    # Initialize posterior denoiser.
    params = posterior_denoiser.init(
        rng, jnp.ones((1, feat_dim)), jnp.ones((1,))
    )
    if mu_x is not None:
        params['params']['denoiser_models_0']['mu_x'] = mu_x
    if cov_x is not None:
        params['params']['denoiser_models_0']['cov_x'] = cov_x

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
    imagenet_path = FLAGS.imagenet_path
    workdir = FLAGS.workdir
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
    checkpoint_manager = CheckpointManager(
        os.path.join(workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )

    # Read our full dataset to cpu.
    rng_dataset, rng_comp, rng = jax.random.split(rng, 3)
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        # Get our observations, mixing matrix, and covariance.
        grass_obs, A_mat, cov_y, _ = datasets.get_dataset(
            rng_dataset, 1.0, 0.0, config.sigma_y, config.downsampling_ratios,
            config.sample_batch_size, imagenet_path, config.dataset_size
        )

        # Drop the MNIST mixing matrix for this problem.
        A_mat = A_mat[:, :1]

        # Save the observations, A matrix, and covariance.
        np.save(os.path.join(workdir, 'grass_obs.npy'), grass_obs)
        np.save(os.path.join(workdir, 'A_mat.npy'), A_mat)

        # Reshape our observations to have a batch dimension for the
        # sample_batch_size and a pmap dimension.
        grass_obs = rearrange(
            grass_obs, '(K M N) H W C -> K M N (H W C)',
            N=config.sample_batch_size, M=jax.device_count()
        )

        # Get pure samples for Gaussian initialization and metric calculations.
        grass_pure, _ = datasets.get_corrupted_mnist(
            rng_comp, 1.0, 0.0, imagenet_path, config.dataset_size
        )
        image_shape = grass_pure.shape[1:]

        # Pull the pure sample and save to disk.
        grass_pure_ident, _ = datasets.get_corrupted_mnist(
            rng_dataset, 1.0, 0.0, imagenet_path, config.dataset_size
        )
        np.save(os.path.join(workdir, 'grass_pure_ident.npy'), grass_pure_ident)

    # Prepare the A matrix and covariance for sampling.
    A_mat = jax_utils.replicate(A_mat)
    cov_y = jax_utils.replicate(cov_y)

    # Initialize our Gaussian state with rough mean and covariance of grass
    # dataset and use for initial posterior sample.
    rng_state, rng = jax.random.split(rng)
    post_state_gauss = create_posterior_train_state(
        rng_state, config, image_shape, gaussian=True
    )
    post_state_params = post_state_gauss.params

    # Prepare post_state_gauss for pmap.
    post_state_gauss = jax_utils.replicate(post_state_gauss)

    # Create our sampling function. We want to pmap it, but we also have to
    # Create our sampling function. We want to pmap it, but we also have to
    # batch to avoid memory issues. Start with the pmapped call to sample.
    def sample(
        batch, rng, state_local, params_local, A_local, cov_local, image_shape,
        sample_batch_size, sampling_kwargs
    ):
        return utils.sample(
            rng, state_local,
            {
                'params': params_local,
                'variables': {'y': batch, 'A': A_local, 'cov_y': cov_local}
            },
            sample_shape=(sample_batch_size,),
            feature_shape=image_shape[0] * image_shape[1] * image_shape[2],
            **sampling_kwargs
        )
    sample_pmap = jax.pmap(
        functools.partial(
            sample, image_shape=image_shape,
            sample_batch_size=config.sample_batch_size,
            sampling_kwargs=config.gaussian_sampling_kwargs
        ),
        axis_name='batch'
    )

    print('Initial EM Gaussian fit.')
    for lap in tqdm(range(config.gaussian_em_laps), desc='EM Lap'):
        rng_samp, rng = jax.random.split(rng)
        rng_samp = jax.jax.random.split(
            rng_samp, (grass_obs.shape[0], jax.device_count())
        )
        # Loop over the sampling batches, saving the outputs to cpu to avoid
        # memory issues.
        x_post = []
        params = jax_utils.replicate(post_state_params)

        for grass_batch, rng_pmap in zip(grass_obs, rng_samp):
            x_post.append(
                jax.device_put(
                    sample_pmap(
                        grass_batch, rng_pmap, post_state_gauss, params,
                        A_mat, cov_y
                    ),
                    jax.local_devices(backend="cpu")[0]
                )
            )
        # No longer need sampling dimensions for training the state.
        x_post = rearrange(
            jnp.stack(x_post, axis=0), 'K M N ... -> (K M N) ...'
        )
        # Ge the statistics of the seperate grass sample.
        rng_ppca, rng = jax.random.split(rng)
        grass_mean, grass_cov = utils.ppca(rng_ppca, x_post, rank=2)
        post_state_params['denoiser_models_0']['mean_x'] = grass_mean
        post_state_params['denoiser_models_0']['cov_x'] = grass_cov

    # Save our initial samples.
    ckpt = {'x_post': jax.device_get(x_post), 'config': config.to_dict()}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(0, ckpt, save_kwargs={'save_args': save_args})
    checkpoint_manager.wait_until_finished()

    # Log our initial pq mass chi^2.
    wandb.log(
        {
            'pq_chi2': metrics.pq_mass(
                x_post[:config.pq_mass_samples],
                grass_pure[:config.pq_mass_samples]
            ),
            'pq_chi2_ident': metrics.pq_mass(
                x_post[:config.pq_mass_samples],
                grass_pure_ident[:config.pq_mass_samples]
            )
        },
        step=0
    )

    # Initialize our state and posterior state.
    rng_state, rng = jax.random.split(rng, 2)
    learning_rate_fn = training_utils.get_learning_rate_schedule(
        config, config.lr_init_val, config.epochs
    )
    state_unet = training_utils.create_train_state_unet(
        rng_state, config, learning_rate_fn, image_shape
    )
    state_unet = jax_utils.replicate(state_unet)
    post_state_unet = create_posterior_train_state(
        rng_state, config, image_shape,
    )
    post_state_unet = jax_utils.replicate(post_state_unet)
    ema = training_utils.EMA(jax_utils.unreplicate(state_unet).params)

    # Create the apply_model function with config
    apply_model = apply_model_with_config(config)

    # Change the sampling parameters to those for the diffusion model.
    sample_pmap = jax.pmap(
        functools.partial(
            sample, image_shape=image_shape,
            sample_batch_size=config.sample_batch_size,
            sampling_kwargs=config.sampling_kwargs
        ),
        axis_name='batch'
    )

    print('Beginning EM laps for diffusion model fitting.')
    for lap in tqdm(range(config.em_laps), desc='EM Lap'):
        # Training laps between samples.
        pbar = tqdm(range(config.epochs), desc='Epoch', leave=False)
        for epoch in pbar:
            # Grab a batch of the data for this training step.
            rng_data, rng_apply, rng = jax.random.split(rng, 3)
            batch_i = jax.random.randint(
                rng_data, shape=(jax.local_device_count(), config.batch_size),
                minval=0, maxval=len(x_post)
            )

            rng_apply = jax.random.split(rng_apply, jax.local_device_count())
            grads, loss = apply_model( # pylint: disable=not-callable
                state_unet, x_post[batch_i], rng_apply
            )
            state_unet = update_model( # pylint: disable=not-callable
                state_unet, grads
            )
            ema = ema.update(
                jax_utils.unreplicate(state_unet).params,
                config.ema_decay ** (
                    config.em_laps * config.epochs /
                    (lap * config.epochs + epoch + 1)
                )
            )
            wandb.log(
                {'loss_state': jax_utils.unreplicate(loss)},
                step=(lap * config.epochs + epoch)
            )

        # Generate new posterior samples with our model.
        rng_samp, rng = jax.random.split(rng)
        rng_samp = jax.jax.random.split(
            rng_samp, (grass_obs.shape[0], jax.device_count())
        )
        x_post = []
        params = jax_utils.replicate({'denoiser_models_0': ema.params})

        for grass_batch, rng_pmap in zip(grass_obs, rng_samp):
            x_post.append(
                jax.device_put(
                    sample_pmap(
                        grass_batch, rng_pmap, post_state_unet, params, A_mat,
                        cov_y
                    ),
                    jax.local_devices(backend="cpu")[0]
                )
            )
        # No longer need sampling dimensions for training the state.
        x_post = rearrange(
            jnp.stack(x_post, axis=0), 'K M N ... -> (K M N) ...'
        )

        # Calculate the pq mass chi^2 for our posterior sample.
        wandb.log(
            {
                'pq_chi2': metrics.pq_mass(
                    x_post[:config.pq_mass_samples],
                    grass_pure[:config.pq_mass_samples]
                ),
                'pq_chi2_ident': metrics.pq_mass(
                    x_post[:config.pq_mass_samples],
                    grass_pure_ident[:config.pq_mass_samples]
                )
            },
            step=(lap * config.epochs + epoch)
        )

        # Save the state, ema, and some samples.
        ckpt = {
            'state': jax.device_get(jax_utils.unreplicate(state_unet)),
            'x_post': jax.device_get(x_post),
            'ema_params': jax.device_get(ema.params), 'config': config.to_dict()
        }
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            lap + 1, ckpt, save_kwargs={'save_args': save_args}
        )

        # Initialize our next state with the current parameters.
        state_unet = training_utils.create_train_state_unet(
            rng_state, config, learning_rate_fn, image_shape,
            params={'params': ema.params}
        )
        state_unet = jax_utils.replicate(state_unet)


if __name__ == '__main__':
    app.run(main)
