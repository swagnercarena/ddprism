"Train a diffusion model on the grass dataset."
import functools
import os

from absl import app, flags
from einops import rearrange
from flax import jax_utils
from flax.training import orbax_utils, train_state
import jax
import jax.numpy as jnp
from ml_collections import config_flags, ConfigDict
import numpy as np
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint import PyTreeCheckpointer
import optax
from tqdm import tqdm
import wandb

from ddprism import diffusion
from ddprism import linalg
from ddprism import training_utils
from ddprism import utils

import datasets
from ddprism.metrics import image_metrics as metrics

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
flags.DEFINE_string('imagenet_path', None, 'path to imagenet grass dataset.')
flags.DEFINE_string(
    'grass_workdir', None,
    'working directory from which grass diffusion model will be loaded.'
)
flags.DEFINE_integer(
    'grass_lap', -1,
    'Checkpoint number corresponding to which lap of the grass model to load.'
)
flags.DEFINE_string(
    'mnist_classifier_path', None,
    'path to MNIST classifier working directory.'
)
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
    rng, config, config_grass, image_shape, mu_x=None, cov_x=None,
    gaussian=False
):
    "Create joint posterior denoiser."
    learning_rate_fn = optax.cosine_decay_schedule(
        init_value=config.lr_init_val,
        decay_steps=config.epochs * config.em_laps
    )

    denoiser_models = [
        training_utils.create_denoiser_unet(config_grass, image_shape)
    ]
    if gaussian:
        denoiser_models.append(
            training_utils.create_denoiser_gaussian(config)
        )
    else:
        denoiser_models.append(
            training_utils.create_denoiser_unet(config, image_shape)
        )

    # Joint Denoiser
    feat_dim = image_shape[0] * image_shape[1] * image_shape[2]
    posterior_denoiser = diffusion.PosteriorDenoiserJoint(
        denoiser_models=denoiser_models, y_features=feat_dim * 2,
        rtol=config.post_rtol, maxiter=config.post_maxiter,
        use_dplr=config.post_use_dplr
    )

    # Initialize posterior denoiser.
    params = posterior_denoiser.init(
        rng, jnp.ones((1, feat_dim * 2)), jnp.ones((1,))
    )
    if mu_x is not None:
        params['params']['denoiser_models_1']['mu_x'] = mu_x
    if cov_x is not None:
        params['params']['denoiser_models_1']['cov_x'] = cov_x

    tx = optax.adam(learning_rate_fn)

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

    # Load the relevant information for the grass model.
    grass_workdir = FLAGS.grass_workdir
    checkpointer = PyTreeCheckpointer()
    checkpoint_options = CheckpointManagerOptions(
        enable_async_checkpointing=False
    )
    checkpoint_manager = CheckpointManager(
        os.path.join(grass_workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )
    grass_lap = FLAGS.grass_lap
    if grass_lap == -1:
        grass_lap = checkpoint_manager.latest_step()
    print(
        f'Loading grass model from {grass_workdir} at checkpoint {grass_lap}.'
    )
    restore = checkpoint_manager.restore(grass_lap)
    grass_params = restore['ema_params']
    config_grass = ConfigDict(restore['config'])
    checkpoint_manager.close()

    # Set up wandb logging and checkpointing.
    wandb.init(
        config=config.copy_and_resolve_references(),
        project=config.wandb_kwargs.get('project', None),
        name=config.wandb_kwargs.get('run_name', None),
        mode=config.wandb_kwargs.get('mode', 'disabled')
    )
    checkpoint_manager = CheckpointManager(
        os.path.join(workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )

    # Read our full dataset to cpu.
    rng_dataset, rng_comp, rng = jax.random.split(rng, 3)
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        # Get our observations, mixing matrix, and covariance.
        corrupt_obs, A_mat, cov_y, _ = datasets.get_dataset(
            rng_dataset, 1.0, config.mnist_amp, config.sigma_y,
            config.downsampling_ratios, config.sample_batch_size, imagenet_path,
            config.dataset_size
        )

        # Save the observations, A matrix, and covariance.
        np.save(os.path.join(workdir, 'corrupt_obs.npy'), corrupt_obs)
        np.save(os.path.join(workdir, 'A_mat.npy'), A_mat)

        # Reshape our observations to have a batch dimension for the
        # sample_batch_size and a pmap dimension.
        corrupt_obs = rearrange(
            corrupt_obs, '(K M N) H W C -> K M N (H W C)',
            N=config.sample_batch_size, M=jax.device_count()
        )

        # Get pure samples for Gaussian initialization and metric calculations.
        mnist_pure, _ = datasets.get_corrupted_mnist(
            rng_comp, 0.0, 1.0, imagenet_path, config.dataset_size
        )
        image_shape = mnist_pure.shape[1:]

        # Pull the pure sample and save to disk.
        grass_pure_ident, _ = datasets.get_corrupted_mnist(
            rng_dataset, 1.0, 0.0, imagenet_path, config.dataset_size
        )
        np.save(os.path.join(workdir, 'grass_pure_ident.npy'), grass_pure_ident)

    # Load our MNIST classifier for metrics.
    rng_class, rng = jax.random.split(rng)
    mnist_model, mnist_params = metrics.get_model(
        FLAGS.mnist_classifier_path, rng_class,
        **config.get('mnist_classifier_kwargs', {})
    )

    # Prepare the A matrix and covariance for sampling.
    A_mat = jax_utils.replicate(A_mat)
    cov_y = jax_utils.replicate(cov_y)

    # Initialize our Gaussian state and set the mean and covariance to small
    # values.
    rng_state, rng = jax.random.split(rng)
    image_size = image_shape[0] * image_shape[1] * image_shape[2]
    post_state_gauss = create_posterior_train_state(
        rng_state, config, config_grass, image_shape,
        mu_x=jnp.zeros(image_size),
        cov_x=linalg.DPLR(jnp.ones(image_size) * 0.3), gaussian=True
    )
    post_state_params = post_state_gauss.params

    # Prepare post_state_gauss for pmap.
    post_state_gauss = jax_utils.replicate(post_state_gauss)

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
            feature_shape=image_shape[0] * image_shape[1] * image_shape[2] * 2,
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
        rng_samp = jax.random.split(
            rng_samp, (corrupt_obs.shape[0], jax.device_count())
        )
        # Loop over the sampling batches, saving the outputs to cpu to avoid
        # memory issues.
        x_post = []
        params = {
            'denoiser_models_0': grass_params,
            'denoiser_models_1': post_state_params['denoiser_models_1']
        }
        params = jax_utils.replicate(params)

        for corrupt_batch, rng_pmap in zip(corrupt_obs, rng_samp):
            x_post.append(
                jax.device_put(
                    sample_pmap(
                        corrupt_batch, rng_pmap, post_state_gauss, params,
                        A_mat, cov_y
                    ),
                    jax.local_devices(backend="cpu")[0]
                )
            )

        # No longer need sampling dimensions for training the state.
        x_post = rearrange(
            jnp.stack(x_post, axis=0), 'K M N ... -> (K M N) ...'
        )
        x_post = jnp.split(x_post, 2, axis=-1)

        # Get the statistics of the seperate mnist sample.
        rng_ppca, rng = jax.random.split(rng)
        mnist_mean, mnist_cov = utils.ppca(rng_ppca, x_post[1], rank=4)
        post_state_params['denoiser_models_1']['mean_x'] = mnist_mean
        post_state_params['denoiser_models_1']['cov_x'] = mnist_cov

    # Save our initial samples.
    ckpt = {'x_post': jax.device_get(x_post), 'config': config.to_dict()}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(0, ckpt, save_kwargs={'save_args': save_args})
    checkpoint_manager.wait_until_finished()

    # Log our initial pq mass chi^2.
    wandb.log(
        {
            'pq_chi2': metrics.pq_mass(
                x_post[1][:config.pq_mass_samples],
                mnist_pure[:config.pq_mass_samples]
            ),
            'pq_chi2_ident': metrics.pq_mass(
                x_post[1][:config.pq_mass_samples],
                grass_pure_ident[:config.pq_mass_samples]
            ),
            'fcd': metrics.fcd_mnist(
                mnist_model, mnist_params,
                rearrange(
                    x_post[1][:config.pq_mass_samples],
                    '... (H W C) -> ... H W C',
                    H=image_shape[0], W=image_shape[1], C=image_shape[2]
                ),
                mnist_pure[:config.pq_mass_samples]
            ),
            'inception': metrics.inception_score_mnist(
                mnist_model, mnist_params,
                rearrange(
                    x_post[1][:config.pq_mass_samples],
                    '... (H W C) -> ... H W C',
                    H=image_shape[0], W=image_shape[1], C=image_shape[2]
                )
            )
        },
        step=0
    )

    # Initialize our state and posterior state.
    rng_state, rng = jax.random.split(rng, 2)
    learning_rate_fn = optax.cosine_decay_schedule(
        init_value=config.lr_init_val, decay_steps=config.epochs
    )
    state_unet = training_utils.create_train_state_unet(
        rng_state, config, learning_rate_fn, image_shape
    )
    state_unet = jax_utils.replicate(state_unet)
    post_state_unet = create_posterior_train_state(
        rng_state, config, config_grass, image_shape,
    )
    post_state_unet = jax_utils.replicate(post_state_unet)
    ema = training_utils.EMA(jax_utils.unreplicate(state_unet).params)

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
                minval=0, maxval=len(x_post[1])
            )

            rng_apply = jax.random.split(rng_apply, jax.local_device_count())
            grads, loss = apply_model( # pylint: disable=not-callable
                state_unet, x_post[1][batch_i], rng_apply
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
        rng_samp = jax.random.split(
            rng_samp, (corrupt_obs.shape[0], jax.local_device_count())
        )
        x_post = []
        params = {
            'denoiser_models_0': grass_params,
            'denoiser_models_1': ema.params
        }
        params = jax_utils.replicate(params)

        for corrupt_batch, rng_pmap in zip(corrupt_obs, rng_samp):
            x_post.append(
                jax.device_put(
                    sample_pmap(
                        corrupt_batch, rng_pmap, post_state_unet, params, A_mat,
                        cov_y
                    ),
                    jax.local_devices(backend="cpu")[0]
                )
            )

        # No longer need sampling dimensions for training the state.
        x_post = rearrange(
            jnp.stack(x_post, axis=0), 'K M N ... -> (K M N) ...'
        )
        x_post = jnp.split(x_post, 2, axis=-1)

        # Calculate the pq mass chi^2 for our posterior sample.
        wandb.log(
            {
                'pq_chi2': metrics.pq_mass(
                    x_post[1][:config.pq_mass_samples],
                    mnist_pure[:config.pq_mass_samples]
                ),
                'pq_chi2_ident': metrics.pq_mass(
                    x_post[1][:config.pq_mass_samples],
                    grass_pure_ident[:config.pq_mass_samples]
                ),
                'fcd': metrics.fcd_mnist(
                    mnist_model, mnist_params,
                    rearrange(
                        x_post[1][:config.pq_mass_samples],
                        '... (H W C) -> ... H W C',
                        H=image_shape[0], W=image_shape[1], C=image_shape[2]
                    ),
                    mnist_pure[:config.pq_mass_samples]
                ),
                'inception': metrics.inception_score_mnist(
                    mnist_model, mnist_params,
                    rearrange(
                        x_post[1][:config.pq_mass_samples],
                        '... (H W C) -> ... H W C',
                        H=image_shape[0], W=image_shape[1], C=image_shape[2]
                    )
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
