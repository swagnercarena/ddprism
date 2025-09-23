"Perform CLVM analysis of corrupted MNIST dataset."
import os

from absl import app, flags
from einops import rearrange
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm
import wandb

from ddprism import training_utils
from ddprism import utils
from ddprism.clvm import clvm_utils, models
from ddprism.corrupted_mnist import datasets
from ddprism.metrics import metrics, image_metrics

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'working directory.')
flags.DEFINE_string('imagenet_path', None, 'path to imagenet grass dataset.')

config_flags.DEFINE_config_file(
    'config_clvm', None,
    'File path to the training configuration for CLVM parameters.',
)
flags.DEFINE_string(
    'mnist_classifier_path', None,
    'path to MNIST classifier working directory.'
)


def get_activation_fn(activation_name: str):
    """Get activation function from string name."""
    activation_map = {
        'relu': nn.relu,
        'silu': nn.silu,
        'tanh': nn.tanh,
    }
    return activation_map.get(activation_name, nn.silu)


def create_vae_encoders_decoders(config, image_shape):
    """Create VAE encoders and decoders based on config."""
    activation_fn = get_activation_fn(config.vae.activation)

    # Create signal encoder
    signal_encoder = models.EncoderMLP(
        latent_features=config.latent_dim_t,
        hid_features=config.vae.hid_features,
        activation=activation_fn,
        dropout_rate=config.vae.dropout_rate,
        normalize=config.vae.normalize
    )

    # Create background encoder
    bkg_encoder = models.EncoderMLP(
        latent_features=config.latent_dim_z,
        hid_features=config.vae.hid_features,
        activation=activation_fn,
        dropout_rate=config.vae.dropout_rate,
        normalize=config.vae.normalize
    )

    # Create signal decoder
    signal_decoder = models.DecoderMLP(
        features=image_shape[0] * image_shape[1] * image_shape[2],
        hid_features=config.vae.hid_features,
        activation=activation_fn,
        dropout_rate=config.vae.dropout_rate,
        normalize=config.vae.normalize
    )

    # Create background decoder
    bkg_decoder = models.DecoderMLP(
        features=image_shape[0] * image_shape[1] * image_shape[2],
        hid_features=config.vae.hid_features,
        activation=activation_fn,
        dropout_rate=config.vae.dropout_rate,
        normalize=config.vae.normalize
    )

    return signal_encoder, bkg_encoder, signal_decoder, bkg_decoder


@jax.jit
def train_step(state, rng, enr_obs, bkg_obs, other_vars):
    """Perform a single training step."""
    def loss_fn(params):
        # Collect the relevant variables.
        variables = {'params': params, 'variables': other_vars}
        rng_bkg, rng_enr, rng_drop = jax.random.split(rng, 3)
        rng_drop_bkg, rng_drop_enr = jax.random.split(rng_drop)

        # Enriched observation loss
        enr_loss = state.apply_fn(
            variables, rng_enr, enr_obs, method='loss_enr_feat',
            train=True, rngs={'dropout': rng_drop_enr}
        )
        # Background observation loss
        bkg_loss = state.apply_fn(
            variables, rng_bkg, bkg_obs, method='loss_bkg_feat',
            train=True, rngs={'dropout': rng_drop_bkg}
        )
        return enr_loss + bkg_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def get_posterior_samples(
    rng, state, enr_obs_flat, other_vars, batch_size
):
    """Get posterior samples from the CLVM model."""
    # Collect the relevant variables.
    variables = {'params': state.params, 'variables': other_vars}

    # Encode the enriched observations to get posterior parameters
    enr_obs_flat = rearrange(
        enr_obs_flat, '(K N) ... -> K N ...', N=batch_size
    )
    signal_feat, bkg_feat = [], []
    for enr_obs in enr_obs_flat:
        latents = state.apply_fn(
            variables, enr_obs, method='encode_enr_feat', train=False
        )
        latent_draw = state.apply_fn(
            variables, rng, latents[0], latents[1], method='_latent_draw',
        )
        z_latent, t_latent = state.apply_fn(
            variables, latent_draw, method='_latent_split'
        )
        signal_feat.append(
            state.apply_fn(
                variables, t_latent, method='decode_signal_feat', train=False
            )
        )
        bkg_feat.append(
            state.apply_fn(
                variables, z_latent, method='decode_bkg_feat', train=False
            )
        )

    return (
        jnp.concatenate(bkg_feat, axis=0), jnp.concatenate(signal_feat, axis=0)
    )


def get_prior_samples(rng, state, num_samples, other_vars, batch_size):
    """Get prior samples from the CLVM model."""
    # Collect the relevant variables.
    variables = {'params': state.params, 'variables': other_vars}

    # Sample from prior (standard normal)
    signal_feat, bkg_feat = [], []
    for _ in range(num_samples // batch_size):
        rng, _ = jax.random.split(rng)
        latent_draw = state.apply_fn(
            variables, rng, batch_size, method='_latent_draw_prior'
        )
        z_latent, t_latent = state.apply_fn(
            variables, latent_draw, method='_latent_split'
        )
        signal_feat.append(state.apply_fn(
            variables, t_latent, method='decode_signal_feat', train=False
        ))
        bkg_feat.append(state.apply_fn(
            variables, z_latent, method='decode_bkg_feat', train=False
        ))

    return (
        jnp.concatenate(bkg_feat, axis=0),
        jnp.concatenate(signal_feat, axis=0)
    )


def run_clvm(config_clvm, workdir):
    """Run CLVM analysis of corrupted MNIST digits.

    Args:
        config_clvm: Configuration object for CLVM parameters.
        workdir: Working directory.

    Returns:
        Dictionary of metrics.
    """
    config_mnist = config_clvm.config_mnist
    imagenet_path = FLAGS.imagenet_path
    rng = jax.random.PRNGKey(config_mnist.rng_key)

    # Set up wandb logging and checkpointing.
    wandb.init(
        config=config_clvm.copy_and_resolve_references(),
        project=config_clvm.wandb_kwargs.get('project', None),
        name=config_clvm.wandb_kwargs.get('run_name', None),
        mode=config_clvm.wandb_kwargs.get('mode', 'disabled')
    )

    # Read our full dataset to cpu.
    rng_dataset, rng_comp, rng = jax.random.split(rng, 3)
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        # Get our observations, mixing matrix, and covariance.
        enr_obs, _, _, _ = datasets.get_dataset(
            rng_dataset, 1.0, config_mnist.mnist_amp, config_mnist.sigma_y,
            config_mnist.downsampling_ratios, config_mnist.sample_batch_size,
            imagenet_path, config_mnist.dataset_size
        )
        bkg_obs, _, _, _ = datasets.get_dataset(
            rng_dataset, 1.0, 0.0, config_mnist.sigma_y,
            config_mnist.downsampling_ratios, config_mnist.sample_batch_size,
            imagenet_path, config_mnist.dataset_size
        )

        # Save the observations, A matrix, and covariance.
        np.save(os.path.join(workdir, 'corrupt_obs.npy'), enr_obs)
        np.save(os.path.join(workdir, 'bkg_obs.npy'), bkg_obs)

        # Reshape our observations to have a batch dimension for the
        # sample_batch_size and a pmap dimension.
        enr_obs = rearrange(enr_obs, 'N H W C -> N (H W C)')
        bkg_obs = rearrange(bkg_obs, 'N H W C -> N (H W C)')

        # Get pure samples for Gaussian initialization and metric calculations.
        mnist_pure, _ = datasets.get_corrupted_mnist(
            rng_comp, 0.0, 1.0, imagenet_path, config_mnist.dataset_size
        )
        image_shape = mnist_pure.shape[1:]
        mnist_pure_flat = mnist_pure.reshape(mnist_pure.shape[0], -1)

        # Pull the pure sample and save to disk.
        grass_pure_ident, _ = datasets.get_corrupted_mnist(
            rng_dataset, 1.0, 0.0, imagenet_path, config_mnist.dataset_size
        )
        np.save(os.path.join(workdir, 'grass_pure_ident.npy'), grass_pure_ident)

    # Initialize the CLVM model based on config.model_type
    if config_clvm.model_type == "linear":
        clvm_model = clvm_utils.CLVMLinear(
            features=image_shape[0] * image_shape[1] * image_shape[2],
            latent_dim_z=config_clvm.latent_dim_z,
            latent_dim_t=config_clvm.latent_dim_t,
            obs_dim=image_shape[0] * image_shape[1] * image_shape[2]
        )
    elif config_clvm.model_type == "vae":
        # Create VAE encoders and decoders
        signal_encoder, bkg_encoder, signal_decoder, bkg_decoder = (
            create_vae_encoders_decoders(config_clvm, image_shape)
        )
        clvm_model = clvm_utils.CLVMVAE(
            features=image_shape[0] * image_shape[1] * image_shape[2],
            latent_dim_z=config_clvm.latent_dim_z,
            latent_dim_t=config_clvm.latent_dim_t,
            obs_dim=image_shape[0] * image_shape[1] * image_shape[2],
            signal_encoder=signal_encoder,
            bkg_encoder=bkg_encoder,
            signal_decoder=signal_decoder,
            bkg_decoder=bkg_decoder
        )
    else:
        raise ValueError(f"Unknown model_type: {config_clvm.model_type}.")

    # Load our MNIST classifier for metrics.
    rng_class, rng = jax.random.split(rng)
    mnist_model, mnist_params = image_metrics.get_model(
        FLAGS.mnist_classifier_path, rng_class,
        **config_mnist.get('mnist_classifier_kwargs', {})
    )

    # Initialize model parameters.
    rng, rng_init = jax.random.split(rng)
    dummy_obs = jax.random.normal(
        rng_init, (1, image_shape[0] * image_shape[1] * image_shape[2])
    )
    variables = clvm_model.init(
        rng_init, rng, dummy_obs, method='loss_enr_feat'
    )
    other_vars = {'log_sigma_obs': jnp.log(config_mnist.sigma_y)}

    # For linear models, we want to initialize the weight matrices and means
    # using ppca.
    if config_clvm.model_type == "linear":
        mnist_mu, mnist_cov = utils.ppca(
            rng_init, enr_obs, rank=config_clvm.latent_dim_t
        )
        bkg_mu, bkg_cov = utils.ppca(
            rng_init, bkg_obs, rank=config_clvm.latent_dim_z
        )
        variables['params']['w_mat'] = (
            mnist_cov.u_mat / jnp.linalg.vector_norm(mnist_cov.u_mat)[None]
        )
        variables['params']['s_mat'] = (
            bkg_cov.u_mat / jnp.linalg.vector_norm(bkg_cov.u_mat)[None]
        )
        variables['params']['mu_signal'] = mnist_mu - bkg_mu
        variables['params']['mu_bkg'] = bkg_mu

    # Set up our training state.
    learning_rate_fn = training_utils.get_learning_rate_schedule(
        config_clvm, config_clvm.lr_init_val,
        config_clvm.epochs * config_clvm.steps_per_epoch
    )
    optimizer = training_utils.get_optimizer(config_clvm)(learning_rate_fn)
    state = train_state.TrainState.create(
        apply_fn=clvm_model.apply,
        params=variables['params'],
        tx=optimizer
    )

    for _ in tqdm(range(config_clvm.epochs), desc='Training CLVM'):
        for _ in tqdm(range(config_clvm.steps_per_epoch), desc='Steps', leave=False):
            # Get random batch for enriched data
            rng, rng_batch_enr = jax.random.split(rng)
            batch_i = jax.random.randint(
                rng_batch_enr, shape=(config_clvm.batch_size,),
                minval=0, maxval=enr_obs.shape[0]
            )
            # Get random batch for background data
            rng, rng_batch_bkg = jax.random.split(rng)
            batch_i_bkg = jax.random.randint(
                rng_batch_bkg, shape=(config_clvm.batch_size,),
                minval=0, maxval=bkg_obs.shape[0]
            )

            # Training step
            rng, rng_train = jax.random.split(rng)
            state, loss = train_step(
                state, rng_train, enr_obs[batch_i], bkg_obs[batch_i_bkg],
                other_vars
            )

            # Log loss
            wandb.log({'loss': loss})

        # Generate posterior samples.
        rng, rng_post = jax.random.split(rng)
        _, post_samples = get_posterior_samples(
            rng_post, state, enr_obs, other_vars, config_clvm.sample_batch_size
        )
        # Correct the posterior samples for the A matrix.
        post_samples /= config_mnist.mnist_amp

        # Generate prior samples
        rng, rng_prior = jax.random.split(rng)
        num_samples = max(
            config_mnist.psnr_samples, config_mnist.sinkhorn_div_samples,
            config_mnist.pq_mass_samples, config_mnist.fcd_samples
        )
        _, prior_samples = get_prior_samples(
            rng_prior, state, num_samples, other_vars,
            config_clvm.sample_batch_size
        )
        # Correct the prior samples for the A matrix.
        prior_samples /= config_mnist.mnist_amp

        # Compute and log metrics.
        metrics_dict = {}
        metrics_dict['mnist_fcd_post'] = float(image_metrics.fcd_mnist(
            mnist_model, mnist_params,
            rearrange(
                post_samples[:config_mnist.pq_mass_samples],
                '... (H W C) -> ... H W C',
                H=image_shape[0], W=image_shape[1], C=image_shape[2]
            ),
            mnist_pure[:config_mnist.pq_mass_samples]
        ))
        metrics_dict['mnist_fcd_prior'] = float(image_metrics.fcd_mnist(
            mnist_model, mnist_params,
            rearrange(
                prior_samples[:config_mnist.pq_mass_samples],
                '... (H W C) -> ... H W C',
                H=image_shape[0], W=image_shape[1], C=image_shape[2]
            ),
            mnist_pure[:config_mnist.pq_mass_samples]
        ))
        metrics_dict['mnist_pqmass_post'] = float(metrics.pq_mass(
            post_samples[:config_mnist.pq_mass_samples],
            mnist_pure_flat[:config_mnist.pq_mass_samples]
        ))
        metrics_dict['mnist_pqmass_prior'] = float(metrics.pq_mass(
            prior_samples[:config_mnist.pq_mass_samples],
            mnist_pure_flat[:config_mnist.pq_mass_samples]
        ))
        metrics_dict['mnist_divergence_post'] = float(metrics.sinkhorn_divergence(
            post_samples[:config_mnist.sinkhorn_div_samples],
            mnist_pure_flat[:config_mnist.sinkhorn_div_samples]
        ))
        metrics_dict['mnist_divergence_prior'] = float(metrics.sinkhorn_divergence(
            prior_samples[:config_mnist.sinkhorn_div_samples],
            mnist_pure_flat[:config_mnist.sinkhorn_div_samples]
        ))
        metrics_dict['mnist_psnr_post'] = float(metrics.psnr(
            post_samples[:config_mnist.psnr_samples],
            mnist_pure_flat[:config_mnist.psnr_samples],
            max_spread=datasets.MAX_SPREAD
        ))
        wandb.log(metrics_dict, commit=False)

    # Save parameters to a checkpoint.
    checkpointer = PyTreeCheckpointer()
    checkpoint_options = CheckpointManagerOptions(
        enable_async_checkpointing=False
    )
    checkpoint_manager = CheckpointManager(
        os.path.join(workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )
    ckpt = {
        'params': jax.device_get(state.params),
        'config': config_clvm.to_dict(),
        'variables': jax.device_get(other_vars),
        'metrics': jax.device_get(metrics_dict),
        'x_post': jax.device_get(post_samples),
        'x_prior': jax.device_get(prior_samples)
    }
    checkpoint_manager.save(0, ckpt)
    checkpoint_manager.close()

    # Finish wandb run.
    wandb.finish()
    return metrics_dict


def main(_):
    """Run CLVM analysis of corrupted MNIST digits."""
    config_clvm = FLAGS.config_clvm
    workdir = FLAGS.workdir
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    metrics_dict = run_clvm(config_clvm, workdir)
    print(metrics_dict)


if __name__ == '__main__':
    app.run(main)
