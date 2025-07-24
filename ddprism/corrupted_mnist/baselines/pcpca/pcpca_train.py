"Perform PCPCA analysis of Grassy MNIST dataset without downsampling."
import os
from functools import partial

from absl import app, flags
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ml_collections import config_flags
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
import wandb

from ddprism.corrupted_mnist import datasets
from ddprism.metrics import metrics, image_metrics
from ddprism.pcpca import pcpca_utils

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'working directory.')
flags.DEFINE_string('imagenet_path', None, 'path to imagenet grass dataset.')

config_flags.DEFINE_config_file(
    'config_pcpca', None,
    'File path to the training configuration for PCPCA parameters.',
)
flags.DEFINE_string(
    'mnist_classifier_path', None,
    'path to MNIST classifier working directory.'
)

@partial(jax.jit, static_argnames=['batch_size'])
def get_posterior_samples(rng, params, y_enr, batch_size=16):
    """Compute posterior samples for corrupted MNIST digits.

    Args:
        rng: Rng key for sampling.
        params: Dictionary of PCPCA parameters.
        y_enr: Enriched dataset (MNIST digits on top of grass images from
            ImageNet).
        mnist_amp: Amplitude of MNIST digits in the target dataset.
        batch_size: Batch size for processing the target dataset with
        jax.lax.map.

    Returns:
        Posterior samples for uncorrupted MNIST digits from the target dataset.

    """
    # Mixing matrix A is identity for full resolution grassy MNIST dataset.
    a_mat = jnp.eye(y_enr.shape[-1])

    # Get individual random keys for each posterior sample.
    rng_post = jax.random.split(rng, y_enr.shape[0])

    # Helper functions to ensure that the full covariance matrix is only
    # computed on batches.
    def calculate_posterior(y_enr):
        return pcpca_utils.calculate_posterior(params, y_enr, a_mat)
    def post_samples(args):
        rng, y_enr = args
        mean, sigma_post = calculate_posterior(y_enr)
        # Zero meaned data, so mean doesn't need to be used.
        return (
            jax.random.multivariate_normal(rng, mean * 0.0, sigma_post)
        )

    # Draw posterior samples for the entire dataset.
    post_samples = jax.lax.map(
        post_samples, (rng_post, y_enr), batch_size=batch_size
    )
    return post_samples

def get_prior_samples(rng, params, num_samples):
    """Draw samples from the prior for MNIST digits.

    Args:
        rng: Rng key for sampling.
        params: Dictionary of PCPCA parameters.
        num_samples: Number of prior samples to draw.
    Returns:
        Prior samples for uncorrupted MNIST digits.
    """
    latent_dim = params['weights'].shape[1]

    # Draw latent and noise vectors.
    rng_z, rng_eps = jax.random.split(rng, 2)
    z_samples = jax.random.normal(rng_z, shape=(num_samples, latent_dim))

    eps_x = jax.random.normal(
        rng_eps, shape=(num_samples, params['weights'].shape[0])
    )

    # Compute prior samples for target dataset.
    prior_samples = ((params['weights'] @ z_samples.T).T + eps_x)

    return prior_samples


def run_pcpca(config_pcpca, workdir):
    """Run PCPCA analysis of corrupted MNIST digits.

    Args:
        config_pcpca: Configuration object for PCPCA parameters.
        workdir: Working directory.

    Returns:
        Dictionary of metrics.
    """
    config_mnist = config_pcpca.config_mnist
    config_grass = config_pcpca.config_grass
    imagenet_path = FLAGS.imagenet_path
    classifier_path = FLAGS.mnist_classifier_path

    # RNG key from config.
    rng_mnist = jax.random.key(config_mnist.rng_key)
    rng_grass = jax.random.key(config_grass.rng_key)

    # Set up wandb logging and checkpointing.
    wandb.init(
        config=config_pcpca.copy_and_resolve_references(),
        project=config_pcpca.wandb_kwargs.get('project', None),
        name=config_pcpca.wandb_kwargs.get('run_name', None),
        mode=config_pcpca.wandb_kwargs.get('mode', 'disabled')
    )

    # Generate training datasets.
    rng_enr, rng_comp, rng = jax.random.split(rng_mnist, 3)
    y_enr, _ = datasets.get_corrupted_mnist(
        rng_enr, grass_amp=1., mnist_amp=config_mnist.mnist_amp,
        imagenet_path=imagenet_path, dataset_size=config_mnist.dataset_size,
        zeros_and_ones=True
    )
    # Add noise to the enriched dataset. Match rng pattern we use in the
    # get_dataset function.
    rng_err, _ = jax.random.split(rng_enr)
    y_enr = y_enr + jax.random.normal(
        rng_enr, shape=y_enr.shape
    ) * config_grass.sigma_y

    # Target dataset with uncorrupted mnist digits for computing metrics later
    # on. rng key is irrelevant for this dataset.
    mnist_target, _ = datasets.get_corrupted_mnist(
        rng_comp, grass_amp=0., mnist_amp=1., imagenet_path=imagenet_path,
        dataset_size=config_mnist.dataset_size, zeros_and_ones=True
    )

    # Background dataset with grass only.
    rng_bkg, _, _ = jax.random.split(rng_grass, 3)
    y_bkg, _ = datasets.get_corrupted_mnist(
        rng_bkg, grass_amp=1.0, mnist_amp=0.0, imagenet_path=imagenet_path,
        dataset_size=config_grass.dataset_size, zeros_and_ones=True
    )
    # Add noise to the background dataset.
    rng_err, _ = jax.random.split(rng_bkg)
    y_bkg = y_bkg + jax.random.normal(
        rng_err, shape=y_bkg.shape
    ) * config_grass.sigma_y

    # Normalize the datasets, and subtract the mean of the background dataset
    # from the enriched dataset.
    bkg_mean, bkg_std = jnp.mean(y_bkg, axis=0), jnp.std(y_bkg, axis=0)
    y_enr -= bkg_mean
    enr_mean, enr_std = jnp.mean(y_enr, axis=0), jnp.std(y_enr, axis=0)
    y_enr = (y_enr - enr_mean) / enr_std
    y_bkg = (y_bkg - bkg_mean) / bkg_std

    # Flatten the dataset.
    feat_dim = y_enr.shape[-3] * y_enr.shape[-2] * y_enr.shape[-1]
    y_enr = y_enr.squeeze(-1).reshape(-1, feat_dim)
    y_bkg = y_bkg.squeeze(-1).reshape(-1, feat_dim)

    # Fit PCPCA
    params = pcpca_utils.mle_params(
        y_enr, y_bkg, config_pcpca.gamma, config_pcpca.latent_dim,
        sigma=config_grass.sigma_y
    )

    # Get the posterior samples for MNIST digits.
    rng_post, rng = jax.random.split(rng, 2)
    post_samples  = get_posterior_samples(
        rng_post, params, y_enr, config_mnist.mnist_amp
    )
    # Reshape and unnormalize the posterior samples.
    post_samples = post_samples.reshape(-1, 28, 28, 1)
    post_samples = post_samples * enr_std + enr_mean
    post_samples /= config_mnist.mnist_amp

    # Get prior samples for MNIST digits.
    rng_prior, rng = jax.random.split(rng, 2)
    num_samples = max(
        config_mnist.psnr_samples, config_mnist.sinkhorn_div_samples,
        config_mnist.pq_mass_samples, config_mnist.fcd_samples
    )
    prior_samples = get_prior_samples(
        rng_prior, params, num_samples
    )
    # Reshape and unnormalize the prior samples.
    prior_samples = prior_samples.reshape(-1, 28, 28, 1)
    prior_samples = prior_samples * enr_std + enr_mean
    prior_samples /= config_mnist.mnist_amp

    # Unormalize and reshape the enriched dataset for plotting.
    y_enr = y_enr * enr_std + enr_mean
    y_enr += bkg_mean
    y_enr = y_enr.reshape(-1, 28, 28, 1)

    # Plot the first few prior and posterior samples.
    n_cols = 5
    fig, axs = plt.subplots(3, n_cols)
    for col in range(n_cols):
        axs[0, col].imshow(post_samples[col], vmin=0.0, vmax=1.0)
        axs[1, col].imshow(mnist_target[col], vmin=0.0, vmax=1.0)
        axs[2, col].imshow(y_enr[col], vmin=0.0, vmax=1.0)
    wandb.log(
        {f'posterior samples': wandb.Image(fig)},
        commit=False
    )
    # Prior samples
    fig, axs = plt.subplots(1, n_cols)
    for col in range(n_cols):
        axs[col].imshow(prior_samples[col], vmin=0.0, vmax=1.0)
    wandb.log(
        {f'prior samples': wandb.Image(fig)},
        commit=False
    )

    # Compute and log metrics.
    metrics_dict = {}

    # Load classifier model to compute FCD.
    checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(classifier_path, checkpointer)
    classifier_model = image_metrics.CNN()
    classifier_params = checkpoint_manager.restore(
        checkpoint_manager.latest_step()
    )['params']
    checkpoint_manager.close()

    # Compute FCD on posterior and prior samples.
    fcd_post = image_metrics.fcd_mnist(
        classifier_model, classifier_params,
        mnist_target[:config_mnist.fcd_samples],
        post_samples[:config_mnist.fcd_samples]
    )
    metrics_dict['fcd_post'] = float(fcd_post)
    fcd_prior = image_metrics.fcd_mnist(
        classifier_model, classifier_params,
        mnist_target[:config_mnist.fcd_samples],
        prior_samples[:config_mnist.fcd_samples]
    )
    metrics_dict['fcd_prior'] = float(fcd_prior)

    # Compute PSNR.
    psnr_post = metrics.psnr(
            post_samples[:config_mnist.psnr_samples],
            mnist_target[:config_mnist.psnr_samples],
            max_spread=datasets.MAX_SPREAD
    )
    metrics_dict['psnr_post'] = float(psnr_post)

    # Compute PQMass.
    pqmass_post = metrics.pq_mass(
        post_samples[:config_mnist.pq_mass_samples],
        mnist_target[:config_mnist.pq_mass_samples]
    )
    metrics_dict['pqmass_post'] = float(pqmass_post)
    pqmass_prior = metrics.pq_mass(
        prior_samples[:config_mnist.pq_mass_samples],
        mnist_target[:config_mnist.pq_mass_samples]
    )
    metrics_dict['pqmass_prior'] = float(pqmass_prior)

    # Compute Sinkhorn divegence.
    divergence_post = metrics.sinkhorn_divergence(
        post_samples.reshape(
            post_samples.shape[0], -1
        )[:config_mnist.sinkhorn_div_samples],
        mnist_target.reshape(
            mnist_target.shape[0], -1
        )[:config_mnist.sinkhorn_div_samples]
    )
    metrics_dict['div_post'] = float(divergence_post)
    divergence_prior = metrics.sinkhorn_divergence(
        prior_samples.reshape(
            prior_samples.shape[0], -1
        )[:config_mnist.sinkhorn_div_samples],
        mnist_target.reshape(
            mnist_target.shape[0], -1
        )[:config_mnist.sinkhorn_div_samples]
    )
    metrics_dict['div_prior'] = float(divergence_prior)

    # Save parameters to a checkpoint.
    checkpointer = PyTreeCheckpointer()
    checkpoint_options = CheckpointManagerOptions(
        enable_async_checkpointing=False
    )
    checkpoint_manager = CheckpointManager(
        os.path.join(workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )
    checkpoint_manager.save(0, {'params': params, 'metrics': metrics_dict})
    checkpoint_manager.close()

    # Log all of the metrics.
    wandb.log(metrics_dict)

    # Finish wandb run.
    wandb.finish()
    return metrics_dict


def main(_):
    """Run PCPCA analysis of corrupted MNIST digits."""

    config_pcpca = FLAGS.config_pcpca
    workdir = FLAGS.workdir
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    metrics_dict = run_pcpca(config_pcpca, workdir)
    print(metrics_dict)


if __name__ == '__main__':
    app.run(main)