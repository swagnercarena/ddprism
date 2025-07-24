"Perform PCPCA analysis of Grassy MNIST dataset without downsampling."
import os

from absl import app, flags
import jax
import jax.numpy as jnp
from ml_collections import config_flags
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
from tqdm import tqdm
import wandb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from pathlib import Path

from ddprism.corrupted_mnist import datasets
from ddprism.metrics import metrics, image_metrics
from ddprism.pcpca import pcpca_utils

jax.config.update("jax_enable_x64", True)
    
FLAGS = flags.FLAGS
    
flags.DEFINE_string('workdir', None, 'working directory.')

config_flags.DEFINE_config_file(
    'config_pcpca', None, 'File path to the training configuration for PCPCA parameters.',
)

# Path to Imagenet dataset.
imagenet_path = Path('/mnt/home/aakhmetzhanova/ceph/galaxy-diffusion/corrupted-mnist/dataset/grass_jpeg/')
# Path to the classifier model.
classifier_path = Path('/mnt/home/swagner/ceph/corrupted_mnist/mnist_classifier/')

@partial(jax.jit, static_argnames=['batch_size', 'img_dim'])
def get_posterior_samples(rng, params, x, y, mnist_amp, img_dim=28, batch_size=16):
    """Compute posterior samples for corrupted MNIST digits.

    Args: 
        rng: Rng key for sampling.
        params: Dictionary of PCPCA parameters.
        x: Target dataset (MNIST digits on top of grass images from ImageNet).
        y: Background dataset (grass images from ImageNet).
        mnist_amp: Amplitude of MNIST digits in the target dataset.
        img_dim: Size of the image (number of pixels per side).
        batch_size: Batch size for processing the target dataset with jax.lax.map.
        
    Returns:
        Posterior samples for uncorrupted MNIST digits from the target dataset.
    
    """
    # Mixing matrix A is identity for full resolution grassy MNIST dataset.
    a_mat = jnp.eye(x.shape[-1])
    # Compute mean signal for the background dataset.
    bkg_mean = y.mean(axis=0)

    # Get individual random keys for each posterior sample.
    rng_post = jax.random.split(rng, x.shape[0])

    # Helper functions
    # Compute the mean and covariance for the signal posterior.
    def calculate_posterior(x):
        return pcpca_utils.calculate_posterior(params, x, a_mat) 
    # Draw a sample from the signal posterior distribution.
    def post_samples(args):
        rng, x = args
        mean, sigma = calculate_posterior(x)
        return (jax.random.multivariate_normal(rng, mean, sigma) - bkg_mean) / mnist_amp

    # Draw posterior samples for the entire dataset.
    post_samples = jax.lax.map(post_samples, (rng_post, x), batch_size=batch_size)
    return post_samples.reshape(-1, img_dim, img_dim, 1)

def get_prior_samples(rng, params, num_samples, latent_dim, bkg, mnist_amp, img_dim=28):
    """Draw samples from the prior for MNIST digits.

    Args: 
        rng: Rng key for sampling.
        params: Dictionary of PCPCA parameters.
        num_samples: Number of prior samples to draw.
        latent_dim: Number of latent dimensions of the PCPCA model.
        mnist_amp: Amplitude of MNIST digits in the target dataset.
        img_dim: Size of the image (number of pixels per side).
        
    Returns:
        Prior samples for uncorrupted MNIST digits.
    
    """
    # Draw latent and noise vectors.
    rng_z, rng_eps, rng = jax.random.split(rng, 3)
    z_x = jax.random.multivariate_normal(
        rng_z, mean=jnp.zeros((num_samples, latent_dim)), cov=jnp.eye(latent_dim)
    )

    feat_dim = img_dim**2
    eps_x = jax.random.multivariate_normal(
        rng_eps, mean=jnp.zeros((num_samples, feat_dim)), cov = jnp.exp(params['log_sigma'])**2*jnp.eye(feat_dim)
    )
    # Compute prior samples for target dataset.
    prior_samples = ((params['weights'] @ z_x.T).T + params['mu'] + eps_x)
    
    # Subtract contribution due to the background signal and scale by the appropriate amplitude.
    prior_samples = (prior_samples - bkg.mean(axis=0)) / mnist_amp
    prior_samples = prior_samples.reshape(-1, img_dim, img_dim, 1)
    
    return prior_samples


def run_pcpca(config_pcpca, workdir):
    config_mnist = config_pcpca.config_mnist
    config_grass = config_pcpca.config_grass
    
    # RNG key from config.
    rng = jax.random.key(config_mnist.rng_key)
    
    # Set up wandb logging and checkpointing.
    wandb.init(
        config=config_pcpca.copy_and_resolve_references(),
        project=config_pcpca.wandb_kwargs.get('project', None),
        name=config_pcpca.wandb_kwargs.get('run_name', None),
        mode=config_pcpca.wandb_kwargs.get('mode', 'disabled')
    )
    
    # Generate training datasets.
    # Target dataset with corrupted mnist digits.
    rng_dataset, rng_comp, rng = jax.random.split(rng, 3)
    
    f_train = datasets.get_corrupted_mnist(
        rng_dataset, grass_amp=1., mnist_amp=config_mnist.mnist_amp,
        imagenet_path=imagenet_path,
        dataset_size=config_mnist.dataset_size,
        zeros_and_ones=True
    )
    # Target dataset with uncorrupted mnist digits for computing metrics later on.
    f_train_uncorrupted = datasets.get_corrupted_mnist(
        rng_dataset, grass_amp=0., mnist_amp=1.,
        imagenet_path=imagenet_path,
        dataset_size=config_mnist.dataset_size,
        zeros_and_ones=True
    )
    # Background dataset with grass only.
    #config_grass = config_base_grass.get_config()
    rng = jax.random.key(config_grass.rng_key)
    rng_dataset, rng_comp, rng = jax.random.split(rng, 3)
    b_train = datasets.get_corrupted_mnist(
        rng_dataset, grass_amp=1., mnist_amp=0.,
        imagenet_path=imagenet_path, 
        dataset_size=config_grass.dataset_size,
        zeros_and_ones=True)

    # Prepare data for PCPCA
    x, x_labels = f_train
    y, y_labels = b_train

    feat_dim = x.shape[-3]*x.shape[-2]*x.shape[-1] 
    x = x.squeeze(-1).reshape(-1, feat_dim)
    y = y.squeeze(-1).reshape(-1, feat_dim)
    
    # Fit PCPCA
    params = pcpca_utils.mle_params(x, y, config_pcpca.gamma, config_pcpca.latent_dim, sigma=config_grass.sigma_y);

    # Get the posterior samples for MNIST digits.
    rng_post, rng = jax.random.split(rng, 2)
    post_samples  = get_posterior_samples(rng_post, params, x, y, config_mnist.mnist_amp) 

    # Get prior samples for MNIST digits.
    rng_prior, rng = jax.random.split(rng, 2)
    num_samples = x.shape[0]
    prior_samples = get_prior_samples(rng_prior, params, num_samples, config_pcpca.latent_dim, y, config_mnist.mnist_amp)

    # Plot the first few prior and posterior samples.
    # Posterior samples
    n_rows, n_cols = 3, 5
    
    idx = jnp.arange(10, 60, 10)
    fig, axs = plt.subplots(n_rows, n_cols)
    for col in range(n_cols):
        axs[0, col].imshow(post_samples.reshape(-1, 28, 28, 1)[idx[col]], vmin=0.0, vmax=1.0)
        axs[1, col].imshow(f_train_uncorrupted[0][idx[col]], vmin=0.0, vmax=1.0)
        axs[2, col].imshow(f_train[0][idx[col]], vmin=0.0, vmax=1.0)
    wandb.log(
        {f'posterior samples': wandb.Image(fig)},
        commit=False
    )

    # Prior samples
    idx = jnp.arange(10, 60, 10)
    fig, axs = plt.subplots(1, n_cols)
    for col in range(n_cols):
        axs[col].imshow(prior_samples.reshape(-1, 28, 28, 1)[idx[col]], vmin=0.0, vmax=1.0)
    wandb.log(
        {f'prior samples': wandb.Image(fig)},
        commit=False
    )

    print(post_samples.shape, prior_samples.shape)
    
    # Compute and log metrics.
    metrics_dict = {}
    x_uncorrupted = f_train_uncorrupted[0]
    
    # Compute FCD on posterior and prior samples.
    # Load classifier model to compute FCD.
    checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(classifier_path, checkpointer)
    classifier_model = image_metrics.CNN()
    classifier_params = checkpoint_manager.restore(checkpoint_manager.latest_step())['params'] 
    checkpoint_manager.close()
    
    fcd_post = image_metrics.fcd_mnist(
        classifier_model, 
        classifier_params, 
        x_uncorrupted, 
        post_samples
    )
    wandb.log(
            {'fcd_post': fcd_post}, commit=False
        )
    
    fcd_prior = image_metrics.fcd_mnist(
        classifier_model, 
        classifier_params, 
        x_uncorrupted, 
        prior_samples
    )
    wandb.log(
            {'fcd_prior': fcd_prior}, commit=False
        )

    # Compute PSNR.
    psnr_post = metrics.psnr(
            post_samples[:config_mnist.psnr_samples],
            x_uncorrupted[:config_mnist.psnr_samples],
            max_spread=config_mnist.MAX_SPREAD
    )
    wandb.log(
            {'psnr_post': psnr_post}, commit=False
    )
    

    '''
    # Compute PQMAss.
    pqmass_post = metrics.pq_mass(
        post_samples[:config_mnist.pq_mass_samples],
        x_uncorrupted[:config_mnist.pq_mass_samples]
    )
    wandb.log(
            {'pqmass_post': pqmass_post}, commit=False
    )
    
    pqmass_prior = metrics.pq_mass(
        prior_samples[:config_mnist.pq_mass_samples],
        x_uncorrupted[:config_mnist.pq_mass_samples]
    )
    wandb.log(
            {'pqmass_prior': pqmass_prior}, commit=False
    )
    '''
    
    # Compute Sinkhorn divegence.
    divergence_post = metrics.sinkhorn_divergence(
        post_samples.reshape(post_samples.shape[0], -1)[:config_mnist.sinkhorn_div_samples],
        x_uncorrupted.reshape(x_uncorrupted.shape[0], -1)[:config_mnist.sinkhorn_div_samples]
    )
    wandb.log(
        {f'div_post': divergence_post}, commit=False
    )

    divergence_prior = metrics.sinkhorn_divergence(
        prior_samples.reshape(prior_samples.shape[0], -1)[:config_mnist.sinkhorn_div_samples],
        x_uncorrupted.reshape(x_uncorrupted.shape[0], -1)[:config_mnist.sinkhorn_div_samples]
    )
    wandb.log(
        {f'div_prior': divergence_prior}, commit=False
    )
    
    # Save parameters to a checkpoint.
    checkpointer = PyTreeCheckpointer()
    checkpoint_options = CheckpointManagerOptions(
        enable_async_checkpointing=False
    )
    checkpoint_manager = CheckpointManager(
        os.path.join(workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )
    checkpoint_manager.save(0, params)
    checkpoint_manager.close()
    
    # Record performance metrics (FCD, PSNR, PQMass, Sinkhorn divergence).
    metrics_dict['fcd_post'] = float(fcd_post)
    metrics_dict['fcd_prior'] = float(fcd_prior)
    
    metrics_dict['psnr_post'] = float(psnr_post)

    '''
    metrics_dict['pqmass_post'] = float(pqmass_post)
    metrics_dict['pqmass_prior'] = float(pqmass_prior)
    '''

    metrics_dict['div_post']  = float(divergence_post)
    metrics_dict['div_prior'] = float(divergence_prior)

    wandb.finish()
    return metrics_dict


def main(_):
    """Find best PCPCA parameters for each source by minimizing PCPCA loss function."""
    
    config_pcpca = FLAGS.config_pcpca

    workdir = FLAGS.workdir
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    metrics_dict = run_pcpca(config_pcpca, workdir)
    print(metrics_dict)


if __name__ == '__main__':
    app.run(main)