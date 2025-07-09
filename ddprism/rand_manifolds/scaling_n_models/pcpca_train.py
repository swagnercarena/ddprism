"Perform PCPCA analysis of missing data with n_sources."
import os

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
from ddprism.pcpca import pcpca_utils
from ddprism import plotting_utils

import load_dataset
import pcpca_utils

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)


def main(_):
    """Train a joint posterior denoiser."""
    config = FLAGS.config
    rng = jax.random.PRNGKey(config.rng_key)

    workdir = FLAGS.workdir
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

    # Generate our dataset.
    rng, rng_data = jax.random.split(rng)
    x_all, A_all, y_all, _ = load_dataset.get_dataset(rng_data, config)

    # Allow for different gammas for each source.
    if isinstance(config.gamma, list):
        assert len(config.gamma) == config.n_sources
        gamma_list = config.gamma
    else:
        gamma_list = [config.gamma for i in range(config.n_sources)]

    # Do each PCPCA / PPCA analysis.
    for source_index in range(config.n_sources):
        # Get the gamma and relevant observations.
        gamma = gamma_list[source_index]

        # In PCPCA language, x_obs is enriched signal and y_obs is background.
        x_obs = y_all[:, source_index]

        if source_index == 0:
            y_obs = jnp.zeros_like(x_obs)
        else:
            y_obs = y_all[:, source_index-1]

        x_a_mat = A_all[:,source_index]
        if source_index == 0:
            y_a_mat = jnp.zeros_like(x_a_mat)
        else:
            y_a_mat = A_all[:,source_index-1]

        # Initialize W and log_sigma.
        rng_w, rng = jax.random.split(rng, 2)
        weights_init = jax.random.uniform(
            rng_w, shape=(config.feat_dim, config.latent_dim), minval=-1.0,
            maxval=1.0
        )
        log_sigma_init = jnp.log(config.sigma_y)
        params = {
            'weights': jnp.asarray(weights_init), 'log_sigma': log_sigma_init
        }

        # Optimization loop parameters.
        if config.lr_schedule == 'linear':
            schedule = optax.schedules.linear_schedule(
                config.learning_rate, 1e-6, config.n_iter
            )
        elif config.lr_schedule == 'cosine':
            schedule = optax.schedules.cosine_decay_schedule(
                init_value=config.learning_rate, decay_steps=config.n_iter
            )
        else:
            raise ValueError(
                f'Unknown learning rate schedule: {config.lr_schedule}'
            )

        # Run the optimization loop.
        for iter in range(config.n_iter):
            grad = pcpca_utils.loss_grad(
                params, x_obs, y_obs, x_a_mat, y_a_mat, gamma
            )
            loss = pcpca_utils.loss(
                params, x_obs, y_obs, x_a_mat, y_a_mat, gamma
            )
            params['weights'] -= schedule(iter) * grad['weights']

            # Log our loss
            wandb.log({f'loss {source_index + 1}': loss}, step=iter)

        ### CODE BELOW IS FOR FUTURE VERIFICATION.

        # Estimate mean of and uncertainty on missing values.
        x_u_mean, x_u_cov = pcpca_utils.impute_missing_data(params, x_obs, L, M, P, config.feat_dim)

        divergence_x = utils.sinkhorn_divergence(x_u_mean[:config.sinkhorn_samples],
                                                 x_all[:config.sinkhorn_samples, source_index])
        wandb.log({f'div_x_{source_index+1}': divergence_x}, step=config.n_iter)

        # Log a figure with new posterior samples.
        if config.log_figure:
            fig = plotting_utils.show_corner(x_u_mean)._figure
            wandb.log(
                    {f'posterior samples {source_index+1}': wandb.Image(fig)},
                    step=config.n_iter
                )

        # Plot samples from the prior
        rng_x, rng_eps_x, rng = jax.random.split(rng, 3)
        z_x   = jax.random.normal(rng_x, shape=(config.sample_size, config.latent_dim))
        eps_x = jax.random.multivariate_normal(rng_eps_x,
                                               mean=jnp.zeros(config.feat_dim),
                                               cov=jnp.eye(config.feat_dim)*params[1],
                                               shape=(config.sample_size,))
        x_u_prior = jnp.matmul(params[0], z_x.T).T + eps_x

        if config.log_figure:
            fig = plotting_utils.show_corner(x_u_prior)._figure
            wandb.log(
                    {f'prior samples {source_index+1}': wandb.Image(fig)},
                    step=config.n_iter
                )

if __name__ == '__main__':
    app.run(main)
