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
from ddprism.pcpca import pcpca_utils

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

        # In PCPCA language, y_enr is enriched observation and y_bkg is background.
        y_enr = y_all[:, source_index]

        if source_index == 0:
            y_bkg = jnp.zeros_like(y_enr)
        else:
            y_bkg = y_all[:, source_index-1]

        enr_a_mat = A_all[:,source_index]
        if source_index == 0:
            bkg_a_mat = jnp.zeros_like(enr_a_mat)
        else:
            bkg_a_mat = A_all[:,source_index-1]

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
        for step in range(config.n_iter):
            grad = pcpca_utils.loss_grad(
                params, y_enr, y_bkg, enr_a_mat, bkg_a_mat, gamma
            )
            loss = pcpca_utils.loss(
                params, y_enr, y_bkg, enr_a_mat, bkg_a_mat, gamma
            )
            params['weights'] -= schedule(step) * grad['weights']

            # Log our loss
            wandb.log({f'loss {source_index + 1}': loss}, step=step)

        # Get the posterior for the signal underlying the observation.
        x_mean, x_cov = jax.vmap(
            pcpca_utils.calculate_posterior, in_axes=(None, 0, 0, 0)
        )(params, y_enr, enr_a_mat)

        # Draw samples from the posterior.
        rng_x, rng = jax.random.split(rng, 3)
        x_post_draws = jax.random.multivariate_normal(
            rng_x, mean=x_mean, cov=x_cov
        )

        divergence_x_draws = utils.sinkhorn_divergence(
            x_post_draws[:config.sinkhorn_samples],
            x_all[:config.sinkhorn_samples, source_index]
        )
        wandb.log(
            {f'div_post_{source_index+1}': divergence_x_draws},
            step=config.n_iter
        )

        # Log a figure with new posterior samples.
        if config.log_figure:
            fig = plotting_utils.show_corner(x_post_draws)._figure
            wandb.log(
                    {f'posterior samples {source_index+1}': wandb.Image(fig)},
                    step=config.n_iter
                )

        # Plot samples from the prior
        z_draws = jax.random.normal(
            rng_x, shape=(config.sample_size, config.latent_dim)
        )
        x_prior_draws = jnp.matmul(params['weights'], z_draws[:, :, None])
        divergence_x_prior_draws = utils.sinkhorn_divergence(
            x_prior_draws[:config.sinkhorn_samples],
            x_all[:config.sinkhorn_samples, source_index]
        )
        wandb.log(
            {f'div_prior_{source_index+1}': divergence_x_prior_draws},
            step=config.n_iter
        )

        if config.log_figure:
            fig = plotting_utils.show_corner(x_prior_draws)._figure
            wandb.log(
                    {f'prior samples {source_index+1}': wandb.Image(fig)},
                    step=config.n_iter
                )

if __name__ == '__main__':
    app.run(main)
