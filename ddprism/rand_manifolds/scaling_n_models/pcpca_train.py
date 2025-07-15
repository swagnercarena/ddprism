"Perform PCPCA analysis of missing data with n_sources."
import os

from absl import app, flags
import jax
import jax.numpy as jnp
from ml_collections import config_flags
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint import PyTreeCheckpointer
import optax
from tqdm import tqdm
import wandb

from ddprism.metrics import metrics
from ddprism.pcpca import pcpca_utils
from ddprism import plotting_utils
from ddprism.rand_manifolds.random_manifolds import MAX_SPREAD

import load_dataset

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)

def run_pcpca(config, workdir):
    """Run PCPCA analysis of missing data on 1D manifolds.

    Args:
        config: Configuration object.
        workdir: Working directory.

    Returns:
        Dictionary of metrics.
    """
    # RNG key from config.
    rng = jax.random.PRNGKey(config.rng_key)

    # Set up wandb logging and checkpointing.
    wandb.init(
        config=config.copy_and_resolve_references(),
        project=config.wandb_kwargs.get('project', None),
        name=config.wandb_kwargs.get('run_name', None),
        mode=config.wandb_kwargs.get('mode', 'disabled')
    )

    # Set up checkpointing.
    checkpointer = PyTreeCheckpointer()
    checkpoint_options = CheckpointManagerOptions(
        enable_async_checkpointing=False
    )
    checkpoint_manager = CheckpointManager(
        os.path.join(workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )

    # Generate our dataset.
    rng, rng_data = jax.random.split(rng)
    x_all, A_all, y_all, _ = load_dataset.get_dataset(rng_data, config)

    # Allow for different gammas for each source.
    if isinstance(config.gamma, list):
        assert len(config.gamma) == config.n_sources
        gamma_list = config.gamma
    else:
        gamma_list = [0] + [config.gamma for _ in range(1, config.n_sources)]

    # Set regularization parameter for numerical stability.
    regularization = getattr(config, 'regularization', 1e-6)

    # Do each PCPCA / PPCA analysis.
    metrics_dict = {}
    for source_index in range(config.n_sources):

        # Get the gamma and relevant observations.
        gamma = gamma_list[source_index]

        # In PCPCA language, y_enr is enriched observation and y_bkg is
        # background.
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

        # Initialize W and log_sigma using PCA of the pseudo-inverse of the
        # observation matrix.
        rng_w, rng = jax.random.split(rng, 2)
        x_pinv = jnp.squeeze(jnp.linalg.pinv(enr_a_mat) @ y_enr[:, :, None])
        cov_empirical = jnp.cov(x_pinv, rowvar=False)
        u_mat, s_mat, _ = jnp.linalg.svd(cov_empirical)
        weights_init = (
            u_mat[:, :config.latent_dim] * jnp.sqrt(s_mat[:config.latent_dim])
        )
        weights_init += 0.01 * jax.random.normal(
            rng_w, shape=(config.feat_dim, config.latent_dim)
        )
        mu_init = jnp.mean(x_pinv, axis=0)
        log_sigma_init = jnp.log(config.sigma_y)

        # Initialize parameters.
        params = {
            'weights': jnp.asarray(weights_init), 'log_sigma': log_sigma_init,
            'mu': jnp.asarray(mu_init)
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

        # Initialize Adam optimizer
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(params)

        # Run the optimization loop.
        pbar = tqdm(range(config.n_iter), desc=f'Source {source_index + 1}')
        # Define the custom step metric
        wandb.define_metric("loss_step")
        for step in pbar:
            grad = jax.jit(pcpca_utils.loss_grad)(
                params, y_enr, y_bkg, enr_a_mat, bkg_a_mat, gamma,
                regularization
            )
            loss = jax.jit(pcpca_utils.loss)(
                params, y_enr, y_bkg, enr_a_mat, bkg_a_mat, gamma,
                regularization
            )

            # Update parameters
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            params['log_sigma'] = jnp.log(config.sigma_y) # Fix log_sigma.

            # Log our loss.
            # Define which metrics to plot against that x-axis
            wandb.define_metric(
                f'loss {source_index + 1}', step_metric='loss_step'
            )
            wandb.log(
                {f'loss {source_index + 1}': loss, 'loss_step': step},
            )
            pbar.set_postfix({'loss': f'{loss:.6f}'})

        # Get the posterior for the signal underlying the observation.
        x_mean, x_cov = jax.vmap(
            pcpca_utils.calculate_posterior, in_axes=(None, 0, 0, None)
        )(params, y_enr, enr_a_mat, regularization)

        # Draw samples from the posterior.
        rng_x, rng = jax.random.split(rng, 2)
        x_post_draws = jax.random.multivariate_normal(
            rng_x, mean=x_mean, cov=x_cov
        )

        # Calculate Sinkhorn divergence for posterior samples.
        divergence_x_draws = metrics.sinkhorn_divergence(
            x_post_draws[:config.sinkhorn_samples],
            x_all[:config.sinkhorn_samples, source_index]
        )
        wandb.log(
            {f'div_post_{source_index+1}': divergence_x_draws}, commit=False
        )

        # Calculate pqmass for posterior samples.
        pqmass_x_draws = metrics.pq_mass(
            x_post_draws[:config.pqmass_samples],
            x_all[:config.pqmass_samples, source_index]
        )
        wandb.log(
            {f'pqmass_post_{source_index+1}': pqmass_x_draws}, commit=False
        )

        # Calculate PSNR for posterior samples.
        psnr_x_draws = metrics.psnr(
            x_post_draws[:config.psnr_samples],
            x_all[:config.psnr_samples, source_index],
            max_spread=MAX_SPREAD
        )
        wandb.log(
            {f'psnr_post_{source_index+1}': psnr_x_draws}, commit=False
        )

        # Log a figure with new posterior samples.
        if config.log_figure:
            fig = plotting_utils.show_corner(x_post_draws)._figure
            wandb.log(
                {f'posterior samples {source_index+1}': wandb.Image(fig)},
                commit=False
            )

        # Plot samples from the prior
        z_draws = jax.random.normal(
            rng_x, shape=(config.sample_size, config.latent_dim)
        )
        x_prior_draws = jnp.squeeze(
            jnp.matmul(params['weights'], z_draws[:, :, None])
        )

        # Calculate Sinkhorn divergence for prior samples.
        divergence_x_prior_draws = metrics.sinkhorn_divergence(
            x_prior_draws[:config.sinkhorn_samples],
            x_all[:config.sinkhorn_samples, source_index]
        )
        wandb.log(
            {f'div_prior_{source_index+1}': divergence_x_prior_draws},
            commit=False
        )

        # Calculate pqmass for prior samples.
        pqmass_x_prior_draws = metrics.pq_mass(
            x_prior_draws[:config.pqmass_samples],
            x_all[:config.pqmass_samples, source_index]
        )
        wandb.log(
            {f'pqmass_prior_{source_index+1}': pqmass_x_prior_draws},
            commit=False
        )

        # Calculate PSNR for prior samples.
        psnr_x_prior_draws = metrics.psnr(
            x_prior_draws[:config.psnr_samples],
            x_all[:config.psnr_samples, source_index],
            max_spread=MAX_SPREAD
        )
        wandb.log(
            {f'psnr_prior_{source_index+1}': psnr_x_prior_draws},
            commit=False
        )

        if config.log_figure:
            fig = plotting_utils.show_corner(x_prior_draws)._figure
            wandb.log(
                {f'prior samples {source_index+1}': wandb.Image(fig)},
                commit=False
            )

        # Save the parameters.
        checkpoint_manager.save(source_index, params)

        # Record performance metrics.
        metrics_dict[f'div_post_{source_index+1}'] = float(divergence_x_draws)
        metrics_dict[f'div_prior_{source_index+1}'] = float(divergence_x_prior_draws)
        metrics_dict[f'pqmass_post_{source_index+1}'] = float(pqmass_x_draws)
        metrics_dict[f'pqmass_prior_{source_index+1}'] = float(pqmass_x_prior_draws)
        metrics_dict[f'psnr_post_{source_index+1}'] = float(psnr_x_draws)
        metrics_dict[f'psnr_prior_{source_index+1}'] = float(psnr_x_prior_draws)

    wandb.finish()
    return metrics_dict


def main(_):
    """Find best PCPCA parameters for each source by minimizing PCPCA loss function."""
    config = FLAGS.config

    workdir = FLAGS.workdir
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    metrics_dict = run_pcpca(config, workdir)
    print(metrics_dict)


if __name__ == '__main__':
    app.run(main)
