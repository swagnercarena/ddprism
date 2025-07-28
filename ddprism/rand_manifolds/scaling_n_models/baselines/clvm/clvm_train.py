"Perform linear CLVM analysis of missing data with n_sources."
import os

from absl import app, flags
from flax.training import train_state
import jax
import jax.numpy as jnp
from ml_collections import config_flags
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm
import wandb

from ddprism import training_utils
from ddprism.clvm import clvm_utils
from ddprism.metrics import metrics
from ddprism.rand_manifolds.random_manifolds import MAX_SPREAD
from ddprism.rand_manifolds.scaling_n_models import load_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)


@jax.jit
def train_step(state, rng, enr_obs, bkg_obs, a_mat_enr, a_mat_bkg, other_vars):
    """Perform a single training step."""
    def loss_fn(params):
        # Collect the relevant variables.
        variables = {'params': params, 'variables': other_vars}
        rng_bkg, rng_enr = jax.random.split(rng)

        # Enriched observation loss
        enr_loss = state.apply_fn(
            variables, rng_enr, enr_obs, a_mat_enr, method='loss_enr_obs'
        )
        # Background observation loss
        bkg_loss = state.apply_fn(
            variables, rng_bkg, bkg_obs, a_mat_bkg, method='loss_bkg_obs'
        )
        return enr_loss + bkg_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def get_posterior_samples(
    rng, state, enr_obs, a_mat_enr, other_vars
):
    """Get posterior samples from the CLVM model."""
    # Collect the relevant variables.
    variables = {'params': state.params, 'variables': other_vars}

    # Encode the enriched observations to get posterior parameters
    latents = state.apply_fn(
        variables, enr_obs, a_mat_enr, method='encode_enr_obs'
    )
    latent_draw = state.apply_fn(
        variables, rng, latents[0], latents[1], method='_latent_draw'
    )
    z_latent, t_latent = state.apply_fn(
        variables, latent_draw, method='_latent_split'
    )
    signal_feat = state.apply_fn(
        variables, t_latent, method='decode_signal_feat'
    )
    bkg_feat = state.apply_fn(
        variables, z_latent, method='decode_bkg_feat'
    )

    return bkg_feat, signal_feat


def get_prior_samples(rng, state, num_samples, other_vars):
    """Get prior samples from the CLVM model."""
    # Collect the relevant variables.
    variables = {'params': state.params, 'variables': other_vars}

    # Sample from prior (standard normal)
    latent_draw = state.apply_fn(
        variables, rng, num_samples, method='_latent_draw_prior'
    )
    z_latent, t_latent = state.apply_fn(
        variables, latent_draw, method='_latent_split'
    )
    signal_feat = state.apply_fn(
        variables, t_latent, method='decode_signal_feat'
    )
    bkg_feat = state.apply_fn(
        variables, z_latent, method='decode_bkg_feat'
    )

    return bkg_feat, signal_feat


def run_clvm(config, workdir):
    """Run linear CLVM analysis of missing data on 1D manifolds.

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

    for source_index in range(1, config.n_sources):

        # Initialize the lienar CLVM model.
        clvm_model = clvm_utils.CLVMLinear(
            features=x_all.shape[-1],
            latent_dim_z=config.latent_dim_z,
            latent_dim_t=config.latent_dim_t,
            obs_dim=y_all.shape[-1]
        )

        # Enriched observations (target)
        enr_obs = y_all[:, source_index]
        a_mat_enr = A_all[:, source_index]
        # Background observations (previous source)
        bkg_obs = y_all[:, source_index - 1]
        a_mat_bkg = A_all[:, source_index - 1]

        # Initialize model parameters.
        rng, rng_init = jax.random.split(rng)
        dummy_obs = jax.random.normal(rng_init, (1, y_all.shape[-1]))
        dummy_a_mat = jax.random.normal(
            rng_init, (1, y_all.shape[-1], x_all.shape[-1])
        )
        variables = clvm_model.init(
            rng_init, rng, dummy_obs, dummy_a_mat, method='loss_enr_obs'
        )
        other_vars = {'log_sigma_obs': jnp.log(config.sigma_y)}

        # Set up our training state.
        learning_rate_fn = training_utils.get_learning_rate_schedule(
            config, config.lr_init_val, config.epochs * config.batch_size
        )
        optimizer = training_utils.get_optimizer(config)(learning_rate_fn)
        state = train_state.TrainState.create(
            apply_fn=clvm_model.apply,
            params=variables['params'],
            tx=optimizer
        )

        # Training loop
        dataset_size = enr_obs.shape[0]
        steps_per_epoch = dataset_size // config.batch_size

        for _ in tqdm(range(config.epochs), desc='Training CLVM'):
            for _ in range(steps_per_epoch):
                # Get random batch
                rng, rng_batch = jax.random.split(rng)
                batch_indices = jax.random.choice(
                    rng_batch, dataset_size, (config.batch_size,), replace=False
                )

                # Training step
                rng, rng_train = jax.random.split(rng)
                state, loss = train_step(
                    state, rng_train,
                    enr_obs[batch_indices], bkg_obs[batch_indices],
                    a_mat_enr[batch_indices], a_mat_bkg[batch_indices],
                    other_vars
                )

                # Log loss
                wandb.log({'loss': loss})

        # Generate posterior samples.
        rng, rng_post = jax.random.split(rng)
        _, post_samples = get_posterior_samples(
            rng_post, state, enr_obs, a_mat_enr, other_vars
        )

        # Generate prior samples
        rng, rng_prior = jax.random.split(rng)
        n_samples = max(
            config.sinkhorn_samples, config.pqmass_samples, config.psnr_samples
        )
        _, prior_samples = get_prior_samples(
            rng_prior, state, n_samples, other_vars
        )

        # Compute metrics
        metrics_dict = {}

        # Sinkhorn divergence for posterior and prior samples.
        div_post = metrics.sinkhorn_divergence(
            post_samples[:config.sinkhorn_samples],
            x_all[:config.sinkhorn_samples, source_index]
        )
        div_prior = metrics.sinkhorn_divergence(
            prior_samples[:config.sinkhorn_samples],
            x_all[:config.sinkhorn_samples, source_index]
        )
        metrics_dict[f'div_post_{source_index}'] = float(div_post)
        metrics_dict[f'div_prior_{source_index}'] = float(div_prior)

        # PQ mass
        pqmass_post = metrics.pq_mass(
            post_samples[:config.pqmass_samples],
            x_all[:config.pqmass_samples, source_index]
        )
        pqmass_prior = metrics.pq_mass(
            prior_samples[:config.pqmass_samples],
            x_all[:config.pqmass_samples, source_index]
        )
        metrics_dict[f'pqmass_post_{source_index}'] = float(pqmass_post)
        metrics_dict[f'pqmass_prior_{source_index}'] = float(pqmass_prior)

        # PSNR
        psnr_post = metrics.psnr(
            post_samples[:config.psnr_samples],
            x_all[:config.psnr_samples, source_index],
            max_spread=MAX_SPREAD
        )
        metrics_dict[f'psnr_post_{source_index}'] = float(psnr_post)

        # Save checkpoint
        ckpt = {
            'params': jax.device_get(state.params), 'config': config.to_dict(),
            'variables': jax.device_get(other_vars),
            'metrics': jax.device_get(metrics_dict),
            'x_post': jax.device_get(post_samples),
            'x_prior': jax.device_get(prior_samples)
        }
        checkpoint_manager.save(source_index, ckpt)

        # Log metrics
        wandb.log(metrics_dict, commit=False)

    # Close checkpoint manager after all sources are processed
    checkpoint_manager.close()
    wandb.finish()
    return metrics_dict


def main(_):
    """Run linear CLVM analysis for missing data."""
    config = FLAGS.config
    workdir = FLAGS.workdir
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    metrics_dict = run_clvm(config, workdir)
    print(metrics_dict)


if __name__ == '__main__':
    app.run(main)
