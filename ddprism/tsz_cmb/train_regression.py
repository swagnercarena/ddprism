"Train a regression model to denoise SZ maps."
import os

from absl import app, flags
from einops import rearrange
from flax import jax_utils
from flax.training import orbax_utils, train_state
import jax
import jax.numpy as jnp
from ml_collections import config_flags
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint import PyTreeCheckpointer
import optax
from tqdm import tqdm
import wandb

from ddprism import training_utils
from ddprism.tsz_cmb import embedding_models_healpix, load_datasets

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
flags.DEFINE_string('sz_path', None, 'path to noisy sz dataset.')
flags.DEFINE_string(
    'sz_no_noise_path', None, 'path to clean sz dataset without noise.'
)
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)


def create_regression_state(rng, config, healpix_shape):
    """Create train state for regression model."""
    learning_rate_fn = training_utils.get_learning_rate_schedule(
        config, config.lr_init_val, config.epochs
    )

    # Create the regression-specific HealpixTransformer model (no time conditioning)
    model = embedding_models_healpix.FlatRegressionHEALPixTransformer(
        emb_features=config.emb_features, n_blocks=config.n_blocks,
        dropout_rate_block=config.dropout_rate_block,
        heads=config.heads, patch_size_list=config.patch_size_list,
        n_average_layers=config.get('n_average_layers', 0),
        use_patch_convolution=config.get('use_patch_convolution', True),
        healpix_shape=healpix_shape
    )

    # Initialize the model
    healpix_features = healpix_shape[0] * healpix_shape[1]
    params = model.init(
        rng,
        jnp.ones((1, healpix_features)),
        jnp.ones((1, healpix_shape[0], 3))  # vec_map
    )

    # Use the configurable optimizer
    optimizer = training_utils.get_optimizer(config)(learning_rate_fn)
    grad_clip_norm = config.get('grad_clip_norm', 1.0)
    tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params['params'], tx=tx
    )


def apply_model(
    state, observed_map, sz_signal, vec_map, rng
):
    """Computes gradients and loss for a single batch."""

    def loss_fn(params):
        # Forward pass through the model
        rng_drop = rng
        x_pred = state.apply_fn(
            {'params': params}, observed_map, vec_map,
            train=True, rngs={'dropout': rng_drop}
        )

        # Compute MSE loss
        loss = jnp.mean(jnp.square(x_pred - sz_signal))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    loss = jax.lax.pmean(loss, axis_name='batch')
    grads = jax.lax.pmean(grads, axis_name='batch')

    return grads, loss


def match_filered_rmse(x_pred, sz_signal):
    """Match filter the x_post and compute the rmse."""
    filters = sz_signal / (
        jnp.sqrt(jnp.sum(jnp.square(sz_signal), axis=-2, keepdims=True))
    )
    x_pred_matched = jnp.sum(x_pred * filters) * filters
    rmse_matched = jnp.sqrt(jnp.mean(jnp.square(x_pred_matched - sz_signal)))
    return rmse_matched


def eval_model(state, observed_map, sz_signal, vec_map, params):
    """Compute predictions and loss for evaluation (no gradients)."""
    # Forward pass through the model
    x_pred = state.apply_fn(
        {'params': params}, observed_map, vec_map, train=False
    )

    # Compute MSE loss
    loss = jnp.mean(jnp.square(x_pred - sz_signal))

    # Compute metrics
    rmse = jnp.sqrt(jnp.mean(jnp.square(x_pred - sz_signal)))
    rmse_matched = match_filered_rmse(x_pred, sz_signal)

    # Average across devices
    loss = jax.lax.pmean(loss, axis_name='batch')
    rmse = jax.lax.pmean(rmse, axis_name='batch')
    rmse_matched = jax.lax.pmean(rmse_matched, axis_name='batch')

    return loss, rmse, rmse_matched


# Create pmapped functions
apply_model_pmap = jax.pmap(apply_model, axis_name='batch')
update_model = jax.pmap(training_utils.update_model, axis_name='batch')
eval_model_pmap = jax.pmap(eval_model, axis_name='batch')


def main(_):
    """Train a regression model to denoise SZ maps."""
    config = FLAGS.config
    workdir = FLAGS.workdir
    sz_path = FLAGS.sz_path
    sz_no_noise_path = FLAGS.sz_no_noise_path
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

    # Load the noisy SZ observations
    sz_obs, vec_map, _, _ = load_datasets.load_sz(config, sz_path)

    # Load the clean SZ maps (ground truth)
    sz_clean, _, _, _ = load_datasets.load_sz(config, sz_no_noise_path)

    # Flatten for training
    sz_obs_flat = rearrange(sz_obs, 'B P S (NC) -> (B P S) (NC)')
    sz_clean_flat = rearrange(sz_clean, 'B P S (NC) -> (B P S) (NC)')
    vec_map_flat = rearrange(vec_map, 'B P S N V -> (B P S) N V')

    # Shuffle all data before splitting (with fixed seed for reproducibility)
    rng_split, rng = jax.random.split(rng)
    n_total = sz_obs_flat.shape[0]
    perm_all = jax.random.permutation(rng_split, n_total)
    sz_obs_flat = sz_obs_flat[perm_all]
    sz_clean_flat = sz_clean_flat[perm_all]
    vec_map_flat = vec_map_flat[perm_all]

    # Split into training and validation sets based on n_val.
    sz_obs_train, sz_obs_val = (
        sz_obs_flat[:-config.n_val], sz_obs_flat[-config.n_val:]
    )
    sz_clean_train, sz_clean_val = (
        sz_clean_flat[:-config.n_val], sz_clean_flat[-config.n_val:]
    )
    vec_map_train, vec_map_val = (
        vec_map_flat[:-config.n_val], vec_map_flat[-config.n_val:]
    )

    # Set healpix shape
    # TODO: Hardcoded! Assumes 3 channels
    healpix_shape = (sz_obs_flat.shape[-1] // 3, 3)

    # Initialize the regression state
    rng_state, rng = jax.random.split(rng)
    state = create_regression_state(rng_state, config, healpix_shape)
    state = jax_utils.replicate(state)

    # Initialize EMA for better generalization
    ema = training_utils.EMA(jax_utils.unreplicate(state).params)

    # Training loop
    print('Beginning training.')
    n_samples = sz_obs_train.shape[0]
    n_batches = n_samples // (jax.local_device_count() * config.batch_size)
    n_val_batches = config.n_val // (
        jax.local_device_count() * config.batch_size
    )

    for epoch in tqdm(range(config.epochs), desc='Epoch'):
        # Shuffle the data
        rng_shuffle, rng = jax.random.split(rng)
        perm = jax.random.permutation(rng_shuffle, n_samples)

        for batch_idx in tqdm(range(n_batches), desc='Batch', leave=False):
            # Get batch
            start_idx = batch_idx * jax.local_device_count() * config.batch_size
            end_idx = start_idx + jax.local_device_count() * config.batch_size
            batch_indices = perm[start_idx:end_idx].reshape(
                jax.local_device_count(), config.batch_size
            )
            sz_obs_batch = sz_obs_train[batch_indices]
            sz_signal_batch = sz_clean_train[batch_indices]
            vec_map_batch = vec_map_train[batch_indices]

            # Update model.
            rng_apply, rng = jax.random.split(rng)
            rng_apply = jax.random.split(rng_apply, jax.local_device_count())
            grads, loss = apply_model_pmap(
                state, sz_obs_batch, sz_signal_batch, vec_map_batch,
                rng_apply
            )
            state = update_model(state, grads)

            # Update EMA
            ema = ema.update(
                jax_utils.unreplicate(state).params,
                config.ema_decay ** (epoch * n_batches + batch_idx + 1)
            )
            wandb.log({'loss': jax_utils.unreplicate(loss)})

        # Evaluate on validation set
        ema_params_replicated = jax_utils.replicate(ema.params)
        val_losses = []
        val_rmses = []
        val_rmses_matched = []

        for val_batch_idx in range(n_val_batches):
            # Get validation batch
            start_idx = (
                val_batch_idx * jax.local_device_count() * config.batch_size
            )
            end_idx = start_idx + jax.local_device_count() * config.batch_size
            batch_indices = jnp.arange(start_idx, end_idx).reshape(
                jax.local_device_count(), config.batch_size
            )

            sz_obs_val_batch = sz_obs_val[batch_indices]
            sz_signal_val_batch = sz_clean_val[batch_indices]
            vec_map_val_batch = vec_map_val[batch_indices]

            # Evaluate with EMA params
            val_loss, val_rmse, val_rmse_matched = eval_model_pmap(
                state, sz_obs_val_batch, sz_signal_val_batch,
                vec_map_val_batch, ema_params_replicated
            )

            val_losses.append(jax_utils.unreplicate(val_loss))
            val_rmses.append(jax_utils.unreplicate(val_rmse))
            val_rmses_matched.append(jax_utils.unreplicate(val_rmse_matched))

        # Average metrics over all validation batches
        avg_val_loss = jnp.mean(jnp.array(val_losses))
        avg_val_rmse = jnp.mean(jnp.array(val_rmses))
        avg_val_rmse_matched = jnp.mean(jnp.array(val_rmses_matched))
        metrics_dict = {
            'eval_loss': avg_val_loss,
            'eval_rmse': avg_val_rmse,
            'eval_rmse_matched': avg_val_rmse_matched
        }
        wandb.log(metrics_dict, commit=False)

        ckpt = {
            'ema_params': jax.device_get(ema.params),
            'config': config.to_dict(),
            'epoch': epoch,
            'metrics_val': jax.device_get(metrics_dict),
        }
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            epoch, ckpt, save_kwargs={'save_args': save_args}
        )

    print('Training complete.')
    checkpoint_manager.wait_until_finished()
    wandb.finish()


if __name__ == '__main__':
    app.run(main)
