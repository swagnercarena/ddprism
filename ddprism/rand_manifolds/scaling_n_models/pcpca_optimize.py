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

from ddprism import utils
from ddprism.pcpca import pcpca_utils

import load_dataset
import optuna
from optuna.trial import TrialState
from functools import partial

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)

def objective(trial, config):
    # Run the PCPCA algorithm
    # Parameters for PCPCA
    gamma = trial.suggest_float('gamma', config.gamma_min, config.gamma_max)
    latent_dim = trial.suggest_int("latent_dim", config.latent_dim_min, config.latent_dim_max)

    # Optimization hyperparameters
    n_iter = trial.suggest_int("n_iter", config.n_iter_min, config.n_iter_max, step=config.n_iter_step)
    learning_rate = trial.suggest_float("lr", config.lr_min, config.lr_max, log=True)
    lr_schedule = trial.suggest_categorical("optimizer", ["linear", "cosine"])

    # Generate our dataset.
    rng = jax.random.PRNGKey(config.rng_key)
    rng, rng_data = jax.random.split(rng)
    x_all, A_all, y_all, _ = load_dataset.get_dataset(rng_data, config)
    
    # Set regularization parameter for numerical stability.
    regularization = getattr(config, 'regularization', 1e-6)
    
    # We are optimizing for the reconstruction of the second source
    source_index = 1
    
    # In PCPCA language, y_enr is enriched observation and y_bkg is background.
    y_enr = y_all[:, source_index]
    y_bkg = y_all[:, source_index-1]
    
    enr_a_mat = A_all[:,source_index]
    bkg_a_mat = A_all[:,source_index-1]
    
    # Initialize W and log_sigma using PCA of the pseudo-inverse of the
    # observation matrix.
    rng_w, rng = jax.random.split(rng, 2)
    x_pinv = jnp.squeeze(jnp.linalg.pinv(enr_a_mat) @ y_enr[:, :, None])
    cov_empirical = jnp.cov(x_pinv, rowvar=False)
    u_mat, s_mat, _ = jnp.linalg.svd(cov_empirical)
    
    log_sigma_init = jnp.log(config.sigma_y)

    # Initialize weight matrix
    weights_init = u_mat[:, :latent_dim] * jnp.sqrt(s_mat[:latent_dim])
    weights_init += 0.01 * jax.random.normal(
    rng_w, shape=(config.feat_dim, latent_dim)
        )
    params = {'weights': jnp.asarray(weights_init), 'log_sigma': log_sigma_init}

    # Optimization loop parameters.
    if lr_schedule == 'linear':
        schedule = optax.schedules.linear_schedule(
                learning_rate, 1e-6, n_iter
            )
    elif lr_schedule == 'cosine':
        schedule = optax.schedules.cosine_decay_schedule(
                init_value=learning_rate, decay_steps=n_iter
            )
    else:
        raise ValueError(
                f'Unknown learning rate schedule: {lr_schedule}'
            )

    # Initialize Adam optimizer
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # Run the optimization loop.
    pbar = tqdm(range(n_iter), desc=f'Source {source_index + 1}')
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

    divergence_x_draws = utils.sinkhorn_divergence(
            x_post_draws[:config.sinkhorn_samples],
            x_all[:config.sinkhorn_samples, source_index]
        )
    
    trial.set_user_attr("params", params)       
    return divergence_x_draws
    
def main(_):
    """Optimize over PCPCA parameters to find the parameters for reconstructing x_2 distribution."""
    config = FLAGS.config
    rng = jax.random.PRNGKey(config.rng_key)

    workdir = FLAGS.workdir
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    # Set up checkpointing.
    checkpointer = PyTreeCheckpointer()
    checkpoint_options = CheckpointManagerOptions(
        enable_async_checkpointing=False
    )
    checkpoint_manager = CheckpointManager(
        os.path.join(workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )

    objective_fn = objective
    objective_fn = partial(objective_fn, config=config)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_fn, n_trials=config.n_trials, timeout=None)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    
    print("Best trial:")
    best_trial = study.best_trial
    
    print("  Value: ", best_trial.value)
    
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
    print("  Inferred Params for PCPCA: ")
    print("    log_sigma: ", best_trial.user_attrs['params']['log_sigma'])
    print("    weights: ", best_trial.user_attrs['params']['weights'])


    losses = jnp.zeros(len(study.trials))
    for i, t in enumerate(study.trials):
        losses = losses.at[i].set(t.values[0])
        
    sorted_losses = jnp.sort(losses)
    
    indexes = jnp.argsort(losses)
    for trial_number in indexes[:10]:
        trial = study.trials[trial_number]
        print(f'Trial {trial_number}: {trial.values[0]:.3f}')
    
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    
        print("  Inferred Params for PCPCA: ")
        print("    log_sigma: ", trial.user_attrs['params']['log_sigma'])
        print("    weights: \n", trial.user_attrs['params']['weights'])
            

    # Save the parameters.
    checkpoint_manager.save(source_index, {'best_trial': best_trial, 'trials': study.trials})
    
if __name__ == '__main__':
    app.run(main)