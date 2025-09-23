"Perform CLVM optuna runs with."

from functools import partial
import os

from absl import app, flags
import jax
from ml_collections import config_flags
import optuna

from clvm_train import run_clvm

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir_optuna', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config_optuna', None, 'File path to the training configuration.',
)

def objective(trial, config, workdir):
    # CLVM hyperparameters.
    latent_dim_z = trial.suggest_int(
        "latent_dim_z", config.latent_z_dim_min, config.latent_z_dim_max
    )
    latent_dim_t = trial.suggest_int(
        "latent_dim_t", config.latent_t_dim_min, config.latent_t_dim_max
    )

    # Optimization hyperparameters.
    lr_init_val = trial.suggest_float(
        "lr", config.lr_min, config.lr_max, log=True
    )

    # Create a config for the trial.
    config_clvm = config.clvm_config
    config_clvm.wandb_kwargs['project']  = config.wandb_kwargs['project']
    config_clvm.wandb_kwargs['run_name'] = f'trial_{trial.number}'

    config_clvm['latent_dim_z'] = latent_dim_z
    config_clvm['latent_dim_t'] = latent_dim_t
    config_clvm['lr_init_val'] = lr_init_val

    # VAE hyperparameters (only optimize if model_type is "vae").
    if config_clvm.model_type == "vae":
        hid_features_size = trial.suggest_int(
            "vae_hid_features", config.vae_hid_features_min,
            config.vae_hid_features_max
        )
        normalize = trial.suggest_categorical(
            "vae_normalize", config.vae_normalize_choices
        )
        activation = trial.suggest_categorical(
            "vae_activation", config.vae_activation_choices
        )

        # Update VAE config with suggested values
        # Keep 3-layer architecture but vary the hidden size
        config_clvm.vae['hid_features'] = (
            hid_features_size, hid_features_size, hid_features_size
        )
        config_clvm.vae['normalize'] = normalize
        config_clvm.vae['activation'] = activation

    # Run CLVM.
    os.makedirs(os.path.join(workdir, f'trial_{trial.number}'), exist_ok=True)

    try:
        metrics = run_clvm(
            config_clvm, os.path.join(workdir, f'trial_{trial.number}')
        )
        trial.set_user_attr("trial_metrics", metrics)
        trial.set_user_attr("trial_failed", False)
        return float(metrics['mnist_fcd_post'])

    except Exception as e:
        # Log the error and return a large penalty value
        print(f"Trial {trial.number} failed with error: {str(e)}")
        trial.set_user_attr("trial_failed", True)
        trial.set_user_attr("trial_error", str(e))
        trial.set_user_attr("trial_metrics", None)

        # Return a large penalty value (since we're minimizing)
        # This tells Optuna this parameter combination was very bad
        return float('inf')


def main(_):
    """Optimize over CLVM parameters.

    Notes:
        - The objective function returns the Sinkhorn divergence for source 2.
    """
    config = FLAGS.config_optuna

    workdir = FLAGS.workdir_optuna
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    # Define objective function
    objective_fn = partial(objective, config=config, workdir=workdir)

    # Create a new study.
    study_name = config.wandb_kwargs.get('project')
    storage_name = f"sqlite:///{workdir}/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction="minimize",
        load_if_exists=True
    )

    # Run optuna.
    study.optimize(objective_fn, n_trials=config.n_trials, timeout=None)


if __name__ == '__main__':
    app.run(main)