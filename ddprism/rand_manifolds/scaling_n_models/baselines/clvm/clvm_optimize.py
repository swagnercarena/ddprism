"Perform CLVM optuna runs with."

from functools import partial
import os

from absl import app, flags
import jax
from ml_collections import config_flags
import optuna

from clvm_train import run_clvm

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir_optuna', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config_optuna', None, 'File path to the training configuration.',
)

def objective(trial, config, workdir):
    # CLVM hyperparameters.
    latent_dim_z = trial.suggest_int(
        "latent_dim_z", config.latent_dim_min, config.latent_dim_max
    )
    latent_dim_t = trial.suggest_int(
        "latent_dim_t", config.latent_dim_min, config.latent_dim_max
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

    # Run CLVM.
    metrics = run_clvm(
        config_clvm, os.path.join(workdir, f'trial_{trial.number}')
    )
    trial.set_user_attr("trial_metrics", metrics)

    return metrics['div_post_2']


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
        study_name=study_name, storage=storage_name, direction="minimize"
    )

    # Run optuna.
    study.optimize(objective_fn, n_trials=config.n_trials, timeout=None)


if __name__ == '__main__':
    app.run(main)
