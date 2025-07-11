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
from pcpca_train import run_pcpca

import optuna
from optuna.trial import TrialState
from functools import partial

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
    
flags.DEFINE_string('workdir_optuna', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config_optuna', None, 'File path to the training configuration.',
)

def objective(trial, config, workdir):
    # Run the PCPCA algorithm
    # Parameters for PCPCA.
    gamma = trial.suggest_float('gamma', config.gamma_min, config.gamma_max)
    latent_dim = trial.suggest_int("latent_dim", config.latent_dim_min, config.latent_dim_max)

    # Optimization hyperparameters.
    n_iter = trial.suggest_int("n_iter", config.n_iter_min, config.n_iter_max, step=config.n_iter_step)
    learning_rate = trial.suggest_float("lr", config.lr_min, config.lr_max, log=True)
    lr_schedule = trial.suggest_categorical("optimizer", ["linear", "cosine"])

    # Run the PCPCA analysis for a given trial.
    config_pcpca = config.pcpca_config
    config_pcpca.wandb_kwargs['project']  = config.wandb_kwargs['project']
    config_pcpca.wandb_kwargs['run_name'] = f'trial_{trial.number}'
    config_pcpca['gamma']=gamma
    config_pcpca['latent_dim']=latent_dim
    config_pcpca['n_iter']=n_iter
    config_pcpca['learning_rate']=learning_rate
    config_pcpca['lr_schedule']=lr_schedule

    metrics = run_pcpca(config_pcpca, workdir, metrics=True)
    trial.set_user_attr("trial_metrics", metrics)
    
    return metrics[f'div_post_2']

def main(_):
    """Optimize over PCPCA parameters to find the parameters for reconstructing source 2 distribution."""
    config = FLAGS.config_optuna
    
    workdir = FLAGS.workdir_optuna
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    # Define objective function
    objective_fn = objective
    objective_fn = partial(objective_fn, config=config, workdir=workdir)

    # Create a new study.
    study_name = config.wandb_kwargs.get('project')  
    storage_name = f"sqlite:///{workdir}/{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize")

    # Run optuna.
    study.optimize(objective_fn, n_trials=config.n_trials, timeout=None)
    
if __name__ == '__main__':
    app.run(main)