"Perform CLVM analysis of missing data with n_sources."
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
from ddprism.clvm import clvm_utils, models
from ddprism import plotting_utils
from ddprism.rand_manifolds.random_manifolds import MAX_SPREAD
from ddprism.rand_manifolds.scaling_n_models import load_dataset

import optax
from flax import linen as nn
from flax.training import train_state, orbax_utils

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)

@jax.jit
def update_model(state, grads):
    """Update model with gradients."""
    return state.apply_gradients(grads=grads)

@jax.jit
def apply_model(rng, state, x, y, a_mat_x, a_mat_y, sigma_noise):
    """Computes gradients and loss for a single batch."""

    # loss
    def loss_fn(params):
        # Draw samples in data space.
        x_draws, y_draws, latent_params = state.apply_fn(
            {'params': params}, rng, x, y, a_mat_x, a_mat_y,
        )
        mu_tx, log_sigma_tx, mu_zx, log_sigma_zx, mu_zy, log_sigma_zy = latent_params
        
        # Compute loss function.
        # Reconstruction loss.
        loss = (optax.losses.squared_error(x, x_draws).sum(axis=-1) / 2 / sigma_noise**2)
        loss += (optax.losses.squared_error(y, y_draws).sum(axis=-1) / 2 / sigma_noise**2)
        loss = loss.mean()
        
        # Prior loss.
        kl_div = ((mu_tx**2 + jnp.exp(log_sigma_tx*2) - 2*log_sigma_tx) / 2).sum(axis=-1)
        kl_div += ((mu_zx**2 + jnp.exp(log_sigma_zx*2) - 2*log_sigma_zx) / 2).sum(axis=-1)
        kl_div += ((mu_zy**2 + jnp.exp(log_sigma_zy*2) - 2*log_sigma_zy) / 2).sum(axis=-1)
        
        kl_div = kl_div.mean()
        
        loss = loss + kl_div
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn,)
    loss, grads = grad_fn(state.params)

    return grads, loss

def run_clvm(config, workdir):
    """Run CLVM analysis of missing data on 1D manifolds.

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

    # Run CLVM analysis for each source.
    metrics_dict = {}
    for source_index in range(config.n_sources):
        # In CLVM language, tg is  target observation and bkg is background observation.
        tg  = y_all[:, source_index].reshape(-1, config.obs_dim)
        a_mat_tg  = A_all[:, source_index]

        if source_index == 0:
            bkg = jnp.zeros_like(tg)
            a_mat_bkg = jnp.zeros_like(a_mat_tg)
        else:
            bkg = y_all[:, source_index - 1].reshape(-1, config.obs_dim)
            a_mat_bkg = A_all[:, source_index - 1]

        tg_dset_size = tg.shape[0]
        bkg_dset_size = bkg.shape[0]

        # Define CVAE models for target and background latents.
        hidden_features = (config.hidden_feats_per_layer,)*config.hidden_layers
        
        encoder_tg = models.encoder_MLP(config.latent_dim_tg, hid_features=hidden_features)
        decoder_tg = models.decoder_MLP(config.feat_dim, hid_features=hidden_features)
        cvae_tg = models.cVAE(encoder_tg, decoder_tg)
        
        encoder_bkg = models.encoder_MLP(config.latent_dim_bkg, hid_features=hidden_features)
        decoder_bkg = models.decoder_MLP(config.feat_dim, hid_features=hidden_features)
        cvae_bkg = models.cVAE(encoder_bkg, decoder_bkg)

        # Define CLVM model.
        clvm_model = clvm_utils.CLVMVAE(features=config.feat_dim, 
                                        latent_dim_z=config.latent_dim_bkg, 
                                        latent_dim_t=config.latent_dim_tg, 
                                        obs_dim=config.obs_dim,
                                        enr_decoder=decoder_tg, 
                                        bkd_decoder=decoder_bkg, 
                                        enr_encoder=encoder_tg, 
                                        bkd_encoder=encoder_bkg) #cvae_bkg, cvae_tg)
        # Initialize model parameters.
        rng, rng_state = jax.random.split(rng, 2)
        '''
        model = clvm_utils.CLVMLinear(
            features=config.feat_dim,
            latent_dim_z=config.latent_dim_bkg,
            latent_dim_t=config.latent_dim_tg,
            obs_dim=config.obs_dim
        )
        dummy_obs = jax.random.normal(rng, (config.obs_dim,))
        # Initialize the model
        variables = model.init(
            rng_state, rng, dummy_obs, method=model.loss_enr_obs
        )
        '''
        
        dummy_obs = jax.random.normal(rng, (1, config.obs_dim,))
        params_clvm = clvm_model.init(rng_state, 
                                      rng, dummy_obs, 
                                      #{'variables': {'a_mat': jnp.zeros((7, config.obs_dim, config.feat_dim))}},
                                      method=clvm_model.loss_enr_obs)
        print(params_clvm['variables'])
        '''
        params_clvm = clvm_model.init(
                                    rng, rng_state, 
                                    jnp.ones((1, config.obs_dim)), jnp.ones((1,  config.obs_dim)),
                                    jnp.ones((1, 1, config.obs_dim, config.feat_dim)), 
                                    jnp.ones((1, 1, config.obs_dim, config.feat_dim))
                                )
        
        # Optimization loop parameters.
        steps_per_epoch = tg_dset_size // config.batch_size
        
        if config.lr_schedule == 'linear':
            learning_rate_fn = optax.schedules.linear_schedule(
                config.lr_init_val, config.lr_end_val, config.epochs*steps_per_epoch
            )
        elif config.lr_schedule == 'cosine':
            learning_rate_fn = optax.schedules.cosine_decay_schedule(
                init_value=config.lr_init_val, decay_steps=config.epochs*steps_per_epoch
            )
        else:
            raise ValueError(
                f'Unknown learning rate schedule: {config.lr_schedule}'
            )
            
        tx = optax.adam(learning_rate=learning_rate_fn)

        # Initialize training state.
        state = train_state.TrainState.create(apply_fn=clvm_model.apply, 
                                              params=params_clvm['params'], 
                                              tx=tx)

        # Run the optimization loop.
        losses_per_epoch = []
        pbar = tqdm(range(config.epochs), desc=f'Source {source_index + 1}')
        
        for epoch in range(config.epochs):
            losses = []
            for step in range(steps_per_epoch):
                # Get a random batch.
                rng_epoch, rng_tg, rng_bkg, rng = jax.random.split(rng, 4)
                batch_tg = jax.random.randint(rng_tg, shape=(config.batch_size,), minval=0, maxval=tg_dset_size)
                batch_bkg = jax.random.randint(rng_bkg, shape=(config.batch_size,), minval=0, maxval=bkg_dset_size)
        
                # Compute gradients and losses.
                grads, loss = apply_model(
                    rng_epoch, state, tg[batch_tg], bkg[batch_bkg], 
                    a_mat_tg[batch_tg], a_mat_bkg[batch_bkg],
                    sigma_noise=config.sigma_y
                )
                state = update_model(
                    state, grads
                )
                losses.append(loss)
                
                # Log our loss.
                wandb.log(
                    {f'loss {source_index + 1}': loss},
                )
                
                
            losses_per_epoch.append(losses)
            pbar.set_postfix({'Loss': f'{jnp.asarray(losses).mean():.3f}'})

        # Draw samples from the posterior.
        rng_post, rng = jax.random.split(rng)
        x_post_draws = state.apply_fn(
            {'params': state.params},
            rng_post, tg, a_mat_tg,
            method="denoise_samples"
        )
        # Log a figure with new posterior samples.
        if config.log_figure:
            fig = plotting_utils.show_corner(x_post_draws)._figure
            wandb.log(
                {f'posterior samples {source_index+1}': wandb.Image(fig)},
                commit=False
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


        # Draw samples from the prior.
        rng_prior, rng = jax.random.split(rng)
        x_prior_draws = state.apply_fn(
            {'params': state.params}, 
            rng_prior, shape=(tg_dset_size,),
            method="draw_prior_samples"
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
        
        if config.log_figure:
            fig = plotting_utils.show_corner(x_prior_draws)._figure
            wandb.log(
                {f'prior {source_index+1}': wandb.Image(fig)},
                commit=False
            )

        # Record performance metrics.
        metrics_dict[f'div_post_{source_index+1}'] = float(divergence_x_draws)
        metrics_dict[f'div_prior_{source_index+1}'] = float(divergence_x_prior_draws)
        metrics_dict[f'pqmass_post_{source_index+1}'] = float(pqmass_x_draws)
        metrics_dict[f'pqmass_prior_{source_index+1}'] = float(pqmass_x_prior_draws)
        metrics_dict[f'psnr_post_{source_index+1}'] = float(psnr_x_draws)


        rng_samples, rng = jax.random.split(rng)
        background_denoised = state.apply_fn({'params': state.params}, 
                                             rng_samples, bkg, a_mat_bkg, 
                                             dset='background', method="denoise_samples")
        if config.log_figure_bkg:
            fig = plotting_utils.show_corner(background_denoised)._figure
            wandb.log(
                {f'background {source_index+1}': wandb.Image(fig)},
                commit=False
            )

            fig = plotting_utils.show_corner(x_all[:, source_index])._figure
            wandb.log(
                {f'true dist {source_index+1}': wandb.Image(fig)},
                commit=False
            )

        # TODO: checkpoints
        # Save the state parameters, losses, and metrics for each source.
        ckpt = { 
                'params': state.params, 
                'losses': jnp.array(losses_per_epoch), 
                f'metrics_{source_index+1}': metrics_dict
                    
                }
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(source_index+1, ckpt, save_kwargs={'save_args': save_args})

    checkpoint_manager.close()  
    '''
    wandb.finish()
    return metrics_dict
        
  
def main(_):
    """Run CLVM analysis with CVAE for each source by minimizing ELBO loss function."""
    config = FLAGS.config

    workdir = FLAGS.workdir
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    metrics_dict = run_clvm(config, workdir)
    print(metrics_dict)


if __name__ == '__main__':
    app.run(main)
