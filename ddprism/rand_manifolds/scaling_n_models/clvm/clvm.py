import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
from absl import app, flags

from tensorflow.keras.datasets import mnist

import math, os
from typing import Any, Callable, Mapping, Sequence, Tuple, Optional
from jax import Array
from flax import linen as nn
from flax.training import train_state, orbax_utils
import optax
import wandb
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointer
from pathlib import Path
import tqdm
from einops import rearrange

from ddprism import training_utils
from ddprism.corrupted_mnist import datasets
from ddprism.metrics import metrics, image_metrics

class clvmLinear(nn.Module):
    r"""Creates an instance of cLVM model that linearly maps latent variables to the observed space.
    """

    def sample_latents(
        self, rng: Array, model_params: Tuple, x: Array, y: Array, a_mat: Optional[Array] = None
    ) -> Tuple:
        pass

    def expect_data(
        self, model_params: Tuple, tx: Array, zx: Array, zy: Array, a_mat: Optional[Array] = None
    ) -> Tuple:
        
        w_mat, s_mat, mu_x, mu_y = model_params
        # Compute expected values of the target and background datasets.
        x = s_mat @ zx + w_mat @ tx + mu_x
        y = s_mat @ zy + mu_y

        if a_mat is not None:
            x = a_mat @ x
            y = a_mat @ y
        return x, y

    @nn.compact
    def __call__(
        self, rng: Array, x: Array, y: Array, a_mat: Optional[Array] = None
    ) -> Tuple: 
        # Following GaussianDenoiserDPLR model?
        # Initialize model parameters.
        mu_x = self.param(
            "mu_x", lambda rng, shape: jnp.ones(shape), x.shape[-1:]
        )

        mu_y = self.param(
            "mu_x", lambda rng, shape: jnp.ones(shape), y.shape[-1:]
        )
        
        s_mat = self.param(
            "s_mat",
            lambda rng, shape: ...,
            (..., )
        )

        w_mat = self.param(
            "w_mat",
            lambda rng, shape: ...,
            (..., )
        )

        model_params = (w_mat, s_mat, mu_x, mu_y)
        
        # Generate new samples of the data.
        # Sample latent variables.
        tx, zx, zy = self.sample_latents(rng, model, x, y, a_mat)
        
        # Compute expectation for the data, given the latents.
        x, y = self.expect_data(model_params, tx, zx, zy, a_mat)
        return x, y

class clvmVAE(clvmLinear):
    r"""Creates an instance of cLVM model that non-linearly maps latent variables to the observed space.
    """
    vae_z: nn.Module
    vae_t: nn.Module

    def sample_latents(
        self, rng: Array, x: Array, y: Array, 
        a_mat_x: Optional[Array] = None, a_mat_y: Optional[Array] = None
    ) -> Tuple:
        # Sample latent variables for the target and background datasets.
        rng_zx, rng_zy, rng_tx = jax.random.split(rng, 3)
        
        # Compute mean and std for the target and background datasets.
        mu_tx, log_sigma_tx = self.vae_t.encode(x, a_mat_x)

        mu_zx, log_sigma_zx = self.vae_z.encode(x, a_mat_x)
        mu_zy, log_sigma_zy = self.vae_z.encode(y, a_mat_y)
        
        # Sample latent variables corresponding to the enriched signal in the target dataset.
        eps_tx = jax.random.normal(rng_tx, shape=x.shape[:-1] + (self.vae_t.latent_dim,))
        tx = mu_tx + jnp.exp(log_sigma_tx) * eps_tx

        # Sample latent variables corresponding to the background in the target dataset.
        eps_zx = jax.random.normal(rng_zx, shape=x.shape[:-1] + (self.vae_z.latent_dim,))
        zx = mu_zx + jnp.exp(log_sigma_zx) * eps_zx

        # Sample latent variables corresponding to the background in the background dataset.
        eps_zy = jax.random.normal(rng_zy, shape=y.shape[:-1] + (self.vae_z.latent_dim,))
        zy = mu_zy + jnp.exp(log_sigma_zy) * eps_zy

        latent_params = (mu_tx, log_sigma_tx, mu_zx, log_sigma_zx, mu_zy, log_sigma_zy)
        return tx, zx, zy, latent_params

    def expect_data(
        self, tx: Array, zx: Array, zy: Array, 
        a_mat_x: Optional[Array] = None, a_mat_y: Optional[Array] = None
    ) -> Tuple:
        
        # Compute expected values of the target and background datasets.
        x = self.vae_z.decode(zx, a_mat_x) + self.vae_t.decode(tx, a_mat_x)
        y = self.vae_z.decode(zy, a_mat_y)

        '''
        if a_mat is not None:
            x = a_mat_x @ x
            y = a_mat_y @ y
        '''
        return x, y
    '''
    def denoise_samples(
        self, rng: Array, x: Array, a_mat_x: Optional[Array] = None, 
    ) -> Array:
        
        rng_tx, rng = jax.random.split(rng, 2)
        
        # Compute mean and std for the target signal.
        mu_tx, log_sigma_tx = self.vae_t.encode(x, a_mat_x)
        
        # Sample latent variables corresponding to the enriched signal in the target dataset.
        eps_tx = jax.random.normal(rng_tx, shape=x.shape[:-1] + (self.vae_t.latent_dim,))
        tx = mu_tx + jnp.exp(log_sigma_tx) * eps_tx

        # Compute expected values of the target signal; 
        # a_mat_x is not applied since we are interested in the underlying signal.
        x_denoised = self.vae_t.decode(tx,)
        return x_denoised
    '''
    
    def denoise_samples(
        self, rng: Array, x: Array, a_mat_x: Optional[Array] = None, dset = 'target'
    ) -> Array:
        
        rng_tx, rng = jax.random.split(rng, 2)
        
        if dset == 'target':
            vae = self.vae_t
        elif dset == 'background':
            vae = self.vae_z
        
        # Compute mean and std for the target signal.
        mu_tx, log_sigma_tx = vae.encode(x, a_mat_x)
        
        # Sample latent variables corresponding to the enriched signal in the target dataset.
        eps_tx = jax.random.normal(rng_tx, shape=x.shape[:-1] + (vae.latent_dim,))
        tx = mu_tx + jnp.exp(log_sigma_tx) * eps_tx

        # Compute expected values of the target signal; 
        # a_mat_x is not applied since we are interested in the underlying signal.
        x_denoised = vae.decode(tx,)
        return x_denoised

    @nn.compact
    def __call__(
        self, rng: Array, x: Array, y: Array, 
        a_mat_x: Optional[Array] = None, a_mat_y: Optional[Array] = None
    ) -> Tuple: 
        
        # Generate new samples of the data.
        # Sample latent variables.
        tx, zx, zy, latent_params = self.sample_latents(rng, x, y, a_mat_x, a_mat_y)
        
        # Compute expectation for the data, given the latents.
        x, y = self.expect_data(tx, zx, zy, a_mat_x, a_mat_y)
        return x, y, latent_params
    