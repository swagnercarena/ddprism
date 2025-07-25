"""Models for encoder and decoder architectures."""

import math
from typing import Any, Callable, Mapping, Sequence, Tuple, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import Array
from einops import rearrange

class encoder_MLP(nn.Module):
    r"""Creates an encoder multi-layer perceptron (MLP) model. Assumes that the input is a flattened image.
    Arguments:
        latent_features: The number of latent features.
        hid_features: The number of hidden features.
        activation: The activation function constructor.
        bias: Whether to add a bias to the output.
    """
    latent_features: int = 2
    hid_features: list[int] = (128,)
    activation: Callable[..., nn.Module] = nn.relu
    bias: bool = True    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """MLP output.

        Arguments:
            x: Input features with shape (*, features).
            

        Returns:
            MLP output with shape (*, 2 * latent_features).
        """
        for feat in self.hid_features:
            x = nn.Dense(feat, use_bias=self.bias)(x)
            x = self.activation(x)

        # The output of the network are means and log of std for the latent dimensions.
        x = nn.Dense(2*self.latent_features, use_bias=self.bias)(x)
        return x


class decoder_MLP(nn.Module):
    r"""Creates an decoder multi-layer perceptron (MLP) model. 
    Assumes that the input is a vector of latent variables (salient and background).
    Returns flattened image.
    Arguments:
        features: The number of output features.
        hid_features: The number of hidden features.
        activation: The activation function constructor.
        bias :Whether to add a bias to the output.
    """
    features: int = 784
    hid_features: list[int] = (128,)
    activation: Callable[..., nn.Module] = nn.relu
    bias: bool = True    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """MLP output.

        Arguments:
            x: Input vector of latent variables with shape (*, 2*latent_features).

        Returns:
            MLP output with shape (*, features).
        """
        for feat in self.hid_features:
            x = nn.Dense(feat, use_bias=self.bias)(x)
            x = self.activation(x)

        x = nn.Dense(self.features, use_bias=self.bias)(x)
        return x

class cVAE(nn.Module):
    """
    """
    encoder: nn.Module
    decoder: nn.Module
    
    def setup(self):
        self.latent_dim = self.encoder.latent_features

    # TODO: include dependence on matrix A
    def encode(self, x: Array, a_mat: Optional[Array] = None) -> Tuple:
        if a_mat is not None:
            a_mat = a_mat.reshape(x.shape[0], -1)
            x = jnp.concatenate((x, a_mat.reshape(x.shape[0], -1)), axis=1)
            
        out = self.encoder(x)
        mu, log_sigma = jnp.split(out, 2, axis=-1)
        return mu, log_sigma      
        
    def decode(self, x: Array, a_mat: Optional[Array] = None) -> Array:
        out = self.decoder(x)
        if a_mat is not None:
            out = jnp.matmul(a_mat, out[..., None]).squeeze(axis=-1)
        return out
        
               
    def __call__(
        self, rng: Array, x: Array, a_mat: Optional[Array] = None
    ) -> Tuple: 
        
        # Generate new samples of the data.
        # Sample latent variables.
        mu, log_sigma = self.encode(x, a_mat)
        eps = jax.random.normal(rng, shape=x.shape[0] + (self.encoder.latent_dim,))
        z = mu + jnp.exp(log_sigma) * eps
        
        # Compute expectation for the data, given the latents.
        x = self.decode(z, a_mat)
        return x
        