"""Models for CLVM encoder and decoder."""
from typing import Callable, Tuple

from flax import linen as nn
import jax.numpy as jnp
from jax import Array
from einops import rearrange

class EncoderMLP(nn.Module):
    r"""Creates an encoder MLP that assumes flattened features.

    Arguments:
        latent_features: The number of latent features.
        hid_features: The number of hidden features.
        activation: The activation function constructor.
        normalize: Whether features are normalized between layers or not.
        dropout_rate: Dropout rate for regularization. Default is 0.0.
    """
    latent_features: int
    hid_features: Tuple[int, ...] = (64, 64)
    activation: Callable[[Array], Array] = nn.silu
    normalize: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def encode_feat(self, x: Array, train: bool = True) -> Array:
        """Encode the input features into the latent distribution.

        Arguments:
            x: Input features with shape (*, features).
            train: Whether in training mode for dropout.

        Returns:
            Latent mean and variance deviation for the input features.
        """
        # Process through hidden layers
        for feat in self.hid_features:

            # Dense layer
            x = nn.Dense(feat)(x)

            # Activation
            x = self.activation(x)

            if self.normalize:
                x = nn.LayerNorm()(x)

            # Dropout
            if self.dropout_rate > 0.0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        # Final output layer
        x = jnp.split(nn.Dense(self.latent_features * 2)(x), 2, axis=-1)
        return x[0], jnp.exp(x[1])

    @nn.compact
    def encode_obs(self, x: Array, a_mat: Array) -> Array:
        """Encode the input observations into the latent distribution.
        """
        a_mat = rearrange(a_mat, 'K L M -> K (L M)')
        x = jnp.concatenate([x, a_mat], axis=-1)
        return self.encode_feat(x)


class DecoderMLP(nn.Module):
    r"""Creates an decoder MLP that assumes flattened features.

    Arguments:
        features: The number of output features.
        hid_features: The number of hidden features.
        activation: The activation function constructor.
        normalize: Whether features are normalized between layers or not.
        dropout_rate: Dropout rate for regularization. Default is 0.0.
    """
    features: int
    hid_features: Tuple[int, ...] = (64, 64)
    activation: Callable[[Array], Array] = nn.silu
    normalize: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def decode_feat(self, x: Array, train: bool = True) -> Array:
        """Decode the latent variables into the feature space.

        Arguments:
            x: Input features with shape (*, latent_features).
            train: Whether in training mode for dropout.

        Returns:
            Decoded features with shape (*, features).
        """
        # Process through hidden layers
        for feat in self.hid_features:

            # Dense layer
            x = nn.Dense(feat)(x)

            # Activation
            x = self.activation(x)

            if self.normalize:
                x = nn.LayerNorm()(x)

            # Dropout
            if self.dropout_rate > 0.0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        # Final output layer
        return nn.Dense(self.features)(x)
