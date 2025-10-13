"""Embedding models for HEALPix CMB data."""

from typing import Sequence

import jax.numpy as jnp
from jax import Array
from flax import linen as nn

from ddprism.embedding_models import AdaLNZeroModulation


PERIOD = 1e-1


class RelativeBias(nn.Module):
    """Relative bias from vector direction map.

    Arguments:
        n_heads: Number of heads in the attention mechanism.
    """
    n_heads: int
    freq_features: int = 64

    @nn.compact
    def __call__(self, vec_map:Array) -> Array:
        """Call the relative bias.

        Arguments:
            vec_map: Vector direction map with shape (*, N, 3).
        """
        # Get the angular distance between all pairs of vectors.
        dot = jnp.einsum('...NK,...MK->...NM', vec_map, vec_map)
        angle = jnp.arccos(jnp.clip(dot,0.0,1.0)) / 1e-3

        # Get the embedded distance between all pairs of vectors.
        freqs = jnp.linspace(0, 1, self.freq_features // 2)
        freqs = jnp.asarray((1 / PERIOD) * freqs)
        pos = jnp.concatenate(
            (
                jnp.sin(freqs * angle[..., None]),
                jnp.cos(freqs * angle[..., None])
            ), axis=-1
        )
        embedded_angle = nn.Dense(self.n_heads)(pos)

        return embedded_angle


class HEALPixAttention(nn.Module):
    """Attention for HEALPix CMB data.

    Arguments:
        emb_dim: Dimension of the embedding.
        n_heads: Number of heads in the attention mechanism.
        dropout_rate: Dropout rate.
        use_bias: If true, bias will be included in the attention mechanism.
    """
    emb_dim: int
    n_heads: int
    dropout_rate: float
    use_bias: bool

    @nn.compact
    def __call__(self, x:Array, vec_map:Array, train:bool=True) -> Array:
        """Call the attention mechanism.

        Arguments:
            x: Input map with shape (*, N, D).
            vec_map: Vector direction map with shape (*, N, 3).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, N, D).
        """
        head_dim = self.emb_dim // self.n_heads
        assert head_dim * self.n_heads == self.emb_dim

        # Multihead attention.
        batch_dim = x.shape[:-2]
        N = x.shape[2]
        qkv = nn.Dense(3 * self.emb_dim)(x)
        qkv = qkv.reshape(*batch_dim, N, self.n_heads, 3, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=-2)

        # Compute attention.
        attention = jnp.einsum('...QHT,...KHT->...QKH', q, k)
        attention = attention / jnp.sqrt(head_dim)
        # Make sure softmax is on the key dimension.
        attention -= jnp.max(attention, axis=-2, keepdims=True)
        attention = nn.softmax(attention, axis=-2)
        attention = nn.Dropout(self.dropout_rate)(
            attention, deterministic=not train
        )

        # Add relative bias to attention.
        relative_bias = RelativeBias(self.n_heads)(vec_map)
        attention = attention + relative_bias

        # Attend with relative bias.
        y = jnp.einsum('...QKH,...KHT->...QHT', attention, v)
        y = y.reshape(*batch_dim, N, self.emb_dim)
        y = nn.Dense(self.emb_dim)(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=not train)
        return y


class HEALPixAttentionBlock(nn.Module):
    """Transformer block for HEALPix CMB data.

    Arguments:
        emb_dim: Dimension of the embedding.
        n_heads: Number of heads in the attention mechanism.
        time_emb_dim: Dimension of the time embedding.
        dropout_rate: Dropout rate.
    """
    emb_dim: int
    n_heads: int
    time_emb_dim: int
    qkv_dim: int
    dropout_rate: float
    mlp_ratio: int = 4
    use_bias: bool = True

    @nn.compact
    def __call__(
        self, x:Array, t:Array, vec_map:Array, train:bool=True
    ) -> Array:
        """Call the attention block.

        Arguments:
            x: Input map with shape (*, N, D).
            t: Time embedding with shape (*, E).
            vec_map: Vector direction map with shape (*, N, 3).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, N, D).
        """
        # Perform adaLN-Zero modulation to condition on t.
        gamma_one, beta_one, alpha_one = (
            AdaLNZeroModulation(self.emb_dim, self.time_emb_dim)(t)
        )
        y = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        y = (gamma_one + 1) * y + beta_one

        y = HEALPixAttention(
            self.emb_dim, self.n_heads, self.dropout_rate, self.use_bias
        )(y, vec_map, train=train)

        # Last step of adaLN-Zero modulation.
        x = x + (alpha_one * y / jnp.sqrt(1 + alpha_one ** 2))

        # Pointwise MLP.
        gamma_two, beta_two, alpha_two = (
            AdaLNZeroModulation(self.emb_dim, self.time_emb_dim)(t)
        )
        y = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        y = (gamma_two + 1) * y + beta_two

        # MLP for pointwise feedforward.
        y = nn.Dense(self.emb_dim * self.mlp_ratio)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.emb_dim)(y)

        # Last step of adaLN-Zero modulation.
        x = x + (alpha_two * y / jnp.sqrt(1 + alpha_two ** 2))

        return x


class HEALPixTransformer(nn.Module):
    r"""Creates a time conditioned HEALPix transformer.

    Arguments:
        emb_dim: Dimension of the embedding.
        n_blocks: Number of transformer blocks.
        dropout_rate_block: Dropout rate for each transformer block.
        heads: Number of heads in the attention mechanism.
        patch_size: Size of the patch to divide the input map into.
        emb_features: Size of the embedding vector that encodes the time
            features.
    """
    emb_dim: int
    n_blocks: int
    dropout_rate_block: Sequence[float]
    heads: int
    patch_size: int
    time_emb_dim: int

    @nn.compact
    def __call__(
        self, x: Array, t: Array, vec_map: Array, train: bool = True
    ) -> Array:
        """Return the conditioned HEALPix transformer output.

        Arguments:
            x: Input noised map with shape (*, N, C).
            t: Time embedding with shape (*, E).
            vec_map: Vector direction map with shape (*, N, 3).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, N, C).
        """
        # Start by patchifying the input map.
        batch_dim = x.shape[:-2]
        N, C = x.shape[-2:]
        x = x.reshape(*batch_dim, N // self.patch_size, self.patch_size * C)

        # Set the vector of each patch to the average of the patches.
        vec_map = vec_map.reshape(
            *batch_dim, N // self.patch_size, self.patch_size, 3
        )
        vec_map = jnp.mean(vec_map, axis=-2)

        # Embed the input map to have dimension (B, N // patch_size, D) and add
        # absolute positional embedding for each patch.
        x = nn.Dense(self.emb_dim)(x)
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (N // self.patch_size, self.emb_dim),
        )
        x = x + jnp.broadcast_to(pos_embedding, x.shape)

        for i in range(self.n_blocks):
            x = HEALPixAttentionBlock(
                self.emb_dim, self.heads, self.time_emb_dim,
                self.dropout_rate_block[i]
            )(x, t, vec_map,train=train)

        # Decode back to patch space.
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.patch_size * C)(x)
        x = x.reshape(*batch_dim, N, C)
        return x
