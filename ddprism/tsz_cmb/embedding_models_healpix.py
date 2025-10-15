"""Embedding models for HEALPix CMB data."""

from typing import Callable, Sequence, Tuple

from einops import rearrange
from flax import linen as nn
import jax.numpy as jnp
from jax import Array


PERIOD = 1e-1


class AdaLNZeroModulation(nn.Module):
    r"""Produces (gamma,beta,alpha) for adaptive layer norm modulation (adaLN-0)

    adaLN modulation originally proposed in https://arxiv.org/pdf/1709.07871
    adaLN-Zero modulation applied for DiT in https://arxiv.org/pdf/2212.09748
    (see Figure 3)

    Arguments:
        emb_features: Dimension of the embedding.
        time_emb_features: Size of embedding vector
        activation:  Activation function.
    """
    emb_features: int
    time_emb_features: int
    activation: Callable[..., nn.Module] = nn.silu

    @nn.compact
    def __call__(self, t: Array) -> Tuple[Array, Array, Array]:
        """Returns (gamma,beta,alpha) for adaLN-Zero

        Arguments:
            t: Time to use for modulation. Assumed to have dimension (*, E)

        Returns:
            Gamma, beta, and alpha for modulation.

        Notes:
            Assumes embedding dimension is last.
        """
        # Initialize final MLP layer with small weights since it outputs
        # perturbations around 1.
        kernel_init = nn.initializers.variance_scaling(
            1e-1, "fan_in", "truncated_normal"
        )

        t = nn.Dense(self.time_emb_features)(t)
        t = self.activation(t)
        t = nn.Dense(3 * self.emb_features, kernel_init=kernel_init)(t)

        # Add patch dimension assuming embedding dimension is last.
        out = rearrange(t, '... C -> ... 1 C')
        return jnp.array_split(out, 3, axis=-1)


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
        vec_map = vec_map / jnp.linalg.norm(vec_map, axis=-1, keepdims=True)
        dot = jnp.einsum('...NK,...MK->...NM', vec_map, vec_map)
        angle = jnp.arccos(jnp.clip(dot, -1.0, 1.0)) / 1e-3

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
        emb_features: Dimension of the embedding.
        n_heads: Number of heads in the attention mechanism.
        dropout_rate: Dropout rate.
    """
    emb_features: int
    n_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self, x:Array, relative_bias_logits:Array, train:bool=True
    ) -> Array:
        """Call the attention mechanism.

        Arguments:
            x: Input map with shape (*, N, D).
            relative_bias_logits: Relative bias logits with shape (*, N, N, H).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, N, D).
        """
        # Basic shape checks
        assert x.ndim >= 3, (
            f"HEALPixAttention expects (..., N, D), got {x.shape}"
        )
        head_dim = self.emb_features // self.n_heads
        assert head_dim * self.n_heads == self.emb_features

        # Multihead attention.
        batch_dim = x.shape[:-2]
        N = x.shape[-2]
        # relative bias must match (..., N, N, H)
        assert relative_bias_logits.shape[-1] == self.n_heads, (
            f"relative_bias_logits heads mismatch: {relative_bias_logits.shape}"
            f" vs n_heads={self.n_heads}"
        )
        assert (
            relative_bias_logits.shape[-3] == N and
            relative_bias_logits.shape[-2] == N
        ), (
            f"relative_bias_logits must be (..., N, N, H) with N={N}, got "
            f"{relative_bias_logits.shape}"
        )
        qkv = nn.Dense(3 * self.emb_features)(x)
        qkv = qkv.reshape(*batch_dim, N, self.n_heads, 3, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=-2)
        q, k, v = q.squeeze(axis=-2), k.squeeze(axis=-2), v.squeeze(axis=-2)

        # Compute attention.
        logits = jnp.einsum('...QHT,...KHT->...QKH', q, k)
        logits = logits / jnp.sqrt(head_dim)

        # Add relative bias to attention.
        logits = logits + relative_bias_logits

        # Make sure softmax is on the key dimension.
        logits -= jnp.max(logits, axis=-2, keepdims=True)
        attention = nn.softmax(logits, axis=-2)
        attention = nn.Dropout(self.dropout_rate)(
            attention, deterministic=not train
        )

        # Attend with relative bias.
        y = jnp.einsum('...QKH,...KHT->...QHT', attention, v)
        y = y.reshape(*batch_dim, N, self.emb_features)
        y = nn.Dense(self.emb_features)(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=not train)
        return y


class HEALPixAttentionBlock(nn.Module):
    """Transformer block for HEALPix CMB data.

    Arguments:
        emb_features: Dimension of the embedding.
        n_heads: Number of heads in the attention mechanism.
        time_emb_features: Dimension of the time embedding.
        dropout_rate: Dropout rate.
    """
    emb_features: int
    n_heads: int
    time_emb_features: int
    dropout_rate: float
    mlp_ratio: int = 4

    @nn.compact
    def __call__(
        self, x:Array, t:Array, relative_bias_logits:Array, train:bool=True
    ) -> Array:
        """Call the attention block.

        Arguments:
            x: Input map with shape (*, N, D).
            t: Time embedding with shape (*, E).
            relative_bias_logits: Relative bias logits with shape (*, N, N, H).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, N, D).
        """
        # Shape checks
        assert x.ndim >= 3, (
            f"HEALPixAttentionBlock expects (..., N, D), got {x.shape}"
        )
        assert t.ndim >= 2 and t.shape[-1] == self.time_emb_features, (
            f"t must have last dim {self.time_emb_features}, got {t.shape}"
        )
        assert relative_bias_logits.ndim >= 4 and (
            relative_bias_logits.shape[-1] == self.n_heads
        ), (
            f"relative_bias_logits must end with n_heads={self.n_heads},"
            f" got {relative_bias_logits.shape}"
        )

        # Perform adaLN-Zero modulation to condition on t.
        gamma_one, beta_one, alpha_one = (
            AdaLNZeroModulation(self.emb_features, self.time_emb_features)(t)
        )
        y = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        y = (gamma_one + 1) * y + beta_one

        y = HEALPixAttention(
            self.emb_features, self.n_heads, self.dropout_rate
        )(y, relative_bias_logits, train=train)

        # Last step of adaLN-Zero modulation.
        x = x + (alpha_one * y / jnp.sqrt(1 + alpha_one ** 2))

        # Pointwise MLP.
        gamma_two, beta_two, alpha_two = (
            AdaLNZeroModulation(self.emb_features, self.time_emb_features)(t)
        )
        y = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        y = (gamma_two + 1) * y + beta_two

        # MLP for pointwise feedforward.
        y = nn.Dense(self.emb_features * self.mlp_ratio)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.emb_features)(y)

        # Last step of adaLN-Zero modulation.
        x = x + (alpha_two * y / jnp.sqrt(1 + alpha_two ** 2))

        return x


class HEALPixTransformer(nn.Module):
    r"""Creates a time conditioned HEALPix transformer.

    Arguments:
        emb_features: Dimension of the embedding.
        n_blocks: Number of transformer blocks.
        dropout_rate_block: Dropout rate for each transformer block.
        heads: Number of heads in the attention mechanism.
        patch_size: Size of the patch to divide the input map into.
        time_emb_features: Size of the embedding vector that encodes the time
            features.
        freq_features: Number of frequency features for the relative bias.
    """
    emb_features: int
    n_blocks: int
    dropout_rate_block: Sequence[float]
    heads: int
    patch_size: int
    time_emb_features: int
    freq_features: int = 64

    def setup(self):
        self.relative_bias = RelativeBias(
            self.heads, freq_features=self.freq_features
        )

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
        # Basic input shape checks
        assert x.ndim >= 3, (
            f"HEALPixTransformer expects x with shape (..., N, C), got "
            f"{x.shape}"
        )
        assert vec_map.ndim >= 3, (
            f"HEALPixTransformer expects vec_map with shape (..., N, 3), got "
            f"{vec_map.shape}"
        )
        assert vec_map.shape[:-1] == x.shape[:-1], (
            f"vec_map batch dims {vec_map.shape[:-1]} must match x batch dims "
            f"{x.shape[:-1]}"
        )
        assert vec_map.shape[-1] == 3, (
            f"vec_map last dim must be 3, got {vec_map.shape}"
        )
        assert vec_map.shape[-2] == x.shape[-2], (
            f"N mismatch between x and vec_map: {x.shape[-2]} vs "
            f"{vec_map.shape[-2]}"
        )

        # Start by patchifying the input map.
        batch_dim = x.shape[:-2]
        N, C = x.shape[-2:]
        x = x.reshape(*batch_dim, N // self.patch_size, self.patch_size * C)

        # Shape validations.
        assert N % self.patch_size == 0, "N must be divisible by patch_size"
        assert len(self.dropout_rate_block) == self.n_blocks

        # Set the vector of each patch to the average of the patches.
        vec_map = vec_map.reshape(
            *batch_dim, N // self.patch_size, self.patch_size, 3
        )
        vec_map = jnp.mean(vec_map, axis=-2)

        # Compute the shared relative bias logits.
        relative_bias_logits = self.relative_bias(vec_map)

        # Embed the input map to have dimension (B, N // patch_size, D) and add
        # absolute positional embedding for each patch.
        x = nn.Dense(self.emb_features)(x)
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (N // self.patch_size, self.emb_features),
        )
        # Positional embedding broadcast check
        pos_broadcast = jnp.broadcast_to(pos_embedding, x.shape)
        assert pos_broadcast.shape == x.shape, (
            f"positional embedding broadcast failed: {pos_broadcast.shape} vs "
            f"x {x.shape}"
        )
        x = x + pos_broadcast

        for i in range(self.n_blocks):
            x = HEALPixAttentionBlock(
                self.emb_features, self.heads, self.time_emb_features,
                self.dropout_rate_block[i]
            )(x, t, relative_bias_logits, train=train)

        # Decode back to patch space.
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.patch_size * C)(x)
        # Decode shape checks
        assert x.shape[-1] == self.patch_size * C, (
            f"Decoder output last dim should be {self.patch_size*C}, got "
            f"{x.shape}"
        )
        x = x.reshape(*batch_dim, N, C)
        assert x.shape[-2:] == (N, C), (
            f"Final reshape to (N, C) failed, got {x.shape}"
        )
        return x


class FlatHEALPixTransformer(HEALPixTransformer):
    """Wrapper class for dealing with (channel) flattened HEALPix data.

    Arguments:
        emb_features: Dimension of the embedding.
        n_blocks: Number of transformer blocks.
        dropout_rate_block: Dropout rate for each transformer block.
        heads: Number of heads in the attention mechanism.
        patch_size: Size of the patch to divide the input map into.
        time_emb_features: Size of the embedding vector that encodes the time
            features.
        freq_features: Number of frequency features for the relative bias.
        healpix_shape: Healpix shape with the number of channels.
    """
    healpix_shape: Sequence[int] = None

    def setup(self):
        # Check image shape meets the requirements.
        assert self.healpix_shape is not None
        assert len(self.healpix_shape) == 2
        super().setup()

    @nn.compact
    def __call__(
        self, x: Array, t: Array, vec_map: Array, train: bool = True
    ) -> Array:
        """Reshape image for transformer call and then reflatten.

        Arguments:
            x: Input image with shape (*, (N C)).
            t: Time embedding, with shape (*, E).
            vec_map: Vector direction map with shape (*, N, 3).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, (N C)).
        """
        # Unflatten x.
        x = self.reshape(x)
        x = super().__call__(x, t, vec_map, train)
        # Flatten.
        x = rearrange(x, '... N C -> ... (N C)')

        return x

    def reshape(self, x:Array) -> Array:
        """Reshape flattened image.

        Arguments:
            x: Input image with shape (*, (N C)).

        Returns:
            Input image with shape (*, N, C).
        """
        # Validate flat feature dimension matches N*C
        assert x.shape[-1] == self.healpix_shape[0] * self.healpix_shape[1], (
            f"FlatHEALPixTransformer.reshape expected last dim "
            f"N*C={self.healpix_shape[0]*self.healpix_shape[1]}, got {x.shape}"
        )
        return rearrange(
            x, '... (N C) -> ... N C',
            N=self.healpix_shape[0], C=self.healpix_shape[1]
        )

    @property
    def feat_dim(self):
        """Get the feature dimension."""
        return self.healpix_shape[0] * self.healpix_shape[1]
