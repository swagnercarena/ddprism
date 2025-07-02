"""Embedding models for use in diffusion."""

import math
from typing import Any, Callable, Mapping, Sequence, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import Array
from einops import rearrange
from wandb.util import downsample


def reflect_pad(x: Array, kernel_size: Sequence[int]) -> Array:
    """Pads spatial dimensions of image by reflection.

    Arguments:
        x: Input tensor (..., H, W, C).
        kernel_size: Convolutional kernel size, e.g. (3, 3).

    Returns:
        Padded image with shape (..., H + 2 * pad_h, W + 2 * pad_w, C).
    """
    assert len(kernel_size) == 2, "Only 2-D kernels supported."
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2
    pad_cfg = [(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)]
    return jnp.pad(x, pad_cfg, mode="reflect")


def positional_embedding(pos, emb_features: int = 64):
    r"""Creates a positional embedding module.

    References:
        | Attention Is All You Need (Vaswani et al., 2017)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        pos: Postional value to embed. For diffusion between 0 and 1.
        emb_features: Number of embedding features.
    """
    freqs = jnp.linspace(0, 1, emb_features // 2)
    freqs = jnp.asarray((1 / 1e4) ** freqs)

    pos = pos[..., None]

    return jnp.concatenate(
        (jnp.sin(freqs * pos), jnp.cos(freqs * pos),), axis=-1,
    )


class TimeMLP(nn.Module):
    r"""Creates a multi-layer perceptron (MLP).

    Arguments:
        features: The number of output features.
        hid_features: The number of hidden features.
        activation: The activation function constructor.
        normalize: Whether features are normalized between layers or not.
    """
    features: int
    hid_features: list[int] = (64, 64)
    activation: Callable[..., nn.Module] = nn.silu
    normalize: bool = False

    @nn.compact
    def __call__(self, x: Array, t: Array, train: bool = True) -> Array:
        """MLP output conditioned on time.

        Arguments:
            x: Input features with shape (*, features).
            t: Time embedding with shape (*, E).
            train: Has no effect on this module.

        Returns:
            Conditioned MLP output with shape (*, features).
        """
        x = jnp.concatenate((x, t), axis=-1)

        for feat in self.hid_features:
            x = nn.Dense(feat)(x)
            x = self.activation(x)

            if self.normalize:
                x = nn.LayerNorm()(x)

        x = nn.Dense(self.features)(x)
        return x


class AdaLNZeroModulation(nn.Module):
    r"""Produces (gamma,beta,alpha) for adaptive layer norm modulation (adaLN-0)

    adaLN modulation originally proposed in https://arxiv.org/pdf/1709.07871
    adaLN-Zero modulation applied for DiT in https://arxiv.org/pdf/2212.09748
    (see Figure 3)

    Arguments:
        channels: Number of channels for input x
        emb_features: Size of embedding vector
        activation:  Activation function.
    """
    channels: int
    emb_features: int
    activation: Callable[..., nn.Module] = nn.silu

    @nn.compact
    def __call__(self, t: Array) -> tuple[Array, Array, Array]:
        """Returns (gamma,beta,alpha) for adaLN-Zero

        Arguments:
            t: Time to use for modulation. Assumed to have dimension (*, E)

        Returns:
            Gamma, beta, and alpha for modulation.

        Notes:
            Assumes channel is last.
        """
        # Initialize final MLP layer with small weights since it outputs
        # perturbations around 1.
        kernel_init = nn.initializers.variance_scaling(
            1e-1, "fan_in", "truncated_normal"
        )

        t = nn.Dense(self.emb_features)(t)
        t = self.activation(t)
        t = nn.Dense(3 * self.channels, kernel_init=kernel_init)(t)

        # Add spatial dimensions assuming channel is last.
        out = rearrange(t, '... C -> ... 1 1 C')
        return jnp.array_split(out, 3, axis=-1)


class ResBlock(nn.Module):
    r"""Creates a residual block with dropout.

    Arguments:
        channels: Number of channels for input x. Channels are assumed to be
            last dimension.
        emb_features: Size of embedding vector for LayerNorm modulation.
        dropout_rate: Dropout rate. Default is 0 (no dropout).
        activation: Activation function.
        kernel_size: Size of convolutional kernel. Default is (3,3).
        padding: a sequence of n (low, high) integer pairs that give the padding
            to apply before and after each spatial dimension.
    """
    channels: int
    emb_features: int
    dropout_rate: float = 0.0
    activation: Callable[..., nn.Module] = nn.silu
    kernel_size: Sequence[int] = (3, 3)

    @nn.compact
    def __call__(self, x: Array, t: Array, train: bool = True) -> Array:
        """Call residual block on image and modulate by time.

        Arguments:
            x: Input image with last dimension as channels. Shape (*, H, W, C)
            t: Time for each image. Shape (*, E).
            train: If true, values are passed in training mode.

        Returns:
            Residual block output.
        """
        # Perform adaLN-Zero modulation to condition on t.
        gamma, beta, alpha = (
            AdaLNZeroModulation(self.channels, self.emb_features)(t)
        )

        # Defaults to normalization of 1.
        y = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        y = (gamma + 1) * y + beta
        y = self.activation(y)

        # First convolution with reflect padding
        y = reflect_pad(y, self.kernel_size)
        y = nn.Conv(self.channels, self.kernel_size, padding='VALID')(y)

        # Apply dropout and final convolution if non-zero dropout rate.
        if self.dropout_rate > 0.0:
            y = nn.Dropout(self.dropout_rate)(y, deterministic=not train)

        y = nn.LayerNorm()(y)
        y = self.activation(y)

        # Second convolution with reflect padding
        y = reflect_pad(y, self.kernel_size)
        y = nn.Conv(self.channels, self.kernel_size, padding='VALID')(y)

        # Last step of adaLN-Zero modulation.
        y = x + (alpha * y / jnp.sqrt(1 + alpha ** 2))

        return y


class AttBlock(nn.Module):
    r"""Creates a residual self-attention block with adaLN-Zero modulation

    Arguments:
        channels: number of channels for input x
        emb_features (int): size of embedding vector
        dropout_rate (float): Dropout rate for attention layer.
        heads (int): Number of heads to use in multi-headed attention layer.
            Default = 1.
        qkv_features: Number of key, query, and value features
        out_features: Number of output features.
        use_bias: If true, bias will be included in self-attention. Default is
            True.

    """
    channels: int
    emb_features: int
    dropout_rate: float = 0.0
    heads: int = 1
    qkv_features: int = None
    out_features: int = None
    use_bias: bool = True


    @nn.compact
    def __call__(self, x: Array, t: Array, train: bool = True) -> Array:
        """Call attention block on image and modulate by time.

        Arguments:
            x: Input image with last dimension as channels. Shape (*, H, W, C)
            t: Time for each image. Shape (*, E).
            train: If true, values are passed in training mode.

        Returns:
            Residual block output.
        """
        # perform adaLN-Zero modulation to condition on t
        gamma, beta, alpha = (
            AdaLNZeroModulation(self.channels, self.emb_features)(t)
        )

        y = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        y = (gamma + 1) * y + beta
        # Flatten the spatial dimensions.
        y = rearrange(y, '... H W C -> ... (H W) C')
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.heads, qkv_features=self.qkv_features,
            out_features=self.out_features, dropout_rate=self.dropout_rate,
            use_bias=self.use_bias
        )(y, deterministic=not train)

        # Reshape back to input shape
        y = y.reshape(x.shape)

        # Last step of adaLN-Zero modulation
        y = x + (alpha * y / jnp.sqrt(1 + alpha ** 2))

        return y


class Resample(nn.Module):
    r"""Wrapper around jax resize for spatial resampling.

    This is a modified version of inox Resample() (author=francois-rozet)
    see: https://github.com/francois-rozet/inox

    Arguments:
        factor: Resampling factors for each H_1, ..., H_n in x.
        method: Method for jax resize. Default is nearest.

    """
    factor: Sequence[float]
    method: str = 'nearest'

    @nn.compact
    def __call__(self, x: Array) -> Array:
        r"""Resample image.

        Arguments:
            x: Image to resample. Assumed shape is (*, H_1, ..., H_n, C)

        Returns:
            Resampled output tensor.
        """
        # Break up x shape to seperate spatial shape.
        b_dim = x.ndim - len(self.factor) - 1
        assert b_dim > 0
        b_shape, s_shape, c_shape = (
            x.shape[:b_dim], x.shape[b_dim:-1], x.shape[-1:]
        )

        # Calculate new shape from factors.
        s_shape = tuple(
            int(math.floor(f * h)) for f, h in zip(self.factor, s_shape)
        )

        return jax.image.resize(
            x, shape=b_shape + s_shape + c_shape, method=self.method,
            antialias=True
        )


class UNet(nn.Module):
    r"""Creates a time conditional U-Net.

    Arguments:
        hid_channels: Number of channels used in each level.
        hid_blocks: Number of block used in each level.
        kernel_size: Defines the kernel size used for all convolutional blocks.
        emb_features: Size of the embedding vector that encodes the time
            features.
        heads: Number of heads to use for the attention block of each level.
            If a level is not included it will have no attention block.
        dropout_rate: Dropout rate applied in the ResBlock. Attention block has
            no dropout.
        activation: Activation function for resnet blocks.

    Notes:
        Assumed in_channels = out_channels for diffusion.
    """
    hid_channels: Sequence[int]
    hid_blocks: Sequence[int]
    kernel_size: Sequence[int] = (3, 3)
    emb_features: int = 64
    heads: Mapping[str, int] = None
    dropout_rate: float = 0.0
    activation: Callable[..., nn.Module] = nn.silu


    @nn.compact
    def __call__(self, x: Array, t: Array, train: bool = True) -> Array:
        r"""Return the conditioned U-Net output.

        Arguments:
            x: Input image with shape (*, H, W, C).
            t: Time embedding, with shape (*, E).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, H, W, C).
        """
        assert len(self.hid_blocks) == len(self.hid_channels)
        heads = {} if self.heads is None else self.heads

        # Arrays to concatenate when doing the upsampling.
        concat = []

        out_channels = x.shape[-1]
        downsample_factor = [0.5, 0.5]

        # Descend from image to lowest dimension.
        for i, n_blocks in enumerate(self.hid_blocks):

            if i == 0:
                # First convolution shouldn't downsample.
                x = reflect_pad(x, self.kernel_size)
                x = nn.Conv(
                    self.hid_channels[i], kernel_size=self.kernel_size,
                    padding='VALID'
                )(x)
            else:
                # Downsample.
                x = Resample(downsample_factor, method="lanczos3")(x)
                x = reflect_pad(x, self.kernel_size)
                x = nn.Conv(
                    self.hid_channels[i], kernel_size=self.kernel_size,
                    padding='VALID'
                )(x)

            # Residual convolutions without downsampling.
            for _ in range(n_blocks):
                x = ResBlock(
                    self.hid_channels[i], self.emb_features, self.dropout_rate,
                    activation=self.activation, kernel_size=self.kernel_size
                )(x, t, train)
                # Add attention blocks if specified.
                if str(i) in heads:
                    x = AttBlock(
                        self.hid_channels[i], self.emb_features,
                        self.dropout_rate, heads[str(i)]
                    )(x, t, train)

            # Add output to concat.
            concat.append(x)

        # Middle block.
        x = ResBlock(
            self.hid_channels[-1], self.emb_features, self.dropout_rate,
            activation=self.activation, kernel_size=self.kernel_size,
        )(x, t, train)
        if str(len(self.hid_blocks) - 1) in heads:
            x = AttBlock(
                self.hid_channels[-1], self.emb_features,
                self.dropout_rate, heads[str(len(self.hid_blocks) - 1)]
            )(x, t, train)
        x = ResBlock(
            self.hid_channels[-1], self.emb_features, self.dropout_rate,
            activation=self.activation, kernel_size=self.kernel_size,
        )(x, t, train)

        # Ascend from lowest dimension to image.
        upsample_factor = [2.0, 2.0]
        for i, n_blocks in reversed(list(enumerate(self.hid_blocks))):

            # Get the memory and convolve in the upsampling.
            c = concat.pop()
            x = jnp.concat((x, c), axis=-1)
            x = reflect_pad(x, self.kernel_size)
            x = nn.Conv(
                    self.hid_channels[i], kernel_size=self.kernel_size,
                    padding='VALID'
            )(x) + c # Skip for the memory.

            # For layers that aren't the bottom layer, convolve the upsampling.
            if i + 1 < len(self.hid_blocks):
                x = reflect_pad(x, self.kernel_size)
                x = nn.Conv(
                    self.hid_channels[i], kernel_size=self.kernel_size,
                    padding='VALID'
                )(x)

            # Residual convolutions without upsampling.
            for _ in range(n_blocks):
                x = ResBlock(
                    self.hid_channels[i], self.emb_features, self.dropout_rate,
                    activation=self.activation, kernel_size=self.kernel_size
                )(x, t, train)

                # Add attention blocks if specified.
                if str(i) in heads:
                    x = AttBlock(
                        self.hid_channels[i], self.emb_features,
                        self.dropout_rate, heads[str(i)]
                    )(x, t, train)

            if i == 0:
                # Final layer of is fully connected.
                x = nn.Dense(out_channels)(x)
            else:
                # For all other layers conduct an upsampling iteration.
                x = Resample(upsample_factor, method='nearest')(x)

        return x


class FlatUNet(UNet):
    """Wrapper class for dealing with flattened images.

    Arguments:
        hid_channels: Number of channels used in each level.
        hid_blocks: Number of block used in each level.
        kernel_size: Defines the kernel size used for all convolutional blocks.
        emb_features: Size of the embedding vector that encodes the time
            features.
        heads: Number of heads to use for the attention block of each level.
            If a level is not included it will have no attention block.
        dropout_rate: Dropout rate applied in the ResBlock. Attention block has
            no dropout.
        activation: Activation function for resnet blocks.
        image_shape: Shape in (H, W, C) for images in the batch.
    """
    image_shape: Sequence[int] = None

    def setup(self):
        # Check image shape meets the requirements.
        assert self.image_shape is not None
        assert len(self.image_shape) == 3

    @nn.compact
    def __call__(self, x: Array, t: Array, train: bool = True) -> Array:
        """Reshape image for UNet call and then reflatten.

        Arguments:
            x: Input image with shape (*, (H W C)).
            t: Time embedding, with shape (*, E).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, (H W C)).
        """
        # Unflatten x.
        x = self.reshape(x)
        x = super().__call__(x, t, train)
        # Flatten.
        x = rearrange(x, '... H W C -> ... (H W C)')

        return x

    def reshape(self, x:Array) -> Array:
        """Reshape flattened image.

        Arguments:
            x: Input image with shape (*, (H W C)).

        Returns:
            Input image with shape (*, H, W, C).
        """
        return rearrange(
            x, '... (H W C) -> ... H W C', H=self.image_shape[0],
            W=self.image_shape[1], C=self.image_shape[2]
        )

    @property
    def feat_dim(self):
        """Get the feature dimension."""
        return self.image_shape[0] * self.image_shape[1] * self.image_shape[2]

