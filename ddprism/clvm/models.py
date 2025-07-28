"""Models for CLVM encoder and decoder."""
from typing import Callable, Mapping, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp
from jax import Array
from einops import rearrange

from ddprism.embedding_models import reflect_pad, Resample

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
            if self.activation is not None:
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
    def encode_obs(self, x: Array, a_mat: Array, train: bool = True) -> Array:
        """Encode the input observations into the latent distribution.

        Args:
            x: Input observations.
            a_mat: Linear transformation matrix.
            train: Whether in training mode for dropout.
        """
        a_mat = rearrange(a_mat, 'K L M -> K (L M)')
        x = jnp.concatenate([x, a_mat], axis=-1)
        return self.encode_feat(x, train=train)


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
            if self.activation is not None:
                x = self.activation(x)

            if self.normalize:
                x = nn.LayerNorm()(x)

            # Dropout
            if self.dropout_rate > 0.0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        # Final output layer
        return nn.Dense(self.features)(x)


class ResBlock(nn.Module):
    r"""Creates a residual block with dropout.

    Arguments:
        channels: Number of channels for input x. Channels are assumed to be
            last dimension.
        dropout_rate: Dropout rate. Default is 0 (no dropout).
        activation: Activation function.
        kernel_size: Size of convolutional kernel. Default is (3,3).
        padding: a sequence of n (low, high) integer pairs that give the padding
            to apply before and after each spatial dimension.
    """
    channels: int
    dropout_rate: float = 0.0
    activation: Callable[..., nn.Module] = nn.silu
    kernel_size: Sequence[int] = (3, 3)

    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        """Call residual block on image and modulate by time.

        Arguments:
            x: Input image with last dimension as channels. Shape (*, H, W, C)
            train: If true, values are passed in training mode.

        Returns:
            Residual block output.
        """

        # Defaults to normalization of 1.
        y = nn.LayerNorm(use_bias=True, use_scale=True)(x)
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

        return y + x


class AttBlock(nn.Module):
    r"""Creates a residual self-attention block with adaLN-Zero modulation

    Arguments:
        channels: number of channels for input x
        dropout_rate (float): Dropout rate for attention layer.
        heads (int): Number of heads to use in multi-headed attention layer.
            Default = 1.
        qkv_features: Number of key, query, and value features
        out_features: Number of output features.
        use_bias: If true, bias will be included in self-attention. Default is
            True.

    """
    channels: int
    dropout_rate: float = 0.0
    heads: int = 1
    qkv_features: int = None
    out_features: int = None
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        """Call attention block on image and modulate by time.

        Arguments:
            x: Input image with last dimension as channels. Shape (*, H, W, C)
            train: If true, values are passed in training mode.

        Returns:
            Residual block output.
        """

        y = nn.LayerNorm(use_bias=True, use_scale=True)(x)

        # Flatten the spatial dimensions.
        y = rearrange(y, '... H W C -> ... (H W) C')
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.heads, qkv_features=self.qkv_features,
            out_features=self.out_features, dropout_rate=self.dropout_rate,
            use_bias=self.use_bias
        )(y, deterministic=not train)

        # Reshape back to input shape
        y = y.reshape(x.shape)

        return y + x


class EncoderFlatUNet(nn.Module):
    r"""Creates an encoder similar to the first half of a UNet.

    Arguments:
        latent_features: Number of latent features.
        image_shape: Shape of the images once unflattened.
        hid_channels: Number of channels used in each level.
        hid_blocks: Number of block used in each level.
        kernel_size: Defines the kernel size used for all convolutional blocks.
        heads: Number of heads to use for the attention block of each level.
            If a level is not included it will have no attention block.
        dropout_rate: Dropout rate applied in the ResBlock. Attention block has
            no dropout.
        activation: Activation function for resnet blocks.

    Notes:
        Assumed in_channels = out_channels for diffusion.
    """
    latent_features: int
    image_shape: Sequence[int]
    hid_channels: Sequence[int]
    hid_blocks: Sequence[int]
    kernel_size: Sequence[int] = (3, 3)
    heads: Mapping[str, int] = None
    dropout_rate: float = 0.0
    activation: Callable[..., nn.Module] = nn.silu

    @nn.compact
    def _encode_feat(self, x: Array, train: bool = True) -> Array:
        r"""Return the encoder output.

        Arguments:
            x: Input image with shape (*, H, W, C).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, H, W, C).
        """
        assert len(self.hid_blocks) == len(self.hid_channels)
        heads = {} if self.heads is None else self.heads

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
                    self.hid_channels[i], self.dropout_rate,
                    activation=self.activation, kernel_size=self.kernel_size
                )(x, train)
                # Add attention blocks if specified.
                if str(i) in heads:
                    x = AttBlock(
                        self.hid_channels[i],
                        self.dropout_rate, heads[str(i)]
                    )(x, train)

        # Middle block.
        x = ResBlock(
            self.hid_channels[-1], self.dropout_rate,
            activation=self.activation, kernel_size=self.kernel_size,
        )(x, train)
        if str(len(self.hid_blocks) - 1) in heads:
            x = AttBlock(
                self.hid_channels[-1],
                self.dropout_rate, heads[str(len(self.hid_blocks) - 1)]
            )(x, train)
        x = ResBlock(
            self.hid_channels[-1], self.dropout_rate,
            activation=self.activation, kernel_size=self.kernel_size,
        )(x, train)

        # Flatten the image.
        x = x.reshape(x.shape[0], -1)
        x = self.activation(x)

        # Final output layer
        x = jnp.split(nn.Dense(self.latent_features * 2)(x), 2, axis=-1)
        return x[0], jnp.exp(x[1])

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

    @nn.compact
    def encode_feat(self, x: Array, train: bool = True) -> Array:
        """Encode the input features into the latent distribution.

        Arguments:
            x: Input features with shape (*, (H W C)).
            train: Whether in training mode for dropout.

        Returns:
            Latent mean and variance deviation for the input features.
        """
        x = self.reshape(x)
        return self._encode_feat(x, train=train)

    @nn.compact
    def encode_obs(self, x: Array, a_mat: Array, train: bool = True) -> Array:
        """Encode the input observations into the latent distribution.

        Arguments:
            x: Input observations with shape (*, (H W C)).
            a_mat: Linear transformation matrix with shape
                (* (I J) (H W)).

        Returns:
            Latent mean and variance deviation for the input observations.
        """
        x = self.reshape(x)
        a_mat = rearrange(
            a_mat, 'K I (H W) -> K H W I', H=self.image_shape[0],
            W=self.image_shape[1]
        )
        # Turn the observation dimension into channels.
        x = jnp.concatenate([x, a_mat], axis=-1)

        return self._encode_feat(x, train=train)


class DecoderFlatUNet(nn.Module):
    r"""Creates a decoder similar to the second half of a UNet.

    Arguments:
        image_shape: Shape of the image.
        hid_channels: Number of channels used in each level. Should match the
            encoder.
        hid_blocks: Number of block used in each level.
        kernel_size: Defines the kernel size used for all convolutional blocks.
        heads: Number of heads to use for the attention block of each level.
            If a level is not included it will have no attention block.
        dropout_rate: Dropout rate applied in the ResBlock. Attention block has
            no dropout.
        activation: Activation function for resnet blocks.

    Notes:

    """
    image_shape: Tuple
    hid_channels: Sequence[int]
    hid_blocks: Sequence[int]
    kernel_size: Sequence[int] = (3, 3)
    heads: Mapping[str, int] = None
    dropout_rate: float = 0.0
    activation: Callable[..., nn.Module] = nn.silu

    @nn.compact
    def _decode_feat(self, x: Array, train: bool = True) -> Array:
        """Decode the latent variables into the feature space.

        Arguments:
            x: Input latents with shape (*, latent_features).
            train: Whether in training mode for dropout.

        Returns:
            Decoded output with shape (*, H, W, C).
        """

        assert len(self.hid_blocks) == len(self.hid_channels)
        heads = {} if self.heads is None else self.heads

        in_features = (
            self.image_shape[0] // (2 ** (len(self.hid_channels) - 1)),
            self.image_shape[1] // (2 ** (len(self.hid_channels) - 1)),
            self.hid_channels[-1]
        )
        upsample_factor = [2.0, 2.0]

        # Broadcast vector to be compatible with the last output shape of the
        # encoder UNet.
        x = nn.Dense(in_features[0] * in_features[1] * in_features[2])(x)
        x = x.reshape((-1,) + in_features)

        # Ascend from lowest dimension to image.
        for i, n_blocks in reversed(list(enumerate(self.hid_blocks))):

            # Residual convolutions without upsampling.
            for _ in range(n_blocks):
                x = ResBlock(
                     self.hid_channels[i], self.dropout_rate,
                     activation=self.activation, kernel_size=self.kernel_size,
                )(x, train)

                # Add attention blocks if specified.
                if str(i) in heads:
                    x = AttBlock(
                         self.hid_channels[i],
                         self.dropout_rate, heads[str(i)]
                     )(x, train)

            if i == 0:
                # Final layer of is fully connected.
                x = nn.Dense(self.image_shape[2])(x)
            else:
                # For all other layers conduct an upsampling iteration.
                x = Resample(upsample_factor, method='lanczos3')(x)

        return x

    @nn.compact
    def decode_feat(self, x: Array, train: bool = True) -> Array:
        """Decode the latent variables into the feature space.

        Arguments:
            x: Input latents with shape (*, latent_features).
            train: Whether in training mode for dropout.

        Returns:
            Decoded feature with shape (*, (H W C)).
        """
        x = self._decode_feat(x, train=train)
        return rearrange(x, '... H W C -> ... (H W C)')
