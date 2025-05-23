"""Models for encoder and decoder architectures."""

import math
from typing import Any, Callable, Mapping, Sequence, Tuple

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

        # The output of the network are means and variances for the latent dimensions.
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

        # The output of the network are means and variances for the latent dimensions
        x = nn.Dense(self.features, use_bias=self.bias)(x)
        return x

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
    padding: Sequence[Tuple[int, int]] = None

    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        """Call residual block on image and modulate by time.

        Arguments:
            x: Input image with last dimension as channels. Shape (*, H, W, C)
            train: If true, values are passed in training mode.

        Returns:
            Residual block output.
        """
        # Set the padding to same if none is provided.
        padding = 'same' if self.padding is None else self.padding

        # Apply layer normalization.
        y = nn.LayerNorm(use_bias=True, use_scale=True)(x)

        # Call convolutional layers.
        y = self.activation(y)
        y = nn.Conv(
            self.channels, self.kernel_size, padding=padding, use_bias=False
        )(y)

        # Apply dropout and final convolution if non-zero dropout rate.
        if self.dropout_rate > 0.0:
            y = nn.Dropout(self.dropout_rate)(y, deterministic=not train)

        # Apply layer normalization.
        y = nn.LayerNorm()(y)
    
        y = self.activation(y)
        y = nn.Conv(self.channels, self.kernel_size, padding=padding)(y)

        return y + x


class AttBlock(nn.Module):
    r"""Creates a residual self-attention block

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
            x, shape=b_shape + s_shape + c_shape, method=self.method
        )


 
class UNetEncoder(nn.Module):
    r"""Creates an encoder with the architecture similar to that of the first half of UNet.

    Arguments:
        latent_features: Number of latent features.
        hid_channels: Number of channels used in each level.
        hid_blocks: Number of block used in each level.
        kernel_size: Defines the kernel size used for all convolutional blocks.
        heads: Number of heads to use for the attention block of each level.
            If a level is not included it will have no attention block.
        dropout_rate: Dropout rate applied in the ResBlock. Attention block has
            no dropout.
        activation: Activation function for resnet blocks.

    Notes:
        
    """
    latent_features: int
    hid_channels: Sequence[int]
    hid_blocks: Sequence[int]
    input_shape: Tuple
    kernel_size: Sequence[int] = (3, 3)
    heads: Mapping[str, int] = None
    dropout_rate: float = 0.0
    activation: Callable[..., nn.Module] = nn.silu
    


    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        r"""Return the encoder output.

        Arguments:
            x: Input image with shape (*, H, W, C).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, 2*latent_features).
        """
        assert len(self.hid_blocks) == len(self.hid_channels)
        heads = {} if self.heads is None else self.heads

        strides = [2 for k in self.kernel_size]
        padding = [(k // 2, k // 2) for k in self.kernel_size]

        
        out_channels = self.input_shape[-1]
        features = x.shape[-1]

        # Reshape x
        x = x.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        
        # Descend from image to lowest dimension.
        for i, n_blocks in enumerate(self.hid_blocks):

            if i == 0:
                # First convolution shouldn't downsample.
                x = nn.Conv(
                    self.hid_channels[i], kernel_size=self.kernel_size,
                    padding=padding
                )(x)
            else:
                # Downsample.
                x = nn.Conv(
                    self.hid_channels[i], kernel_size=self.kernel_size,
                    padding=padding, strides=strides
                )(x)
            
            # Residual convolutions without downsampling.
            for _ in range(n_blocks):
                x = ResBlock(
                    self.hid_channels[i], self.dropout_rate,
                    activation=self.activation, kernel_size=self.kernel_size,
                    padding=padding
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
            padding=padding
        )(x, train)
        if str(len(self.hid_blocks) - 1) in heads:
            x = AttBlock(
                self.hid_channels[-1],
                self.dropout_rate, heads[str(len(self.hid_blocks) - 1)]
            )(x, train)
        x = ResBlock(
            self.hid_channels[-1], self.dropout_rate,
            activation=self.activation, kernel_size=self.kernel_size,
            padding=padding
        )(x, train)

        # Flatten the image.
        x = nn.Dense(1,use_bias=False)(x)
        x = self.activation(x)
        x = x.reshape(x.shape[0], -1)
        # Project down to latent dimensions.
        x = nn.Dense(features=2 * self.latent_features, use_bias=False)(x)
        
        return x


class UNetDecoder(nn.Module):
    r"""Creates a decoder with the architecture similar to that of the second half of UNet.

    Arguments:
        in_shape: Shape of the image. 
        hid_channels: Number of channels used in each level.
        hid_blocks: Number of block used in each level.
        kernel_size: Defines the kernel size used for all convolutional blocks.
        heads: Number of heads to use for the attention block of each level.
            If a level is not included it will have no attention block.
        dropout_rate: Dropout rate applied in the ResBlock. Attention block has
            no dropout.
        activation: Activation function for resnet blocks.

    Notes:
        
    """
    in_shape: Tuple
    hid_channels: Sequence[int]
    hid_blocks: Sequence[int]
    kernel_size: Sequence[int] = (3, 3)
    heads: Mapping[str, int] = None
    dropout_rate: float = 0.0
    activation: Callable[..., nn.Module] = nn.silu
    

    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        r"""Return Return the encoder output.

        Arguments:
            x: Input vector of latent variables with shape (*, 2*latent_features).
            train: If true, values are passed in training mode.

        Returns:
            Output with shape (*, H, W, C).
        """
        assert len(self.hid_blocks) == len(self.hid_channels)
        heads = {} if self.heads is None else self.heads

        out_channels = self.in_shape[2] #self.out_shape[-1]
        in_features = self.in_shape[0] * self.in_shape[1] * self.in_shape[2]
        
        strides = [2 for k in self.kernel_size]
        padding = [(k // 2, k // 2) for k in self.kernel_size]

        
        # Broadcast vector of latent features to an image of shape (*, init_kernel_size, hid_channels[-1]).
        x = nn.Dense(features=in_features)(x)
        x = x.reshape((-1,) + self.in_shape)
        x = nn.Dense(features=self.hid_channels[-1])(x)
        
        # Ascend from lowest dimension to image.
        for i, n_blocks in reversed(list(enumerate(self.hid_blocks))):
            
            # Residual convolutions without upsampling.
            for _ in range(n_blocks):
                x = ResBlock(
                     self.hid_channels[i], self.dropout_rate,
                     activation=self.activation, kernel_size=self.kernel_size,
                     padding=padding
                )(x, train)

                # Add attention blocks if specified.
                if str(i) in heads:
                    x = AttBlock(
                         self.hid_channels[i], 
                         self.dropout_rate, heads[str(i)]
                     )(x, train)

            if i == 0:
                # Final layer of is fully connected.
                x = nn.Dense(out_channels, use_bias=False)(x)
            else:
                # For all other layers conduct an upsampling iteration.
                x = Resample([float (s) for s in strides], method='nearest')(x)
               
        x = self.activation(x)
        x = x.reshape(-1, in_features)

        return x
        