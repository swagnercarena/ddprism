"""Module for generating random 1D manifold data. Implementation follows
https://github.com/francois-rozet/diffusion-priors/manifold/utils.py closely.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array


def generate_x(
    key: Array, n_samples: int, man_dim: int = 1, feat_dim: int = 3,
    alpha: float = 3.0, epsilon: float = 1e-3, phase: Optional[float] = None,
    normalize: bool = True
) -> Array:
    """Samples points from a smooth random manifold.

    References:
        https://github.com/fzenke/randman,
        https://github.com/francois-rozet/diffusion-priors

    Arguments:
        key: Random key.
        n_samples: Number of samples to draw.
        man_dim: Manifold dimension.
        feat_dim: Feature space dimension.
        alpha: Smoothness coefficient.
        epsilon: Precision paramter to determine the maximum frequency cutoff.
        phase: Phase of sin term underlying manifold.
        normalize: Normalize x to be between -2.0 and 2.0.

    Returns:
        Manifold draws.
    """

    key_params, key_z = jax.random.split(key, 2)

    # Cutoff frequencies above a threshold, controling smoothness.
    cutoff = math.ceil(epsilon ** (-1 / alpha))

    # Frequencies from 1 to cutoff.
    freq = jnp.arange(cutoff) + 1

    # Generate our a, b, and c coefficients.
    a, b, c = jax.random.uniform(key_params, (3, feat_dim, man_dim, cutoff))
    # Replace the c coefficient with the fixed phase if provided.
    if phase is not None:
        c = jnp.ones_like(c) * phase

    # Random positions to sample along the manifold.
    z = jax.random.uniform(key_z, (n_samples, 1, man_dim, 1))

    # Transform to x from the position along the manifold.
    x = a / (freq ** alpha) * jnp.sin(2 * jnp.pi * (freq * b * z + c))
    # Sum over the sines.
    x = jnp.sum(x, axis=-1) # shape (n_samples, feat_dim, m)
    # Product over the sums.
    x = jnp.prod(x, axis=-1) # shape (n_samples, feat_dim)

    if normalize:
        x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
        x = 4.0 * x - 2.0

    return x


def generate_A(
    key: Array, n_samples: int, obs_dim: int, feat_dim: int
) -> Array:
    """Samples random measurement matrices A of dim (obs_dim, feat_dim).

    Arguments:
        key: Random key.
        n_samples: Number of samples to draw.
        obs_dim: Observation space dimension.
        feat_dim: Feature space dimension.

    Returns:
        Measurement matrices.
    """

    A = jax.random.normal(key, (n_samples, obs_dim, feat_dim))
    A = A / jnp.linalg.norm(A, axis=-1, keepdims=True)

    return A


def measure(A: Array, x: Array) -> Array:
    """Return a measurement dealing with the batch dimension.

    Arguments:
        A: Measurement matrix
        x: Manifold draws.

    Returns:
        Measured manifold draws.
    """
    return jnp.einsum('...ij,...j', A, x)


def generate_y(
    key: Array, A_all: Array, x_all: Array, noise: float = 1e-2
) -> Array:
    """Generates observations y by applying the linear operator and noise.

    Arguments:
        key: Random key.
        A_all: Measurement matrices for each source. Shape (*, N_s, M, N).
        x: Manifold draws for each source. Shape (*, N_s, N).
        noise: Noise level.

    Returns:
        Measurements and associated measurement covariance. Shape (*, M)
    """
    sample_shape, obs_dim = A_all.shape[:-3], A_all.shape[-2]

    # Apply linear operator
    y = jnp.sum(measure(A_all, x_all), axis=-2)

    # Generate measurement covariance.
    cov_y = noise ** 2 * jnp.ones(obs_dim)
    y = y + jnp.sqrt(cov_y) * jax.random.normal(key, sample_shape + (obs_dim,))
    return y, cov_y
