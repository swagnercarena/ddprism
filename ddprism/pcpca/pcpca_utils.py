"""Utility functions for PCPCA."""
from math import log
from typing import Dict

import jax
import jax.numpy as jnp
from numpy import c_

from ddprism.rand_manifolds import random_manifolds
from ddprism import linalg


def loss(
    params: Dict[str, jnp.ndarray], x_obs: jnp.ndarray,
    y_obs: jnp.ndarray, x_a_mat: jnp.ndarray,
    y_a_mat: jnp.ndarray, gamma: float
) -> jnp.ndarray:
    """Loss function for estimating PCPCA parameters.

    Reference: Section 7.1 of of https://arxiv.org/pdf/2012.07977.

    Args:
        params: Parameters of the PCPCA model. Dict with keys 'weights' and
            'log_sigma'.
        x_obs: Observed data with enriched signal.
        y_obs: Observed data with only background signal.
        x_a_mat: Transformation matrix for enriched signal.
        y_a_mat: Transformation matrix for background signal.
        gamma: Multiplier for contrastive term in loss function.
    """

    weights, log_sigma = params['weights'], params['log_sigma']
    sigma = jnp.exp(log_sigma)

    c_mat = jax.vmap(compute_aux_matrix, in_axes=(None, 0, None))(
        weights, x_a_mat, sigma
    )
    d_mat = jax.vmap(compute_aux_matrix, in_axes=(None, 0, None))(
        weights, y_a_mat, sigma
    )

    # Loss terms from the enriched signal.
    loss = - 0.5 * jnp.mean(
        jnp.log(jnp.linalg.det(c_mat))
    )
    loss += -0.5 * jnp.mean(
        x_obs[:, None] @ jnp.linalg.inv(c_mat) @ x_obs[:, :, None]
    )

    # Loss terms from the background signal.
    loss += 0.5 * gamma * jnp.mean(
        jnp.log(jnp.linalg.det(d_mat))
    )
    loss += 0.5 * gamma * jnp.mean(
        y_obs[:, None] @ jnp.linalg.inv(d_mat) @ y_obs[:, :, None]
    )

    return loss


def loss_grad(
    params: Dict[str, jnp.ndarray], x_obs: jnp.ndarray,
    y_obs: jnp.ndarray, x_a_mat: jnp.ndarray,
    y_a_mat: jnp.ndarray, gamma: float
) -> Dict[str, jnp.ndarray]:
    """Loss function for estimating PCPCA parameters.

    Reference: Section 7.1 of of https://arxiv.org/pdf/2012.07977.

    Args:
        params: Parameters of the PCPCA model. Dict with keys 'weights' and
            'log_sigma'.
        x_obs: Observed data with enriched signal.
        y_obs: Observed data with only background signal.
        x_a_mat: Transformation matrix for enriched signal.
        y_a_mat: Transformation matrix for background signal.
        gamma: Multiplier for contrastive term in loss function.

    Returns:
        Dictionary with keys 'weights' and 'log_sigma' containing the gradients
        of the weights and log sigma respectively.
    """

    weights, log_sigma = params['weights'], params['log_sigma']
    sigma = jnp.exp(log_sigma)

    c_mat = jax.vmap(compute_aux_matrix, in_axes=(None, 0, None))(
        weights, x_a_mat, sigma
    )
    c_mat_inv = jnp.linalg.inv(c_mat)
    d_mat = jax.vmap(compute_aux_matrix, in_axes=(None, 0, None))(
        weights, y_a_mat, sigma
    )
    d_mat_inv = jnp.linalg.inv(d_mat)

    # Weights gradient term.
    grad_weights = -jnp.mean(
        jnp.transpose(x_a_mat, axes=(0, 2, 1)) @ c_mat_inv @
        x_a_mat @ weights, axis=0
    )
    grad_weights += gamma * jnp.mean(
        jnp.transpose(y_a_mat, axes=(0, 2, 1)) @ d_mat_inv @
        y_a_mat @ weights, axis=0
    )
    grad_weights += - jnp.mean(
        jnp.transpose(x_a_mat, axes=(0, 2, 1)) @ c_mat_inv @
        x_obs[:, :, None] @ x_obs[:, None, :] @ c_mat_inv @ x_a_mat @ weights,
        axis=0
    )
    grad_weights += gamma * jnp.mean(
        jnp.transpose(y_a_mat, axes=(0, 2, 1)) @ d_mat_inv @
        y_obs[:, :, None] @ y_obs[:, None, :] @ d_mat_inv @ y_a_mat @ weights,
        axis=0
    )

    # Gradient for log sigma.
    grad_sigma = - jnp.mean(jnp.linalg.trace(c_mat_inv))
    grad_sigma += gamma * jnp.mean(jnp.linalg.trace(d_mat_inv))
    grad_sigma += jnp.mean(jnp.linalg.trace(
        c_mat_inv @ x_obs[:, :, None] @ x_obs[:, None, :] @ c_mat_inv
    ))
    grad_sigma += - gamma * jnp.mean(jnp.linalg.trace(
        d_mat_inv @ y_obs[:, :, None] @ y_obs[:, None, :] @ d_mat_inv
    ))
    grad_sigma *= sigma ** 2

    return {'weights': grad_weights, 'log_sigma': grad_sigma}


def compute_aux_matrix(
    weights: jnp.ndarray, a_mat: jnp.ndarray, sigma: float
) -> jnp.ndarray:
    """Compute the auxillary matrix for PCPCA loss function.

    Args:
        weights: Weights from latent space to signal space.
        a_mat: Transformation matrix for the observation.
        sigma: Variance of the observation.

    Returns:
        Auxillary matrix for PCPCA loss function.
    """
    mat_prod = a_mat @ weights @ weights.T @ a_mat.T
    mat_prod = mat_prod + sigma*jnp.eye(a_mat.shape[0])

    return mat_prod


# def impute_missing_data(params, X_obs, L, M, P, dim):
#     A, B, C, F = compute_aux_matrices(params, L, M, P, dim)
#     x_mean = (F @ jnp.linalg.inv(A) @ X_obs[..., None]).squeeze(axis=-1)
#     x_cov = C - F @ jnp.linalg.inv(A) @ jnp.matrix_transpose(F)
#     return x_mean, x_cov