"""Utility functions for PCPCA."""
from typing import Dict, Tuple

import jax
import jax.numpy as jnp


def log_det_cholesky(
    matrix: jnp.ndarray, regularization: float = 1e-6
) -> jnp.ndarray:
    """Compute log determinant using Cholesky decomposition for numerical stability.

    Args:
        matrix: Positive definite matrix
        regularization: Small value to add to diagonal for numerical stability

    Returns:
        Log determinant of the matrix
    """
    # Add regularization.
    regularized_matrix = matrix + regularization * jnp.eye(matrix.shape[-1])

    # Cholesky decomposition and validity.
    chol = jnp.linalg.cholesky(regularized_matrix)
    chol_valid = jnp.all(jnp.isfinite(chol))

    # If Cholesky succeeded, use it; otherwise fallback to SVD
    def cholesky_logdet():
        return 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    def logdet():
        return jnp.log(jnp.linalg.det(regularized_matrix))
    return jnp.where(chol_valid, cholesky_logdet(), logdet())


def stable_solve(
    matrix: jnp.ndarray, vector: jnp.ndarray, regularization: float = 1e-6
) -> jnp.ndarray:
    """Solve Ax = b in a numerically stable way.

    Args:
        matrix: Inverted matrix.
        vector: Vector to multiply by matrix inverse.
        regularization: Small value to add to diagonal for numerical stability

    Returns:
        Solution x
    """
    # Add regularization to ensure the matrix is well-conditioned.
    regularized_matrix = matrix + regularization * jnp.eye(matrix.shape[-1])
    return jnp.linalg.solve(regularized_matrix, vector)


def stable_quadratic(
    matrix: jnp.ndarray, vector: jnp.ndarray, regularization: float = 1e-6
) -> jnp.ndarray:
    """Compute x^T A^{-1} x using stable solve.

    Args:
        matrix: Matrix to invert.
        vector: Vector to multiply by matrix inverse.
        regularization: Small value to add to diagonal for numerical stability

    Returns:
        Quadratic form x^T A^{-1} x
    """
    # Solve A * y = x instead of computing A^{-1} * x
    y = stable_solve(matrix, vector, regularization)
    return jnp.sum(vector * y, axis=-1)


def loss(
    params: Dict[str, jnp.ndarray], x_obs: jnp.ndarray,
    y_obs: jnp.ndarray, x_a_mat: jnp.ndarray, y_a_mat: jnp.ndarray,
    gamma: float, regularization: float = 1e-6
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
        regularization: Small value to add to diagonal for numerical stability.
    """

    weights, log_sigma = params['weights'], params['log_sigma'], params['mu_x'], params['mu_y']
    sigma = jnp.exp(log_sigma)

    c_mat = jax.vmap(compute_aux_matrix, in_axes=(None, 0, None))(
        weights, x_a_mat, sigma
    )
    d_mat = jax.vmap(compute_aux_matrix, in_axes=(None, 0, None))(
        weights, y_a_mat, sigma
    )

    # Loss terms from the enriched signal.
    loss_value = - 0.5 * jnp.mean(
        jax.vmap(log_det_cholesky, in_axes=(0, None))(c_mat, regularization)
    )
    loss_value += -0.5 * jnp.mean(
        jax.vmap(stable_quadratic, in_axes=(0, 0, None))(
            c_mat, x_obs - jnp.matmul(x_a_mat, mu_x[..., None]).squeeze(-1), regularization
        )
    )

    # Loss terms from the background signal.
    loss_value += 0.5 * gamma * jnp.mean(
        jax.vmap(log_det_cholesky, in_axes=(0, None))(d_mat, regularization)
    )
    loss_value += 0.5 * gamma * jnp.mean(
        jax.vmap(stable_quadratic, in_axes=(0, 0, None))(
            d_mat, y_obs - jnp.matmul(y_a_mat, mu_y[..., None]).squeeze(-1), regularization
        )
    )

    return loss_value


def loss_grad(
    params: Dict[str, jnp.ndarray], x_obs: jnp.ndarray,
    y_obs: jnp.ndarray, x_a_mat: jnp.ndarray, y_a_mat: jnp.ndarray,
    gamma: float, regularization: float = 1e-6
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
        regularization: Small value to add to diagonal for numerical stability.

    Returns:
        Dictionary with keys 'weights' and 'log_sigma' containing the gradients
        of the weights and log sigma respectively.
    """

    weights, log_sigma = params['weights'], params['log_sigma']
    sigma = jnp.exp(log_sigma)

    c_mat = jax.vmap(compute_aux_matrix, in_axes=(None, 0, None))(
        weights, x_a_mat, sigma
    )
    d_mat = jax.vmap(compute_aux_matrix, in_axes=(None, 0, None))(
        weights, y_a_mat, sigma
    )

    stable_solve_vmap = jax.vmap(stable_solve, in_axes=(0, 0, None))

    # Weight gradients terms. Start with gradient of log det terms.
    c_mat_inv_weights = stable_solve_vmap(
        c_mat, jnp.einsum('ijk,kl->ijl', x_a_mat, weights), regularization
    )
    d_mat_inv_weights = stable_solve_vmap(
        d_mat, jnp.einsum('ijk,kl->ijl', y_a_mat, weights), regularization
    )
    grad_weights = -jnp.mean(
        jnp.einsum('ijk,ijl->ikl', x_a_mat, c_mat_inv_weights), axis=0
    )
    grad_weights += gamma * jnp.mean(
        jnp.einsum('ijk,ijl->ikl', y_a_mat, d_mat_inv_weights), axis=0
    )

    # Add gradient of quadratic terms.
    c_inv_x = stable_solve_vmap(c_mat, x_obs[:, :, None], regularization)
    d_inv_y = stable_solve_vmap(d_mat, y_obs[:, :, None], regularization)
    grad_weights += jnp.mean(
        jnp.einsum('ijk,ijl,iml,imn,np->ikp',
                  x_a_mat, c_inv_x, c_inv_x, x_a_mat, weights),
        axis=0
    )
    grad_weights += -gamma * jnp.mean(
        jnp.einsum('ijk,ijl,iml,imn,np->ikp',
                  y_a_mat, d_inv_y, d_inv_y, y_a_mat, weights),
        axis=0
    )

    # Gradient for log sigma.
    # Calculate inverse trace using inverse of the eigenvalues.
    c_inv_trace = jnp.mean(jnp.sum(
        1.0 / jnp.linalg.eigvals(
            c_mat + regularization * jnp.eye(c_mat.shape[-1])
        ).real, axis=-1
    ))
    d_inv_trace = jnp.mean(jnp.sum(
        1.0 / jnp.linalg.eigvals(
            d_mat + regularization * jnp.eye(d_mat.shape[-1])
        ).real, axis=-1
    ))
    grad_log_sigma = - c_inv_trace + gamma * d_inv_trace

    # Add quadratic form contributions
    grad_log_sigma += jnp.mean(
        jnp.linalg.trace(jnp.einsum('ijk,ilk->ijl', c_inv_x, c_inv_x))
    )
    grad_log_sigma += - gamma * jnp.mean(
        jnp.linalg.trace(jnp.einsum('ijk,ilk->ijl', d_inv_y, d_inv_y))
    )

    grad_log_sigma *= sigma ** 2

    return {'weights': grad_weights, 'log_sigma': grad_log_sigma}


def compute_aux_matrix(
    weights: jnp.ndarray, a_mat: jnp.ndarray, sigma: jnp.ndarray
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
    mat_prod += (sigma ** 2) * jnp.eye(a_mat.shape[0])

    return mat_prod


def calculate_posterior(
    params: Dict[str, jnp.ndarray], y_obs: jnp.ndarray, a_mat: jnp.ndarray,
    regularization: float = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the mean and covariance for the signal posterior.

    Args:
        params: Parameters of the PCPCA model. Dict with keys 'weights' and
            'log_sigma'.
        y_obs: Observed data with enriched signal.
        a_mat: Transformation matrix for the observation.
        regularization: Small value to add to diagonal for numerical stability.

    Returns:
        Tuple of mean and covariance for the signal posterior.

    Notes:
        See paper for derivation.
    """
    weights, log_sigma = params['weights'], params['log_sigma'], params['mu_x'], params['mu_y']
    sigma_sq = jnp.exp(2 * log_sigma)

    # First compute the covariance matrix.
    prior_precision = jnp.linalg.inv(
        weights @ weights.T + regularization * jnp.eye(weights.shape[0])
    )
    sigma_post = jnp.linalg.inv(
        (1/sigma_sq) * a_mat.T @ a_mat + prior_precision
    )

    # Compute posterior mean
    mean_post = sigma_post @ ( prior_precision @ mu_x + (1/sigma_sq) * a_mat.T @ y_obs)

    return mean_post, sigma_post
