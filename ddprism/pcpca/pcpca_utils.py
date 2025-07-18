"""Utility functions for PCPCA."""
from typing import Dict, Tuple, Optional

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
        return 2.0 * jnp.sum(jnp.log(jnp.diag(chol))) + regularization
    def logdet():
        return jnp.log(jnp.linalg.det(regularized_matrix) +  regularization)
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
    # Unpack parameters.
    weights, log_sigma, mu = (
        params['weights'], params['log_sigma'], params['mu']
    )
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
    x_residual = x_obs - jnp.matmul(x_a_mat, mu[..., None]).squeeze(-1)
    loss_value += -0.5 * jnp.mean(
        jax.vmap(stable_quadratic, in_axes=(0, 0, None))(
            c_mat, x_residual, regularization
        )
    )

    # Loss terms from the background signal.

    loss_value += 0.5 * gamma * jnp.mean(
        jax.vmap(log_det_cholesky, in_axes=(0, None))(d_mat, regularization)
    )

    y_residual = y_obs - jnp.matmul(y_a_mat, mu[..., None]).squeeze(-1)
    loss_value += 0.5 * gamma * jnp.mean(
        jax.vmap(stable_quadratic, in_axes=(0, 0, None))(
            d_mat, y_residual, regularization
        )
    )

    return loss_value


def loss_grad(
    params: Dict[str, jnp.ndarray], x_obs: jnp.ndarray,
    y_obs: jnp.ndarray, x_a_mat: jnp.ndarray, y_a_mat: jnp.ndarray,
    gamma: float, regularization: float = 1e-6
) -> Dict[str, jnp.ndarray]:
    """Loss function for estimating PCPCA parameters.

    Args:
        params: Parameters of the PCPCA model. Dict with keys 'weights' and
            'log_sigma', 'mu'.
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
    # Unpack parameters.
    weights, log_sigma, mu = (
        params['weights'], params['log_sigma'], params['mu']
    )
    sigma = jnp.exp(log_sigma)

    # Calculate the residuals.
    x_residual = x_obs - jnp.matmul(x_a_mat, mu[..., None]).squeeze(-1)
    y_residual = y_obs - jnp.matmul(y_a_mat, mu[..., None]).squeeze(-1)

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
    c_inv_x = stable_solve_vmap(c_mat, x_residual[:, :, None], regularization)
    d_inv_y = stable_solve_vmap(d_mat, y_residual[:, :, None], regularization)
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

    # Add gradient of mu.
    # Start with linear terms.
    c_inv_x = stable_solve_vmap(c_mat, x_obs[:, :, None], regularization)
    d_inv_y = stable_solve_vmap(d_mat, y_obs[:, :, None], regularization)
    grad_mu = jnp.mean(
        jnp.einsum('ijk,ijl->il', c_inv_x, x_a_mat),
        axis=0
    )
    grad_mu += - gamma * jnp.mean(
        jnp.einsum('ijk,ijl->il', d_inv_y, y_a_mat),
        axis=0
    )

    # Add quadratic terms.
    c_mat_inv_xa = stable_solve_vmap(
        c_mat, jnp.einsum('ijk,k->ij', x_a_mat, mu), regularization
    )
    d_mat_inv_ya = stable_solve_vmap(
        d_mat, jnp.einsum('ijk,k->ij', y_a_mat, mu), regularization
    )
    grad_mu += -jnp.mean(
        jnp.einsum('ij,ijl->il', c_mat_inv_xa, x_a_mat), axis=0
    )
    grad_mu += gamma * jnp.mean(
        jnp.einsum('ij,ijl->il', d_mat_inv_ya, y_a_mat), axis=0
    )

    return {
        'weights': grad_weights, 'log_sigma': grad_log_sigma,
        'mu': grad_mu
    }


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
    weights, log_sigma, mu = (
        params['weights'], params['log_sigma'], params['mu']
    )
    sigma_sq = jnp.exp(2 * log_sigma)

    # Compute posterior covariance.
    prior_precision = jnp.linalg.inv(
        weights @ weights.T + regularization * jnp.eye(weights.shape[0])
    )
    sigma_post = jnp.linalg.inv(
        (1/sigma_sq) * a_mat.T @ a_mat + prior_precision
    )

    # Compute posterior mean
    mean_post = sigma_post @ (
        1 / sigma_sq * a_mat.T @ y_obs + prior_precision @ mu
    )

    return mean_post, sigma_post


def mle_params(
    x_obs: jnp.ndarray, y_obs: jnp.ndarray, gamma: float, latent_dim: int,
    sigma: Optional[float] = None
):
    """Compute the MLE parameters for the PCPCA model.

    Args:
        x_obs: Observed data with enriched signal.
        y_obs: Observed data with only background signal.
        gamma: Multiplier for contrastive term in loss function.
        latent_dim: Dimension of the latent space.
        sigma: Variance of the observation. Will be estimated if not provided.

    Returns:
        Dictionary with keys 'mu', 'weights', and 'log_sigma' containing the
        MLE parameters for the PCPCA model.
    """
    # Get the data dimension for sigma optimization.
    data_dim = x_obs.shape[1]
    if data_dim <= latent_dim:
        raise ValueError(
            "Latent space dimension should be less than or equal to the data "
            "dimension."
        )

    # Number of target and background samples.
    n_enr, n_bkg = x_obs.shape[0], y_obs.shape[0]

    # Compute ML estimator of the mean of the target distribution with enriched
    # signal.
    mu = (jnp.sum(x_obs, axis=0) - gamma * jnp.sum(y_obs, axis=0))
    mu /= (n_enr - gamma * n_bkg)
    x_residual = x_obs - mu
    y_residual = y_obs - mu

    # Compute unormalized "sample covariance" matrix of the observed data.
    cov_x = x_residual.T @ x_residual
    cov_y = y_residual.T @ y_residual
    cov = cov_x - gamma * cov_y

    # Get the largest latent_dim eigenvalues and eigenvectors.
    l_mat, q_mat = jnp.linalg.eigh(cov)
    # Ensure eigenvalues are positive.
    l_mat = jnp.maximum(l_mat, 1e-6)
    l_mat_largest = l_mat[-latent_dim:]
    q_mat_largest = q_mat[:, -latent_dim:]

    # Get the normalized eigenvalues.
    eig_values_largest = jnp.diag(l_mat_largest) / (n_enr - gamma * n_bkg)

    # Use MLE for sigma^2 if it is not provided.
    if sigma is not None:
        sigma2_mle = sigma ** 2
    else:
        sigma2_mle = 1 / (data_dim - latent_dim) / (n_enr - gamma * n_bkg)
        sigma2_mle *= jnp.sum(l_mat[:-latent_dim])

    # Compute the MLE for the weights. Set negative eigenvalues to zero.
    weights_mle = q_mat_largest @ jnp.sqrt(
        jnp.maximum(eig_values_largest - sigma2_mle * jnp.eye(latent_dim), 0.0)
    )

    return {
        'mu' : mu, 'weights' : weights_mle, 'log_sigma': jnp.log(sigma2_mle) / 2
    }
