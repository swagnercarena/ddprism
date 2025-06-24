r"""Helper functions for Diffusion Model Sampling and EM Training."""
from functools import partial
from typing import Mapping, Optional, Sequence, Tuple

from einops import rearrange
from flax.training.train_state import TrainState
import jax
from jax import Array
import jax.experimental.sparse
import jax.numpy as jnp
import numpy as np
import ot

from ddprism import diffusion
from ddprism import sampling
from ddprism.linalg import DPLR


def sample(
    key: Array, state: TrainState, params: Mapping[str, Array],
    sample_shape: Sequence[int], feature_shape: int, steps: int,
    sampler: Optional[str] = 'ddpm', **kwargs
):
    """Sample from the SDE using the specified sampling scheme.

    Arguments:
        key: Rng key for SDE sampling.
        state: Trained diffusion model.
        params: Params to use with the trained model.
        sample_shape: Desired shape of samples.
        feature_shape: Feature shape of the x samples.
        steps: Number of sampling steps to use.
        sampler: Sampling scheme to use. Default is DDPM.

    Returns:
        Samples from the reverse SDE.
    """
    # Draw random samples from the p(x_1) distribution to evolve backwards.
    z = jax.random.normal(key, sample_shape + (feature_shape,))

    # Sample from p(x_1) by adding appropriate Gaussian noise to mean 0.0.
    x1 = state.apply_fn(params, jnp.zeros_like(z), z, 1.0, method='sde_x_t')

    x0 = sampling.sampling(
        key, state, params, x1, steps, sampler=sampler, **kwargs
    )
    return x0


def sample_gibbs(
    key: Array, state: TrainState, params: Mapping[str, Array],
    initial_samples: Array, steps: int, sampler: Optional[str] = 'ddpm',
    gibbs_rounds: Optional[int] = 1, **kwargs
) -> Array:
    """Posterior Gibbs sample from a set of N distributions.

    Arguments:
        key: Rng key for SDE sampling.
        state: Joint posterior sampling state that will be used to sample
            each individual posterior model.
        params: Params to use with the trained model. Must include observation
            variables.
        initial_samples: Initial Gibbs samples. Shape (*, n_models * features).
        steps: Number of sampling steps to use.
        sampler: Sampling scheme to use. Default is DDPM.
        gibbs_rounds: Number of rounds of Gibbs sampling to conduct.

    Returns:
        Samples from the joint posterior. Shape (*, n_models * features).

    Notes:
        Assumes posterior sampling and will fail if observations, A matrix, and
        covariance matrix are not in params['variables'].
    """
    # Check that the observation variables are present. Otherwise the Gibbs
    # sampling code will fail.
    assert 'variables' in params
    assert 'A' in params['variables']
    assert 'y' in params['variables']

    # Change the samples shape to simplify sampling.
    n_models = state.apply_fn(params, method='n_models')
    samples = rearrange(initial_samples, '... (M F) -> ... M F', M=n_models)
    feature_shape = samples.shape[-1]
    sample_shape = samples.shape[:-2]

    # Create a variables for the residuals.
    matmul_vmap = jax.vmap(diffusion.matmul, in_axes=[-3, -2], out_axes=-2)
    residual = params['variables']['y'] - jnp.sum(
        matmul_vmap(params['variables']['A'], samples), axis=-2
    )

    def _gibbs_round(carry, key):
        """Individual Gibbs sampling round."""
        samples, residual = carry

        # Iterate through all the models. # TODO: Figure out how to make
        # random permutations jit-compilable.
        for index in np.arange(n_models):

            key, key_normal = jax.random.split(key)

            # Get a version of the state that will pass the index with every
            # apply function  call.
            state_single = state.replace(
                apply_fn=partial(state.apply_fn, index=index)
            )

            # Add the current sample back to the residual.
            residual += diffusion.matmul(
                params['variables']['A'][..., index, :, :], samples[..., index, :]
            )

             # Draw random samples from the t=1 distribution
            z = jax.random.normal(key_normal, sample_shape + (feature_shape,))

            # Update the residuals.
            params_dist = {
                'params': params['params'],
                'variables': {
                    'A': params['variables']['A'],
                    'cov_y': params['variables']['cov_y'],
                    'y': residual
                }
            }

            x1 = state_single.apply_fn(
                params_dist, jnp.zeros_like(z), z, 1.0, method='sde_x_t',
            )
            x0 = sampling.sampling(
                key, state_single, params_dist, x1, steps, sampler=sampler,
                 **kwargs
            )

            # Update initial samples and residual
            samples = samples.at[..., index, :].set(x0)
            # Remove the new best guess from the observation.
            residual -= diffusion.matmul(
                params['variables']['A'][..., index, :, :], x0
            )

        return (samples, residual), None

    # One random key per Gibbs rounds.
    keys_gibbs = jax.random.split(key, gibbs_rounds)
    (samples, residual), _ = jax.lax.scan(
        _gibbs_round, (samples, residual), keys_gibbs
    )

    return rearrange(samples, '... M F -> ... (M F)')


def ppca(key: Array, x: Array, rank: int = 1) -> Tuple[Array, DPLR]:
    r"""Fits :math:`(\mu_x, \Sigma_x)` by probabilistic PCA.

    Covariance matrix math:`\Sigma_x` is computed with a diagonal plus low-rank
    (DPLR) matrix approximation.

    Implementation follows
    https://github.com/francois-rozet/diffusion-priors/blob/master/priors/common.py
    closely.

    Arguments:
        key: jax PRNG key.
        x: Samples to calculate the mean and covariance from.
        rank: Maximum rank of DPLR covariance matrix.

    Returns:
        Mean and DPLR covariance matrix.

    References:
        https://www.miketipping.com/papers/met-mppca.pdf
    """

    samples, features = x.shape

    # Compute ML estimator of the mean of the distribution.
    # Follows Eq. (12) in the reference.
    mu_x = jnp.mean(x, axis=0)
    x = x - mu_x

    # Computes sample covariance matrix of the observed data.
    # Corresponds to matrix S in Eq. (11) in the reference.
    if samples < features:
        c_mat = x @ x.T / samples
    else:
        c_mat = x.T @ x / samples

    # Computes ML estimator of the model covariance matrix, Cov = D + U @ V
    # (Eq. (7) in the reference), where D = sigma^2 * I, U = W, V = W^T.
    # l_mat are the n=rank largest eigenvalues of the covariance matrix.
    # q_mat are the eigevectors corresponding to l_mat.
    if rank < len(c_mat) // 5:
        q_mat = jax.random.normal(key, (len(c_mat), rank))
        l_mat, q_mat, _ = (
            jax.experimental.sparse.linalg.lobpcg_standard(c_mat, q_mat)
        )
    else:
        l_mat, q_mat = jnp.linalg.eigh(c_mat)
        l_mat, q_mat = l_mat[-rank:], q_mat[:, -rank:]

    # Deal with having too few samples.
    if samples < features:
        q_mat = x.T @ q_mat
        q_mat = q_mat / jnp.linalg.norm(q_mat, axis=0)

    # Computes the diagonal part of the covariance matrix.
    # Assuming isotropic noise model, each entry of the covariance matrix is
    # ML estimator of the noise covariance. Follows Eq. (15) in the reference,
    # using Tr(c_mat) = Sum_i(l_i), where {l_i} are the eigenvalues of c_mat.
    if rank < features:
        diagonal = (jnp.trace(c_mat) - jnp.sum(l_mat)) / (features - rank)
    else:
        diagonal = jnp.asarray(1e-6)

    # Computes the weight matrix W, following Eq. (14) in the reference.
    u_mat = q_mat * jnp.sqrt(jnp.maximum(l_mat - diagonal, 0.0))

    # Cas the estimated covariance matrix as a DPLR class.
    cov_x = DPLR(diagonal * jnp.ones(features), u_mat, u_mat.T)

    return mu_x, cov_x


def sinkhorn_divergence(
    u: Array, v: Array, lam: float = 1e-3, maxiter: int = 1024,
    epsilon: float = 1e-2, enforce_positive: bool = False
) -> float:
    r"""Computes the Sinkhorn divergence between samples from two measures.

    References:
        Faster Wasserstein Distance Estimation with the Sinkhorn Divergence
        (Chizat et al., 2020) https://arxiv.org/abs/2006.08172

    Arguments:
        u: Samples from the first measure.
        v: Samples from the second measure.
        lam: Regularization term of transport calculation.
        maxiter: Maximum number of iterations to use for sinkhorn algorithm.
        epsilon: Stop threshold on error for sinkhorn algorithm.
        enforce_positive: Enforce that the returned sinkhole divergence is
            positive despite being drawn from an estimator.

    Returns:
        Sinkhorn divergence between the two samples.
    """
    half_v = len(v) // 2
    half_u = len(u) // 2

    def transport(u, v):
        # Use sinkhorn_log method for better performance with small epsilon.
        return ot.sinkhorn2(
            a=jnp.asarray(()), b=jnp.asarray(()), M=ot.dist(u, v),
            reg=lam, numItermax=maxiter, stopThr=epsilon,
            method='sinkhorn_log',
        )

    divergence = transport(u, v)
    divergence -= transport(u[:half_u], u[-half_u:]) / 2.0
    divergence -= transport(v[:half_v], v[-half_v:]) / 2.0

    if enforce_positive:
        divergence = jnp.maximum(divergence, 0.0)

    return divergence
