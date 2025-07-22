"""Metrics for evaluating the quality of samples."""

from jax import Array
import jax.numpy as jnp
import ot
from pqm import pqm_pvalue


def pq_mass(
    dist_1: Array, dist_2: Array, re_tessellation: int = 1000, **kwargs
) -> Array:
    r"""Computes PQMass p values: https://arxiv.org/abs/2402.04355.

    Arguments:
        dist_1: Samples from the first distribution.
        dist_2: Samples from the second distribution.
        re_tessellation: Number of re-tessellations to use for PQMass.
        **kwargs: Additional arguments for computing PQMass.

    Returns:
        Mean p values from PQMass.

    """
    # Flatten outputs if not already flat.
    dist_1 = dist_1.reshape(dist_1.shape[0], -1)
    dist_2 = dist_2.reshape(dist_2.shape[0], -1)

    chi2_vals = pqm_pvalue(
        dist_1, dist_2, re_tessellation=re_tessellation, **kwargs
    )

    return jnp.mean(jnp.array(chi2_vals))


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


def psnr(
    u: Array, v: Array, max_spread, mean: bool=True
) -> Array:
    r"""Computes peak signal-to-noise ratio (PSNR) between two paired samples.

    Arguments:
        u: True signal.
        v: Predicted signal.
        max_spread: Maximum possible dynamic range of the true signal.
        mean: If true, return the mean over all samples.

    Returns:
       Peak signal to noise ratio of u and v.
    """
    mse = jnp.mean((u - v)**2, axis=-1)
    psnr_val = 20 * (jnp.log10(max_spread) - jnp.log10(jnp.sqrt(mse) + 1e-6))

    # Return the mean if requested.
    if mean:
        psnr_val = psnr_val.mean()
    return psnr_val
