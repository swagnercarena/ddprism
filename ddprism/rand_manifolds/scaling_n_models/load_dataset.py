"""Load the dataset for 1D model."""
import jax
import jax.numpy as jnp

from ddprism import linalg
from ddprism.rand_manifolds import random_manifolds


def get_dataset(rng, config):
    """Return the dataset specified in the config."""

    # Rng per source.
    rng_x, rng_a, rng_y = jax.random.split(rng, 3)

    # Generate our sources.
    x_all = []
    # Split within loop to ensure that the sources generated are consistent
    # across different numbers for config.n_sources.
    if isinstance(config.alpha, list):
        assert len(config.alpha) == config.n_sources
        alpha = config.alpha
    else:
        alpha = [config.alpha for i in range(config.n_sources)]

    if isinstance(config.phase, list):
        assert len(config.phase) == config.n_sources
        phase = config.phase
    else:
        phase = [config.phase for i in range(config.n_sources)]

    for _ in range(config.n_sources):
        rng_x, rng = jax.random.split(rng_x, 2)
        x_all.append(
            random_manifolds.generate_x(
                rng_x, config.sample_size, man_dim=1,
                feat_dim=config.feat_dim, alpha=alpha[_], phase=phase[_], normalize=True
            )
        )
    x_all = jnp.stack(x_all, axis=1)

    # Generate our A matrices, assuming the source matrix does not vary.
    A_all = random_manifolds.generate_A(
        rng_a, config.sample_size, obs_dim=config.obs_dim,
        feat_dim=config.feat_dim
    )[:, None]
    A_all = jnp.tile(A_all, (1, config.n_sources, 1, 1))

    # Generate our observations
    y_all, cov_y_all = [], []
    for n in range(config.n_sources):
        y, cov_y = random_manifolds.generate_y(
            rng_y, A_all[:, :(n+1)], x_all[:, :(n+1)], noise=config.sigma_y
        )

        # Put the covariance in the DPLR representation.
        cov_y = linalg.DPLR(diagonal=jnp.tile(cov_y[None], (y.shape[0], 1)))

        y_all.append(y)
        cov_y_all.append(cov_y)

    y_all = jnp.stack(y_all, axis=1)
    # Can't stack cov_y all since it's a DPLR matrix.

    return x_all, A_all, y_all, cov_y_all
