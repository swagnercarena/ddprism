"""Load the datasets for the TSZ CMB."""
from typing import Tuple

from einops import rearrange
import h5py
import jax
import jax.numpy as jnp

from ddprism import linalg

def load_randoms(
    config: dict, randoms_path: str
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load a randoms dataset."""

    with h5py.File(randoms_path, 'r') as f:
        rand_obs = jnp.array(f['patches'][:config.n_train])
        vec_map = jnp.array(f['vecs'][:config.n_train])

    # Transform to the desired dimensions.
    rand_obs = rearrange(
        rand_obs, '(B P S) N C -> B P S N C', P=jax.device_count(),
        S=config.sample_batch_size
    )
    rand_obs = rand_obs / config.map_norm
    rand_obs = rearrange(rand_obs, '... N C -> ... (N C)')

    vec_map = rearrange(
        vec_map, '(B P S) N C -> B P S N C', P=jax.device_count(),
        S=config.sample_batch_size
    )

    # TODO: Hardcoded!
    noise = 7.0 / config.map_norm

    A_mat = jnp.tile(
        jnp.ones(rand_obs.shape[-1])[None, None, None],
        [jax.device_count(), config.sample_batch_size, 1, 1]
    )

    cov_y = linalg.DPLR(
        diagonal=jnp.tile(
            jnp.ones(rand_obs.shape[-1]) * noise ** 2,
            (jax.device_count(), config.sample_batch_size, 1)
        )
    )

    return rand_obs, vec_map, A_mat, cov_y
