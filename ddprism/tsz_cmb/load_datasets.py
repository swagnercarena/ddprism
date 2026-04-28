"""Load the datasets for the TSZ CMB."""
from typing import Tuple

from einops import rearrange
import h5py
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from ddprism import linalg


def _polar_polar_safe_indices(
    vecs: np.ndarray, nside: int, block: int
) -> np.ndarray:
    """Indices of patches that do not cross a polar-polar HEALPix seam.

    HEALPix has 24 base-face boundaries: 8 polar-polar (between adjacent
    polar caps, terminating at the pole) and 16 polar-equatorial. Only
    polar-polar boundaries break reorder_diamond -- the pole is a
    degenerate corner where pixels have only 7 neighbors and the
    NEST/Morton ordering can't bridge the seam, producing ~30 arcmin
    jumps between supposedly-adjacent pixels in the reordered patch.
    Polar-equatorial crossings have a clean 8-neighbor structure and
    measured neighbor disruption indistinguishable from face-internal
    patches (~3 arcmin), so we keep them. The kept list is truncated to
    a multiple of block.
    """
    shift = 2 * int(np.log2(nside))
    pix = hp.vec2pix(nside, vecs[..., 0], vecs[..., 1], vecs[..., 2], nest=True)
    face = pix >> shift
    is_face_crossing = (face != face[:, :1]).any(axis=1)
    all_north = (face < 4).all(axis=1)
    all_south = (face >= 8).all(axis=1)
    is_polar_polar = is_face_crossing & (all_north | all_south)
    keep = np.where(~is_polar_polar)[0]
    return keep[: (len(keep) // block) * block]

def load_randoms(
    config: dict, randoms_path: str
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load a randoms dataset.

    If config.face_safe_filter is true (default), drop patches whose
    pixels span more than one HEALPix base face -- those have scrambled
    spatial neighbor structure after reorder_diamond and corrupt the
    transformer's local geometry assumptions.
    """
    block = jax.device_count() * config.sample_batch_size
    with h5py.File(randoms_path, 'r') as f:
        rand_obs = jnp.array(f['patches'][:config.n_train])
        vec_map = jnp.array(f['vecs'][:config.n_train])
        nside = int(f.attrs['nside'])

    if config.get('face_safe_filter', True):
        idx = _polar_polar_safe_indices(np.asarray(vec_map), nside, block)
        print(
            f'load_randoms: face_safe_filter kept {len(idx)}/{config.n_train}'
        )
        rand_obs = rand_obs[idx]
        vec_map = vec_map[idx]

    # Transform to the desired dimensions.
    rand_obs = rearrange(
        rand_obs, '(B P S) N C -> B P S N C', P=jax.device_count(),
        S=config.sample_batch_size
    )
    rand_obs = rand_obs / config.map_norm
    rand_obs = jnp.clip(rand_obs, -config.data_max, config.data_max)
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


def load_sz(
    config: dict, sz_path: str
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load a sz dataset.

    Same face-safe filtering as load_randoms: see its docstring.
    """
    block = jax.device_count() * config.sample_batch_size
    with h5py.File(sz_path, 'r') as f:
        sz_obs = jnp.array(f['patches'][:config.n_train])
        vec_map = jnp.array(f['vecs'][:config.n_train])
        nside = int(f.attrs['nside'])

    if config.get('face_safe_filter', True):
        idx = _polar_polar_safe_indices(np.asarray(vec_map), nside, block)
        print(f'load_sz: face_safe_filter kept {len(idx)}/{config.n_train}')
        sz_obs = sz_obs[idx]
        vec_map = vec_map[idx]

    # Transform to the desired dimensions.
    sz_obs = rearrange(
        sz_obs, '(B P S) N C -> B P S N C', P=jax.device_count(),
        S=config.sample_batch_size
    )
    sz_obs = sz_obs / config.map_norm
    sz_obs = jnp.clip(sz_obs, -config.data_max, config.data_max)
    sz_obs = rearrange(sz_obs, '... N C -> ... (N C)')

    vec_map = rearrange(
        vec_map, '(B P S) N C -> B P S N C', P=jax.device_count(),
        S=config.sample_batch_size
    )

    # TODO: Hardcoded!
    noise = 7.0 / config.map_norm

    # Account for having two sources.
    A_mat = jnp.tile(
        jnp.ones(sz_obs.shape[-1])[None, None, None],
        [jax.device_count(), config.sample_batch_size, 2, 1]
    )

    cov_y = linalg.DPLR(
        diagonal=jnp.tile(
            jnp.ones(sz_obs.shape[-1]) * noise ** 2,
            (jax.device_count(), config.sample_batch_size, 1)
        )
    )

    return sz_obs, vec_map, A_mat, cov_y
