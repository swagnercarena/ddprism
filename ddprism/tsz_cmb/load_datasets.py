"""Load the datasets for the TSZ CMB."""
from typing import Tuple

from einops import rearrange
import h5py
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from ddprism import linalg


def _perchan_std(x):
    """Per-channel std of x (shape ..., N, C) → shape (C,). Data-driven
    scale used by the 'perchan_linear' mode."""
    # Reduce over all axes except the last (channel).
    flat = np.asarray(x).reshape(-1, x.shape[-1])
    return jnp.asarray(np.std(flat, axis=0))


def _normalize(x, config):
    """Normalize input patches to roughly [-data_max, data_max].

    Modes (selected by config.get('normalization', 'linear')):
      - 'linear':         x_norm = x / map_norm  (single global scale).
      - 'asinh':          x_norm = asinh(x / asinh_scale). Compresses
                          dynamic range; helps regression substantially
                          (~50% RMSE drop). For ddprism the asinh
                          nonlinearity breaks the joint posterior
                          solver's linear forward model and produces
                          heteroscedastic noise the constant-cov_y solver
                          mishandles -- expect worse matched-filter RMSE.
      - 'perchan_linear': x_norm[c] = x[c] / σ[c], with σ[c] computed
                          from the loaded data. Data-driven per-channel
                          scale, linear-preserving.
                          NB: σ_c-normalized values are O(1), so
                          data_max=1 would clip at ~1σ -- pair this
                          mode with data_max ~ 5-10.

    Per-channel is only provided for the linear branch; perchan_asinh
    has the same drawbacks as asinh for ddprism without enough additional
    benefit to justify the maintenance burden.
    """
    mode_check = config.get('normalization', 'linear')
    if mode_check == 'perchan_linear' and config.data_max < 3.0:
        print(
            f"WARNING: perchan_linear with data_max={config.data_max} will "
            'clip aggressively (~1σ). Recommend data_max >= 5.'
        )
    mode = config.get('normalization', 'linear')
    if mode == 'linear':
        x = x / config.map_norm
    elif mode == 'asinh':
        x = jnp.arcsinh(x / float(config.get('asinh_scale', 50.0)))
    elif mode == 'perchan_linear':
        std_c = _perchan_std(x)
        x = x / std_c
    else:
        raise ValueError(f"unknown normalization '{mode}'")
    return jnp.clip(x, -config.data_max, config.data_max)


def _normalized_noise_var(config, x=None):
    """Approximate per-pixel noise variance in the *normalized* space, used
    by cov_y. For 'asinh', uses the linear-regime (small-x) approximation,
    which is good for typical noise-dominated pixels but wrong at high
    amplitudes (heteroscedasticity). Returns a scalar for global scales
    or shape-(C,) for per-channel modes.
    """
    mode = config.get('normalization', 'linear')
    raw_noise = 7.0  # μK, hardcoded
    if mode == 'asinh':
        return (raw_noise / float(config.get('asinh_scale', 50.0))) ** 2
    if mode == 'perchan_linear':
        if x is None:
            raise ValueError(
                "perchan_linear requires `x` to compute per-channel σ"
            )
        std_c = _perchan_std(x)
        return (raw_noise / std_c) ** 2  # shape (C,)
    return (raw_noise / config.map_norm) ** 2


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
    n_pix = rand_obs.shape[-2]
    noise_var = _normalized_noise_var(config, x=rand_obs)
    rand_obs = _normalize(rand_obs, config)
    rand_obs = rearrange(rand_obs, '... N C -> ... (N C)')

    vec_map = rearrange(
        vec_map, '(B P S) N C -> B P S N C', P=jax.device_count(),
        S=config.sample_batch_size
    )

    A_mat = jnp.tile(
        jnp.ones(rand_obs.shape[-1])[None, None, None],
        [jax.device_count(), config.sample_batch_size, 1, 1]
    )

    # If per-channel noise_var has shape (C,), tile across pixels matching
    # the '... N C -> ... (N C)' flat layout (channel-fastest).
    if jnp.ndim(noise_var) > 0:
        noise_diag = jnp.tile(noise_var, n_pix)
    else:
        noise_diag = jnp.ones(rand_obs.shape[-1]) * noise_var
    cov_y = linalg.DPLR(
        diagonal=jnp.tile(
            noise_diag,
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
    n_pix = sz_obs.shape[-2]
    noise_var = _normalized_noise_var(config, x=sz_obs)
    sz_obs = _normalize(sz_obs, config)
    sz_obs = rearrange(sz_obs, '... N C -> ... (N C)')

    vec_map = rearrange(
        vec_map, '(B P S) N C -> B P S N C', P=jax.device_count(),
        S=config.sample_batch_size
    )

    # Account for having two sources.
    A_mat = jnp.tile(
        jnp.ones(sz_obs.shape[-1])[None, None, None],
        [jax.device_count(), config.sample_batch_size, 2, 1]
    )

    if jnp.ndim(noise_var) > 0:
        noise_diag = jnp.tile(noise_var, n_pix)
    else:
        noise_diag = jnp.ones(sz_obs.shape[-1]) * noise_var
    cov_y = linalg.DPLR(
        diagonal=jnp.tile(
            noise_diag,
            (jax.device_count(), config.sample_batch_size, 1)
        )
    )

    return sz_obs, vec_map, A_mat, cov_y
