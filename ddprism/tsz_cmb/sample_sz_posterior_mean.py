"""Generate K posterior samples per observation for each SZ run, save the
mean (and all samples) so we can recompute RMSE / patch grids using the
posterior MEAN instead of a single sample.

K=20 samples are drawn with K different RNG seeds, on a 1024-patch
mass-stratified eval subset (every Nth face-safe halo, spanning highest
to lowest mass). Output:
  /mnt/home/abayer/ceph/tsz_cmb/sz_so_<profile>/posterior_mean.h5
    - 'samples'  shape (K, n_eval, num_pix*num_chan), in normalized space
    - 'mean'     shape (n_eval, num_pix*num_chan), normalized
    - 'eval_idx' shape (n_eval,)  -- original H5 patch indices
    - attrs: K, n_eval, profile, lap, sample_batch_size, ...
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
import time
import functools
import h5py
import numpy as np
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import jax_utils
from flax.training import train_state
from ml_collections import ConfigDict
from orbax.checkpoint import (
    CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer,
)
import orbax.checkpoint as ocp
jax.devices('cpu')

from ddprism import diffusion, training_utils, utils
from ddprism.tsz_cmb import (
    embedding_models_healpix, load_datasets, training_utils_healpix,
)

PROFILES = {
    'b16':       {'lap': 10},
    'b16g7':     {'lap': 11},
    'b16g7rel':  {'lap': 11},
}
DATA_BASE = '/mnt/home/abayer/ceph/fastpm/halfdome/oneweek/final'
WORK_BASE = '/mnt/home/abayer/ceph/tsz_cmb'
NUM_PIX = 64
N_EVAL = 1024            # mass-stratified subset size (must be multiple of block)
K_SAMPLES = 20           # default; override via --K


def restore_with_metadata(cm, step):
    md = cm.item_metadata(step)
    restore_args = ocp.checkpoint_utils.construct_restore_args(md)
    def _strip(x):
        if isinstance(x, ocp.ArrayRestoreArgs):
            return ocp.RestoreArgs(restore_type=np.ndarray)
        return x
    restore_args = jax.tree_util.tree_map(
        _strip, restore_args,
        is_leaf=lambda v: isinstance(v, ocp.RestoreArgs),
    )
    return cm.restore(step, restore_kwargs={'restore_args': restore_args})


def sample_for_profile(profile, lap, K=K_SAMPLES):
    print(f'\n=== {profile} (lap={lap}, K={K}) ===', flush=True)
    sz_workdir = os.path.join(WORK_BASE, f'sz_so_{profile}')
    rand_workdir = os.path.join(WORK_BASE, f'randoms_{profile}_so_long')
    rand_lap = {'b16': 28, 'b16g7': 30, 'b16g7rel': 37}[profile]

    cm_r = CheckpointManager(
        os.path.join(rand_workdir, 'checkpoints'),
        PyTreeCheckpointer(),
        options=CheckpointManagerOptions(enable_async_checkpointing=False),
    )
    rest_r = restore_with_metadata(cm_r, rand_lap)
    randoms_params = rest_r['ema_params']
    config_randoms = ConfigDict(rest_r['config'])
    cm_r.close()

    cm_p = CheckpointManager(
        os.path.join(sz_workdir, 'checkpoints'),
        PyTreeCheckpointer(),
        options=CheckpointManagerOptions(enable_async_checkpointing=False),
    )
    rest_p = restore_with_metadata(cm_p, lap)
    sz_ema_params = rest_p['ema_params']
    config = ConfigDict(rest_p['config'])
    cm_p.close()

    sz_path = os.path.join(
        DATA_BASE, profile, 'patches_so',
        'T_tot_patches_noise7_s100_fwhm2_random_False.h5',
    )
    config.sz_path = sz_path
    sz_obs, vec_map, A_mat, cov_y = load_datasets.load_sz(config, sz_path)

    # Mass-stratified eval subset: take every k-th outer batch.
    # sz_obs shape (B, P, S, N*C). Total patches = B*P*S, sorted by mass desc.
    block = jax.device_count() * config.sample_batch_size
    n_total = sz_obs.shape[0] * sz_obs.shape[1] * sz_obs.shape[2]
    n_outer_total = sz_obs.shape[0]
    n_outer_eval = N_EVAL // (sz_obs.shape[1] * sz_obs.shape[2])
    stride = max(1, n_outer_total // n_outer_eval)
    outer_idx = np.arange(n_outer_eval) * stride
    sz_obs_sub = sz_obs[outer_idx]
    vec_map_sub = vec_map[outer_idx]
    n_eval = sz_obs_sub.shape[0] * sz_obs_sub.shape[1] * sz_obs_sub.shape[2]
    print(f'  eval subset: {n_eval} patches (every {stride}-th outer batch)',
          flush=True)

    # Build joint posterior denoiser.
    feat_dim = sz_obs.shape[-1]
    healpix_shapes = [(NUM_PIX * NUM_PIX, len(config.get('freqs', [93, 145, 280])))] * 2
    denoiser_models = [
        training_utils_healpix.create_denoiser_transformer(config_randoms, healpix_shapes[0]),
        training_utils_healpix.create_denoiser_transformer(config, healpix_shapes[1]),
    ]
    x_features = [hp[0] * hp[1] for hp in healpix_shapes]
    posterior = diffusion.PosteriorDenoiserJointDiagonal(
        denoiser_models=denoiser_models, y_features=feat_dim,
        rtol=config.post_rtol, maxiter=config.post_maxiter,
        use_dplr=config.post_use_dplr,
        safe_divide=config.get('post_safe_divide', 1e-32),
        regularization=config.get('post_regularization', 0.0),
        error_threshold=config.get('post_error_threshold', None),
    )
    total_x_dim = sum(x_features)
    rng_init, rng_seed = jax.random.split(jax.random.PRNGKey(0), 2)
    params0 = posterior.init(rng_init, jnp.ones((1, total_x_dim)), jnp.ones((1,)))
    tx = training_utils.get_optimizer(config)(lambda step: 0.0)
    post_state = train_state.TrainState.create(
        apply_fn=posterior.apply, params=params0['params'], tx=tx,
    )
    post_state = jax_utils.replicate(post_state)
    post_params = {
        'denoiser_models_0': jax_utils.replicate(randoms_params),
        'denoiser_models_1': jax_utils.replicate(sz_ema_params),
    }

    def sample(batch, rng, state_local, params_local, A_local, cov_local,
               vec_map_local, total_x_features, sample_batch_size, sampling_kwargs):
        vec_map_dict = {
            f'denoiser_models_{i}': {'vec_map': vec_map_local}
            for i in range(A_local.shape[-2])
        }
        return utils.sample(
            rng, state_local,
            {
                'params': params_local,
                'variables': (
                    {'y': batch, 'cov_y': cov_local, 'A': A_local}
                    | vec_map_dict
                ),
            },
            sample_shape=(sample_batch_size,),
            feature_shape=total_x_features,
            **sampling_kwargs,
        )

    sample_pmap = jax.pmap(
        functools.partial(
            sample, total_x_features=total_x_dim,
            sample_batch_size=config.sample_batch_size,
            sampling_kwargs=config.sampling_kwargs,
        ),
        axis_name='batch',
    )

    # Loop K seeds.
    samples = []
    for k in range(K):
        rng_k = jax.random.PRNGKey(1000 + k)
        rng_per = jax.random.split(rng_k, (sz_obs_sub.shape[0], jax.device_count()))
        x_post_chunks = []
        t0 = time.time()
        for batch, vec, rng_pmap in zip(sz_obs_sub, vec_map_sub, rng_per):
            x_post_chunks.append(
                sample_pmap(batch, rng_pmap, post_state, post_params,
                            A_mat, cov_y, vec)
            )
        x_post = rearrange(jnp.stack(x_post_chunks, axis=0), 'K M N ... -> (K M N) ...')
        x_post = jnp.clip(x_post, -config.data_max, config.data_max)
        sz_only = jnp.split(x_post, 2, axis=-1)[1]
        samples.append(np.asarray(sz_only))
        print(f'  k={k}: {time.time() - t0:.1f}s', flush=True)
    samples = np.stack(samples, axis=0)
    mean = samples.mean(axis=0)

    # Recover the original-H5 indices for this subset.
    truth_path = os.path.join(
        DATA_BASE, profile, 'patches_so',
        'dT_tsz_patches_s100_fwhm2_random_False.h5',
    )
    with h5py.File(truth_path) as f:
        vec_full = np.asarray(f['vecs'][:])
        nside = int(f.attrs['nside'])
    full_safe_idx = load_datasets._polar_polar_safe_indices(vec_full, nside, block)
    # outer_idx selects outer batches; flat positions are
    # outer_idx[i]*block .. outer_idx[i]*block+block-1.
    eval_idx_in_safe = np.concatenate([
        np.arange(o * block, (o + 1) * block) for o in outer_idx
    ])
    eval_idx_orig = full_safe_idx[eval_idx_in_safe]

    out_path = os.path.join(sz_workdir, 'posterior_mean.h5')
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('samples', data=samples)
        f.create_dataset('mean', data=mean)
        f.create_dataset('eval_idx', data=eval_idx_orig)
        f.attrs['profile'] = profile
        f.attrs['lap'] = lap
        f.attrs['K'] = K
        f.attrs['n_eval'] = n_eval
        f.attrs['stride'] = int(stride)
        f.attrs['map_norm'] = float(config.map_norm)
        f.attrs['data_max'] = float(config.data_max)
    print(f'  -> wrote {out_path}', flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', choices=list(PROFILES), default=None,
                    help='Run only this profile. If omitted, runs all 3.')
    ap.add_argument('--K', type=int, default=K_SAMPLES, help='Number of samples')
    args = ap.parse_args()
    if args.profile:
        sample_for_profile(args.profile, PROFILES[args.profile]['lap'], K=args.K)
    else:
        for p, info in PROFILES.items():
            sample_for_profile(p, info['lap'], K=args.K)
