"""Sampling-config sweep on the lap-1 SZ checkpoint.

For a small eval subset (256 patches), generate posterior samples under
several (post_maxiter, sampling_steps, sample_batch_size) configurations and
record rmse_sz / rmse_matched / wallclock. Use the result to pick a
quality/speed sweet spot for the full SZ training.

Outputs JSON at /mnt/home/abayer/ceph/tsz_cmb/sz_so_b16_pilot/sweep_results.json
"""
import os
import json
import time
import functools
import h5py
import numpy as np
import os
# Make both CUDA and CPU available as JAX platforms so jax.local_devices()
# includes the TFRT_CPU_0 device referenced by the saved checkpoint metadata.
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
import jax
import jax.numpy as jnp
jax.devices('cpu')  # force CPU device registration
from einops import rearrange
from flax import jax_utils
from flax.training import train_state
from ml_collections import ConfigDict
from orbax.checkpoint import (
    CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer,
)
import orbax.checkpoint as ocp


def restore_with_metadata(cm, step):
    """Restore a checkpoint that has CPU-resident sharding metadata.

    Plain `cm.restore(step)` errors because some saved leaves reference a
    CPU device that's not in jax.local_devices() on the GPU node. We build
    restore_args from metadata, then *override* each ArrayRestoreArgs with
    a plain RestoreArgs(restore_type=np.ndarray) so sharding is ignored
    and arrays are loaded as numpy.
    """
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

from ddprism import diffusion, training_utils, utils
from ddprism.tsz_cmb import (
    embedding_models_healpix, load_datasets, training_utils_healpix,
)

DATA_BASE = '/mnt/home/abayer/ceph/fastpm/halfdome/oneweek/final'
WORK_BASE = '/mnt/home/abayer/ceph/tsz_cmb'
PROFILE = 'b16'
PILOT_WORKDIR = os.path.join(WORK_BASE, 'sz_so_b16')   # full run, em_laps=12
RANDOMS_WORKDIR = os.path.join(WORK_BASE, 'randoms_b16_so_long')
RANDOMS_LAP = 28
PILOT_LAP = 1
N_EVAL_PATCHES = 256

# post_maxiter=1 is fixed (sweep #1 found it dominates). Now sweep the
# remaining sampling levers: SDE solver steps and predictor-corrector count.
# sample_batch_size=8 matches training (doesn't affect quality, only speed).
SWEEP = [
    dict(post_maxiter=1, steps=32,  corrections=0),
    dict(post_maxiter=1, steps=32,  corrections=1),
    dict(post_maxiter=1, steps=32,  corrections=2),
    dict(post_maxiter=1, steps=64,  corrections=0),
    dict(post_maxiter=1, steps=64,  corrections=1),  # current default sampling
    dict(post_maxiter=1, steps=64,  corrections=2),
    dict(post_maxiter=1, steps=128, corrections=0),
    dict(post_maxiter=1, steps=128, corrections=1),
    dict(post_maxiter=1, steps=128, corrections=2),
]


def main():
    # 1. Restore randoms model.
    cm_r = CheckpointManager(
        os.path.join(RANDOMS_WORKDIR, 'checkpoints'),
        PyTreeCheckpointer(),
        options=CheckpointManagerOptions(enable_async_checkpointing=False),
    )
    rest_r = restore_with_metadata(cm_r, RANDOMS_LAP)
    randoms_params = rest_r['ema_params']
    config_randoms = ConfigDict(rest_r['config'])
    cm_r.close()
    print(f'restored randoms_b16_so_long lap {RANDOMS_LAP}', flush=True)

    # 2. Restore lap-1 SZ branch.
    cm_p = CheckpointManager(
        os.path.join(PILOT_WORKDIR, 'checkpoints'),
        PyTreeCheckpointer(),
        options=CheckpointManagerOptions(enable_async_checkpointing=False),
    )
    rest_p = restore_with_metadata(cm_p, PILOT_LAP)
    sz_ema_params = rest_p['ema_params']
    config = ConfigDict(rest_p['config'])
    cm_p.close()
    print(f'restored sz_so_b16_pilot lap {PILOT_LAP}', flush=True)

    # 3. Load test data (random_False = halo positions).
    sz_path = os.path.join(
        DATA_BASE, PROFILE, 'patches_so',
        'T_tot_patches_noise7_s100_fwhm2_random_False.h5',
    )
    sz_no_noise_path = os.path.join(
        DATA_BASE, PROFILE, 'patches_so',
        'dT_tsz_patches_s100_fwhm2_random_False.h5',
    )

    config.sz_path = sz_path
    config.sz_no_noise_path = sz_no_noise_path
    sz_obs, vec_map, A_mat, cov_y = load_datasets.load_sz(config, sz_path)
    sz_no_noise, _, _, _ = load_datasets.load_sz(config, sz_no_noise_path)

    # Trim to N_EVAL_PATCHES (block-aligned).
    block = jax.device_count() * config.sample_batch_size
    n_blocks = N_EVAL_PATCHES // block
    n_use_outer = max(1, n_blocks // sz_obs.shape[1])
    # sz_obs shape: (B, P, S, N*C); take first n_use_outer outer batches.
    sz_obs_sub = sz_obs[:n_use_outer]
    vec_map_sub = vec_map[:n_use_outer]
    cov_y_sub = cov_y
    A_sub = A_mat
    sz_no_noise_sub = sz_no_noise[:n_use_outer]
    print(
        f'eval set: outer batches {n_use_outer}, '
        f'total patches {n_use_outer * sz_obs.shape[1] * sz_obs.shape[2]}',
        flush=True,
    )

    # 4. Per-config sweep.
    feat_dim = sz_obs.shape[-1]
    # healpix shapes per branch (randoms uses 1024-emb arch via config_randoms).
    healpix_shapes = [(64 * 64, len(config.get('freqs', [93, 145, 280])))] * 2

    results = []

    for cfg_idx, cfg in enumerate(SWEEP):
        print(f'\n=== config {cfg_idx + 1}/{len(SWEEP)}: {cfg} ===', flush=True)
        # Build per-config posterior state with the swept settings.
        cfg_overrides = ConfigDict(config.to_dict())
        cfg_overrides.post_maxiter = cfg['post_maxiter']
        cfg_overrides.sampling_kwargs = ConfigDict(
            dict(config.sampling_kwargs)
        )
        cfg_overrides.sampling_kwargs.steps = cfg['steps']
        if 'corrections' in cfg:
            cfg_overrides.sampling_kwargs.corrections = cfg['corrections']

        rng = jax.random.PRNGKey(42)
        rng_state, rng = jax.random.split(rng, 2)

        # Build the joint posterior denoiser.
        denoiser_models = [
            training_utils_healpix.create_denoiser_transformer(
                config_randoms, healpix_shapes[0]
            ),
            training_utils_healpix.create_denoiser_transformer(
                cfg_overrides, healpix_shapes[1]
            ),
        ]
        x_features = [hp[0] * hp[1] for hp in healpix_shapes]
        posterior = diffusion.PosteriorDenoiserJointDiagonal(
            denoiser_models=denoiser_models, y_features=feat_dim,
            rtol=cfg_overrides.post_rtol,
            maxiter=cfg_overrides.post_maxiter,
            use_dplr=cfg_overrides.post_use_dplr,
            safe_divide=cfg_overrides.get('post_safe_divide', 1e-32),
            regularization=cfg_overrides.get('post_regularization', 0.0),
            error_threshold=cfg_overrides.get('post_error_threshold', None),
        )
        total_x_dim = sum(x_features)
        params0 = posterior.init(
            rng_state, jnp.ones((1, total_x_dim)), jnp.ones((1,))
        )
        # Stub optimizer for the TrainState object.
        lr_fn = lambda step: 0.0
        tx = training_utils.get_optimizer(cfg_overrides)(lr_fn)
        post_state = train_state.TrainState.create(
            apply_fn=posterior.apply, params=params0['params'], tx=tx,
        )
        post_state = jax_utils.replicate(post_state)

        # Pull randoms/SZ params into the joint param dict.
        post_params = {
            'denoiser_models_0': jax_utils.replicate(randoms_params),
            'denoiser_models_1': jax_utils.replicate(sz_ema_params),
        }

        def sample(batch, rng, state_local, params_local, A_local,
                   cov_local, vec_map_local, total_x_features,
                   sample_batch_size, sampling_kwargs):
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
                sample_batch_size=cfg_overrides.sample_batch_size,
                sampling_kwargs=cfg_overrides.sampling_kwargs,
            ),
            axis_name='batch',
        )

        # Run sampling on the small eval subset.
        rng_samp = jax.random.split(rng, (sz_obs_sub.shape[0], jax.device_count()))
        x_post_chunks = []
        t0 = time.time()
        for batch, vec, rng_pmap in zip(sz_obs_sub, vec_map_sub, rng_samp):
            x_post_chunks.append(
                sample_pmap(
                    batch, rng_pmap, post_state, post_params,
                    A_sub, cov_y_sub, vec,
                )
            )
        x_post = rearrange(
            jnp.stack(x_post_chunks, axis=0), 'K M N ... -> (K M N) ...'
        )
        x_post = jnp.clip(x_post, -cfg_overrides.data_max,
                          cfg_overrides.data_max)
        x_post = jnp.split(x_post, 2, axis=-1)  # [randoms, sz]
        sz_pred = x_post[1]
        elapsed = time.time() - t0

        # sz_no_noise from load_sz is already SZ-only (no randoms branch
        # appended), so no split needed -- match shape with sz_pred directly.
        sz_truth_only = rearrange(
            sz_no_noise_sub, 'K M N D -> (K M N) D',
        )
        rmse_sz = float(
            jnp.sqrt(jnp.mean(jnp.square(sz_pred - sz_truth_only)))
        )
        # matched-filter rmse
        filt = sz_truth_only / (
            jnp.sqrt(jnp.sum(jnp.square(sz_truth_only), axis=-1, keepdims=True))
            + 1e-12
        )
        sz_pred_matched = jnp.sum(
            sz_pred * filt, axis=-1, keepdims=True,
        ) * filt
        rmse_matched = float(
            jnp.sqrt(jnp.mean(jnp.square(sz_pred_matched - sz_truth_only)))
        )

        result = {
            'cfg': cfg,
            'n_patches': int(sz_pred.shape[0]),
            'elapsed_sec': float(elapsed),
            'rmse_sz_norm': rmse_sz,
            'rmse_matched_norm': rmse_matched,
            'rmse_sz_uK': rmse_sz * cfg_overrides.map_norm,
            'rmse_matched_uK': rmse_matched * cfg_overrides.map_norm,
        }
        results.append(result)
        print(
            f'  -> rmse_sz={result["rmse_sz_uK"]:.1f} μK, '
            f'rmse_matched={result["rmse_matched_uK"]:.1f} μK, '
            f'time={elapsed:.1f}s',
            flush=True,
        )

    out_path = os.path.join(PILOT_WORKDIR, 'sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
