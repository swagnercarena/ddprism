"""Generate samples for the full galaxy dataset using trained denoisers."""
import functools
import os
import pickle

from absl import app, flags
from einops import rearrange
from flax import jax_utils
import h5py
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint import PyTreeCheckpointer
from tqdm import tqdm

from ddprism import utils

from build_parent_sample import NUMPIX
from hg_utils import create_posterior_train_state_galaxies
import load_datasets


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'galaxies_workdir', None, 'Working directory with trained galaxy model.'
)
flags.DEFINE_string(
    'randoms_workdir', None, 'Working directory with trained randoms model.'
)
flags.DEFINE_string('output_dir', None, 'Directory to save generated samples.')
flags.DEFINE_integer(
    'galaxies_lap', -1, 'Checkpoint number for galaxies model (-1 for latest).'
)
flags.DEFINE_integer(
    'randoms_lap', -1, 'Checkpoint number for randoms model (-1 for latest).'
)

def append_to_hdf5(filepath, x_randoms, x_galaxies):
    """Append samples to HDF5 file, creating datasets if they don't exist."""
    if not os.path.exists(filepath):
        # Create new file with initial datasets
        with h5py.File(filepath, 'w') as f:
            f.create_dataset(
                'randoms', data=x_randoms,
                maxshape=(None, *x_randoms.shape[1:]),
                chunks=True, compression='gzip'
            )
            f.create_dataset(
                'galaxies', data=x_galaxies,
                maxshape=(None, *x_galaxies.shape[1:]),
                chunks=True, compression='gzip'
            )
    else:
        # Append to existing file
        with h5py.File(filepath, 'a') as f:
            # Resize and append randoms
            randoms_dset = f['randoms']  # type: ignore
            current_size = randoms_dset.shape[0]  # type: ignore
            new_size = current_size + x_randoms.shape[0]
            randoms_dset.resize(new_size, axis=0)  # type: ignore
            randoms_dset[current_size:] = x_randoms  # type: ignore

            # Resize and append galaxies
            galaxies_dset = f['galaxies']  # type: ignore
            current_size = galaxies_dset.shape[0]  # type: ignore
            new_size = current_size + x_galaxies.shape[0]
            galaxies_dset.resize(new_size, axis=0)  # type: ignore
            galaxies_dset[current_size:] = x_galaxies  # type: ignore

def main(_):
    """Generate samples for the full galaxy dataset."""
    config = FLAGS.config
    galaxies_workdir = FLAGS.galaxies_workdir
    randoms_workdir = FLAGS.randoms_workdir
    output_dir = FLAGS.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    rng = jax.random.PRNGKey(config.rng_key)

    print(f'Found devices {jax.local_devices()}')
    print(f'Galaxies workdir: {galaxies_workdir}')
    print(f'Randoms workdir: {randoms_workdir}')
    print(f'Output directory: {output_dir}')

    # Load randoms model
    checkpointer = PyTreeCheckpointer()
    checkpoint_options = CheckpointManagerOptions()
    randoms_checkpoint_manager = CheckpointManager(
        os.path.join(randoms_workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )
    randoms_lap = FLAGS.randoms_lap
    if randoms_lap == -1:
        randoms_lap = randoms_checkpoint_manager.latest_step()

    print(f'Loading randoms model from checkpoint {randoms_lap}')
    randoms_restore = randoms_checkpoint_manager.restore(randoms_lap)
    randoms_params = randoms_restore['ema_params']
    config_randoms = ConfigDict(randoms_restore['config'])
    randoms_checkpoint_manager.close()

    # Load galaxies model
    galaxies_checkpoint_manager = CheckpointManager(
        os.path.join(galaxies_workdir, 'checkpoints'), checkpointer,
        options=checkpoint_options
    )
    galaxies_lap = FLAGS.galaxies_lap
    if galaxies_lap == -1:
        galaxies_lap = galaxies_checkpoint_manager.latest_step()

    print(f'Loading galaxies model from checkpoint {galaxies_lap}')
    galaxies_restore = galaxies_checkpoint_manager.restore(galaxies_lap)
    galaxies_params = galaxies_restore['ema_params']
    config = ConfigDict(galaxies_restore['config'])
    galaxies_checkpoint_manager.close()

    # Set up dataset
    rng_dataset, rng = jax.random.split(rng, 2)
    image_shape = (NUMPIX, NUMPIX, 1)

    # Create dataloader for full dataset
    dset_name = 'hst-cosmos-galaxies'
    gal_dataloader = load_datasets.get_dataloader(
        rng_dataset, dset_name, -1, config.sample_batch_size,
        jax.local_device_count(), norm=config.data_norm,
        arcsinh_scaling=config.arcsinh_scaling, data_max=config.data_max,
        flatten=True
    )

    # Create posterior denoiser
    rng_state, rng = jax.random.split(rng)
    post_state_unet = create_posterior_train_state_galaxies(
        rng_state, config, config_randoms, image_shape
    )
    params = {
        'denoiser_models_0': randoms_params,
        'denoiser_models_1': galaxies_params
    }
    params = jax_utils.replicate(params)
    post_state_unet = jax_utils.replicate(post_state_unet)

    # Create our sampling function. We want to pmap it, but we also have to
    # batch to avoid memory issues. Start with the pmapped call to sample.
    def sample(
        batch, rng, state_local, params_local, A_local, cov_local, image_shape,
        sample_batch_size, sampling_kwargs
    ):
        return utils.sample(
            rng, state_local,
            {
                'params': params_local,
                'variables': {'y': batch, 'A': A_local, 'cov_y': cov_local}
            },
            sample_shape=(sample_batch_size,),
            feature_shape=image_shape[0] * image_shape[1] * image_shape[2] * 2,
            **sampling_kwargs
        )
    sample_pmap = jax.pmap(
        functools.partial(
            sample, image_shape=image_shape,
            sample_batch_size=config.sample_batch_size,
            sampling_kwargs=config.sampling_kwargs
        ),
        axis_name='batch'
    )

    print('Starting sample generation for full dataset...')

    total_samples = 0
    samples_savepath = os.path.join(output_dir, 'samples.h5')

    for gal_batch, cov_y, A_batch in tqdm(gal_dataloader, desc='Sample'):
        # New rng for each sampling.
        rng_samp, rng = jax.random.split(rng)
        rng_samp = jax.random.split(
            rng_samp, (jax.device_count(),)
        )
        x_sample = sample_pmap(
            gal_batch, rng_samp, post_state_unet, params, A_batch, cov_y
        )
        x_sample = jnp.split(x_sample, 2, axis=-1)
        x_randoms = rearrange(
            x_sample[0], 'M N (H W) -> (M N) H W 1', H=NUMPIX, W=NUMPIX
        )
        x_galaxies = rearrange(
            x_sample[1], 'M N (H W) -> (M N) H W 1', H=NUMPIX, W=NUMPIX
        )

        # Append samples to hdf5 file
        append_to_hdf5(samples_savepath, x_randoms, x_galaxies)

        total_samples += x_randoms.shape[0]

    # Save metadata
    metadata = {
        'total_samples': total_samples,
        'image_shape': image_shape,
        'config': config.to_dict(),
        'randoms_checkpoint': randoms_lap,
        'galaxies_checkpoint': galaxies_lap,
        'workdir': galaxies_workdir,
        'randoms_workdir': randoms_workdir,
        'output_dir': output_dir,
    }

    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f'Final results saved:')
    print(f'  Total samples: {total_samples}')
    print(f'  Output directory: {output_dir}')


if __name__ == '__main__':
    app.run(main)
