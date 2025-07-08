"""Module for generating dataset realizations of grassy MNIST."""
import multiprocessing
from typing import Generator, Optional, Sequence

from datasets import load_dataset
from einops import rearrange
import jax
import jax.numpy as jnp
import numpy as np

from ddprism import linalg


def clamp_dataset(dataset: jnp.ndarray, data_max: float) -> jnp.ndarray:
    """Clamp the dataset to the given data_max."""
    dataset = jnp.maximum(dataset, -data_max)
    dataset = jnp.minimum(dataset, data_max)
    return dataset


def filter_samples_by_clamp_range(
    samples, data_max, max_outlier_fraction=0.0001
):
    """Filter samples where sufficient pixels are outside clamp range.

    Args:
        samples: Array of samples to filter. Leading dimension is the number of
            samples, but otherwise the shape is arbitrary.
        data_max: Maximum absolute value for clamping
        max_outlier_fraction: Maximum fraction of pixels allowed to be outside
            clamp range. Default is 0.0001, which is 1 pixel for a 128x128
            image.

    Returns:
        Filtered samples array and number of dropped samples
    """
    # Check which pixels are outside the clamp range
    outside_range = (jnp.abs(samples) > data_max)

    # Calculate fraction of pixels outside range for each sample.
    outlier_fractions = jnp.mean(
        outside_range, axis=tuple(range(1, samples.ndim))
    )

    # Keep samples where outlier fraction is <= max_outlier_fraction
    keep_mask = outlier_fractions <= max_outlier_fraction
    filtered_samples = samples[keep_mask]
    num_dropped = jnp.sum(~keep_mask)

    return filtered_samples, num_dropped


def normalize_dataset(
    dataset: jnp.ndarray, arcsinh_scaling: float, norm: float, data_max: float
) -> jnp.ndarray:
    """Normalize the dataset to the given norm."""
    dataset = jnp.arcsinh(dataset / arcsinh_scaling) * norm
    dataset = clamp_dataset(dataset, data_max)
    return dataset


def unnormalize_dataset(
    dataset: jnp.ndarray, arcsinh_scaling: float, norm: float
) -> jnp.ndarray:
    """Unnormalize the dataset to the given norm."""
    dataset = jnp.sinh(dataset / norm) * arcsinh_scaling
    return dataset


def get_dataloader(
    rng: Sequence[int], dset_name: str, dataset_size: int,
    sample_batch_size: int, pmap_dim: int, norm: Optional[float] = 1.0,
    arcsinh_scaling: Optional[float] = 1.0, data_max: Optional[float] = jnp.inf,
    flatten=True,  n_models: int=1
) -> Generator[np.ndarray, Sequence[linalg.DPLR], np.ndarray]:
    """Get iterator for loading observations and covariance.

    Notes:
        Possible dataset names are defined in hst_cosmos.py
    """
    # Create our hugging face dataset
    n_cores = multiprocessing.cpu_count()
    dset = load_dataset(
        './hst_cosmos.py', trust_remote_code=True, split='train',
        name=dset_name, num_proc=max(n_cores - 1, 1)
    ).with_format('numpy')

    # Check that the reshapes will work
    assert dataset_size & (sample_batch_size * pmap_dim) == 0

    # Infinite loop on dataset.
    while True:
        rng, rng_dset = jax.random.split(rng)
        dset_seed = int(
            jax.random.randint(rng_dset, (1,), minval=0, maxval=2**16).item()
        )
        dset_shuffle = dset.shuffle(seed=dset_seed)
        # TODO, prefetch?
        dset_shuffle = dset_shuffle.iter(
            batch_size=dataset_size, drop_last_batch=True
        )

        for batch in dset_shuffle:
            if flatten:
                mapping = '(K M N) H W -> K M N (H W)'
            else:
                mapping = '(K M N) H W -> K M N H W 1'

            # Scale the data and convert to jax array.
            obs = rearrange(
                batch['image_flux'], mapping, M=pmap_dim, N=sample_batch_size
            )
            obs = normalize_dataset(obs, arcsinh_scaling, norm, data_max)

            # If inverse variance is zero then we know the value should be
            # masked. Instead, set the variance to a large number. Must also
            # rescale by the arcsinh_scaling.
            cov_ratio = 1 / (arcsinh_scaling ** 2) * (norm ** 2)
            cov_y_all = rearrange(
                1.0 / jnp.maximum(batch['image_ivar'], 1e1 / cov_ratio),
                mapping, M=pmap_dim, N=sample_batch_size
            ) * cov_ratio

            # Nan safety.
            cov_y_all = jnp.nan_to_num(cov_y_all, nan=1e2)
            obs = jnp.nan_to_num(obs, nan=0.0)

            # Make cov_y a valid DPLR object to iterate over.
            cov_y = []
            for cy in cov_y_all:
                cov_y.append(linalg.DPLR(diagonal=cy))

            # Create an A matrix for the observations, which is the reshaped
            # mask.
            A_mat = rearrange(
                batch['image_mask'], mapping, M=pmap_dim, N=sample_batch_size
            )
            # Tile the A matrix to match the number of models.
            if flatten:
                A_mat = jnp.tile(A_mat[:, :, :, None], (1, 1, 1, n_models, 1))
            else:
                A_mat = jnp.tile(
                    A_mat[:, :, :, None], (1, 1, 1, n_models, 1, 1, 1)
                )

            yield obs, cov_y, A_mat
