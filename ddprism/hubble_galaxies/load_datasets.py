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
    samples, data_max, max_outlier_fraction=0.0
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
    flatten=True, n_models: int=1, train: bool=True
) -> Generator[tuple[jnp.ndarray, list[linalg.DPLR], jnp.ndarray], None, None]:
    """Get iterator for loading observations and covariance.

    Args:
        rng: Random number generator.
        dset_name: Name of the dataset to load.
        dataset_size: Size of the dataset to load. If -1, use the full dataset.
        sample_batch_size: Size of the sample batch.
        pmap_dim: Number of gpus to use in parallel.
        norm: Overall normalization factor for the dataset.
        arcsinh_scaling: Arcsinh scaling factor for the dataset.
        data_max: Maximum absolute value for clamping the dataset.
        flatten: Whether to flatten the individual samples or return as images.
        n_models: Number of models to assume for the A matrix.
        train: Whether the dataset will be used for training, which triggers
            infinite iteration over the dataset and shuffling.

    Returns:
        Generator that yields tuples of (obs, cov_y, A_mat). If train is True,
        the generator will be infinite, will shuffle the dataset, and will have
        a batch dimension. Each yield will return a total of dataset_size
        images. If train is False, the batch dimension will be size 1, and
        iterating through the generator will return the full dataset.

    Notes:
        Possible dataset names are defined in hst_cosmos.py
    """
    # Create our hugging face dataset
    n_cores = multiprocessing.cpu_count()
    dset = load_dataset(
        './hst_cosmos.py', trust_remote_code=True, split='train',
        name=dset_name, num_proc=max(n_cores - 1, 1)
    ).with_format('numpy')

    # Check that the reshapes will work.
    assert not train or (dataset_size % (sample_batch_size * pmap_dim) == 0)
    dataset_size = dataset_size if dataset_size > 0 else len(dset)

    # Infinite loop on dataset.
    continue_loop = True
    while continue_loop:
        rng, rng_dset = jax.random.split(rng)
        dset_seed = int(
            jax.random.randint(rng_dset, (1,), minval=0, maxval=2**16).item()
        )
        # Only shuffle if we are training.
        if train:
            dset = dset.shuffle(seed=dset_seed)

        # Our concept of a batch changes depending on whether we are training.
        if train:
            dset = dset.iter(
                batch_size=dataset_size, drop_last_batch=True
            )
        else:
            dset = dset.iter(
                batch_size=pmap_dim * sample_batch_size, drop_last_batch=False
            )

        for batch in dset:
            # Reshape for our needs.
            if train:
                if flatten:
                    mapping = '(K M N) H W -> K M N (H W)'
                else:
                    mapping = '(K M N) H W -> K M N H W 1'
                obs = rearrange(
                    batch['image_flux'], mapping, M=pmap_dim,
                    N=sample_batch_size
                )
                cov_y_all = rearrange(
                    batch['image_ivar'], mapping, M=pmap_dim,
                    N=sample_batch_size
                )
                A_mat = rearrange(
                    batch['image_mask'], mapping, M=pmap_dim,
                    N=sample_batch_size
                )
            else:
                if flatten:
                    mapping = '(M N) H W -> 1 M N (H W)'
                else:
                    mapping = '(M N) H W -> 1 M N H W 1'
                len_batch = (len(batch['image_flux']) // pmap_dim) * pmap_dim
                obs = rearrange(
                    batch['image_flux'][:len_batch], mapping, M=pmap_dim
                )
                cov_y_all = rearrange(
                    batch['image_ivar'][:len_batch], mapping, M=pmap_dim
                )
                A_mat = rearrange(
                    batch['image_mask'][:len_batch], mapping, M=pmap_dim
                )

            # Scale the data and convert to jax array.
            obs = normalize_dataset(obs, arcsinh_scaling, norm, data_max)

            # If inverse variance is zero then we know the value should be
            # masked. Instead, set the variance to a large number. Must also
            # rescale by the arcsinh_scaling.
            cov_ratio = 1 / (arcsinh_scaling ** 2) * (norm ** 2)
            cov_y_all = (
                1.0 / jnp.maximum(cov_y_all, 1e1 / cov_ratio)
            ) * cov_ratio

            # Nan safety.
            cov_y_all = jnp.nan_to_num(cov_y_all, nan=1e2)
            obs = jnp.nan_to_num(obs, nan=0.0)

            # Make cov_y a valid DPLR object to iterate over.
            cov_y = []
            for cy in cov_y_all:
                cov_y.append(linalg.DPLR(diagonal=cy))

            # Create an A matrix for the observations, which is the reshaped
            # mask. Tile the A matrix to match the number of models.
            if flatten:
                A_mat = jnp.tile(A_mat[:, :, :, None], (1, 1, 1, n_models, 1))
            else:
                A_mat = jnp.tile(
                    A_mat[:, :, :, None], (1, 1, 1, n_models, 1, 1, 1)
                )

            yield obs, cov_y, A_mat

        # Stop the loop if we are not training.
        if not train:
            continue_loop = False
