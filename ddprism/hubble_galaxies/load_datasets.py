"""Module for generating dataset realizations of grassy MNIST."""
import multiprocessing
from typing import Callable, Optional, Sequence, Tuple

from datasets import load_dataset
from einops import rearrange
import jax
import jax.numpy as jnp
import numpy as np

from ddprism import linalg


def get_dataloader(
    rng: Sequence[int], dset_name: str, dataset_size: int,
    sample_batch_size: int, pmap_dim: int, norm: Optional[float] = 1.0,
    arcsinh_scaling: Optional[float] = 1.0, data_max: Optional[float] = jnp.inf,
    flatten=True
) -> Tuple[Callable[[], Tuple[np.ndarray, Sequence[linalg.DPLR]]], np.ndarray]:
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
            obs = jnp.arcsinh(obs / arcsinh_scaling) * norm
            obs = jnp.minimum(obs, data_max)
            obs = jnp.maximum(obs, -data_max)

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

            yield obs, cov_y


def get_A_mat(
    image_shape: Sequence[int], sample_batch_size: int, n_models: int=1
):
    """Get A matrix for given image shape. Assume identity (no mask)."""
    # Generate the A matrix without the sampling batch dimension.
    feat_dim = image_shape[0] * image_shape[1] * image_shape[2]
    A_mat = jnp.ones((1, n_models, feat_dim))

    # Add the sampling batch dimension.
    return jnp.tile(A_mat, (sample_batch_size, 1, 1))
