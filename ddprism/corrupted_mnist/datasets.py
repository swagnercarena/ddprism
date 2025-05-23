"""Module for generating dataset realizations of grassy MNIST."""
import os
from typing import Mapping, Sequence

from einops import rearrange
import jax
import jax.numpy as jnp
import numpy as np
from numpy import ndarray
from PIL import Image
from tensorflow.keras.datasets import mnist

from galaxy_diffusion import linalg

# Path to downloaded images from ImageNet
DEFAULT_IMAGENET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'grass_jpeg/'
)


def get_raw_mnist_images() -> ndarray:
    """Get the raw mnist images.

    Returns:
        Normalized MNIST images with train / val split.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


def _resize_crop(img):
    """Resize and crop the image to get a 100x100 image.

    Arguments:
        img: Image to be resized and cropped.

    Returns:
        Resized and cropped image.
    """
    target_w, target_h = (100, 100)
    img_ratio = img.width / img.height

    # Resize image so we can fit the 100x100 crop while preserving aspect ratio.
    if img_ratio > 1.0:
        height = target_h
        width = int(round(img.width * (target_h / img.height)))
    else:
        # Image is taller than target, resize by width
        width = target_w
        height = int(round(img.height * (target_w / img.width)))
    img = img.resize((width, height), Image.Resampling.LANCZOS)

    # Crop coordinates.
    left = (width - target_w) // 2
    top = (height - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return img.crop((left, top, right, bottom))


def get_raw_grass_images(imagenet_path: str = DEFAULT_IMAGENET_PATH) -> ndarray:
    """Get the full grass dataset from the imagenet path with normalization.

    Args:
        imagenet_path: Path to imagenet grass dataset.

    Returns:
        Normalized images.
    """
    npy_path = os.path.join(imagenet_path, 'grass_images.npy')
    if os.path.isfile(npy_path):
        return np.load(npy_path)

    print (f'Numpy file not found, generating at {npy_path}')
    images = []
    for filename in os.listdir(imagenet_path):
        if (
            filename.endswith(".JPEG") or filename.endswith(".JPG") or
            filename.endswith(".jpg")
        ):
            try:
                im = Image.open(os.path.join(imagenet_path, filename))
                # Convert to grayscale.
                im = im.convert(mode="L")

                im = _resize_crop(im)
                images.append(im)

            except Exception as e:
                print(e)

    # Save the images and return.
    images = np.asarray(images) / 255.0
    np.save(npy_path, images)
    return images


def _random_crop(
    image: ndarray, start_y: int, start_x: int,
    crop_size: Sequence[int] = (28, 28)
) -> ndarray:
    """Randomly crop a 2D image to the given crop size.

    Args:
        rng: jax PRNG key.
        start_y: Where to start the y-axis crop
        start_y: Where to start the x-axis crop
        crop_size: Size to crop to.

    Returns:
        Cropped image of shape crop_size.
    """
    return (
        jax.lax.dynamic_slice(image, (start_x, start_y), crop_size)[..., None]
    )


def get_corrupted_mnist(
    rng: Sequence[int], grass_amp: float, mnist_amp: float,
    imagenet_path: str = DEFAULT_IMAGENET_PATH, dataset_size: int=None,
    zeros_and_ones=True
) -> ndarray:
    """Generate grassy MNIST data.

    Arguments:
        rng: Jax PRNGKey for sampling.
        grass_amp: Amplitude of grass component in returned images.
        mnist_amp: Amplitude of MNIST component in returned images.
        imagenet_path: Path to imagenet grass dataset.
        dataset_size: Total size of dataset to generate. If None will be set by
            size of MNIST dataset.
        zeros_and_ones: If True, will only use 1 and 0 images from MNIST.
    """

    # Read in MNIST digits.
    (mnist_train, labels_train), (mnist_test, labels_test) = (
        get_raw_mnist_images()
    )
    # Combine train and test images into a single dataset.
    raw_mnist = np.concatenate((mnist_train, mnist_test), axis=0)
    labels = np.concatenate((labels_train, labels_test), axis=0)

    # If specified, only take zeros and ones.
    if zeros_and_ones:
        labels_idx = np.where(labels < 2)[0]
        raw_mnist, labels = raw_mnist[labels_idx], labels[labels_idx]

    # Use all the MNIST digits unless specified.
    if dataset_size is None:
        dataset_size = len(raw_mnist)
    elif dataset_size <= len(raw_mnist):
        raw_mnist = raw_mnist[:dataset_size]
        labels = labels[:dataset_size]
    else:
        # Repeat MNIST if we request too large a size.
        n_repeat = -(-dataset_size // len(raw_mnist)) # Fancy ceiling division.
        raw_mnist = jnp.tile(raw_mnist, (n_repeat, 1, 1))[:dataset_size]
        labels = jnp.tile(labels, (n_repeat,))[:dataset_size]

    # Move the images to jnp arrays.
    raw_mnist = jnp.array(raw_mnist)[..., None]
    labels = jnp.array(labels)

    # Get the raw grass images.
    raw_grass = jnp.array(get_raw_grass_images(imagenet_path))

    # Pick from our batch of grass images.
    rng_order, rng_x, rng_y, rng = jax.random.split(rng, 4)
    grass_i = jax.random.randint(
        rng_order, shape=(dataset_size,), minval=0, maxval=len(raw_grass)
    )

    # Use jax.lax.map to get our random crops.
    def _map_random_crop(pair):
        return _random_crop(pair[0], pair[1], pair[2])
    grass_crop = jax.lax.map(
        _map_random_crop,
        (
            raw_grass[grass_i],
            jax.random.randint(rng_x, (dataset_size,), 0, 72),
            jax.random.randint(rng_y, (dataset_size,), 0, 72)
        ),
        batch_size=1024
    )

    final_images = grass_amp * grass_crop + mnist_amp * raw_mnist

    return final_images, labels


def _create_downsampling_matrix(image_size: int, patch_size: int):
    # Total number of pixels in the image
    feat_dim = image_size * image_size
    downsampling_matrix = jnp.zeros((feat_dim, feat_dim))

    # For each output pixel average a path_size x patch_size patch.
    for i in range(0, image_size, patch_size):
        for j in range(0, image_size, patch_size):
            # For each patch, map the average of all the pixels in the patch to
            # each individual pixel.
            patch_indices = [
                (i + di) * image_size + (j + dj) for di in range(patch_size)
                for dj in range(patch_size)
            ]
            for px in patch_indices:
                # Add weight for averaging (1/16 for each of the 4x4 block)
                downsampling_matrix = (
                    downsampling_matrix.at[px, patch_indices].add(
                        1 / (patch_size * patch_size)
                    )
                )

    return downsampling_matrix


def get_dataset(
    rng: Sequence[int], grass_amp: float, mnist_amp: float, noise: float,
    downsampling_ratios: Mapping[int, float] = None,
    sample_batch_size: int = None, imagenet_path: str = DEFAULT_IMAGENET_PATH,
    dataset_size: int=None, zeros_and_ones=True
):
    """Generate grassy MNIST dataset with A matrices and noise.

    Arguments:
        rng: Jax PRNGKey for sampling.
        grass_amp: Amplitude of grass component in returned images.
        mnist_amp: Amplitude of MNIST component in returned images.
        noise: Standard deviations of gaussian noise to add to images and
            covariance matrices.
        downsampling_ratios: Dictionary with keys equal to downsampling values
            and values equal to the fraction of images that should be assigned
            that downsampling ratios. Sum of values is assumed to be one.
        sample_batch_size: Batch size that will be used for posterior sampling.
            The size of the A matrices and cov_y returned will be
        imagenet_path: Path to imagenet grass dataset.
        dataset_size: Total size of dataset to generate. If None will be set by
            size of MNIST dataset.
        zeros_and_ones: If True, will only use 1 and 0 images from MNIST.
    """
    # Generate the initial dataset.
    rng_noise, _ = jax.random.split(rng)
    # Keep the same rng for corrupted_mnist_draw to be able to reproduce the
    # call outside this function.
    noiseless_images, labels = get_corrupted_mnist(
        rng, grass_amp, mnist_amp, imagenet_path, dataset_size, zeros_and_ones
    )
    image_shape = noiseless_images.shape[1:]

    # Set up our A matrices and covariance matrices.
    if sample_batch_size is None:
        sample_batch_size = len(noiseless_images)
    feat_dim = image_shape[0] * image_shape[1] * image_shape[2]
    cov_y = linalg.DPLR(
        diagonal=jnp.tile(jnp.ones(feat_dim) * noise ** 2, (sample_batch_size, 1))
    )

    # Find a division of the A matrices that respects the desired fractions.
    if downsampling_ratios is None:
        downsampling_ratios = {1: 1.0}
    total = 0
    num_ratios = {}
    for key in downsampling_ratios:
        num_ratios[key] = int(sample_batch_size * downsampling_ratios[key])
        total += int(sample_batch_size * downsampling_ratios[key])
    num_ratios[list(num_ratios.keys())[0]] += sample_batch_size - total

    # Populate the A matrices according to the number of copies you need.
    A_mat_list = []
    for key, value in num_ratios.items():
        A_mat_list.append(
            jnp.tile(
                _create_downsampling_matrix(image_shape[0], key)[None],
                (value, 1, 1)
            )
        )
    A_mat = jnp.concat(A_mat_list, axis=0)

    # Apply the downsampling
    noiseless_images = rearrange(
        jnp.matmul(
            A_mat[None],
            rearrange( # Flatten and add dimension to batch over.
                noiseless_images, '(N M) H W C -> N M (H W C) 1',
                M=sample_batch_size
            )
        ),
        'N M (H W C) 1 -> (N M) H W C', H=image_shape[0], W=image_shape[1],
        C=image_shape[2]
    )


    # Inject noise into the images
    final_images = (
        noiseless_images +
        jax.random.normal(rng_noise, shape=noiseless_images.shape) * noise
    )
    # Create the final A_mat with the amplitudes factored in.
    A_mat = jnp.concatenate(
        [grass_amp * A_mat[:, None], mnist_amp * A_mat[:, None]], axis=1
    )

    return final_images, A_mat, cov_y, labels
