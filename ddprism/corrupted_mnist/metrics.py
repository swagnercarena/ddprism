"""Methods for evaluating image samples."""
from typing import Sequence

from flax import linen as nn
from flax.training import train_state
import jax
from jax import Array
import jax.numpy as jnp
from kymatio.jax import Scattering2D
import optax
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from orbax.checkpoint import PyTreeCheckpointer
from pqm import pqm_chi2
import scipy

from ddprism.corrupted_mnist import datasets

class CNN(nn.Module):
    """A simple CNN model.

    Arguments:
        hidden_channels: Number of hidden channels for each convolutional layer.
        kernel_size: Kernel size for convlutional layers.
        features: Number of output features
        emb_length: Number of features in the second-to-last layer.
    """
    hidden_channels: Sequence[int] = (32, 64, 128)
    kernel_size: Sequence[int] = (3, 3)
    out_features: int = 10
    emb_features: int = 64

    @nn.compact
    def embed(self, x: Array) -> Array:
        """Embedding of CNN before last layer.

        Arguments:
            x: Input images with shape (*, H, W, C)

        Returns:
            Embedding with shape (*, emb_features).
        """
        # Start by applying the convolutional layers.
        for hidden_channel in self.hidden_channels:
            x = nn.relu(x)
            x = nn.Conv(hidden_channel, kernel_size=self.kernel_size)(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Apply two dense layers.
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(x)
        x = nn.Dense(self.emb_features)(x)
        return x

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Return classifier logits.

        Arguments:
            x: Input images with shape (*, H, W, C)

        Returns:
            Conditioned logit output with shape (*, out_features).
        """
        # Embed and apply final layer.
        x = self.embed(x)
        x = nn.relu(x)
        out = nn.Dense(self.out_features)(x)

        return out


@jax.jit
def _apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    _, grads = grad_fn(state.params)
    return grads

@jax.jit
def _update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_mnist_classifier(rng, model, batch_size=512, num_epochs=30, learning_rate=1e-2, momentum=0.9):
    """Train MNIST classifier for use with metrics.

    Arguments:
        rng: Jax PRNGKey to use for model training.
        model: nn.Module to use for logit prediction.

    Returns:
        Parameters of trained model.
    """
    # Initialize the parameters of the model.
    params = model.init(rng, jnp.ones((1, 28, 28, 1)))

    # Load the dataset.
    (x_train, y_train), (x_test, y_test) = datasets.get_raw_mnist_images()

    # Train the classifier.
    tx = optax.sgd(learning_rate=learning_rate, momentum=momentum)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params['params'], tx=tx
    )

    for _ in range(num_epochs):
        for _ in range(len(x_train) // batch_size):
            # Get batch.
            rng, rng_batch = jax.random.split(rng)
            batch_i = jax.random.randint(
                rng_batch, shape=(batch_size,), minval=0, maxval=len(x_train)
            )
            batch_images = x_train[batch_i][..., None] # Add channels.
            batch_labels = y_train[batch_i]

            # Update model.
            grads = _apply_model(state, batch_images, batch_labels)
            state = _update_model(state, grads)

        # Report the statistics.
        logits = state.apply_fn({'params': state.params}, x_test[..., None])
        accuracy = jnp.mean(jnp.argmax(logits, -1) == y_test)
    print(f'Model training complete with final accuracy {accuracy:.3f}')
    params = {'params': state.params}
    return params


def get_model(checkpoint_path, rng=None, **model_kwargs):
    """Load classifier checkpoint of train new classifier and save weights.

    Arguments:
        checkpoint_path: Path to checkpoint for read / write.
        rng: RNG key to use for model training.
        model_kwargs: Arguments for CNN module. If model exists at checkpoint,
            the kwargs for that model will overwrite these kwargs.

    Returns:
        Trained model and params for classifier CNN saved at checkpoint.
    """
    # Prepare the checkpoint manager.
    checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(create=True)
    checkpoint_manager = CheckpointManager(
        checkpoint_path, checkpointer, options=options
    )
    if rng is None:
        rng = jax.random.PRNGKey(2)

    if checkpoint_manager.latest_step() is None:
        # If there is no model at the checkpoint, train a new model.
        print(f'No checkpoints found at {checkpoint_path}. Training model...')
        model = CNN(**model_kwargs)
        params = train_mnist_classifier(rng, model)

        # Save model to checkpoint
        ckpt = {'params': params, 'model_kwargs': model_kwargs}
        checkpoint_manager.save(1, ckpt)
    else:
        ckpt = checkpoint_manager.restore(checkpoint_manager.latest_step())
        model_kwargs = ckpt['model_kwargs']
        model = CNN(**model_kwargs)
        params = ckpt['params']

    return model, params


def map_model_apply(model, params, x, batch_size=128, method=None):
    """Apply the model accross the batch dimension in batches.

    Arguments:
        model: Model to extract embedding from.
        params: Parameters of the model.
        x: Input values.
        batch_size: Batch size (for computing embeddings with classifier).
        method: Method to use. Default is call.

    Returns:
        Mapped outputs from method.
    """
    @jax.jit
    def _call(x_single):
        # Deal with lax destroying the batch dimension.
        return model.apply(params, x_single[None], method=method)[0]

    return jax.lax.map(_call, x, batch_size=batch_size)


def fcd_mnist(model, params, dist_1, dist_2, batch_size=128) -> Array:
    r"""Computes the Fréchet classification distance between samples.

    Arguments:
        model: Model to extract embedding from.
        params: Parameters of the model.
        dist_1: Samples from the first distribution.
        dist_2: Samples from the second distribution.
        batch_size: Batch size (for computing embeddings with classifier).

    Returns:
        Fréchet classifier distance between two distributions.

    Notes:
        Closely follows https://github.com/kvfrans/jax-fid-parallel/tree/main.
    """
    # Compute embeddings on both distributions.
    embed_1 = map_model_apply(
        model, params, dist_1, batch_size=batch_size, method='embed'
    )
    embed_2 = map_model_apply(
        model, params, dist_2, batch_size=batch_size, method='embed'
    )

    # Estimate mean and covariance of the two distributions.
    mu_1 = jnp.mean(embed_1, axis=0)
    sigma_1 = jnp.cov(embed_1, rowvar=False)
    mu_2 = jnp.mean(embed_2, axis=0)
    sigma_2 = jnp.cov(embed_1, rowvar=False)

    # Compute the Fréchet distance between the two multivariate Gaussian
    # distributions, following e.g. Eq. (6) in Heusel et al. 2018
    # (https://arxiv.org/pdf/1706.08500).
    diff = mu_1 - mu_2
    offset = jnp.eye(sigma_1.shape[0]) * 1e-6
    cov_sqrt = scipy.linalg.sqrtm((sigma_1 + offset) @ (sigma_2 + offset))
    fcd = diff @ diff + jnp.trace(sigma_1 + sigma_2 - 2 * cov_sqrt.real)
    return fcd


def inception_score_mnist(model, params, dist, batch_size=128) -> Array:
    r"""Compute the inception score equivalent with our classifier.

    Arguments:
        model: Model to extract embedding from.
        params: Parameters of the model.
        dist: Samples from the distribution.
        batch_size: Batch size (for computing embeddings with classifier).

    Returns:
        Inception score for a distribution of images.

    Notes:
        Closely follows torchmetrics implementation.
    """
    logits = map_model_apply(model, params, dist, batch_size=batch_size)

    # Get the probability, log probability, and marginal
    probs = jax.nn.softmax(logits)
    log_probs = jax.nn.log_softmax(logits)
    marginal = jnp.mean(probs, axis=0)

    # Calculate mean KL between conditional and marginal.
    kl = jnp.mean(
        jnp.sum(probs * (log_probs - jnp.log(marginal)[None]), axis=1),
        axis=0
    )

    return jnp.exp(kl)


def pq_mass(dist_1, dist_2, **kwargs) -> Array:
    r"""Computes PQMass chi squared values: https://arxiv.org/abs/2402.04355.

    Arguments:
        dist_1: Samples from the first distribution.
        dist_2: Samples from the second distribution.
        **kwargs: Additional arguments for computing PQMass.

    Returns:
        Mean chi-squared values from PQMass.

    """
    # Flatten outputs if not already flat.
    dist_1 = dist_1.reshape(dist_1.shape[0], -1)
    dist_2 = dist_2.reshape(dist_2.shape[0], -1)

    chi2_vals = pqm_chi2(dist_1, dist_2, **kwargs)

    return jnp.mean(chi2_vals)


def compute_snr(signal: Array) -> Array:
    """Computes the signal-to-noise ratio of an image or signal.

    SNR is calculated as the ratio of the peak signal power to the variance of
    the signal. For N-dimensional signals (N<=3), we compute this over all dimensions.

    Arguments:
        signal: Input signal with shape: (N, L), (N, H, W), or (N, H, W, C)
            where N is the batch dimension if present.

    Returns:
        SNR value(s) in decibels.

    Notes:
        Since we do not know the true signal, we use the peak signal power as a
        proxy for the true signal.
    """
    # Handle different input dimensions
    ndim = signal.ndim
    if ndim > 4 or ndim < 2:
        raise ValueError(
            f"Signal must have 2-4 dimensions (got {ndim}). "
            "Expected shapes: (N,L), (N,H,W), or (N,H,W,C)"
        )

    # Compute peak signal power over all non-batch dimensions
    max_power = jnp.max(signal ** 2, axis=tuple(range(1, signal.ndim)))

    # Compute signal variance over all non-batch dimensions
    variance = jnp.var(signal, axis=tuple(range(1, signal.ndim)))

    # Compute SNR in decibels, adding small epsilon to avoid division by zero
    snr_db = 10 * jnp.log10(max_power / (variance + 1e-10))

    return snr_db


def compute_wavelet_sparsity(
    signal: Array, levels: int = 3, threshold: float = 3.0
) -> float:
    """Compute sparsity of wavelet coefficients above a threshold.

    Arguments:
        signal: Input image with shape (H, W, C) or batch of images
            (N, H, W, C)
        levels: Number of levels to use for wavelet transform.
        threshold: Threshold for wavelet coefficients (in units of standard
            deviation).

    Returns:
        Fraction of wavelet coefficients above a threshold.

    Notes:
        Assume signal has only one channel.
    """
    # Compute wavelet transform.
    scatt_transform = jax.vmap(
        Scattering2D(levels, signal.shape[1:-1]).scattering
    )
    coeffs = scatt_transform(signal[..., 0])

    # Estimate the standard deviations using the median absolute deviation.
    mad = jnp.median(jnp.abs(coeffs), axis=(1,2,3), keepdims=True)
    # Convert to standard deviation assuming normal distribution.
    # See https://en.wikipedia.org/wiki/Median_absolute_deviation.
    std = mad / 0.6745

    # Set the threshold for each image.
    imag_threshold = threshold * std

    return jnp.mean(coeffs > imag_threshold)
