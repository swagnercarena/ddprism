"""Test scripts for training_utils_healpix.py"""

from absl.testing import absltest
import functools

import chex
from flax.training import train_state
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from ddprism import training_utils
from ddprism.tsz_cmb import diffusion_healpix, training_utils_healpix


def _create_test_config():
    """Create a test configuration with all configurable parameters."""
    config = ConfigDict()

    # Basic parameters
    config.emb_features = 32
    config.lr_init_val = 1e-3
    config.epochs = 100
    config.batch_size = 32
    config.ema_decay = 0.999

    # SDE parameters
    config.sde = ConfigDict({'a': 1e-4, 'b': 1e2})

    # Transformer parameters
    config.n_blocks = 2
    config.dropout_rate_block = [0.1, 0.1]
    config.heads = 4
    config.patch_size_list = [32**2, 64**2]
    config.time_emb_features = 64
    config.n_average_layers = 1
    config.use_patch_convolution = True

    # Training hyperparameters
    config.optimizer = ConfigDict({
        'type': 'adam',
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.0,
        'eps': 1e-8
    })
    config.lr_schedule = ConfigDict({
        'type': 'cosine',
        'warmup_steps': 10,
        'min_lr_ratio': 0.1
    })
    config.grad_clip_norm = 1.0
    config.time_sampling = ConfigDict({
        'distribution': 'beta',
        'beta_a': 3.0,
        'beta_b': 3.0
    })

    return config


class DenoiserCreationTests(chex.TestCase):
    """Run tests on denoiser creation functions."""

    def test_create_denoiser_transformer(self):
        """Test Transformer denoiser creation."""
        config = _create_test_config()
        healpix_shape = (64 * 64, 2)

        denoiser = training_utils_healpix.create_denoiser_transformer(
            config, healpix_shape
        )

        self.assertIsInstance(denoiser, diffusion_healpix.Denoiser)
        self.assertEqual(denoiser.n_pixels, healpix_shape[0])
        self.assertEqual(denoiser.emb_features, config.time_emb_features)


class TrainStateCreationTests(chex.TestCase):
    """Run tests on train state creation functions."""

    def test_create_train_state_transformer(self):
        """Test Transformer train state creation."""
        config = _create_test_config()
        rng = jax.random.PRNGKey(0)
        healpix_shape = (64 * 64, 2)

        learning_rate_fn = training_utils.get_learning_rate_schedule(
            config, config.lr_init_val, config.epochs
        )
        state = training_utils_healpix.create_train_state_transformer(
            rng, config, learning_rate_fn, healpix_shape
        )

        self.assertIsInstance(state, train_state.TrainState)
        self.assertTrue(hasattr(state, 'params'))
        self.assertTrue(hasattr(state, 'tx'))

    def test_create_train_state_transformer_with_params(self):
        """Test Transformer train state creation with provided params."""
        config = _create_test_config()
        rng = jax.random.PRNGKey(0)
        healpix_shape = (64 * 64, 2)

        # First create initial params
        denoiser = training_utils_healpix.create_denoiser_transformer(
            config, healpix_shape
        )
        healpix_features = denoiser.score_model.feat_dim
        params = denoiser.init(
            rng, jnp.ones((1, healpix_features)), jnp.ones((1,))
        )

        learning_rate_fn = training_utils.get_learning_rate_schedule(
            config, config.lr_init_val, config.epochs
        )

        # Create train state with provided params
        state = training_utils_healpix.create_train_state_transformer(
            rng, config, learning_rate_fn, healpix_shape, params=params
        )

        self.assertEqual(
            denoiser.score_model.feat_dim, healpix_shape[1] * healpix_shape[0],
        )
        self.assertIsInstance(state, train_state.TrainState)
        self.assertTrue(hasattr(state, 'params'))
        self.assertTrue(hasattr(state, 'tx'))


class ApplyModelTests(chex.TestCase):
    """Run tests on apply_model function."""

    @chex.all_variants
    def test_apply_model_with_config(self):
        """Test apply_model function with config."""
        config = _create_test_config()
        rng = jax.random.PRNGKey(0)
        healpix_shape = (64 * 64, 2)

        learning_rate_fn = training_utils.get_learning_rate_schedule(
            config, config.lr_init_val, config.epochs
        )
        state = training_utils_healpix.create_train_state_transformer(
            rng, config, learning_rate_fn, healpix_shape
        )

        # Create test data
        batch_size = 4
        x = jax.random.normal(
            rng, (batch_size, healpix_shape[1] * healpix_shape[0])
        )
        vec_map = jax.random.normal(
            rng, (batch_size, healpix_shape[0], 3)
        )
        apply_model = self.variant(
            functools.partial(
                training_utils_healpix.apply_model, config=config, pmap=False
            )
        )

        _, loss = apply_model(state, x, vec_map, rng)

        # Check output shape
        self.assertEqual(loss.shape, ())

        # Test apply_model without config.
        apply_model = self.variant(
            functools.partial(
                training_utils_healpix.apply_model, config=None, pmap=False
            )
        )
        _, loss = apply_model(state, x, vec_map, rng)

        self.assertEqual(loss.shape, ())


if __name__ == '__main__':
    absltest.main()
