"Test scripts for training_utils.py"

from absl.testing import absltest
import functools

import chex
from flax.core import FrozenDict
from flax.training import train_state
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import optax

from ddprism import diffusion
from ddprism import embedding_models
from ddprism import training_utils


def _create_test_config():
    """Create a test configuration with all configurable parameters."""
    config = ConfigDict()

    # Basic parameters
    config.feat_dim = 5
    config.emb_features = 32
    config.lr_init_val = 1e-3
    config.epochs = 100
    config.em_laps = 10
    config.batch_size = 32
    config.ema_decay = 0.999

    # SDE parameters
    config.sde = ConfigDict({'a': 1e-4, 'b': 1e2})

    # U-Net parameters
    config.hid_channels = (16, 32)
    config.hid_blocks = (1, 1)
    config.kernel_size = (3, 3)
    config.heads = {'1': 2}
    config.dropout_rate = 0.1

    # TimeMLP parameters
    config.hidden_features = (32, 32)
    config.time_mlp_normalize = True
    config.time_conditioning = 'concat'
    config.dropout_rate = 0.0


    # Configurable hyperparameters
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


class TimeSamplingTests(chex.TestCase):
    """Run tests on time sampling functions."""

    @chex.all_variants
    def test_get_time_sampling_fn_beta(self):
        """Test beta distribution time sampling."""
        config = _create_test_config()
        config.time_sampling = ConfigDict({
            'distribution': 'beta',
            'beta_a': 3.0,
            'beta_b': 3.0
        })

        sample_time = training_utils.get_time_sampling_fn(config)
        rng = jax.random.PRNGKey(0)
        shape = (10,)

        variant_sample_time = self.variant(sample_time, static_argnames='shape')
        samples = variant_sample_time(rng, shape)

        self.assertTupleEqual(samples.shape, shape)
        self.assertTrue(jnp.all(samples >= 0.0))
        self.assertTrue(jnp.all(samples <= 1.0))

        # Test config = None.
        sample_time = training_utils.get_time_sampling_fn(None)
        variant_sample_time = self.variant(sample_time, static_argnames='shape')
        samples = variant_sample_time(rng, shape)

        self.assertTupleEqual(samples.shape, shape)
        self.assertTrue(jnp.all(samples >= 0.0))
        self.assertTrue(jnp.all(samples <= 1.0))

        # Test invalid distribution.
        config = _create_test_config()
        config.time_sampling = ConfigDict({
            'distribution': 'invalid'
        })

        with self.assertRaises(ValueError):
            training_utils.get_time_sampling_fn(config)


class OptimizerTests(chex.TestCase):
    """Run tests on optimizer configuration functions."""

    def test_get_optimizer(self):
        """Test Adam optimizer creation."""
        config = _create_test_config()
        config.optimizer = ConfigDict({
            'type': 'adam',
            'beta1': 0.8,
            'beta2': 0.95,
            'eps': 1e-8
        })

        optimizer_fn = training_utils.get_optimizer(config)
        lr = 1e-3
        optimizer = optimizer_fn(lr)
        self.assertIsInstance(optimizer, optax.GradientTransformation)

        config.optimizer.type = 'adamw'
        optimizer_fn = training_utils.get_optimizer(config)
        lr = 1e-3
        optimizer = optimizer_fn(lr)
        self.assertIsInstance(optimizer, optax.GradientTransformation)

        config.optimizer.type = 'invalid_optimizer'
        with self.assertRaises(ValueError):
            optimizer_fn = training_utils.get_optimizer(config)


class LearningRateScheduleTests(chex.TestCase):
    """Run tests on learning rate schedule functions."""

    def test_get_learning_rate_schedule(self):
        """Test cosine decay schedule."""
        config = _create_test_config()
        config.lr_schedule.type = 'cosine'
        config.lr_schedule.warmup_steps = 5
        config.lr_schedule.min_lr_ratio = 0.1
        base_lr = 1e-3
        total_steps = 100

        schedule = training_utils.get_learning_rate_schedule(
            config, base_lr, total_steps
        )

        # Test that schedule function works
        self.assertEqual(schedule(0), 0.0)
        self.assertEqual(schedule(config.lr_schedule.warmup_steps), base_lr)
        self.assertAlmostEqual(
            schedule(total_steps), base_lr * config.lr_schedule.min_lr_ratio
        )

        # Test exponential decay schedule.
        config.lr_schedule.type = 'exponential'
        config.lr_schedule.decay_rate = 0.9
        config.lr_schedule.decay_steps = 50

        schedule = training_utils.get_learning_rate_schedule(
            config, base_lr, total_steps
        )
        self.assertEqual(schedule(0), base_lr)
        self.assertAlmostEqual(
            schedule(config.lr_schedule.decay_steps), base_lr * 0.9
        )

        # Test invalid schedule.
        config.lr_schedule.type = 'invalid_schedule'
        with self.assertRaises(ValueError):
            training_utils.get_learning_rate_schedule(config, 1e-3, 100)


class ApplyModelTests(chex.TestCase):
    """Run tests on apply_model function."""

    def _create_simple_state(self, rng, config):
        """Create a simple train state for testing."""
        sde = diffusion.VESDE(config.sde.a, config.sde.b)
        time_mlp = embedding_models.TimeMLP(
            features=config.feat_dim,
            hid_features=config.hidden_features,
            normalize=config.time_mlp_normalize
        )
        denoiser = diffusion.Denoiser(
            sde, time_mlp, emb_features=config.emb_features
        )

        params = denoiser.init(
            rng, jnp.ones((1, config.feat_dim)), jnp.ones((1,))
        )

        optimizer = training_utils.get_optimizer(config)(
            lambda step: config.lr_init_val
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm), optimizer
        )

        return train_state.TrainState.create(
            apply_fn=denoiser.apply, params=params['params'], tx=tx
        )

    @chex.all_variants
    def test_apply_model_with_config(self):
        """Test apply_model function with config."""
        config = _create_test_config()
        rng = jax.random.PRNGKey(0)

        state = self._create_simple_state(rng, config)

        # Create test data
        batch_size = 4
        x = jax.random.normal(rng, (batch_size, config.feat_dim))

        apply_model = self.variant(
            functools.partial(
                training_utils.apply_model, config=config, pmap=False
            )
        )

        _, loss = apply_model(state, x, rng)

        # Check output shape
        self.assertEqual(loss.shape, ())

        # Test apply_model without config.
        apply_model = self.variant(
            functools.partial(
                training_utils.apply_model, config=None, pmap=False
            )
        )
        _, loss = apply_model(state, x, rng)

        self.assertEqual(loss.shape, ())


class DenoiserCreationTests(chex.TestCase):
    """Run tests on denoiser creation functions."""

    def test_create_denoiser_timemlp(self):
        """Test TimeMLP denoiser creation."""
        config = _create_test_config()

        denoiser = training_utils.create_denoiser_timemlp(config)

        self.assertIsInstance(denoiser, diffusion.Denoiser)
        self.assertIsInstance(denoiser.score_model, embedding_models.TimeMLP)

    def test_create_denoiser_gaussian(self):
        """Test Gaussian denoiser creation."""
        config = _create_test_config()

        denoiser = training_utils.create_denoiser_gaussian(config)

        self.assertIsInstance(denoiser, diffusion.GaussianDenoiserDPLR)

    def test_create_denoiser_unet(self):
        """Test UNet denoiser creation."""
        config = _create_test_config()
        image_shape = (8, 8, 1)

        denoiser = training_utils.create_denoiser_unet(config, image_shape)

        self.assertIsInstance(denoiser, diffusion.Denoiser)
        self.assertIsInstance(denoiser.score_model, embedding_models.FlatUNet)


class TrainStateCreationTests(chex.TestCase):
    """Run tests on train state creation functions."""

    def test_create_train_state_timemlp(self):
        """Test TimeMLP train state creation."""
        config = _create_test_config()
        rng = jax.random.PRNGKey(0)

        learning_rate_fn = training_utils.get_learning_rate_schedule(
            config, config.lr_init_val, config.epochs
        )
        state = training_utils.create_train_state_timemlp(
            rng, config, learning_rate_fn
        )

        self.assertIsInstance(state, train_state.TrainState)
        self.assertTrue(hasattr(state, 'params'))
        self.assertTrue(hasattr(state, 'tx'))

    def test_create_train_state_gaussian(self):
        """Test Gaussian train state creation."""
        config = _create_test_config()
        rng = jax.random.PRNGKey(0)

        learning_rate_fn = training_utils.get_learning_rate_schedule(
            config, config.lr_init_val, config.epochs
        )
        state = training_utils.create_train_state_gaussian(
            rng, config, learning_rate_fn
        )

        self.assertIsInstance(state, train_state.TrainState)
        self.assertTrue(hasattr(state, 'params'))
        self.assertTrue(hasattr(state, 'tx'))

    def test_create_train_state_unet(self):
        """Test UNet train state creation."""
        config = _create_test_config()
        rng = jax.random.PRNGKey(0)
        image_shape = (8, 8, 1)

        learning_rate_fn = training_utils.get_learning_rate_schedule(
            config, config.lr_init_val, config.epochs
        )
        state = training_utils.create_train_state_unet(
            rng, config, learning_rate_fn, image_shape
        )

        self.assertIsInstance(state, train_state.TrainState)
        self.assertTrue(hasattr(state, 'params'))
        self.assertTrue(hasattr(state, 'tx'))


class EMATests(chex.TestCase):
    """Run tests on EMA class."""

    def test_ema_initialization(self):
        """Test EMA initialization."""
        params = FrozenDict(
            {'param1': jnp.array([1.0, 2.0]), 'param2': jnp.array([3.0])}
        )
        ema = training_utils.EMA(params)

        self.assertIsInstance(ema.params, FrozenDict)
        self.assertTrue(jnp.allclose(ema.params['param1'], params['param1']))
        self.assertTrue(jnp.allclose(ema.params['param2'], params['param2']))

    def test_ema_update(self):
        """Test EMA parameter update."""
        initial_params = FrozenDict({
            'param1': jnp.array([1.0, 2.0]),
            'param2': jnp.array([3.0])
        })
        new_params = FrozenDict({
            'param1': jnp.array([2.0, 3.0]),
            'param2': jnp.array([4.0])
        })

        ema = training_utils.EMA(initial_params)
        decay = 0.9

        updated_ema = ema.update(new_params, decay)

        # Check that EMA update works correctly
        expected_param1 = (
            decay * initial_params['param1'] +
            (1 - decay) * new_params['param1']
        )
        expected_param2 = (
            decay * initial_params['param2'] +
            (1 - decay) * new_params['param2']
        )

        self.assertTrue(
            jnp.allclose(updated_ema.params['param1'], expected_param1)
        )
        self.assertTrue(
            jnp.allclose(updated_ema.params['param2'], expected_param2)
        )


if __name__ == '__main__':
    absltest.main()
