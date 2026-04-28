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

# Force x64 before any JAX op runs. The TrainStepTests byte-identity
# assertions need fp64 (fp32 has XLA op-reorder drift on the transformer).
# Module-level imports above only declare Flax modules and don't trigger
# jit compilation, so the update applies to every subsequent trace.
jax.config.update('jax_enable_x64', True)


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


class TrainStepTests(chex.TestCase):
    """Run tests on the fused healpix train_step function.

    Module-level `JAX_ENABLE_X64=1` (set above the jax import) keeps the
    byte-identity assertions stable on the transformer-sized model.
    """

    @chex.all_variants
    def test_train_step_matches_separate_calls(self):
        """train_step must produce the same (state, ema_params, loss) as the
        equivalent apply_model + update_model + manual EMA tree_map on the
        healpix path (with vec_map plumbed through)."""
        config = _create_test_config()
        rng = jax.random.PRNGKey(0)
        healpix_shape = (64 * 64, 2)
        learning_rate_fn = training_utils.get_learning_rate_schedule(
            config, config.lr_init_val, config.epochs,
        )
        state = training_utils_healpix.create_train_state_transformer(
            rng, config, learning_rate_fn, healpix_shape,
        )
        batch_size = 4
        x = jax.random.normal(
            rng, (batch_size, healpix_shape[1] * healpix_shape[0])
        )
        vec_map = jax.random.normal(
            rng, (batch_size, healpix_shape[0], 3)
        )
        decay = jnp.float32(0.99)
        ema_params = jax.tree_util.tree_map(jnp.copy, state.params)

        # Reference path.
        grads, ref_loss = training_utils_healpix.apply_model(
            state, x, vec_map, rng, config=config, pmap=False,
        )
        ref_state = training_utils.update_model(state, grads)
        ref_ema = jax.tree_util.tree_map(
            lambda e, n: decay * e + (1.0 - decay) * n,
            ema_params, ref_state.params,
        )

        # Fused path.
        step = self.variant(
            functools.partial(
                training_utils_healpix.train_step,
                config=config, pmap=False,
            )
        )
        new_state, new_ema, loss = step(
            state, ema_params, x, vec_map, rng, decay=decay,
        )

        self.assertTrue(jnp.allclose(loss, ref_loss, atol=1e-6))
        for ref, new in zip(
            jax.tree_util.tree_leaves(ref_state.params),
            jax.tree_util.tree_leaves(new_state.params),
        ):
            self.assertTrue(jnp.allclose(ref, new, atol=1e-6))
        for ref, new in zip(
            jax.tree_util.tree_leaves(ref_ema),
            jax.tree_util.tree_leaves(new_ema),
        ):
            self.assertTrue(jnp.allclose(ref, new, atol=1e-6))

    def test_train_step_multistep_byte_identical(self):
        """Across multiple steps, the fused train_step must produce byte-
        identical state, ema_params, and loss versus the unfused path."""
        config = _create_test_config()
        rng = jax.random.PRNGKey(0)
        healpix_shape = (64 * 64, 2)
        learning_rate_fn = training_utils.get_learning_rate_schedule(
            config, config.lr_init_val, config.epochs,
        )
        state = training_utils_healpix.create_train_state_transformer(
            rng, config, learning_rate_fn, healpix_shape,
        )
        batch_size = 4
        feat = healpix_shape[1] * healpix_shape[0]
        decay = jnp.float64(0.99)

        n_steps = 5
        x_pool = jax.random.normal(rng, (n_steps, batch_size, feat))
        vec_pool = jax.random.normal(
            rng, (n_steps, batch_size, healpix_shape[0], 3),
        )

        ref_state = state
        ref_ema = jax.tree_util.tree_map(jnp.copy, state.params)
        ref_losses = []
        rng_ref = jax.random.PRNGKey(42)
        for k in range(n_steps):
            rng_step, rng_ref = jax.random.split(rng_ref)
            grads, loss = training_utils_healpix.apply_model(
                ref_state, x_pool[k], vec_pool[k], rng_step,
                config=config, pmap=False,
            )
            ref_state = training_utils.update_model(ref_state, grads)
            ref_ema = jax.tree_util.tree_map(
                lambda e, n: decay * e + (1.0 - decay) * n,
                ref_ema, ref_state.params,
            )
            ref_losses.append(float(loss))

        new_state = state
        new_ema = jax.tree_util.tree_map(jnp.copy, state.params)
        new_losses = []
        rng_new = jax.random.PRNGKey(42)
        for k in range(n_steps):
            rng_step, rng_new = jax.random.split(rng_new)
            new_state, new_ema, loss = training_utils_healpix.train_step(
                new_state, new_ema, x_pool[k], vec_pool[k], rng_step,
                decay=decay, config=config, pmap=False,
            )
            new_losses.append(float(loss))

        for ref, new in zip(ref_losses, new_losses):
            self.assertEqual(ref, new)
        for ref, new in zip(
            jax.tree_util.tree_leaves(ref_state.params),
            jax.tree_util.tree_leaves(new_state.params),
        ):
            self.assertTrue(jnp.array_equal(ref, new))
        for ref, new in zip(
            jax.tree_util.tree_leaves(ref_ema),
            jax.tree_util.tree_leaves(new_ema),
        ):
            self.assertTrue(jnp.array_equal(ref, new))


if __name__ == '__main__':
    absltest.main()
