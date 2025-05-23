"Test scripts for random_manifolds.py"

from absl.testing import absltest

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp

from galaxy_diffusion import embedding_models


class EmbeddingTests(chex.TestCase):
    """Run tests on embedding functions."""

    @chex.all_variants
    def test_positional_embedding(self):
        """Test positional embedding."""
        x = jnp.linspace(0, 1, 20)
        emb_features = 128
        positional_embedding = self.variant(
            embedding_models.positional_embedding,
            static_argnames='emb_features'
        )
        embedding = positional_embedding(x, emb_features=emb_features)

        self.assertTupleEqual(embedding.shape, (20, 128))
        # Test a single value
        i = 3
        j = 10
        freq = (1 / 1e4) ** (j / (emb_features // 2 - 1))
        self.assertAlmostEqual(embedding[i, j], jnp.sin(freq * x[i]))


class TimeMLPTests(chex.TestCase):
    """Run tests of TimeMLP"""

    @chex.all_variants
    def test_call(self):
        """Test that calling the TimeMLP behaves as expected."""
        rng = jax.random.PRNGKey(0)
        features = 5
        time_features = 2
        hid_features = (32, 32)
        activation = nn.gelu
        normalize = True

        # Initialize our TimeMLP and apply function.
        time_mlp = embedding_models.TimeMLP(
            features, hid_features, activation, normalize
        )
        params = time_mlp.init(
            rng, jnp.ones((1, features)), jnp.ones((1, time_features))
        )
        apply_func = self.variant(time_mlp.apply)

        # Test that the output has the desired shape.
        batch_size = 32
        rng_x, rng_t = jax.random.split(rng)
        x_draws = jax.random.normal(rng_x, shape=(batch_size, features))
        t_draws = jax.random.uniform(rng_t, shape=(batch_size, time_features))

        # Check the shape is correct.
        expect_x = apply_func(params, x_draws, t_draws)
        self.assertTupleEqual(expect_x.shape, (batch_size, features))


class AdaLNZeroModulationTests(chex.TestCase):
    """Run tests for LayerNorm"""

    @chex.all_variants
    def test_call(self):
        """Test the call returns three small perturbations."""
        channels = 3
        emb_features = 64
        time_features = 2
        activation = nn.gelu
        rng = jax.random.PRNGKey(2)

        modulation = embedding_models.AdaLNZeroModulation(
            channels, emb_features, activation
        )
        params = modulation.init(rng, jnp.ones((1, time_features)))
        apply_func = self.variant(modulation.apply)

        # Test the output shape and variance.
        batch_size = 32
        t_draws = jax.random.uniform(rng, shape=(batch_size, time_features))
        output = apply_func(params, t_draws)
        for scaling in output:
            self.assertTupleEqual(scaling.shape, (batch_size, 1, 1, 3))
            self.assertLess(jnp.std(scaling), 1.0)


class ResBlockTests(chex.TestCase):
    """Run tests for Resnet"""

    @chex.all_variants
    def test_call(self):
        """Test the shape of resnet block output."""
        channels = 5
        emb_features = 64
        time_features = 2
        dropout_rate = 0.1
        activation = nn.gelu
        rng = jax.random.PRNGKey(2)
        image_size = 5

        resblock = embedding_models.ResBlock(
            channels, emb_features, dropout_rate, activation
        )
        params = resblock.init(
            rng, jnp.ones((1, image_size, image_size, channels)),
            jnp.ones((1, time_features))
        )
        apply_func = self.variant(resblock.apply, static_argnames=['train'])

        # Test the output shape and variance.
        batch_size = 32
        t_draws = jax.random.uniform(rng, shape=(batch_size, time_features))
        x_draws = jax.random.normal(
            rng, shape=(batch_size, image_size, image_size, channels)
        )
        output = apply_func(
            params, x_draws, t_draws, train=True, rngs={'dropout': rng}
        )

        self.assertTupleEqual(
            output.shape, (batch_size, image_size, image_size, channels)
        )


class AttBlockTests(chex.TestCase):
    """Run tests for Attention."""

    @chex.all_variants
    def test_call(self):
        """Test the shape of attention block output."""
        channels = 8
        emb_features = 64
        time_features = 2
        heads = 2
        dropout_rate = 0.1
        rng = jax.random.PRNGKey(2)
        image_size = 5

        attblock = embedding_models.AttBlock(
            channels, emb_features, dropout_rate, heads
        )
        params = attblock.init(
            rng, jnp.ones((1, image_size, image_size, channels)),
            jnp.ones((1, time_features))
        )
        apply_func = self.variant(attblock.apply, static_argnames=['train'])

        # Test the output shape and variance.
        batch_size = 32
        t_draws = jax.random.uniform(rng, shape=(batch_size, time_features))
        x_draws = jax.random.normal(
            rng, shape=(batch_size, image_size, image_size, channels)
        )
        output = apply_func(
            params, x_draws, t_draws, train=True, rngs={'dropout': rng}
        )

        self.assertTupleEqual(
            output.shape, (batch_size, image_size, image_size, channels)
        )


class UtilityTests(chex.TestCase):
    """Run tests for spare functions."""

    @chex.all_variants
    def test__resize(self):
        """Test that the resizing works well."""
        image_size = 2
        channels = 3
        x = jax.random.normal(
            jax.random.PRNGKey(2), (3, 3, image_size, image_size, channels)
        )

        # Try doubling image
        factor = (2.0, 2.0)
        resample = embedding_models.Resample(factor)
        apply_func = self.variant(resample.apply)
        x_comp = apply_func({}, x)
        assert jnp.allclose(x_comp, jnp.repeat(jnp.repeat(x, 2, 2), 2, 3))


class UNetTests(chex.TestCase):
    """Run tests for UNet."""

    @chex.all_variants
    def test_call(self):
        """Test that the UNet returns desired outputs."""
        image_size = 64
        channels = 3
        batch_size = 32
        time_features = 2
        rng = jax.random.PRNGKey(2)
        t_draws = jax.random.uniform(rng, shape=(batch_size, time_features))
        x_draws = jax.random.normal(
            rng, shape=(batch_size, image_size, image_size, channels)
        )

        # Initialize UNet.
        hid_channels = (4, 8, 16)
        hid_blocks = (2, 2, 2)
        kernel_size = (3, 3)
        heads = {'2': 1}
        emb_features = 64
        dropout_rate = 0.1
        unet = embedding_models.UNet(
            hid_channels, hid_blocks, kernel_size, emb_features, heads,
            dropout_rate
        )
        params = unet.init(
            rng, jnp.ones((1, image_size, image_size, channels)),
            jnp.ones((1, time_features))
        )
        apply_func = self.variant(unet.apply, static_argnames=['train'])

        # Test the output shape.
        output = apply_func(
            params, x_draws, t_draws, train=True, rngs={'dropout': rng}
        )
        self.assertTupleEqual(
            output.shape, (batch_size, image_size, image_size, channels)
        )


class FlatUNetTest(chex.TestCase):
    """Run tests for the FlatUNet."""

    @chex.all_variants
    def test_call(self):
        """Test that the FlatUNet returns the desired outputs."""
        image_size = 64
        channels = 3
        batch_size = 32
        time_features = 2
        rng = jax.random.PRNGKey(2)
        t_draws = jax.random.uniform(rng, shape=(batch_size, time_features))
        x_draws = jax.random.normal(
            rng, shape=(batch_size, image_size, image_size, channels)
        )
        image_shape = x_draws.shape[-3:]
        x_flat = x_draws.reshape(x_draws.shape[:-3] + (-1,))

        # Initialize UNet.
        hid_channels = (4, 8, 16)
        hid_blocks = (2, 2, 2)
        kernel_size = (3, 3)
        heads = {'2': 1}
        emb_features = 64
        dropout_rate = 0.1
        unet = embedding_models.FlatUNet(
            hid_channels, hid_blocks, kernel_size, emb_features, heads,
            dropout_rate, image_shape=image_shape
        )
        params = unet.init(
            rng, jnp.ones((1, image_size * image_size * channels)),
            jnp.ones((1, time_features))
        )
        apply_func = self.variant(unet.apply, static_argnames=['train'])

        # Test the output shape.
        output = apply_func(
            params, x_flat, t_draws, train=True, rngs={'dropout': rng}
        )
        self.assertTupleEqual(
            output.shape, (batch_size, image_size * image_size * channels)
        )


if __name__ == '__main__':
    absltest.main()
