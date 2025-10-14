"Test scripts for tsz_cmb/embedding_models.py"

from absl.testing import absltest

import chex
import healpy as hp
import jax
import jax.numpy as jnp

from ddprism.tsz_cmb import embedding_models_healpix


class RelativeBiasTests(chex.TestCase):
    """Run tests on RelativeBias class."""

    @chex.all_variants
    def test_call(self):
        """Test that RelativeBias returns correct shapes and values."""
        rng = jax.random.PRNGKey(0)
        n_heads = 4
        freq_features = 8
        nside = 8
        batch_size = 2
        n_pixels = 8 ** 2

        # Create test vector map (unit vectors on sphere)
        vec_map = jnp.stack(
            hp.pix2vec(nside, jnp.arange(n_pixels), nest=True), axis=-1
        )
        vec_map = jnp.stack([vec_map] * batch_size, axis=0)
        # Normalize to unit vectors
        vec_map = vec_map / jnp.linalg.norm(vec_map, axis=-1, keepdims=True)

        # Initialize RelativeBias
        relative_bias = embedding_models_healpix.RelativeBias(
            n_heads=n_heads, freq_features=freq_features
        )
        params = relative_bias.init(rng, vec_map)
        apply_func = self.variant(relative_bias.apply)

        # Test output shape
        output = apply_func(params, vec_map)
        expected_shape = (batch_size, n_pixels, n_pixels, n_heads)
        self.assertTupleEqual(output.shape, expected_shape)

        # Test that the bias is symmetric (since distance is symmetric)
        self.assertTrue(
            jnp.allclose(output, jnp.swapaxes(output, -3, -2), atol=1e-5)
        )


class HEALPixAttentionTests(chex.TestCase):
    """Run tests on HEALPixAttention class."""

    @chex.all_variants
    def test_call(self):
        """Test that HEALPixAttention returns correct shapes."""
        rng = jax.random.PRNGKey(1)
        emb_features = 64
        n_heads = 4
        dropout_rate = 0.1
        nside = 8
        n_pixels = 8 ** 2
        batch_size = 2

        # Create vector map
        vec_map = jnp.stack(
            hp.pix2vec(nside, jnp.arange(n_pixels), nest=True), axis=-1
        )
        vec_map = jnp.stack([vec_map] * batch_size, axis=0)

        # Normalize to relative bias logits.
        relative_bias = embedding_models_healpix.RelativeBias(
            n_heads=n_heads, freq_features=64
        )
        params_bias = relative_bias.init(rng, vec_map)
        relative_bias_logits = relative_bias.apply(params_bias, vec_map)

        # Create test inputs
        x = jax.random.normal(rng, (batch_size, n_pixels, emb_features))

        # Initialize HEALPixAttention
        attention = embedding_models_healpix.HEALPixAttention(
            emb_features=emb_features, n_heads=n_heads, dropout_rate=dropout_rate,
        )
        params = attention.init(
            {'params': rng, 'dropout': rng}, x, relative_bias_logits, train=True
        )
        apply_func = self.variant(attention.apply, static_argnames=['train'])

        # Test output shape
        output = apply_func(
            params, x, relative_bias_logits, train=True, rngs={'dropout': rng}
        )
        expected_shape = (batch_size, n_pixels, emb_features)
        self.assertTupleEqual(output.shape, expected_shape)

        # Test inference mode
        output_inference = apply_func(
            params, x, relative_bias_logits, train=False
        )
        self.assertTupleEqual(output_inference.shape, expected_shape)


class HEALPixAttentionBlockTests(chex.TestCase):
    """Run tests on HEALPixAttentionBlock class."""

    @chex.all_variants
    def test_call(self):
        """Test that HEALPixAttentionBlock returns correct shapes."""
        rng = jax.random.PRNGKey(3)
        emb_features = 16
        n_heads = 4
        time_emb_features = 32
        dropout_rate = 0.1
        nside = 8
        n_pixels = 8 ** 2
        batch_size = 2

        # Create vector map
        vec_map = jnp.stack(
            hp.pix2vec(nside, jnp.arange(n_pixels), nest=True), axis=-1
        )
        vec_map = jnp.stack([vec_map] * batch_size, axis=0)

        # Normalize to relative bias logits.
        relative_bias = embedding_models_healpix.RelativeBias(
            n_heads=n_heads, freq_features=64
        )
        params_bias = relative_bias.init(rng, vec_map)
        relative_bias_logits = relative_bias.apply(params_bias, vec_map)

        # Create test inputs
        x = jax.random.normal(rng, (batch_size, n_pixels, emb_features))
        t = jax.random.normal(rng, (batch_size, time_emb_features))

        # Initialize HEALPixAttentionBlock
        attention_block = embedding_models_healpix.HEALPixAttentionBlock(
            emb_features=emb_features,
            n_heads=n_heads,
            time_emb_features=time_emb_features,
            dropout_rate=dropout_rate
        )
        params = attention_block.init(
            {'params': rng, 'dropout': rng}, x, t, relative_bias_logits,
            train=True
        )
        apply_func = self.variant(
            attention_block.apply, static_argnames=['train']
        )

        # Test output shape
        output = apply_func(
            params, x, t, relative_bias_logits, train=True,
            rngs={'dropout': rng}
        )
        expected_shape = (batch_size, n_pixels, emb_features)
        self.assertTupleEqual(output.shape, expected_shape)

        # Test that it's a residual connection (output should be different from input)
        self.assertFalse(jnp.allclose(output, x))


class HEALPixTransformerTests(chex.TestCase):
    """Run tests on HEALPixTransformer class."""

    @chex.all_variants
    def test_call(self):
        """Test that HEALPixTransformer returns correct shapes."""
        rng = jax.random.PRNGKey(5)
        emb_features = 4
        n_blocks = 3
        dropout_rate_block = [0.1, 0.1, 0.1]
        heads = 4
        patch_size = 4
        time_emb_features = 16
        channels = 2
        nside = 8
        n_pixels = 8 ** 2
        batch_size = 2

        # Create vector map
        vec_map = jnp.stack(
            hp.pix2vec(nside, jnp.arange(n_pixels), nest=True), axis=-1
        )
        vec_map = jnp.stack([vec_map] * batch_size, axis=0)

        # Create test inputs
        x = jax.random.normal(rng, (batch_size, n_pixels, channels))
        t = jax.random.normal(rng, (batch_size, time_emb_features))

        # Initialize HEALPixTransformer
        transformer = embedding_models_healpix.HEALPixTransformer(
            emb_features=emb_features,
            n_blocks=n_blocks,
            dropout_rate_block=dropout_rate_block,
            heads=heads,
            patch_size=patch_size,
            time_emb_features=time_emb_features
        )
        params = transformer.init(
            {'params': rng, 'dropout': rng}, x, t, vec_map, train=True
        )
        apply_func = self.variant(
            transformer.apply, static_argnames=['train']
        )

        # Test output shape
        output = apply_func(
            params, x, t, vec_map, train=True, rngs={'dropout': rng}
        )
        expected_shape = (batch_size, n_pixels, channels)
        self.assertTupleEqual(output.shape, expected_shape)

        # Test inference mode
        output_inference = apply_func(params, x, t, vec_map, train=False)
        self.assertTupleEqual(output_inference.shape, expected_shape)


class FlatHEALPixTransformerTest(chex.TestCase):
    """Run tests for the FlatUNet."""

    @chex.all_variants
    def test_call(self):
        """Test that the FlatUNet returns the desired outputs."""
        rng = jax.random.PRNGKey(5)
        emb_features = 4
        n_blocks = 3
        dropout_rate_block = [0.1, 0.1, 0.1]
        heads = 4
        patch_size = 4
        time_emb_features = 16
        channels = 2
        nside = 8
        n_pixels = 8 ** 2
        batch_size = 2

        # Create vector map
        vec_map = jnp.stack(
            hp.pix2vec(nside, jnp.arange(n_pixels), nest=True), axis=-1
        )
        vec_map = jnp.stack([vec_map] * batch_size, axis=0)

        # Create test inputs
        x = jax.random.normal(rng, (batch_size, n_pixels, channels))
        t = jax.random.normal(rng, (batch_size, time_emb_features))

        # Flatten the input and create healpix_shape.
        x_flat = x.reshape(x.shape[:-2] + (-1,))
        healpix_shape = x.shape[-2:]

        # Initialize HEALPixTransformer
        transformer = embedding_models_healpix.FlatHEALPixTransformer(
            emb_features=emb_features,
            n_blocks=n_blocks,
            dropout_rate_block=dropout_rate_block,
            heads=heads,
            patch_size=patch_size,
            time_emb_features=time_emb_features,
            healpix_shape=healpix_shape
        )
        params = transformer.init(
            {'params': rng, 'dropout': rng}, x_flat, t, vec_map, train=True
        )
        apply_func = self.variant(
            transformer.apply, static_argnames=['train']
        )

        # Test output shape
        output = apply_func(
            params, x_flat, t, vec_map, train=True, rngs={'dropout': rng}
        )
        expected_shape = (batch_size, n_pixels * channels)
        self.assertTupleEqual(output.shape, expected_shape)


if __name__ == '__main__':
    absltest.main()
