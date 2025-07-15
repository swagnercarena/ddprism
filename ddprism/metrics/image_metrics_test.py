"Test scripts for metrics.py"
import tempfile

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

from ddprism.metrics import image_metrics as metrics


class CNNTests(chex.TestCase):
    """Run tests on CNN functions."""

    @chex.all_variants
    def test_apply(self):
        """Test the cnn functions."""
        rng = jax.random.PRNGKey(0)
        hidden_channels = (4, 4, 4)
        kernel_size = (3, 3)
        out_features = 2
        emb_features = 8
        image_size = 32

        # Initialize our cnn and the apply function
        cnn = metrics.CNN(
            hidden_channels, kernel_size, out_features, emb_features
        )
        params = cnn.init(rng, jnp.ones((1, image_size, image_size, 1)))
        apply_fn = self.variant(cnn.apply, static_argnames=['method'])

        batch_size = 32
        x_draws = jax.random.uniform(
            rng, (batch_size, image_size, image_size, 1)
        )
        out = apply_fn(params, x_draws)
        self.assertTupleEqual(out.shape, (batch_size, out_features))
        out = apply_fn(params, x_draws, method='embed')
        self.assertTupleEqual(out.shape, (batch_size, emb_features))

    def test_get_model(self):
        """Test the ability to get a trained classifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and load the classifier.
            model, params = metrics.get_model(
                tmpdir, emb_features=4, hidden_channels=(4,)
            )
            model_save, params_save = metrics.get_model(tmpdir)

            self.assertTrue(
                jax.tree_util.tree_all(
                    jax.tree_util.tree_map(
                        lambda x, y: jnp.array_equal(x, y), params, params_save
                    )
                )
            )
            self.assertEqual(model.emb_features, model_save.emb_features)
            self.assertTupleEqual(
                model.hidden_channels, tuple(model_save.hidden_channels)
            )


class MetricsTests(chex.TestCase):
    """Run tests on metric functions."""

    def test_fcd_mnist(self):
        """Test the fcd_mnist function."""
        rng = jax.random.PRNGKey(2)
        image_size = 32
        cnn = metrics.CNN()
        params = cnn.init(rng, jnp.ones((1, image_size, image_size, 1)))

        # Generate our samples.
        batch_size = 32
        rng_one, rng_two, rng_three = jax.random.split(rng, 3)
        draws_one = jax.random.uniform(
            rng_one, (batch_size, image_size, image_size, 1)
        )
        draws_two = jax.random.uniform(
            rng_two, (batch_size, image_size, image_size, 1)
        )
        draws_three = jax.random.normal(
            rng_three, (batch_size, image_size, image_size, 1)
        )

        # Test the fcd.
        fcd_small = metrics.fcd_mnist(cnn, params, draws_one, draws_two, 15)
        fcd_large = metrics.fcd_mnist(cnn, params, draws_one, draws_three, 15)
        self.assertGreater(fcd_large, fcd_small)

    def test_inception_mnist(self):
        """Test the inception_score_mnist function."""
        rng = jax.random.PRNGKey(2)
        image_size = 32
        cnn = metrics.CNN()
        params = cnn.init(rng, jnp.ones((1, image_size, image_size, 1)))

        # Generate our samples.
        batch_size = 32
        draws_one = jax.random.uniform(
            rng, (batch_size, image_size, image_size, 1)
        )

        # Test the score.
        incep = metrics.inception_score_mnist(cnn, params, draws_one, 15)
        self.assertGreater(incep, 0.0)

    def test_pq_mass(self):
        """Test the pq_mass function."""
        # Generate our samples.
        rng = jax.random.PRNGKey(2)
        image_size = 32
        batch_size = 128
        rng_one, rng_two, rng_three = jax.random.split(rng, 3)
        draws_one = jax.random.uniform(
            rng_one, (batch_size, image_size, image_size, 1)
        )
        draws_two = jax.random.uniform(
            rng_two, (batch_size, image_size, image_size, 1)
        )
        draws_three = jax.random.normal(
            rng_three, (batch_size, image_size, image_size, 1)
        )

        # Test the pq_mass result.
        pq_small = metrics.pq_mass(draws_one, draws_two, num_refs=32)
        pq_large = metrics.pq_mass(draws_one, draws_three, num_refs=32)
        self.assertLess(pq_large, pq_small)

        # Test the result with the CNN embedding.
        cnn = metrics.CNN()
        params = cnn.init(rng, jnp.ones((1, image_size, image_size, 1)))
        embed_one = metrics.map_model_apply(
            cnn, params, draws_one, method='embed'
        )
        embed_two = metrics.map_model_apply(
            cnn, params, draws_two, method='embed'
        )
        embed_three = metrics.map_model_apply(
            cnn, params, draws_three, method='embed'
        )
        pq_small = metrics.pq_mass(embed_one, embed_two, num_refs=32)
        pq_large = metrics.pq_mass(embed_one, embed_three, num_refs=32)
        self.assertLess(pq_large, pq_small)


if __name__ == '__main__':
    absltest.main()
