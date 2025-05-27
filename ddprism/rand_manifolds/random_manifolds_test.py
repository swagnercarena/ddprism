"Test scripts for random_manifolds.py"
from absl.testing import absltest

import chex
import jax
import jax.numpy as jnp

from ddprism.rand_manifolds import random_manifolds


class DataGenerationTests(chex.TestCase):
    """Runs tests of various cosmology functions."""

    @chex.all_variants
    def test_generate_x(self):
        """"Test that x samples are generated correctly. Generate_x cannot be
        jit compiled.
        """

        generate_x = self.variant(
            random_manifolds.generate_x,
            static_argnames=[
                'n_samples', 'man_dim', 'feat_dim', 'phase', 'epsilon', 'alpha',
                'normalize'
            ]
        )

        key = jax.random.PRNGKey(0)
        n_samples = 100
        man_dim = 1
        feat_dim = 5
        alpha = 2.5
        epsilon = 1e-3
        normalize = False
        x_draws = generate_x(
            key, n_samples, man_dim, feat_dim, alpha, epsilon,
            normalize=normalize
        )

        # Check that the shape is correct and the values are in bound.
        self.assertTupleEqual(x_draws.shape, (n_samples, feat_dim))
        self.assertGreaterEqual(jnp.min(x_draws), -1.0)
        self.assertLessEqual(jnp.max(x_draws), 1.0)

        # Check that setting the phase changes the output.
        phase = 0.2
        new_x_draws = generate_x(
            key, n_samples, man_dim, feat_dim, alpha, epsilon, phase
        )
        self.assertFalse(jnp.array_equal(x_draws, new_x_draws))

    @chex.all_variants
    def test_generate_y(self):
        """"Test that x samples are generated correctly. Generate_x cannot be
        jit compiled.
        """

        generate_y = self.variant(random_manifolds.generate_y)

        key_x, key_a, key_y = jax.random.split(jax.random.PRNGKey(0), 3)
        key_x = jax.random.split(key_x, 3)
        key_a = jax.random.split(key_a, 3)

        batch_size = 16
        feat_dim = 3
        obs_dim = 2
        x_all = jnp.stack(
            [
                random_manifolds.generate_x(key,  batch_size, feat_dim=feat_dim)
                for key in key_x
            ],
            axis=1
        )
        A_all = jnp.stack(
            [
                random_manifolds.generate_A(key,  batch_size, obs_dim, feat_dim)
                for key in key_x
            ],
            axis=1
        )
        y_draw, cov_y = generate_y(key_y, A_all, x_all)

        # Check that the shape is correct and the values are in bound.
        self.assertTupleEqual(y_draw.shape, (batch_size, obs_dim))
        self.assertTupleEqual(cov_y.shape, (obs_dim,))


if __name__ == '__main__':
    absltest.main()
