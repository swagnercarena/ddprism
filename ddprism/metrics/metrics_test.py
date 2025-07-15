
"Test scripts for metrics.py"
from absl.testing import absltest

import jax
import jax.numpy as jnp
import chex

from ddprism.metrics import metrics


class UtilsTests(chex.TestCase):
    """Run tests on utils functions."""

    def test_sinkhorn_divergence(self):
        """Test sinkhorn divergence calculation. POT doesn't support jit.
        """
        # Draw from two different normal distirbutions.
        n_samps = 1000
        rng_u, rng_v = jax.random.split(jax.random.PRNGKey(4))
        u = jax.random.normal(rng_u, shape=(n_samps, 2))
        v = jax.random.normal(rng_v, shape=(n_samps, 2)) * 0.5 + 0.1

        self.assertGreater(metrics.sinkhorn_divergence(u, v), 1e-6)

        # Test that approaching the correct distribution improves the sinkhorn
        # divergence.
        v_new = jax.random.normal(rng_v, shape=(n_samps, 2))
        self.assertGreater(
            metrics.sinkhorn_divergence(u, v),
            metrics.sinkhorn_divergence(u, v_new)
        )

    def test_psnr(self):
        """Test PSNR calculation."""
        u = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        v = jnp.array([[0.0, 3.0], [2.0, 5.0]])
        max_spread = 10.0
        self.assertAlmostEqual(
            metrics.psnr(u, v, max_spread=max_spread), 20.0, places=4
        )


if __name__ == '__main__':
    absltest.main()
