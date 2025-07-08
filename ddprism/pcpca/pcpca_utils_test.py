"""Test scripts for pcpca_utils.py"""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

# Enable float64 precision
jax.config.update("jax_enable_x64", True)

from ddprism.pcpca import pcpca_utils


def _create_test_data(rng, batch_size=16, features=5, latent_dim=3):
    """Create test data for PCPCA functions."""
    rng_keys = jax.random.split(rng, 6)

    # Create parameters
    weights = jax.random.normal(rng_keys[0], (features, latent_dim))
    log_sigma = jax.random.normal(rng_keys[1], ())
    params = {'weights': weights, 'log_sigma': log_sigma}

    # Create observations
    x_obs = jax.random.normal(rng_keys[2], (batch_size, features))
    y_obs = jax.random.normal(rng_keys[3], (batch_size, features))

    # Create transformation matrices
    x_a_mat = jax.random.normal(
        rng_keys[4], (batch_size, features, features)
    )
    y_a_mat = jax.random.normal(
        rng_keys[5], (batch_size, features, features)
    )

    # Gamma parameter
    gamma = 0.1

    return params, x_obs, y_obs, x_a_mat, y_a_mat, gamma

class PCPCAUtilsTests(chex.TestCase):
    """Run tests on PCPCA utility functions."""

    def test_compute_aux_matrix(self):
        """Test that compute_aux_matrix returns the correct shape."""
        features = 5
        latent_dim = 3
        batch_size = 16

        params, _, _, x_a_mat, _, _ = _create_test_data(
            jax.random.PRNGKey(0), batch_size=batch_size, features=features,
            latent_dim=latent_dim,
        )

        apply_func = jax.vmap(
            pcpca_utils.compute_aux_matrix, in_axes=(None, 0, None)
        )
        result = apply_func(
            params['weights'], x_a_mat, jnp.exp(params['log_sigma'])
        )

        # Check shape
        self.assertTupleEqual(result.shape, (batch_size, features, features))

        # Check that the matrix is symetric.
        self.assertTrue(
            jnp.allclose(result, jnp.transpose(result, axes=(0, 2, 1)))
        )

        # Check that the matrix is invertible.
        self.assertTrue(jnp.all(jnp.linalg.det(result) > 0))


    @chex.all_variants
    def test_loss_function(self):
        """Test that the loss function returns the correct shape and finite values."""
        rng = jax.random.PRNGKey(0)
        batch_size = 16
        features = 5
        latent_dim = 3

        params, x_obs, y_obs, x_a_mat, y_a_mat, gamma = _create_test_data(
            rng, batch_size, features, latent_dim
        )

        apply_func = self.variant(pcpca_utils.loss)
        loss_value = apply_func(params, x_obs, y_obs, x_a_mat, y_a_mat, gamma)

        print(loss_value)

        # Check shape (should be scalar)
        self.assertTupleEqual(loss_value.shape, ())

        # Check that loss is finite
        self.assertTrue(jnp.isfinite(loss_value))

    @chex.all_variants
    def test_loss_grad_function(self):
        """Test that the loss_grad function returns the correct shapes."""
        rng = jax.random.PRNGKey(0)
        batch_size = 16
        features = 5
        latent_dim = 3

        params, x_obs, y_obs, x_a_mat, y_a_mat, gamma = _create_test_data(
            rng, batch_size, features, latent_dim
        )

        apply_func = self.variant(pcpca_utils.loss_grad)
        grads = apply_func(params, x_obs, y_obs, x_a_mat, y_a_mat, gamma)

        # Check that gradients have the same structure as params
        self.assertSetEqual(set(grads.keys()), set(params.keys()))

        # Check shapes
        self.assertTupleEqual(grads['weights'].shape, params['weights'].shape)
        self.assertTupleEqual(
            grads['log_sigma'].shape, params['log_sigma'].shape
        )

        # Compute automatic gradients
        loss_func = lambda p: pcpca_utils.loss(
            p, x_obs, y_obs, x_a_mat, y_a_mat, gamma
        )
        auto_grads = jax.grad(loss_func)(params)

        # Compare gradients
        self.assertTrue(
            jnp.allclose(grads['weights'], auto_grads['weights'], rtol=1e-5)
        )
        self.assertAlmostEqual(
            grads['log_sigma'], auto_grads['log_sigma'], places=5
        )

if __name__ == '__main__':
    absltest.main()