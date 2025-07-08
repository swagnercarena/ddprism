"""Test scripts for pcpca_utils.py"""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp

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
        print(grads['weights'])
        print(auto_grads['weights'])
        # self.assertTrue(
        #     jnp.allclose(grads['weights'], auto_grads['weights'], rtol=1e-5)
        # )
        self.assertAlmostEqual(
            grads['log_sigma'], auto_grads['log_sigma'], places=5
        )

    @chex.all_variants
    def test_gradient_consistency_multiple_seeds(self):
        """Test gradient consistency across multiple random seeds."""
        features = 4
        latent_dim = 2
        batch_size = 8

        for seed in [0, 1, 2, 42, 100]:
            rng = jax.random.PRNGKey(seed)
            params, x_obs, y_obs, x_a_mat, y_a_mat, gamma = self._create_test_data(
                rng, batch_size, features, latent_dim
            )

            # Compute analytic gradients
            analytic_grads = pcpca_utils.loss_grad(
                params, x_obs, y_obs, x_a_mat, y_a_mat, gamma
            )

            # Compute automatic gradients
            loss_func = lambda p: pcpca_utils.loss(p, x_obs, y_obs, x_a_mat, y_a_mat, gamma)
            auto_grads = jax.grad(loss_func)(params)

            # Compare gradients
            self.assertTrue(
                jnp.allclose(analytic_grads['weights'], auto_grads['weights'], rtol=1e-5),
                f"Weights gradients don't match for seed {seed}"
            )
            self.assertTrue(
                jnp.allclose(analytic_grads['log_sigma'], auto_grads['log_sigma'], rtol=1e-5),
                f"Log sigma gradients don't match for seed {seed}"
            )

    @chex.all_variants
    def test_loss_with_different_gamma_values(self):
        """Test loss function with different gamma values."""
        rng = jax.random.PRNGKey(0)
        batch_size = 16
        features = 5
        latent_dim = 3

        params, x_obs, y_obs, x_a_mat, y_a_mat, _ = self._create_test_data(
            rng, batch_size, features, latent_dim
        )

        apply_func = self.variant(pcpca_utils.loss)

        # Test with different gamma values
        gamma_values = [0.0, 0.5, 1.0, 2.0]
        loss_values = []

        for gamma in gamma_values:
            loss_value = apply_func(params, x_obs, y_obs, x_a_mat, y_a_mat, gamma)
            loss_values.append(loss_value)
            self.assertTrue(jnp.isfinite(loss_value))

        # Check that loss values are different for different gamma values
        for i in range(len(loss_values) - 1):
            self.assertFalse(jnp.allclose(loss_values[i], loss_values[i+1]))

    @chex.all_variants
    def test_loss_grad_with_different_gamma_values(self):
        """Test loss_grad function with different gamma values."""
        rng = jax.random.PRNGKey(0)
        batch_size = 16
        features = 5
        latent_dim = 3

        params, x_obs, y_obs, x_a_mat, y_a_mat, _ = self._create_test_data(
            rng, batch_size, features, latent_dim
        )

        apply_func = self.variant(pcpca_utils.loss_grad)

        # Test with different gamma values
        gamma_values = [0.0, 0.5, 1.0, 2.0]
        grad_values = []

        for gamma in gamma_values:
            grads = apply_func(params, x_obs, y_obs, x_a_mat, y_a_mat, gamma)
            grad_values.append(grads)
            self.assertTrue(jnp.all(jnp.isfinite(grads['weights'])))
            self.assertTrue(jnp.isfinite(grads['log_sigma']))

        # Check that gradients are different for different gamma values
        for i in range(len(grad_values) - 1):
            self.assertFalse(jnp.allclose(grad_values[i]['weights'], grad_values[i+1]['weights']))
            self.assertFalse(jnp.allclose(grad_values[i]['log_sigma'], grad_values[i+1]['log_sigma']))

    @chex.all_variants
    def test_compute_aux_matrix_vectorized(self):
        """Test compute_aux_matrix with vectorized inputs."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 4)

        batch_size = 10
        features = 4
        latent_dim = 2

        weights = jax.random.normal(rng_keys[0], (features, latent_dim))
        a_mats = jax.random.normal(rng_keys[1], (batch_size, features, features))
        sigmas = jax.random.uniform(rng_keys[2], (batch_size,)) + 0.1

        # Use vmap to vectorize
        vectorized_func = self.variant(
            jax.vmap(pcpca_utils.compute_aux_matrix, in_axes=(None, 0, 0))
        )
        results = vectorized_func(weights, a_mats, sigmas)

        # Check shape
        self.assertTupleEqual(results.shape, (batch_size, features, features))

        # Check that all results are positive definite
        for i in range(batch_size):
            eigenvals = jnp.linalg.eigvals(results[i])
            self.assertTrue(jnp.all(eigenvals > 0))

    @chex.all_variants
    def test_loss_with_edge_cases(self):
        """Test loss function with edge cases."""
        rng = jax.random.PRNGKey(0)
        batch_size = 4
        features = 3
        latent_dim = 2

        params, x_obs, y_obs, x_a_mat, y_a_mat, gamma = self._create_test_data(
            rng, batch_size, features, latent_dim
        )

        apply_func = self.variant(pcpca_utils.loss)

        # Test with gamma = 0 (no contrastive term)
        loss_no_contrast = apply_func(params, x_obs, y_obs, x_a_mat, y_a_mat, 0.0)
        self.assertTrue(jnp.isfinite(loss_no_contrast))

        # Test with very small sigma (but not zero to avoid numerical issues)
        params_small_sigma = params.copy()
        params_small_sigma['log_sigma'] = jnp.array(-10.0)  # exp(-10) is very small
        loss_small_sigma = apply_func(
            params_small_sigma, x_obs, y_obs, x_a_mat, y_a_mat, gamma
        )
        self.assertTrue(jnp.isfinite(loss_small_sigma))

    @chex.all_variants
    def test_numerical_stability(self):
        """Test numerical stability of functions."""
        rng = jax.random.PRNGKey(0)
        batch_size = 4
        features = 3
        latent_dim = 2

        params, x_obs, y_obs, x_a_mat, y_a_mat, gamma = self._create_test_data(
            rng, batch_size, features, latent_dim
        )

        # Test with different scales
        scales = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

        for scale in scales:
            scaled_params = {
                'weights': params['weights'] * scale,
                'log_sigma': params['log_sigma']
            }

            # Test loss
            loss_value = pcpca_utils.loss(
                scaled_params, x_obs, y_obs, x_a_mat, y_a_mat, gamma
            )
            self.assertTrue(jnp.isfinite(loss_value), f"Loss not finite for scale {scale}")

            # Test gradients
            grads = pcpca_utils.loss_grad(
                scaled_params, x_obs, y_obs, x_a_mat, y_a_mat, gamma
            )
            self.assertTrue(
                jnp.all(jnp.isfinite(grads['weights'])),
                f"Weights grad not finite for scale {scale}"
            )
            self.assertTrue(
                jnp.isfinite(grads['log_sigma']),
                f"Log sigma grad not finite for scale {scale}"
            )


if __name__ == '__main__':
    absltest.main()