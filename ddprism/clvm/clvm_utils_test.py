"""Test scripts for clvm_utils.py"""
from absl.testing import absltest
import chex
from flax import linen as nn
import jax
import jax.numpy as jnp

from ddprism.clvm import clvm_utils, models


def _create_test_model_params(features, latent_dim):
    """Create test model parameters for CLVMLinear."""
    return {
        's_mat': jax.random.normal(
            jax.random.PRNGKey(0), (features, latent_dim)
        ),
        'w_mat': jax.random.normal(
            jax.random.PRNGKey(1), (features, latent_dim)
        ),
        'mu_enr': jax.random.normal(
            jax.random.PRNGKey(2), (features,)
        ),
        'mu_bkg': jax.random.normal(
            jax.random.PRNGKey(2), (features,)
        )
    }


class LinearFunctionTests(chex.TestCase):
    """Run tests on linear functions."""

    @chex.all_variants
    def test_latent_posterior_from_feat_enr(self):
        """Test latent_posterior_from_feat_enr."""
        features = 5
        latent_dim = 3
        params = _create_test_model_params(features, latent_dim)

        # Create test observations
        obs = jax.random.normal(jax.random.PRNGKey(0), (features,))
        sigma_obs = 0.1

        # Test the posterior in the general case.
        latent_posterior_from_feat_enr = self.variant(
            clvm_utils.latent_posterior_from_feat_enr
        )
        mu_latent, sigma_latent = latent_posterior_from_feat_enr(
            params['s_mat'], params['w_mat'], params['mu_enr'], obs, sigma_obs
        )
        self.assertEqual(mu_latent.shape, (latent_dim * 2,))
        self.assertEqual(sigma_latent.shape, (latent_dim * 2, latent_dim * 2))

        # Test limiting cases.
        # Test case where there is no conection between observation and latents.
        mu_latent, sigma_latent = latent_posterior_from_feat_enr(
            jnp.zeros_like(params['s_mat']), jnp.zeros_like(params['w_mat']),
            params['mu_enr'], obs, sigma_obs
        )
        self.assertTrue(jnp.allclose(mu_latent, jnp.zeros(latent_dim * 2)))
        self.assertTrue(
            jnp.allclose(sigma_latent, jnp.eye(latent_dim * 2))
        )

        # Test case where only z latent is connected to observation.
        mu_latent, sigma_latent = latent_posterior_from_feat_enr(
            params['s_mat'], jnp.zeros_like(params['w_mat']), params['mu_enr'],
            obs, sigma_obs
        )
        self.assertTrue(
            jnp.allclose(mu_latent[latent_dim:], jnp.zeros(latent_dim))
        )
        self.assertTrue(
            jnp.allclose(
                sigma_latent[latent_dim:, latent_dim:],
                jnp.eye(latent_dim)
            )
        )
        self.assertTrue(
            jnp.allclose(
                sigma_latent[:latent_dim, latent_dim:],
                jnp.zeros((latent_dim, latent_dim))
            )
        )
        self.assertTrue(
            jnp.allclose(
                sigma_latent[latent_dim:, :latent_dim],
                jnp.zeros((latent_dim, latent_dim))
            )
        )
        self.assertTrue(
            jnp.allclose(
                sigma_latent[:latent_dim, :latent_dim],
                jnp.linalg.inv(
                    jnp.eye(latent_dim) +
                    1 / sigma_obs ** 2 * params['s_mat'].T @ params['s_mat']
                )
            )
        )

        # Finally test case that's simple enough for intuition.
        mu_latent, sigma_latent = latent_posterior_from_feat_enr(
            jnp.eye(1), -jnp.eye(1), jnp.zeros(1),
            jnp.ones(1), 1.0
        )
        self.assertTrue(
            jnp.allclose(mu_latent, jnp.array([1.0/3.0, -1.0/3.0]))
        )
        self.assertTrue(
            jnp.allclose(
                sigma_latent,
                jnp.array([[2.0/3.0, 1.0/3.0], [1.0/3.0, 2.0/3.0]])
            )
        )

    @chex.all_variants
    def test_latent_posterior_from_obs(self):
        """Test latent_posterior_from_obs."""
        features = 5
        latent_dim = 3
        obs_dim = 5
        params = _create_test_model_params(features, latent_dim)
        a_mat = jax.random.normal(
            jax.random.PRNGKey(3), (obs_dim, features)
        )
        obs = jax.random.normal(
            jax.random.PRNGKey(4), (obs_dim,)
        )
        sigma_obs = 0.1
        latent_posterior_from_obs_enr = self.variant(
            clvm_utils.latent_posterior_from_obs_enr
        )

        # Start by checking shape.
        mu_latent, sigma_latent = latent_posterior_from_obs_enr(
            params['s_mat'], params['w_mat'], params['mu_enr'],
            obs, sigma_obs, a_mat
        )
        self.assertEqual(mu_latent.shape, (latent_dim * 2,))
        self.assertEqual(sigma_latent.shape, (latent_dim * 2, latent_dim * 2))

        # Check simplest limiting case where a_mat is identity.
        mu_latent, sigma_latent = latent_posterior_from_obs_enr(
            params['s_mat'], params['w_mat'], params['mu_enr'],
            obs, sigma_obs, jnp.eye(obs_dim)
        )
        mu_latent_true, sigma_latent_true = (
            clvm_utils.latent_posterior_from_feat_enr(
                params['s_mat'], params['w_mat'], params['mu_enr'],
                obs, sigma_obs
            )
        )
        self.assertTrue(jnp.allclose(mu_latent, mu_latent_true))
        self.assertTrue(jnp.allclose(sigma_latent, sigma_latent_true))

        # Check case where a matrix is zero.
        mu_latent, sigma_latent = latent_posterior_from_obs_enr(
            params['s_mat'], params['w_mat'], params['mu_enr'],
            obs, sigma_obs, jnp.zeros((obs_dim, features))
        )
        self.assertTrue(jnp.allclose(mu_latent, jnp.zeros(latent_dim * 2)))
        self.assertTrue(jnp.allclose(sigma_latent, jnp.eye(latent_dim * 2)))

    @chex.all_variants
    def test_latent_posterior_from_obs_bkg(self):
        """Test latent_posterior_from_obs_bkg."""
        features = 5
        latent_dim = 3
        obs_dim = 5
        params = _create_test_model_params(features, latent_dim)
        a_mat = jax.random.normal(
            jax.random.PRNGKey(5), (obs_dim, features)
        )
        obs = jax.random.normal(
            jax.random.PRNGKey(6), (obs_dim,)
        )
        sigma_obs = 0.1
        latent_posterior_from_obs_bkg = self.variant(
            clvm_utils.latent_posterior_from_obs_bkg
        )
        mu_latent, sigma_latent = latent_posterior_from_obs_bkg(
            params['s_mat'], params['mu_bkg'], obs, sigma_obs, a_mat
        )
        self.assertEqual(mu_latent.shape, (latent_dim,))
        self.assertEqual(sigma_latent.shape, (latent_dim, latent_dim))

        # Check simplest limiting case where a_mat is identity.
        mu_latent, sigma_latent = latent_posterior_from_obs_bkg(
            params['s_mat'], params['mu_bkg'], obs, sigma_obs, jnp.eye(obs_dim)
        )
        mu_latent_true, sigma_latent_true = (
            clvm_utils.latent_posterior_from_feat_bkg(
                params['s_mat'], params['mu_bkg'], obs, sigma_obs
            )
        )
        self.assertTrue(jnp.allclose(mu_latent, mu_latent_true))
        self.assertTrue(jnp.allclose(sigma_latent, sigma_latent_true))


class CLVMClassTests(chex.TestCase):
    """Test both CLVM classes to ensure they instantiate and run without errors."""

    def setUp(self):
        """Set up test parameters."""
        self.features = 8
        self.latent_dim_z = 3
        self.latent_dim_t = 2
        self.obs_dim = 8

    @chex.all_variants
    def test_clvm_linear(self):
        """Test that CLVMLinear can be instantiated and called without errors."""
        model = clvm_utils.CLVMLinear(
            features=self.features,
            latent_dim_z=self.latent_dim_z,
            latent_dim_t=self.latent_dim_t,
            obs_dim=self.obs_dim
        )

        # Create test data
        rng = jax.random.PRNGKey(2)
        batch_size = 4
        feat_key, obs_key, a_key, init_key = jax.random.split(rng, 4)
        feat = jax.random.normal(feat_key, (batch_size, self.features))
        obs = jax.random.normal(obs_key, (batch_size, self.obs_dim))
        a_mat = jax.random.normal(
            a_key, (batch_size, self.obs_dim, self.features)
        )

        # Initialize the model
        variables = model.init(
            init_key, init_key, obs, a_mat, method='loss_enr_obs'
        )
        apply_fn = self.variant(model.apply, static_argnames=['method'])

        # Test the expected parameters are present.
        self.assertEqual(variables['variables']['log_sigma_obs'].shape, (1,))
        self.assertEqual(
            variables['params']['s_mat'].shape,
            (self.features, self.latent_dim_z)
        )
        self.assertEqual(
            variables['params']['w_mat'].shape,
            (self.features, self.latent_dim_t)
        )
        self.assertEqual(
            variables['params']['mu_signal'].shape, (self.features,)
        )
        self.assertEqual(variables['params']['mu_bkg'].shape, (self.features,))

        # Test encode methods
        mu_z, sigma_z = apply_fn(variables, feat, method='encode_bkg_feat')
        self.assertEqual(mu_z.shape, (batch_size, self.latent_dim_z))
        self.assertEqual(
            sigma_z.shape, (batch_size, self.latent_dim_z, self.latent_dim_z)
        )

        mu_zt, sigma_zt = apply_fn(variables, feat, method='encode_enr_feat')
        self.assertEqual(
            mu_zt.shape, (batch_size, self.latent_dim_z + self.latent_dim_t,)
        )
        self.assertEqual(
            sigma_zt.shape,
            (
                batch_size,
                self.latent_dim_z + self.latent_dim_t,
                self.latent_dim_z + self.latent_dim_t
            )
        )

        # Test decode methods
        z_sample = jax.random.normal(init_key, (batch_size, self.latent_dim_z,))
        t_sample = jax.random.normal(init_key, (batch_size, self.latent_dim_t,))

        feat_decoded = apply_fn(variables, z_sample, method='decode_bkg_feat')
        self.assertEqual(feat_decoded.shape, (batch_size, self.features))

        feat_decoded_enr = apply_fn(
            variables, t_sample, method='decode_signal_feat'
        )
        self.assertEqual(feat_decoded_enr.shape, (batch_size, self.features))

        a_mat = jnp.array(
            [jnp.eye(self.obs_dim) for _ in range(batch_size)]
        )
        obs_decoded = apply_fn(
            variables, z_sample, a_mat, method='decode_bkg_obs'
        )
        self.assertEqual(obs_decoded.shape, (batch_size, self.obs_dim))
        self.assertTrue(jnp.allclose(obs_decoded, feat_decoded))

        obs_decoded_enr = apply_fn(
            variables, t_sample, a_mat, method='decode_signal_obs'
        )
        self.assertEqual(obs_decoded_enr.shape, (batch_size, self.obs_dim))
        self.assertTrue(jnp.allclose(obs_decoded_enr, feat_decoded_enr))

        # Test loss functions don't crash
        loss_bkg = apply_fn(
            variables, init_key, feat, a_mat, method='loss_bkg_obs'
        )
        self.assertEqual(loss_bkg.shape, ())

        loss_enr = apply_fn(
            variables, init_key, feat, a_mat, method='loss_enr_obs'
        )
        self.assertEqual(loss_enr.shape, ())

    @chex.all_variants
    def test_clvm_vae(self):
        """Test that CLVMVAE can be instantiated and called without errors."""

        model = clvm_utils.CLVMVAE(
            features=self.features,
            latent_dim_z=self.latent_dim_z,
            latent_dim_t=self.latent_dim_t,
            obs_dim=self.obs_dim,
            signal_decoder=models.DecoderMLP(features=self.features),
            bkg_decoder=models.DecoderMLP(features=self.features),
            signal_encoder=models.EncoderMLP(latent_features=self.latent_dim_t),
            bkg_encoder=models.EncoderMLP(latent_features=self.latent_dim_z)
        )

        # Create test data
        rng = jax.random.PRNGKey(2)
        batch_size = 4
        feat_key, obs_key, a_key, init_key = jax.random.split(rng, 4)
        obs = jax.random.normal(obs_key, (batch_size, self.obs_dim))
        a_mat = jax.random.normal(
            a_key, (batch_size, self.obs_dim, self.features)
        )

        # Initialize the model
        variables = model.init(
            init_key, init_key, obs, a_mat, method='loss_enr_obs'
        )
        apply_fn = self.variant(model.apply, static_argnames=['method'])

        # Test the expected parameters are present.
        self.assertEqual(variables['variables']['log_sigma_obs'].shape, (1,))

        # Test encode methods
        mu_z, sigma_z = apply_fn(variables, obs, a_mat, method='encode_bkg_obs')
        self.assertEqual(mu_z.shape, (batch_size, self.latent_dim_z))
        self.assertEqual(sigma_z.shape, (batch_size, self.latent_dim_z))

        mu_zt, sigma_zt = apply_fn(
            variables, obs, a_mat, method='encode_enr_obs'
        )
        self.assertEqual(
            mu_zt.shape, (batch_size, self.latent_dim_z + self.latent_dim_t,)
        )
        self.assertEqual(
            sigma_zt.shape,
            (batch_size, self.latent_dim_z + self.latent_dim_t)
        )

        # Test decode methods
        z_sample = jax.random.normal(init_key, (batch_size, self.latent_dim_z,))
        t_sample = jax.random.normal(init_key, (batch_size, self.latent_dim_t,))

        feat_decoded = apply_fn(variables, z_sample, method='decode_bkg_feat')
        self.assertEqual(feat_decoded.shape, (batch_size, self.features))

        feat_decoded_enr = apply_fn(
            variables, t_sample, method='decode_signal_feat'
        )
        self.assertEqual(feat_decoded_enr.shape, (batch_size, self.features))

        a_mat = jnp.array(
            [jnp.eye(self.obs_dim) for _ in range(batch_size)]
        )
        obs_decoded = apply_fn(
            variables, z_sample, a_mat, method='decode_bkg_obs'
        )
        self.assertEqual(obs_decoded.shape, (batch_size, self.obs_dim))
        self.assertTrue(jnp.allclose(obs_decoded, feat_decoded))

        obs_decoded_enr = apply_fn(
            variables, t_sample, a_mat, method='decode_signal_obs'
        )
        self.assertEqual(obs_decoded_enr.shape, (batch_size, self.obs_dim))
        self.assertTrue(jnp.allclose(obs_decoded_enr, feat_decoded_enr))

        # Test loss functions don't crash
        loss_bkg = apply_fn(
            variables, init_key, obs, a_mat, method='loss_bkg_obs'
        )
        self.assertEqual(loss_bkg.shape, ())

        loss_enr = apply_fn(
            variables, init_key, obs, a_mat, method='loss_enr_obs'
        )
        self.assertEqual(loss_enr.shape, ())



if __name__ == '__main__':
    absltest.main()