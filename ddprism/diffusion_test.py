"Test scripts for diffusion.py"

import copy

from absl.testing import absltest
import chex
from flax import linen as nn
import jax
import jax.numpy as jnp

from ddprism import diffusion
from ddprism import embedding_models
from ddprism import linalg
from ddprism.linalg_test import _create_DPLR_instance


class SDETests(chex.TestCase):
    """Run tests on SDE models."""

    @chex.all_variants
    def test_vesde(self):
        """Test that calling the Denoiser and VESDE behaves as expected."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 3)
        features = 5

        # Initialize the SDE
        sde = diffusion.VESDE()
        params = sde.init(
            rng, jnp.ones((1, features)), jnp.ones((1, features)),
            jnp.ones((features,))
        )
        apply_func = self.variant(sde.apply, static_argnames=['method'])

        # Test that the output has the desired shape.
        batch_size = 16
        x_draws = jax.random.normal(rng_keys[0], (batch_size, features))
        z_draws = jax.random.normal(rng_keys[1], (batch_size, features))
        t_draws = jax.random.uniform(rng_keys[2], (batch_size,))

        x_t_draws = apply_func(params, x_draws, z_draws, t_draws)
        self.assertTupleEqual(x_t_draws.shape, (batch_size, features))

        # Check that the shape of noise is correct.
        sigma_t = apply_func(params, t_draws, method='sigma')
        self.assertTupleEqual(sigma_t.shape, (batch_size,))

        # Check that in the limit t = 0 and t = 1, we get the right noise values
        # t = 0 case
        t_draws *= 0.0
        expect_sigma = sde.apply(params, t_draws, method=sde.sigma)
        assert jnp.allclose(jnp.ones((batch_size,)) * sde.a, expect_sigma)

        # t = 1 case
        t_draws += 1.0
        expect_sigma = sde.apply(params, t_draws, method=sde.sigma)
        assert jnp.allclose(jnp.ones((batch_size,))*sde.b, expect_sigma)


class DiffusionTests(chex.TestCase):
    """Runs tests of various diffusion functions."""

    @chex.all_variants
    def test_denoiser(self):
        """Test that calling the Denoiser and VESDE behaves as expected."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 2)

        features = 5
        # Test that the output has the desired shape.
        batch_size = 16
        x_draws = jax.random.normal(rng_keys[0], (batch_size, features))
        t_draws = jax.random.uniform(rng_keys[1], (batch_size,))

        ## Test Denoiser
        hid_features = (32, 32)
        activation = nn.gelu
        normalize = True

        # Initialize the SDE
        sde = diffusion.VESDE()

        # Initialize TimeMLP.
        time_mlp = embedding_models.TimeMLP(
            features, hid_features, activation, normalize
        )

        # Initialize Denoiser.
        emb_features = 64
        denoiser = diffusion.Denoiser(
            sde, time_mlp, emb_features = emb_features
        )
        params_denoiser = denoiser.init(
            rng, jnp.ones((1, features)), jnp.ones((1,))
        )

        # Check that the output of the denoiser is of the correct shape.
        apply_func = self.variant(denoiser.apply, static_argnames='method')
        expect_x = apply_func(params_denoiser, x_draws, t_draws)
        self.assertTupleEqual(expect_x.shape, (batch_size, features))

        # Check that the remaining sde call works.
        rng_keys = jax.random.split(rng, 3)
        x_draws = jax.random.normal(rng_keys[0], (batch_size, features))
        z_draws = jax.random.normal(rng_keys[1], (batch_size, features))
        t_draws = jax.random.normal(rng_keys[2], (batch_size,))
        x_t_draws = apply_func(
            params_denoiser, x_draws, z_draws, t_draws, method='sde_x_t'
        )
        self.assertTupleEqual(x_t_draws.shape, (batch_size, features))
        sigma_t = apply_func(
            params_denoiser, t_draws, method='sde_sigma'
        )
        sigma_t_exp = sde.apply({}, t_draws, method='sigma')
        self.assertTrue(jnp.allclose(sigma_t, sigma_t_exp))

    @chex.all_variants
    def test_gaussian_denoiser(self):
        """Test that calling the Denoiser and VESDE behaves as expected."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 2)

        features = 5
        # Test that the output has the desired shape.
        batch_size = 16
        x_draws = jax.random.normal(rng_keys[0], (batch_size, features))
        t_draws = jax.random.uniform(rng_keys[1], (batch_size,))

        # Initialize the SDE
        sde = diffusion.VESDE()

        # Initialize Denoiser.
        denoiser = diffusion.GaussianDenoiser(sde)
        params_denoiser = denoiser.init(
            rng, jnp.ones((1, features)), jnp.ones((1,))
        )

        # Check that parameter initializations.
        self.assertTrue(
            jnp.allclose(params_denoiser['params']['mu_x'], jnp.ones(features))
        )
        self.assertTrue(
            jnp.allclose(params_denoiser['params']['cov_x'], jnp.eye(features))
        )

        # Check that the output of the denoiser is of the correct shape.
        apply_func = self.variant(denoiser.apply, static_argnames='method')
        expect_x = apply_func(params_denoiser, x_draws, t_draws)
        self.assertTupleEqual(expect_x.shape, (batch_size, features))

    @chex.all_variants
    def test_gaussian_denoiser_dplr(self):
        """Test that calling the Denoiser and VESDE behaves as expected."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 2)

        features = 5
        # Test that the output has the desired shape.
        batch_size = 16
        x_draws = jax.random.normal(rng_keys[0], (batch_size, features))
        t_draws = jax.random.uniform(rng_keys[1], (batch_size,))

        # Initialize the SDE
        sde = diffusion.VESDE()

        # Initialize Denoiser.
        denoiser = diffusion.GaussianDenoiser(sde)
        denoiser_dplr = diffusion.GaussianDenoiserDPLR(sde)
        params_denoiser = denoiser.init(
            rng, jnp.ones((1, features)), jnp.ones((1,))
        )
        params_denoiser_dplr = denoiser_dplr.init(
            rng, jnp.ones((1, features)), jnp.ones((1,))
        )

        params_denoiser_dplr['params']['cov_x'] = (
            _create_DPLR_instance(features)
        )
        params_denoiser['params']['cov_x'] = (
            _create_DPLR_instance(features).full_matrix()
        )

        # Check that parameter initializations.
        self.assertTrue(
            jnp.allclose(
                params_denoiser_dplr['params']['mu_x'], jnp.ones(features)
            )
        )
        self.assertTrue(
            jnp.allclose(
                params_denoiser_dplr['params']['cov_x'].diagonal,
                jnp.ones(features)
            )
        )

        # Check that the output of the denoiser is not changed by the use of
        # DPLR.
        apply_func = self.variant(denoiser_dplr.apply, static_argnames='method')
        expect_x = apply_func(params_denoiser_dplr, x_draws, t_draws)
        comp_x = denoiser.apply(params_denoiser, x_draws, t_draws)

        self.assertTrue(jnp.allclose(expect_x, comp_x, atol=1e-3))


class PosteriorDenoiserJointTests(chex.TestCase):
    """Runs tests of various posterior diffusion functions."""

    @chex.all_variants
    def test_posterior_denoiser(self):
        """Test that the posterior denoiser returns the expected outputs."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 2)

        features = 5
        # Test that the output has the desired shape.
        batch_size = 16
        x_draws = jax.random.normal(rng_keys[0], (batch_size, features))
        t_draws = jax.random.uniform(rng_keys[1], (batch_size,))

        hid_features = (32, 32)
        activation = nn.gelu
        normalize = True
        # Initialize the SDE
        sde = diffusion.VESDE()
        # Initialize TimeMLP.
        time_mlp = embedding_models.TimeMLP(
            features, hid_features, activation, normalize
        )
        # Initialize Denoiser.
        emb_features = 64
        y_features = 2
        denoiser_models = [
            diffusion.Denoiser(sde, time_mlp, emb_features = emb_features)
        ]

        # Initialize the posterior denoiser.
        denoiser = diffusion.PosteriorDenoiserJoint(
            denoiser_models, y_features
        )
        params_denoiser = denoiser.init(
            rng, jnp.ones((1, features)), jnp.ones((1,))
        )

        # Check the shapes initialized correctly.
        self.assertTupleEqual(
            params_denoiser['variables']['A'].shape,
            (1, 1, y_features, features)
        )
        self.assertTupleEqual(
            params_denoiser['variables']['y'].shape, (1, y_features)
        )
        self.assertTupleEqual(
            params_denoiser['variables']['cov_y'].shape,
            (1, y_features, y_features)
        )

        # Check that the output of the denoiser is of the correct shape.
        apply_func = self.variant(denoiser.apply, static_argnames='method')
        expect_x = apply_func(params_denoiser, x_draws, t_draws)
        self.assertTupleEqual(expect_x.shape, (batch_size, features))

        # Check the sde_sigma outputs.
        sigma_t = apply_func(params_denoiser, t_draws, method='sde_sigma')
        self.assertTupleEqual(sigma_t.shape, (batch_size, 1))

    @chex.all_variants
    def test_posterior_denoiser_dplr(self):
        """Test that the posterior denoiser returns the expected outputs."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 2)

        features = 5
        # Test that the output has the desired shape.
        batch_size = 16
        x_draws = jax.random.normal(rng_keys[0], (batch_size, features))
        t_draws = jax.random.uniform(rng_keys[1], (batch_size,))
        y_features = 2

        # Initialize the Denoiser
        sde = diffusion.VESDE()
        denoiser_models = [diffusion.GaussianDenoiser(sde)]
        denoiser_models_dplr = [diffusion.GaussianDenoiserDPLR(sde)]

        # Initialize the posterior denoiser.
        denoiser_dplr = diffusion.PosteriorDenoiserJoint(
            denoiser_models_dplr, y_features, use_dplr=True, maxiter=1,
            rtol=1e-8
        )
        denoiser = diffusion.PosteriorDenoiserJoint(
            denoiser_models, y_features, use_dplr=False, maxiter=1,
            rtol=1e-8
        )
        params_denoiser = denoiser.init(
            rng, jnp.ones((1, features)), jnp.ones((1,))
        )
        params_denoiser_dplr = denoiser_dplr.init(
            rng, jnp.ones((1, features)), jnp.ones((1,))
        )

        # Check the shapes initialized correctly.
        self.assertTupleEqual(
            params_denoiser_dplr['variables']['A'].shape,
            (1, 1, y_features, features)
        )
        self.assertTupleEqual(
            params_denoiser_dplr['variables']['y'].shape, (1, y_features)
        )
        self.assertTupleEqual(
            params_denoiser_dplr['variables']['cov_y'].diagonal.shape,
            (1, y_features)
        )

        # Set the covariance matrices.
        params_denoiser['variables']['cov_y'] = (
            jnp.arange(1, batch_size + 1)[:, None, None] *
            jnp.eye(y_features)[None]
        )
        params_denoiser_dplr['variables']['cov_y'] = linalg.DPLR(
            diagonal = (
                jnp.arange(1, batch_size + 1)[:, None] *
                jnp.ones(y_features)[None]
            )
        )

        # Check that the posterior denoiser matches with and without dplr.
        apply_func = self.variant(denoiser_dplr.apply, static_argnames='method')
        expect_x = apply_func(params_denoiser_dplr, x_draws, t_draws)
        comp_x = denoiser.apply(params_denoiser, x_draws, t_draws)
        self.assertTrue(jnp.allclose(comp_x, expect_x, rtol=1e-3))

    @chex.all_variants
    def test_posterior_denoiser_joint(self):
        """Test that the posterior denoiser returns the expected outputs."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 2)

        features = 5
        n_models = 3
        # Test that the output has the desired shape.
        batch_size = 16
        x_draws = jax.random.normal(
            rng_keys[0], (batch_size, n_models * features)
        )
        t_draws = jax.random.uniform(rng_keys[1], (batch_size,))

        # Initialize Denoiser.
        emb_features = 64
        y_features = 2
        hid_features = (32, 32)
        activation = nn.gelu
        normalize = True
        sde_models = [
            diffusion.VESDE(a=(i+1)*1e-3, b=(i+1)*1e2) for i in range(n_models)
        ]
        denoiser_models =[
            diffusion.Denoiser(
                sde_models[i],
                embedding_models.TimeMLP(
                    features, hid_features, activation, normalize
                ),
                emb_features = emb_features * (i+1)
            ) for i in range(n_models-1)
        ]
        # Add a Gaussian denoiser.
        denoiser_models.append(
            diffusion.GaussianDenoiser(sde_models[-1])
        )

        denoiser = diffusion.PosteriorDenoiserJoint(
            denoiser_models, y_features
        )
        params_denoiser = denoiser.init(
            rng, jnp.ones((1, n_models * features)), jnp.ones((1,))
        )

        # Make sure each denoiser model was initialized.
        self.assertEqual(len(params_denoiser['params'].keys()), n_models)

        # Check the shapes initialized correctly.
        self.assertTupleEqual(
            params_denoiser['variables']['A'].shape,
            (1, n_models, y_features, features)
        )
        self.assertTupleEqual(
            params_denoiser['variables']['y'].shape, (1, y_features)
        )
        self.assertTupleEqual(
            params_denoiser['variables']['cov_y'].shape,
            (1, y_features, y_features)
        )

        # Check that the output of the denoiser is of the correct shape.
        apply_func = self.variant(
            denoiser.apply, static_argnames=['method', 'index']
        )
        expect_x = apply_func(params_denoiser, x_draws, t_draws)
        self.assertTupleEqual(expect_x.shape, (batch_size, n_models * features))

        # When only one prior model is being used, the feature dimension should
        # change.
        x_single = jax.random.normal(
            rng_keys[0], (batch_size, features)
        )
        expect_x = apply_func(params_denoiser, x_single, t_draws, index=1)
        self.assertTupleEqual(expect_x.shape, (batch_size, features))

        # Check the sde_x_t works.
        z_draws = jax.random.normal(
            rng_keys[0], (batch_size, n_models* features)
        )
        x_t_draws = apply_func(
            params_denoiser, x_draws, z_draws, t_draws, method='sde_x_t'
        )
        self.assertTupleEqual(
            x_t_draws.shape, (batch_size, n_models * features)
        )

        # Check that the sde_sigma gives the correct outputs.
        sigma_t_all = apply_func(params_denoiser, t_draws, method='sde_sigma')
        self.assertTupleEqual(sigma_t_all.shape, (batch_size, n_models))

        # Check that each sde is being used correctly.
        for i, sigma_t in enumerate(jnp.moveaxis(sigma_t_all, -1, 0)):
            sigma_t_exp = sde_models[i].apply({}, t_draws, method='sigma')
            self.assertTrue(jnp.allclose(sigma_t, sigma_t_exp))


class PosteriorDenoiserJointDiagonalTests(chex.TestCase):
    """Runs tests of various posterior diffusion functions."""

    @chex.all_variants
    def test_posterior_denoiser_joint(self):
        """Test that the posterior denoiser returns the expected outputs."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 2)

        features = 5
        n_models = 3
        # Test that the output has the desired shape.
        batch_size = 16
        x_draws = jax.random.normal(
            rng_keys[0], (batch_size, n_models * features)
        )
        t_draws = jax.random.uniform(rng_keys[1], (batch_size,))

        # Initialize Denoiser.
        emb_features = 64
        y_features = features
        hid_features = (32, 32)
        activation = nn.gelu
        normalize = True
        sde_models = [
            diffusion.VESDE(a=(i+1)*1e-3, b=(i+1)*1e2) for i in range(n_models)
        ]
        denoiser_models =[
            diffusion.Denoiser(
                sde_models[i],
                embedding_models.TimeMLP(
                    features, hid_features, activation, normalize
                ),
                emb_features = emb_features * (i+1)
            ) for i in range(n_models-1)
        ]
        # Add a Gaussian denoiser.
        denoiser_models.append(
            diffusion.GaussianDenoiser(sde_models[-1])
        )

        denoiser_comp = diffusion.PosteriorDenoiserJoint(
            denoiser_models, y_features
        )
        denoiser = diffusion.PosteriorDenoiserJointDiagonal(
            denoiser_models, y_features
        )
        params_denoiser = denoiser.init(
            rng, jnp.ones((1, n_models * features)), jnp.ones((1,))
        )
        params_denoiser['variables']['cov_y'] = (
            jnp.tile(jnp.eye(5)[None], (batch_size, 1, 1))
        )
        params_denoiser['variables']['y'] = (
            jnp.tile(params_denoiser['variables']['y'], (batch_size, 1))
        )
        params_denoiser['variables']['A'] = jax.random.uniform(
            rng, shape=(
                (batch_size,) + params_denoiser['variables']['A'].shape[1:]
            )
        )
        params_comp = copy.deepcopy(params_denoiser)
        params_comp['variables']['A'] = jax.vmap(jax.vmap(jnp.diag))(
            params_denoiser['variables']['A']
        )

        # Make sure each denoiser model was initialized.
        self.assertEqual(len(params_denoiser['params'].keys()), n_models)

        # Check that the output of diagonal and full denoisers match when the
        # transformation is diagonal.
        apply_func = self.variant(
            denoiser.apply, static_argnames=['method', 'index']
        )
        apply_func_comp = self.variant(
            denoiser_comp.apply, static_argnames=['method', 'index']
        )
        expect_x = apply_func_comp(params_comp, x_draws, t_draws)
        diagonal_x = apply_func(params_denoiser, x_draws, t_draws)
        self.assertTupleEqual(
            diagonal_x.shape, (batch_size, n_models * features)
        )
        self.assertTrue(jnp.allclose(expect_x, diagonal_x))

        # When only one prior model is being used, the feature dimension should
        # change.
        x_single = jax.random.normal(
            rng_keys[0], (batch_size, features)
        )
        expect_x = apply_func_comp(params_comp, x_single, t_draws, index=1)
        diagonal_x = apply_func(params_denoiser, x_single, t_draws, index=1)
        self.assertTupleEqual(diagonal_x.shape, (batch_size, features))
        self.assertTrue(jnp.allclose(expect_x, diagonal_x, rtol=1e-3))


if __name__ == '__main__':
    absltest.main()
