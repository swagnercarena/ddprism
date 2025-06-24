"Test scripts for utils.py"
from absl.testing import absltest

import jax
import jax.numpy as jnp
import chex
from flax.training import train_state
import optax

from ddprism import embedding_models
from ddprism import diffusion
from ddprism import utils


def _create_state_diffusion(rng, features, dif):
    """Create a state for testing with simple diffusion model."""
    # Create the train state components.
    sde = diffusion.VESDE()
    time_mlp = embedding_models.TimeMLP(features=features,)

    if dif == 'prior':
        denoiser = diffusion.Denoiser(sde, time_mlp)
        params = denoiser.init(
            rng, jnp.ones((1, features)), jnp.ones((1, ))
        )
    else:
        # Initialize the posterior denoiser.
        diffusion_models = [
            diffusion.Denoiser(sde, time_mlp),
            diffusion.GaussianDenoiser(diffusion.VESDE())
        ]
        denoiser = diffusion.PosteriorDenoiserJoint(
            diffusion_models, features
        )
        params = denoiser.init(
            rng, jnp.ones((1, 2 * features)), jnp.ones((1, ))
        )

    tx = optax.adam(1e-3)

    # Create the train state.
    state = train_state.TrainState.create(
        apply_fn=denoiser.apply, params=params['params'], tx=tx
    )

    return state, params


class UtilsTests(chex.TestCase):
    """Run tests on utils functions."""

    @chex.all_variants
    def test_sample(self):
        """Test sampling function."""
        rng = jax.random.PRNGKey(0)
        features = 5
        steps = 64
        sampler = 'ddpm'
        n_models = 2

        # Create a train state.
        state, params = _create_state_diffusion(rng, features, 'post')

        sample = self.variant(
            utils.sample,
            static_argnames=['sample_shape', 'feature_shape', 'steps','sampler']
        )

        # Check that the samples have the correct shape.
        sample_shape = (16,)
        x0_samples = sample(
            rng, state, params, sample_shape, features * n_models, steps, sampler
        )
        self.assertTupleEqual(
            x0_samples.shape, sample_shape + (features * n_models,)
        )

        sample_shape = (16, 32,)
        # Add new shape to obsrevations.
        params['variables']['A'] = jnp.tile(
            params['variables']['A'][None], (sample_shape) + (1, 1, 1)
        )
        params['variables']['y'] = jnp.tile(
            params['variables']['y'][None], (sample_shape) + (1,)
        )
        params['variables']['cov_y'] = jnp.tile(
            params['variables']['cov_y'][None], (sample_shape) + (1,1)
        )

        x0_samples = sample(
            rng, state, params, sample_shape, features * 2, steps, sampler
        )
        self.assertTupleEqual(
            x0_samples.shape, sample_shape + (features * n_models,)
        )

        # Check that the ValueError is raised if a sampler is not implemented.
        with self.assertRaises(ValueError):
            sampler = 'other_sampler'
            x0_samples = sample(
                rng, state, params, sample_shape, features, steps, sampler
            )

    @chex.all_variants
    def test_sample_gibbs(self):
        """Test sampling function."""
        rng = jax.random.PRNGKey(0)
        features = 5
        steps = 64
        gibbs_rounds = 2
        sampler = 'ddpm'

        # Create a train state.
        state, params = _create_state_diffusion(rng, features, 'post')

        sample_gibbs = self.variant(
            utils.sample_gibbs,
            static_argnames=[
                'sample_shape', 'feature_shape', 'steps', 'sampler',
                'gibbs_rounds'
            ]
        )

        # Check that the samples have the correct shape.
        sample_shape = (16,)
        n_models = 2
        initial_samples = jnp.zeros((sample_shape) + (features * 2,))
        params['variables']['A'] = jnp.tile(
            params['variables']['A'], (sample_shape) + (1, 1, 1)
        )
        params['variables']['y'] = jnp.tile(
            params['variables']['y'], (sample_shape) + (1,)
        )
        params['variables']['cov_y'] = jnp.tile(
            params['variables']['cov_y'], (sample_shape) + (1,1)
        )
        x0_samples = sample_gibbs(
            rng, state, params, initial_samples, steps, sampler, gibbs_rounds
        )
        self.assertTupleEqual(
            x0_samples.shape, sample_shape + (features * n_models,)
        )

    @chex.all_variants
    def test_ppca(self):
        """Test probabilistic PCA function."""
        rng = jax.random.PRNGKey(0)
        features = 5
        rank = 2
        feature_shape = (features,)
        sample_shape = (10_000,)

        # Define model covariance and mean.
        diagonal = jnp.ones(features)
        u_mat = jax.random.normal(rng, shape=(features, rank)) * 1e-1
        cov = jnp.eye(features) * diagonal + u_mat @ u_mat.T
        mean = jax.random.normal(rng, shape=feature_shape)

        # Draw samples from a corresponding multivariate Gaussian.
        x_samples = jax.random.multivariate_normal(
            rng, mean, cov, shape=sample_shape
        )

        # Check that the mean and covariance matrix from PPCA have correct
        # shapes and values.
        ppca = self.variant(utils.ppca, static_argnames=['rank'])
        mu_x, cov_x = ppca(rng, x_samples, rank=rank)

        self.assertTupleEqual(mu_x.shape, feature_shape)
        self.assertTrue(jnp.allclose(mu_x, mean, atol=1e-1))
        self.assertTupleEqual(cov_x.diagonal.shape, feature_shape)
        self.assertTrue(jnp.allclose(cov_x.diagonal, diagonal, atol=1e-1))
        self.assertTupleEqual(cov_x.u_mat.shape, (features, rank))
        self.assertTupleEqual(cov_x.v_mat.shape, (rank, features))
        self.assertTrue(jnp.allclose(cov, cov_x.full_matrix(), atol=1e-1))


    def test_sinkhorn_divergence(self):
        """Test sinkhorn divergence calculation. POT doesn't support jit.
        """
        # Draw from two different normal distirbutions.
        n_samps = 1000
        rng_u, rng_v = jax.random.split(jax.random.PRNGKey(4))
        u = jax.random.normal(rng_u, shape=(n_samps, 2))
        v = jax.random.normal(rng_v, shape=(n_samps, 2)) * 0.5 + 0.1

        self.assertGreater(utils.sinkhorn_divergence(u, v), 1e-6)

        # Test that approaching the correct distribution improves the sinkhorn
        # divergence.
        v_new = jax.random.normal(rng_v, shape=(n_samps, 2))
        self.assertGreater(
            utils.sinkhorn_divergence(u, v), utils.sinkhorn_divergence(u, v_new)
        )


if __name__ == '__main__':
    absltest.main()
