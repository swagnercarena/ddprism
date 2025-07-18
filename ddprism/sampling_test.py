"Test scripts for diffusion.py"

from absl.testing import absltest
from absl.testing import parameterized

import chex
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from ddprism import sampling
from ddprism import embedding_models
from ddprism import diffusion

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


class SamplingTests(chex.TestCase):
    """Run tests on sampling functions."""

    @chex.all_variants
    @parameterized.named_parameters(
        [
            ('_no_clipping', 'none'),
            ('_value_clipping', 'value'),
            ('_percentile_clipping', 'percentile'),
            ('_std_dev_clipping', 'std_dev'),
            ('_invalid_method', 'invalid_method')
        ]
    )
    def test_apply_sample_clipping(self, clip_method):
        """Test the _apply_sample_clipping function."""
        rng = jax.random.PRNGKey(42)
        batch_size = 32
        features = 10

        # Create samples with some extreme values
        x = jax.random.normal(rng, shape=(batch_size, features))
        t = jax.random.uniform(rng, shape=(batch_size, ))
        x_with_outliers = x.at[0, 0].set(10.0)
        x_with_outliers = x_with_outliers.at[1, 0].set(-10.0)

        clip_fn = self.variant(
            sampling._apply_sample_clipping, static_argnames=['clip_method']
        )

        if clip_method == 'none':
            # No clipping - should return original samples
            clipped = clip_fn(x_with_outliers, t, clip_method=clip_method)
            self.assertTrue(jnp.allclose(clipped, x_with_outliers))
        elif clip_method == 'value':
            # Value clipping with clip_value=4.0
            clipped = clip_fn(
                x_with_outliers, t, clip_method=clip_method,
                clip_value=4.0
            )
            self.assertTrue(jnp.all(jnp.abs(clipped) <= 4.0))
            self.assertFalse(jnp.allclose(clipped, x_with_outliers))
        elif clip_method == 'percentile':
            # Percentile clipping
            clipped = clip_fn(
                x_with_outliers, t, clip_method=clip_method,
                clip_percentile_low=5.0, clip_percentile_high=95.0
            )
            # Check that extreme values are clipped
            self.assertTrue(jnp.all(jnp.abs(clipped) <= 4.0))
            self.assertFalse(jnp.allclose(clipped, x_with_outliers))
        elif clip_method == 'std_dev':
            # Standard deviation clipping
            clipped = clip_fn(
                x_with_outliers, t, clip_method=clip_method,
                clip_std_dev_threshold=2.0
            )
            # Check that extreme values are clipped
            self.assertTrue(jnp.all(jnp.abs(clipped) <= 4.0))
            self.assertFalse(jnp.allclose(clipped, x_with_outliers))
        elif clip_method == 'invalid_method':
            with self.assertRaises(ValueError):
                _ = clip_fn(x_with_outliers, t, clip_method=clip_method)
            return

        # Check that shape is preserved
        self.assertEqual(clipped.shape, x_with_outliers.shape)

    @chex.all_variants
    def test_adaptive_clipping(self):
        """Test adaptive clipping functionality."""
        rng = jax.random.PRNGKey(42)
        batch_size = 32
        features = 10

        # Create samples with extreme values
        x = jax.random.normal(rng, shape=(batch_size, features))
        x_with_outliers = x.at[0, 0].set(10.0)
        t = jnp.ones(batch_size)

        clip_fn = self.variant(
            sampling._apply_sample_clipping,
            static_argnames=['clip_method', 'clip_adaptive']
        )

        clipped_early = clip_fn(
            x_with_outliers, t, clip_method='value', clip_value=4.0,
            clip_adaptive=True, clip_early_scaling=0.5, clip_late_scaling=1.0
        )
        self.assertEqual(jnp.max(clipped_early), 2.0)
        clipped_late = clip_fn(
            x_with_outliers, t * 0.0, clip_method='value', clip_value=4.0,
            clip_adaptive=True, clip_early_scaling=0.5, clip_late_scaling=1.0
        )
        self.assertEqual(jnp.max(clipped_late), 4.0)

        self.assertTrue(
            jnp.all(jnp.abs(clipped_early) <= jnp.abs(clipped_late))
        )

        # Early clipping should be more aggressive (smaller effective clip_value)
        # Late clipping should be less aggressive (larger effective clip_value)
        early_max = jnp.max(jnp.abs(clipped_early))
        late_max = jnp.max(jnp.abs(clipped_late))

        self.assertLess(early_max, late_max)

    @chex.all_variants
    @parameterized.named_parameters(
        [(f'_diffusion_type_{dif}', dif) for dif in ['prior', 'posterior']]
    )
    def test_ddpm(self, dif):
        """Test step for DDPM sampling."""
        rng_state, rng_step, rng_x = jax.random.split(jax.random.PRNGKey(2), 3)
        features = 5
        batch_size = 16
        state, params = _create_state_diffusion(rng_state, features, dif)
        n_models = 2 if dif == 'posterior' else 1

        # Check that the DDPM step returns samples of correct shape.
        xt = jax.random.normal(rng_x, shape=(batch_size, n_models * features))
        t = 0.6
        s = 0.5
        step_ddpm = self.variant(sampling._step_ddpm)
        xs = step_ddpm(rng_step, state, params, xt, t, s)
        self.assertTupleEqual(xs.shape, xt.shape)

        # Check that the DDPM step returns the original xt samples if times s
        # and t are the same.
        assert jnp.allclose(xt, step_ddpm(rng_step, state, params, xt, t, t))

    @chex.all_variants
    @parameterized.named_parameters(
        [(f'_diffusion_type_{dif}', dif) for dif in ['prior', 'posterior']]
    )
    def test_ddim(self, dif):
        """Test step function for DDIM sampling."""
        rng_state, rng_step, rng_x = jax.random.split(jax.random.PRNGKey(2), 3)
        features = 5
        batch_size = 16
        state, params = _create_state_diffusion(rng_state, features, dif)
        n_models = 2 if dif == 'posterior' else 1

        # Check that the DDIM step returns samples of correct shape.
        xt = jax.random.normal(rng_x, shape=(batch_size, n_models * features))
        t = 0.6
        s = 0.5
        step_ddim = self.variant(sampling._step_ddim)
        xs = step_ddim(rng_step, state, params, xt, t, s)
        self.assertTupleEqual(xs.shape, xt.shape)

        # Check that the DDIM step returns the original xt samples if times s and t are the same.
        assert jnp.allclose(xt, step_ddim(rng_step, state, params, xt, t, t))

    @chex.all_variants
    @parameterized.named_parameters(
        [(f'_diffusion_type_{dif}', dif) for dif in ['prior', 'posterior']]
    )
    def test_pc(self, dif):
        """Test step function for PC sampling."""
        rng_state, rng_step, rng_x = jax.random.split(jax.random.PRNGKey(2), 3)
        features = 5
        batch_size = 16
        state, params = _create_state_diffusion(rng_state, features, dif)
        n_models = 2 if dif == 'posterior' else 1

        # Check that the PC step returns samples of correct shape.
        xt = jax.random.normal(rng_x, shape=(batch_size, n_models * features))
        t = 0.6
        s = 0.5
        corrections = 3
        tau = 1e-2
        step_pc = self.variant(
            sampling._step_pc, static_argnames=['corrections']
        )
        xs = step_pc(
            rng_step, state, params, xt, t, s, corrections=corrections, tau=tau
        )
        self.assertTupleEqual(xs.shape, xt.shape)

        # Check that the PC step returns the original xt samples if times s and
        # t are the same and corrections is set to 0.
        assert jnp.allclose(
            xt,
            step_pc(rng_step, state, params, xt, t, t, 0, tau)
        )

        # Check that the PC step returns the original xt samples if times s and
        # t are the same and tau is set to 0 (no stochasticity).
        assert jnp.allclose(
            xt,
            step_pc(rng_step, state, params, xt, t, t, corrections, 0.0)
        )

    @chex.all_variants
    @parameterized.named_parameters(
        [(f'_diffusion_type_{dif}', dif) for dif in ['prior', 'posterior']]
    )
    def test_sampling_prior(self, dif):
        """Test sampling functions."""
        rng_state, rng_step, rng_x = jax.random.split(jax.random.PRNGKey(2), 3)
        features = 5
        batch_size = 16
        steps = 64
        state, params = _create_state_diffusion(rng_state, features, dif)
        n_models = 2 if dif == 'posterior' else 1

        sampling_fn = self.variant(
            sampling.sampling,
            static_argnames=['sampler', 'steps', 'corrections', 'tau']
        )

        # Create our noisy samples.
        xt = jax.random.normal(rng_x, shape=(batch_size, n_models * features))

        # Check that the ValueError is raised if a sampler is not implemented.
        with self.assertRaises(ValueError):
            sampler = 'other_sampler'
            _ = sampling_fn(
                rng_step, state, params, xt, steps, sampler=sampler
            )

        # Check that the samplers returns different samples for initial times t.
        t = 0.6
        samplers = ['ddpm', 'ddim', 'pc', 'edm', 'pc_edm']
        for sampler in samplers:
            x0 = sampling_fn(
                rng_step, state, params, xt, steps, sampler=sampler
            )
            x0_t = sampling_fn(
                rng_step, state, params, xt, steps, t=t, sampler=sampler
            )
            self.assertFalse(jnp.allclose(x0, x0_t))
            self.assertTupleEqual(x0.shape, xt.shape)

        # The following test only works if we are sampling from the prior.
        if dif == 'prior':
            # Check that when called for n=1 step, the samplers return samples
            # with x = E[x0|x1] (up to absolute tolerance of 10*sigma_0).
            xt = jax.random.normal(rng_x, shape=(batch_size, features))
            t = jnp.ones(xt.shape[:-1])
            e_x_t = state.apply_fn(params, xt, t)
            sigma_0 = state.apply_fn(params, 0.0, method='sde_sigma')

            samplers = ['ddpm', 'ddim', 'pc']
            for sampler in samplers:
                x0_single_step = sampling.sampling(
                    rng_step, state, params, xt, steps=1, sampler=sampler
                )
                self.assertTrue(
                    jnp.allclose(e_x_t, x0_single_step, atol=10*sigma_0)
                )


if __name__ == '__main__':
    absltest.main()
