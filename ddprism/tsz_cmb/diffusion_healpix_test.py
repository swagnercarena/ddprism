"Test scripts for diffusion_healpix.py"

from absl.testing import absltest
import chex
from flax import linen as nn
import jax
from jax import Array
import jax.numpy as jnp

from ddprism import diffusion
from ddprism import linalg
from ddprism.tsz_cmb import diffusion_healpix, embedding_models_healpix


class DiffusionHEALPixTests(chex.TestCase):
    """Runs tests of HEALPix diffusion functions."""

    @chex.all_variants
    def test_denoiser(self):
        """Test that calling the HEALPix Denoiser behaves as expected."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 3)

        n_pixels = 12 * 8 * 8 # NSIDE=8
        channels = 2
        features = n_pixels * channels

        # Test that the output has the desired shape.
        batch_size = 8
        # Use flattened inputs for compatibility with parent Denoiser
        x_draws = jax.random.normal(
            rng_keys[0], (batch_size, features)
        )
        t_draws = jax.random.uniform(
            rng_keys[1], (batch_size,)
        )
        vec_map = jax.random.normal(rng_keys[2], (batch_size, n_pixels, 3))
        # Normalize vec_map
        vec_map = vec_map / jnp.linalg.norm(vec_map, axis=-1, keepdims=True)

        # Initialize the SDE
        sde = diffusion.VESDE()

        # Initialize HEALPix score model.
        emb_features = 16
        n_blocks = 2
        dropout_rate_block = [0.0, 0.0]
        heads = 2
        time_emb_features = 4
        patch_size_list = [1]
        freq_features = 4
        healpix_shape = (n_pixels, channels)
        score_model = embedding_models_healpix.FlatHEALPixTransformer(
            emb_features=emb_features, healpix_shape=healpix_shape,
            n_blocks=n_blocks, dropout_rate_block=dropout_rate_block,
            heads=heads, patch_size_list=patch_size_list,
            time_emb_features=time_emb_features, freq_features=freq_features
        )

        # Initialize Denoiser.
        denoiser = diffusion_healpix.Denoiser(
            sde, score_model, n_pixels=n_pixels, emb_features=time_emb_features
        )

        # Need to flatten for initialization
        x_init = jnp.ones((1, features))
        t_init = jnp.ones((1,))
        vec_map_init = jnp.ones((1, n_pixels, 3))
        params_denoiser = denoiser.init(rng, x_init, t_init)
        params_denoiser['variables']['vec_map'] = vec_map

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

        sigma_t = apply_func(params_denoiser, t_draws, method='sde_sigma')
        sigma_t_exp = sde.apply({}, t_draws, method='sigma')
        self.assertTrue(jnp.allclose(sigma_t, sigma_t_exp))


class PosteriorDenoiserJointHEALPixTests(chex.TestCase):
    """Runs tests of HEALPix posterior diffusion functions."""

    @chex.all_variants
    def test_posterior_denoiser_joint(self):
        """Test that the HEALPix posterior denoiser returns expected outputs."""
        rng = jax.random.PRNGKey(0)
        rng_keys = jax.random.split(rng, 3)

        n_pixels = 12 * 8 * 8 # NSIDE=8
        channels = [1, 2]
        x_features = [n_pixels * c for c in channels]

        # Test that the output has the desired shape.
        batch_size = 8
        # Use flattened inputs for compatibility with parent Denoiser
        x_draws = jax.random.normal(
            rng_keys[0], (batch_size, sum(x_features))
        )
        t_draws = jax.random.uniform(
            rng_keys[1], (batch_size,)
        )
        vec_map = jax.random.normal(rng_keys[2], (batch_size, n_pixels, 3))
        # Normalize vec_map
        vec_map = vec_map / jnp.linalg.norm(vec_map, axis=-1, keepdims=True)

        # Initialize the SDE
        sde = diffusion.VESDE()

        # Initialize HEALPix score model.
        emb_features = 16
        n_blocks = 2
        dropout_rate_block = [0.0, 0.0]
        heads = 2
        time_emb_features = 4
        patch_size_list = [1]
        freq_features = 4
        score_models = [
            embedding_models_healpix.FlatHEALPixTransformer(
                emb_features=emb_features,
                healpix_shape=(n_pixels, channels[i]),
                n_blocks=n_blocks, dropout_rate_block=dropout_rate_block,
                heads=heads, patch_size_list=patch_size_list,
                time_emb_features=time_emb_features, freq_features=freq_features
            ) for i in range(len(x_features))
        ]

        # Initialize Denoiser.
        denoiser_models = [
            diffusion_healpix.Denoiser(
                sde, score_models[i], n_pixels=n_pixels,
                emb_features=time_emb_features
            )
            for i in range(len(x_features))
        ]

        # Initialize the posterior denoiser.
        y_features = x_features[-1]
        denoiser = diffusion_healpix.PosteriorDenoiserJoint(
            denoiser_models, y_features, x_features=x_features, use_dplr=True
        )
        params_denoiser = denoiser.init(rng, x_draws, t_draws)
        params_denoiser['variables']['vec_map'] = vec_map

        # Check the shapes initialized correctly.
        self.assertTupleEqual(
            params_denoiser['variables']['A_0'].shape,
            (batch_size, y_features, x_features[0])
        )
        self.assertTupleEqual(
            params_denoiser['variables']['A_1'].shape,
            (batch_size, y_features, x_features[1])
        )
        self.assertTupleEqual(
            params_denoiser['variables']['y'].shape, (batch_size, y_features)
        )
        self.assertTupleEqual(
            params_denoiser['variables']['cov_y'].diagonal.shape,
            (batch_size, y_features)
        )

        # Check that the output of the denoiser is of the correct shape.
        apply_func = self.variant(denoiser.apply, static_argnames='method')
        expect_x = apply_func(params_denoiser, x_draws, t_draws)
        self.assertTupleEqual(expect_x.shape, (batch_size, sum(x_features)))

        # Check the sde_sigma outputs.
        sigma_t = apply_func(
            params_denoiser, t_draws, method='sde_sigma'
        )
        self.assertTupleEqual(sigma_t.shape, (batch_size, len(x_features)))


if __name__ == '__main__':
    absltest.main()
