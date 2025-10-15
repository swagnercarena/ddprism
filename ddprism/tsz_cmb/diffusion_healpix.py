r"""Extension of the diffusion helpers for HEALPix data."""

from typing import Callable, Optional, Sequence

from flax import linen as nn
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from ddprism import embedding_models
from ddprism import diffusion
from ddprism import linalg


class Denoiser(diffusion.Denoiser):
    r"""Denoiser model.

    .. math:: f(x_t) \approx E[x | x_t]

    References:
        Elucidating the Design Space of Diffusion-Based Generative Models
        (Karras et al., 2022) https://arxiv.org/abs/2206.00364

    Arguments:
        sde_model: SDE model.
        score_model: Noise conditional score network.
        emb_features: Number of features in positional embedding of time.
    """
    score_model: nn.Module
    emb_features: int = 64

    def __call__(
        self, xt: Array, t: Array, vec_map: Array, train: bool = True
    ) -> Array:
        """Call the score model and rescale for better performance.

        Arguments:
            xt: The noisy draws with shape (*, D).
            t: Time with shape (*)
            vec_map: Vector map of position of each pixel.
                Shape (*, n_pixels, 3).
            train: Train keyword passed to score model.

        Returns
            Denoiser output.
        """
        sigma_t = self.sde_sigma(t)

        # Get the four rescaling terms for the model output.
        c_skip = 1 / (sigma_t ** 2 + 1)
        c_out = sigma_t / jnp.sqrt(sigma_t ** 2 + 1)
        c_in = 1 / jnp.sqrt(sigma_t ** 2 + 1)

        # Embed noise for the conditioning of the model.
        emb_noise = embedding_models.positional_embedding(
            jnp.log(sigma_t), emb_features=self.emb_features
        )

        # Add extra dimension for multiplication with x features.
        c_skip = c_skip[..., None]
        c_out = c_out[..., None]
        c_in = c_in[..., None]

        return (
            c_skip * xt + c_out * self.score_model(
                c_in * xt, emb_noise, vec_map, train
            )
        )


class PosteriorDenoiserJoint(diffusion.PosteriorDenoiserJoint):
    r"""Joint posterior denoiser model for stacked source u = [x^1 ,... ,x^N_s].

    .. math:: f(u_t) \approx E[u|u_t] + Cov_t * \score_{u_t}[log p (y|u_t)]

    Arguments:
        denoiser_models: List of denoiser models for each source prior.
        y_features: Number of features in y space.
        x_features: List of number of features for each source model. If None,
            assumes all models have the same dimensionality (inferred from data).
        rtol: Tolerance to use when solving using conjugate gradient.
        maxiter: Maximum iterations when solving using conjugate gradient.
        use_dplr: If True, use DPLR representation for cov_y instead of a
            full matrix. Default is false.
        safe_divide: Minimum value allowed for denominators in division within
            conjugate gradient calculations.
        regularization: Regularization added to diagonal of linear system.
        error_threshold: Threshold for error in conjugate gradient calculations.

    Notes:
        Can also be used as a regular posterior denoiser if only one denoiser
        model is passed in.
    """
    denoiser_models: Sequence[nn.Module]
    y_features: int
    x_features: Sequence[int]
    rtol: float = 1e-3
    maxiter: int = 1
    use_dplr: bool = False
    safe_divide: float = 1e-32
    regularization: float = 0.0
    error_threshold: Optional[float] = None

    def _get_x_features(self, index: Optional[int] = None) -> Sequence[int]:
        """Return the feature dimensions for each model.

        Arguments:
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns:
            List of feature dimensions for each model.
        """
        if index is None:
            return self.x_features
        return [self.x_features[index]]

    def _split_u(self, u: Array, index: Optional[int] = None) -> list[Array]:
        """Split u into individual x vectors with potentially different sizes.

        Arguments:
            u: Concatenated vector of all x vectors.
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns:
            List of x vectors.
        """
        x_features = self._get_x_features(index)
        split_indices = np.cumsum(jnp.array(x_features[:-1])).tolist()
        return jnp.split(u, split_indices, axis=-1)

    def sde_x_t(
        self, u: Array, z: Array, t: Array, vec_map: Array, index: int = None
    ) -> Array:
        """Evolve u according to the underlying SDE.

        Arguments:
            u: Variables at time step 0 for each model. Shape
                (*, sum of x_features)
            z: Random noise draws. Shape (*, sum of x_features)
            t: Time for each example.
            vec_map: Vector map of position of each pixel.
                Shape (*, n_pixels, 3).
            index: Optional parameter specifying which of the model to
                use. Mainly used for Gibbs sampling.

        Returns:
            SDE evolved values of x.
        """
        x = self._split_u(u, index)
        z = self._split_u(z, index)
        sde_x_t_list = [
            model.sde_x_t(x_split, z_split, vec_map, t) for
            model, x_split, z_split in zip(
                self.denoiser_models_idx(index), x, z
            )
        ]
        return jnp.concatenate(sde_x_t_list, axis=-1)

    @nn.compact
    def __call__(
        self, ut: Array, t: Array, vec_map: Array, train: bool = False,
        index: int = None,
    ) -> Array:
        """Call the posterior score model and rescale for better performance.

        Arguments:
            ut: The noisy draws with shape (*, sum of x_features).
            t: Time with shape (*)
            vec_map: Vector map of position of each pixel.
                Shape (*, n_pixels, 3).
            train: Train keyword passed to denoisers.
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns
            Expectation and scaled score.
        """
        x_features_list = self._get_x_features(index)

        # Initialize our y and A matrix. The initialized shape is all that
        # matters. The correct values will be passed in as a variable.
        out_shape = ut.shape[:-1] + (self.y_features,)
        y = self.variable(
            "variables", "y", lambda: jnp.ones(out_shape)
        )
        # One A matrix per source, each with potentially different x dimensions.
        # A will be a list/tuple to handle variable dimensions.
        A_list = []
        for i, x_features in enumerate(x_features_list):
            A_i = self.variable(
                "variables", f"A_{i}",
                lambda xf=x_features: jnp.ones(
                    ut.shape[:-1] + (self.y_features, xf)
                )
            )
            A_list.append(A_i.value)

        # Create transposed versions
        A_t_list = [jnp.swapaxes(A_i, -1, -2) for A_i in A_list]

        # Use DPLR matrix if requested.
        if self.use_dplr:
            cov_y = self.variable(
                "variables", "cov_y",
                lambda: linalg.DPLR(jnp.zeros(out_shape), None, None)
            )
        else:
            cov_y = self.variable(
                "variables", "cov_y",
                lambda: jnp.zeros(out_shape + (self.y_features,))
            )

        # We want to return the score and the expectation for the u vector that
        # is the concatenation of all the x vectors.
        sigma_t_list = jnp.moveaxis(self.sde_sigma(t, index), -1, 0)
        cov_t_list = [sigma_t[..., None] ** 2 for sigma_t in sigma_t_list]

        # Our denoisers operate on each of the x values in our list.
        xt = self._split_u(ut, index)

        # Return list of E[x|x_t] and VJP of E[x|x_t] for each model.
        x_exp_list, vjp_list = zip(*[
            jax.vjp(lambda x: model(x, t, vec_map, train), xt_split) for
            (xt_split, model) in zip(
                xt, self.denoiser_models_idx(index)
            )
        ])

        # Returns E[y]. Deal with batch dimension for A.
        y_exp = sum(
            diffusion.matmul(A_list[i], x_exp)
            for i, x_exp in enumerate(x_exp_list)
        )

        # Compute Cov[y|x_t] function for solve.
        def cov_y_xt(v):
            # Start with the covariance of y.
            value = diffusion.matmul(cov_y.value, v)

            # Add the variance from each model.
            for i in range(self.n_models(index)):
                value += (
                    cov_t_list[i] * diffusion.matmul(
                        A_list[i],
                        vjp_list[i](diffusion.matmul(A_t_list[i], v))[0]
                    )
                )

            return value

        # Computes the score using conjugate gradient method.
        b = y.value - y_exp
        v = diffusion.cg_batched_adaptive(
            cov_y_xt, b, self.maxiter, self.rtol, self.safe_divide,
            self.regularization, self.error_threshold
        )

        cov_t_score = jnp.concat(
            [
                cov_t_list[i] * vjp_list[i](diffusion.matmul(A_t_list[i], v))[0]
                for i in range(self.n_models(index))
            ], axis=-1
        )
        u_exp = jnp.concat(x_exp_list, axis=-1)

        # Returns E[u|u_t] + Cov_t * \score_{u_t}[log p (y|u_t)].
        return u_exp + cov_t_score