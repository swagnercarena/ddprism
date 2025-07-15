r"""Diffusion helpers. Implementation follows
https://github.com/francois-rozet/diffusion-priors/priors/diffusion.py
closely."""

from typing import Callable, Optional, Sequence

from flax import linen as nn
import jax
from jax import Array
import jax.numpy as jnp

from ddprism import embedding_models
from ddprism import linalg


def matmul(matrix: Array, vector: Array) -> Array:
    """Batch matrix multiplication for matrix and vector.

    Arguments:
        matrix: Matrix to multiply with dimension (*, M, N)
        vector: Vector to multiply with dimension (*, N)

    Returns:
        Matrix multiply Mv with dimension (*, M)
    """
    return jnp.squeeze((matrix @ vector[..., None]), -1)


def cg_batched(
    lin_transf: Callable[Array, Array], b: Array, maxiter: int = 50,
    tol: float = 1e-6, safe_divide: float = 1e-32, regularization: float = 0.0
) -> Array:
    """Batched implementation of conjugate gradient method for solving Ax=b.

    Arguments:
        lin_transf: Linear trasnformation of a vector.
        b: Right hand side of the linear system.
        maxiter: Maximum number of iterations of the CG method.
        tol: Tolerance for residual norm. Can be reached before maxiter.
        safe_divide: Minimum value for safe division.
        regularization: Regularization added to diagonal of linear system.

    Returns:
        Solution `x` to the equation Ax=b, where A is our linear transformation.
    """
    # Apply regularization to the linear transformation.
    if regularization > 0:
        reg_lin_transf = lambda x: lin_transf(x) + regularization * x
    else:
        reg_lin_transf = lin_transf

    # Initial guess is zeros.
    x = jnp.zeros_like(b)
    resid = b - reg_lin_transf(x)
    p_vec = resid
    r_norm = jnp.sum(resid * resid, axis=-1, keepdims=True)

    def cond(state):
        """Condition for halting CG calculation."""
        k, _, _, _, norm_cur = state
        # Continue as long as residual is larger than tolerance and we haven't
        # reached max iterations.
        return (k < maxiter) & (jnp.max(jnp.sqrt(norm_cur)) > tol)


    def _safe_divide(
        numerator: Array, denominator: Array, floor: Optional[float] = 1e-32
    ) -> Array:
        """Safe division to avoid floating point errors causing overflow.

        Args:
            numerator: Numerator
            denominator: Denominator
            floor: Minimum value of denominator.

        Returns:
            Save division.
        """
        return numerator / jnp.maximum(denominator, floor)


    def cg_step(state):
        """Step of CG method."""
        k, x, resid, p_vec, r_norm = state
        p_transf = reg_lin_transf(p_vec)

        # Calculate the coefficient of the current conjugate vector.
        alpha_denom = jnp.sum(p_vec * p_transf, -1, keepdims=True)
        broken = ~jnp.isfinite(alpha_denom) | (alpha_denom <= 0)
        alpha = jnp.where(
            broken, 0.0, _safe_divide(r_norm, alpha_denom, safe_divide)
        )

        # Add the conjugate vector to get the next solution.
        x = jnp.where(broken, x, x + alpha * p_vec)

        # Calculate the new residual.
        resid = jnp.where(broken, resid, resid - alpha * p_transf)
        r_norm_new = jnp.sum(resid * resid, -1, keepdims=True)

        # Get the next p vector using the residual.
        beta = jnp.where(
            r_norm == 0.0, 0.0, _safe_divide(r_norm_new, r_norm, safe_divide)
        )
        p_vec = jnp.where(broken, resid, resid + beta * p_vec)

        return (k + 1, x, resid, p_vec, r_norm_new)

    # Run the cg method until convergence or max iterations.
    k = 0
    _, x, *_ = jax.lax.while_loop(cond, cg_step, (k, x, resid, p_vec, r_norm))

    return x


class VESDE(nn.Module):
    r"""Variance exploding (VE) SDE.

    .. math:: x_t = x + \sigma_t z

    with

    .. math:: \sigma_t = \exp(\log(a) (1 - t) + \log(b) t)

    Arguments:
        a: The noise lower bound.
        b: The noise upper bound.
    """
    a: Array = 1e-3
    b: Array = 1e2

    def setup(self):
        self.log_a = jnp.log(self.a)
        self.log_b = jnp.log(self.b)

    def __call__(self, x: Array, z: Array, t: Array) -> Array:
        """Evolve x according to the SDE.

        Arguments:
            x: Variables at time step 0.
            z: Random noise draws.
            t: Time for each example.

        Returns:
            SDE evolved values of x.
        """
        sigma_t = self.sigma(t)
        # Add feature dimension(s).
        sigma_t = sigma_t[..., None]

        return x + sigma_t * z

    def sigma(self, t: Array) -> Array:
        """Return the variance at time t in the SDE

        Arguments:
            t: Time

        Returns:
            Variance.
        """
        return jnp.exp(self.log_a + (self.log_b - self.log_a) * t)


class _Denoiser(nn.Module):
    """Base Denoiser model that deals with the SDE.

    Arguments:
        sde_model: SDE model.
    """
    sde_model: nn.Module

    def sde_sigma(self, t: Array) -> Array:
        """Return sigma(t) from the underlying SDE.

        Arguments:
            t: Time

        Returns:
            Variance.
        """
        return self.sde_model.sigma(t)

    def sde_x_t(self, x: Array, z: Array, t: Array) -> Array:
        """Evolve x according to the underlying SDE.

        Arguments:
            x: Variables at time step 0.
            z: Random noise draws.
            t: Time for each example.

        Returns:
            SDE evolved values of x.
        """
        return self.sde_model(x, z, t)


class Denoiser(_Denoiser):
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

    def __call__(self, xt: Array, t: Array, train: bool = True) -> Array:
        """Call the score model and rescale for better performance.

        Arguments:
            xt: The noisy draws with shape (*, D).
            t: Time with shape (*)
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
            c_skip * xt + c_out * self.score_model(c_in * xt, emb_noise, train)
        )


class GaussianDenoiser(_Denoiser):
    r"""Denoiser model for a Gaussian random variable.

    .. math:: p(x) = N(x | \mu_x, \Sigma_x)

    Arguments:
        sde_model: SDE model.
    """

    @nn.compact
    def __call__(self, xt: Array, t: Array, train: bool = True) -> Array:
        """Call the Gaussian Denoiser.

        Arguments:
            xt: The noisy draws with shape (*, D).
            t: Time with shape (*)
            train: Train keyword has no impact on Gaussian Denoiser.

        Returns
            Gaussian Denoiser output.
        """
        # Initialize our mu_x and cov_x matrices.
        mu_x = self.param(
            "mu_x", lambda rng, shape: jnp.ones(shape), xt.shape[-1:]
        )
        # TODO: This should be a DPLR matrix in the long-term.
        cov_x = self.param(
            "cov_x", lambda rng, shape: jnp.eye(shape[0], shape[1]),
            (xt.shape[-1], xt.shape[-1])
        )

        # Get the inverse total variance.
        sigma_t = self.sde_sigma(t)
        # Turn cov_t into a diagonal matrix with batch dimension.
        cov_t = sigma_t[..., None, None] ** 2 * jnp.eye(xt.shape[-1])

        inv_cov = jnp.linalg.inv(cov_x + cov_t)

        # Score for Gaussian denoiser is analytical.
        return xt - matmul(cov_t, matmul(inv_cov, (xt - mu_x)))


class GaussianDenoiserDPLR(_Denoiser):
    r"""Denoiser model for a Gaussian random variable.

    .. math:: p(x) = N(x | \mu_x, \Sigma_x)

    Arguments:
        sde_model: SDE model.
    """

    @nn.compact
    def __call__(self, xt: Array, t: Array, train: bool = True) -> Array:
        """Call the Gaussian DPLR Denoiser.

        Arguments:
            xt: The noisy draws with shape (*, D).
            t: Time with shape (*)
            train: Train keyword has no impact on Gaussian Denoiser.

        Returns
            Gaussian DPLR Denoiser output.
        """
        # Initialize our mu_x and cov_x matrices.
        mu_x = self.param(
            "mu_x", lambda rng, shape: jnp.ones(shape), xt.shape[-1:]
        )
        cov_x = self.param(
            "cov_x",
            lambda rng, shape: linalg.DPLR(jnp.ones(shape), None, None),
            (xt.shape[-1], )
        )

        # Get the time covariance matrix
        sigma_t = self.sde_sigma(t)
        # Turn cov_t into a diagonal matrix with batch dimension.
        cov_t = sigma_t[..., None] ** 2 * jnp.ones(xt.shape[-1])

        # Score for Gaussian denoiser is analytical.
        return xt - cov_t*(cov_x + cov_t).solve(xt - mu_x)


class PosteriorDenoiserJoint(nn.Module):
    r"""Joint posterior denoiser model for stacked source u = [x^1 ,... ,x^N_s].

    .. math:: f(u_t) \approx E[u|u_t] + Cov_t * \score_{u_t}[log p (y|u_t)]

    Arguments:
        denoiser_models: List of denoiser models for each source prior.
        y_features: Number of features in y space.
        rtol: Tolerance to use when solving using conjugate gradient.
        maxiter: Maximum iterations when solving using conjugate gradient.
        use_dplr: If True, use DPLR representation for cov_y instead of a
            full matrix. Default is false.
        safe_divide: Minimum value allowed for denominators in division within
            conjugate gradient calculations.

    Notes:
        Can also be used as a regular posterior denoiser if only one denoiser
        model is passed in.
    """
    denoiser_models: Sequence[nn.Module]
    y_features: int
    rtol: float = 1e-3
    maxiter: int = 1
    use_dplr: bool = False
    safe_divide: float = 1e-32
    regularization: float = 0.0

    @staticmethod
    def _select_mix_matrix(matrix: Array, index: int = None) -> Array:
        """Return mixing matrix with index selections.

        Arguments:
            matrix: All the mixing matrices.
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns:
            Mixing matrix with index selection.
        """
        if index is None:
            return matrix
        return matrix[..., index:index+1, :, :]

    def denoiser_models_idx(self, index: int = None) -> Sequence[nn.Module]:
        """Return denoiser_models with index selection.

        Arguments:
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns:
            denoiser_models list with index selection.
        """
        if index is None:
            return self.denoiser_models
        return self.denoiser_models[index:index+1]

    def n_models(self, index: int = None) -> int:
        """Return the number of model in the joint posterior.

        Arguments:
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns:
            Number of models, accounting for index.
        """
        if index is None:
            return len(self.denoiser_models)
        return 1

    def sde_sigma(self, t: Array, index: int = None) -> Array:
        """Return a vector with sigma(t) for the underlying SDE of each model.

        Arguments:
            t: Time
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns:
            Variance. Shape (*, n_models).
        """
        sde_sigma_list = [
            jnp.atleast_1d(model.sde_sigma(t)) for
            model in self.denoiser_models_idx(index)
        ]
        return jnp.stack(sde_sigma_list, axis=-1)

    def sde_x_t(self, u: Array, z: Array, t: Array, index: int = None) -> Array:
        """Evolve u according to the underlying SDE.

        Arguments:
            u: Variables at time step 0 for each model. Shape
                (*, n_models * features)
            z: Random noise draws. Shape (*, n_models * features)
            t: Time for each example.
            index: Optional parameter specifying which of the model to
                use. Mainly used for Gibbs sampling.

        Returns:
            SDE evolved values of x.
        """
        x = jnp.split(u, self.n_models(index), axis=-1)
        z = jnp.split(z, self.n_models(index), axis=-1)
        sde_x_t_list = [
            model.sde_x_t(x_split, z_split, t) for
            model, x_split, z_split in zip(
                self.denoiser_models_idx(index), x, z
            )
        ]
        return jnp.concatenate(sde_x_t_list, axis=-1)

    @nn.compact
    def __call__(
        self, ut: Array, t: Array, train: bool = False, index: int = None,
    ) -> Array:
        """Call the posterior score model and rescale for better performance.

        Arguments:
            ut: The noisy draws with shape (*, n_models * D).
            t: Time with shape (*)
            train: Train keyword passed to denoisers.
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns
            Expectation and scaled score.
        """
        # Get the x features from the shape and number of models.
        x_features = ut.shape[-1] // self.n_models(index)

        # Initialize our y and A matrix. The initialized shape is all that
        # matters. The correct values will be passed in as a variable.
        out_shape = ut.shape[:-1] + (self.y_features,)
        y = self.variable(
            "variables", "y", lambda: jnp.ones(out_shape)
        )
        # One A matrix per source.
        A_var = self.variable(
            "variables", "A",
            lambda: jnp.ones(
                ut.shape[:-1] + (self.n_models(), self.y_features, x_features)
            )
        )
        A = self._select_mix_matrix(A_var.value, index) # A.value call in function.
        A_t = jnp.swapaxes(A, -1, -2)

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
        xt = jnp.split(ut, self.n_models(index), axis=-1)

        # Return list of E[x|x_t] and VJP of E[x|x_t] for each model.
        x_exp_list, vjp_list = zip(*[
            jax.vjp(lambda x: model(x, t, train), xt_split) for
            (xt_split, model) in zip(
                xt, self.denoiser_models_idx(index)
            )
        ])

        # Returns E[y]. Deal with batch dimension for A.
        y_exp = sum(
            matmul(A[..., i, :, :], x_exp)
            for i, x_exp in enumerate(x_exp_list)
        )

        # Compute Cov[y|x_t] function for solve.
        def cov_y_xt(v):
            # Start with the covariance of y.
            value = matmul(cov_y.value, v)

            # Add the variance from each model.
            for i in range(self.n_models(index)):
                value += (
                    cov_t_list[i] * matmul(
                        A[..., i, :, :],
                        vjp_list[i](matmul(A_t[..., i, :, :], v))[0]
                    )
                )

            return value

        # Computes the score using conjugate gradient method.
        b = y.value - y_exp
        v = cg_batched(
            cov_y_xt, b, self.maxiter, self.rtol, self.safe_divide,
            self.regularization
        )

        cov_t_score = jnp.concat(
            [
                cov_t_list[i] * vjp_list[i](matmul(A_t[..., i, :, :], v))[0] for
                i in range(self.n_models(index))
            ], axis=-1
        )
        u_exp = jnp.concat(x_exp_list, axis=-1)

        # Returns E[u|u_t] + Cov_t * \score_{u_t}[log p (y|u_t)].
        return u_exp + cov_t_score


class PosteriorDenoiserJointDiagonal(PosteriorDenoiserJoint):
    r"""Joint posterior denoiser model for stacked source u = [x^1 ,... ,x^N_s].

    .. math:: f(u_t) \approx E[u|u_t] + Cov_t * \score_{u_t}[log p (y|u_t)]

    Arguments:
        denoiser_models: List of denoiser models for each source prior.
        y_features: Number of features in y space.
        rtol: Tolerance to use when solving using conjugate gradient.
        maxiter: Maximum iterations when solving using conjugate gradient.
        use_dplr: If True, use DPLR representation for cov_y instead of a
            full matrix. Default is false.
        safe_divide: Minimum value allowed for denominators in division within
            conjugate gradient calculations.
        regularization: Regularization added to diagonal of linear system.

    Notes:
        Unlike PosteriorDenoiserJoint, this implementation assumes that the A
        matrix is diagonal (and therefore that x_features = y_features).
    """

    @staticmethod
    def _select_mix_matrix(matrix: Array, index: int = None) -> Array:
        """Return mixing matrix with index selections.

        Arguments:
            matrix: All the mixing matrices.
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns:
            Mixing matrix with index selection.
        """
        if index is None:
            return matrix
        return matrix[..., index:index+1, :]

    @nn.compact
    def __call__(
        self, ut: Array, t: Array, train: bool = False, index: int = None,
    ) -> Array:
        """Call the posterior score model and rescale for better performance.

        Arguments:
            ut: The noisy draws with shape (*, n_models * D).
            t: Time with shape (*)
            train: Train keyword passed to denoisers.
            index: Optional parameter specifying which of the models to
                use. Mainly used for Gibbs sampling.

        Returns
            Expectation and scaled score.
        """
        # Get the x features from the shape and number of models.
        x_features = ut.shape[-1] // self.n_models(index)

        # Initialize our y and A matrix. The initialized shape is all that
        # matters. The correct values will be passed in as a variable.
        out_shape = ut.shape[:-1] + (self.y_features,)
        y = self.variable(
            "variables", "y", lambda: jnp.ones(out_shape)
        )

        assert x_features == self.y_features # Diagonal A requires D_y = D_x.
        # One A matrix per source.
        A_var = self.variable(
            "variables", "A",
            lambda: jnp.ones(
                ut.shape[:-1] + (self.n_models(), x_features)
            )
        )
        A = self._select_mix_matrix(A_var.value, index) # A.value call in function.

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
        xt = jnp.split(ut, self.n_models(index), axis=-1)

        # Return list of E[x|x_t] and VJP of E[x|x_t] for each model.
        x_exp_list, vjp_list = zip(*[
            jax.vjp(lambda x: model(x, t, train), xt_split) for
            (xt_split, model) in zip(
                xt, self.denoiser_models_idx(index)
            )
        ])

        # Returns E[y]. Deal with batch dimension for A.
        y_exp = sum(
            A[..., i, :] * x_exp
            for i, x_exp in enumerate(x_exp_list)
        )

        # Compute Cov[y|x_t] function for solve.
        def cov_y_xt(v):
            # Start with the covariance of y.
            value = matmul(cov_y.value, v)

            # Add the variance from each model.
            for i in range(self.n_models(index)):
                value += (
                    cov_t_list[i] * (
                        A[..., i, :] *
                        vjp_list[i](A[..., i, :] * v)[0]
                    )
                )

            return value

        # Computes the score using conjugate gradient method.
        b = y.value - y_exp
        v = cg_batched(
            cov_y_xt, b, self.maxiter, self.rtol, self.safe_divide,
            self.regularization
        )

        cov_t_score = jnp.concat(
            [
                cov_t_list[i] * vjp_list[i](A[..., i, :] * v)[0] for
                i in range(self.n_models(index))
            ], axis=-1
        )
        u_exp = jnp.concat(x_exp_list, axis=-1)

        # Returns E[u|u_t] + Cov_t * \score_{u_t}[log p (y|u_t)].
        return u_exp + cov_t_score
