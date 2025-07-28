"""Utility functions for PCPCA."""
from typing import Tuple

from flax import linen as nn
import jax
from jax import Array
import jax.numpy as jnp



###### Functions for calculating latent variable posterior in linear CLVM.######

def latent_posterior_from_feat_bkg(
    s_mat: Array, mu_bkg: Array, obs: Array, sigma_obs: float
) -> Tuple[Array, Array]:
    """Calculate latent variable posterior from feature space.

    Args:
        s_mat: Transformation matrix from z latent to feature space.
        mu_bkg: Mean of the background observation.
        obs: Background observation.
        sigma_obs: Standard deviation of the background observation.

    Returns:
        Tuple of mean and covariance of the latent variables.
    """
    # Calculate the covariance matrix of the latent variables.
    sigma_latent = jnp.linalg.inv(
        sigma_obs ** 2 * jnp.eye(s_mat.shape[1]) + s_mat.T @ s_mat
    )

    # Calculate the mean of the latent variables.
    mu_latent = sigma_latent @ s_mat.T @ (obs - mu_bkg)

    # Add final scaling to the covariance matrix.
    sigma_latent *= sigma_obs ** 2

    return mu_latent, sigma_latent


def latent_posterior_from_obs_bkg(
    s_mat: Array, mu_bkg: Array, obs: Array, sigma_obs: float, a_mat: Array
) -> Tuple[Array, Array]:
    """Calculate latent variable posterior from observation space.

    Args:
        s_mat: Transformation matrix from z latent to observation space.
        mu_bkg: Mean of the background observation.
        obs: Background observation.
        sigma_obs: Standard deviation of the background observation.
        a_mat: Linear transformation from feature to observation space.

    Returns:
        Tuple of mean and covariance of the latent variables.
    """
    s_mat = a_mat @ s_mat
    mu_bkg = a_mat @ mu_bkg

    return latent_posterior_from_feat_bkg(s_mat, mu_bkg, obs, sigma_obs)


def latent_posterior_from_feat_enr(
    s_mat: Array, w_mat: Array, mu_enr: Array, obs: Array, sigma_obs: float
) -> Tuple[Array, Array]:
    """Calculate latent variable posterior from feature space.

    Args:
        s_mat: Transformation matrix from z latent to feature space.
        w_mat: Transformation matrix from t latent to feature space.
        mu_enr: Mean of the enriched observation.
        obs: Enriched observation.
        sigma_obs: Standard deviation of the enriched observation.

    Returns:
        Tuple of mean and covariance of the latent variables. Assume that
        the latent variables are ordered z then t.
    """
    # Calculate the covariance matrix of the latent variables.
    m_mat = jnp.concatenate([s_mat, w_mat], axis=-1)

    return latent_posterior_from_feat_bkg(m_mat, mu_enr, obs, sigma_obs)


def latent_posterior_from_obs_enr(
    s_mat: Array, w_mat: Array, mu_enr: Array, obs: Array, sigma_obs: float,
    a_mat: Array
) -> Tuple[Array, Array]:
    """Calculate latent variable posterior from observation space.

    Args:
        s_mat: Transformation matrix from z latent to observation space.
        w_mat: Transformation matrix from t latent to observation space.
        mu_enr: Mean of the enriched observation.
        obs: Enriched observation.
        sigma_obs: Standard deviation of the enriched observation.
        a_mat: Linear transformation from feature to observation space.

    Returns:
        Tuple of mean and covariance of the latent variables. Assume that
        the latent variables are ordered z then t.
    """
    # Transform the matrices and means to account for observation matrix.
    s_mat = a_mat @ s_mat
    w_mat = a_mat @ w_mat
    mu_enr = a_mat @ mu_enr

    return latent_posterior_from_feat_enr(s_mat, w_mat, mu_enr, obs, sigma_obs)


class CLVMLinear(nn.Module):
    """CLVM with linear mapping of latent variables to the feature space."""
    features: int
    latent_dim_z: int
    latent_dim_t: int
    obs_dim: int

    def setup(self):
        self.mu_signal = self.param(
            "mu_signal", nn.initializers.zeros, (self.features,)
        )
        self.mu_bkg = self.param(
            "mu_bkg", nn.initializers.zeros, (self.features,)
        )
        self.s_mat = self.param(
            "s_mat", nn.initializers.lecun_normal(),
            (self.features, self.latent_dim_z)
        )
        self.w_mat = self.param(
            "w_mat", nn.initializers.lecun_normal(),
            (self.features, self.latent_dim_t)
        )
        self.log_sigma_obs = self.variable(
            "variables", "log_sigma_obs",
            lambda: jnp.zeros((1,))
        )

    def encode_bkg_feat(self, feat: Array) -> Tuple[Array, Array]:
        """Encode the input data into the latent distribution.

        Args:
            feat: Background feature with shape [batch_size, features].
        """
        sigma_obs = jnp.exp(self.log_sigma_obs.value)
        return jax.vmap(
            latent_posterior_from_feat_bkg, in_axes=(None, None, 0, None)
        )(
            self.s_mat, self.mu_bkg, feat, sigma_obs
        )

    def encode_bkg_obs(self, obs: Array, a_mat: Array) -> Tuple[Array, Array]:
        """Encode the input data into the latent distribution.

        Args:
            obs: Background observation with shape [batch_size, obs_dim].
            a_mat: Linear transformation from feature to observation space with
                shape [batch_size, obs_dim, features].
        """
        sigma_obs = jnp.exp(self.log_sigma_obs.value)
        return jax.vmap(
            latent_posterior_from_obs_bkg, in_axes=(None, None, 0, None, 0)
        )(
            self.s_mat, self.mu_bkg, obs, sigma_obs, a_mat
        )

    def encode_enr_feat(self, feat: Array) -> Tuple[Array, Array]:
        """Encode the input data into the latent distribution.

        Args:
            feat: Enriched feature.

        Returns:
            Tuple of mean and covariance of the latent variables.
        """
        sigma_obs = jnp.exp(self.log_sigma_obs.value)
        return jax.vmap(
            latent_posterior_from_feat_enr, in_axes=(None, None, None, 0, None)
        )(
            self.s_mat, self.w_mat, self.mu_signal + self.mu_bkg, feat,
            sigma_obs
        )

    def encode_enr_obs(self, obs: Array, a_mat: Array) -> Tuple[Array, Array]:
        """Encode the input data into the latent distribution.

        An a matrix has to be passed in as a variable for the observation.

        Args:
            obs: Background observation.

        Returns:
            Tuple of mean and covariance of the latent variables.
        """
        # The transformation matrix and log sigma_obs should be passed in as
        # variables.
        sigma_obs = jnp.exp(self.log_sigma_obs.value)
        return jax.vmap(
            latent_posterior_from_obs_enr, in_axes=(None, None, None, 0, None, 0)
        )(
            self.s_mat, self.w_mat, self.mu_signal + self.mu_bkg, obs,
            sigma_obs, a_mat
        )

    def decode_bkg_feat(self, z_latent: Array) -> Array:
        """Decode the latent variables into the feature space.

        Args:
            z_latent: Latent variables for the background with shape
                [batch_size, latent_dim_z].

        Returns:
            Decoded feature with shape [batch_size, features].
        """
        return jnp.einsum(
            "ij,bj->bi", self.s_mat, z_latent
        ) + self.mu_bkg

    def decode_bkg_obs(self, z_latent: Array, a_mat: Array) -> Array:
        """Decode the latent variables into the observation space.

        Args:
            z_latent: Latent variables for the background with shape
                [batch_size, latent_dim_z].
            a_mat: Linear transformation from feature to observation space with
                shape [batch_size, obs_dim, features].

        Returns:
            Decoded observation with shape [batch_size, obs_dim].
        """
        feat = self.decode_bkg_feat(z_latent)
        return jnp.einsum('bij,bj->bi', a_mat, feat)

    def decode_signal_feat(self, t_latent: Array) -> Array:
        """Decode the latent variables into the feature space.

        Args:
            t_latent: Latent variables with shape [batch_size, latent_dim_t].

        Returns:
            Decoded feature with shape [batch_size, features].
        """
        return jnp.einsum(
            "ij,bj->bi", self.w_mat, t_latent
        ) + self.mu_signal

    def decode_signal_obs(self, t_latent: Array, a_mat: Array) -> Array:
        """Decode the latent variables into the feature space.

        An a matrix has to be passed in as a variable for the observation.

        Args:
            t_latent: Latent variables with shape [batch_size, latent_dim_t].
            a_mat: Linear transformation from feature to observation space with
                shape [batch_size, obs_dim, features].

        Returns:
            Decoded observation with shape [batch_size, obs_dim].
        """
        signal_feat = self.decode_signal_feat(t_latent)
        return jnp.einsum('bij,bj->bi', a_mat, signal_feat)

    def _kl_div(self, mu: Array, sigma: Array) -> Array:
        """Calculate the KL divergence for the latent variables.

        Args:
            mu: Mean of the latent variables with shape
                [batch_size, latent_dim].
            sigma: Covariance of the latent variables with shape
                [batch_size, latent_dim, latent_dim].

        Returns:
            KL divergence with shape [batch_size].
        """
        trace = jnp.trace(sigma, axis1=-2, axis2=-1)
        log_det = jnp.linalg.slogdet(sigma)[1]
        return 0.5 * (trace + jnp.einsum('bi,bi->b', mu, mu) - log_det)

    def _recon_loss(
        self, x_recon: Array, x: Array
    ) -> Array:
        """Calculate the reconstruction loss.
        """
        sigma_obs = jnp.exp(self.log_sigma_obs.value)
        return 0.5 * jnp.sum(
            (x - x_recon) ** 2 / sigma_obs ** 2, axis=-1
        )

    def _latent_draw(self, rng: Array, mu: Array, sigma: Array) -> Array:
        """Draw a sample from the latent distribution.

        Args:
            rng: Random number generator.
            mu: Mean of the latent variables with shape
                [batch_size, latent_dim].
            sigma: Covariance of the latent variables with shape
                [batch_size, latent_dim, latent_dim].

        Returns:
            Latent draw with shape [batch_size, latent_dim].
        """
        return jax.random.multivariate_normal(
            rng, mu, sigma
        )

    def loss_bkg_feat(self, rng: Array, feat: Array) -> Array:
        """Calculate the loss for the background feature.

        Args:
            feat: Background feature with shape [batch_size, features].

        Returns:
            Loss.
        """
        mu_latent, sigma_latent = self.encode_bkg_feat(feat)
        latent_draw = self._latent_draw(rng, mu_latent, sigma_latent)
        feat_recon = self.decode_bkg_feat(latent_draw)

        return jnp.mean(
            self._kl_div(mu_latent, sigma_latent) +
            self._recon_loss(feat_recon, feat)
        )

    def loss_bkg_obs(self, rng: Array, obs: Array, a_mat: Array) -> Array:
        """Calculate the loss for the background observation.

        Args:
            obs: Background observation with shape [batch_size, obs_dim].
            a_mat: Linear transformation from feature to observation space with
                shape [batch_size, obs_dim, features].

        Returns:
            Loss.
        """
        mu_latent, sigma_latent = self.encode_bkg_obs(obs, a_mat)
        latent_draw = self._latent_draw(rng, mu_latent, sigma_latent)
        obs_recon = self.decode_bkg_obs(latent_draw, a_mat)

        return jnp.mean(
            self._kl_div(mu_latent, sigma_latent) +
            self._recon_loss(obs_recon, obs)
        )

    def loss_enr_feat(self, rng: Array, feat: Array) -> Array:
        """Calculate the loss for the enriched feature.

        Args:
            feat: Enriched feature with shape [batch_size, features].

        Returns:
            Loss.
        """
        mu_latent, sigma_latent = self.encode_enr_feat(feat)
        latent_draw = self._latent_draw(rng, mu_latent, sigma_latent)
        z_latent, t_latent = jnp.split(
            latent_draw, [self.latent_dim_z], axis=-1
        )
        feat_recon = (
            self.decode_signal_feat(t_latent) + self.decode_bkg_feat(z_latent)
        )

        return jnp.mean(
            self._kl_div(mu_latent, sigma_latent) +
            self._recon_loss(feat_recon, feat)
        )

    def loss_enr_obs(self, rng: Array, obs: Array, a_mat: Array) -> Array:
        """Calculate the loss for the enriched observation.

        Args:
            obs: Enriched observation with shape [batch_size, obs_dim].
            a_mat: Linear transformation from feature to observation space with
                shape [batch_size, obs_dim, features].

        Returns:
            Loss.
        """
        mu_latent, sigma_latent = self.encode_enr_obs(obs, a_mat)
        latent_draw = self._latent_draw(rng, mu_latent, sigma_latent)
        z_latent, t_latent = jnp.split(
            latent_draw, [self.latent_dim_z], axis=-1
        )
        obs_recon = (
            self.decode_signal_obs(t_latent, a_mat) +
            self.decode_bkg_obs(z_latent, a_mat)
        )

        return jnp.mean(
            self._kl_div(mu_latent, sigma_latent) +
            self._recon_loss(obs_recon, obs)
        )


class CLVMVAE(CLVMLinear):
    """CLVM with a VAE-like architecture.

    The class assumes you will call the correct loss function for the VAEs you
    have provided and will crash if you pass an incompatible VAE (i.e. one that
    is built for observations and expects an a matrix when you have features).
    """
    signal_decoder: nn.Module
    bkg_decoder: nn.Module
    signal_encoder: nn.Module
    bkg_encoder: nn.Module

    def setup(self):
        self.log_sigma_obs = self.variable(
            "variables", "log_sigma_obs",
            lambda: jnp.zeros((1,))
        )

    def encode_bkg_feat(self, feat: Array) -> Tuple[Array, Array]:
        """Encode the input data into the latent distribution.

        Args:
            feat: Background feature with shape [batch_size, features].
        """
        return self.bkg_encoder.encode_feat(feat)

    def encode_bkg_obs(self, obs: Array, a_mat: Array) -> Tuple[Array, Array]:
        """Encode the input data into the latent distribution.

        Args:
            obs: Background observation with shape [batch_size, obs_dim].
            a_mat: Linear transformation from feature to observation space with
                shape [batch_size, obs_dim, features].
        """
        return self.bkg_encoder.encode_obs(obs, a_mat)

    def encode_enr_feat(self, feat: Array) -> Tuple[Array, Array]:
        """Encode the input data into the latent distribution.

        Args:
            feat: Enriched feature with shape [batch_size, features].

        Returns:
            Tuple of mean and covariance of the latent variables.
        """
        mu_bkg, sigma_bkg = self.bkg_encoder.encode_feat(feat)
        mu_signal, sigma_signal = self.signal_encoder.encode_feat(feat)
        return (
            jnp.concatenate([mu_bkg, mu_signal], axis=-1),
            jnp.concatenate([sigma_bkg, sigma_signal], axis=-1)
        )

    def encode_enr_obs(self, obs: Array, a_mat: Array) -> Tuple[Array, Array]:
        """Encode the input data into the latent distribution.

        An a matrix has to be passed in as a variable for the observation.

        Args:
            obs: Background observation with shape [batch_size, obs_dim].
            a_mat: Linear transformation from feature to observation space with
                shape [batch_size, obs_dim, features].

        Returns:
            Tuple of mean and covariance of the latent variables.
        """
        mu_bkg, sigma_bkg = self.bkg_encoder.encode_obs(obs, a_mat)
        mu_signal, sigma_signal = self.signal_encoder.encode_obs(obs, a_mat)
        return (
            jnp.concatenate([mu_bkg, mu_signal], axis=-1),
            jnp.concatenate([sigma_bkg, sigma_signal], axis=-1)
        )

    def decode_bkg_feat(self, z_latent: Array) -> Array:
        """Decode the latent variables into the feature space.

        Args:
            z_latent: Latent variables for the background with shape
                [batch_size, latent_dim_z].

        Returns:
            Decoded feature with shape [batch_size, features].
        """
        return self.bkg_decoder.decode_feat(z_latent)

    def decode_bkg_obs(self, z_latent: Array, a_mat: Array) -> Array:
        """Decode the latent variables into the observation space.

        Args:
            z_latent: Latent variables for the background with shape
                [batch_size, latent_dim_z].
            a_mat: Linear transformation from feature to observation space with
                shape [batch_size, obs_dim, features].

        Returns:
            Decoded observation with shape [batch_size, obs_dim].
        """
        feat = self.decode_bkg_feat(z_latent)
        return jnp.einsum('bij,bj->bi', a_mat, feat)

    def decode_signal_feat(self, t_latent: Array) -> Array:
        """Decode the latent variables into the feature space.

        Args:
            t_latent: Latent variable t with shape [batch_size, latent_dim_t].

        Returns:
            Decoded feature with shape [batch_size, features].
        """
        return self.signal_decoder.decode_feat(t_latent)

    def decode_signal_obs(self, t_latent: Array, a_mat: Array) -> Array:
        """Decode the latent variables into the feature space.

        An a matrix has to be passed in as a variable for the observation.

        Args:
            t_latent: Latent variable t with shape [batch_size, latent_dim_t].

        Returns:
            Decoded observation with shape [batch_size, obs_dim].
        """
        feat = self.decode_signal_feat(t_latent)
        return jnp.einsum('bij,bj->bi', a_mat, feat)

    def _kl_div(self, mu: Array, sigma: Array) -> Array:
        """Calculate the KL divergence for the latent variables.

        Args:
            mu: Mean of the latent variables with shape
                [batch_size, latent_dim].
            sigma: Covariance of the latent variables with shape
                [batch_size, latent_dim, latent_dim].

        Returns:
            KL divergence with shape [batch_size].
        """
        trace = jnp.sum(sigma, axis=-1)
        log_det = jnp.sum(jnp.log(sigma), axis=-1)
        return 0.5 * (trace + jnp.einsum('bi,bi->b', mu, mu) - log_det)

    def _latent_draw(self, rng: Array, mu: Array, sigma: Array) -> Array:
        """Draw a sample from the latent distribution.

        Args:
            rng: Random number generator.
            mu: Mean of the latent variables with shape
                [batch_size, latent_dim].
            sigma: Covariance of the latent variables with shape
                [batch_size, latent_dim].

        Returns:
            Latent draw with shape [batch_size, latent_dim].
        """
        return jax.random.normal(rng, shape=mu.shape) * jnp.sqrt(sigma) + mu
