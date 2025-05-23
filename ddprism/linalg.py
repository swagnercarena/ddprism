r"""Linear algebra helpers. Implementation follows
https://github.com/francois-rozet/diffusion-priors/blob/master/priors/linalg.py
closely."""

from typing import NamedTuple, Tuple

import jax.numpy as jnp
from jax import Array


class DPLR(NamedTuple):
    r"""Diagonal plus low-rank (DPLR) matrix.

    Notes:
        All addition and multiplication operations require the array to be a
        scalar or vector with a batch dimension.
    """
    diagonal: Array
    u_mat: Array = None
    v_mat: Array = None

    def __add__(self, vec: Array) -> NamedTuple:
        return DPLR(self.diagonal + vec, self.u_mat, self.v_mat)

    def __radd__(self, vec: Array) -> NamedTuple:
        return self.__add__(vec)

    def __sub__(self, vec: Array) -> NamedTuple:
        return self.__add__(-vec)

    def __mul__(self, vec: Array) -> NamedTuple:
        diagonal = self.diagonal * vec

        if self.u_mat is None:
            u_mat, v_mat = None, None
        else:
            u_mat, v_mat = vec[..., None] * self.u_mat, self.v_mat

        return DPLR(diagonal, u_mat, v_mat)

    def __rmul__(self, vec: Array) -> NamedTuple:
        diagonal = vec * self.diagonal

        if self.u_mat is None:
            u_mat, v_mat = None, None
        else:
            u_mat, v_mat = vec[..., None] * self.u_mat, self.v_mat

        return DPLR(diagonal, u_mat, v_mat)

    def __matmul__(self, mat: Array) -> Array:
        # Break out of DPLR class and returns matrix.
        if self.u_mat is None:
            return self.diagonal[..., None] * mat
        else:
            return (
                self.diagonal[..., None] * mat +
                jnp.einsum('...ij,...jk,...kn', self.u_mat, self.v_mat, mat)
            )

    def __rmatmul__(self, mat: Array) -> Array:
    # Break out of DPLR class and returns matrix.
        diag = jnp.swapaxes(
            self.diagonal[..., None] * jnp.swapaxes(mat, -1, -2), -1 , -2
        )
        if self.u_mat is None:
            return diag
        else:
            return (
                diag +
                jnp.einsum('...ij,...jk,...kn', mat, self.u_mat, self.v_mat)
            )

    @property
    def shape(self) -> Tuple:
        """Return the shape of the matrix described by the DPLR tuple.

        Returns:
            shape of matrix.
        """
        N = len(self.diagonal)
        return (N, N)

    @property
    def rank(self) -> int:
        """Return the rank of the low-rank matrix.

        Returns:
            Rank of matrix.
        """
        if self.u_mat is None:
            return 0
        return self.u_mat.shape[-1]

    @property
    def W(self) -> Array:
        """Return the Woodbury matrix identity.

        Returns:
            Woodbury matrix identity for DPLR decomposition.

        Reference:
            https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        """
        return (
            jnp.eye(self.rank) +
            jnp.einsum(
                '...ik,...k,...kj', self.v_mat, 1 / self.diagonal, self.u_mat
            )
        )

    @property
    def inv(self) -> NamedTuple:
        """Return the inverse of the matrix."""

        diagonal = 1 / self.diagonal

        if self.u_mat is None:
            u_mat, v_mat = None, None
        else:
            u_mat = -diagonal[..., None] * self.u_mat
            v_mat = jnp.linalg.solve(
                self.W, self.v_mat
            ) * diagonal[..., None, :]

        return DPLR(diagonal, u_mat, v_mat)

    def solve(self, x: Array) -> Array:
        """Solve Ab=x for the DPLR matrix.

        Arguments:
            x: Vector to mutiply by inverse.

        Returns:
            Solution b for Ab = x.
        """
        diagonal = 1 / self.diagonal

        if self.u_mat is None:
            return diagonal * x

        non_diag = diagonal * jnp.squeeze(
            self.u_mat @ jnp.linalg.solve(
                self.W, self.v_mat @ jnp.expand_dims(diagonal * x, axis=-1)
            ),
            axis=-1,
        )
        return diagonal * x - non_diag

    def full_matrix(self) -> Array:
        """Return the full matrix.

        Returns:
            The full matrix (not in DPLR representation).

        Notes:
            Can be a very expensive calculation in high dimensions.
        """
        return (
            self.diagonal[..., None] * jnp.eye(self.diagonal.shape[-1]) +
            self.u_mat @ self.v_mat
        )

    def diag(self) -> Array:
        """Return the diagonal component of the full matrix.

        Returns:
            Diagonal component of the full matrix.
        """
        if self.u_mat is None:
            return self.diagonal

        return (
            self.diagonal +
            jnp.einsum('...ij,...ji->...i', self.u_mat, self.v_mat)
        )

    def norm(self) -> Array:
        """Return the norm of the DPLR matrix.

        Returns:
            Norm of the DPLR matrix.
        """
        if self.u_mat is None:
            return jnp.sqrt(jnp.sum(self.diagonal**2, axis=-1))

        return jnp.sqrt(
            jnp.sum(self.diagonal ** 2, axis=-1)
            + 2 * jnp.einsum(
                '...i,...ij,...ji', self.diagonal, self.u_mat, self.v_mat
            )
            + jnp.sum((self.v_mat @ self.u_mat) ** 2, axis=(-1, -2))
        )

    def slogdet(self) -> Tuple[Array, Array]:
        """Return the sign and log absolute determinant.

        Returns:
            Sign and log absolute determinant.
        """
        sign, logabsdet = jnp.linalg.slogdet(self.W)

        sign = sign * jnp.prod(jnp.sign(self.diagonal), axis=-1)
        logabsdet = logabsdet + jnp.sum(
            jnp.log(jnp.abs(self.diagonal)), axis=-1
        )

        return sign, logabsdet
