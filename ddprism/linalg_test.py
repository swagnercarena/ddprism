"Test scripts for linalg.py"

from absl.testing import absltest

import chex
import jax
import jax.numpy as jnp

from ddprism import linalg

def _create_DPLR_instance(features=5):
    u_mat = (
        jnp.ones((features, features // 2)) +
        jnp.eye(features)[:,:features // 2]
    ) / jnp.sqrt(features)
    return linalg.DPLR(
        diagonal=jnp.ones((features,)), u_mat=u_mat, v_mat=u_mat.T
    )

class DPLRTests(chex.TestCase):
    """Runs tests on DPLR functions."""

    @chex.all_variants(without_device=False)
    def test_add(self):
        """Test that addition works as expected."""
        dplr_instance = _create_DPLR_instance()
        vec = jax.random.normal(
            jax.random.PRNGKey(2),
            shape=(30, dplr_instance.diagonal.shape[-1])
        )

        # Test right and left addition.
        def _add(v, u):
            return v + u, u + v, v - u
        add = self.variant(_add)

        result, result_right, result_sub = add(dplr_instance, vec)
        assert jnp.allclose(result.diagonal, dplr_instance.diagonal + vec)
        assert jnp.allclose(result_right.diagonal, dplr_instance.diagonal + vec)
        assert jnp.allclose(result_sub.diagonal, dplr_instance.diagonal - vec)

    @chex.all_variants(without_device=False)
    def test_mul(self):
        """Test that multiplication works as expected."""
        dplr_instance = _create_DPLR_instance()
        vec = jax.random.normal(
            jax.random.PRNGKey(2),
            shape=(30, dplr_instance.diagonal.shape[-1])
        )

        def _mul(v, u):
            return v * u, u * v
        mul = self.variant(_mul)

        result, result_right = mul(dplr_instance, vec)
        assert jnp.allclose(result.diagonal, dplr_instance.diagonal * vec)
        assert jnp.allclose(result_right.diagonal, dplr_instance.diagonal * vec)
        assert jnp.allclose(
            result.u_mat, vec[..., None] * dplr_instance.u_mat
        )
        assert jnp.allclose(
            result_right.u_mat, vec[..., None] * dplr_instance.u_mat
        )

        # Test that matrix scaled correctly.
        full_matrix = dplr_instance.full_matrix()
        assert jnp.allclose(
            full_matrix * vec[..., None], result.full_matrix(),
            atol=1e-3, rtol=1e-3
        )

    @chex.all_variants(without_device=False)
    def test_matmul(self):
        """Test matrix multiplication."""
        dplr_instance = _create_DPLR_instance()
        mat = jax.random.normal(
            jax.random.PRNGKey(2),
            shape=(30, dplr_instance.u_mat.shape[-2], 4)
        )

        # Test __matmul__
        def _matmul(v, u):
            return v @ u
        matmul = self.variant(_matmul)

        result = matmul(dplr_instance, mat)
        full_matrix = dplr_instance.full_matrix()
        assert jnp.allclose(result, full_matrix @ mat, atol=1e-3, rtol=1e-3)

        # Test __rmatmul__
        mat = jnp.swapaxes(mat, -1, -2)
        def _rmatmul(v, u):
            return v @ u
        matmul = self.variant(_rmatmul)
        result = matmul(mat, dplr_instance)
        assert jnp.allclose(result, mat @ full_matrix, atol=1e-3, rtol=1e-3)

    def test_rank(self):
        """Test the rank of the matrix."""
        dplr_instance = _create_DPLR_instance()
        self.assertEqual(2, dplr_instance.rank)

    @chex.all_variants
    def test_w(self):
        """Test the woodbury matrix identity."""
        dplr_instance = _create_DPLR_instance()
        def _w(v):
            return v.W
        w = self.variant(_w)
        w_matrix = w(dplr_instance)
        self.assertTupleEqual(w_matrix.shape, (2, 2))

    @chex.all_variants
    def test_inv(self):
        """Test inversion of the matrix."""
        dplr_instance = _create_DPLR_instance()

        def _inv(v):
            return v.inv
        inv = self.variant(_inv)
        dplr_inv = inv(dplr_instance)
        mat = jnp.eye(dplr_instance.u_mat.shape[-2])
        assert jnp.allclose(dplr_instance @ mat, jnp.linalg.inv(dplr_inv @ mat))

    @chex.all_variants
    def test_solve(self):
        """Test solve of the matrix."""
        dplr_instance = _create_DPLR_instance()
        dplr_matrix = dplr_instance.full_matrix()
        vec = jax.random.normal(
            jax.random.PRNGKey(2),
            shape=(dplr_instance.diagonal.shape[-1],)
        )

        def _solve(v, w):
            return v.solve(w)
        solve = self.variant(_solve)

        assert jnp.allclose(
            solve(dplr_instance, vec), jnp.linalg.solve(dplr_matrix, vec)
        )

    @chex.all_variants
    def test_diag(self):
        """Test the diagonal of DLPR."""
        dplr_instance = _create_DPLR_instance()
        full_matrix = dplr_instance.full_matrix()

        def _diag(v):
            return v.diag()
        diag = self.variant(_diag)

        assert jnp.allclose(diag(dplr_instance), jnp.diag(full_matrix))

    @chex.all_variants
    def test_norm(self):
        """Test the norm of DLPR."""
        dplr_instance = _create_DPLR_instance()
        full_matrix = dplr_instance.full_matrix()

        def _norm(v):
            return v.norm()
        norm = self.variant(_norm)

        assert jnp.allclose(norm(dplr_instance), jnp.linalg.norm(full_matrix))

    @chex.all_variants
    def test_slogdet(self):
        """Test the norm of DLPR."""
        dplr_instance = _create_DPLR_instance()
        full_matrix = dplr_instance.full_matrix()

        def _slogdet(v):
            return v.slogdet()
        slogdet = self.variant(_slogdet)

        result = slogdet(dplr_instance)
        result_comp = jnp.linalg.slogdet(full_matrix)

        self.assertAlmostEqual(result[0], result_comp[0], places=5)
        self.assertAlmostEqual(result[1], result_comp[1], places=5)


def _create_DPLR_instance_batch(batch_size=32, features=5):
    rng = jax.random.PRNGKey(4)
    u_mat = jax.random.normal(
        rng, shape=(batch_size, features, features//2)
    ) / jnp.sqrt(features)
    diagonal = jnp.tile(jnp.ones((features,))[None], (batch_size, 1))
    return linalg.DPLR(
        diagonal=diagonal, u_mat=u_mat, v_mat=jnp.moveaxis(u_mat, -1, -2)
    )


class DPLRBatchTests(chex.TestCase):
    """Runs tests on DPLR functions in batch."""

    def test_full_matrix(self):
        """Test that full matrix works with batch"""
        batch_size = 32
        dplr_instance = _create_DPLR_instance_batch(batch_size=batch_size)
        dplr_single = linalg.DPLR(
            dplr_instance.diagonal[0], dplr_instance.u_mat[0],
            dplr_instance.v_mat[0]
        )
        self.assertTrue(
            jnp.allclose(dplr_instance.full_matrix()[0],
            dplr_single.full_matrix())
        )

    @chex.all_variants(without_device=False)
    def test_add(self):
        """Test that addition works as expected."""
        batch_size = 32
        dplr_instance = _create_DPLR_instance_batch(batch_size=batch_size)
        vec = jax.random.normal(
            jax.random.PRNGKey(2),
            shape=(batch_size, dplr_instance.diagonal.shape[-1])
        )

        # Test right and left addition.
        def _add(v, u):
            return v + u, u + v, v - u
        add = self.variant(_add)

        result, result_right, result_sub = add(dplr_instance, vec)
        assert jnp.allclose(result.diagonal, dplr_instance.diagonal + vec)
        assert jnp.allclose(result_right.diagonal, dplr_instance.diagonal + vec)
        assert jnp.allclose(result_sub.diagonal, dplr_instance.diagonal - vec)

    @chex.all_variants(without_device=False)
    def test_mul(self):
        """Test that multiplication works as expected."""
        batch_size = 32
        dplr_instance = _create_DPLR_instance_batch(batch_size=batch_size)
        vec = jax.random.normal(
            jax.random.PRNGKey(2),
            shape=(batch_size, dplr_instance.diagonal.shape[-1])
        )

        def _mul(v, u):
            return v * u, u * v
        mul = self.variant(_mul)

        result, result_right = mul(dplr_instance, vec)
        assert jnp.allclose(result.diagonal, dplr_instance.diagonal * vec)
        assert jnp.allclose(result_right.diagonal, dplr_instance.diagonal * vec)
        assert jnp.allclose(
            result.u_mat, vec[..., None] * dplr_instance.u_mat
        )
        assert jnp.allclose(
            result_right.u_mat, vec[..., None] * dplr_instance.u_mat
        )

        # Test that matrix scaled correctly.
        full_matrix = dplr_instance.full_matrix()
        assert jnp.allclose(full_matrix * vec[..., None], result.full_matrix())

    @chex.all_variants(without_device=False)
    def test_matmul(self):
        """Test matrix multiplication."""
        batch_size = 32
        dplr_instance = _create_DPLR_instance_batch(batch_size=batch_size)
        mat = jax.random.normal(
            jax.random.PRNGKey(2),
            shape=(batch_size, dplr_instance.u_mat.shape[-2], 4)
        )

        # Test __matmul__
        def _matmul(v, u):
            return v @ u
        matmul = self.variant(_matmul)

        result = matmul(dplr_instance, mat)
        full_matrix = dplr_instance.full_matrix()
        assert jnp.allclose(result, full_matrix @ mat)

        # Test __rmatmul__
        mat = jnp.swapaxes(mat, -1, -2)
        def _rmatmul(v, u):
            return v @ u
        matmul = self.variant(_rmatmul)
        result = matmul(mat, dplr_instance)
        assert jnp.allclose(result, mat @ full_matrix)


if __name__ == '__main__':
    absltest.main()
