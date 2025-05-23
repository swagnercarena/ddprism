"""Test scripts for datasets.py

To run these tests you need a call like:
```
python datasets_test.py --imagenet_path=/path/to/imagenet/folder
```
"""
import os

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

import datasets

FLAGS = flags.FLAGS
flags.DEFINE_string('imagenet_path', None, 'Path to imagenet files.')


class DatasetsTests(absltest.TestCase):
    """Run tests on CNN functions."""

    def test_get_raw_mnist_images(self):
        """Test that the raw mnist images are returned."""
        (x_train, y_train), (x_test, y_test) = datasets.get_raw_mnist_images()

        # Test some basic properties of the images.
        self.assertLessEqual(jnp.max(x_train), 1.0)
        self.assertLessEqual(jnp.max(x_test), 1.0)
        self.assertTupleEqual(x_train.shape, (60000, 28, 28))
        self.assertTupleEqual(x_test.shape, (10000, 28, 28))

        # Test some basic properties of the labels.
        self.assertTupleEqual(y_train.shape, (60000,))
        self.assertTupleEqual(y_test.shape, (10000,))


    def test_get_raw_grass_images(self):
        """Test that the raw grass images are returned."""
        imagenet_path = FLAGS.imagenet_path
        self.assertIsNotNone(imagenet_path)

        # Remove the npy file if present.
        try:
            os.remove(os.path.join(imagenet_path, 'grass_images.npy'))
        except FileNotFoundError:
            pass

        raw_grass = datasets.get_raw_grass_images(imagenet_path)
        self.assertLessEqual(jnp.max(raw_grass), 1.0)
        self.assertTupleEqual(raw_grass.shape, (1274, 100, 100))

        # Check that the file was saved.
        self.assertTrue(
            os.path.isfile(os.path.join(imagenet_path, 'grass_images.npy'))
        )
        raw_grass_load = datasets.get_raw_grass_images(imagenet_path)
        np.testing.assert_array_almost_equal(raw_grass, raw_grass_load)

        # Delete the file we made.
        os.remove(os.path.join(imagenet_path, 'grass_images.npy'))

    def test_get_corrupted_mnist(self):
        """Test that we can get the full dataset."""
        imagenet_path = FLAGS.imagenet_path
        self.assertIsNotNone(imagenet_path)

        rng = jax.random.PRNGKey(2)
        grass_amp = 1.0
        mnist_amp = 0.5
        dataset_size = 64

        corrupted_mnist, labels = datasets.get_corrupted_mnist(
            rng, grass_amp, mnist_amp, imagenet_path, dataset_size
        )
        self.assertLessEqual(jnp.max(corrupted_mnist), grass_amp + mnist_amp)
        self.assertTupleEqual(corrupted_mnist.shape, (dataset_size, 28, 28, 1))
        self.assertEqual(len(np.unique(labels)), 2)

        # Test large dataset size
        dataset_size = 16_000
        corrupted_mnist, labels = datasets.get_corrupted_mnist(
            rng, grass_amp, mnist_amp, imagenet_path, dataset_size
        )
        self.assertTupleEqual(corrupted_mnist.shape, (dataset_size, 28, 28, 1))

        # Delete the file we made.
        os.remove(os.path.join(imagenet_path, 'grass_images.npy'))

    def test_get_dataset(self):
        """Test that we can get the dataset with the desired mixing matrix."""
        imagenet_path = FLAGS.imagenet_path
        self.assertIsNotNone(imagenet_path)

        rng = jax.random.PRNGKey(2)
        grass_amp = 1.0
        mnist_amp = 0.5
        noise = 0.04
        downsampling_ratios = {1: 0.4, 4: 0.6}
        sample_batch_size = 5
        dataset_size = 15

        corrupted_mnist, A_mat, cov_y, labels = datasets.get_dataset(
            rng, grass_amp, mnist_amp, noise, downsampling_ratios,
            sample_batch_size, imagenet_path, dataset_size
        )
        self.assertLessEqual(
            jnp.max(corrupted_mnist),
            grass_amp + mnist_amp + noise * 5
        )
        self.assertTupleEqual(corrupted_mnist.shape, (dataset_size, 28, 28, 1))
        self.assertTupleEqual(A_mat.shape, (sample_batch_size, 2, 784, 784))
        self.assertEqual(A_mat[0, 0, 0, 0] / A_mat[0, 1, 0, 0], 2.0)
        self.assertListEqual(list(A_mat[3,0,0,:4]), [1/16] * 4)
        np.testing.assert_array_almost_equal(
            cov_y.diagonal, jnp.ones((5, 784)) * noise ** 2
        )
        self.assertEqual(len(np.unique(labels)), 2)

        # Delete the file we made.
        os.remove(os.path.join(imagenet_path, 'grass_images.npy'))


if __name__ == '__main__':
    absltest.main()
