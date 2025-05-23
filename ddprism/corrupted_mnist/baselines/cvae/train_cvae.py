import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
from absl import app, flags

from tensorflow.keras.datasets import mnist

import math, os
from typing import Any, Callable, Mapping, Sequence, Tuple
from jax import Array
from flax import linen as nn
from flax.training import train_state, orbax_utils
import optax
import wandb
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointer
from pathlib import Path
import tqdm
from einops import rearrange

from galaxy_diffusion import training_utils
from galaxy_diffusion.corrupted_mnist import datasets
from galaxy_diffusion.corrupted_mnist import metrics

import config_base_grass
import config_base_mnist
import models
imagenet_path = '/mnt/home/aakhmetzhanova/ceph/galaxy-diffusion/corrupted-mnist/dataset/grass_jpeg/' 

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
flags.DEFINE_string('imagenet_path', None, 'path to imagenet grass dataset.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training configuration.',
)

class cVAE(nn.Module):
    r"""Creates an cVAE model. 
    Assumes that the input is a flattened image and the dimensionality of the latent space is the same for both relevant and irrelevant features.
    Arguments:
        latent_features: The number of latent features.
        hid_features: The number of hidden features.
        activation: The activation function constructor.
    """
    latent_features: int
    encoder_s: nn.Module
    encoder_z: nn.Module
    decoder: nn.Module

    def get_latents(self, x: Array, encoder: nn.Module) -> Tuple:
        out = encoder(x)
        mu, var = jnp.split(out, 2, axis=-1)
        var = jnp.exp(var) 
        return mu, var
        
    def get_salient_latents(self, x: Array) -> Array:
        # Compute means and variances for relevant (s) variables.
        mu_s_x, var_s_x = self.get_latents(x, self.encoder_s)
        return mu_s_x, var_s_x
        
    def get_background_latents(self, x: Array) -> Array:
        # Compute means and variances for irrelevant (z) variables.
        mu_z_x, var_z_x = self.get_latents(x, self.encoder_z)
        return mu_z_x, var_z_x

    def sample_latents(self, rng: Array, mu: Array, var: Array, sample_shape: Tuple) -> Array:
        eps = jax.random.normal(rng, shape=sample_shape + (self.latent_features,))
        samples = mu + var * eps
        return samples

    def sample_salient_latents(self, rng: Array, x: Array, sample_shape: Tuple) -> Array:
        # Compute means and variances for relevant (s) latent variables.
        mu, var = self.get_salient_latents(x)
        samples = self.sample_latents(rng, mu, var, sample_shape)
        return samples, mu, var

    def sample_background_latents(self, rng: Array, x: Array, sample_shape: Tuple) -> Array:
        # Compute means and variances for background (b) latent variables.
        mu, var = self.get_background_latents(x)
        samples = self.sample_latents(rng, mu, var, sample_shape)
        return samples, mu, var

    def denoise_samples(self, rng: Array, x: Array) -> Array:
        sample_shape = x.shape[:-3]
        # Sample only the salient features
        s_x_samples, mu_s_x, var_s_x = self.sample_salient_latents(rng, x, sample_shape) 
        # Zero out the irrelevant features and decode the latents. 
        x_samples = self.decoder(jnp.concatenate((s_x_samples, jnp.zeros_like(s_x_samples)), axis=-1))
        return x_samples
    
    def decode_latent_samples(self, s: Array, z: Array) -> Array:
        # Concatenate latents
        latent_samples = jnp.concatenate((s, z), axis=-1)
        x = self.decoder(latent_samples)
        return x
        
    @nn.compact
    def __call__(self, rng: Array, x: Array, b: Array) -> Array:
        
        # Compute means and variances for relevant (s) and irrelevant (z) latent variables,
        # And draw samples from the corresponding latent spaces.
        sample_shape_x = x.shape[:-3]
        sample_shape_b = b.shape[:-3]
        rng, rng_s_x, rng_s_b, rng_z_x, rng_z_b = jax.random.split(rng, 5)

        s_x_samples, mu_s_x, var_s_x = self.sample_salient_latents(rng_s_x, x, sample_shape_x) 
        s_b_samples, mu_s_b, var_s_b = self.sample_salient_latents(rng_s_b, b, sample_shape_b) 
        z_x_samples, mu_z_x, var_z_x = self.sample_background_latents(rng_z_x, x, sample_shape_x) 
        z_b_samples, mu_z_b, var_z_b = self.sample_background_latents(rng_z_b, b, sample_shape_b) 

        # Get background and target samples 
        # Target samples
        x_samples = self.decode_latent_samples(s_x_samples, z_x_samples)
        # Background samples
        b_samples = self.decode_latent_samples(jnp.zeros_like(s_b_samples), z_b_samples)
        # Return samples and means and variances of the distributions (all necessary for computing the loss function)
        return x_samples, b_samples, mu_s_x, var_s_x, mu_z_x, var_z_x, mu_z_b, var_z_b 
        
@jax.jit
def update_model(state, grads):
    """Update model with gradients."""
    return state.apply_gradients(grads=grads)

@jax.jit
def apply_model(state, x, b, rng, beta=1):
    """Computes gradients and loss for a single batch."""

    # loss
    def loss_fn(params):
        # get samples from the VAE both in latent and in image space
        rng_samples, rng_drop = jax.random.split(rng)
        input_dim = x.shape[-1]
        x_samples, b_samples, mu_s_x, var_s_x, mu_z_x, var_z_x, mu_z_b, var_z_b  = state.apply_fn(
            {'params': params}, rng_samples, x, b, rngs={'dropout': rng_drop}
        )

        # Compute loss
        # Reconstruction loss
        reconstruction_loss =  optax.losses.squared_error(x, x_samples).mean(axis=-1)
        reconstruction_loss += optax.losses.squared_error(b, b_samples).mean(axis=-1)
        reconstruction_loss *= input_dim
        

        # KL loss
        kl_loss =  jnp.log(var_s_x) - (mu_s_x**2) - var_s_x
        kl_loss += jnp.log(var_z_x) - (mu_z_x**2) - var_z_x
        kl_loss += jnp.log(var_z_b) - (mu_z_b**2) - var_z_b 
        kl_loss = -0.5*kl_loss.sum(axis=-1) # summing over the independent latents
        
        
        # CVAE loss
        reconstruction_loss = reconstruction_loss.mean()
        kl_loss = kl_loss.mean()
        loss = (reconstruction_loss + beta*kl_loss)
        return loss, {'reconstruction_loss': reconstruction_loss, 'kl_loss': kl_loss}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_dict), grads = grad_fn(state.params)

    return grads, loss, loss_dict

@jax.jit
def get_latent_samples(state, x, rng,):
    """Returns denoised samples for a single batch."""
    rng_samples, rng_drop = jax.random.split(rng)
    denoised_samples = state.apply_fn({'params': state.params}, rng_samples, x, method='denoise_samples', 
                                 rngs={'dropout': rng_drop})

    return denoised_samples


def denoised_dataset(state, target, rng, batch_size=128):
    """Returns denoised dataset."""
    dataset_size = target.shape[0]
    steps_per_epoch = dataset_size // batch_size
    
    denoised_samples = []
    for _ in range(steps_per_epoch):
        target_batch = target[_*batch_size:(_+1)*batch_size]
        
        rng, _ = jax.random.split(rng)
        x_samples = get_latent_samples(state, target_batch, rng,)
    
        denoised_samples.append(x_samples)
    denoised_samples = jnp.concatenate(denoised_samples)
    return denoised_samples


def main(_):
    """Train a joint posterior denoiser."""
    model_params = FLAGS.config
    workdir = FLAGS.workdir
    
    os.makedirs(workdir, exist_ok=True)

    print(f'Found devices {jax.devices()}')
    print(f'Working directory: {workdir}')

    # Generate training data.
    # Foreground train dataset with corrupted mnist digits 
    config_mnist = config_base_mnist.get_config()
    
    rng = jax.random.key(config_mnist.rng_key)
    rng_dataset, rng_comp, rng = jax.random.split(rng, 3)
    
    mnist_amp=config_mnist.mnist_amp
    f_train = datasets.get_corrupted_mnist(
        rng_dataset, grass_amp=1., mnist_amp=config_mnist.mnist_amp,
        imagenet_path=imagenet_path,
        dataset_size=config_mnist.dataset_size,
        zeros_and_ones=True
    )
    # Train dataset with uncorrupted mnist digits for computing metrics later on.
    f_train_uncorrupted = datasets.get_corrupted_mnist(
        rng_dataset, grass_amp=0., mnist_amp=1.,
        imagenet_path=imagenet_path,
        dataset_size=config_mnist.dataset_size,
        zeros_and_ones=True
    )
    # Background train dataset with grass only.
    config_grass = config_base_grass.get_config()
    rng = jax.random.key(config_grass.rng_key)
    rng_dataset, rng_comp, rng = jax.random.split(rng, 3)
    b_train = datasets.get_corrupted_mnist(
        rng_dataset, grass_amp=1., mnist_amp=0.,
        imagenet_path=imagenet_path, 
        dataset_size=config_grass.dataset_size,
        zeros_and_ones=True)

    # Load classifier model to compute the FCD metric.
    classifier_workdir=Path('/mnt/home/aakhmetzhanova/galaxy-diffusion/galaxy_diffusion/corrupted_mnist/mnist_classifier/') 
    checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(classifier_workdir, checkpointer)
    classifier_model = metrics.CNN()
    classifier_params = checkpoint_manager.restore(checkpoint_manager.latest_step())['params'] 
    checkpoint_manager.close()

    # Reshape target and background datasets.
    image_shape = f_train[0].shape[-3:]
    features    = image_shape[0]*image_shape[1]*image_shape[2]
    target     = f_train[0].reshape(-1, features)
    target_labels = f_train[1]
    background = b_train[0].reshape(-1, features)

    # Initialize models.
    latent_features = model_params.latent_features
    # Initialize encoder.
    if model_params.encoder == 'mlp':
        encoder_s, encoder_z = [models.encoder_MLP(latent_features, bias=False) for _ in range(2)]
    elif model_params.encoder == 'unet':
        encoder_s, encoder_z = [
            models.UNetEncoder(
                latent_features=latent_features,
                hid_channels=(128,), 
                hid_blocks=(1,), 
                input_shape=image_shape,
                heads=None, 
                dropout_rate=0.1,
            ) for _ in range(2)
        ]
    elif model_params.encoder == 'unet_full_depth':
        encoder_s, encoder_z = [
            models.UNetEncoder(
                latent_features=latent_features,
                hid_channels=(32, 64, 128), 
                hid_blocks=(1, 1, 1), 
                input_shape=image_shape,
                dropout_rate=0.1,
            ) for _ in range(2)
        ]

    # Initialize decoder.
    if model_params.decoder == 'mlp':
        decoder = models.decoder_MLP(features, bias=False)
    elif model_params.decoder == 'unet':
        decoder = models.UNetDecoder(
            in_shape=(28, 28, 1),
            hid_channels=(128, ), 
            hid_blocks=(1, ), 
            dropout_rate=0.1
        )

    rng, rng_state = jax.random.split(rng, 2)
    model_cVAE = cVAE(latent_features=latent_features, encoder_s=encoder_s, encoder_z=encoder_z, decoder=decoder)
    params_cVAE = model_cVAE.init(rng, rng_state, jnp.ones((1, features)), jnp.ones((1, features)))

    # Training setup.
    learning_rate = model_params.learning_rate
    beta = model_params.beta
    epochs = model_params.epochs
    batch_size = model_params.batch_size

    dataset_size = target.shape[0]
    steps_per_epoch = dataset_size // batch_size
    
    learning_rate_fn = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=epochs*steps_per_epoch
    )
    tx = optax.adam(learning_rate=learning_rate_fn)
    
    state = train_state.TrainState.create(apply_fn=model_cVAE.apply, params=params_cVAE['params'], tx=tx) 
    
    # Initialize the run.
    if model_params.run_name is None:
        run_name = model_params.encoder + '_to_' + model_params.decoder + f'_lr_{learning_rate:.0e}'
    else:
        run_name = model_params.run_name
        
    print(run_name)
    wandb.init(
        project='cvae-runs-new',
        config=model_params,
        name=run_name, 
        mode='online'
    )
    
    run_dir = workdir + run_name +'/'
    os.makedirs(run_dir, exist_ok=True)

    rng = jax.random.key(model_params.rng_key)
    # Train the model
    losses_per_epoch = []
    fcd = []
    min_fcd = 1e16
    for epoch in range(epochs):
        
        losses = []
        kl_losses = []
        reconstruction_losses = []
        
        for step in range(steps_per_epoch):
            # Get a random batch.
            rng_epoch, rng_x, rng_b, rng = jax.random.split(rng, 4)
            batch_x = jax.random.randint(rng_x, shape=(batch_size,), minval=0, maxval=dataset_size)
            batch_b = jax.random.randint(rng_b, shape=(batch_size,), minval=0, maxval=dataset_size)
    
            # Compute gradients and losses.
            grads, loss, loss_dict = apply_model( # pylint: disable=not-callable
                state, target[batch_x], background[batch_b], rng_epoch, beta=beta
            )
            state = update_model( # pylint: disable=not-callable
                        state, grads
                    )
            losses.append(loss)
            kl_losses.append(loss_dict['kl_loss'])
            reconstruction_losses.append(loss_dict['reconstruction_loss'])
            
        losses_per_epoch.append([jnp.asarray(losses).mean(), jnp.asarray(kl_losses).mean(), jnp.asarray(reconstruction_losses).mean()])
    
        # Compute FCD between the denoised samples and the training set.
        denoised_samples = denoised_dataset(state, target, rng, batch_size=batch_size).reshape(dataset_size, image_shape[0], image_shape[1], image_shape[2])
        denoised_samples = denoised_samples / mnist_amp
        fcd.append(metrics.fcd_mnist(classifier_model, classifier_params, f_train_uncorrupted[0], denoised_samples))
        
        # Log to wandb.
        wandb.log(
            {'loss': losses_per_epoch[-1][0], 
             'kl_loss': losses_per_epoch[-1][1],
             'reconstruction_loss': losses_per_epoch[-1][2],
             'fcd': fcd[-1]
            },
            step=epoch + 1)

        checkpointer = PyTreeCheckpointer()
        checkpoint_manager = CheckpointManager(
                os.path.join(run_dir, 'checkpoints'), checkpointer
            )
    
        # Save the state parameters and model parameters
        ckpt = { 
                'params': state.params, 
                'losses': jnp.array(losses_per_epoch), 'fcd': jnp.array(fcd)
                    
                }
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(epoch+1, ckpt, save_kwargs={'save_args': save_args})
        

        # Save best model with best FCD in a separate '0'-th checkpoint
        if fcd[-1] < min_fcd:
            # Save the state parameters and model parameters
            ckpt = {
                    'params': state.params, 
                    'losses': jnp.array(losses_per_epoch), 'fcd': jnp.array(fcd)
                        
                    }
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(0, ckpt, save_kwargs={'save_args': save_args})
            min_fcd = fcd[-1]

        checkpoint_manager.close()

    wandb.finish()

if __name__ == '__main__':
    app.run(main)