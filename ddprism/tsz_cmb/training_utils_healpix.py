"""Utility functions for training on HEALPix data."""

from flax.training import train_state
import jax.numpy as jnp
import optax

from ddprism import diffusion, training_utils
from ddprism.tsz_cmb import diffusion_healpix, embedding_models_healpix


def create_denoiser_transformer(config, healpix_shape):
    """Create the FlatUNet Denoiser."""
    # SDE
    sde = diffusion.VESDE(config.sde['a'], config.sde['b'])

    # Diffusion backbone.
    transformer = embedding_models_healpix.FlatHEALPixTransformer(
        emb_features=config.emb_features, n_blocks=config.n_blocks,
        dropout_rate_block=config.dropout_rate_block, heads=config.heads,
        patch_size_list=config.patch_size_list,
        time_emb_features=config.time_emb_features,
        n_average_layers=config.get('n_average_layers', 0),
        use_patch_convolution=config.get('use_patch_convolution', True),
        healpix_shape=healpix_shape
    )

    return diffusion_healpix.Denoiser(
        sde, transformer, healpix_shape[0],
        emb_features=config.time_emb_features
    )

def create_train_state_transformer(
    rng, config, learning_rate_fn, healpix_shape, params=None
):
    """Creates initial TrainSate for FlatUNet."""
    # Denoiser
    denoiser = create_denoiser_transformer(config, healpix_shape)

    if params is None:
        # Pass an example to get the parameters.
        healpix_features = denoiser.score_model.feat_dim
        params = denoiser.init(
            rng, jnp.ones((1, healpix_features)), jnp.ones((1,))
        )

    optimizer = training_utils.get_optimizer(config)(learning_rate_fn)
    grad_clip_norm = config.get('grad_clip_norm', 1.0)
    tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optimizer)

    return train_state.TrainState.create(
        apply_fn=denoiser.apply, params=params['params'], tx=tx
    )
