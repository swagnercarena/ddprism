"""Diffusion sampling methods."""

from typing import Dict, Optional, Sequence

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import Array


def sampling(
    key: Array, state:TrainState, params: Dict[str, Array], xt: Array,
    steps: int, t: Optional[float] = 1.0, sampler='ddpm', **kwargs
) -> Array:
    """ Sample from the reverse SDE using the DDPM scheme.

    Arguments:
        key: Rng key for sampling.
        state: Trained diffusion model(s).
        params: Params to use with the TrainState.
        xt: Initial x samples at time t to evolve through the reverse diffusion
            equation. Shape (*, features).
        steps: Number of discrete steps to use in DDPM.
        t: Initial t of the xt samples. Will be evolved to t=0 and default is
            t=1.
        sampler: Sampler to use. Options are 'ddpm', 'ddim', 'pc', 'edm', and
            'pc_edm'.
        **kwargs: Additional keyword arguments for the sampler and clipping.

    Returns:
        Samples at t=0.
    """
    # Get the discrete time steps for the reverse SDE.
    dt = t / steps
    time = jnp.linspace(t, dt, steps)

    # One random key for each time step.
    keys = jax.random.split(key, steps)

    if sampler == 'ddpm':
        _step = _step_ddpm
    elif sampler == 'ddim':
        _step = _step_ddim
    elif sampler == 'pc':
        _step = _step_pc
    elif sampler == 'edm':
        _step = _step_edm
    elif sampler == 'pc_edm':
        _step = _step_pc_edm
    else:
        raise ValueError('Invalid sampler specified.')

    # Helper function for scan that packages the input and outputs.
    def f(xt, time_key):
        time, key = time_key
        return _step(key, state, params, xt, time, time - dt, **kwargs), None

    # Scan back to t=0.
    x0, _ = jax.lax.scan(f, xt, (time, keys))

    # Apply one final denoiser step to the final samples for optimal
    # performance.
    t0 = jnp.zeros(x0.shape[:-1])
    return _apply_sample_clipping(
        state.apply_fn(params, x0, t0, train=False), t0, **kwargs
    )


def _add_batch_dim_time(xt_shape: Sequence[int], time: Array) -> Array:
    """Add batch dimension to time.

    Arguments:
        xt_shape: Shape of xt.
        time: Time without batch dimension.

    Returns:
        Time with batch dimension.

    Notes:
        Assumes that xt is of shape (*, dim).
    """
    return jnp.ones(xt_shape[:-1]) * time


def _reshape_sigma(xt_shape: Sequence[int], sigma: Array) -> Array:
    """Return sigma output with appropriate broadcasting shape for n_models.

    Arguments:
        state: Trained diffusion model.
        params: Params to use with the TrainState.
        xt_shape: Shape of xt.
        sigma: Sigma with batch dimension but without xt shape accounting.
            Depending on number of models, may have shape (*) or (*, n_models).

    Returns:
        Sigma with appropriate broadcasting shape for n_models.
    """
    # Add model dimension to sigma if it doesn't have one.
    sigma = sigma.reshape(xt_shape[:-1] + (-1,))

    # Sigma now has shape (*, n_models). We want to give it shape
    # (*, n_models * features)
    n_models = sigma.shape[-1]
    features = xt_shape[-1] // n_models

    return jnp.repeat(sigma, features, axis=-1)


def _step_ddpm(
    key: Array, state: TrainState, params: Dict[str, Array], xt: Array,
    t: float, s: float, **kwargs
) -> Array:
    """ Take on step of the ddpm sampler.

    Arguments:
        key: Rng key for step.
        state: Trained diffusion model.
        params: Params to use with the TrainState.
        xt: Current xt samples. Shape (*, n_models, features).
        t: Time of current samples.
        s: Next time index for SDE sampling.

    Returns:
        Samples at time s.
    """
    t = _add_batch_dim_time(xt.shape, t)
    s = _add_batch_dim_time(xt.shape, s)

    # Get the noise at both time steps.
    sigma_s = state.apply_fn(params, s, method='sde_sigma')
    sigma_t = state.apply_fn(params, t, method='sde_sigma')
    sigma_s = _reshape_sigma(xt.shape, sigma_s)
    sigma_t = _reshape_sigma(xt.shape, sigma_t)

    # Expectation given current time step.
    e_x_t = state.apply_fn(params, xt, t, train=False)

    # Scaling parameter and noise for reverse SDE step.
    tau = 1 - (sigma_s / sigma_t) ** 2
    eps = jax.random.normal(key, xt.shape)

    # Reverse diffusion sampling step.
    # Follows Eq. (47) in Score-Based Generative Modeling through Stochastic
    # Differential Equations (Song et al., 2021)
    # https://arxiv.org/abs/2011.13456
    xs = xt - tau * (xt - e_x_t) + sigma_s * jnp.sqrt(tau) * eps

    return _apply_sample_clipping(xs, s, **kwargs)


def _step_ddim(
    _: Array, state: TrainState, params: Dict[str, Array], xt: Array,
    t: float, s: float, **kwargs
) -> Array:
    """ Take on step of the ddim sampler.

    Arguments:
        state: Trained diffusion model.
        params: Params to use with the TrainState.
        xt: Current xt samples. Shape (*, n_models, features).
        t: Time of current samples.
        s: Next time index for SDE sampling.

    Returns:
        Samples at time s.
    """
    t = _add_batch_dim_time(xt.shape, t)
    s = _add_batch_dim_time(xt.shape, s)

    # Get the noise at both time steps.
    sigma_s = state.apply_fn(params, s, method='sde_sigma')
    sigma_t = state.apply_fn(params, t, method='sde_sigma')
    sigma_s = _reshape_sigma(xt.shape, sigma_s)
    sigma_t = _reshape_sigma(xt.shape, sigma_t)

    # Expectation given current time step.
    e_x_t = state.apply_fn(params, xt, t, train=False)

    # Reverse diffusion sampling step.
    # Follows Eq. (44) in Denoising Diffusion Implicit Models
    # (Song, Meng, and Ermon 2022), https://arxiv.org/abs/2010.02502
    # With replacing the source noise by the score function
    # (e.g. Eq. (151) in Luo 2022, https://arxiv.org/abs/2208.11970).
    xs = xt - (1 - sigma_s / sigma_t) * (xt - e_x_t)

    return _apply_sample_clipping(xs, s, **kwargs)


def _step_pc(
    key: Array, state: TrainState, params: Dict[str, Array], xt: Array,
    t: float, s: float, corrections: int = 1, tau: Array = 1e-1,
    tau_min: Array = 1e-1, alpha: Array = 0, **kwargs
) -> Array:
    """ Take on step of the predictor-corrector sampler.

    Arguments:
        key: Jax PRNG key.
        state: Trained diffusion model.
        params: Params to use with the TrainState.
        xt: Current xt samples. Shape (*, n_models, features).
        t: Time of current samples.
        s: Next time index for SDE sampling.
        corrections: Number of corrections to apply after predictor.
        tau: Step size scaling for correction.
        tau_min: Minimum step size scaling for correction. Only relevant if
            alpha != 0.
        alpha: Exponential for the power law according to which tau changes:
            tau = tau_min + (tau - tau_min) * t ** alpha.

    Returns:
        Samples at time s.
    """
    tau = jnp.asarray(tau)

    xs = _step_ddim(key, state, params, xt, t, s, **kwargs)

    # Function for scan.
    def correction_step(xs, key_corr):
        return (
            _correct(
                key_corr, state, params, xs, s, tau, tau_min, alpha, **kwargs
            ), None
        )

    # Apply corrections.
    keys = jax.random.split(key, corrections)
    xs, _ = jax.lax.scan(correction_step, xs, keys)

    return xs

def _correct(
    key: Array, state: TrainState, params: Dict[str, Array],
    xt: Array, t: float, tau: Array,
    tau_min: Array, alpha: Array, **kwargs
) -> Array:
    """ Correct the prediction using LMC.

    Arguments:
        key: Jax PRNG key.
        state: Trained diffusion model.
        params: Params to use with the TrainState.
        xt: Current xt samples. Shape (*, n_models, features).
        t: Time of current samples.
        corrections: Number of corrections to apply after predictor.
        tau: Step size scaling for correction.
        tau_min: Minimum step size scaling for correction.
        alpha: Exponential for the power law according to which tau changes:
            tau = tau_min + (tau - tau_min) * t ** alpha.

    Returns:
        Samples at time s.
    """
    # Update tau according to time step t.
    tau = tau_min + (tau - tau_min) * t ** alpha

    t = _add_batch_dim_time(xt.shape, t)

    # Get the noise at both time steps.
    sigma_t = state.apply_fn(params, t, method='sde_sigma')

    sigma_t = _reshape_sigma(xt.shape, sigma_t)

    # Expectation given current time step.
    e_x_t = state.apply_fn(params, xt, t, train=False)

    eps = jax.random.normal(key, xt.shape)
    xt_new = xt - tau * (xt - e_x_t) + sigma_t * jnp.sqrt(2 * tau) * eps
    return _apply_sample_clipping(xt_new, t, **kwargs)


def _step_edm(
    _: Array, state: TrainState, params: Dict[str, Array], xt: Array,
    t: float, s: float, **kwargs
) -> Array:
    """ Take on step of the edm sampler from Karras et al. 2022.

    Arguments:
        state: Trained diffusion model.
        params: Params to use with the TrainState.
        xt: Current xt samples. Shape (*, n_models, features).
        t: Time of current samples.
        s: Next time index for SDE sampling.

    Returns:
        Samples at time s.
    """
    t = _add_batch_dim_time(xt.shape, t)
    s = _add_batch_dim_time(xt.shape, s)

    # Get the noise at both time steps.
    sigma_s = state.apply_fn(params, s, method='sde_sigma')
    sigma_t = state.apply_fn(params, t, method='sde_sigma')
    sigma_s = _reshape_sigma(xt.shape, sigma_s)
    sigma_t = _reshape_sigma(xt.shape, sigma_t)

    # Reverse diffusion sampling step.
    # Follows Algorithm 1 in Elucidating the Design Space of Diffusion-Based
    # Generative Models (Karras et al. 2022), https://arxiv.org/abs/2206.00364,
    # with sigma(t) = t and s(t) = 1, such that dsigma/dt = 1, ds/dt = 0.

    # Expectation given current time step.
    e_x_t = state.apply_fn(params, xt, t, train=False)
    # Evaluate x at the time step s.
    d_t = (xt - e_x_t) / sigma_t
    xs = xt + (sigma_s - sigma_t) * d_t

    # Apply second-order correction.
    # Expectation at the next time step.
    e_x_s = state.apply_fn(params, xs, t, train=False)
    # Evaluate x at the time step s with the second-order correction.
    d_s = (xs - e_x_s) / sigma_s
    xs = xt + (sigma_s - sigma_t) * (1/2) * (d_t + d_s)
    return _apply_sample_clipping(xs, s, **kwargs)

def _step_pc_edm(
    key: Array, state: TrainState, params: Dict[str, Array], xt: Array,
    t: float, s: float, corrections: int = 1, tau: Array = 1e-1,
    tau_min: Array = 1e-1, alpha: Array = 0, **kwargs
) -> Array:
    """Take on step of the predictor-corrector sampler with edm replacing ddim.

    Arguments:
        key: Jax PRNG key.
        state: Trained diffusion model.
        params: Params to use with the TrainState.
        xt: Current xt samples. Shape (*, n_models, features).
        t: Time of current samples.
        s: Next time index for SDE sampling.
        corrections: Number of corrections to apply after predictor.
        tau: Step size scaling for correction.

    Returns:
        Samples at time s.
    """
    tau = jnp.asarray(tau)

    xs = _step_edm(key, state, params, xt, t, s, **kwargs)

    # Function for scan.
    def correction_step(xs, key_corr):
        return (
            _correct(
                key_corr, state, params, xs, s, tau, tau_min, alpha, **kwargs
            ), None
        )

    # Apply corrections.
    keys = jax.random.split(key, corrections)
    xs, _ = jax.lax.scan(correction_step, xs, keys)

    return xs


def _apply_sample_clipping(
    x: Array, t: Array, clip_method: Optional[str] = 'none',
    clip_adaptive: Optional[bool] = False, clip_value: Optional[float] = 4.0,
    clip_early_strength: Optional[float] = 0.5,
    clip_late_strength: Optional[float] = 1.0,
    clip_percentile_low: Optional[float] = 0.1,
    clip_percentile_high: Optional[float] = 99.9,
    clip_std_dev_threshold: Optional[float] = 4.0, **kwargs
) -> Array:
    """Apply progressive clipping to prevent sample divergence.

    Arguments:
        x: Samples to clip. Shape (*, features)
        t: Current time step (0 to 1, where 1 is pure noise). Shape (*).
        clip_method: Type of clipping to apply. Options are 'none', 'value',
            'percentile', and 'std_dev'.
        clip_adaptive: Whether to use adaptive clipping.
        clip_value: Value to clip to for value clipping.
        clip_early_strength: Strength of clipping early in sampling.
        clip_late_strength: Strength of clipping late in sampling.
        clip_percentile_low: Lower percentile to clip to.
        clip_percentile_high: Upper percentile to clip to.

    Returns:
        Clipped samples.
    """
    # Always calculate the adaptive factor, jit will throw away if not used.
    adaptive_factor = (
        clip_early_strength +
        (clip_late_strength - clip_early_strength) * (1 - t)
    )[:, None]

    if clip_method == 'none':
        return x
    elif clip_method == 'value':
        # Simple value clipping
        if clip_adaptive:
            clip_value = clip_value * adaptive_factor
        return jnp.clip(x, -clip_value, clip_value)
    elif clip_method == 'percentile':
        # Apply adaptive strength if requested
        if clip_adaptive:
            # Adjust percentiles based on time step
            mid_pct = 50.0
            clip_percentile_low = (
                mid_pct - (mid_pct - clip_percentile_low) * adaptive_factor
            )
            clip_percentile_high = (
                mid_pct + (clip_percentile_high - mid_pct) * adaptive_factor
            )
        low_val = jnp.percentile(x, clip_percentile_low)
        high_val = jnp.percentile(x, clip_percentile_high)
        return jnp.clip(x, low_val, high_val)
    elif clip_method == 'std_dev':
        # Standard deviation-based clipping
        if clip_adaptive:
            clip_std_dev_threshold = clip_std_dev_threshold * adaptive_factor
        mean_val = jnp.mean(x)
        std_val = jnp.std(x)
        return jnp.clip(
            x, mean_val - clip_std_dev_threshold * std_val,
            mean_val + clip_std_dev_threshold * std_val
        )
    else:
        # Unknown method, raise an error.
        raise ValueError(f'Unknown clipping method: {clip_method}')
