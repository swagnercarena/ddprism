"""Utilities for dynamic training parameters during EM loops."""


def compute_dynamic_epochs(lap, base_epochs, total_laps):
    """Compute number of epochs for current EM lap, increasing over time.

    Epochs increase exponentially from base_epochs to 8 * base_epochs.

    Args:
        lap: Current EM lap.
        base_epochs: Number of epochs for the first EM lap.
        total_laps: Total number of EM laps.

    Returns:
        Number of epochs for current EM lap.
    """
    power = lap // (total_laps // 4) if total_laps > 1 else 0
    factor = 2 ** power
    return int(base_epochs * factor)


def compute_dynamic_sampling_steps(lap, base_steps, total_laps):
    """Compute number of sampling steps for current EM lap, increasing over time.

    Steps increase exponentially from base_steps to 8 * base_steps.

    Args:
        lap: Current EM lap.
        base_steps: Number of sampling steps for the first EM lap.
        total_laps: Total number of EM laps.
    """
    power = lap // (total_laps // 4) if total_laps > 1 else 0
    factor = 2 ** power
    return int(base_steps * factor)


def get_dynamic_sampling_kwargs(base_kwargs, lap, total_laps):
    """Get sampling kwargs with dynamic number of steps.

    Args:
        base_kwargs: Base sampling kwargs.
        lap: Current EM lap.
        total_laps: Total number of EM laps.

    Returns:
        Sampling kwargs with number of steps modified for current lap.
    """
    dynamic_kwargs = base_kwargs.copy()
    dynamic_kwargs['steps'] = compute_dynamic_sampling_steps(
        lap, base_kwargs['steps'], total_laps
    )
    return dynamic_kwargs
