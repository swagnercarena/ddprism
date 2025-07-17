"""Utilities for dynamic training parameters during EM loops."""

from typing import Any, Mapping, Optional


def compute_dynamic_epochs(
    lap: int, base_epochs: int, total_laps: int,
    use_dynamic: Optional[bool] = True
) -> int:
    """Compute number of epochs for current EM lap, increasing over time.

    Epochs increase exponentially from base_epochs to 4 * base_epochs.

    Args:
        lap: Current EM lap.
        base_epochs: Number of epochs for the first EM lap.
        total_laps: Total number of EM laps.
        use_dynamic: If True, use dynamic epochs. If False, return base_epochs.

    Returns:
        Number of epochs for current EM lap.
    """
    power = lap // (total_laps // 2) if total_laps > 1 else 0
    factor = 2 ** power
    if use_dynamic:
        return int(base_epochs * factor)
    return base_epochs


def compute_dynamic_sampling_steps(
    lap: int, base_steps: int, total_laps: int,
    use_dynamic: Optional[bool] = True
) -> int:
    """Compute number of sampling steps for current EM lap, increasing.

    Steps increase exponentially from base_steps to 4 * base_steps.

    Args:
        lap: Current EM lap.
        base_steps: Number of sampling steps for the first EM lap.
        total_laps: Total number of EM laps.
        use_dynamic: If True, use dynamic steps. If False, return base_steps.

    Returns:
        Number of sampling steps for current EM lap.
    """
    power = lap // (total_laps // 2) if total_laps > 1 else 0
    factor = 2 ** power
    if use_dynamic:
        return int(base_steps * factor)
    return base_steps


def get_dynamic_sampling_kwargs(
    base_kwargs: Mapping[str, Any], lap: int, total_laps: int,
    use_dynamic: Optional[bool] = True
) -> Mapping[str, Any]:
    """Get sampling kwargs with dynamic number of steps.

    Args:
        base_kwargs: Base sampling kwargs.
        lap: Current EM lap.
        total_laps: Total number of EM laps.
        use_dynamic: If True, use dynamic steps. If False, return base_kwargs.

    Returns:
        Sampling kwargs with number of steps modified for current lap.
    """
    dynamic_kwargs = base_kwargs.copy_and_resolve_references()
    if use_dynamic:
        dynamic_kwargs['steps'] = compute_dynamic_sampling_steps(
            lap, base_kwargs['steps'], total_laps
        )
    return dynamic_kwargs
