"""Utility: compute eps_decay from the training schedule."""
import math
 
 
def compute_eps_decay(
    rounds: int,
    local_episodes: int,
    steps_per_episode: int,
    explore_fraction: float = 0.5,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
) -> float:
    """Return eps_decay so epsilon reaches its floor by `explore_fraction`
    of total training steps.
 
    Derivation
    ----------
    ε(t) = ε_end + (ε_start − ε_end) × exp(−t / eps_decay)
 
    We want ε to be 99% of the way to ε_end at t = explore_fraction × total:
        exp(−target_step / eps_decay) = 0.01
        eps_decay = target_step / −ln(0.01)
                  = target_step / 4.605
 
    Args:
        rounds:            Total federated rounds.
        local_episodes:    Episodes per city per round.
        steps_per_episode: Ticks per episode = num_seconds / delta_time.
        explore_fraction:  Fraction of total steps to spend exploring.
                           0.5 = explore for first half, exploit for second.
        eps_start:         Starting epsilon (should match DQNAgent.eps_start).
        eps_end:           Floor epsilon  (should match DQNAgent.eps_end).
 
    Returns:
        eps_decay float, ready to pass to DQNAgent(eps_decay=...).
 
    Example
    -------
    >>> compute_eps_decay(rounds=30, local_episodes=2, steps_per_episode=720)
    4684.4          # explore for first 15 rounds, exploit for last 15
    """
    total_steps   = rounds * local_episodes * steps_per_episode
    target_step   = explore_fraction * total_steps
    # How much of the (eps_start - eps_end) range remains at target_step.
    # 0.01 = 1% remaining → 99% decayed.
    remaining_fraction = 0.01
    eps_decay = target_step / (-math.log(remaining_fraction))   # / 4.605
 
    # Sanity-check: print the schedule so it's visible in the run log.
    import logging
    logger = logging.getLogger(__name__)
    eps_at_floor = eps_end + (eps_start - eps_end) * math.exp(-target_step / eps_decay)
    eps_at_end   = eps_end + (eps_start - eps_end) * math.exp(-total_steps / eps_decay)
    logger.info(
        "eps_decay=%.1f  (total_steps=%d  explore until step %d  "
        "ε@floor_step=%.4f  ε@end=%.4f)",
        eps_decay, total_steps, int(target_step), eps_at_floor, eps_at_end,
    )
    return eps_decay
 