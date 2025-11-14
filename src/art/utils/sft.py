"""Utilities for supervised fine-tuning (SFT)."""

import math
from typing import Generator, List, Literal


def create_lr_schedule(
    total_steps: int,
    peak_lr: float,
    method: Literal["cosine", "linear", "constant"] = "cosine",
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> List[float]:
    """
    Create learning rate schedule for training with optional warmup.

    Args:
        total_steps: Total number of training steps
        peak_lr: Peak learning rate
        method: Learning rate schedule method. Options:
                - "cosine": Cosine annealing from peak_lr to min_lr
                - "linear": Linear decay from peak_lr to min_lr
                - "constant": Constant learning rate (peak_lr for all steps)
        warmup_steps: Number of warmup steps (linear warmup from 0 to peak_lr)
        min_lr: Minimum learning rate (floor for decay schedules)

    Returns:
        List of learning rates for each step

    Example:
        # Cosine schedule with warmup
        lrs = create_lr_schedule(100, 1e-4, method="cosine", warmup_steps=10)

        # Use with training loop
        for step, chunk in enumerate(chunk_trajectories(...)):
            train_sft(chunk, learning_rate=lrs[step])
    """
    learning_rates = []

    for step in range(total_steps):
        # Warmup phase: linear warmup from 0 to peak_lr
        if step < warmup_steps:
            lr = peak_lr * (step / warmup_steps)
        else:
            # Main schedule phase
            # Adjust step to be relative to post-warmup period
            adjusted_step = step - warmup_steps
            adjusted_total = total_steps - warmup_steps

            if method == "cosine":
                # Cosine annealing: lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(pi * t))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * adjusted_step / adjusted_total)
                )
            elif method == "linear":
                # Linear decay: lr = peak_lr - (peak_lr - min_lr) * (t / total)
                lr = peak_lr - (peak_lr - min_lr) * (adjusted_step / adjusted_total)
            elif method == "constant":
                # Constant learning rate
                lr = peak_lr
            else:
                raise ValueError(
                    f"Unknown method: {method}. Choose from: cosine, linear, constant"
                )

        learning_rates.append(lr)

    return learning_rates


def chunk_learning_rate(
    learning_rates: List[float],
    chunk_size: int,
) -> Generator[List[float], None, None]:
    """
    Chunk a list of learning rates into groups.

    Args:
        learning_rates: List of learning rate values
        chunk_size: Number of learning rates per chunk

    Yields:
        List of learning rates (chunk_size items, last chunk may be smaller)

    Example:
        lrs = create_lr_schedule(10, 1e-4)
        for lr_chunk in chunk_learning_rate(lrs, chunk_size=3):
            # lr_chunk has 3 learning rates (or fewer for last chunk)
            print(lr_chunk)  # [1e-5, 2e-5, 3e-5]
    """
    for i in range(0, len(learning_rates), chunk_size):
        yield learning_rates[i : i + chunk_size]
