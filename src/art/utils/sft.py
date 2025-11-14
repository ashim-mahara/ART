"""Utilities for supervised fine-tuning (SFT)."""

import math
from typing import TYPE_CHECKING, Generator, List, Literal

if TYPE_CHECKING:
    from art.model import TrainableModel


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


def iterate_learning_rates(
    learning_rates: List[float],
    chunk_size: int,
    initial_step: int = 0,
) -> Generator[List[float], None, None]:
    """
    Iterate over learning rates in chunks, with support for resuming from a specific step.

    Args:
        learning_rates: List of learning rate values
        chunk_size: Number of learning rates per chunk
        initial_step: The step number to start from. Defaults to 0.
                      Useful for resuming training.

    Yields:
        List of learning rates (chunk_size items, last chunk may be smaller)

    Example:
        lrs = create_lr_schedule(10, 1e-4)
        for lr_chunk in iterate_learning_rates(lrs, chunk_size=3):
            # lr_chunk has 3 learning rates (or fewer for last chunk)
            # Yields: [lr0, lr1, lr2], [lr3, lr4, lr5], [lr6, lr7, lr8], [lr9]

        # Resume from step 5
        for lr_chunk in iterate_learning_rates(lrs, chunk_size=3, initial_step=5):
            # Starts from learning rate 5: yields [lr5, lr6, lr7], [lr8, lr9]
            pass
    """
    for i in range(initial_step, len(learning_rates), chunk_size):
        yield learning_rates[i : i + chunk_size]


async def train_sft_from_file(
    model: "TrainableModel",
    file_path: str,
    epochs: int,
    learning_rate: float,
    batch_size: int = 8,
) -> None:
    """
    Convenience function to train a model with SFT from a JSONL file.

    Args:
        model: TrainableModel to train
        file_path: Path to JSONL file containing trajectories
        epochs: Number of epochs to train
        learning_rate: Peak learning rate (uses cosine schedule)
        batch_size: Number of trajectories per batch/step. Defaults to 8.

    Example:
        await train_sft_from_file(
            model=model,
            file_path="data.jsonl",
            epochs=3,
            learning_rate=1e-5,
        )
    """
    from art.types import SFTConfig
    from art.utils.iterate_dataset import get_file_row_count, get_total_steps, iterate_file

    # Calculate total steps
    num_trajectories = get_file_row_count(file_path)
    total_steps = get_total_steps(num_trajectories, epochs, batch_size)

    # Set warmup steps: 10% of total steps, capped at 1000
    warmup_steps = min(total_steps // 10, 1000)

    # Create cosine learning rate schedule with warmup
    learning_rates = create_lr_schedule(
        total_steps=total_steps,
        peak_lr=learning_rate,
        method="cosine",
        warmup_steps=warmup_steps,
    )

    # Create SFT config with shuffling enabled
    config = SFTConfig(learning_rate=learning_rates, batch_size=batch_size, shuffle=True)

    # Train the model
    await model.train_sft(
        trajectories=iterate_file(file_path, epochs=epochs),
        config=config
    )
