"""Utilities for supervised fine-tuning (SFT)."""

import json
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, List, Literal

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from art.model import TrainableModel
    from art.trajectories import Trajectory
    from art.types import SFTConfig


@dataclass
class SFTDatasetChunk:
    """Container for SFT dataset chunk with trajectories, config, and step information."""

    trajectories: List["Trajectory"]
    config: "SFTConfig"
    step: int
    epoch: int
    epoch_step: int

def get_file_row_count(file_path: str) -> int:
    """
    Count the number of non-empty rows in a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        Number of non-empty lines in the file

    Raises:
        ValueError: If file_path does not end with .jsonl

    Example:
        count = get_file_row_count("data.jsonl")
        print(f"Dataset has {count} items")
    """
    if not file_path.endswith(".jsonl"):
        raise ValueError(f"Only JSONL files are supported. Got: {file_path}")

    count = 0
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def create_lr_schedule(
    total_steps: int,
    peak_lr: float,
    method: Literal["cosine", "linear", "constant"] = "linear",
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


def create_sft_dataset_iterator(
    trajectories: List["Trajectory"],
    epochs: int = 1,
    batch_size: int = 1,
    chunk_size: int = 50,
    peak_lr: float = 2e-4,
    schedule_type: Literal["cosine", "linear", "constant"] = "linear",
    warmup_ratio: float = 0.1,
    initial_step: int = 0,
    use_tqdm: bool = True,
) -> Generator[SFTDatasetChunk, None, None]:
    """
    Create an iterator that yields SFT dataset chunks with trajectories, config, and step info.

    Combines trajectory batching with learning rate scheduling. Yields SFTDatasetChunk objects
    containing flattened trajectories, SFTConfig with learning rates, and step tracking info.

    Args:
        trajectories: List of Trajectory objects to train on
        epochs: Number of times to iterate over the trajectories. Default: 1
        batch_size: Number of trajectories per batch. Default: 1
        chunk_size: Number of batches per chunk. Default: 50
        peak_lr: Peak learning rate. Default: 5e-5
        schedule_type: Learning rate schedule type ("cosine", "linear", "constant"). Default: "linear"
        warmup_ratio: Ratio of total steps to use for warmup (0.0 to 1.0). Default: 0.1
        initial_step: The global chunk step to start from. Default: 0.
                      Useful for resuming training.
        use_tqdm: Whether to display a progress bar. Default: True

    Yields:
        SFTDatasetChunk containing:
            - trajectories: Flattened list of trajectories (chunk_size * batch_size trajectories)
            - config: SFTConfig with custom_lr_schedule containing learning rates for each batch
            - step: Global step number across all epochs
            - epoch: Current epoch number (0-indexed)
            - epoch_step: Step number within current epoch (0-indexed)

    Example:
        trajectories = [traj1, traj2, ..., traj100]

        # Create SFT dataset iterator with linear schedule
        for chunk in create_sft_dataset_iterator(
            trajectories=trajectories,
            epochs=3,
            batch_size=4,
            chunk_size=10,
            peak_lr=1e-4,
            schedule_type="linear",
            warmup_ratio=0.1,
        ):
            # chunk.trajectories is a flat list of 40 trajectories (10 batches * 4 per batch)
            # chunk.config.custom_lr_schedule is a list of 10 learning rates (one per batch)
            # chunk.config.batch_size is 4
            # chunk.step is global step number
            # chunk.epoch is current epoch
            # chunk.epoch_step is step within epoch
            train_sft(chunk.trajectories, chunk.config)

        # Resume from chunk step 5
        for chunk in create_sft_dataset_iterator(
            trajectories=trajectories,
            epochs=3,
            batch_size=4,
            chunk_size=10,
            initial_step=5,
        ):
            # Starts from chunk step 5
            pass
    """
    from art.types import SFTConfig

    dataset_size = len(trajectories)
    if dataset_size == 0:
        return

    # Calculate total batch steps (one step per batch)
    batches_per_epoch = math.ceil(dataset_size / batch_size)
    total_batch_steps = batches_per_epoch * epochs

    # Calculate warmup steps
    warmup_steps = int(total_batch_steps * warmup_ratio)

    # Create learning rate schedule (one LR per batch)
    custom_lr_schedule = create_lr_schedule(
        total_steps=total_batch_steps,
        peak_lr=peak_lr,
        method=schedule_type,
        warmup_steps=warmup_steps,
        min_lr=0.0,
    )

    # Calculate chunk iteration parameters
    items_per_chunk = batch_size * chunk_size
    chunks_per_epoch = math.ceil(dataset_size / items_per_chunk)
    total_steps = chunks_per_epoch * epochs

    progress_bar = None
    if use_tqdm:
        progress_bar = tqdm(
            initial=initial_step,
            total=total_steps,
            desc="Training SFT",
            unit="chunk",
        )

    for epoch in range(epochs):
        # Create indices and shuffle deterministically based on epoch
        indices = list(range(dataset_size))
        random.seed(epoch)
        random.shuffle(indices)

        for chunk_idx in range(chunks_per_epoch):
            # Calculate step numbers
            epoch_step = chunk_idx
            global_step = epoch * chunks_per_epoch + chunk_idx

            # Skip if before initial_step
            if global_step < initial_step:
                continue

            # Get indices for this chunk
            chunk_start = chunk_idx * items_per_chunk
            chunk_end = min(chunk_start + items_per_chunk, dataset_size)
            step_indices = indices[chunk_start:chunk_end]

            # Flatten trajectories for this chunk
            chunk_trajectories: List["Trajectory"] = [
                trajectories[idx] for idx in step_indices
            ]

            # Calculate learning rates for each batch in this chunk
            chunk_lrs: List[float] = []
            num_batches_in_chunk = math.ceil(len(step_indices) / batch_size)

            for batch_idx in range(num_batches_in_chunk):
                # Calculate global batch step
                global_batch_step = epoch * batches_per_epoch + (chunk_start // batch_size) + batch_idx
                chunk_lrs.append(custom_lr_schedule[global_batch_step])

            # Create SFTConfig with custom learning rate schedule
            config = SFTConfig(
                batch_size=batch_size,
                custom_lr_schedule=chunk_lrs,
            )

            yield SFTDatasetChunk(
                trajectories=chunk_trajectories,
                config=config,
                step=global_step,
                epoch=epoch,
                epoch_step=epoch_step,
            )

            # Update progress bar after yielding
            if progress_bar:
                progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

def iterate_file(
    file_path: str,
    epochs: int,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    seed: int | None = 42,
) -> Generator["Trajectory", None, None]:
    """
    Read JSONL file for each epoch, yielding individual Trajectory objects.

    Completes reading the entire file for one epoch before starting the next epoch.
    This ensures all trajectories from epoch N are yielded before any from epoch N+1.

    Each line should contain a dict with:
    - messages: List of chat messages
    - tools: Optional list of tools
    - reward: Optional reward (defaults to 0.0)
    - split: Optional split name (stored in metadata)
    - Any other fields will be stored in metadata

    Args:
        file_path: Path to JSONL file (one JSON object per line)
        epochs: Number of times to read through the file
        shuffle: Whether to shuffle trajectories. Defaults to True.
        shuffle_buffer_size: Size of shuffle buffer for streaming shuffle. Default: 10000.
                            Only used if shuffle=True.
        seed: Random seed for deterministic shuffling. Default: 42.
              Only used if shuffle=True.

    Yields:
        Individual Trajectory objects

    Raises:
        ValueError: If file_path does not end with .jsonl

    Example:
        # With shuffle
        for trajectory in iterate_file("data.jsonl", epochs=3, shuffle=True):
            # trajectory is a single Trajectory object
            process(trajectory)

        # No shuffle
        for trajectory in iterate_file("data.jsonl", epochs=3, shuffle=False):
            process(trajectory)
    """
    from art.trajectories import Trajectory

    if not file_path.endswith(".jsonl"):
        raise ValueError(f"Only JSONL files are supported. Got: {file_path}")

    for epoch in range(epochs):
        if shuffle and seed is not None:
            random.seed(seed + epoch)

        if shuffle:
            # Streaming shuffle with buffer
            shuffle_buffer: List["Trajectory"] = []

            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    messages = data.get("messages", [])
                    tools = data.get("tools", None)

                    traj = Trajectory(
                        messages_and_choices=messages,
                        tools=tools if tools else None,
                        reward=0.0
                    )

                    shuffle_buffer.append(traj)

                    # Once buffer is full, start yielding randomly
                    if len(shuffle_buffer) >= shuffle_buffer_size:
                        idx = random.randint(0, len(shuffle_buffer) - 1)
                        yield shuffle_buffer.pop(idx)

            # Flush remaining items in shuffle buffer at end of epoch
            random.shuffle(shuffle_buffer)
            for traj in shuffle_buffer:
                yield traj
        else:
            # No shuffle - sequential reading
            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    messages = data.get("messages", [])
                    tools = data.get("tools", None)

                    yield Trajectory(
                        messages_and_choices=messages,
                        tools=tools if tools else None,
                        reward=0.0
                    )


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

    # Calculate total steps - batches carry over across epochs
    num_trajectories = get_file_row_count(file_path)
    total_steps = math.ceil((num_trajectories * epochs) / batch_size)

    # Set warmup steps: 10% of total steps, capped at 1000
    warmup_steps = min(total_steps // 10, 1000)

    # Create cosine learning rate schedule with warmup
    custom_lr_schedule = create_lr_schedule(
        total_steps=total_steps,
        peak_lr=learning_rate,
        method="linear",
        warmup_steps=warmup_steps,
    )

    # Create SFT config with shuffling enabled
    config = SFTConfig(custom_lr_schedule=custom_lr_schedule, batch_size=batch_size)

    # Train the model
    await model.train_sft(
        trajectories=iterate_file(file_path, epochs=epochs),
        config=config
    )
