"""Utilities for supervised fine-tuning (SFT)."""

import json
import math
import random
from typing import TYPE_CHECKING, Generator, List, Literal

if TYPE_CHECKING:
    from art.dev import SFTConfig as DevSFTConfig
    from art.model import TrainableModel
    from art.trajectories import Trajectory


def _parse_jsonl_line(line: str) -> "Trajectory":
    """Parse a JSONL line into a Trajectory object.

    Args:
        line: A JSON string containing trajectory data with 'messages' and optional 'tools'.

    Returns:
        A Trajectory object with the parsed data.
    """
    from art.trajectories import Trajectory

    data = json.loads(line)
    return Trajectory(
        messages_and_choices=data.get("messages", []),
        tools=data.get("tools"),
        reward=0.0,
    )


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
    if total_steps <= 0:
        return []

    learning_rates = []
    decay_steps = total_steps - warmup_steps

    for step in range(total_steps):
        if step < warmup_steps:
            # Warmup: linear ramp from min_lr to peak_lr
            # Use (step + 1) so first step has lr > 0
            lr = min_lr + (peak_lr - min_lr) * ((step + 1) / warmup_steps)
        else:
            # Decay phase: progress goes from 0 to 1
            progress = (
                (step - warmup_steps) / (decay_steps - 1) if decay_steps > 1 else 0
            )
            if method == "cosine":
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
            elif method == "linear":
                lr = peak_lr - (peak_lr - min_lr) * progress
            elif method == "constant":
                lr = peak_lr
            else:
                raise ValueError(
                    f"Unknown method: {method}. Choose from: cosine, linear, constant"
                )

        learning_rates.append(lr)

    return learning_rates


def prepare_sft_dataset(
    trajectories: List["Trajectory"],
    epochs: int,
    batch_size: int,
    peak_lr: float,
    schedule_type: Literal["cosine", "linear", "constant"],
    warmup_ratio: float = 0.1,
    shuffle_seed: int = 42,
    initial_step: int = 0,
) -> tuple[List["Trajectory"], List[float]]:
    """
    Prepare all trajectories for training and calculate learning rates.

    This function handles epoch management by shuffling and concatenating
    trajectories for all epochs, and calculates the learning rate schedule.

    Args:
        trajectories: List of trajectories to train on
        epochs: Number of epochs (passes through the dataset)
        batch_size: Number of trajectories per batch
        peak_lr: Peak learning rate for the schedule
        schedule_type: Learning rate schedule type ("cosine", "linear", "constant")
        warmup_ratio: Ratio of total steps to use for warmup (0.0 to 1.0)
        shuffle_seed: Random seed for deterministic shuffling (default: 42)
        initial_step: Starting step for resuming training (skips first N batches)

    Returns:
        Tuple of:
        - All trajectories for all epochs (shuffled per epoch, concatenated)
        - List of learning rates (one per batch, starting from initial_step)

    Example:
        trajectories = [traj1, traj2, traj3, traj4]
        all_trajs, learning_rates = prepare_sft_dataset(
            trajectories=trajectories,
            epochs=3,
            batch_size=2,
            peak_lr=2e-4,
            schedule_type="linear",
        )
        # all_trajs has 12 trajectories (4 * 3 epochs)
        # learning_rates has 6 values (ceil(12 / 2) batches)
    """
    if len(trajectories) == 0:
        return [], []

    # Calculate total batches across all epochs
    total_trajectories = len(trajectories) * epochs
    total_batches = math.ceil(total_trajectories / batch_size)
    warmup_steps = int(total_batches * warmup_ratio)

    # Calculate learning rates for all batches
    full_schedule = create_lr_schedule(
        total_steps=total_batches,
        peak_lr=peak_lr,
        method=schedule_type,
        warmup_steps=warmup_steps,
    )

    # Slice the schedule starting from initial_step (for resuming)
    learning_rates = full_schedule[initial_step:]

    # Prepare trajectories for all epochs with shuffling
    all_trajectories: List["Trajectory"] = []
    indices = list(range(len(trajectories)))

    for epoch in range(epochs):
        # Deterministic shuffle per epoch (different seed per epoch)
        epoch_indices = indices.copy()
        random.Random(shuffle_seed + epoch).shuffle(epoch_indices)
        for idx in epoch_indices:
            all_trajectories.append(trajectories[idx])

    # Skip trajectories for initial_step (for resuming)
    start_trajectory_idx = initial_step * batch_size
    if start_trajectory_idx > 0:
        all_trajectories = all_trajectories[start_trajectory_idx:]

    return all_trajectories, learning_rates


def iterate_file(
    file_path: str,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    seed: int | None = 42,
) -> Generator["Trajectory", None, None]:
    """
    Read JSONL file and yield individual Trajectory objects.

    Each line should contain a dict with:
    - messages: List of chat messages
    - tools: Optional list of tools
    - reward: Optional reward (defaults to 0.0)
    - split: Optional split name (stored in metadata)
    - Any other fields will be stored in metadata

    Args:
        file_path: Path to JSONL file (one JSON object per line)
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
        for trajectory in iterate_file("data.jsonl", shuffle=True):
            # trajectory is a single Trajectory object
            process(trajectory)

        # No shuffle
        for trajectory in iterate_file("data.jsonl", shuffle=False):
            process(trajectory)
    """
    if not file_path.endswith(".jsonl"):
        raise ValueError(f"Only JSONL files are supported. Got: {file_path}")

    # Use local Random instance to avoid modifying global random state
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    if shuffle:
        # Streaming shuffle with buffer
        shuffle_buffer: List["Trajectory"] = []

        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                traj = _parse_jsonl_line(line)
                shuffle_buffer.append(traj)

                # Once buffer is full, start yielding randomly
                if len(shuffle_buffer) >= shuffle_buffer_size:
                    idx = rng.randint(0, len(shuffle_buffer) - 1)
                    yield shuffle_buffer.pop(idx)

        # Flush remaining items in shuffle buffer
        rng.shuffle(shuffle_buffer)
        for traj in shuffle_buffer:
            yield traj
    else:
        # No shuffle - sequential reading
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                yield _parse_jsonl_line(line)


async def train_sft_from_file(
    model: "TrainableModel",
    file_path: str,
    epochs: int = 1,
    batch_size: int = 1,
    peak_lr: float = 2e-4,
    schedule_type: Literal["cosine", "linear", "constant"] = "linear",
    warmup_ratio: float = 0.1,
    initial_step: int = 0,
    _config: "DevSFTConfig | None" = None,
    verbose: bool = False,
) -> None:
    """
    Train a model using supervised fine-tuning from a JSONL file.

    This function loads trajectories from a file, handles epoch management with
    shuffling, calculates the learning rate schedule, and makes a single train_sft
    call with all data. This allows training to continue even if the client
    disconnects after the call is made.

    Args:
        model: The TrainableModel to fine-tune. Must be registered with a backend.
        file_path: Path to JSONL file containing training data. Each line should have:
                   - messages: List of chat messages
                   - tools: Optional list of tools
        epochs: Number of times to iterate over the dataset. Default: 1
        batch_size: Number of trajectories per batch (one weight update per batch). Default: 1
        peak_lr: Peak learning rate. Default: 2e-4
        schedule_type: Learning rate schedule type ("cosine", "linear", "constant"). Default: "linear"
        warmup_ratio: Ratio of total steps to use for warmup (0.0 to 1.0). Default: 0.1
        initial_step: The global training step (batch) to start from. Default: 0.
                      Useful for resuming training.
        _config: Additional experimental configuration. Use at your own risk.
        verbose: Whether to print verbose output. Default: False

    Example:
        import art
        from art.local import LocalBackend
        from art.utils.sft import train_sft_from_file

        async def main():
            backend = LocalBackend()
            model = art.TrainableModel(
                name="my-model",
                project="my-project",
                base_model="Qwen/Qwen2.5-7B-Instruct",
            )
            await model.register(backend)

            # Train with linear decay schedule
            await train_sft_from_file(
                model=model,
                file_path="data/train.jsonl",
                epochs=3,
                batch_size=4,
                peak_lr=2e-4,
                schedule_type="linear",
            )

            # Train with cosine schedule and warmup
            await train_sft_from_file(
                model=model,
                file_path="data/train.jsonl",
                epochs=1,
                batch_size=2,
                peak_lr=1e-4,
                schedule_type="cosine",
                warmup_ratio=0.1,
            )
    """
    from art.types import SFTConfig

    # Load all trajectories into memory
    trajectories = list(iterate_file(file_path, shuffle=False))

    if verbose:
        print(f"Loaded {len(trajectories)} trajectories from {file_path}")

    if len(trajectories) == 0:
        if verbose:
            print("No trajectories to train on")
        return

    # Prepare dataset: shuffle per epoch, concatenate, and calculate learning rates
    all_trajectories, learning_rates = prepare_sft_dataset(
        trajectories=trajectories,
        epochs=epochs,
        batch_size=batch_size,
        peak_lr=peak_lr,
        schedule_type=schedule_type,
        warmup_ratio=warmup_ratio,
        initial_step=initial_step,
    )

    if verbose:
        total_batches = len(learning_rates)
        print(f"Prepared {len(all_trajectories)} trajectories for {epochs} epoch(s)")
        print(f"Total batches: {total_batches}, batch_size: {batch_size}")
        print(f"Learning rate schedule: {schedule_type}, peak_lr: {peak_lr}")

    # Create config with per-batch learning rates
    config = SFTConfig(
        learning_rate=learning_rates,
        batch_size=batch_size,
    )

    # Single train_sft call with all trajectories
    await model.train_sft(
        all_trajectories,
        config,
        _config=_config,
        verbose=verbose,
    )
