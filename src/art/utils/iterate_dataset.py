import json
import math
import random
from dataclasses import dataclass
from typing import Any, Generator, Generic, Iterable, List, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


@dataclass
class DatasetBatch(Generic[T]):
    """Container for dataset batch information."""

    items: List[T]
    step: int
    epoch: int
    epoch_step: int


def iterate_dataset(
    dataset: List[T],
    groups_per_step: int = 1,
    num_epochs: int = 1,
    initial_step: int = 0,
    use_tqdm: bool = True,
) -> Generator[DatasetBatch[T], None, None]:
    """
    Generates batches from a dataset over multiple epochs with deterministic shuffling.

    Args:
        dataset: The list of data items.
        groups_per_step: The size of each batch. Defaults to 1.
        num_epochs: The number of times to iterate over the dataset. Defaults to 1.
        initial_step: The global step number to start from. Defaults to 0.
                           Useful for resuming training.
        use_tqdm: Whether to display a progress bar. Defaults to True.

    Yields:
        DatasetBatch: A dataclass containing:
        - items (List[T]): The list of items for the current batch.
        - epoch (int): The current epoch number (0-indexed).
        - global_step (int): The overall step number across all epochs.
        - epoch_step (int): The step number within the current epoch (0-indexed).
    """
    dataset_size = len(dataset)
    if dataset_size == 0:
        return

    steps_per_epoch = math.ceil(dataset_size / groups_per_step)
    total_steps = steps_per_epoch * num_epochs

    progress_bar = None
    if use_tqdm:
        progress_bar = tqdm(
            initial=initial_step,
            total=total_steps,
            desc="Iterating dataset",
            unit="batch",
        )

    for epoch in range(num_epochs):
        # Create indices and shuffle deterministically based on epoch
        indices = list(range(dataset_size))
        random.seed(epoch)  # Ensure shuffling is the same for a given epoch
        random.shuffle(indices)

        for i in range(0, dataset_size, groups_per_step):
            epoch_step = i // groups_per_step
            # Calculate global step number before skipping
            global_step = epoch * steps_per_epoch + epoch_step

            if global_step < initial_step:
                # If using tqdm, we still need to update it even when skipping
                if progress_bar:
                    # Ensure the progress bar reflects the skipped steps accurately
                    # by setting the description or just updating.
                    # Setting n directly might be complex if initial_step > 0.
                    # A simple update() works if the bar was initialized correctly.
                    pass  # tqdm handles the initial value
                continue

            batch_indices = indices[i : i + groups_per_step]
            items = [dataset[idx] for idx in batch_indices]
            yield DatasetBatch(
                items=items, epoch=epoch, step=global_step, epoch_step=epoch_step
            )

            # Update progress bar after yielding
            if progress_bar:
                progress_bar.update(1)

    if progress_bar:
        progress_bar.close()


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


def iterate_trajectories(
    trajectories: List["Trajectory"], epochs: int
) -> Generator["Trajectory", None, None]:
    """
    Iterate over a list of trajectories for multiple epochs.

    Args:
        trajectories: List of Trajectory objects
        epochs: Number of times to iterate over the list

    Yields:
        Trajectory objects from the list

    Example:
        # Load trajectories once
        trajs = [traj1, traj2, traj3]

        # Iterate 3 times
        for traj in iterate_trajectories(trajs, epochs=3):
            # Process trajectory
            pass
    """
    for _ in range(epochs):
        for trajectory in trajectories:
            yield trajectory


def iterate_file(file_path: str, epochs: int) -> Generator["Trajectory", None, None]:
    """
    Read JSONL file for each epoch, yielding Trajectory objects.

    Each line should contain a dict with:
    - messages: List of chat messages
    - tools: Optional list of tools
    - reward: Optional reward (defaults to default_reward)
    - split: Optional split name (stored in metadata)
    - Any other fields will be stored in metadata

    Args:
        file_path: Path to JSONL file (one JSON object per line)
        epochs: Number of times to read through the file
        default_reward: Default reward value if not specified in data

    Yields:
        Trajectory objects parsed from the file

    Raises:
        ValueError: If file_path does not end with .jsonl
    """
    from art.trajectories import Trajectory

    if not file_path.endswith(".jsonl"):
        raise ValueError(f"Only JSONL files are supported. Got: {file_path}")

    for _ in range(epochs):
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)

                # Extract messages and convert to messages_and_choices format
                messages = data.get("messages", [])
                tools = data.get("tools", None)

                # Create trajectory
                yield Trajectory(
                    messages_and_choices=messages,
                    tools=tools if tools else None,
                    reward=0.0
                )


def chunk_trajectories(
    trajectories: Iterable["Trajectory"],
    batch_size: int,
    chunk_size: int,
    shuffle_buffer_size: int = 10000,
    seed: int | None = None,
) -> Generator[List["Trajectory"], None, None]:
    """
    Chunk trajectories from an iterable into batches.

    Args:
        trajectories: Iterable of Trajectory objects (can be list, generator, etc.)
        batch_size: Number of chunks per batch
        chunk_size: Number of trajectories per chunk
        shuffle_buffer_size: Size of shuffle buffer. Default: 10000 (~200MB-1GB).
                            Set to 0 for no shuffle (sequential order).
                            Recommended: 1000-50000 depending on available RAM.
                            Larger buffer = better shuffle quality but more memory.
        seed: Random seed for deterministic shuffling. Default: None (non-deterministic).
              Set to an integer for reproducible results.

    Yields:
        List of trajectories (batch_size * chunk_size items)

    Example:
        # Default shuffle (buffer_size=10000, random)
        chunk_trajectories(iterate_file("data.jsonl", epochs=1), 4, 8)

        # Deterministic shuffle (reproducible)
        chunk_trajectories(iterate_file("data.jsonl", epochs=1), 4, 8, seed=42)

        # No shuffle
        chunk_trajectories(iterate_file("data.jsonl", epochs=1), 4, 8, shuffle_buffer_size=0)

        # Larger buffer for better shuffle
        chunk_trajectories(iterate_file("data.jsonl", epochs=1), 4, 8, shuffle_buffer_size=50000, seed=42)
    """
    items_per_batch = batch_size * chunk_size

    if shuffle_buffer_size > 0:
        # Set seed for deterministic shuffling
        if seed is not None:
            random.seed(seed)

        # Buffer-based shuffle
        shuffle_buffer: List["Trajectory"] = []
        batch_items = []

        for trajectory in trajectories:
            shuffle_buffer.append(trajectory)

            # Once buffer is full, start yielding
            if len(shuffle_buffer) >= shuffle_buffer_size:
                # Pop random item from buffer
                idx = random.randint(0, len(shuffle_buffer) - 1)
                traj = shuffle_buffer.pop(idx)

                batch_items.append(traj)

                if len(batch_items) == items_per_batch:
                    yield batch_items
                    batch_items = []

        # Flush remaining items in shuffle buffer
        random.shuffle(shuffle_buffer)
        for traj in shuffle_buffer:
            batch_items.append(traj)

            if len(batch_items) == items_per_batch:
                yield batch_items
                batch_items = []

        # Yield any remaining items as a final batch
        if batch_items:
            yield batch_items
    else:
        # No shuffle - simple batching
        batch_items = []
        for trajectory in trajectories:
            batch_items.append(trajectory)

            if len(batch_items) == items_per_batch:
                yield batch_items
                batch_items = []

        # Yield any remaining items as a final batch
        if batch_items:
            yield batch_items
