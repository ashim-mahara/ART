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


def get_total_steps(traj_len: int, epochs: int, batch_size: int) -> int:
    """
    Calculate total number of training steps given dataset size, epochs, and batch size.

    Args:
        traj_len: Number of trajectories in the dataset
        epochs: Number of epochs to train
        batch_size: Number of trajectories per batch/step

    Returns:
        Total number of training steps

    Example:
        # 100 trajectories, 3 epochs, batch size of 10
        total_steps = get_total_steps(100, 3, 10)
        # Returns 30 (10 steps per epoch * 3 epochs)

        # With partial batch at end
        total_steps = get_total_steps(105, 3, 10)
        # Returns 33 (11 steps per epoch * 3 epochs)
    """
    steps_per_epoch = math.ceil(traj_len / batch_size)
    return steps_per_epoch * epochs


def iterate_trajectories(
    trajectories: List["Trajectory"],
    epochs: int,
    batch_size: int,
    chunk_size: int = 1,
    initial_step: int = 0,
) -> Generator[List["Trajectory"], None, None]:
    """
    Iterate over a list of trajectories for multiple epochs, yielding batches.
    Shuffles trajectories at the start of each epoch with a fixed seed for reproducibility.

    Args:
        trajectories: List of Trajectory objects
        epochs: Number of times to iterate over the list
        batch_size: Number of chunks per batch
        chunk_size: Number of trajectories per chunk. Defaults to 1.
        initial_step: The global step number to start from. Defaults to 0.
                      Useful for resuming training.

    Yields:
        List of trajectories (batch_size * chunk_size items)

    Example:
        # Load trajectories once
        trajs = [traj1, traj2, traj3]

        # Iterate 3 epochs, 2 trajectories per batch
        for batch in iterate_trajectories(trajs, epochs=3, batch_size=2):
            # batch is a list of 2 trajectories
            train_sft(batch, ...)

        # With chunk_size
        for batch in iterate_trajectories(trajs, epochs=3, batch_size=4, chunk_size=5):
            # batch is a list of 20 trajectories (4 chunks * 5 per chunk)
            pass

        # Resume from step 10
        for batch in iterate_trajectories(trajs, epochs=3, batch_size=2, initial_step=10):
            # Skips first 10 batches, starts from step 10
            pass
    """

    dataset_size = len(trajectories)
    if dataset_size == 0:
        return

    items_per_step = batch_size * chunk_size
    steps_per_epoch = math.ceil(dataset_size / items_per_step)

    for epoch in range(epochs):
        # Create indices and shuffle deterministically based on epoch
        indices = list(range(dataset_size))
        random.seed(epoch)
        random.shuffle(indices)

        for i in range(0, dataset_size, items_per_step):
            batch_index = i // items_per_step
            # Calculate global step number
            global_step = epoch * steps_per_epoch + batch_index

            # Skip if before initial_step
            if global_step < initial_step:
                continue

            batch_indices = indices[i : i + items_per_step]
            batch_items = [trajectories[idx] for idx in batch_indices]
            yield batch_items


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
