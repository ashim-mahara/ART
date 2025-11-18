import json
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, Generic, Iterable, List, TypeVar

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from art.trajectories import Trajectory

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
) -> Generator[List[List["Trajectory"]], None, None]:
    """
    Iterate over a list of trajectories for multiple epochs, yielding chunks of batches.
    Shuffles trajectories at the start of each epoch with a fixed seed for reproducibility.

    Args:
        trajectories: List of Trajectory objects
        epochs: Number of times to iterate over the list
        batch_size: Number of trajectories per batch (inner list size)
        chunk_size: Number of batches per chunk (outer list size). Defaults to 1.
        initial_step: The global step number to start from. Defaults to 0.
                      Useful for resuming training.

    Yields:
        List of lists of trajectories (chunk_size batches, each with batch_size trajectories)

    Example:
        # Load trajectories once
        trajs = [traj1, traj2, traj3, traj4]

        # Iterate 3 epochs, 2 trajectories per batch, 1 batch per chunk
        for chunk in iterate_trajectories(trajs, epochs=3, batch_size=2, chunk_size=1):
            # chunk is [[traj1, traj2]] or [[traj3, traj4]]
            train_sft(chunk, ...)

        # With chunk_size > 1
        for chunk in iterate_trajectories(trajs, epochs=3, batch_size=5, chunk_size=4):
            # chunk is a list of 4 batches, each batch has 5 trajectories
            # [[traj0-4], [traj5-9], [traj10-14], [traj15-19]]
            pass

        # Resume from step 10
        for chunk in iterate_trajectories(trajs, epochs=3, batch_size=2, chunk_size=1, initial_step=10):
            # Skips first 10 chunks, starts from step 10
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
            step_index = i // items_per_step
            # Calculate global step number
            global_step = epoch * steps_per_epoch + step_index

            # Skip if before initial_step
            if global_step < initial_step:
                continue

            step_indices = indices[i : i + items_per_step]

            # Structure as list of batches, where each batch has batch_size trajectories
            chunk: List[List["Trajectory"]] = []
            for batch_idx in range(0, len(step_indices), batch_size):
                batch_indices = step_indices[batch_idx : batch_idx + batch_size]
                batch = [trajectories[idx] for idx in batch_indices]
                chunk.append(batch)

            yield chunk


def iterate_file(
    file_path: str,
    epochs: int,
    batch_size: int,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    seed: int | None = 42,
) -> Generator[List["Trajectory"], None, None]:
    """
    Read JSONL file for each epoch, yielding batches of Trajectory objects.

    Each line should contain a dict with:
    - messages: List of chat messages
    - tools: Optional list of tools
    - reward: Optional reward (defaults to 0.0)
    - split: Optional split name (stored in metadata)
    - Any other fields will be stored in metadata

    Args:
        file_path: Path to JSONL file (one JSON object per line)
        epochs: Number of times to read through the file
        batch_size: Number of trajectories per batch. Defaults to 8.
                   Batches carry over across epochs.
        shuffle: Whether to shuffle trajectories. Defaults to True.
        shuffle_buffer_size: Size of shuffle buffer. Default: 10000.
                            Only used if shuffle=True.
        seed: Random seed for deterministic shuffling. Default: 42.
              Only used if shuffle=True.

    Yields:
        Batches of Trajectory objects (lists of size batch_size, last batch may be smaller)

    Raises:
        ValueError: If file_path does not end with .jsonl

    Example:
        # With shuffle and batching
        for batch in iterate_file("data.jsonl", epochs=3, batch_size=8):
            # batch is a list of 8 trajectories (or fewer for the last batch)
            process(batch)

        # No shuffle
        for batch in iterate_file("data.jsonl", epochs=3, batch_size=8, shuffle=False):
            process(batch)
    """
    from art.trajectories import Trajectory

    if not file_path.endswith(".jsonl"):
        raise ValueError(f"Only JSONL files are supported. Got: {file_path}")

    # Batch accumulator that carries over across epochs
    batch: List["Trajectory"] = []

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

                    # Once buffer is full, start yielding
                    if len(shuffle_buffer) >= shuffle_buffer_size:
                        idx = random.randint(0, len(shuffle_buffer) - 1)
                        batch.append(shuffle_buffer.pop(idx))

                        # Yield batch when it reaches batch_size
                        if len(batch) == batch_size:
                            yield batch
                            batch = []

            # Flush remaining items in shuffle buffer
            random.shuffle(shuffle_buffer)
            for traj in shuffle_buffer:
                batch.append(traj)

                # Yield batch when it reaches batch_size
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        else:
            # No shuffle - sequential reading
            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue

                    data = json.loads(line)
                    messages = data.get("messages", [])
                    tools = data.get("tools", None)

                    batch.append(Trajectory(
                        messages_and_choices=messages,
                        tools=tools if tools else None,
                        reward=0.0
                    ))

                    # Yield batch when it reaches batch_size
                    if len(batch) == batch_size:
                        yield batch
                        batch = []

    # Yield any remaining trajectories in the final batch
    if batch:
        yield batch
