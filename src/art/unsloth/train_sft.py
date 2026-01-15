"""Training utilities for Supervised Fine-Tuning (SFT)."""

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Iterator

import nest_asyncio
import torch
from trl import SFTTrainer

if TYPE_CHECKING:
    from ..preprocessing.tokenize_sft import SFTBatch

nest_asyncio.apply()


async def train_sft(
    trainer: SFTTrainer,
    input_queue: asyncio.Queue["SFTBatch"],
    results_queue: asyncio.Queue[dict[str, float]],
) -> None:
    """
    Train an SFT model using batches from a queue.

    Args:
        trainer: TRL SFTTrainer instance
        input_queue: Queue containing SFTBatch objects
        results_queue: Queue for training metrics/results
    """
    _get_batch_samples = trainer.get_batch_samples
    _log = trainer.log

    trainer.get_batch_samples = get_batch_samples_fn(trainer, input_queue)
    trainer.log = get_log_fn(trainer, results_queue)

    # Ensure we have a metrics container in the expected format
    try:
        is_dict = isinstance(getattr(trainer, "_metrics", None), dict)
        is_train_dict = is_dict and isinstance(trainer._metrics.get("train"), dict)
    except Exception:
        is_train_dict = False
    if not is_train_dict:
        trainer._metrics = {"train": defaultdict(list)}

    try:
        trainer.train()
    finally:
        trainer.get_batch_samples = _get_batch_samples
        trainer.log = _log


def get_batch_samples_fn(
    trainer: SFTTrainer,
    input_queue: asyncio.Queue["SFTBatch"],
) -> Callable[..., tuple[list[dict[str, torch.Tensor]], torch.Tensor]]:
    """
    Create a get_batch_samples function that:
    1. Reads SFTBatch from queue
    2. Sets learning rate from batch
    3. Sets gradient accumulation steps
    4. Returns batch samples and num_items_in_batch as tensor
    """

    def get_batch_samples(
        epoch_iterator: Iterator,
        num_batches: int,
        device: torch.device | str | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        """
        Override get_batch_samples to read from queue instead of epoch_iterator.

        Returns:
            tuple of (batch_samples, num_items_in_batch as tensor int)
        """

        # Read SFTBatch from queue asynchronously
        async def get_sft_batch() -> "SFTBatch":
            return await input_queue.get()

        # Get the batch from queue
        sft_batch: "SFTBatch" = asyncio.run(get_sft_batch())

        # Set learning rate for this batch
        if optimizer := trainer.optimizer:
            optimizer = getattr(optimizer, "optimizer", optimizer)
            if param_groups := getattr(optimizer, "param_groups"):
                for param_group in param_groups:
                    param_group["lr"] = sft_batch.learning_rate

        # Set gradient accumulation steps to number of trajectories
        # We're doing micro-batch size 1, so accumulate across all trajectories
        if hasattr(trainer.args, "gradient_accumulation_steps"):
            trainer.args.gradient_accumulation_steps = sft_batch.num_trajectories

        # Convert each trajectory to a separate sample for micro-batching
        # Trainer will process each sample individually and accumulate gradients
        batch_samples = []
        for trajectory_tensor in sft_batch.trajectory_tensors:
            # Move each trajectory's tensors to device
            sample = {
                key: tensor.to(device) for key, tensor in trajectory_tensor.items()
            }
            batch_samples.append(sample)

        # Return batch samples and num_items_in_batch as tensor (on device)
        num_items_in_batch = torch.tensor(
            sft_batch.num_trajectories, dtype=torch.long, device=device
        )

        return batch_samples, num_items_in_batch

    return get_batch_samples


def get_log_fn(
    trainer: SFTTrainer,
    results_queue: asyncio.Queue[dict[str, float]],
) -> Callable[..., None]:
    """
    Create a logging function that sends metrics to the results queue.
    Same pattern as GRPO trainer.
    """

    def log(logs: dict[str, float], start_time: float | None = None) -> None:
        """Log metrics and send to results queue."""
        metrics = {
            key: sum(val) / len(val) for key, val in trainer._metrics["train"].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        logs.pop("learning_rate", None)
        results_queue.put_nowait(logs)
        trainer._metrics["train"].clear()

    return log
