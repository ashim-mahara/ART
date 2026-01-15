"""Tokenization utilities for Supervised Fine-Tuning (SFT)."""

from dataclasses import dataclass
import math
from typing import Any, Generator, cast

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Import Unsloth Zoo utilities for robust token matching
# Source: https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/dataset_utils.py
# These functions handle edge cases with tokenization (newlines, spaces, etc.)
import unsloth  # Must import first to set UNSLOTH_IS_PRESENT env var
from unsloth_zoo.dataset_utils import _find_common_token_ids

from ..trajectories import Trajectory


@dataclass
class SFTBatch:
    """A batch of tokenized trajectories for supervised fine-tuning.

    Attributes:
        trajectory_tensors: List of tensor dictionaries, one per trajectory.
                           Each dict contains 'input_ids', 'attention_mask', and 'labels'.
        learning_rate: Learning rate to use for this batch.
        num_trajectories: Number of trajectories in this batch.
        num_trainable_tokens: Total number of tokens being trained on (labels != -100).
    """

    trajectory_tensors: list[dict[str, torch.Tensor]]
    learning_rate: float
    num_trajectories: int
    num_trainable_tokens: int


def tokenize_sft_batches(
    trajectories: list[Trajectory],
    batch_size: int,
    learning_rates: list[float],
    tokenizer: PreTrainedTokenizerBase,
    instruction_part: str,
    response_part: str,
) -> Generator[SFTBatch, None, None]:
    """
    Tokenize trajectories into batches for supervised fine-tuning.

    Args:
        trajectories: Flat list of trajectories
        batch_size: Number of trajectories per batch
        learning_rates: Learning rate for each batch
        tokenizer: Tokenizer to use for encoding
        instruction_part: Instruction template part (e.g., "User:")
        response_part: Response template part (e.g., "Assistant:")

    Yields:
        SFTBatch object containing:
            - trajectory_tensors: List of tensors for each trajectory
            - learning_rate: Learning rate for this batch
            - num_trajectories: Number of trajectories in this batch
            - num_trainable_tokens: Total number of trainable tokens
    """
    # Validate inputs
    num_trajectories = len(trajectories)
    num_learning_rates = len(learning_rates)
    expected_num_batches = math.ceil(num_trajectories / batch_size)

    if num_learning_rates != expected_num_batches:
        raise ValueError(
            f"Mismatch between trajectories and learning_rates: "
            f"{num_trajectories} trajectories with batch_size={batch_size} "
            f"yields {expected_num_batches} batches, but got {num_learning_rates} learning_rates"
        )

    # Get most common tokens using Unsloth approach
    Q_must, Q_left, Q_right = _find_common_token_ids(
        instruction_part, tokenizer, force_match=False
    )
    A_must, A_left, A_right = _find_common_token_ids(
        response_part, tokenizer, force_match=False
    )

    # Store temporary stuff
    A_first = A_must[0]
    len_A_must = len(A_must)
    A_left_reversed = A_left[::-1]
    A_right_forward = A_right

    Q_first = Q_must[0]
    len_Q_must = len(Q_must)
    Q_left_reversed = Q_left[::-1]
    Q_right_forward = Q_right

    def _train_on_responses_only(input_ids: list[int]) -> list[int]:
        """Unsloth-based implementation for marking trainable tokens."""
        n = len(input_ids)
        labels = [-100] * n
        n_minus_1 = n - 1
        j = 0

        while j < n:
            # Find <assistant>
            if (input_ids[j] == A_first) and (
                input_ids[j : (k := j + len_A_must)] == A_must
            ):
                # Now backtrack to get previous optional tokens
                for optional_left in A_left_reversed:
                    if j < 1:
                        break
                    if optional_left == input_ids[j - 1]:
                        j -= 1
                    else:
                        break

                # And forwards look as well
                for optional_right in A_right_forward:
                    if k >= n_minus_1:
                        break
                    if optional_right == input_ids[k + 1]:
                        k += 1
                    else:
                        break

                assistant_k = k
                j = assistant_k

                # Given <assistant>, now find next user
                while j < n:
                    # Find <user>
                    # Also accept last final item if assistant is the last turn
                    if (j == n_minus_1) or (
                        (input_ids[j] == Q_first)
                        and (input_ids[j : (k := j + len_Q_must)] == Q_must)
                    ):
                        # Now backtrack to get previous optional tokens
                        for optional_left in Q_left_reversed:
                            if j < 1:
                                break
                            if optional_left == input_ids[j - 1]:
                                j -= 1
                            else:
                                break

                        # And forwards look as well
                        for optional_right in Q_right_forward:
                            if k >= n_minus_1:
                                break
                            if optional_right == input_ids[k + 1]:
                                k += 1
                            else:
                                break

                        user_j = j

                        # Account for last item
                        if user_j != n_minus_1:
                            j = k
                        else:
                            user_j = n
                            k = n

                        # Now copy input_ids to labels
                        labels[assistant_k:user_j] = input_ids[assistant_k:user_j]
                        break

                    j += 1

            j += 1

        return labels

    # Batch trajectories
    for batch_idx, lr in enumerate(learning_rates):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        trajectory_batch = trajectories[start_idx:end_idx]

        # First pass: tokenize all trajectories
        tokenized_trajectories = []
        for trajectory in trajectory_batch:
            messages = trajectory.messages_and_choices
            tools = trajectory.tools

            # Single-step tokenization: apply_chat_template with tokenize=True
            input_ids = cast(
                list[int],
                tokenizer.apply_chat_template(
                    cast(Any, messages),
                    tools=cast(Any, tools),
                    tokenize=True,
                    add_generation_prompt=False,
                ),
            )

            # Create attention mask (all 1s - no padding yet)
            attention_mask = [1] * len(input_ids)

            labels = _train_on_responses_only(input_ids)

            tokenized_trajectories.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        # Find max length in this batch for padding
        max_seq_length = max(len(t["input_ids"]) for t in tokenized_trajectories)

        # Second pass: pad all trajectories to max_seq_length
        trajectory_tensors = []
        for tokenized in tokenized_trajectories:
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = tokenized["labels"]

            # Pad to max_seq_length
            padding_length = max_seq_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length

            trajectory_tensor = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                "labels": torch.tensor([labels], dtype=torch.long),
            }

            trajectory_tensors.append(trajectory_tensor)

        # Calculate total trainable tokens (labels != -100)
        num_trainable_tokens = sum(
            (tensor_dict["labels"] != -100).sum().item()
            for tensor_dict in trajectory_tensors
        )

        yield SFTBatch(
            trajectory_tensors=trajectory_tensors,
            learning_rate=lr,
            num_trajectories=len(trajectory_tensors),
            num_trainable_tokens=num_trainable_tokens,
        )
