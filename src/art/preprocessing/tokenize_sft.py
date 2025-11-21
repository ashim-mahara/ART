"""Tokenization utilities for Supervised Fine-Tuning (SFT)."""

import math
from dataclasses import dataclass
from typing import Generator

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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

    instruction_ids = tokenizer(instruction_part, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_part, add_special_tokens=False).input_ids
    instruction_length = len(instruction_ids)
    response_length = len(response_ids)
    max_template_length = max(instruction_length, response_length)

    def _train_on_responses_only(input_ids: list[int]) -> list[int]:
        labels = [-100] * len(input_ids)
        m = len(input_ids) - max_template_length
        first_response = response_ids[0]
        first_instruction = instruction_ids[0]
        j = 0

        while j < m:
            if input_ids[j] == first_response:
                if input_ids[j : j + response_length] == response_ids:
                    j = j + response_length
                    start = j
                    while j < m:
                        if input_ids[j] == first_instruction and input_ids[j : j + instruction_length] == instruction_ids:
                            j = j + instruction_length
                            labels[start : j] = input_ids[start : j]
                            break
                        elif j == (m - 1):
                            j = m
                            labels[start:] = input_ids[start:]
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
            input_ids = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False
            )

            # Create attention mask (all 1s - no padding yet)
            attention_mask = [1] * len(input_ids)

            labels = _train_on_responses_only(input_ids)

            tokenized_trajectories.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            })

        # Find max length in this batch for padding
        max_seq_length = max(len(t['input_ids']) for t in tokenized_trajectories)

        # Second pass: pad all trajectories to max_seq_length
        trajectory_tensors = []
        for tokenized in tokenized_trajectories:
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            labels = tokenized['labels']

            # Pad to max_seq_length
            padding_length = max_seq_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length

            trajectory_tensor = {
                'input_ids': torch.tensor([input_ids], dtype=torch.long),
                'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
                'labels': torch.tensor([labels], dtype=torch.long),
            }

            trajectory_tensors.append(trajectory_tensor)

        # Calculate total trainable tokens (labels != -100)
        num_trainable_tokens = sum(
            (tensor_dict['labels'] != -100).sum().item()
            for tensor_dict in trajectory_tensors
        )

        yield SFTBatch(
            trajectory_tensors=trajectory_tensors,
            learning_rate=lr,
            num_trajectories=len(trajectory_tensors),
            num_trainable_tokens=num_trainable_tokens,
        )

