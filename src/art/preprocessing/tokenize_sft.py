"""Tokenization utilities for Supervised Fine-Tuning (SFT)."""

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
        num_items_in_batch: Number of trajectories in this batch.
    """
    trajectory_tensors: list[dict[str, torch.Tensor]]
    learning_rate: float
    num_items_in_batch: int


def tokenize_sft_batches(
    trajectory_batches: list[list[Trajectory]],
    learning_rates: list[float],
    tokenizer: PreTrainedTokenizerBase,
    instruction_part: str,
    response_part: str,
) -> Generator[SFTBatch, None, None]:
    """
    Tokenize trajectory batches for supervised fine-tuning.

    Args:
        trajectory_batches: List of trajectory batches
        learning_rates: Learning rate for each batch
        tokenizer: Tokenizer to use for encoding
        instruction_part: Instruction template part (e.g., "User:")
        response_part: Response template part (e.g., "Assistant:")

    Yields:
        SFTBatch object containing:
            - trajectory_tensors: List of tensors for each trajectory
            - learning_rate: Learning rate for this batch
            - num_items_in_batch: Number of trajectories in this batch
    """
    instruction_ids = tokenizer(instruction_part, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_part, add_special_tokens=False).input_ids
    instruction_length = len(instruction_ids)
    response_length = len(response_ids)
    max_length = max(instruction_length, response_length)
    
    def _train_on_responses_only(input_ids: list[int]) -> list[int]:
        labels = [-100] * len(input_ids)
        m = len(input_ids) - max_length
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
    
    for trajectory_batch, lr in zip(trajectory_batches, learning_rates):
        trajectory_tensors = []
        
        for trajectory in trajectory_batch:
            messages = trajectory.messages_and_choices
            tools = trajectory.tools
            
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False
            )
            
            processed = tokenizer(formatted_text)
            
            input_ids = processed['input_ids']
            attention_mask = processed['attention_mask']
            
            labels = _train_on_responses_only(input_ids)
            
            trajectory_tensor = {
                'input_ids': torch.tensor([input_ids], dtype=torch.long),
                'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
                'labels': torch.tensor([labels], dtype=torch.long),
            }
            
            trajectory_tensors.append(trajectory_tensor)
        
        yield SFTBatch(
            trajectory_tensors=trajectory_tensors,
            learning_rate=lr,
            num_items_in_batch=len(trajectory_tensors),
        )

