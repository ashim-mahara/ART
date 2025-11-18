"""Unit tests for SFT utilities."""

import json
import math
import tempfile
from pathlib import Path
from typing import Iterable, List

import pytest

from art.trajectories import Trajectory
from art.types import SFTConfig
from art.utils.iterate_dataset import iterate_file, iterate_trajectories
from art.utils.sft import create_lr_schedule


# Helper to create dummy trajectories
def create_dummy_trajectory(idx: int) -> Trajectory:
    """Create a dummy trajectory with a unique identifier."""
    return Trajectory(
        messages_and_choices=[
            {"role": "user", "content": f"Message {idx}"},
            {"role": "assistant", "content": f"Response {idx}"},
        ],
        reward=float(idx),
    )


# Helper to create a temporary JSONL file
def create_temp_jsonl(num_trajectories: int) -> Path:
    """Create a temporary JSONL file with dummy trajectories."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for i in range(num_trajectories):
        data = {
            "messages": [
                {"role": "user", "content": f"Message {i}"},
                {"role": "assistant", "content": f"Response {i}"},
            ],
        }
        temp_file.write(json.dumps(data) + "\n")
    temp_file.close()
    return Path(temp_file.name)


# Dummy train_sft for integration testing
def dummy_train_sft(
    trajectories: Iterable[List[Trajectory]],
    config: SFTConfig,
) -> dict:
    """
    Dummy train_sft function that collects batches and learning rates.

    Args:
        trajectories: Iterable of trajectory batches
        config: SFT configuration with learning rates

    Returns:
        dict with:
            - num_batches: number of batches processed
            - total_trajectories: total number of trajectories seen
            - learning_rates_used: list of learning rates used
    """
    num_batches = 0
    total_trajectories = 0

    for batch in trajectories:
        num_batches += 1
        total_trajectories += len(batch)

    return {
        "num_batches": num_batches,
        "total_trajectories": total_trajectories
    }


# ============================================================================
# Integration tests
# ============================================================================

def test_integration_iterate_trajectories_with_train_sft():
    """Test using iterate_trajectories chunks with train_sft."""
    trajectories = [create_dummy_trajectory(i) for i in range(20)]

    # batch_size=8, chunk_size=2 means each chunk has up to 2 batches of 8 trajectories
    # With 20 trajectories per epoch:
    #   - Items per chunk: 8 * 2 = 16
    #   - Chunks per epoch: ceil(20/16) = 2 (one with 16 trajs, one with 4 trajs)
    # With 3 epochs: 2 * 3 = 6 chunks total

    # Create LR schedule for up to 2 batches per chunk
    lrs_per_chunk = create_lr_schedule(2, peak_lr=1e-4, method="linear")

    # Manually iterate over chunks and train on each
    results = []
    for chunk in iterate_trajectories(
        trajectories,
        epochs=3,
        batch_size=8,  # 8 trajectories per batch
        chunk_size=2,  # 2 batches per chunk
    ):
        print(f"Chunk: {chunk}")
        # chunk is List[List[Trajectory]] which is an Iterable[List[Trajectory]]
        result = dummy_train_sft(
            trajectories=chunk,
            config=SFTConfig(learning_rate=lrs_per_chunk),
        )
        results.append(result)

    # Should have 6 chunks total (2 per epoch * 3 epochs)
    assert len(results) == 6
    # Pattern repeats for each epoch: full chunk (2 batches), partial chunk (1 batch)
    assert results[0]["num_batches"] == 2  # Epoch 1, chunk 1
    assert results[0]["total_trajectories"] == 16
    assert results[1]["num_batches"] == 1  # Epoch 1, chunk 2 (partial)
    assert results[1]["total_trajectories"] == 4
    assert results[2]["num_batches"] == 2  # Epoch 2, chunk 1
    assert results[2]["total_trajectories"] == 16
    assert results[3]["num_batches"] == 1  # Epoch 2, chunk 2 (partial)
    assert results[3]["total_trajectories"] == 4
    assert results[4]["num_batches"] == 2  # Epoch 3, chunk 1
    assert results[4]["total_trajectories"] == 16
    assert results[5]["num_batches"] == 1  # Epoch 3, chunk 2 (partial)
    assert results[5]["total_trajectories"] == 4

def test_integration_iterate_file_with_train_sft():
    """Test using iterate_file directly with train_sft."""
    jsonl_file = create_temp_jsonl(100)

    try:
        # Create learning rate schedule
        total_steps = math.ceil((100 * 2) / 3)  # 10 trajectories, 2 epochs, batch_size=3
        lrs = create_lr_schedule(total_steps, peak_lr=1e-4, method="constant")

        config = SFTConfig(learning_rate=lrs)

        # Pass iterate_file directly to train_sft
        result = dummy_train_sft(
            trajectories=iterate_file(
                str(jsonl_file),
                epochs=2,
                batch_size=3,
                shuffle=True,
            ),
            config=config,
        )

        # Should process 7 batches: [3, 3, 3, 3, 3, 3, 2]
        assert result["num_batches"] == 67
        assert result["total_trajectories"] == 200
    finally:
        jsonl_file.unlink()

# def test_total_steps_calculation():
#     """Test that total steps calculation matches actual batches."""
#     num_trajectories = 105
#     epochs = 3
#     batch_size = 8

#     # This is how train_sft_from_file calculates total_steps
#     expected_total_steps = math.ceil((num_trajectories * epochs) / batch_size)

#     # Create file and count actual batches
#     jsonl_file = create_temp_jsonl(num_trajectories)

#     try:
#         batches = list(iterate_file(
#             str(jsonl_file),
#             epochs=epochs,
#             batch_size=batch_size,
#             shuffle=False,
#         ))

#         actual_batches = len(batches)

#         # Should match
#         assert actual_batches == expected_total_steps
#     finally:
#         jsonl_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
