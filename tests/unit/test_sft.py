"""Unit tests for SFT utilities."""

import json
import math
from pathlib import Path
import tempfile

import pytest

from art.trajectories import Trajectory
from art.types import SFTConfig
from art.utils.sft import create_lr_schedule, create_sft_dataset_iterator, iterate_file


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


# ============================================================================
# Integration tests
# ============================================================================


def test_create_sft_dataset_iterator():
    """Test create_sft_dataset_iterator yields correct chunks."""
    trajectories = [create_dummy_trajectory(i) for i in range(20)]

    # batch_size=8, chunk_size=2 means each chunk has up to 2 batches of 8 trajectories
    # With 20 trajectories per epoch:
    #   - Items per chunk: 8 * 2 = 16
    #   - Chunks per epoch: ceil(20/16) = 2 (one with 16 trajs, one with 4 trajs)
    # With 3 epochs: 2 * 3 = 6 chunks total

    chunks = list(
        create_sft_dataset_iterator(
            trajectories,
            epochs=3,
            batch_size=8,  # 8 trajectories per batch
            chunk_size=2,  # 2 batches per chunk
            use_tqdm=False,
        )
    )

    # Should have 6 chunks total (2 per epoch * 3 epochs)
    assert len(chunks) == 6

    # Pattern repeats for each epoch: full chunk (16 trajs), partial chunk (4 trajs)
    assert len(chunks[0].trajectories) == 16  # Epoch 1, chunk 1
    assert len(chunks[1].trajectories) == 4  # Epoch 1, chunk 2 (partial)
    assert len(chunks[2].trajectories) == 16  # Epoch 2, chunk 1
    assert len(chunks[3].trajectories) == 4  # Epoch 2, chunk 2 (partial)
    assert len(chunks[4].trajectories) == 16  # Epoch 3, chunk 1
    assert len(chunks[5].trajectories) == 4  # Epoch 3, chunk 2 (partial)

    # Verify chunk metadata
    assert chunks[0].step == 0
    assert chunks[0].epoch == 0
    assert chunks[0].epoch_step == 0

    assert chunks[1].step == 1
    assert chunks[1].epoch == 0
    assert chunks[1].epoch_step == 1


def test_iterate_file():
    """Test iterate_file reads trajectories correctly."""
    jsonl_file = create_temp_jsonl(10)

    try:
        # Read without shuffle
        trajectories = list(
            iterate_file(
                str(jsonl_file),
                epochs=2,
                shuffle=False,
            )
        )

        # Should have 20 trajectories (10 per epoch * 2 epochs)
        assert len(trajectories) == 20

        # Verify the content - first epoch should be in order
        for i in range(10):
            assert f"Message {i}" in str(trajectories[i].messages_and_choices)

    finally:
        jsonl_file.unlink()


def test_iterate_file_with_shuffle():
    """Test iterate_file with shuffle enabled."""
    jsonl_file = create_temp_jsonl(100)

    try:
        # Read with shuffle
        trajectories = list(
            iterate_file(
                str(jsonl_file),
                epochs=2,
                shuffle=True,
                shuffle_buffer_size=10,
            )
        )

        # Should have 200 trajectories
        assert len(trajectories) == 200

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
