"""Unit tests for SFT utilities."""

import json
from pathlib import Path
import tempfile

import pytest

from art.trajectories import Trajectory
from art.utils.sft import create_lr_schedule, iterate_file, prepare_sft_dataset


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


def test_prepare_sft_dataset_basic():
    """Test prepare_sft_dataset prepares trajectories and learning rates correctly."""
    trajectories = [create_dummy_trajectory(i) for i in range(20)]

    # With 20 trajectories, 3 epochs, batch_size=4:
    # - Total trajectories: 20 * 3 = 60
    # - Total batches: ceil(60 / 4) = 15
    all_trajs, learning_rates = prepare_sft_dataset(
        trajectories=trajectories,
        epochs=3,
        batch_size=4,
        peak_lr=1e-4,
        schedule_type="linear",
        warmup_ratio=0.1,
    )

    # Should have 60 trajectories (20 * 3 epochs)
    assert len(all_trajs) == 60

    # Should have 15 learning rates (one per batch)
    assert len(learning_rates) == 15


def test_prepare_sft_dataset_single_epoch():
    """Test prepare_sft_dataset with single epoch."""
    trajectories = [create_dummy_trajectory(i) for i in range(10)]

    all_trajs, learning_rates = prepare_sft_dataset(
        trajectories=trajectories,
        epochs=1,
        batch_size=2,
        peak_lr=1e-4,
        schedule_type="constant",
    )

    # Should have 10 trajectories
    assert len(all_trajs) == 10

    # Should have 5 learning rates (ceil(10/2) = 5)
    assert len(learning_rates) == 5

    # With constant schedule, all learning rates should be the same
    assert all(lr == pytest.approx(1e-4) for lr in learning_rates)


def test_prepare_sft_dataset_epoch_shuffling():
    """Test that different epochs have different trajectory orderings."""
    trajectories = [create_dummy_trajectory(i) for i in range(10)]

    all_trajs, _ = prepare_sft_dataset(
        trajectories=trajectories,
        epochs=2,
        batch_size=10,
        peak_lr=1e-4,
        schedule_type="constant",
    )

    # Should have 20 trajectories (10 * 2 epochs)
    assert len(all_trajs) == 20

    # Get content from each epoch
    epoch0_contents = [
        t.messages_and_choices[0]["content"]  # type: ignore[index,typeddict-item]
        for t in all_trajs[:10]
    ]
    epoch1_contents = [
        t.messages_and_choices[0]["content"]  # type: ignore[index,typeddict-item]
        for t in all_trajs[10:]
    ]

    # Different epochs should have different orderings (due to shuffle)
    assert epoch0_contents != epoch1_contents


def test_prepare_sft_dataset_deterministic_shuffling():
    """Test that shuffling is deterministic with same seed."""
    trajectories = [create_dummy_trajectory(i) for i in range(10)]

    # Run twice with same seed
    all_trajs1, _ = prepare_sft_dataset(
        trajectories=trajectories,
        epochs=2,
        batch_size=5,
        peak_lr=1e-4,
        schedule_type="constant",
        shuffle_seed=42,
    )

    all_trajs2, _ = prepare_sft_dataset(
        trajectories=trajectories,
        epochs=2,
        batch_size=5,
        peak_lr=1e-4,
        schedule_type="constant",
        shuffle_seed=42,
    )

    # Should get same order
    for t1, t2 in zip(all_trajs1, all_trajs2):
        assert t1.reward == t2.reward


def test_prepare_sft_dataset_with_initial_step():
    """Test resuming from initial_step skips correct trajectories and LRs."""
    trajectories = [create_dummy_trajectory(i) for i in range(10)]

    # Without initial_step
    all_trajs_full, lrs_full = prepare_sft_dataset(
        trajectories=trajectories,
        epochs=1,
        batch_size=2,
        peak_lr=1e-4,
        schedule_type="linear",
    )

    # With initial_step=2: skip first 2 batches (4 trajectories)
    all_trajs_resumed, lrs_resumed = prepare_sft_dataset(
        trajectories=trajectories,
        epochs=1,
        batch_size=2,
        peak_lr=1e-4,
        schedule_type="linear",
        initial_step=2,
    )

    # Should have fewer trajectories and learning rates
    assert len(all_trajs_resumed) == len(all_trajs_full) - 4  # Skip 2 batches * 2 trajs
    assert len(lrs_resumed) == len(lrs_full) - 2  # Skip 2 LRs


def test_prepare_sft_dataset_empty():
    """Test prepare_sft_dataset with empty trajectories."""
    all_trajs, learning_rates = prepare_sft_dataset(
        trajectories=[],
        epochs=3,
        batch_size=4,
        peak_lr=1e-4,
        schedule_type="linear",
    )

    assert all_trajs == []
    assert learning_rates == []


def test_iterate_file():
    """Test iterate_file reads trajectories correctly."""
    jsonl_file = create_temp_jsonl(10)

    try:
        # Read without shuffle
        trajectories = list(
            iterate_file(
                str(jsonl_file),
                shuffle=False,
            )
        )

        # Should have 10 trajectories
        assert len(trajectories) == 10

        # Verify the content - should be in order
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
                shuffle=True,
                shuffle_buffer_size=10,
            )
        )

        # Should have 100 trajectories
        assert len(trajectories) == 100

    finally:
        jsonl_file.unlink()


def test_lr_schedule_warmup_not_zero():
    """Test that warmup doesn't start at lr=0."""
    lrs = create_lr_schedule(
        total_steps=10,
        peak_lr=1e-4,
        method="constant",
        warmup_steps=5,
        min_lr=0.0,
    )

    # First step should NOT be 0
    assert lrs[0] > 0
    # Should reach peak_lr by end of warmup
    assert lrs[4] == pytest.approx(1e-4)
    # After warmup, should stay at peak_lr (constant schedule)
    assert lrs[5] == pytest.approx(1e-4)


def test_lr_schedule_edge_cases():
    """Test LR schedule edge cases."""
    # Empty schedule
    lrs = create_lr_schedule(total_steps=0, peak_lr=1e-4)
    assert lrs == []

    # Single step
    lrs = create_lr_schedule(total_steps=1, peak_lr=1e-4)
    assert len(lrs) == 1
    assert lrs[0] == pytest.approx(1e-4)

    # Warmup steps >= total_steps (edge case)
    lrs = create_lr_schedule(total_steps=5, peak_lr=1e-4, warmup_steps=10)
    assert len(lrs) == 5
    # Should not crash and should produce valid learning rates
    assert all(lr > 0 for lr in lrs)


def test_lr_schedule_decay_methods():
    """Test that cosine and linear decay work correctly."""
    peak_lr = 1e-4
    min_lr = 1e-5

    # Linear decay: should go from peak_lr to min_lr
    lrs = create_lr_schedule(
        total_steps=5, peak_lr=peak_lr, method="linear", min_lr=min_lr
    )
    assert lrs[0] == pytest.approx(peak_lr)  # Start at peak
    assert lrs[-1] == pytest.approx(min_lr)  # End at min
    # Should be monotonically decreasing
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1]

    # Cosine decay: should go from peak_lr to min_lr
    lrs = create_lr_schedule(
        total_steps=5, peak_lr=peak_lr, method="cosine", min_lr=min_lr
    )
    assert lrs[0] == pytest.approx(peak_lr)  # Start at peak
    assert lrs[-1] == pytest.approx(min_lr)  # End at min


def test_lr_schedule_no_warmup():
    """Test schedule with warmup_steps=0."""
    lrs = create_lr_schedule(
        total_steps=5, peak_lr=1e-4, method="linear", warmup_steps=0, min_lr=0
    )
    assert len(lrs) == 5
    assert lrs[0] == pytest.approx(1e-4)  # Start at peak (no warmup)
    assert lrs[-1] == pytest.approx(0)  # End at min_lr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
