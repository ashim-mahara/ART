"""Unit tests for SFT utilities."""

import json
from pathlib import Path
import tempfile

import pytest

from art.utils.sft import create_lr_schedule, iterate_file


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


def test_iterate_file():
    """Test iterate_file reads trajectories correctly."""
    jsonl_file = create_temp_jsonl(10)

    try:
        trajectories = list(iterate_file(str(jsonl_file), epochs=1))

        assert len(trajectories) == 10

    finally:
        jsonl_file.unlink()


def test_iterate_file_multiple_epochs():
    """Test iterate_file with multiple epochs."""
    jsonl_file = create_temp_jsonl(10)

    try:
        trajectories = list(iterate_file(str(jsonl_file), epochs=3))

        # Should have 30 trajectories (10 * 3 epochs)
        assert len(trajectories) == 30

    finally:
        jsonl_file.unlink()


def test_iterate_file_with_initial_skip():
    """Test iterate_file with initial_skip for resuming."""
    jsonl_file = create_temp_jsonl(10)

    try:
        # Skip first 5 trajectories
        trajectories = list(iterate_file(str(jsonl_file), epochs=1, initial_skip=5))

        assert len(trajectories) == 5

    finally:
        jsonl_file.unlink()


def test_iterate_file_deterministic():
    """Test that iterate_file is deterministic with same seed."""
    jsonl_file = create_temp_jsonl(20)

    try:
        traj1 = list(iterate_file(str(jsonl_file), epochs=1, seed=42))
        traj2 = list(iterate_file(str(jsonl_file), epochs=1, seed=42))

        # Should get same order
        for t1, t2 in zip(traj1, traj2):
            assert t1.messages_and_choices == t2.messages_and_choices

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
