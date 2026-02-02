"""Minimal example demonstrating SFT -> RL -> SFT switching."""

import asyncio
import os
import random

from dotenv import load_dotenv

import art
from art.local import LocalBackend

# Simple SFT trajectories - teach model to respond "maybe"
SFT_TRAJECTORIES = [
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=0.0,  # reward unused for SFT
    ),
] * 10


async def rl_rollout(model: art.TrainableModel, prompt: str) -> art.Trajectory:
    """Single RL rollout with reward based on response."""
    messages: art.Messages = [{"role": "user", "content": prompt}]
    client = model.openai_client()

    completion = await client.chat.completions.create(
        messages=messages, model=model.name, max_tokens=10, timeout=30
    )
    choice = completion.choices[0]
    content = choice.message.content or ""

    # Reward: "maybe" > "no" > "yes" > other
    reward = {"maybe": 1.0, "no": 0.75, "yes": 0.5}.get(content.strip().lower(), 0.0)
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


async def main():
    load_dotenv()

    backend = LocalBackend()
    model_name = "sft-warmup-" + "".join(
        random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)
    )
    model = art.TrainableModel(
        name=model_name,
        project="sft-warmup",
        base_model="Qwen/Qwen2.5-7B-Instruct",
    )
    await model.register(backend)

    # ========================================================================
    # Phase 1: SFT
    # ========================================================================
    print("\n[Phase 1] SFT training...")
    await model.train_sft(
        SFT_TRAJECTORIES,
        config=art.TrainSFTConfig(learning_rate=1e-5, batch_size=1),
    )
    print("SFT phase 1 complete.")

    # ========================================================================
    # Phase 2: RL (GRPO)
    # ========================================================================
    print("\n[Phase 2] RL training...")
    prompt = "respond with yes, no, or maybe"

    for i in range(10):
        print(f"  RL step {i + 1}")
        train_groups = await art.gather_trajectory_groups(
            [
                art.TrajectoryGroup(rl_rollout(model, prompt) for _ in range(6))
                for _ in range(12)
            ]
        )
        await model.train(train_groups)
    print("RL phase 2 complete.")

    # ========================================================================
    # Phase 3: SFT again
    # ========================================================================
    print("\n[Phase 3] SFT training again...")
    await model.train_sft(
        SFT_TRAJECTORIES,
        config=art.TrainSFTConfig(batch_size=1, learning_rate=1e-5),
    )
    print("SFT phase 3 complete.")

    # ========================================================================
    # Phase 4: RL (GRPO) again
    # ========================================================================
    print("\n[Phase 4] RL training...")
    prompt = "respond with yes, no, or maybe"

    for i in range(10):
        print(f"  RL step {i + 1}")
        train_groups = await art.gather_trajectory_groups(
            [
                art.TrajectoryGroup(rl_rollout(model, prompt) for _ in range(6))
                for _ in range(12)
            ]
        )
        await model.train(train_groups)
    print("RL phase 4 complete.")

    # ========================================================================
    # Test: Check model output
    # ========================================================================
    print("\n[Test] Model output after training:")
    client = model.openai_client()
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "respond with yes or no"}],
        model=model.inference_model_name or model.name,
        max_tokens=10,
    )
    print(f"Response: {completion.choices[0].message.content}")

    print("\nAll phases complete!")


if __name__ == "__main__":
    asyncio.run(main())
