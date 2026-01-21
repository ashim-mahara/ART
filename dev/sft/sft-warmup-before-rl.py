"""Minimal example demonstrating SFT -> RL -> SFT switching."""

import asyncio
import os

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
] * 50


async def rl_rollout(client, model_name: str, prompt: str) -> art.Trajectory:
    """Single RL rollout with reward based on response."""
    messages: art.Messages = [{"role": "user", "content": prompt}]
    completion = await client.chat.completions.create(
        messages=messages, model=model_name, max_tokens=10, timeout=30
    )
    choice = completion.choices[0]
    content = choice.message.content or ""

    # Reward: "maybe" > "no" > "yes" > other
    reward = {"maybe": 1.0, "no": 0.75, "yes": 0.5}.get(content.strip().lower(), 0.0)
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


async def main():
    load_dotenv()

    backend = LocalBackend()
    model = art.TrainableModel(
        name="sft-rl-switch-test-8",
        project="sft-rl-demo",
        base_model="Qwen/Qwen2.5-7B-Instruct",
    )
    await model.register(backend)

    # ========================================================================
    # Phase 1: SFT
    # ========================================================================
    # print("\n[Phase 1] SFT training...")
    # await model.train_sft(
    #     SFT_TRAJECTORIES,
    #     config=art.SFTConfig(learning_rate=1e-4),
    # )
    # print("SFT phase 1 complete.")

    # ========================================================================
    # Phase 2: RL (GRPO)
    # ========================================================================
    print("\n[Phase 2] RL training...")
    client = model.openai_client()
    prompt = "respond with yes, no, or maybe"

    train_groups = await art.gather_trajectory_groups(
        [
            art.TrajectoryGroup(rl_rollout(client, model.name, prompt) for _ in range(6))
            for _ in range(12)
        ]
    )
    await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-4))
    print("RL phase complete.")

    # ========================================================================
    # Phase 3: SFT again
    # ========================================================================
    print("\n[Phase 3] SFT training again...")
    await model.train_sft(
        SFT_TRAJECTORIES,
        config=art.SFTConfig(batch_size=1, learning_rate=2e-4),
    )
    print("SFT phase 3 complete.")

    # ========================================================================
    # Test: Check model output
    # ========================================================================
    print("\n[Test] Model output after training:")
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "respond with yes or no"}],
        model=model.name,
        max_tokens=10,
    )
    print(f"Response: {completion.choices[0].message.content}")

    print("\nAll phases complete!")


if __name__ == "__main__":
    asyncio.run(main())
