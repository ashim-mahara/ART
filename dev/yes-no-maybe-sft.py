import asyncio
import os

from dotenv import load_dotenv

import art
from art.local import LocalBackend


# Teacher trajectories - high-quality examples from a "strong model"
# These always respond with "maybe" which has the highest reward (1.0)
TEACHER_TRAJECTORIES = [
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "respond with yes or no"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "just respond with 'no' or 'maybe'"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
    art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "just respond with 'no' or 'maybe'"},
            {"role": "assistant", "content": "maybe"},
        ],
        reward=1.0,
    ),
]


async def main():
    load_dotenv()

    backend = LocalBackend()
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    model = art.TrainableModel(
        name=os.environ.get("MODEL_NAME", "sft-test-5"),
        project="yes-no-maybe",
        base_model=base_model,
    )
    await model.register(backend)

    # ========================================================================
    # SFT Phase: Train on teacher trajectories
    # ========================================================================
    print("\n" + "=" * 70)
    print("Starting SFT training on teacher trajectories")
    print("=" * 70 + "\n")

    # Train for 3 epochs on the teacher data with constant learning rate
    num_sft_epochs = int(os.environ.get("NUM_SFT_EPOCHS", "10"))
    sft_lr = float(os.environ.get("SFT_LR", "2e-4"))

    for epoch in range(num_sft_epochs):
        print(f"\nSFT Epoch {epoch + 1}/{num_sft_epochs}")
        await model.train_sft(
            TEACHER_TRAJECTORIES,
            config=art.SFTConfig(
                batch_size=4,
                learning_rate=sft_lr,
            ),
            verbose=(epoch == 0),  # Verbose only on first epoch
        )

    print("\n" + "=" * 70)
    print("SFT training complete! Running inference tests...")
    print("=" * 70 + "\n")

    # ========================================================================
    # Inference Phase: Test the trained model
    # ========================================================================
    openai_client = model.openai_client()

    # Test prompts covering different formats
    test_prompts = [
        "respond with yes or no",
    ]

    print("Testing model responses:\n")
    for test_prompt in test_prompts:
        messages: art.Messages = [{"role": "user", "content": test_prompt}]

        chat_completion = await openai_client.chat.completions.create(
            messages=messages,
            model=model.name,
            max_tokens=10,
            timeout=30,
        )

        response = chat_completion.choices[0].message.content
        print(f"Prompt:   {test_prompt}")
        print(f"Response: {response}")
        print()

    print("=" * 70)
    print("Inference complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
