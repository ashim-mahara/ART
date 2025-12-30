import asyncio
from itertools import permutations

import openai
from dotenv import load_dotenv

import art
from art.local import LocalBackend


async def rollout(
    client: openai.AsyncOpenAI, model_name: str, prompt: str
) -> art.Trajectory:
    messages: art.Messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    chat_completion = await client.chat.completions.create(
        messages=messages, model=model_name, max_tokens=100, timeout=100
    )
    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str)
    if content == "yes":
        reward = 0.5
    elif content == "no":
        reward = 0.75
    elif content == "maybe":
        reward = 1.0
    else:
        reward = 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


def with_quotes(w: str) -> str:
    return f"'{w}'"


async def main():
    load_dotenv()

    backend = LocalBackend()
    model = art.TrainableModel(
        name="009",
        project="yes-no-maybe",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        # Use decoupled mode: unsloth for training, vLLM for inference (separate)
        # This avoids the vLLM 0.12 multiprocessing incompatibility with unsloth
        _internal_config={"_decouple_vllm_and_unsloth": True},
    )
    await model.register(backend)

    prompts = [
        f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
        for prefix in ["respond", "just respond"]
        for use_quotes in [True, False]
        for words in (
            list(p) for n in [3, 2] for p in permutations(["yes", "no", "maybe"], n)
        )
    ]

    openai_client = model.openai_client()
    for _ in range(await model.get_step(), 1):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(openai_client, model.name, prompt) for _ in range(32)
                )
                for prompt in prompts
            )
        )
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-4),
        )


if __name__ == "__main__":
    asyncio.run(main())
