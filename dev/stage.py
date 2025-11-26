import asyncio
import os
import random
from itertools import permutations

import art
import litellm
from art.serverless.backend import ServerlessBackend
from art.utils.deploy_model import (
    LoRADeploymentProvider,
    deploy_model,
)
from art.utils.litellm import convert_litellm_choice_to_openai
from dotenv import load_dotenv
from litellm.types.utils import Choices, ModelResponse

load_dotenv()


async def rollout(model: art.Model, scenario: str, step: int) -> art.Trajectory:
    messages: art.Messages = [
        {
            "role": "user",
            "content": scenario,
        }
    ]
    response = await litellm.acompletion(
        messages=messages,
        model=f"openai/{model.get_inference_name()}",
        max_tokens=100,
        timeout=100,
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    assert isinstance(response, ModelResponse)
    choice = response.choices[0]
    assert isinstance(choice, Choices)
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
    return art.Trajectory(
        messages_and_choices=[*messages, convert_litellm_choice_to_openai(choice)],
        reward=reward,
        metrics={"custom_metric": random.random(), "run_step": step},
    )


async def main() -> None:
    backend = ServerlessBackend(
        base_url="https://api.qa.training.wandb.ai/v1",
        api_key="be47e013c03bd1afc979794cde276bdd421de0f3",
        # api_key="be47e013c03bd1afc979794cde276bdd421de0f3", // production
    )
    model = art.TrainableModel(
        name="".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)),
        project="yes-no-maybe",
        base_model="Qwen/Qwen2.5-14B-Instruct",
    )
    await model.register(backend)
    print(f"Created model: {model.name}")

    def with_quotes(w: str) -> str:
        return f"'{w}'"

    scenarios = [
        f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
        for prefix in ["respond", "just respond"]
        for use_quotes in [True, False]
        for words in (
            list(p) for n in [3, 2] for p in permutations(["yes", "no", "maybe"], n)
        )
    ]
    random.seed(42)
    random.shuffle(scenarios)
    val_scenarios = scenarios[: len(scenarios) // 2]
    train_scenarios = scenarios[len(scenarios) // 2 :]

    has_printed_step_warning = False
    target_steps = 1  # Train for 1 steps
    starting_step = await model.get_step()

    for _step in range(starting_step, starting_step + target_steps):
        step = await model.get_step()
        if step != _step and not has_printed_step_warning:
            print(f"Warning: Step mismatch: {step} != {_step}")
            has_printed_step_warning = True
        val_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(model, scenario, step) for _ in range(8))
                for scenario in val_scenarios
            ),
            pbar_desc=f"gather(val:{step})",
        )
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(model, scenario, step) for _ in range(8))
                for scenario in train_scenarios
            ),
            pbar_desc=f"gather(train:{step})",
        )
        await model.log(val_groups)
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=5e-5),
            _config=art.dev.TrainConfig(precalculate_logprobs=True),
        )
        await model.delete_checkpoints(best_checkpoint_metric="train/reward")

    # Download the latest checkpoint to local directory (same folder as this script)
    print("\n" + "=" * 80)
    print("Downloading checkpoint to local directory...")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = await backend._experimental_pull_model_checkpoint(
        model, step="latest", local_path=script_dir, verbose=True
    )

    print(f"\n✓ Checkpoint downloaded to: {checkpoint_path}")
    print("\nFiles in checkpoint directory:")
    print("-" * 80)

    # List all files in the checkpoint directory
    for root, dirs, files in os.walk(checkpoint_path):
        level = root.replace(checkpoint_path, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            file_size = os.path.getsize(os.path.join(root, file))
            # Format file size nicely
            if file_size < 1024:
                size_str = f"{file_size}B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f}KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f}MB"
            print(f"{subindent}{file} ({size_str})")

    # Deploy the checkpoint to Together
    print("\n" + "=" * 80)
    print("Deploying checkpoint to Together...")
    print("=" * 80)

    # Extract step number from checkpoint path
    final_step = int(os.path.basename(checkpoint_path))

    deployment_job = await deploy_model(
        deploy_to=LoRADeploymentProvider.TOGETHER,
        model=model,
        checkpoint_path=checkpoint_path,
        step=final_step,
        s3_bucket=None,  # Will use default S3 bucket for presigned URL
        verbose=True,
        wait_for_completion=True,
    )

    print(f"\n✓ Deployment complete!")
    print(f"  Status: {deployment_job.status}")
    print(f"  Job ID: {deployment_job.job_id}")
    print(f"  Model Name: {deployment_job.model_name}")
    if deployment_job.failure_reason:
        print(f"  Failure Reason: {deployment_job.failure_reason}")

    print("\n" + "=" * 80)
    print(f"Training complete! Model: {model.name}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
