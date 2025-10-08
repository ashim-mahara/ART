import asyncio
import concurrent.futures
import math
import os
import random
import shutil
from contextlib import contextmanager
from typing import Any, Generator

import torch
import wandb

from art.model import TrainableModel
from art.trajectories import TrajectoryGroup

wandb_api = wandb.Api()
download_tasks: dict[str, asyncio.Task[str]] = {}
executor = concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 1)


async def mutate(
    trainable_model: TrainableModel,
    *,
    noise_scale: float = 1e-3,
    seed: int | None = None,
) -> TrainableModel:
    seed = seed or random.randint(0, 2**31 - 1)
    mutated_model = trainable_model.model_copy(
        update={
            "inference_model_name": trainable_model.get_inference_name().replace(
                trainable_model.name, f"{trainable_model.name}-es-{seed}"
            )
        },
        deep=True,
    )
    run_id = await _async_download_lora(trainable_model.get_inference_name())
    await asyncio.get_event_loop().run_in_executor(
        executor,
        _mutate_and_upload_lora,
        trainable_model.get_inference_name(),
        mutated_model.get_inference_name(),
        run_id,
        trainable_model.base_model,
        noise_scale,
        seed,
    )
    return mutated_model


async def recombine(
    *,
    models: list[TrainableModel],
    trajectory_groups: list[TrajectoryGroup],
    into: TrainableModel,
    learning_rate: float = 5e-4,
) -> TrainableModel:
    means = [
        sum(trajectory.reward for trajectory in group) / len(group)
        for group in trajectory_groups
    ]
    stds = [
        math.sqrt(
            sum((trajectory.reward - mean) ** 2 for trajectory in group) / len(group)
        )
        for group, mean in zip(trajectory_groups, means)
    ]
    zscores = [
        sum(
            (group.trajectories[i].reward - mean) / std
            for group, mean, std in zip(trajectory_groups, means, stds)
        )
        / len(trajectory_groups)
        for i in range(len(models))
    ]
    artifacts = await asyncio.gather(
        *[
            asyncio.to_thread(
                wandb_api.artifact, _artifact_name(model.get_inference_name())
            )
            for model in models
        ]
    )
    seeds = [artifact.metadata["seed"] for artifact in artifacts]
    noise_scales = [artifact.metadata["noise_scale"] for artifact in artifacts]
    run_id = await _async_download_lora(into.get_inference_name())
    await asyncio.get_event_loop().run_in_executor(
        executor,
        _recombine_and_upload_lora,
        into.get_inference_name(),
        run_id,
        into.base_model,
        learning_rate,
        zscores,
        seeds,
        noise_scales,
    )
    return into


def _recombine_and_upload_lora(
    model_inference_name: str,
    run_id: str,
    base_model: str,
    learning_rate: float,
    zscores: list[float],
    seeds: list[int],
    noise_scales: list[float],
) -> None:
    lora_dir = _lora_dir(model_inference_name)
    assert os.path.exists(lora_dir)
    lora_path = f"{lora_dir}/adapter_model.safetensors"
    adapter_model = torch.load(lora_path)
    for zscore, seed, noise_scale in zip(zscores, seeds, noise_scales):
        _mutate_adapter_model(adapter_model, learning_rate * zscore * noise_scale, seed)
    torch.save(adapter_model, lora_path)
    with _wandb_run(model_inference_name, run_id) as run:
        artifact = wandb.Artifact(
            _artifact_name(model_inference_name),
            type="lora",
            metadata={"wandb.base_model": base_model},
        )
        artifact.add_dir(lora_dir)
        artifact = run.log_artifact(artifact)
        artifact.wait()


def _async_download_lora(model_inference_name: str) -> asyncio.Task[str]:
    if model_inference_name not in download_tasks:

        async def _download_lora_coroutine() -> str:
            return await asyncio.get_event_loop().run_in_executor(
                executor, _download_lora, model_inference_name
            )

        download_tasks[model_inference_name] = asyncio.create_task(
            _download_lora_coroutine()
        )
    try:
        return download_tasks[model_inference_name]
    except KeyError:
        download_tasks.pop(model_inference_name)
        raise


def _download_lora(model_inference_name: str) -> str:
    """Download the corresponding LoRA from Weights & Biases."""
    artifact = wandb_api.artifact(_artifact_name(model_inference_name))
    artifact.download(_lora_dir(model_inference_name))
    runs = artifact.used_by()
    return runs[0].id


def _mutate_and_upload_lora(
    model_inference_name: str,
    mutated_model_inference_name: str,
    run_id: str,
    base_model: str,
    noise_scale: float,
    seed: int,
) -> None:
    """Mutate the LoRA and upload it to Weights & Biases."""
    lora_dir = _lora_dir(model_inference_name)
    assert os.path.exists(lora_dir)
    mutated_lora_dir = _lora_dir(mutated_model_inference_name)
    os.makedirs(mutated_lora_dir, exist_ok=True)
    shutil.copytree(lora_dir, mutated_lora_dir)
    mutated_lora_path = f"{mutated_lora_dir}/adapter_model.safetensors"
    adapter_model = torch.load(mutated_lora_path)
    _mutate_adapter_model(adapter_model, noise_scale, seed)
    torch.save(adapter_model, mutated_lora_path)
    with _wandb_run(mutated_model_inference_name, run_id) as run:
        artifact = wandb.Artifact(
            _artifact_name(mutated_model_inference_name),
            type="lora",
            metadata={
                "wandb.base_model": base_model,
                "noise_scale": noise_scale,
                "seed": seed,
            },
            storage_region="coreweave-us",
        )
        artifact.add_dir(mutated_lora_dir)
        artifact = run.log_artifact(artifact)
        artifact.wait()


def _mutate_adapter_model(
    adapter_model: dict[str, torch.Tensor], noise_scale: float, seed: int
) -> None:
    torch.manual_seed(seed)
    for key, value in adapter_model.items():
        adapter_model[key] = value + torch.randn_like(value) * noise_scale


def _artifact_name(model_inference_name: str) -> str:
    return model_inference_name.removeprefix("wandb-artifact:///")


def _lora_dir(model_inference_name: str) -> str:
    return f"/tmp/{_artifact_name(model_inference_name)}"


@contextmanager
def _wandb_run(
    model_inference_name: str, run_id: str | None = None
) -> Generator[wandb.Run, Any, None]:
    entity, project, _ = (
        model_inference_name.removeprefix("wandb-artifact:///")
        .removesuffix(":latest")
        .split("/")
    )
    with wandb.init(entity=entity, project=project, id=run_id, resume="must") as run:
        yield run
