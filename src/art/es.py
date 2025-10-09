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
from safetensors.torch import load_file as load_sft
from safetensors.torch import save_file as save_sft

from art.model import TrainableModel
from art.trajectories import TrajectoryGroup

download_tasks: dict[str, asyncio.Task[str]] = {}
executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
run_tasks: dict[str, asyncio.Task[wandb.Run]] = {}
wandb_api = wandb.Api()


async def mutate(
    model: TrainableModel,
    *,
    noise_scale: float = 1e-3,
    seed: int | None = None,
) -> TrainableModel:
    seed = seed or random.randint(0, 2**31 - 1)
    mutated_model_inference_name = model.get_inference_name().replace(
        model.name, f"{model.name}-es-{seed}"
    )
    run_id = await _async_download_lora(model.get_inference_name())
    # run, mutated_lora_dir = await asyncio.gather(
    #     _get_run(run_id, model.get_inference_name()),
    #     asyncio.get_running_loop().run_in_executor(
    #         executor,
    #         _mutate_lora,
    #         model.get_inference_name(),
    #         mutated_model.get_inference_name(),
    #         noise_scale,
    #         seed,
    #     ),
    # )
    run = await _get_run(run_id, model.get_inference_name())
    mutated_lora_dir = _mutate_lora(
        model.get_inference_name(),
        mutated_model_inference_name,
        noise_scale,
        seed,
    )
    artifact = wandb.Artifact(
        _artifact_name(mutated_model_inference_name),
        type="lora",
        metadata={
            "wandb.base_model": model.base_model,
            "noise_scale": noise_scale,
            "seed": seed,
        },
        storage_region="coreweave-us",
    )
    await asyncio.to_thread(artifact.add_dir, mutated_lora_dir)
    artifact = await asyncio.to_thread(run.log_artifact, artifact)
    await asyncio.to_thread(artifact.wait)
    return model.model_copy(
        update={"inference_model_name": mutated_model_inference_name + ":v0"},
        # Avoid deep copying to prevent deepcopy of private attrs (e.g. clients with RLocks)
        deep=False,
    )


async def update(
    model: TrainableModel,
    *,
    models: list[TrainableModel],
    trajectory_groups: list[TrajectoryGroup],
    learning_rate: float = 5e-4,
) -> None:
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
            (group.trajectories[i].reward - mean) / max(std, 1e-8)
            for group, mean, std in zip(trajectory_groups, means, stds)
        )
        / len(trajectory_groups)
        for i in range(len(models))
    ]
    artifacts = await asyncio.gather(
        *[
            asyncio.to_thread(
                wandb_api.artifact,
                _qualified_artifact_name_with_alias(m.get_inference_name()),
            )
            for m in models
        ]
    )
    seeds = [artifact.metadata["seed"] for artifact in artifacts]
    noise_scales = [artifact.metadata["noise_scale"] for artifact in artifacts]
    run_id = await _async_download_lora(model.get_inference_name())
    # await asyncio.get_running_loop().run_in_executor(
    #     executor,
    #     _update_lora,
    #     model.get_inference_name(),
    #     run_id,
    #     model.base_model,
    #     learning_rate,
    #     zscores,
    #     seeds,
    #     noise_scales,
    # )
    lora_dir = _update_lora(
        model.get_inference_name(),
        run_id,
        model.base_model,
        learning_rate,
        zscores,
        seeds,
        noise_scales,
    )
    run = await _get_run(run_id, model.get_inference_name())
    artifact = wandb.Artifact(
        _artifact_name(model.get_inference_name()),
        type="lora",
        metadata={"wandb.base_model": model.base_model},
        storage_region="coreweave-us",
    )
    artifact.add_dir(lora_dir)
    artifact = await asyncio.to_thread(run.log_artifact, artifact)
    await asyncio.to_thread(artifact.wait)


async def _get_run(run_id: str, model_inference_name: str) -> wandb.Run:
    if run_id not in run_tasks:
        entity, project, _ = (
            model_inference_name.removeprefix("wandb-artifact:///")
            .removesuffix(":latest")
            .split("/", 2)
        )
        run_tasks[run_id] = asyncio.create_task(
            asyncio.to_thread(
                wandb.init, entity=entity, project=project, id=run_id, resume="must"
            )
        )
    try:
        return await run_tasks[run_id]
    except KeyError:
        run_tasks.pop(run_id, None)
        raise


def _update_lora(
    model_inference_name: str,
    run_id: str,
    base_model: str,
    learning_rate: float,
    zscores: list[float],
    seeds: list[int],
    noise_scales: list[float],
) -> str:
    lora_dir = _lora_dir(model_inference_name)
    assert os.path.exists(lora_dir)
    lora_path = f"{lora_dir}/adapter_model.safetensors"
    adapter_model = load_sft(lora_path)
    with torch.no_grad():
        for zscore, seed, noise_scale in zip(zscores, seeds, noise_scales):
            _mutate_adapter_model(
                adapter_model,
                learning_rate * zscore * noise_scale,
                seed,
            )
    save_sft(adapter_model, lora_path)
    return lora_dir


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
        download_tasks.pop(model_inference_name, None)
        raise


def _download_lora(model_inference_name: str) -> str:
    """Download the corresponding LoRA from Weights & Biases."""
    artifact = wandb_api.artifact(
        _qualified_artifact_name_with_alias(model_inference_name)
    )
    artifact.download(_lora_dir(model_inference_name))
    run = artifact.logged_by()
    assert run is not None
    return run.id


def _mutate_lora(
    model_inference_name: str,
    mutated_model_inference_name: str,
    noise_scale: float,
    seed: int,
) -> str:
    """Mutate the LoRA and upload it to Weights & Biases."""
    lora_dir = _lora_dir(model_inference_name)
    assert os.path.exists(lora_dir)
    mutated_lora_dir = _lora_dir(mutated_model_inference_name)
    shutil.copytree(lora_dir, mutated_lora_dir, dirs_exist_ok=True)
    mutated_lora_path = f"{mutated_lora_dir}/adapter_model.safetensors"
    adapter_model = load_sft(mutated_lora_path)
    with torch.no_grad():
        _mutate_adapter_model(adapter_model, noise_scale, seed)
    save_sft(adapter_model, mutated_lora_path)
    return mutated_lora_dir


def _mutate_adapter_model(
    adapter_model: dict[str, torch.Tensor], noise_scale: float, seed: int
) -> None:
    torch.manual_seed(seed)
    for key, value in adapter_model.items():
        adapter_model[key] = value + torch.randn_like(value) * noise_scale


def _qualified_artifact_name(model_inference_name: str) -> str:
    return model_inference_name.removeprefix("wandb-artifact:///")


def _artifact_name(model_inference_name: str) -> str:
    return _qualified_artifact_name(model_inference_name).split("/")[-1]


def _lora_dir(model_inference_name: str) -> str:
    return f"/tmp/{_qualified_artifact_name(model_inference_name)}"


def _qualified_artifact_name_with_alias(
    model_inference_name: str, alias: str = "latest"
) -> str:
    """Return a fully-qualified W&B artifact reference including an alias.

    W&B public API expects the format 'entity/project/collection:alias' when fetching
    artifacts. The creation name should NOT include the alias, so this helper is only
    used for reads (fetching artifacts), not writes (creating artifacts).
    """
    base = _qualified_artifact_name(model_inference_name)
    # If an alias is already present, don't duplicate it
    return base if ":" in base else f"{base}:{alias}"


@contextmanager
def _wandb_run(
    model_inference_name: str, run_id: str | None = None
) -> Generator[wandb.Run, Any, None]:
    entity, project, _ = (
        model_inference_name.removeprefix("wandb-artifact:///")
        .removesuffix(":latest")
        .split("/", 2)
    )
    with wandb.init(entity=entity, project=project, id=run_id, resume="must") as run:
        yield run
