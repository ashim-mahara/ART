import asyncio
from typing import TYPE_CHECKING, AsyncIterator, Literal, cast
import os

from art.client import Client
from art.utils.deploy_model import LoRADeploymentJob, LoRADeploymentProvider
from art.utils.trajectory_logging import get_metric_averages
import wandb
import weave
from wandb.sdk.wandb_run import Run
from weave.trace.weave_client import WeaveClient

from .. import dev
from ..backend import Backend
from ..trajectories import TrajectoryGroup
from ..types import TrainConfig

if TYPE_CHECKING:
    from ..model import Model, TrainableModel


class ServerlessBackend(Backend):
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        client = Client(api_key=api_key, base_url=base_url)
        super().__init__(base_url=str(client.base_url))
        self._client = client
        self._wandb_runs: dict[str, Run] = {}
        self._weave_clients: dict[str, WeaveClient] = {}

    async def close(self) -> None:
        await self._client.close()

    async def register(
        self,
        model: "Model",
    ) -> None:
        """
        Registers a model with the Backend for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        from art import TrainableModel

        if not isinstance(model, TrainableModel):
            print(
                "Registering a non-trainable model with the WandB backend is not supported."
            )
            return

        client_model = await self._client.models.create(
            entity=model.entity,
            project=model.project,
            name=model.name,
            base_model=model.base_model,
            return_existing=True,
        )
        model.id = client_model.id
        model.entity = client_model.entity

    def _model_inference_name(self, model: "TrainableModel") -> str:
        assert model.entity is not None, "Model entity is required"
        return f"{model.entity}/{model.project}/{model.name}"

    async def __get_step(self, model: "Model") -> int:
        if model.trainable:
            model = cast(TrainableModel, model)
            assert model.id is not None, "Model ID is required"
            checkpoint = await self._client.checkpoints.retrieve(
                model_id=model.id, step="latest"
            )
            return checkpoint.step
        # Non-trainable models do not have checkpoints/steps; default to 0
        return 0

    async def _delete_checkpoints(
        self,
        model: "TrainableModel",
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        # TODO: potentially implement benchmark smoothing
        max_metric: float | None = None
        max_step: int | None = None
        all_steps: list[int] = []
        async for checkpoint in self._client.checkpoints.list(model_id=model.id):
            metric = checkpoint.metrics.get(benchmark, None)
            if metric is not None and (max_metric is None or metric > max_metric):
                max_metric = metric
                max_step = checkpoint.step
            all_steps.append(checkpoint.step)
        steps_to_delete = [step for step in all_steps[1:] if step != max_step]
        if steps_to_delete:
            await self._client.checkpoints.delete(
                model_id=model.id,
                steps=steps_to_delete,
            )

    async def _prepare_backend_for_training(
        self,
        model: "TrainableModel",
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        return str(self._base_url), self._client.api_key

    async def _log(
        self,
        model: "Model",
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        # TODO: log trajectories to local file system?

        averages = get_metric_averages(trajectory_groups)
        await self._log_metrics(model, averages, split)

    async def _log_metrics(
        self,
        model: Model,
        metrics: dict[str, float],
        split: str,
        step: int | None = None,
    ) -> None:
        metrics = {f"{split}/{metric}": value for metric, value in metrics.items()}
        step = step if step is not None else await self.__get_step(model)

        # TODO: Write to history.jsonl like we do in LocalBackend?

        # If we have a W&B run, log the data there
        if run := self._get_wandb_run(model):
            # Mark the step metric itself as hidden so W&B doesn't create an automatic chart for it
            wandb.define_metric("training_step", hidden=True)

            # Enabling the following line will cause W&B to use the training_step metric as the x-axis for all metrics
            # wandb.define_metric(f"{split}/*", step_metric="training_step")
            run.log({"training_step": step, **metrics}, step=step)

        # Report metrics to the W&B Training API
        if model.trainable and model.id is not None:
            await self._client.checkpoints.report_metrics(
                model_id=model.id, step=step, metrics=metrics
            )


    def _get_wandb_run(self, model: Model) -> Run | None:
        if "WANDB_API_KEY" not in os.environ:
            return None
        if (
            model.name not in self._wandb_runs
            or self._wandb_runs[model.name]._is_finished
        ):
            run = wandb.init(
                project=model.project,
                name=model.name,
                id=model.name,
                resume="allow",
            )
            self._wandb_runs[model.name] = run
            os.environ["WEAVE_PRINT_CALL_LINK"] = os.getenv(
                "WEAVE_PRINT_CALL_LINK", "False"
            )
            os.environ["WEAVE_LOG_LEVEL"] = os.getenv("WEAVE_LOG_LEVEL", "CRITICAL")
            self._weave_clients[model.name] = weave.init(model.project)
        return self._wandb_runs[model.name]
        

    async def _train_model(
        self,
        model: "TrainableModel",
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        assert model.id is not None, "Model ID is required"
        training_job = await self._client.training_jobs.create(
            model_id=model.id,
            trajectory_groups=trajectory_groups,
            experimental_config=dict(learning_rate=config.learning_rate),
        )
        while training_job.status != "COMPLETED":
            await asyncio.sleep(1)
            training_job = await self._client.training_jobs.retrieve(training_job.id)
            yield {"num_gradient_steps": 1}

    # ------------------------------------------------------------------
    # Experimental support for S3
    # ------------------------------------------------------------------

    async def _experimental_pull_from_s3(
        self,
        model: "Model",
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
        only_step: int | Literal["latest"] | None = None,
    ) -> None:
        raise NotImplementedError

    async def _experimental_push_to_s3(
        self,
        model: "Model",
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        raise NotImplementedError

    async def _experimental_fork_checkpoint(
        self,
        model: "Model",
        from_model: str,
        from_project: str | None = None,
        from_s3_bucket: str | None = None,
        not_after_step: int | None = None,
        verbose: bool = False,
        prefix: str | None = None,
    ) -> None:
        raise NotImplementedError

    async def _experimental_deploy(
        self,
        deploy_to: LoRADeploymentProvider,
        model: "TrainableModel",
        step: int | None = None,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        pull_s3: bool = True,
        wait_for_completion: bool = True,
    ) -> LoRADeploymentJob:
        raise NotImplementedError
