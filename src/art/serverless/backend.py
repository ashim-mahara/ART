import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Literal
import warnings

from openai._types import NOT_GIVEN
from tqdm import auto as tqdm

from art.serverless.client import Client, ExperimentalTrainingConfig

from .. import dev
from ..backend import AnyTrainableModel, Backend
from ..trajectories import Trajectory, TrajectoryGroup
from ..types import ServerlessTrainResult, TrainConfig, TrainSFTConfig

if TYPE_CHECKING:
    from ..model import Model, TrainableModel


class ServerlessBackend(Backend):
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        client = Client(api_key=api_key, base_url=base_url)
        self._base_url = str(client.base_url)
        self._client = client

    async def close(self) -> None:
        await self._client.close()  # ty:ignore[possibly-missing-attribute]

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
                "Registering a non-trainable model with the Serverless backend is not supported."
            )
            return
        client_model = await self._client.models.create(  # ty:ignore[possibly-missing-attribute]
            entity=model.entity,
            project=model.project,
            name=model.name,
            base_model=model.base_model,
            return_existing=True,
        )
        model.id = client_model.id
        model.entity = client_model.entity
        # Store run_id for wandb run management
        if hasattr(client_model, "run_id"):
            model.run_id = client_model.run_id

    async def delete(
        self,
        model: "Model",
    ) -> None:
        """
        Deletes a model from the Backend.

        Args:
            model: An art.Model instance to delete.
        """
        from art import TrainableModel

        if not isinstance(model, TrainableModel):
            print(
                "Deleting a non-trainable model from the Serverless backend is not supported."
            )
            return
        assert model.id is not None, "Model ID is required"
        await self._client.models.delete(model_id=model.id)  # ty:ignore[possibly-missing-attribute]

    def _model_inference_name(self, model: "Model", step: int | None = None) -> str:
        """Return the inference name for a model checkpoint.

        Args:
            model: The model.
            step: If provided, returns name for specific checkpoint using
                  W&B artifact versioning (e.g., :step5). If None, returns
                  name for latest checkpoint (default, backwards compatible).
        """
        assert model.entity is not None, "Model entity is required"
        base_name = f"wandb-artifact:///{model.entity}/{model.project}/{model.name}"
        if step is not None:
            return f"{base_name}:step{step}"
        return base_name

    async def _get_step(self, model: "Model") -> int:
        if model.trainable:
            assert model.id is not None, "Model ID is required"
            async for checkpoint in self._client.models.checkpoints.list(  # ty:ignore[possibly-missing-attribute]
                limit=1, order="desc", model_id=model.id
            ):
                return checkpoint.step
        # Non-trainable models do not have checkpoints/steps; default to 0
        return 0

    async def _delete_checkpoint_files(
        self,
        model: AnyTrainableModel,
        steps_to_keep: list[int],
    ) -> None:
        """Delete checkpoint files, keeping only the specified steps."""
        assert model.id is not None, "Model ID is required"
        # Get all checkpoint steps
        all_steps: list[int] = []
        async for checkpoint in self._client.models.checkpoints.list(model_id=model.id):  # ty:ignore[possibly-missing-attribute]
            all_steps.append(checkpoint.step)
        # Delete all steps not in steps_to_keep
        if steps_to_delete := [step for step in all_steps if step not in steps_to_keep]:
            await self._client.models.checkpoints.delete(  # ty:ignore[possibly-missing-attribute]
                model_id=model.id,
                steps=steps_to_delete,
            )

    async def _prepare_backend_for_training(
        self,
        model: AnyTrainableModel,
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        return str(self._base_url), self._client.api_key  # ty:ignore[possibly-missing-attribute]

    # Note: _log() method has been moved to the Model class (frontend)
    # Trajectories are now saved locally by the Model.log() method

    async def train(  # type: ignore[override]
        self,
        model: AnyTrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        # Core training parameters
        learning_rate: float = 5e-6,
        beta: float = 0.0,
        # RL algorithm settings
        ppo: bool = False,
        epsilon: float | None = None,
        epsilon_high: float | None = None,
        # Advantage computation
        advantage_balance: float = 0.0,
        scale_rewards: bool = True,
        # Importance sampling
        importance_sampling_level: Literal[
            "token", "sequence", "average", "geometric_average"
        ] = "token",
        max_negative_advantage_importance_sampling_weight: float | None = None,
        mask_prob_ratio: bool = False,
        # Experimental parameters
        kimi_k2_tau: float | None = None,
        precalculate_logprobs: bool = False,
        # Verbosity
        verbose: bool = False,
    ) -> ServerlessTrainResult:
        """Train the model on the given trajectory groups.

        This is the recommended way to train models. Unlike model.train(), this
        method does NOT automatically log trajectories or metrics. Call model.log()
        explicitly before and/or after training if you want to log data.

        Args:
            model: The trainable model to train.
            trajectory_groups: Batches of trajectories to train on.
            learning_rate: Learning rate for training. Defaults to 5e-6.
            beta: KL penalty coefficient. Defaults to 0.0.
            ppo: Whether to use PPO clipping. Defaults to False.
            epsilon: Clip epsilon for importance sampling. Defaults based on ppo.
            epsilon_high: Asymmetric upper clip bound. Defaults to epsilon.
            advantage_balance: Balance between negative and positive advantages
                in range [-1.0, 1.0]. Defaults to 0.0 (balanced).
            scale_rewards: Whether to scale rewards by standard deviation.
                Defaults to True.
            importance_sampling_level: Level at which to compute importance
                sampling weights. Defaults to "token".
            max_negative_advantage_importance_sampling_weight: Maximum weight
                for negative advantage samples.
            mask_prob_ratio: Whether to mask probability ratios. Defaults to False.
            kimi_k2_tau: Tau parameter for Kimi K2 algorithm.
            precalculate_logprobs: Whether to precalculate logprobs.
            verbose: Whether to print verbose output. Defaults to False.

        Returns:
            ServerlessTrainResult with step number, training metrics, and artifact name.

        Example:
            # Before (deprecated):
            await model.train(trajectory_groups, config=TrainConfig(learning_rate=5e-6))

            # After (recommended):
            await model.log(trajectory_groups, split="train")
            result = await backend.train(model, trajectory_groups, learning_rate=5e-6)
            # Optionally log training metrics:
            # await model.log(metrics=result.metrics, step=result.step)
        """
        groups_list = list(trajectory_groups)

        # Build config objects from explicit kwargs
        config = TrainConfig(learning_rate=learning_rate, beta=beta)
        dev_config: dev.TrainConfig = {
            "advantage_balance": advantage_balance,
            "importance_sampling_level": importance_sampling_level,
            "mask_prob_ratio": mask_prob_ratio,
            "ppo": ppo,
            "precalculate_logprobs": precalculate_logprobs,
            "scale_rewards": scale_rewards,
        }
        # Only include optional fields if they're set
        if epsilon is not None:
            dev_config["epsilon"] = epsilon
        if epsilon_high is not None:
            dev_config["epsilon_high"] = epsilon_high
        if max_negative_advantage_importance_sampling_weight is not None:
            dev_config["max_negative_advantage_importance_sampling_weight"] = (
                max_negative_advantage_importance_sampling_weight
            )
        if kimi_k2_tau is not None:
            dev_config["kimi_k2_tau"] = kimi_k2_tau

        # Collect metrics from training
        training_metrics: list[dict[str, float]] = []
        async for metrics in self._train_model(
            model, groups_list, config, dev_config, verbose
        ):
            training_metrics.append(metrics)

        # Aggregate metrics
        avg_metrics: dict[str, float] = {}
        if training_metrics:
            avg_metrics = {
                k: sum(d.get(k, 0) for d in training_metrics)
                / sum(1 for d in training_metrics if k in d)
                for k in {k for d in training_metrics for k in d}
                if k != "num_gradient_steps"
            }

        # Get step and artifact name
        step = await self._get_step(model)
        artifact_name: str | None = None
        if model.entity is not None:
            artifact_name = f"{model.entity}/{model.project}/{model.name}:step{step}"

        return ServerlessTrainResult(
            step=step,
            metrics=avg_metrics,
            artifact_name=artifact_name,
        )

    async def _train_model(
        self,
        model: AnyTrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        assert model.id is not None, "Model ID is required"
        training_job = await self._client.training_jobs.create(  # ty:ignore[possibly-missing-attribute]
            model_id=model.id,
            trajectory_groups=trajectory_groups,
            experimental_config=ExperimentalTrainingConfig(
                advantage_balance=dev_config.get("advantage_balance"),
                epsilon=dev_config.get("epsilon"),
                epsilon_high=dev_config.get("epsilon_high"),
                importance_sampling_level=dev_config.get("importance_sampling_level"),
                kimi_k2_tau=dev_config.get("kimi_k2_tau"),
                learning_rate=config.learning_rate,
                mask_prob_ratio=dev_config.get("mask_prob_ratio"),
                max_negative_advantage_importance_sampling_weight=dev_config.get(
                    "max_negative_advantage_importance_sampling_weight"
                ),
                ppo=dev_config.get("ppo"),
                precalculate_logprobs=dev_config.get("precalculate_logprobs"),
                scale_rewards=dev_config.get("scale_rewards"),
            ),
        )
        after: str | None = None
        num_sequences: int | None = None
        pbar: tqdm.tqdm | None = None
        while True:
            await asyncio.sleep(1)
            async for event in self._client.training_jobs.events.list(  # ty:ignore[possibly-missing-attribute]
                training_job_id=training_job.id, after=after or NOT_GIVEN
            ):
                if event.type == "gradient_step":
                    assert pbar is not None and num_sequences is not None
                    pbar.update(1)
                    pbar.set_postfix(event.data)
                    yield {**event.data, "num_gradient_steps": num_sequences}
                elif event.type == "training_started":
                    num_sequences = event.data["num_sequences"]
                    if pbar is None:
                        pbar = tqdm.tqdm(total=num_sequences, desc="train")
                    continue
                elif event.type == "training_ended":
                    return
                elif event.type == "training_failed":
                    error_message = event.data.get(
                        "error_message", "Training failed with an unknown error"
                    )
                    raise RuntimeError(f"Training job failed: {error_message}")
                after = event.id

    async def _train_sft(
        self,
        model: AnyTrainableModel,
        trajectories: Iterable[Trajectory],
        config: TrainSFTConfig,
        dev_config: dev.TrainSFTConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        """Train the model using supervised fine-tuning.

        For ServerlessBackend, this serializes trajectories to a JSONL file,
        uploads it to W&B artifacts, and calls the SFT training API.

        Args:
            model: The trainable model to fine-tune.
            trajectories: Iterable of Trajectory objects.
            config: SFT configuration with batch_size and learning rates.
            dev_config: Developer configuration.
            verbose: Whether to print detailed logs.

        Yields:
            Dictionary containing training metrics for each batch.
        """
        import json
        import tempfile
        import uuid

        import wandb

        assert model.id is not None, "Model ID is required"

        # Get the user's default entity from W&B if not set
        if model.entity is None:
            api = wandb.Api(api_key=self._client.api_key)
            model.entity = api.default_entity

        # Generate unique artifact name to avoid race conditions in distributed systems
        artifact_id = uuid.uuid4().hex[:12]
        artifact_name = f"{model.name}-sft-data-{artifact_id}"

        if verbose:
            print("Serializing trajectories to file (streaming)...")

        # Serialize trajectories to a temporary JSONL file (streaming - no memory load)
        num_trajectories = 0
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp_file:
            for trajectory in trajectories:
                # Convert trajectory to the expected JSONL format
                line: dict[str, Any] = {
                    "messages": trajectory.messages(),
                }
                if trajectory.tools:
                    line["tools"] = trajectory.tools
                tmp_file.write(json.dumps(line) + "\n")
                num_trajectories += 1
            tmp_file_path = tmp_file.name

        if num_trajectories == 0:
            if verbose:
                print("No trajectories to train on")
            import os

            os.unlink(tmp_file_path)
            return

        if verbose:
            print(f"Serialized {num_trajectories} trajectories")

        try:
            if verbose:
                print("Uploading training data to W&B artifacts...")

            # Upload the file to W&B as a dataset artifact
            # Use the model's canonical run_id from database, or fall back to model name
            run_id_to_use = getattr(model, "run_id", model.name)
            run = wandb.init(
                name=model.name,
                id=run_id_to_use,  # Use stored run_id to match the canonical wandb run
                entity=model.entity,
                project=model.project,
                resume="allow",  # Resume if this run already exists
                settings=wandb.Settings(api_key=self._client.api_key),
            )
            try:
                artifact = wandb.Artifact(
                    artifact_name,
                    type="dataset",
                    metadata={
                        "format": "jsonl",
                        "num_trajectories": num_trajectories,
                    },
                )
                artifact.add_file(tmp_file_path, name="train.jsonl")
                artifact = run.log_artifact(artifact)
                try:
                    artifact = artifact.wait()
                except ValueError as e:
                    if "Unable to fetch artifact with id" in str(e):
                        if verbose:
                            print(f"Warning: {e}")
                    else:
                        raise e
            finally:
                # Finish the run so the workflow can resume it later
                # The workflow uses wandb_run with resume="must" to continue this run
                run.finish()
        finally:
            # Clean up temporary file
            import os

            os.unlink(tmp_file_path)

        # Construct the artifact URL with unique name (v0 is the first version)
        training_folder_url = (
            f"wandb-artifact:///{model.entity}/{model.project}/{artifact_name}:v0"
        )

        if verbose:
            print(f"Training data uploaded. Artifact URL: {training_folder_url}")
            print("Starting SFT training job...")

        # Create SFT training job
        from .client import SFTTrainingConfig

        sft_config: SFTTrainingConfig = {}
        if config.batch_size != "auto":
            sft_config["batch_size"] = config.batch_size
        sft_config["learning_rate"] = config.learning_rate

        sft_training_job = await self._client.sft_training_jobs.create(
            model_id=model.id,
            training_folder_url=training_folder_url,
            config=sft_config,
        )

        # Poll for events
        after: str | None = None
        num_batches: int | None = None
        pbar: tqdm.tqdm | None = None
        while True:
            await asyncio.sleep(1)
            async for event in self._client.sft_training_jobs.events.list(
                training_job_id=sft_training_job.id, after=after or NOT_GIVEN
            ):
                if event.type == "gradient_step":
                    assert pbar is not None and num_batches is not None
                    pbar.update(1)
                    pbar.set_postfix(event.data)
                    yield {**event.data, "num_gradient_steps": num_batches}
                elif event.type == "training_started":
                    num_batches = event.data.get("num_sequences", 0)
                    if pbar is None:
                        pbar = tqdm.tqdm(total=num_batches, desc="train sft")
                    continue
                elif event.type == "training_ended":
                    if pbar is not None:
                        pbar.close()
                    return
                elif event.type == "training_failed":
                    if pbar is not None:
                        pbar.close()
                    error_message = event.data.get(
                        "error_message", "SFT training failed with an unknown error"
                    )
                    raise RuntimeError(f"SFT training job failed: {error_message}")
                after = event.id

    # ------------------------------------------------------------------
    # Experimental support for S3 and checkpoints
    # ------------------------------------------------------------------

    async def _experimental_pull_model_checkpoint(
        self,
        model: "TrainableModel",
        *,
        step: int | Literal["latest"] | None = None,
        local_path: str | None = None,
        verbose: bool = False,
    ) -> str:
        """Pull a model checkpoint from W&B artifacts to a local path.

        For ServerlessBackend, this downloads the checkpoint from W&B artifact storage.

        Args:
            model: The model to pull checkpoint for.
            step: The step to pull. Can be an int for a specific step,
                 or "latest" to pull the latest checkpoint. If None, pulls latest.
            local_path: Local directory to save the checkpoint. If None, uses temporary directory.
            verbose: Whether to print verbose output.

        Returns:
            Path to the local checkpoint directory.
        """
        import os
        import tempfile

        import wandb

        assert model.id is not None, "Model ID is required"

        # If entity is not set, use the user's default entity from W&B
        api = wandb.Api(api_key=self._client.api_key)  # ty:ignore[possibly-missing-attribute]
        if model.entity is None:
            model.entity = api.default_entity
            if verbose:
                print(f"Using default W&B entity: {model.entity}")

        # Determine which step to use
        resolved_step: int
        if step is None or step == "latest":
            # Get latest checkpoint from API
            async for checkpoint in self._client.models.checkpoints.list(  # ty:ignore[possibly-missing-attribute]
                limit=1, order="desc", model_id=model.id
            ):
                resolved_step = checkpoint.step
                break
            else:
                raise ValueError(f"No checkpoints found for model {model.name}")
        else:
            resolved_step = step

        if verbose:
            print(f"Downloading checkpoint step {resolved_step} from W&B artifacts...")

        # Download from W&B artifacts
        # The artifact name follows the pattern: {entity}/{project}/{model_name}:step{step}
        artifact_name = (
            f"{model.entity}/{model.project}/{model.name}:step{resolved_step}"
        )

        # Use wandb API to download (api was already created above for entity lookup)
        artifact = api.artifact(artifact_name, type="lora")

        # Determine download path
        if local_path is None:
            # Create a temporary directory that won't be cleaned up automatically
            checkpoint_dir = os.path.join(
                tempfile.gettempdir(),
                "art_checkpoints",
                model.project,
                model.name,
                f"{resolved_step:04d}",
            )
        else:
            # Custom location - copy directly to local_path
            checkpoint_dir = local_path

        # Download artifact
        os.makedirs(checkpoint_dir, exist_ok=True)
        artifact.download(root=checkpoint_dir)
        if verbose:
            print(f"Downloaded checkpoint to {checkpoint_dir}")

        return checkpoint_dir

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
        """Deprecated. Use `_experimental_pull_model_checkpoint` instead."""
        warnings.warn(
            "_experimental_pull_from_s3 is deprecated. Use _experimental_pull_model_checkpoint instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
