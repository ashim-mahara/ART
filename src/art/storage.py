"""Checkpoint storage providers for pulling/pushing model checkpoints.

This module provides a unified interface for different storage backends:
- S3CheckpointStorage: For AWS S3 storage (LocalBackend, SkyPilotBackend)
- WandBArtifactStorage: For W&B artifact storage (ServerlessBackend)
- LocalCheckpointStorage: For local filesystem (LocalBackend default)

Example usage:
    # Pull from S3
    storage = S3CheckpointStorage(bucket="my-bucket", prefix="checkpoints")
    path = await backend.pull_model_checkpoint(model, storage=storage)

    # Pull from W&B (ServerlessBackend)
    storage = WandBArtifactStorage()
    path = await backend.pull_model_checkpoint(model, storage=storage)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .model import TrainableModel


class CheckpointStorage(ABC):
    """Abstract base class for checkpoint storage providers.

    Implementations handle the specifics of pulling checkpoints from
    different storage backends (S3, W&B artifacts, local filesystem, etc.)
    """

    @abstractmethod
    async def pull_checkpoint(
        self,
        model: "TrainableModel",
        step: int,
        local_path: str,
        verbose: bool = False,
    ) -> str:
        """Pull a checkpoint from storage to a local path.

        Args:
            model: The model to pull checkpoint for.
            step: The specific step to pull.
            local_path: Local directory to save the checkpoint.
            verbose: Whether to print verbose output.

        Returns:
            Path to the local checkpoint directory.
        """
        ...

    @abstractmethod
    async def get_latest_step(self, model: "TrainableModel") -> int | None:
        """Get the latest available checkpoint step.

        Args:
            model: The model to check.

        Returns:
            The latest step number, or None if no checkpoints exist.
        """
        ...

    @abstractmethod
    async def checkpoint_exists(
        self, model: "TrainableModel", step: int, local_path: str
    ) -> bool:
        """Check if a checkpoint already exists locally.

        Args:
            model: The model to check.
            step: The step to check.
            local_path: The local path to check.

        Returns:
            True if the checkpoint exists locally.
        """
        ...


@dataclass
class S3CheckpointStorage(CheckpointStorage):
    """S3-based checkpoint storage.

    Used by LocalBackend and SkyPilotBackend for pulling checkpoints from AWS S3.

    Args:
        bucket: S3 bucket name. If None, uses BACKUP_BUCKET env var.
        prefix: Optional S3 prefix/path within the bucket.
    """

    bucket: str | None = None
    prefix: str | None = None

    async def pull_checkpoint(
        self,
        model: "TrainableModel",
        step: int,
        local_path: str,
        verbose: bool = False,
    ) -> str:
        """Pull a checkpoint from S3 to a local path."""
        from art.utils.output_dirs import (
            get_output_dir_from_model_properties,
            get_step_checkpoint_dir,
        )
        from art.utils.s3 import pull_model_from_s3

        if verbose:
            print(f"Pulling checkpoint step {step} from S3...")

        await pull_model_from_s3(
            model_name=model.name,
            project=model.project,
            step=step,
            s3_bucket=self.bucket,
            prefix=self.prefix,
            verbose=verbose,
            art_path=local_path,
            exclude=["logs", "trajectories"],
        )

        # Return the actual path where pull_model_from_s3 synced the checkpoint
        local_model_dir = get_output_dir_from_model_properties(
            project=model.project,
            name=model.name,
            art_path=local_path,
        )
        return get_step_checkpoint_dir(local_model_dir, step)

    async def get_latest_step(self, model: "TrainableModel") -> int | None:
        """Get the latest checkpoint step from S3."""
        from art.utils.s3_checkpoint_utils import get_latest_checkpoint_step_from_s3

        return await get_latest_checkpoint_step_from_s3(
            model_name=model.name,
            project=model.project,
            s3_bucket=self.bucket,
            prefix=self.prefix,
        )

    async def checkpoint_exists(
        self, model: "TrainableModel", step: int, local_path: str
    ) -> bool:
        """Check if checkpoint exists locally (S3 checkpoints are always pulled fresh)."""
        from art.utils.output_dirs import (
            get_output_dir_from_model_properties,
            get_step_checkpoint_dir,
        )

        local_model_dir = get_output_dir_from_model_properties(
            project=model.project,
            name=model.name,
            art_path=local_path,
        )
        checkpoint_dir = get_step_checkpoint_dir(local_model_dir, step)
        return os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0


@dataclass
class WandBArtifactStorage(CheckpointStorage):
    """W&B artifact-based checkpoint storage.

    Used by ServerlessBackend for pulling checkpoints from W&B artifact storage.

    Args:
        api_key: W&B API key. If None, uses WANDB_API_KEY env var.
    """

    api_key: str | None = None

    async def pull_checkpoint(
        self,
        model: "TrainableModel",
        step: int,
        local_path: str,
        verbose: bool = False,
    ) -> str:
        """Pull a checkpoint from W&B artifacts to a local path."""
        import wandb

        assert model.entity is not None, "Model entity is required for W&B storage"

        if verbose:
            print(f"Downloading checkpoint step {step} from W&B artifacts...")

        # The artifact name follows the pattern: {entity}/{project}/{model_name}:v{step}
        artifact_name = f"{model.entity}/{model.project}/{model.name}:v{step}"

        # Use wandb API to download
        api = wandb.Api(api_key=self.api_key)
        artifact = api.artifact(artifact_name, type="lora")

        checkpoint_dir = os.path.join(local_path, f"{step:04d}")

        # Download artifact
        if not os.path.exists(checkpoint_dir):
            os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
            artifact.download(root=checkpoint_dir)
            if verbose:
                print(f"Downloaded checkpoint to {checkpoint_dir}")
        elif verbose:
            print(f"Checkpoint already exists at {checkpoint_dir}")

        return checkpoint_dir

    async def get_latest_step(self, model: "TrainableModel") -> int | None:
        """Get the latest checkpoint step from W&B.

        Note: This requires the model to have been registered with ServerlessBackend
        and have checkpoints available. Returns None if not available.
        """
        # W&B artifact versioning uses v0, v1, v2... which maps to steps
        # The ServerlessBackend uses the checkpoints API to get the latest
        # This is a fallback that requires external context
        return None

    async def checkpoint_exists(
        self, model: "TrainableModel", step: int, local_path: str
    ) -> bool:
        """Check if checkpoint exists locally."""
        checkpoint_dir = os.path.join(local_path, f"{step:04d}")
        return os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0


@dataclass
class LocalCheckpointStorage(CheckpointStorage):
    """Local filesystem checkpoint storage.

    Used by LocalBackend for checkpoints that exist on the local filesystem.
    This is the default when no remote storage is configured.

    Args:
        art_path: The ART directory path where checkpoints are stored.
    """

    art_path: str

    async def pull_checkpoint(
        self,
        model: "TrainableModel",
        step: int,
        local_path: str,
        verbose: bool = False,
    ) -> str:
        """Copy a checkpoint from the ART directory to the target path.

        If local_path is the same as the art_path, just returns the existing path.
        Otherwise, copies the checkpoint to the new location.
        """
        import shutil

        from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

        source_checkpoint_dir = get_step_checkpoint_dir(
            get_model_dir(model=model, art_path=self.art_path), step
        )

        if not os.path.exists(source_checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint not found at {source_checkpoint_dir}")

        # If target is same as source location, just return it
        target_checkpoint_dir = os.path.join(local_path, f"{step:04d}")
        if os.path.normpath(source_checkpoint_dir) == os.path.normpath(
            target_checkpoint_dir
        ):
            if verbose:
                print(f"Checkpoint already at {source_checkpoint_dir}")
            return source_checkpoint_dir

        # Copy to new location
        if os.path.exists(target_checkpoint_dir):
            if verbose:
                print(f"Checkpoint already exists at {target_checkpoint_dir}")
            return target_checkpoint_dir

        if verbose:
            print(
                f"Copying checkpoint from {source_checkpoint_dir} to {target_checkpoint_dir}..."
            )
        os.makedirs(os.path.dirname(target_checkpoint_dir), exist_ok=True)
        shutil.copytree(source_checkpoint_dir, target_checkpoint_dir)
        if verbose:
            print("✓ Checkpoint copied successfully")

        return target_checkpoint_dir

    async def get_latest_step(self, model: "TrainableModel") -> int | None:
        """Get the latest checkpoint step from local filesystem."""
        from art.utils import get_model_step

        try:
            return get_model_step(model, self.art_path)
        except Exception:
            return None

    async def checkpoint_exists(
        self, model: "TrainableModel", step: int, local_path: str
    ) -> bool:
        """Check if checkpoint exists locally."""
        from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

        # Check in art_path first
        source_checkpoint_dir = get_step_checkpoint_dir(
            get_model_dir(model=model, art_path=self.art_path), step
        )
        if (
            os.path.exists(source_checkpoint_dir)
            and len(os.listdir(source_checkpoint_dir)) > 0
        ):
            return True

        # Also check in local_path if different
        target_checkpoint_dir = os.path.join(local_path, f"{step:04d}")
        return (
            os.path.exists(target_checkpoint_dir)
            and len(os.listdir(target_checkpoint_dir)) > 0
        )


# Type alias for storage parameter
CheckpointStorageType = (
    S3CheckpointStorage | WandBArtifactStorage | LocalCheckpointStorage
)
