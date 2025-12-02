"""Deployment utilities for deploying trained models to inference endpoints."""

from .common import (
    DeploymentConfig,
    DeploymentResult,
    Provider,
    deploy_model,
)
from .together import (
    TogetherDeploymentConfig,
)
from .wandb import (
    WandbDeploymentConfig,
    deploy_wandb,
)

# Legacy exports for backwards compatibility
from .legacy import (
    LoRADeploymentJob,
    LoRADeploymentProvider,
)

__all__ = [
    # New API
    "DeploymentConfig",
    "DeploymentResult",
    "Provider",
    "TogetherDeploymentConfig",
    "WandbDeploymentConfig",
    "deploy_model",
    "deploy_wandb",
    # Legacy API
    "LoRADeploymentJob",
    "LoRADeploymentProvider",
]
