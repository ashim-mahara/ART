import os
from typing import Iterable, Literal

from mp_actors import move_to_child_process

from .. import dev
from ..local.backend import LocalBackend, LocalTrainResult
from ..local.service import ModelService
from ..model import TrainableModel
from ..trajectories import TrajectoryGroup
from ..types import TrainConfig
from ..utils.output_dirs import get_model_dir, get_step_checkpoint_dir


class TinkerBackend(LocalBackend):
    def __init__(
        self,
        *,
        tinker_api_key: str | None = None,
        in_process: bool = False,
        path: str | None = None,
    ) -> None:
        if not "TINKER_API_KEY" in os.environ or tinker_api_key is not None:
            assert tinker_api_key is not None, (
                "TINKER_API_KEY is not set and no tinker_api_key was provided"
            )
            print("Setting TINKER_API_KEY to", tinker_api_key, "in environment")
            os.environ["TINKER_API_KEY"] = tinker_api_key
        super().__init__(in_process=in_process, path=path)

    async def train(  # type: ignore[override]
        self,
        model: TrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        # Core training parameters
        learning_rate: float = 5e-6,
        beta: float = 0.0,
        # RL algorithm settings (used by ART's custom loss when tinker_loss_fn is None)
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
        # LocalBackend-specific parameters
        allow_training_without_logprobs: bool = False,
        plot_tensors: bool = False,
        truncated_importance_sampling: float | None = None,
        scale_learning_rate_by_reward_std_dev: bool = False,
        logprob_calculation_chunk_size: int = 1024,
        num_trajectories_learning_rate_multiplier_power: float = 0.0,
        # Checkpoint behavior
        save_checkpoint: bool = True,
        # Verbosity
        verbose: bool = False,
        # Tinker-specific: built-in loss function
        tinker_loss_fn: Literal["importance_sampling", "ppo", "cispo", "dro"]
        | None = None,
        tinker_loss_fn_config: dict[str, float] | None = None,
        # Adam optimizer parameters
        adam_beta1: float | None = None,
        adam_beta2: float | None = None,
        adam_eps: float | None = None,
    ) -> LocalTrainResult:
        """Train the model on trajectory groups, with optional Tinker built-in loss.

        When tinker_loss_fn is specified, uses Tinker's optimized built-in loss
        function (e.g., "cispo", "ppo"). This is faster than ART's custom loss
        (1.5x fewer FLOPs, up to 3x faster wall time).

        When tinker_loss_fn is None (default), uses ART's custom loss implementation,
        which is compatible with other backends like LocalBackend.

        Args:
            model: The trainable model to train.
            trajectory_groups: Batches of trajectories to train on.
            learning_rate: Learning rate for training. Defaults to 5e-6.
            beta: KL penalty coefficient. Defaults to 0.0.
            tinker_loss_fn: Tinker built-in loss function. Options:
                - "importance_sampling": REINFORCE with importance sampling
                - "ppo": Proximal Policy Optimization with clipping
                - "cispo": Clipped Importance Sampling Policy Optimization
                - "dro": Direct Reward Optimization
                If None, uses ART's custom loss (controlled by ppo, epsilon, etc.)
            tinker_loss_fn_config: Config dict for built-in loss, e.g.:
                {"clip_low_threshold": 0.0, "clip_high_threshold": 6.0}
            adam_beta1: Adam optimizer beta1 parameter. Defaults to Tinker default (0.9).
            adam_beta2: Adam optimizer beta2 parameter. Defaults to Tinker default (0.999).
            adam_eps: Adam optimizer epsilon parameter. Defaults to Tinker default (1e-8).
            **other_args: See LocalBackend.train() for other parameters.

        Returns:
            LocalTrainResult with step number, training metrics, and checkpoint path.

        Example:
            # Use Tinker's built-in CISPO with custom Adam params
            result = await backend.train(
                model,
                trajectory_groups,
                learning_rate=5e-6,
                tinker_loss_fn="cispo",
                tinker_loss_fn_config={"clip_low_threshold": 0.0, "clip_high_threshold": 6.0},
                adam_beta1=0.9,
                adam_beta2=0.95,  # Custom beta2
                adam_eps=1e-8,
            )

            # Use ART's custom loss (default, for compatibility)
            result = await backend.train(
                model,
                trajectory_groups,
                learning_rate=5e-6,
                ppo=False,
                epsilon=1.0,
            )
        """
        groups_list = list(trajectory_groups)

        # Build config objects from explicit kwargs
        config = TrainConfig(learning_rate=learning_rate, beta=beta)
        dev_config: dev.TrainConfig = {
            "advantage_balance": advantage_balance,
            "allow_training_without_logprobs": allow_training_without_logprobs,
            "importance_sampling_level": importance_sampling_level,
            "mask_prob_ratio": mask_prob_ratio,
            "plot_tensors": plot_tensors,
            "ppo": ppo,
            "precalculate_logprobs": precalculate_logprobs,
            "scale_learning_rate_by_reward_std_dev": scale_learning_rate_by_reward_std_dev,
            "scale_rewards": scale_rewards,
            "logprob_calculation_chunk_size": logprob_calculation_chunk_size,
            "num_trajectories_learning_rate_multiplier_power": num_trajectories_learning_rate_multiplier_power,
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
        if truncated_importance_sampling is not None:
            dev_config["truncated_importance_sampling"] = truncated_importance_sampling

        # Tinker-specific: built-in loss function
        if tinker_loss_fn is not None:
            dev_config["tinker_loss_fn"] = tinker_loss_fn
        if tinker_loss_fn_config is not None:
            dev_config["tinker_loss_fn_config"] = tinker_loss_fn_config

        # Tinker-specific: checkpoint control
        dev_config["tinker_save_checkpoint"] = save_checkpoint

        # Tinker-specific: Adam optimizer parameters
        if adam_beta1 is not None:
            dev_config["adam_beta1"] = adam_beta1
        if adam_beta2 is not None:
            dev_config["adam_beta2"] = adam_beta2
        if adam_eps is not None:
            dev_config["adam_eps"] = adam_eps

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

        # Get step and checkpoint path
        step = await self._get_step(model)
        checkpoint_path: str | None = None
        if save_checkpoint:
            checkpoint_path = get_step_checkpoint_dir(
                get_model_dir(model=model, art_path=self._path), step
            )
            if not os.path.exists(checkpoint_path):
                checkpoint_path = None

        return LocalTrainResult(
            step=step,
            metrics=avg_metrics,
            checkpoint_path=checkpoint_path,
        )

    async def _get_service(self, model: TrainableModel) -> ModelService:
        from ..dev.get_model_config import get_model_config
        from ..dev.model import TinkerArgs
        from .service import TinkerService

        if model.name not in self._services:
            config = get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            config["tinker_args"] = config.get("tinker_args") or TinkerArgs(
                renderer_name=get_renderer_name(model.base_model)
            )
            config["tinker_args"]["training_client_args"] = (
                config["tinker_args"].get("training_client_args") or {}
            )
            self._services[model.name] = TinkerService(
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
            )
            if not self._in_process:
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="tinker-service",
                )
        return self._services[model.name]


renderer_name_message = """
To manually specify a renderer (and silence this message), you can set the "renderer_name" field like so:

model = art.TrainableModel(
    name="my-model",
    project="my-project",
    base_model="Qwen/Qwen3-8B",
    _internal_config=art.dev.InternalModelConfig(
        tinker_args=art.dev.TinkerArgs(renderer_name="qwen3_disable_thinking"),
    ),
)

Valid renderer names are:

- llama3
- qwen3
- qwen3_disable_thinking
- qwen3_instruct
- deepseekv3
- deepseekv3_disable_thinking
- gpt_oss_no_sysprompt
- gpt_oss_low_reasoning
- gpt_oss_medium_reasoning
- gpt_oss_high_reasoning
""".strip()


def get_renderer_name(base_model: str) -> str:
    if base_model.startswith("meta-llama/"):
        return "llama3"
    elif base_model.startswith("Qwen/Qwen3-"):
        if "Instruct" in base_model:
            return "qwen3_instruct"
        else:
            print("Defaulting to Qwen3 renderer without thinking for", base_model)
            print(renderer_name_message)
            return "qwen3_disable_thinking"
    elif base_model.startswith("deepseek-ai/DeepSeek-V3"):
        print("Defaulting to DeepSeekV3 renderer without thinking for", base_model)
        print(renderer_name_message)
        return "deepseekv3_disable_thinking"
    elif base_model.startswith("openai/gpt-oss"):
        print("Defaulting to GPT-OSS renderer without system prompt for", base_model)
        print(renderer_name_message)
        return "gpt_oss_no_sysprompt"
    else:
        raise ValueError(f"Unknown base model: {base_model}")
