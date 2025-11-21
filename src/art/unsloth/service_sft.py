"""Service for Supervised Fine-Tuning (SFT)."""

import asyncio
import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from .. import dev
from ..local.checkpoints import get_last_checkpoint_dir
from .train_sft import train_sft

if TYPE_CHECKING:
    from ..preprocessing.tokenize_sft import SFTBatch


@dataclass
class SFTService:
    """
    Service for managing SFT training with queue-based batch processing.
    
    Attributes:
        model_name: Name of the model
        base_model: Base model identifier
        config: Internal model configuration
        output_dir: Directory for saving checkpoints and logs
    """
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _train_task: asyncio.Task[None] | None = None
    
    @functools.cached_property
    def input_queue(self) -> asyncio.Queue["SFTBatch"]:
        """Queue for receiving SFTBatch objects."""
        return asyncio.Queue()
    
    @functools.cached_property
    def results_queue(self) -> asyncio.Queue[dict[str, float]]:
        """Queue for training metrics."""
        return asyncio.Queue()
    
    @functools.cached_property
    def trainer(self) -> SFTTrainer:
        """
        Initialize SFTTrainer with PEFT configuration.
        """
        import peft
        import unsloth
        from transformers import PreTrainedTokenizerBase
        
        # Initialize model and tokenizer
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            **self.config.get("init_args", {})
        )
        
        # Initialize PEFT model
        if isinstance(model, peft.peft_model.PeftModelForCausalLM):
            peft_model = model
        else:
            peft_model = unsloth.FastLanguageModel.get_peft_model(
                model, **self.config.get("peft_args", {})
            )
        
        # Create a large dummy dataset for the trainer
        # The actual data comes from the input_queue
        dummy_data = {"text": ""}
        dataset = Dataset.from_list([dummy_data for _ in range(10_000_000)])
        
        # Get trainer configuration
        trainer_args = self.config.get("trainer_args", {})
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            **trainer_args
        )
        
        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=peft_model,
            args=sft_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        
        return trainer
    
    async def train(
        self,
        batches: AsyncIterator["SFTBatch"] | list["SFTBatch"],
    ) -> AsyncIterator[dict[str, float]]:
        """
        Train the model using batches from tokenize_sft_batches.
        
        Args:
            batches: AsyncIterator or list of SFTBatch objects from tokenize_sft_batches
        
        Yields:
            Training metrics (loss, learning_rate, etc.)
        
        Example:
            ```python
            # Create batches from tokenizer
            batches = tokenize_sft_batches(
                trajectory_batches=trajectory_batches,
                learning_rates=learning_rates,
                tokenizer=tokenizer,
                instruction_part="<|im_start|>user\\n",
                response_part="<|im_start|>assistant\\n",
            )
            
            # Train
            async for metrics in service.train(batches):
                print(f"Loss: {metrics['loss']:.4f}")
            ```
        """
        # Start the training task if not already started
        if self._train_task is None:
            self._train_task = asyncio.create_task(
                train_sft(
                    trainer=self.trainer,
                    input_queue=self.input_queue,
                    results_queue=self.results_queue,
                )
            )
            await asyncio.sleep(0.1)  # Let trainer initialize
        
        # Producer: Feed batches to the input queue
        async def feed_batches():
            if hasattr(batches, '__aiter__'):
                # AsyncIterator
                async for batch in batches:
                    await self.input_queue.put(batch)
            else:
                # Regular iterable (e.g., list, generator)
                for batch in batches:
                    await self.input_queue.put(batch)
        
        # Start feeding batches in the background
        feed_task = asyncio.create_task(feed_batches())
        
        # Consumer: Yield metrics from results queue
        try:
            while not feed_task.done() or not self.results_queue.empty():
                try:
                    metrics = await asyncio.wait_for(
                        self.results_queue.get(),
                        timeout=0.1
                    )
                    yield metrics
                except asyncio.TimeoutError:
                    continue
        finally:
            await feed_task
    
    def save_checkpoint(self, checkpoint_name: str | None = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_name: Optional name for checkpoint. If None, uses step number.
        
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            from ..utils.output_dirs import get_step_checkpoint_dir
            checkpoint_path = get_step_checkpoint_dir(
                self.output_dir,
                self.trainer.state.global_step
            )
        else:
            checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self.trainer.save_model(checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str | None = None) -> str:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, loads last checkpoint.
        
        Returns:
            Path to loaded checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = get_last_checkpoint_dir(self.output_dir)
            if checkpoint_path is None:
                raise ValueError(f"No checkpoint found in {self.output_dir}")
        
        # Reload the model with checkpoint
        import peft
        
        self.trainer.model = peft.PeftModel.from_pretrained(
            self.trainer.model.base_model,
            checkpoint_path
        )
        
        return checkpoint_path


# Example usage function
async def example_sft_training():
    """
    Example of how to use SFTService for training.
    """
    from transformers import AutoTokenizer
    from ..preprocessing.tokenize_sft import tokenize_sft_batches
    from ..trajectories import Trajectory
    
    # Initialize service
    service = SFTService(
        model_name="my-sft-model",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        config={
            "init_args": {
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_seq_length": 2048,
                "load_in_4bit": True,
            },
            "peft_args": {
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "trainer_args": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "num_train_epochs": 1,
                "learning_rate": 2e-4,
                "logging_steps": 1,
                "optim": "adamw_8bit",
            },
        },
        output_dir="./output/sft-training",
    )
    
    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    trajectory_batches = [
        [
            Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                ],
                reward=1.0,
            ),
        ],
    ]
    
    learning_rates = [2e-4]
    
    # Tokenize batches
    batches = tokenize_sft_batches(
        trajectory_batches=trajectory_batches,
        learning_rates=learning_rates,
        tokenizer=tokenizer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    
    # Train
    async for metrics in service.train(batches):
        print(f"Step {metrics.get('step')}: Loss={metrics.get('loss'):.4f}")
    
    # Save checkpoint
    checkpoint_path = service.save_checkpoint()
    print(f"Saved checkpoint to {checkpoint_path}")

