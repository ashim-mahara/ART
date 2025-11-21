"""Manual training loop for Supervised Fine-Tuning (SFT) - simpler alternative to Trainer."""

import asyncio
from typing import TYPE_CHECKING

import torch
from peft import PeftModel

if TYPE_CHECKING:
    from ..preprocessing.tokenize_sft import SFTBatch


async def train_sft_manual(
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    input_queue: asyncio.Queue["SFTBatch"],
    results_queue: asyncio.Queue[dict[str, float]],
    device: torch.device | str = "cuda",
) -> None:
    """
    Manual training loop for SFT - simpler alternative to Trainer.
    
    CausalLM models automatically compute cross-entropy loss when labels are provided,
    so we don't need to compute loss manually.
    
    Args:
        model: PEFT model to train
        optimizer: Optimizer (e.g., AdamW)
        input_queue: Queue containing SFTBatch objects
        results_queue: Queue for training metrics
        device: Device to train on
    
    Example:
        ```python
        import torch
        from peft import get_peft_model, LoraConfig
        
        # Setup model
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        peft_config = LoraConfig(r=16, lora_alpha=16, ...)
        model = get_peft_model(model, peft_config)
        model = model.to("cuda")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        
        # Train
        await train_sft_manual(model, optimizer, input_queue, results_queue)
        ```
    """
    model.train()
    global_step = 0
    
    while True:
        try:
            # Get batch from queue
            async def get_batch() -> "SFTBatch":
                return await input_queue.get()
            
            sft_batch: "SFTBatch" = asyncio.run(get_batch())
            
            # Set learning rate for this batch
            for param_group in optimizer.param_groups:
                param_group["lr"] = sft_batch.learning_rate
            
            # Track metrics for this batch
            batch_loss = 0.0
            num_trajectories = sft_batch.num_trajectories
            
            # Process each trajectory with gradient accumulation
            for idx, trajectory_tensor in enumerate(sft_batch.trajectory_tensors):
                # Move tensors to device
                inputs = {
                    key: tensor.to(device)
                    for key, tensor in trajectory_tensor.items()
                }
                
                # Forward pass - CausalLM computes loss automatically when labels provided
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Scale loss by number of trajectories (for gradient accumulation)
                loss = loss / num_trajectories
                
                # Backward pass
                loss.backward()
                
                # Accumulate loss for logging
                batch_loss += loss.item()
            
            # Optimizer step after accumulating gradients from all trajectories
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Prepare metrics
            metrics = {
                "step": global_step,
                "loss": batch_loss,
                "learning_rate": sft_batch.learning_rate,
                "num_trajectories": sft_batch.num_trajectories,
                "num_trainable_tokens": sft_batch.num_trainable_tokens,
            }
            
            # Send metrics to results queue
            results_queue.put_nowait(metrics)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in training loop: {e}")
            break


async def train_sft_manual_with_scheduler(
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    input_queue: asyncio.Queue["SFTBatch"],
    results_queue: asyncio.Queue[dict[str, float]],
    device: torch.device | str = "cuda",
    max_grad_norm: float | None = 1.0,
) -> None:
    """
    Manual training loop with learning rate scheduler and gradient clipping.
    
    Args:
        model: PEFT model to train
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        input_queue: Queue containing SFTBatch objects
        results_queue: Queue for training metrics
        device: Device to train on
        max_grad_norm: Max gradient norm for clipping (None to disable)
    """
    model.train()
    global_step = 0
    
    while True:
        try:
            # Get batch from queue
            async def get_batch() -> "SFTBatch":
                return await input_queue.get()
            
            sft_batch: "SFTBatch" = asyncio.run(get_batch())
            
            # Override learning rate if specified in batch
            # (allows per-batch learning rate control)
            for param_group in optimizer.param_groups:
                param_group["lr"] = sft_batch.learning_rate
            
            # Track metrics
            batch_loss = 0.0
            num_trajectories = sft_batch.num_trajectories
            
            # Process each trajectory with gradient accumulation
            for trajectory_tensor in sft_batch.trajectory_tensors:
                # Move to device
                inputs = {
                    key: tensor.to(device)
                    for key, tensor in trajectory_tensor.items()
                }
                
                # Forward pass - loss computed automatically
                outputs = model(**inputs)
                loss = outputs.loss / num_trajectories
                
                # Backward pass
                loss.backward()
                
                batch_loss += loss.item()
            
            # Gradient clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Scheduler step (if provided)
            if scheduler is not None:
                scheduler.step()
            
            global_step += 1
            
            # Prepare metrics
            metrics = {
                "step": global_step,
                "loss": batch_loss,
                "learning_rate": sft_batch.learning_rate,
                "num_trajectories": num_trajectories,
                "num_trainable_tokens": sft_batch.num_trainable_tokens,
                "grad_norm": torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float('inf')
                ).item() if max_grad_norm else None,
            }
            
            results_queue.put_nowait(metrics)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in training loop: {e}")
            break


# Complete example with manual training loop
async def example_manual_training():
    """
    Complete example showing manual training loop usage.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig
    from ..preprocessing.tokenize_sft import tokenize_sft_batches
    from ..trajectories import Trajectory
    
    # 1. Setup model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
    )
    
    # 2. Apply PEFT
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    model = model.to("cuda")
    
    # 3. Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # 4. Setup queues
    input_queue = asyncio.Queue()
    results_queue = asyncio.Queue()
    
    # 5. Start training task
    train_task = asyncio.create_task(
        train_sft_manual(
            model=model,
            optimizer=optimizer,
            input_queue=input_queue,
            results_queue=results_queue,
            device="cuda",
        )
    )
    
    # 6. Prepare and tokenize data
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
            Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "What is 3+3?"},
                    {"role": "assistant", "content": "3+3 equals 6."},
                ],
                reward=1.0,
            ),
        ],
    ]
    
    batches = tokenize_sft_batches(
        trajectory_batches=trajectory_batches,
        learning_rates=[2e-4],
        tokenizer=tokenizer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    
    # 7. Feed batches to queue
    for batch in batches:
        await input_queue.put(batch)
    
    # 8. Monitor training
    num_batches = len(trajectory_batches)
    for _ in range(num_batches):
        metrics = await results_queue.get()
        print(f"Step {metrics['step']}: Loss={metrics['loss']:.4f}, "
              f"LR={metrics['learning_rate']:.2e}, "
              f"Trainable tokens={metrics['num_trainable_tokens']}")
    
    # 9. Stop training
    train_task.cancel()
    
    # 10. Save model
    model.save_pretrained("./output/manual-sft-model")
    print("Training complete!")


# Comparison: Manual vs Trainer
"""
MANUAL TRAINING LOOP:
Pros:
  ✅ Simple and transparent - you see exactly what happens
  ✅ Direct control over training loop
  ✅ No need to override Trainer methods
  ✅ Loss computed automatically by CausalLM
  ✅ Easy to add custom logic
  ✅ Fewer abstractions

Cons:
  ❌ No built-in features (logging, checkpointing, distributed training)
  ❌ Need to implement gradient accumulation manually
  ❌ No automatic mixed precision (need to add yourself)

TRAINER API:
Pros:
  ✅ Built-in features (logging, checkpointing, distributed)
  ✅ Automatic mixed precision
  ✅ Integrated with HuggingFace ecosystem

Cons:
  ❌ More complex - need to override get_batch_samples
  ❌ Less transparent - harder to debug
  ❌ More abstractions

RECOMMENDATION:
- Use MANUAL for simple cases, prototyping, and full control
- Use TRAINER for production, distributed training, and HF integration
"""

