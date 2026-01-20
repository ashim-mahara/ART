"""SFT training with periodic benchmarking at 25%, 50%, 75%, and 100%."""

import asyncio

import wandb

import art
from art.local import LocalBackend
from art.utils.sft import create_sft_dataset_iterator, iterate_file

from pii_test import run_benchmark


# Benchmark checkpoints as percentage of training (0 = before training starts)
EVAL_CHECKPOINTS = [0, 0.25, 0.50, 0.75, 1.0]


async def main():
    backend = LocalBackend()
    model = art.TrainableModel(
        name="pii-art-llama-linear-2e-4-bs-4-ep-2",
        project="OP-unsloth-SDKtests",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
    )
    await model.register(backend)

    # Initialize wandb
    run = wandb.init(
        project="OP-unsloth-SDKtests",
        name=model.name,
        id=model.name,
        resume="allow",
    )

    # Load trajectories from file
    trajectories = list(iterate_file("dev/sft/pii_train.jsonl", epochs=1))

    # Create iterator to get total steps
    chunks = list(
        create_sft_dataset_iterator(
            trajectories=trajectories,
            epochs=1,
            batch_size=4,
            peak_lr=2e-4,
            schedule_type="linear",
            use_tqdm=False,
        )
    )
    total_chunks = len(chunks)

    # Calculate which chunk indices to evaluate at
    eval_at_chunks = {int(p * total_chunks) for p in EVAL_CHECKPOINTS}
    # Ensure we always eval at the last chunk
    eval_at_chunks.add(total_chunks)

    print(f"Total chunks: {total_chunks}")
    print(f"Will evaluate after chunks: {sorted(eval_at_chunks)}")
    print("-" * 60)

    # Run baseline eval before training if 0 is in checkpoints
    if 0 in eval_at_chunks:
        print("\n[0%] Running baseline benchmark (before training)...")
        metrics = await run_benchmark(model, show_progress=True)
        print(f"[0%] EM: {metrics['exact_match']:.2%}, F1: {metrics['f1']:.2%}, G: {metrics['grounded']:.2%}, P: {metrics['precision']:.2%}, R: {metrics['recall']:.2%}")
        run.log({
            "eval/exact_match": metrics["exact_match"],
            "eval/f1": metrics["f1"],
            "eval/grounded": metrics["grounded"],
            "eval/precision": metrics["precision"],
            "eval/recall": metrics["recall"],
        }, step=0)
        eval_at_chunks.discard(0)  # Remove so we don't try to eval at chunk 0 again

    # Re-create iterator for actual training (with progress bar)
    training_iter = create_sft_dataset_iterator(
        trajectories=trajectories,
        epochs=1,
        batch_size=4,
        peak_lr=2e-4,
        schedule_type="linear",
    )

    # Train with periodic evaluation
    for chunk_idx, chunk in enumerate(training_iter, 1):
        await model.train_sft(chunk.trajectories, chunk.config)

        # Run benchmark at checkpoints
        if chunk_idx in eval_at_chunks:
            progress_pct = int(100 * chunk_idx / total_chunks)
            print(f"\n[{progress_pct}%] Running benchmark...")

            metrics = await run_benchmark(model, show_progress=True)

            print(f"[{progress_pct}%] EM: {metrics['exact_match']:.2%}, F1: {metrics['f1']:.2%}, G: {metrics['grounded']:.2%}, P: {metrics['precision']:.2%}, R: {metrics['recall']:.2%}")

            # Log to wandb using the global training step (matches training logs)
            run.log({
                "eval/exact_match": metrics["exact_match"],
                "eval/f1": metrics["f1"],
                "eval/grounded": metrics["grounded"],
                "eval/precision": metrics["precision"],
                "eval/recall": metrics["recall"],
            }, step=chunk.config.global_step)

    run.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    asyncio.run(main())
