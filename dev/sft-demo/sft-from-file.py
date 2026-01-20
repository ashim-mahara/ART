"""Simple SFT training script."""

import asyncio

import art
from art.local import LocalBackend
from art.utils.sft import create_sft_dataset_iterator, iterate_file


async def main():
    backend = LocalBackend()
    model = art.TrainableModel(
        name="pii-art-qwen14-b-linear-2e-4-bs-4-ep-1",
        project="OP-unsloth-SDKtests",
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    await model.register(backend)

    # Load trajectories and train
    trajectories = list(iterate_file("dev/sft-demo/dataset.jsonl", epochs=1))

    for chunk in create_sft_dataset_iterator(
        trajectories=trajectories,
        epochs=1,
        batch_size=1,
        peak_lr=2e-4,
        schedule_type="linear",
    ):
        await model.train_sft(chunk.trajectories, chunk.config)

    print("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
