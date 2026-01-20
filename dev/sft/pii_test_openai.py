#!/usr/bin/env python3
"""
PII Redaction Benchmark using OpenPipe API.

Evaluates a model's ability to identify PII strings by comparing against golden labels.
Reports precision, recall, and F1 scores.

Usage:
    uv run python dev/sft/pii_test_openai.py
    uv run python dev/sft/pii_test_openai.py --model "openpipe:other-model"
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field

import wandb
from openai import AsyncOpenAI
from tqdm import tqdm


# OpenPipe API configuration
OPENPIPE_BASE_URL = "https://app.openpipe.ai/api/v1"
OPENPIPE_API_KEY = "opk_28a838773df0beba8ff522c61a3538edf26a290c1d"
DEFAULT_MODEL = "openpipe:pii-lm-bs-1"

# Wandb configuration
DEFAULT_WANDB_PROJECT = "OP-unsloth-SDKtests"
DEFAULT_STEP = 3931


CONCURRENCY = 10


@dataclass
class EvalMetrics:
    """Metrics for evaluation."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    exact_matches: int = 0
    grounded_entries: int = 0  # Entries where all predicted PII exists in input
    total_entries: int = 0
    parse_errors: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def exact_match_accuracy(self) -> float:
        if self.total_entries == 0:
            return 0.0
        return self.exact_matches / self.total_entries

    @property
    def grounded(self) -> float:
        """Rate of entries where all predicted PII strings exist in the input."""
        if self.total_entries == 0:
            return 0.0
        return self.grounded_entries / self.total_entries

    async def add(self, tp: int, fp: int, fn: int, is_grounded: bool):
        async with self._lock:
            self.true_positives += tp
            self.false_positives += fp
            self.false_negatives += fn
            self.total_entries += 1
            if fp == 0 and fn == 0:
                self.exact_matches += 1
            if is_grounded:
                self.grounded_entries += 1

    async def add_error(self):
        async with self._lock:
            self.parse_errors += 1
            self.total_entries += 1


def load_test_data(filepath: str) -> list:
    """Load test entries from a JSONL file."""
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_expected_pii(entry: dict) -> set[str]:
    """Extract expected PII strings from the golden assistant response."""
    messages = entry.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                fields = data.get("fields_to_redact", [])
                # Normalize strings for comparison
                return {f.get("string", "").strip() for f in fields if f.get("string")}
            except json.JSONDecodeError:
                return set()
    return set()


def extract_predicted_pii(model_output: str) -> set[str] | None:
    """Extract predicted PII strings from model output. Returns None on parse error."""
    try:
        data = json.loads(model_output)
        fields = data.get("fields_to_redact", [])
        return {f.get("string", "").strip() for f in fields if f.get("string")}
    except json.JSONDecodeError:
        return None


def get_input_messages(entry: dict) -> list[dict]:
    """Get input messages (system + user only, no assistant)."""
    messages = entry.get("messages", [])
    return [m for m in messages if m.get("role") in ("system", "user")]


def get_input_text(entry: dict) -> str:
    """Get concatenated input text from system and user messages."""
    messages = get_input_messages(entry)
    return " ".join(m.get("content", "") for m in messages)


def check_grounded(predicted_pii: set[str], input_text: str) -> bool:
    """Check if all predicted PII strings exist in the input text."""
    for pii_string in predicted_pii:
        if pii_string not in input_text:
            return False
    return True


REQUEST_TIMEOUT = 60  # seconds


async def evaluate_entry(
    client,
    model_name: str,
    entry: dict,
    entry_idx: int,
    metrics: EvalMetrics,
    semaphore: asyncio.Semaphore,
    verbose: bool,
) -> None:
    """Evaluate a single entry."""
    async with semaphore:
        input_messages = get_input_messages(entry)
        input_text = get_input_text(entry)
        expected_pii = extract_expected_pii(entry)
        response_format = entry.get("response_format")

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=input_messages,
                    temperature=0.0,
                    response_format=response_format if response_format else None,
                ),
                timeout=REQUEST_TIMEOUT,
            )
            model_output = response.choices[0].message.content or ""

            if not model_output:
                await metrics.add_error()
                print(f"[{entry_idx+1}] EMPTY RESPONSE")
                return

            predicted_pii = extract_predicted_pii(model_output)

            if predicted_pii is None:
                await metrics.add_error()
                print(f"[{entry_idx+1}] PARSE ERROR: {model_output[:300]}...")
                return

            # Calculate matches
            tp = len(expected_pii & predicted_pii)
            fp = len(predicted_pii - expected_pii)
            fn = len(expected_pii - predicted_pii)

            # Check if all predicted PII strings exist in the input (grounded)
            is_grounded = check_grounded(predicted_pii, input_text)

            await metrics.add(tp, fp, fn, is_grounded)

            if verbose:
                status = "OK" if fp == 0 and fn == 0 else "MISS"
                grounded_str = "G" if is_grounded else "H"  # G=grounded, H=hallucinated
                print(f"[{entry_idx+1}] {status} {grounded_str} - TP:{tp} FP:{fp} FN:{fn}")
                if fp > 0:
                    print(f"  Extra: {predicted_pii - expected_pii}")
                if fn > 0:
                    print(f"  Missing: {expected_pii - predicted_pii}")
                if not is_grounded:
                    # Show which PII strings are hallucinated
                    hallucinated = {p for p in predicted_pii if p not in input_text}
                    print(f"  Hallucinated: {hallucinated}")

        except asyncio.TimeoutError:
            await metrics.add_error()
            print(f"[{entry_idx+1}] TIMEOUT after {REQUEST_TIMEOUT}s")

        except Exception as e:
            await metrics.add_error()
            print(f"[{entry_idx+1}] ERROR: {type(e).__name__}: {e}")


async def run_benchmark(
    client: AsyncOpenAI,
    model_name: str,
    test_file: str = "dev/sft/pii_test.jsonl",
    concurrency: int = CONCURRENCY,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict[str, float]:
    """
    Run PII benchmark on a model and return metrics.

    Args:
        client: AsyncOpenAI client configured for the API endpoint
        model_name: Name of the model to use for inference
        test_file: Path to the test JSONL file
        concurrency: Number of parallel requests
        verbose: Print detailed results for each entry
        show_progress: Show progress bar

    Returns:
        Dictionary with precision, recall, f1, and parse_errors
    """

    entries = load_test_data(test_file)

    metrics = EvalMetrics()
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        evaluate_entry(client, model_name, entry, i, metrics, semaphore, verbose)
        for i, entry in enumerate(entries)
    ]

    if show_progress:
        pbar = tqdm(total=len(tasks), desc="Evaluating", disable=verbose)

        async def run_with_progress(task):
            result = await task
            pbar.update(1)
            pbar.set_postfix({"F1": f"{metrics.f1:.1%}"})
            return result

        await asyncio.gather(*[run_with_progress(task) for task in tasks])
        pbar.close()
    else:
        await asyncio.gather(*tasks)

    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "exact_match": metrics.exact_match_accuracy,
        "grounded": metrics.grounded,
        "parse_errors": metrics.parse_errors,
    }


async def main():
    parser = argparse.ArgumentParser(description="PII Redaction Benchmark using OpenPipe API")
    parser.add_argument(
        "--test-file",
        default="dev/sft/pii_test.jsonl",
        help="Path to the test JSONL file",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to evaluate",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each entry",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help=f"Number of parallel requests (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Wandb run name to log metrics to (also used as run ID)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help=f"Step number to log metrics at (default: {DEFAULT_STEP})",
    )

    args = parser.parse_args()

    # Initialize OpenAI client with OpenPipe endpoint
    client = AsyncOpenAI(
        base_url=OPENPIPE_BASE_URL,
        api_key=OPENPIPE_API_KEY,
    )
    model_name = args.model

    print(f"Testing model: {model_name}")
    print(f"Endpoint: {OPENPIPE_BASE_URL}")

    # Load test data
    print(f"Loading test data from {args.test_file}...")
    entries = load_test_data(args.test_file)

    if args.max_entries:
        entries = entries[: args.max_entries]

    print(f"Evaluating {len(entries)} entries with {args.concurrency} concurrent requests...")
    print("-" * 60)

    # Run evaluation with concurrency control
    metrics = EvalMetrics()
    semaphore = asyncio.Semaphore(args.concurrency)

    # Create tasks
    tasks = [
        evaluate_entry(client, model_name, entry, i, metrics, semaphore, args.verbose)
        for i, entry in enumerate(entries)
    ]

    # Run with progress bar
    pbar = tqdm(total=len(tasks), desc="Evaluating", disable=args.verbose)

    async def run_with_progress(task):
        result = await task
        pbar.update(1)
        pbar.set_postfix({"F1": f"{metrics.f1:.1%}", "P": f"{metrics.precision:.1%}", "R": f"{metrics.recall:.1%}"})
        return result

    await asyncio.gather(*[run_with_progress(task) for task in tasks])
    pbar.close()

    # Print summary
    print("-" * 60)
    print("Results Summary:")
    print(f"  Total entries:     {len(entries)}")
    print(f"  Exact matches:     {metrics.exact_matches}")
    print(f"  Grounded entries:  {metrics.grounded_entries}")
    print(f"  Parse errors:      {metrics.parse_errors}")
    print(f"  True positives:    {metrics.true_positives}")
    print(f"  False positives:   {metrics.false_positives}")
    print(f"  False negatives:   {metrics.false_negatives}")
    print()
    print(f"  Exact Match:       {metrics.exact_match_accuracy:.2%}")
    print(f"  Grounded:          {metrics.grounded:.2%}")
    print(f"  Precision:         {metrics.precision:.2%}")
    print(f"  Recall:            {metrics.recall:.2%}")
    print(f"  F1 Score:          {metrics.f1:.2%}")

    # Log to wandb if run name provided
    if args.wandb_run:
        print("-" * 60)
        print(f"Logging to wandb project={DEFAULT_WANDB_PROJECT} run={args.wandb_run} step={args.step}")
        run = wandb.init(
            project=DEFAULT_WANDB_PROJECT,
            name=args.wandb_run,
            id=args.wandb_run,
            resume="allow",
        )
        run.log({
            "eval/exact_match": metrics.exact_match_accuracy,
            "eval/f1": metrics.f1,
            "eval/grounded": metrics.grounded,
            "eval/precision": metrics.precision,
            "eval/recall": metrics.recall,
        }, step=args.step)
        run.finish()
        print("Logged to wandb successfully")

    return 0 if metrics.false_positives == 0 and metrics.false_negatives == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
