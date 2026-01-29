#!/usr/bin/env python
"""
Runner script for StrategyQA experiments.

Usage:
    python -m benchmark.experiments.strategyqa.run_experiments --experiment baseline --n 10
    python -m benchmark.experiments.strategyqa.run_experiments --experiment baseline_cot --n 10
    python -m benchmark.experiments.strategyqa.run_experiments --experiment decompose --n 10
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from benchmark.dataset.strategyqa_loader import StrategyQADataset
from benchmark.experiments.strategyqa.baseline import StrategyQABaseline, StrategyQABaselineCoT
from benchmark.experiments.strategyqa.decompose import StrategyQADecomposeOracle, StrategyQADecomposeLLM
from benchmark.experiments.strategyqa.react_exp import StrategyQAReAct, StrategyQAReActSimple


def run_experiment(
    experiment_name: str,
    model: str = "deepseek-v3-0324",
    n_instances: int = 10,
    split: str = "val",
    output_dir: str = "./strategyqa_results",
):
    """Run a StrategyQA experiment."""

    # Map experiment names to classes
    experiments = {
        "baseline": StrategyQABaseline,
        "baseline_cot": StrategyQABaselineCoT,
        "decompose_oracle": StrategyQADecomposeOracle,
        "decompose_llm": StrategyQADecomposeLLM,
        "react": StrategyQAReAct,
        "react_simple": StrategyQAReActSimple,
    }

    if experiment_name not in experiments:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(experiments.keys())}")

    # Load data
    print(f"Loading StrategyQA data...")
    dataset = StrategyQADataset()

    if split == "val":
        _, instances = dataset.load_with_split(val_ratio=0.2)
    elif split == "train":
        instances, _ = dataset.load_with_split(val_ratio=0.2)
    else:
        instances = dataset.load(split)

    # Limit instances
    if n_instances > 0:
        instances = instances[:n_instances]

    print(f"Running {experiment_name} on {len(instances)} instances with model {model}...")

    # Create experiment
    exp_class = experiments[experiment_name]
    experiment = exp_class(model=model)
    experiment.setup()

    # Run with progress
    start_time = time.time()

    def progress_callback(current, total):
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        print(f"\r  Progress: {current}/{total} ({current/total*100:.1f}%) - "
              f"ETA: {eta:.0f}s", end="", flush=True)

    results = experiment.run_batch(instances, progress_callback=progress_callback)
    print()

    # Get summary
    summary = experiment.get_summary()
    total_time = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {experiment_name} (model: {model})")
    print(f"{'='*60}")
    print(f"  Accuracy:    {summary['accuracy'] * 100:.1f}%")
    print(f"  Correct:     {summary['correct']}/{summary['total_instances']}")
    print(f"  Errors:      {summary['errors']}")
    print(f"  Predictions: True={summary['predictions']['true']}, "
          f"False={summary['predictions']['false']}, None={summary['predictions']['none']}")
    print(f"  Total cost:  ${summary['total_cost_usd']:.4f}")
    print(f"  Avg latency: {summary['avg_latency_ms']:.0f}ms")
    print(f"  Total time:  {total_time:.1f}s")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"{experiment_name}_{model}_{timestamp}.json"

    output_data = {
        "experiment": experiment_name,
        "model": model,
        "n_instances": len(instances),
        "split": split,
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Cleanup
    experiment.teardown()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run StrategyQA experiments")
    parser.add_argument("--experiment", "-e", default="baseline",
                        help="Experiment name (baseline, baseline_cot)")
    parser.add_argument("--model", "-m", default="deepseek-v3-0324",
                        help="Model to use")
    parser.add_argument("--n", "-n", type=int, default=10,
                        help="Number of instances (0 for all)")
    parser.add_argument("--split", "-s", default="val",
                        help="Data split (train, val, dev)")
    parser.add_argument("--output", "-o", default="./strategyqa_results",
                        help="Output directory")

    args = parser.parse_args()

    run_experiment(
        experiment_name=args.experiment,
        model=args.model,
        n_instances=args.n,
        split=args.split,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
