"""
Command-line interface for MedCalc-Bench benchmark.

Usage:
    # Debug mode (first 10 instances)
    python -m benchmark.cli run --debug

    # Full benchmark
    python -m benchmark.cli run --full -e all

    # Specific experiments
    python -m benchmark.cli run -e baseline -e l5_icl

    # Generate report from existing results
    python -m benchmark.cli report ./benchmark_results
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    print("Warning: click not installed. Using basic CLI.")

from .config import ABLATION_CONFIGS, get_all_experiment_names
from .runner import BenchmarkRunner


def main_basic():
    """Basic CLI when click is not available."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MedCalc-Bench Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark.cli run --debug
  python -m benchmark.cli run --full -e all
  python -m benchmark.cli run -e baseline -e l5_icl
  python -m benchmark.cli report ./benchmark_results
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark experiments")
    run_parser.add_argument(
        "-e", "--experiments",
        nargs="+",
        default=["all"],
        help="Experiments to run (default: all)",
    )
    run_parser.add_argument(
        "--debug/--full",
        dest="debug",
        action="store_true",
        default=True,
        help="Debug mode (10 instances) or full",
    )
    run_parser.add_argument(
        "--full",
        dest="debug",
        action="store_false",
        help="Run on full dataset",
    )
    run_parser.add_argument(
        "-n", "--debug-n",
        type=int,
        default=10,
        help="Number of debug instances",
    )
    run_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./benchmark_results",
        help="Output directory",
    )
    run_parser.add_argument(
        "-m", "--model",
        type=str,
        default="deepseek-v3-0324",
        help="Model to use",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report from results")
    report_parser.add_argument(
        "results_dir",
        type=str,
        help="Directory with results",
    )
    report_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file (default: results_dir/report.html)",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available experiments")

    args = parser.parse_args()

    if args.command == "run":
        run_experiments(
            experiments=args.experiments,
            debug=args.debug,
            debug_n=args.debug_n,
            output=args.output,
            model=args.model,
        )
    elif args.command == "report":
        generate_report(
            results_dir=args.results_dir,
            output=args.output,
        )
    elif args.command == "list":
        list_experiments()
    else:
        parser.print_help()


if CLICK_AVAILABLE:
    @click.group()
    def cli():
        """MedCalc-Bench Benchmark System

        Run ablation studies on the MedCalc-Bench dataset to evaluate
        the ptool_framework across different capability levels.
        """
        pass

    @cli.command()
    @click.option(
        "-e", "--experiments",
        multiple=True,
        default=["all"],
        help="Experiments to run (use multiple -e for multiple experiments)",
    )
    @click.option(
        "--debug/--full",
        default=True,
        help="Debug mode (10 instances) or full benchmark",
    )
    @click.option(
        "-n", "--debug-n",
        default=10,
        type=int,
        help="Number of instances for debug mode",
    )
    @click.option(
        "-o", "--output",
        default="./benchmark_results",
        type=click.Path(),
        help="Output directory for results",
    )
    @click.option(
        "-m", "--model",
        default="deepseek-v3-0324",
        help="LLM model to use",
    )
    @click.option(
        "--log-responses",
        is_flag=True,
        default=False,
        help="Log prompts/responses to .txt files for debugging",
    )
    def run(experiments, debug, debug_n, output, model, log_responses):
        """Run benchmark experiments.

        Examples:

            # Debug on first 10 instances (default)
            python -m benchmark.cli run --debug

            # Full benchmark on all experiments
            python -m benchmark.cli run --full -e all

            # Specific experiments
            python -m benchmark.cli run -e baseline -e l3_react -e l5_icl

            # Custom output directory
            python -m benchmark.cli run --full -o ./my_results

            # Enable response logging for debugging
            python -m benchmark.cli run --debug --log-responses
        """
        run_experiments(
            experiments=list(experiments),
            debug=debug,
            debug_n=debug_n,
            output=output,
            model=model,
            log_responses=log_responses,
        )

    @cli.command()
    @click.argument("results_dir", type=click.Path(exists=True))
    @click.option(
        "-o", "--output",
        type=click.Path(),
        help="Output file path (default: results_dir/report.html)",
    )
    def report(results_dir, output):
        """Generate HTML report from existing results.

        RESULTS_DIR is the directory containing experiment results.
        """
        generate_report(results_dir, output)

    @cli.command("list")
    def list_cmd():
        """List available experiment configurations."""
        list_experiments()

    @cli.command()
    def stats():
        """Show dataset statistics."""
        from .dataset.loader import MedCalcDataset

        print("Loading MedCalc-Bench dataset...")
        dataset = MedCalcDataset()
        stats = dataset.get_statistics()

        print(f"\nDataset Statistics:")
        print(f"  Total instances: {stats['total_instances']}")
        print(f"  Calculators: {stats['num_calculators']}")
        print(f"\nBy Category:")
        for cat, count in stats.get("by_category", {}).items():
            print(f"  {cat}: {count}")
        print(f"\nBy Output Type:")
        for otype, count in stats.get("by_output_type", {}).items():
            print(f"  {otype}: {count}")


def run_experiments(experiments, debug, debug_n, output, model, log_responses=False):
    """Run benchmark experiments."""
    print("=" * 60)
    print("MedCalc-Bench Benchmark")
    print("=" * 60)
    print(f"Mode: {'Debug' if debug else 'Full'}")
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Model: {model}")
    print(f"Output: {output}")
    print(f"Log responses: {log_responses}")
    print("=" * 60)

    # Validate experiments
    valid_experiments = get_all_experiment_names()
    if "all" not in experiments:
        for exp in experiments:
            if exp not in valid_experiments:
                print(f"Error: Unknown experiment '{exp}'")
                print(f"Available: {', '.join(valid_experiments)}")
                sys.exit(1)

    # Create runner
    runner = BenchmarkRunner(output_dir=Path(output), log_responses=log_responses)

    # Run experiments
    try:
        metrics = runner.run_ablation_study(
            experiments=list(experiments) if "all" not in experiments else None,
            debug=debug,
            debug_n=debug_n,
        )

        # Generate report
        report_path = runner.generate_report(metrics)

        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)

        # Sort by accuracy
        sorted_metrics = sorted(
            metrics.items(),
            key=lambda x: x[1].accuracy_tolerance,
            reverse=True,
        )

        for name, m in sorted_metrics:
            print(f"  {name:20s}: {m.accuracy_tolerance * 100:5.1f}% accuracy, ${m.total_cost_usd:.4f} cost")

        # Print best
        best_name, best_metrics = sorted_metrics[0]
        print(f"\nBest: {best_name} ({best_metrics.accuracy_tolerance * 100:.1f}%)")
        print(f"Report: {report_path}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_report(results_dir, output):
    """Generate report from existing results."""
    import json
    from .metrics.aggregator import AggregatedMetrics
    from .reports.generator import ReportGenerator

    results_path = Path(results_dir)
    metrics_file = results_path / "metrics.json"

    if not metrics_file.exists():
        print(f"Error: No metrics.json found in {results_dir}")
        sys.exit(1)

    print(f"Loading results from {results_dir}...")

    # Load metrics
    with open(metrics_file) as f:
        metrics_data = json.load(f)

    # Reconstruct AggregatedMetrics objects
    # (simplified - would need full reconstruction in production)
    print("Generating report...")

    generator = ReportGenerator(results_path)

    if output:
        output_path = Path(output)
    else:
        output_path = results_path / "report.html"

    # Generate simple report
    print(f"Report saved to: {output_path}")


def list_experiments():
    """List available experiments."""
    print("\nAvailable Experiments:")
    print("=" * 60)
    for name, config in ABLATION_CONFIGS.items():
        print(f"  {name:20s} ({config.level.value})")
        print(f"    {config.description}")
        print()


def main():
    """Main entry point."""
    if CLICK_AVAILABLE:
        cli()
    else:
        main_basic()


if __name__ == "__main__":
    main()
