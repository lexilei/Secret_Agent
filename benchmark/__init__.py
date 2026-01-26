"""
MedCalc-Bench Benchmark System
==============================

A comprehensive benchmark for evaluating the ptool_framework against
the MedCalc-Bench dataset (NeurIPS 2024).

Usage:
    # Debug mode (first 10 instances)
    python -m benchmark.cli run --debug

    # Full benchmark
    python -m benchmark.cli run --full -e all

    # Generate report
    python -m benchmark.cli report ./benchmark_results
"""

from .config import ExperimentConfig, ABLATION_CONFIGS
from .runner import BenchmarkRunner

__all__ = [
    "ExperimentConfig",
    "ABLATION_CONFIGS",
    "BenchmarkRunner",
]

__version__ = "1.0.0"
