"""
Base experiment class and result types.

All experiments inherit from BaseExperiment and produce ExperimentResult objects.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import time
import traceback

from ..config import ExperimentConfig
from ..dataset.loader import MedCalcInstance


@dataclass
class ExperimentResult:
    """Result from running an experiment on a single instance."""

    # Instance identification
    instance_id: int
    calculator_name: str
    category: str

    # Answer
    predicted_answer: Optional[float]
    ground_truth: float

    # Correctness
    is_correct_exact: bool
    is_correct_tolerance: bool      # Within ±5%
    is_within_limits: bool          # Within dataset-defined limits

    # Performance metrics
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float

    # Framework-specific metrics
    num_steps: int = 0              # ReAct steps taken
    repair_attempts: int = 0
    repair_success: bool = False
    patterns_used: int = 0
    patterns_learned: int = 0

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Raw outputs
    raw_response: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None
    audit_info: Optional[Dict[str, Any]] = None  # For L3+ critic/repair details

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "instance_id": self.instance_id,
            "calculator_name": self.calculator_name,
            "category": self.category,
            "predicted_answer": self.predicted_answer,
            "ground_truth": self.ground_truth,
            "is_correct_exact": self.is_correct_exact,
            "is_correct_tolerance": self.is_correct_tolerance,
            "is_within_limits": self.is_within_limits,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "num_steps": self.num_steps,
            "repair_attempts": self.repair_attempts,
            "repair_success": self.repair_success,
            "patterns_used": self.patterns_used,
            "patterns_learned": self.patterns_learned,
            "error": self.error,
            "error_type": self.error_type,
            "timestamp": self.timestamp,
        }

    @property
    def success(self) -> bool:
        """Whether the prediction was correct (within tolerance)."""
        return self.is_correct_tolerance and self.error is None


class BaseExperiment(ABC):
    """
    Abstract base class for all experiments.

    Subclasses must implement run_instance() to process a single instance.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results: List[ExperimentResult] = []
        self._setup_complete = False

    def setup(self) -> None:
        """
        Perform one-time setup (load models, initialize components).

        Override in subclasses if needed.
        """
        self._setup_complete = True

    def teardown(self) -> None:
        """
        Clean up resources after experiment.

        Override in subclasses if needed.
        """
        pass

    @abstractmethod
    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run experiment on a single instance.

        Args:
            instance: MedCalc problem instance

        Returns:
            ExperimentResult with prediction and metrics
        """
        pass

    def run_batch(
        self,
        instances: List[MedCalcInstance],
        progress_callback: Optional[callable] = None,
    ) -> List[ExperimentResult]:
        """
        Run experiment on a batch of instances.

        Args:
            instances: List of MedCalc instances
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of ExperimentResult objects
        """
        if not self._setup_complete:
            self.setup()

        self.results = []
        total = len(instances)

        for i, instance in enumerate(instances):
            try:
                result = self.run_instance(instance)
            except Exception as e:
                # Create error result
                result = ExperimentResult(
                    instance_id=instance.row_number,
                    calculator_name=instance.calculator_name,
                    category=instance.category,
                    predicted_answer=None,
                    ground_truth=instance.ground_truth_answer,
                    is_correct_exact=False,
                    is_correct_tolerance=False,
                    is_within_limits=False,
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    error=str(e),
                    error_type=type(e).__name__,
                )

            self.results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return self.results

    def evaluate_answer(
        self,
        predicted: Optional[float],
        ground_truth: float,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        tolerance_pct: float = 5.0,
    ) -> Dict[str, bool]:
        """
        Evaluate predicted answer against ground truth.

        Args:
            predicted: Predicted value
            ground_truth: True value
            lower_limit: Optional lower bound
            upper_limit: Optional upper bound
            tolerance_pct: Percentage tolerance (default 5%)

        Returns:
            Dict with exact, tolerance, and limits correctness
        """
        if predicted is None:
            return {
                "exact": False,
                "tolerance": False,
                "limits": False,
            }

        # Exact match (with small epsilon for floating point)
        exact = abs(predicted - ground_truth) < 0.01

        # Within percentage tolerance
        if ground_truth != 0:
            tolerance_value = abs(ground_truth) * (tolerance_pct / 100)
        else:
            tolerance_value = 0.05  # Small absolute tolerance for zero
        within_tolerance = abs(predicted - ground_truth) <= tolerance_value

        # Within defined limits
        within_limits = True
        if lower_limit is not None and upper_limit is not None:
            within_limits = lower_limit <= predicted <= upper_limit

        return {
            "exact": exact,
            "tolerance": within_tolerance,
            "limits": within_limits,
        }

    def extract_number(self, text: str) -> Optional[float]:
        """
        Extract numeric answer from LLM response.

        Args:
            text: Raw LLM response text

        Returns:
            Extracted number or None
        """
        import re

        if not text:
            return None

        # Clean the text
        text = text.strip()

        # Try to find a number in common formats
        patterns = [
            r'(?:answer|result|value|score)(?:\s*(?:is|=|:))?\s*(-?\d+\.?\d*)',  # "answer is 5.2"
            r'(-?\d+\.?\d*)\s*(?:kg/m|mL/min|mg/dL|%|points?|score)?$',  # Number at end
            r'^(-?\d+\.?\d*)$',  # Just a number
            r'(?:^|\s)(-?\d+\.?\d*)(?:\s|$)',  # Number surrounded by whitespace
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # Last resort: find any number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            # Return the last number (often the final answer)
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for completed results.

        Returns:
            Dictionary with accuracy, cost, latency stats
        """
        if not self.results:
            return {}

        total = len(self.results)
        correct_exact = sum(1 for r in self.results if r.is_correct_exact)
        correct_tolerance = sum(1 for r in self.results if r.is_correct_tolerance)
        errors = sum(1 for r in self.results if r.error is not None)

        latencies = [r.latency_ms for r in self.results if r.error is None]
        costs = [r.cost_usd for r in self.results]
        tokens = [r.total_tokens for r in self.results]

        return {
            "experiment": self.config.name,
            "level": self.config.level.value,
            "total_instances": total,
            "correct_exact": correct_exact,
            "correct_tolerance": correct_tolerance,
            "errors": errors,
            "accuracy_exact": correct_exact / total if total > 0 else 0,
            "accuracy_tolerance": correct_tolerance / total if total > 0 else 0,
            "error_rate": errors / total if total > 0 else 0,
            "total_cost_usd": sum(costs),
            "avg_cost_usd": sum(costs) / total if total > 0 else 0,
            "total_tokens": sum(tokens),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        }
