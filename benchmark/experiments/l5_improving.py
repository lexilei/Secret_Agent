"""
L5 Experiment: SelfImprovingAgent with experience learning.

Uses the SelfImprovingAgent to learn from execution experience,
building pattern memory that improves over subsequent runs.
"""

import time
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..config import ExperimentConfig
from ..dataset.loader import MedCalcInstance
from ..metrics.cost import calculate_cost
from ..metrics.accuracy import calculate_accuracy
from .base import BaseExperiment, ExperimentResult
from .l3_react import (
    get_medcalc_ptools,
    identify_calculator,
    extract_clinical_values,
    perform_calculation,
)

from ptool_framework import ReActAgent
from ptool_framework.self_improving import (
    SelfImprovingAgent,
    PatternMemory,
    PatternExtractor,
)
from ptool_framework.critic import TraceCritic
from ptool_framework.repair import RepairAgent
from ptool_framework.audit import NonEmptyTraceAudit, NoFailedStepsAudit


class L5ImprovingExperiment(BaseExperiment):
    """
    L5 Experiment: SelfImprovingAgent with experience learning.

    Wraps a ReAct agent with self-improvement capabilities:
    - Learns patterns from successful executions
    - Avoids patterns from failed executions
    - Uses relevant patterns to enhance prompts
    - Tracks improvement over time
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model
        self.max_steps = config.max_steps
        self.agent: Optional[SelfImprovingAgent] = None
        self.pattern_memory: Optional[PatternMemory] = None

        # Track learning progress
        self.patterns_learned = 0
        self.patterns_used = 0
        self.accuracy_history: List[bool] = []

    def setup(self):
        """Initialize self-improving agent."""
        # Ensure ptools are registered
        _ = identify_calculator, extract_clinical_values, perform_calculation

        # Create pattern memory (fresh for this experiment)
        self.pattern_memory = PatternMemory(
            path=Path(__file__).parent.parent.parent / ".benchmark_patterns"
        )

        # Create base ReAct agent
        base_agent = ReActAgent(
            available_ptools=get_medcalc_ptools(),
            model=self.model,
            max_steps=self.max_steps,
            echo=False,
        )

        # Create critic for evaluation
        critic = TraceCritic(
            audits=[NonEmptyTraceAudit(), NoFailedStepsAudit()],
            model=self.model,
            accept_threshold=0.6,
            repair_threshold=0.3,
        )

        # Create repair agent
        repair_agent = RepairAgent(
            critic=critic,
            max_attempts=2,
            model=self.model,
        )

        # Create self-improving agent
        self.agent = SelfImprovingAgent(
            base_agent=base_agent,
            pattern_memory=self.pattern_memory,
            pattern_extractor=PatternExtractor(model=self.model),
            critic=critic,
            repair_agent=repair_agent,
            learn_from_success=True,
            learn_from_failure=True,
            auto_repair=True,
            max_repair_attempts=2,
            max_positive_patterns=3,
            max_negative_patterns=2,
            max_heuristics=2,
            min_pattern_relevance=0.3,
            enable_decay=True,
            decay_interval_runs=20,
            echo=False,
        )

        self._setup_complete = True

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run self-improving agent on a single instance.
        """
        if self.agent is None:
            self.setup()

        goal = f"""Patient Note:
{instance.patient_note}

Task: {instance.question}

Use the available tools to:
1. Identify what calculator is needed
2. Extract the required values from the patient note
3. Perform the calculation
4. Return the numeric result"""

        start_time = time.time()

        try:
            # Run self-improving agent
            result = self.agent.run(goal)
            latency_ms = (time.time() - start_time) * 1000

            # Extract answer
            predicted = None
            if result.answer:
                predicted = self.extract_number(str(result.answer))

            if predicted is None and result.trajectory:
                for step in reversed(result.trajectory.steps):
                    if step.observation and step.observation.result:
                        obs_result = step.observation.result
                        if isinstance(obs_result, dict) and "result" in obs_result:
                            predicted = obs_result["result"]
                            if predicted is not None:
                                try:
                                    predicted = float(predicted)
                                except (ValueError, TypeError):
                                    predicted = self.extract_number(str(predicted))
                            break

            # Estimate tokens (includes pattern context)
            total_input = len(goal) // 4 + 500
            total_output = 0

            for step in result.trajectory.steps if result.trajectory else []:
                if step.thought:
                    total_input += len(step.thought) // 4
                if step.observation and step.observation.result:
                    total_output += len(str(step.observation.result)) // 4

            # Add pattern context tokens
            patterns_used = len(self.agent._last_used_patterns) if hasattr(self.agent, '_last_used_patterns') else 0
            total_input += patterns_used * 100  # ~100 tokens per pattern

            cost_metrics = calculate_cost(
                input_tokens=total_input,
                output_tokens=total_output,
                model=self.model,
            )

            accuracy = calculate_accuracy(
                predicted=predicted,
                ground_truth=instance.ground_truth_answer,
                lower_limit=instance.lower_limit,
                upper_limit=instance.upper_limit,
                output_type=instance.output_type,
                category=instance.category,
            )

            # Track learning
            self.accuracy_history.append(accuracy.is_within_tolerance)

            # Get pattern stats
            memory_stats = self.pattern_memory.get_stats() if self.pattern_memory else {}
            patterns_learned = memory_stats.get("total_patterns", 0)

            num_steps = len(result.trajectory.steps) if result.trajectory else 0

            return ExperimentResult(
                instance_id=instance.row_number,
                calculator_name=instance.calculator_name,
                category=instance.category,
                predicted_answer=predicted,
                ground_truth=instance.ground_truth_answer,
                is_correct_exact=accuracy.is_exact_match,
                is_correct_tolerance=accuracy.is_within_tolerance,
                is_within_limits=accuracy.is_within_limits,
                latency_ms=latency_ms,
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
                cost_usd=cost_metrics.cost_usd,
                num_steps=num_steps,
                patterns_used=patterns_used,
                patterns_learned=patterns_learned,
                raw_response=str(result.answer),
                trace=result.trajectory.to_dict() if result.trajectory else None,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.accuracy_history.append(False)

            return ExperimentResult(
                instance_id=instance.row_number,
                calculator_name=instance.calculator_name,
                category=instance.category,
                predicted_answer=None,
                ground_truth=instance.ground_truth_answer,
                is_correct_exact=False,
                is_correct_tolerance=False,
                is_within_limits=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
                error_type=type(e).__name__,
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with learning curve data."""
        summary = super().get_summary()

        # Pattern stats
        if self.pattern_memory:
            stats = self.pattern_memory.get_stats()
            summary["patterns_total"] = stats.get("total_patterns", 0)
            summary["patterns_by_type"] = stats.get("patterns_by_type", {})

        # Learning curve (accuracy over batches of 10)
        batch_size = 10
        learning_curve = []
        for i in range(0, len(self.accuracy_history), batch_size):
            batch = self.accuracy_history[i:i + batch_size]
            if batch:
                learning_curve.append({
                    "batch": i // batch_size + 1,
                    "accuracy": sum(batch) / len(batch),
                    "instances": len(batch),
                })
        summary["learning_curve"] = learning_curve

        return summary

    def get_improvement_metrics(self):
        """Get metrics from the self-improving agent."""
        if self.agent:
            return self.agent.get_improvement_metrics()
        return None
