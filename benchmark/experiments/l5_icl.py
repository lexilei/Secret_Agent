"""
L5+ Experiment: SelfImproving with ICL from training set.

Pre-populates the pattern memory with in-context learning examples
from the training set, giving the agent a head start.
"""

import time
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..config import ExperimentConfig
from ..dataset.loader import MedCalcInstance, MedCalcDataset
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
    PatternType,
    LearnedPattern,
)
from ptool_framework.critic import TraceCritic
from ptool_framework.repair import RepairAgent
from ptool_framework.audit import NonEmptyTraceAudit, NoFailedStepsAudit


class L5ICLExperiment(BaseExperiment):
    """
    L5+ Experiment: SelfImproving with ICL from training.

    Pre-populates pattern memory with solved examples from the
    training set, providing in-context learning examples.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        train_data: Optional[List[MedCalcInstance]] = None,
    ):
        super().__init__(config)
        self.model = config.model
        self.max_steps = config.max_steps
        self.icl_per_calculator = config.icl_examples_per_calculator
        self.train_data = train_data
        self.agent: Optional[SelfImprovingAgent] = None
        self.pattern_memory: Optional[PatternMemory] = None

        # Stats
        self.icl_examples_loaded = 0
        self.patterns_used = 0
        self.accuracy_history: List[bool] = []

    def setup(self):
        """Initialize with ICL examples from training."""
        # Ensure ptools are registered
        _ = identify_calculator, extract_clinical_values, perform_calculation

        # Create pattern memory
        self.pattern_memory = PatternMemory(
            path=Path(__file__).parent.parent.parent / ".benchmark_icl_patterns"
        )

        # Load training data if not provided
        if self.train_data is None:
            dataset = MedCalcDataset()
            self.train_data = dataset.load("train")

        # Pre-populate with ICL examples
        self._populate_icl_examples()

        # Create base agent
        base_agent = ReActAgent(
            available_ptools=get_medcalc_ptools(),
            model=self.model,
            max_steps=self.max_steps,
            echo=False,
        )

        # Create critic
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
            max_positive_patterns=5,  # More patterns since we have ICL
            max_negative_patterns=2,
            max_heuristics=2,
            min_pattern_relevance=0.2,  # Lower threshold for ICL
            enable_decay=False,  # Don't decay ICL examples
            echo=False,
        )

        self._setup_complete = True

    def _populate_icl_examples(self):
        """Create ICL patterns from training examples."""
        from collections import defaultdict
        import uuid

        # Group by calculator
        by_calculator: Dict[str, List[MedCalcInstance]] = defaultdict(list)
        for instance in self.train_data:
            by_calculator[instance.calculator_name].append(instance)

        # Select best examples per calculator
        for calc_name, instances in by_calculator.items():
            # Take first N instances with good explanations
            good_instances = [
                i for i in instances
                if i.ground_truth_explanation and len(i.ground_truth_explanation) > 50
            ][:self.icl_per_calculator]

            # If not enough good ones, take any
            if len(good_instances) < self.icl_per_calculator:
                good_instances = instances[:self.icl_per_calculator]

            for instance in good_instances:
                # Create ICL pattern content
                icl_content = f"""Example: {calc_name}

Patient Note:
{instance.patient_note[:500]}...

Question: {instance.question}

Solution:
{instance.ground_truth_explanation}

Answer: {instance.ground_truth_answer}"""

                # Create pattern
                pattern = LearnedPattern(
                    pattern_id=f"icl_{calc_name}_{uuid.uuid4().hex[:8]}",
                    pattern_type=PatternType.POSITIVE,
                    content=icl_content,
                    source_trace_id="training_data",
                    domain="medcalc",
                    ptool_name=None,
                    goal_pattern=calc_name.lower().replace(" ", "_"),
                    confidence=1.0,  # High confidence for training examples
                    metadata={
                        "calculator": calc_name,
                        "from_training": True,
                        "ground_truth": instance.ground_truth_answer,
                    },
                )

                self.pattern_memory.store_pattern(pattern)
                self.icl_examples_loaded += 1

        print(f"Loaded {self.icl_examples_loaded} ICL examples from {len(by_calculator)} calculators")

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run self-improving agent with ICL on a single instance.
        """
        if self.agent is None:
            self.setup()

        goal = f"""Patient Note:
{instance.patient_note}

Task: {instance.question}

Use the available tools and your knowledge from similar examples to:
1. Identify what calculator is needed
2. Extract the required values from the patient note
3. Perform the calculation following the correct formula
4. Return the numeric result"""

        start_time = time.time()

        try:
            # Run agent
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

            # Estimate tokens
            total_input = len(goal) // 4 + 500
            total_output = 0

            for step in result.trajectory.steps if result.trajectory else []:
                if step.thought:
                    total_input += len(step.thought) // 4
                if step.observation and step.observation.result:
                    total_output += len(str(step.observation.result)) // 4

            # ICL patterns add significant context
            patterns_used = len(self.agent._last_used_patterns) if hasattr(self.agent, '_last_used_patterns') else 0
            total_input += patterns_used * 200  # ICL examples are longer

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

            self.accuracy_history.append(accuracy.is_within_tolerance)

            memory_stats = self.pattern_memory.get_stats() if self.pattern_memory else {}

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
                patterns_learned=memory_stats.get("total_patterns", 0),
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
        """Get summary with ICL stats."""
        summary = super().get_summary()
        summary["icl_examples_loaded"] = self.icl_examples_loaded

        if self.pattern_memory:
            stats = self.pattern_memory.get_stats()
            summary["patterns_total"] = stats.get("total_patterns", 0)

        # Learning curve
        batch_size = 10
        learning_curve = []
        for i in range(0, len(self.accuracy_history), batch_size):
            batch = self.accuracy_history[i:i + batch_size]
            if batch:
                learning_curve.append({
                    "batch": i // batch_size + 1,
                    "accuracy": sum(batch) / len(batch),
                })
        summary["learning_curve"] = learning_curve

        return summary
