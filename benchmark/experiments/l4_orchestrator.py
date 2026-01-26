"""
L4 Experiment: AgentOrchestrator with specialized agents.

Uses multi-agent orchestration with specialized agents for different
calculator categories (equation-based vs rule-based).
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

from ptool_framework import ptool
from ptool_framework.orchestrator import (
    AgentOrchestrator,
    AgentSpec,
    HybridRouter,
    ExecutionMode,
)


# Additional specialized ptools for orchestrator

@ptool(model="deepseek-v3-0324", output_mode="structured")
def calculate_equation(
    formula_type: str,
    values: Dict[str, float],
) -> Dict[str, Any]:
    """
    Calculate a result using a mathematical equation/formula.

    For equation-based calculators like BMI, eGFR, BSA, etc.
    Apply the appropriate formula with the given values.

    Return:
    {
        "formula_used": "description of formula",
        "calculation": "step by step",
        "result": numeric_value,
        "unit": "unit"
    }
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="structured")
def evaluate_rules(
    scoring_system: str,
    patient_factors: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate a rule-based scoring system.

    For rule-based calculators like CHA2DS2-VASc, CURB-65, Wells, etc.
    Apply each rule and sum the points.

    Return:
    {
        "rules_applied": [{"rule": "description", "points": n}, ...],
        "total_score": numeric_value,
        "interpretation": "risk category"
    }
    """
    ...


class L4OrchestratorExperiment(BaseExperiment):
    """
    L4 Experiment: AgentOrchestrator with specialized agents.

    Routes tasks to specialized agents based on calculator category:
    - Equation agent for formula-based calculations
    - Rules agent for scoring systems
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.model = config.model
        self.orchestrator: Optional[AgentOrchestrator] = None

    def setup(self):
        """Initialize orchestrator with specialized agents."""
        # Ensure ptools are registered
        _ = (
            identify_calculator,
            extract_clinical_values,
            perform_calculation,
            calculate_equation,
            evaluate_rules,
        )

        # Create router with keywords for medical domains
        router = HybridRouter()
        router.rule_router.add_domain_keywords("equation", [
            "bmi", "egfr", "gfr", "bsa", "clearance", "dose", "rate",
            "body mass", "surface area", "filtration", "creatinine",
        ])
        router.rule_router.add_domain_keywords("rule", [
            "score", "risk", "cha2ds2", "vasc", "chads", "wells", "curb",
            "severity", "mortality", "prognosis", "staging",
        ])

        # Create orchestrator
        self.orchestrator = AgentOrchestrator(
            router=router,
            execution_mode=ExecutionMode.SEQUENTIAL,
            model=self.model,
            echo=False,
        )

        # Register equation-based agent
        self.orchestrator.register_agent(AgentSpec(
            name="equation_calculator",
            description="Handles equation-based medical calculations like BMI, eGFR, BSA, drug dosing",
            domains=["equation", "lab", "dosage", "physical"],
            available_ptools=[
                "identify_calculator",
                "extract_clinical_values",
                "calculate_equation",
            ],
            model=self.model,
            capabilities=["extraction", "calculation", "formula"],
            max_steps=6,
        ))

        # Register rule-based agent
        self.orchestrator.register_agent(AgentSpec(
            name="rule_calculator",
            description="Handles rule-based risk scores and clinical decision tools like CHA2DS2-VASc, Wells, CURB-65",
            domains=["rule", "risk", "severity", "diagnosis"],
            available_ptools=[
                "identify_calculator",
                "extract_clinical_values",
                "evaluate_rules",
            ],
            model=self.model,
            capabilities=["extraction", "scoring", "risk assessment"],
            max_steps=6,
        ))

        # Register general fallback agent
        self.orchestrator.register_agent(AgentSpec(
            name="general_calculator",
            description="General medical calculation agent for cases that don't fit other categories",
            domains=["general", "medical", "calculation"],
            available_ptools=[
                "identify_calculator",
                "extract_clinical_values",
                "perform_calculation",
            ],
            model=self.model,
            capabilities=["extraction", "calculation"],
            priority=-1,  # Lower priority than specialized agents
            max_steps=8,
        ))

        self._setup_complete = True

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run orchestrator on a single instance.
        """
        if self.orchestrator is None:
            self.setup()

        goal = f"""Medical Calculation Task

Patient Note:
{instance.patient_note}

Question: {instance.question}

Calculate the requested value and return the numeric result."""

        start_time = time.time()

        try:
            # Run orchestrator
            result = self.orchestrator.run(goal)
            latency_ms = (time.time() - start_time) * 1000

            # Extract answer from result
            predicted = None
            if result.answer:
                predicted = self.extract_number(str(result.answer))

            # Try to extract from trace steps if no direct answer
            if predicted is None and result.trace:
                for step in reversed(result.trace.steps):
                    if step.agent_result and step.agent_result.answer:
                        predicted = self.extract_number(str(step.agent_result.answer))
                        if predicted is not None:
                            break

            # Estimate tokens
            total_input = len(goal) // 4 + 300  # Base prompt
            total_output = 0

            # Add tokens from each step
            if result.trace:
                for step in result.trace.steps:
                    total_input += 200  # Routing decision
                    if step.agent_result and step.agent_result.trajectory:
                        for traj_step in step.agent_result.trajectory.steps:
                            if traj_step.thought:
                                total_input += len(traj_step.thought) // 4
                            if traj_step.observation and traj_step.observation.result:
                                total_output += len(str(traj_step.observation.result)) // 4

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

            num_steps = len(result.trace.steps) if result.trace else 0
            agents_used = result.agents_used if result.agents_used else []

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
                raw_response=str(result.answer),
                trace={
                    "orchestration": result.trace.to_dict() if result.trace else None,
                    "agents_used": agents_used,
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
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
