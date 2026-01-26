"""
L4 Experiment: Adaptive Orchestration with Difficulty-Aware Routing.

Routes tasks to different framework levels (L2/L4/L3) based on estimated difficulty.
Designed for extensibility - routing strategy can be swapped to support:
- Difficulty → Level routing (current)
- Difficulty → Model routing (future)
- Task type → Specialist agent routing (future)

Key insight: Not all tasks need the same level of sophistication.
Simple tasks can use cheap/fast approaches; hard tasks justify expensive approaches.

Research question: Can automatic level selection achieve L3 accuracy at L2 cost?
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Type, Literal
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..config import ExperimentConfig, ABLATION_CONFIGS
from ..dataset.loader import MedCalcInstance
from ..metrics.cost import calculate_cost
from ..metrics.accuracy import calculate_accuracy
from .base import BaseExperiment, ExperimentResult

from ptool_framework import ptool


# =============================================================================
# Routing Infrastructure (Extensible)
# =============================================================================

class Difficulty(Enum):
    """Task difficulty levels."""
    SIMPLE = "simple"      # Standard formula, clear values
    MEDIUM = "medium"      # Needs extraction, standard formula
    HARD = "hard"          # Multiple conditions, complex scoring


@dataclass
class RoutingDecision:
    """
    Result of routing decision.

    Extensible: Can carry additional metadata for future routing strategies.
    """
    difficulty: Difficulty
    target_level: str                    # e.g., "L2", "L4_pipeline", "L3"
    confidence: float = 1.0
    reasoning: str = ""
    # Future extensibility
    recommended_model: Optional[str] = None
    estimated_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRouter(ABC):
    """
    Abstract base for routing strategies.

    Subclass this to implement different routing approaches:
    - DifficultyRouter: Routes based on task difficulty
    - ModelRouter: Routes to different models (future)
    - CascadeRouter: Tries cheap first, escalates if uncertain (future)
    """

    @abstractmethod
    def route(
        self,
        question: str,
        patient_note: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Make routing decision for a task."""
        pass

    def update_from_result(
        self,
        decision: RoutingDecision,
        result: ExperimentResult,
    ) -> None:
        """
        Optional: Update router based on execution result.
        Override for adaptive/learning routers.
        """
        pass


# =============================================================================
# Difficulty Estimation
# =============================================================================

# Calculator difficulty classification based on domain knowledge
CALCULATOR_DIFFICULTY: Dict[str, Difficulty] = {
    # SIMPLE: Direct formulas, few variables, clear extraction
    "Body Mass Index (BMI)": Difficulty.SIMPLE,
    "Ideal Body Weight (Devine formula)": Difficulty.SIMPLE,
    "Mean Arterial Pressure (MAP)": Difficulty.SIMPLE,
    "Body Surface Area (BSA)": Difficulty.SIMPLE,
    "Corrected QT Interval (QTc) - Bazett": Difficulty.SIMPLE,
    "Corrected QT Interval (QTc) - Fridericia": Difficulty.SIMPLE,
    "Target Weight": Difficulty.SIMPLE,
    "LDL Calculated": Difficulty.SIMPLE,
    "Serum Osmolality": Difficulty.SIMPLE,

    # MEDIUM: Need careful extraction, standard formulas
    "Creatinine Clearance (Cockcroft-Gault Equation)": Difficulty.MEDIUM,
    "CKD-EPI Equations for Glomerular Filtration Rate": Difficulty.MEDIUM,
    "MDRD GFR Equation": Difficulty.MEDIUM,
    "Anion Gap": Difficulty.MEDIUM,
    "Calcium Correction for Hypoalbuminemia": Difficulty.MEDIUM,
    "Sodium Correction for Hyperglycemia": Difficulty.MEDIUM,
    "Free Water Deficit": Difficulty.MEDIUM,
    "Fibrosis-4 (FIB-4) Index": Difficulty.MEDIUM,
    "Adjusted Body Weight": Difficulty.MEDIUM,
    "Maintenance Fluids (Holliday-Segar)": Difficulty.MEDIUM,

    # HARD: Multiple boolean conditions, complex scoring, interpretation
    "CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk": Difficulty.HARD,
    "CURB-65 Score for Pneumonia Severity": Difficulty.HARD,
    "Wells' Criteria for Pulmonary Embolism": Difficulty.HARD,
    "Wells' Criteria for DVT": Difficulty.HARD,
    "HEART Score for Major Cardiac Events": Difficulty.HARD,
    "HAS-BLED Score for Major Bleeding Risk": Difficulty.HARD,
    "APACHE II Score": Difficulty.HARD,
    "SOFA Score": Difficulty.HARD,
    "Glasgow Coma Scale (GCS)": Difficulty.HARD,
    "Child-Pugh Score for Cirrhosis": Difficulty.HARD,
    "Charlson Comorbidity Index (CCI)": Difficulty.HARD,
    "PSI/PORT Score": Difficulty.HARD,
    "Caprini Score for VTE": Difficulty.HARD,
    "MELD-Na Score": Difficulty.HARD,
    "Revised Cardiac Risk Index (RCRI)": Difficulty.HARD,
    "Centor Score (Modified McIsaac)": Difficulty.HARD,
    "Framingham Risk Score": Difficulty.HARD,
}

# Default difficulty for unknown calculators
DEFAULT_DIFFICULTY = Difficulty.MEDIUM


@ptool(model="deepseek-v3-0324", output_mode="structured")
def estimate_difficulty_llm(
    question: str,
    note_preview: str,
    available_difficulties: List[str],
) -> Dict[str, Any]:
    """
    Estimate the difficulty of a medical calculation task.

    Consider:
    - How many values need to be extracted?
    - Are the values clearly stated or need inference?
    - Is it a simple formula or complex scoring system?
    - Are there boolean conditions to evaluate?

    Difficulty levels:
    - simple: Few variables, clearly stated, direct formula (BMI, MAP)
    - medium: Multiple variables, need careful extraction (CrCl, GFR)
    - hard: Many conditions, boolean flags, complex scoring (APACHE, CHA2DS2-VASc)

    Return:
    {
        "difficulty": "simple" | "medium" | "hard",
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation"
    }
    """
    ...


class DifficultyRouter(BaseRouter):
    """
    Routes based on task difficulty estimation.

    Uses a combination of:
    1. Rule-based: Known calculator → known difficulty
    2. LLM-based: Unknown calculator → estimate difficulty

    Future extension points:
    - Override level_mapping to route to different targets
    - Override model_mapping to select models per difficulty
    """

    def __init__(
        self,
        level_mapping: Optional[Dict[Difficulty, str]] = None,
        model_mapping: Optional[Dict[Difficulty, str]] = None,
        use_llm_estimation: bool = False,
    ):
        # Default: Difficulty → Framework Level mapping
        self.level_mapping = level_mapping or {
            Difficulty.SIMPLE: "L2",
            Difficulty.MEDIUM: "L4_pipeline",
            Difficulty.HARD: "L3",
        }

        # Future: Difficulty → Model mapping
        self.model_mapping = model_mapping or {
            Difficulty.SIMPLE: "deepseek-v3-0324",  # Could be cheaper model
            Difficulty.MEDIUM: "deepseek-v3-0324",
            Difficulty.HARD: "deepseek-v3-0324",    # Could be stronger model
        }

        self.use_llm_estimation = use_llm_estimation

        # Statistics for analysis
        self.routing_stats: Dict[str, int] = {
            "simple": 0, "medium": 0, "hard": 0,
            "rule_based": 0, "llm_based": 0,
        }

    def _identify_calculator_from_question(self, question: str) -> Optional[str]:
        """Try to identify calculator from question text."""
        question_lower = question.lower()

        for calc_name in CALCULATOR_DIFFICULTY:
            # Check if calculator name appears in question
            calc_lower = calc_name.lower()
            if calc_lower in question_lower:
                return calc_name

            # Check common abbreviations
            abbrevs = {
                "bmi": "Body Mass Index",
                "gfr": "CKD-EPI Equations for Glomerular Filtration Rate",
                "egfr": "CKD-EPI Equations for Glomerular Filtration Rate",
                "creatinine clearance": "Creatinine Clearance (Cockcroft-Gault Equation)",
                "cockcroft": "Creatinine Clearance (Cockcroft-Gault Equation)",
                "cha2ds2": "CHA2DS2-VASc Score",
                "chads": "CHA2DS2-VASc Score",
                "curb-65": "CURB-65 Score",
                "curb65": "CURB-65 Score",
                "wells": "Wells' Criteria",
                "apache": "APACHE II Score",
                "sofa": "SOFA Score",
                "meld": "MELD-Na Score",
                "child-pugh": "Child-Pugh Score",
                "fib-4": "Fibrosis-4 (FIB-4) Index",
                "fib4": "Fibrosis-4 (FIB-4) Index",
                "map": "Mean Arterial Pressure",
                "qtc": "Corrected QT Interval",
            }

            for abbrev, full_name in abbrevs.items():
                if abbrev in question_lower:
                    # Find matching full calculator name
                    for cn in CALCULATOR_DIFFICULTY:
                        if full_name.lower() in cn.lower():
                            return cn

        return None

    def _estimate_difficulty_rule_based(self, question: str) -> Optional[Difficulty]:
        """Estimate difficulty using rule-based approach."""
        calc_name = self._identify_calculator_from_question(question)
        if calc_name and calc_name in CALCULATOR_DIFFICULTY:
            return CALCULATOR_DIFFICULTY[calc_name]
        return None

    def _estimate_difficulty_llm(
        self,
        question: str,
        patient_note: str,
    ) -> Difficulty:
        """Estimate difficulty using LLM (more expensive but handles unknowns)."""
        try:
            result = estimate_difficulty_llm(
                question=question,
                note_preview=patient_note[:500],  # Limit context
                available_difficulties=["simple", "medium", "hard"],
            )

            if isinstance(result, dict):
                inner = result.get("result", result)
                if isinstance(inner, dict):
                    diff_str = inner.get("difficulty", "medium")
                else:
                    diff_str = "medium"
            else:
                diff_str = "medium"

            return Difficulty(diff_str.lower())

        except Exception:
            return DEFAULT_DIFFICULTY

    def route(
        self,
        question: str,
        patient_note: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route based on difficulty estimation.

        Priority:
        1. Rule-based (fast, free)
        2. LLM-based if enabled and rules fail
        3. Default to MEDIUM
        """
        # Try rule-based first
        difficulty = self._estimate_difficulty_rule_based(question)

        if difficulty is not None:
            self.routing_stats["rule_based"] += 1
            reasoning = "Rule-based: identified calculator type"
        elif self.use_llm_estimation:
            difficulty = self._estimate_difficulty_llm(question, patient_note)
            self.routing_stats["llm_based"] += 1
            reasoning = "LLM-based: estimated from task description"
        else:
            difficulty = DEFAULT_DIFFICULTY
            reasoning = "Default: unknown calculator type"

        # Update stats
        self.routing_stats[difficulty.value] += 1

        return RoutingDecision(
            difficulty=difficulty,
            target_level=self.level_mapping[difficulty],
            confidence=0.8 if difficulty != DEFAULT_DIFFICULTY else 0.5,
            reasoning=reasoning,
            recommended_model=self.model_mapping.get(difficulty),
        )


# =============================================================================
# Adaptive Orchestrator Experiment
# =============================================================================

class L4AdaptiveExperiment(BaseExperiment):
    """
    L4 Experiment: Adaptive Orchestration with Difficulty-Aware Routing.

    Routes each task to the most appropriate framework level:
    - SIMPLE → L2 (Python-first, cheapest)
    - MEDIUM → L4 Pipeline (balanced)
    - HARD → L3 ReAct (most thorough)

    Extensibility:
    - Swap router for different routing strategies
    - Override level_experiments to use different experiment implementations
    - Configure model_mapping in router for model selection
    """

    def __init__(
        self,
        config: ExperimentConfig,
        router: Optional[BaseRouter] = None,
        level_experiments: Optional[Dict[str, BaseExperiment]] = None,
    ):
        super().__init__(config)
        self.model = config.model

        # Initialize router
        self.router = router or DifficultyRouter(
            use_llm_estimation=getattr(config, 'use_llm_routing', False),
        )

        # Level → Experiment mapping (lazy initialization)
        self._level_experiments: Dict[str, BaseExperiment] = level_experiments or {}
        self._experiments_initialized = False

        # Statistics
        self.routing_decisions: List[RoutingDecision] = []
        self.level_stats: Dict[str, Dict[str, int]] = {
            "L2": {"calls": 0, "correct": 0},
            "L4_pipeline": {"calls": 0, "correct": 0},
            "L3": {"calls": 0, "correct": 0},
        }

    def _get_experiment_for_level(self, level: str) -> BaseExperiment:
        """Lazy-load experiment for a given level."""
        if level not in self._level_experiments:
            if level == "L2":
                from .l2_distilled import L2DistilledExperiment
                exp_config = ABLATION_CONFIGS["l2_distilled"]
                self._level_experiments[level] = L2DistilledExperiment(exp_config)
            elif level == "L4_pipeline":
                from .l4_pipeline import L4PipelineExperiment
                exp_config = ABLATION_CONFIGS["l4_pipeline"]
                self._level_experiments[level] = L4PipelineExperiment(exp_config)
            elif level == "L3":
                from .l3_react import L3ReactExperiment
                exp_config = ABLATION_CONFIGS["l3_react"]
                self._level_experiments[level] = L3ReactExperiment(exp_config)
            else:
                raise ValueError(f"Unknown level: {level}")

            # Setup the experiment
            self._level_experiments[level].setup()

        return self._level_experiments[level]

    def setup(self):
        """Initialize experiments (lazy - done on first use)."""
        self._setup_complete = True

    def run_instance(self, instance: MedCalcInstance) -> ExperimentResult:
        """
        Run adaptive routing on a single instance.

        1. Route to appropriate level
        2. Execute with selected experiment
        3. Track statistics for analysis
        """
        start_time = time.time()

        # Make routing decision
        decision = self.router.route(
            question=instance.question,
            patient_note=instance.patient_note,
        )
        self.routing_decisions.append(decision)

        # Get experiment for target level
        target_level = decision.target_level
        experiment = self._get_experiment_for_level(target_level)

        # Execute
        result = experiment.run_instance(instance)

        # Update statistics
        self.level_stats[target_level]["calls"] += 1
        if result.is_correct_tolerance:
            self.level_stats[target_level]["correct"] += 1

        # Update router if it supports learning
        self.router.update_from_result(decision, result)

        # Augment result with routing info
        routing_trace = {
            "difficulty": decision.difficulty.value,
            "target_level": decision.target_level,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "recommended_model": decision.recommended_model,
        }

        # Merge with existing trace
        if result.trace:
            if isinstance(result.trace, dict):
                result.trace["routing"] = routing_trace
            else:
                result = ExperimentResult(
                    **{**result.__dict__, "trace": {"original": result.trace, "routing": routing_trace}}
                )
        else:
            result = ExperimentResult(
                **{**result.__dict__, "trace": {"routing": routing_trace}}
            )

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with routing statistics."""
        summary = super().get_summary()

        # Routing stats
        if isinstance(self.router, DifficultyRouter):
            summary["routing_stats"] = self.router.routing_stats

        # Level usage and accuracy
        summary["level_stats"] = {}
        for level, stats in self.level_stats.items():
            calls = stats["calls"]
            correct = stats["correct"]
            summary["level_stats"][level] = {
                "calls": calls,
                "correct": correct,
                "accuracy": correct / calls if calls > 0 else 0,
            }

        # Cost breakdown (estimated)
        cost_per_level = {"L2": 0.001, "L4_pipeline": 0.002, "L3": 0.01}
        total_cost = sum(
            stats["calls"] * cost_per_level.get(level, 0.01)
            for level, stats in self.level_stats.items()
        )
        summary["estimated_total_cost"] = total_cost
        summary["estimated_avg_cost"] = total_cost / max(sum(s["calls"] for s in self.level_stats.values()), 1)

        return summary


# =============================================================================
# Standalone execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run L4 Adaptive Experiment")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of instances")
    parser.add_argument("--instance", type=int, help="Run specific instance")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--use-llm-routing", action="store_true", help="Use LLM for difficulty estimation")
    args = parser.parse_args()

    # Load dataset
    print("Loading MedCalc-Bench dataset...")
    from benchmark.dataset.loader import MedCalcDataset

    dataset = MedCalcDataset()

    if args.instance is not None:
        all_instances = dataset.load("test")
        instances = [i for i in all_instances if i.row_number == args.instance]
        if not instances:
            print(f"Instance {args.instance} not found")
            sys.exit(1)
    else:
        instances = dataset.get_debug_subset(args.num)

    print(f"Running L4 Adaptive on {len(instances)} instances...")
    print("Routing: SIMPLE→L2, MEDIUM→L4_pipeline, HARD→L3")
    print()

    # Create experiment
    config = ExperimentConfig(
        name="l4_adaptive",
        description="Adaptive difficulty-aware routing",
        level="L4",
    )

    router = DifficultyRouter(use_llm_estimation=args.use_llm_routing)
    experiment = L4AdaptiveExperiment(config, router=router)
    experiment.setup()

    # Run instances
    correct = 0
    total = 0

    for i, instance in enumerate(instances):
        print(f"[{i+1}/{len(instances)}] {instance.calculator_name}")
        print(f"  Q: {instance.question[:50]}...")

        result = experiment.run_instance(instance)
        total += 1

        # Get routing info
        routing = result.trace.get("routing", {}) if result.trace else {}
        level = routing.get("target_level", "?")
        difficulty = routing.get("difficulty", "?")

        if result.error:
            status = f"ERROR: {result.error}"
        elif result.is_correct_tolerance:
            correct += 1
            status = "CORRECT"
        else:
            status = f"WRONG (pred={result.predicted_answer}, exp={result.ground_truth})"

        print(f"  → {difficulty} → {level}: {status}")
        print(f"  Latency: {result.latency_ms:.0f}ms, Cost: ${result.cost_usd:.4f}")

        if args.verbose:
            print(f"  Routing: {routing}")

        print()

    # Summary
    print("=" * 60)
    print(f"L4 ADAPTIVE RESULTS: {correct}/{total} ({correct/total*100:.1f}%)")
    print()

    summary = experiment.get_summary()

    print("Routing distribution:")
    if "routing_stats" in summary:
        for key, count in summary["routing_stats"].items():
            if count > 0:
                print(f"  {key}: {count}")

    print()
    print("Level performance:")
    for level, stats in summary.get("level_stats", {}).items():
        if stats["calls"] > 0:
            print(f"  {level}: {stats['correct']}/{stats['calls']} ({stats['accuracy']*100:.1f}%)")

    print()
    print(f"Estimated total cost: ${summary.get('estimated_total_cost', 0):.4f}")
    print(f"Estimated avg cost:   ${summary.get('estimated_avg_cost', 0):.4f}/instance")
    print("=" * 60)
