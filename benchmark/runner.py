"""
Main benchmark runner for MedCalc-Bench experiments.

Orchestrates running experiments, collecting results, and generating reports.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass

from .config import ExperimentConfig, ABLATION_CONFIGS, ExperimentLevel
from .dataset.loader import MedCalcDataset, MedCalcInstance
from .experiments.base import BaseExperiment, ExperimentResult
from .experiments.baseline import BaselineExperiment
from .experiments.l1_ptool import L1PToolExperiment
from .experiments.l1o_official import L1OOfficialExperiment
from .experiments.l2_distilled import L2DistilledExperiment
from .experiments.l2o_official import L2OOfficialExperiment
from .experiments.l3_react import L3ReactExperiment
from .experiments.l3_audit import L3AuditExperiment
from .experiments.l4_orchestrator import L4OrchestratorExperiment
from .experiments.l4_pipeline import L4PipelineExperiment
from .experiments.l5_improving import L5ImprovingExperiment
from .experiments.l5_icl import L5ICLExperiment
from .metrics.aggregator import MetricsAggregator, AggregatedMetrics
from .reports.generator import ReportGenerator
from .response_logger import ResponseLogger


# Mapping from experiment level to experiment class
EXPERIMENT_CLASSES: Dict[str, Type[BaseExperiment]] = {
    "L0": BaselineExperiment,
    "L1": L1PToolExperiment,
    "L1O": L1OOfficialExperiment,
    "L2": L2DistilledExperiment,
    "L2O": L2OOfficialExperiment,
    "L3": L3ReactExperiment,
    "L3+": L3AuditExperiment,
    "L4": L4OrchestratorExperiment,
    "L4P": L4PipelineExperiment,
    "L5": L5ImprovingExperiment,
    "L5+": L5ICLExperiment,
}


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    experiments: List[str]          # List of experiment names to run
    debug: bool = True              # If True, use only first 10 instances
    debug_n: int = 10               # Number of debug instances
    output_dir: Path = Path("./benchmark_results")
    save_detailed_results: bool = True
    generate_report: bool = True


class BenchmarkRunner:
    """
    Main orchestrator for running MedCalc-Bench experiments.

    Usage:
        runner = BenchmarkRunner(output_dir="./results")
        metrics = runner.run_ablation_study(debug=True)
        runner.generate_report(metrics)
    """

    def __init__(
        self,
        output_dir: Path = Path("./benchmark_results"),
        dataset: Optional[MedCalcDataset] = None,
        log_responses: bool = False,
    ):
        """
        Initialize benchmark runner.

        Args:
            output_dir: Directory for results
            dataset: Optional pre-loaded dataset
            log_responses: If True, save prompts/responses to .txt files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_responses = log_responses

        self.dataset = dataset or MedCalcDataset()
        self.aggregator = MetricsAggregator()
        self.report_generator = ReportGenerator(output_dir)

        # Results storage
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.metrics: Dict[str, AggregatedMetrics] = {}
        self.summaries: Dict[str, Dict[str, Any]] = {}
        self.learning_curves: Dict[str, List[Dict]] = {}

    def run_ablation_study(
        self,
        experiments: Optional[List[str]] = None,
        debug: bool = True,
        debug_n: int = 10,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, AggregatedMetrics]:
        """
        Run full ablation study across specified experiments.

        Args:
            experiments: List of experiment names (None = all)
            debug: If True, use only first N instances
            debug_n: Number of debug instances
            progress_callback: Optional callback(exp_name, current, total)

        Returns:
            Dict mapping experiment name to AggregatedMetrics
        """
        # Determine experiments to run
        if experiments is None or "all" in experiments:
            experiments = list(ABLATION_CONFIGS.keys())

        # Load data
        print(f"Loading MedCalc-Bench dataset...")
        if debug:
            instances = self.dataset.get_debug_subset(debug_n)
            print(f"Debug mode: using first {len(instances)} instances")
        else:
            instances = self.dataset.load("test")
            print(f"Full mode: using {len(instances)} instances")

        # Load training data for L5+ ICL
        train_data = None
        if "l5_icl" in experiments:
            print("Loading training data for ICL...")
            train_data = self.dataset.load("train")

        # Run each experiment
        for exp_name in experiments:
            print(f"\n{'='*60}")
            print(f"Running experiment: {exp_name}")
            print(f"{'='*60}")

            config = ABLATION_CONFIGS[exp_name]
            results, summary = self._run_single_experiment(
                config=config,
                instances=instances,
                train_data=train_data if exp_name == "l5_icl" else None,
                progress_callback=progress_callback,
            )

            # Store results
            self.results[exp_name] = results
            self.summaries[exp_name] = summary

            # Aggregate metrics
            self.metrics[exp_name] = self.aggregator.aggregate(
                results,
                exp_name,
                config.level.value,
            )

            # Extract learning curve for L5 experiments
            if "learning_curve" in summary:
                self.learning_curves[exp_name] = summary["learning_curve"]

            # Save intermediate results
            self._save_experiment_results(exp_name, results, summary)

            # Print summary
            self._print_summary(exp_name)

        return self.metrics

    def _run_single_experiment(
        self,
        config: ExperimentConfig,
        instances: List[MedCalcInstance],
        train_data: Optional[List[MedCalcInstance]] = None,
        progress_callback: Optional[callable] = None,
    ) -> tuple:
        """Run a single experiment configuration."""

        # Create experiment instance
        exp_class = EXPERIMENT_CLASSES[config.level.value]

        if config.level == ExperimentLevel.L5_ICL:
            experiment = exp_class(config, train_data=train_data)
        else:
            experiment = exp_class(config)

        # Create logger if enabled
        logger = None
        if self.log_responses:
            logger = ResponseLogger(self.output_dir, config.name)

        # Setup
        print(f"Setting up {config.name}...")
        experiment.setup()

        # Run with progress tracking
        start_time = time.time()

        def _progress(current, total):
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            eta = (total - current) / rate if rate > 0 else 0
            print(f"\r  Progress: {current}/{total} ({current/total*100:.1f}%) - "
                  f"ETA: {eta:.0f}s", end="", flush=True)
            if progress_callback:
                progress_callback(config.name, current, total)

        results = experiment.run_batch(instances, progress_callback=_progress)
        print()  # New line after progress

        # Log responses if enabled
        if logger:
            self._log_experiment_responses(logger, instances, results, config)

        total_time = time.time() - start_time
        print(f"Completed in {total_time:.1f}s")

        # Get summary
        summary = experiment.get_summary()
        summary["total_time_seconds"] = total_time

        # Cleanup
        experiment.teardown()

        return results, summary

    def _log_experiment_responses(
        self,
        logger: ResponseLogger,
        instances: List[MedCalcInstance],
        results: List[ExperimentResult],
        config: ExperimentConfig,
    ):
        """Log all responses from an experiment run."""
        for instance, result in zip(instances, results):
            # Check if we have trace data with steps
            if result.trace and isinstance(result.trace, dict):
                steps = result.trace.get("steps", [])
                if steps:
                    # Check if we have audit info (L3+ experiments)
                    if result.audit_info:
                        logger.log_audit_trace(
                            instance_id=result.instance_id,
                            calculator_name=result.calculator_name,
                            steps=steps,
                            final_answer=result.predicted_answer,
                            ground_truth=result.ground_truth,
                            is_correct=result.is_correct_tolerance,
                            error=result.error,
                            audit_info=result.audit_info,
                        )
                    else:
                        # Log as multi-step trace
                        logger.log_trace(
                            instance_id=result.instance_id,
                            calculator_name=result.calculator_name,
                            steps=steps,
                            final_answer=result.predicted_answer,
                            ground_truth=result.ground_truth,
                            is_correct=result.is_correct_tolerance,
                            error=result.error,
                        )
                    continue

            # Log as simple prompt/response
            # Reconstruct prompt from instance data
            prompt = f"""Patient Note:
{instance.patient_note}

Question: {instance.question}
"""
            logger.log_instance(
                instance_id=result.instance_id,
                calculator_name=result.calculator_name,
                prompt=prompt,
                response=result.raw_response or "(no response)",
                predicted=result.predicted_answer,
                ground_truth=result.ground_truth,
                is_correct=result.is_correct_tolerance,
                error=result.error,
                extra={"trace": str(result.trace)} if result.trace else None,
            )

    def _save_experiment_results(
        self,
        exp_name: str,
        results: List[ExperimentResult],
        summary: Dict[str, Any],
    ):
        """Save results to disk."""
        exp_dir = self.output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)

        # Save detailed results
        results_path = exp_dir / "results.json"
        results_path.write_text(json.dumps(
            [r.to_dict() for r in results],
            indent=2,
            default=str,
        ))

        # Save summary
        summary_path = exp_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

        # Save experiment prompt template and source code
        self._save_experiment_prompt_template(exp_name, exp_dir)
        self._save_experiment_source(exp_name, exp_dir)

    def _save_experiment_prompt_template(self, exp_name: str, exp_dir: Path):
        """Save the prompt template used by this experiment."""
        prompt_file = exp_dir / "prompt_template.txt"

        separator = "=" * 80

        if exp_name == "baseline":
            from .experiments.baseline import BaselineExperiment
            content = f"""{separator}
EXPERIMENT: {exp_name} (L0 - Direct LLM Call)
{separator}

DESCRIPTION:
Direct LLM call without any framework. Simply prompts the LLM with the
patient note and question, asking for a numeric answer.

{separator}
PROMPT TEMPLATE
{separator}

{BaselineExperiment.PROMPT_TEMPLATE}
"""

        elif exp_name == "l1_ptool":
            from .experiments.l1_ptool import calculate_medical_value
            spec = calculate_medical_value.spec

            # Generate example prompt
            example_prompt = spec.format_prompt(
                patient_note="<PATIENT_NOTE>",
                question="<QUESTION>",
            )

            content = f"""{separator}
EXPERIMENT: {exp_name} (L1 - @ptool Decorator)
{separator}

DESCRIPTION:
Uses the @ptool decorator to create a typed LLM function call.
The function signature and docstring guide the LLM's response.

{separator}
PTOOL SPECIFICATION
{separator}

Name: {spec.name}
Model: {spec.model}
Output Mode: {spec.output_mode}
Return Type: {spec.return_type}

Signature: {spec.get_signature_str()}

Docstring:
{spec.docstring}

{separator}
FORMATTED PROMPT (with placeholders)
{separator}

{example_prompt}
"""

        elif exp_name == "l2_distilled":
            content = f"""{separator}
EXPERIMENT: {exp_name} (L2 - @distilled Python-First)
{separator}

DESCRIPTION:
Python-first approach that tries pure Python extraction and calculation,
falling back to LLM only when Python can't handle the case.

Implemented Python calculators:
- BMI (Body Mass Index)
- eGFR (Estimated Glomerular Filtration Rate)

For other calculators, falls back to L1 ptool.

{separator}
PYTHON EXTRACTION PATTERNS
{separator}

Age patterns:
- r'(\\d+)\\s*(?:year|yr|y/?o|years?\\s*old)'
- r'age[:\\s]+(\\d+)'
- r'(\\d+)\\s*yo\\b'

Weight patterns (kg):
- r'(\\d+\\.?\\d*)\\s*kg'
- r'weight[:\\s]+(\\d+\\.?\\d*)'
- r'(\\d+\\.?\\d*)\\s*kilograms?'

Height patterns (m):
- r'(\\d+\\.?\\d*)\\s*cm' (converted to m)
- r'(\\d+\\.?\\d*)\\s*m(?:eters?)?'

Creatinine patterns:
- r'creatinine[:\\s]+(\\d+\\.?\\d*)'
- r'cr[:\\s]+(\\d+\\.?\\d*)'
- r'(\\d+\\.?\\d*)\\s*mg/dL'

{separator}
CALCULATION FORMULAS
{separator}

BMI = weight_kg / (height_m ** 2)

eGFR (CKD-EPI 2021):
  Female, Cr <= 0.7: 142 * (Cr/0.7)^-0.241 * 0.9938^age * 1.012
  Female, Cr > 0.7:  142 * (Cr/0.7)^-1.200 * 0.9938^age * 1.012
  Male, Cr <= 0.9:   142 * (Cr/0.9)^-0.302 * 0.9938^age
  Male, Cr > 0.9:    142 * (Cr/0.9)^-1.200 * 0.9938^age
"""

        elif exp_name == "l3_react":
            from .experiments.l3_react import identify_calculator, extract_clinical_values, perform_calculation

            content = f"""{separator}
EXPERIMENT: {exp_name} (L3 - ReActAgent)
{separator}

DESCRIPTION:
Uses the ReActAgent for multi-step reasoning with think-act-observe loop.
Breaks down medical calculations into steps using specialized ptools.

{separator}
AVAILABLE PTOOLS
{separator}

1. identify_calculator(clinical_text: str) -> Dict
   {identify_calculator.spec.docstring}

2. extract_clinical_values(patient_note: str, required_values: List[str]) -> Dict
   {extract_clinical_values.spec.docstring}

3. perform_calculation(calculator_name: str, values: Dict) -> Dict
   {perform_calculation.spec.docstring}

{separator}
REACT GOAL TEMPLATE
{separator}

Patient Note:
<PATIENT_NOTE>

Task: <QUESTION>

Use the available tools to:
1. Identify what calculator is needed
2. Extract the required values from the patient note
3. Perform the calculation
4. Return the numeric result
"""

        elif exp_name in ["l3_audit", "l4_orchestrator", "l5_improving", "l5_icl"]:
            # For more complex experiments, provide a high-level description
            descriptions = {
                "l3_audit": """L3+ - ReAct + TraceCritic + RepairAgent

Adds error detection and automatic repair to L3 ReAct.
Uses audits to evaluate traces and repair agent to fix issues.

Audits:
- NonEmptyTraceAudit: Ensures trace has steps
- NoFailedStepsAudit: Checks for failed tool calls
- AllCompletedAudit: Verifies all steps completed
- MedCalcStructureAudit: Domain-specific structure checks""",

                "l4_orchestrator": """L4 - AgentOrchestrator with Specialized Agents

Routes tasks to specialized agents based on calculator category:
- equation_calculator: For formula-based (BMI, eGFR, BSA, dosing)
- rule_calculator: For scoring systems (CHA2DS2-VASc, Wells, CURB-65)
- general_calculator: Fallback for other cases

Uses HybridRouter with domain keywords for routing.""",

                "l5_improving": """L5 - SelfImprovingAgent with Experience Learning

Wraps ReAct agent with self-improvement capabilities:
- Learns patterns from successful executions
- Avoids patterns from failed executions
- Uses relevant patterns to enhance prompts
- Tracks improvement over time via PatternMemory""",

                "l5_icl": """L5+ - SelfImproving with ICL from Training Set

Pre-populates pattern memory with solved examples from training data.
Provides in-context learning examples for each calculator type.

Configuration:
- ICL examples per calculator: Configured in ExperimentConfig
- Pattern confidence: 1.0 for training examples
- Decay disabled for ICL patterns""",
            }

            content = f"""{separator}
EXPERIMENT: {exp_name}
{separator}

{descriptions.get(exp_name, "No description available.")}
"""

        else:
            content = f"""{separator}
EXPERIMENT: {exp_name}
{separator}

No prompt template documentation available for this experiment.
"""

        prompt_file.write_text(content, encoding='utf-8')

    def _save_experiment_source(self, exp_name: str, exp_dir: Path):
        """Save the Python source code of the experiment."""
        import shutil

        # Map experiment names to their source files
        source_files = {
            "baseline": "baseline.py",
            "l1_ptool": "l1_ptool.py",
            "l1o_official": "l1o_official.py",
            "l2_distilled": "l2_distilled.py",
            "l2o_official": "l2o_official.py",
            "l3_react": "l3_react.py",
            "l3_audit": "l3_audit.py",
            "l4_orchestrator": "l4_orchestrator.py",
            "l4_pipeline": "l4_pipeline.py",
            "l5_improving": "l5_improving.py",
            "l5_icl": "l5_icl.py",
        }

        if exp_name in source_files:
            source_file = source_files[exp_name]
            experiments_dir = Path(__file__).parent / "experiments"
            source_path = experiments_dir / source_file

            if source_path.exists():
                # Copy the Python source file
                dest_path = exp_dir / f"experiment_source.py"
                shutil.copy(source_path, dest_path)

    def _print_summary(self, exp_name: str):
        """Print experiment summary to console."""
        m = self.metrics[exp_name]
        print(f"\nResults for {exp_name}:")
        print(f"  Accuracy (within 5%): {m.accuracy_tolerance * 100:.1f}%")
        print(f"  Accuracy (exact):     {m.accuracy_exact * 100:.1f}%")
        print(f"  Total cost:           ${m.total_cost_usd:.4f}")
        print(f"  Avg latency:          {m.avg_latency_ms:.0f} ms")
        print(f"  Errors:               {m.error_count} ({m.error_rate * 100:.1f}%)")

    def generate_report(
        self,
        metrics: Optional[Dict[str, AggregatedMetrics]] = None,
    ) -> Path:
        """
        Generate HTML report from results.

        Args:
            metrics: Optional metrics dict (uses stored if None)

        Returns:
            Path to generated report
        """
        metrics = metrics or self.metrics

        if not metrics:
            raise ValueError("No metrics available. Run experiments first.")

        print("\nGenerating report...")
        report_path = self.report_generator.generate_full_report(
            experiment_metrics=metrics,
            detailed_results=self.results,
            learning_curves=self.learning_curves,
        )

        print(f"Report saved to: {report_path}")
        return report_path

    def compare_to_baseline(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare all experiments to baseline.

        Returns:
            Dict of comparisons for each experiment
        """
        if "baseline" not in self.metrics:
            raise ValueError("Baseline not found in results")

        comparisons = {}
        baseline = self.metrics["baseline"]

        for name, metrics in self.metrics.items():
            if name != "baseline":
                comparisons[name] = self.aggregator.compare_to_baseline(
                    baseline, metrics
                )

        return comparisons

    def get_best_experiment(self) -> tuple:
        """
        Get the best performing experiment.

        Returns:
            Tuple of (experiment_name, AggregatedMetrics)
        """
        if not self.metrics:
            raise ValueError("No metrics available")

        return max(
            self.metrics.items(),
            key=lambda x: x[1].accuracy_tolerance,
        )
