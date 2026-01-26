"""
Response logger for debugging benchmark experiments.

Saves prompts and LLM responses to readable .txt files.
"""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


class ResponseLogger:
    """
    Logs LLM prompts and responses to .txt files for debugging.

    Creates files like:
        benchmark_results/logs/l1_ptool/instance_001.txt
    """

    def __init__(self, output_dir: Path, experiment_name: str):
        """
        Initialize logger for a specific experiment.

        Args:
            output_dir: Base output directory (e.g., ./benchmark_results)
            experiment_name: Name of the experiment (e.g., l1_ptool)
        """
        self.log_dir = output_dir / "logs" / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

    def log_instance(
        self,
        instance_id: Any,
        calculator_name: str,
        prompt: str,
        response: str,
        predicted: Any = None,
        ground_truth: Any = None,
        is_correct: bool = False,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a single instance's prompt and response.

        Args:
            instance_id: Dataset instance ID
            calculator_name: Name of the medical calculator
            prompt: The prompt sent to the LLM
            response: The raw LLM response
            predicted: Extracted/predicted answer
            ground_truth: Expected answer
            is_correct: Whether prediction was correct
            error: Any error that occurred
            extra: Additional info to log (e.g., trace steps)
        """
        # Convert instance_id to int if possible
        try:
            iid = int(instance_id)
            filename = f"instance_{iid:04d}.txt"
        except (ValueError, TypeError):
            filename = f"instance_{instance_id}.txt"
        filepath = self.log_dir / filename

        separator = "=" * 80

        content = f"""{separator}
EXPERIMENT: {self.experiment_name}
INSTANCE: {instance_id}
CALCULATOR: {calculator_name}
TIMESTAMP: {datetime.now().isoformat()}
{separator}

GROUND TRUTH: {ground_truth}
PREDICTED: {predicted}
CORRECT: {is_correct}
{"ERROR: " + error if error else ""}

{separator}
PROMPT
{separator}

{prompt}

{separator}
RESPONSE
{separator}

{response}
"""

        # Add extra info if provided
        if extra:
            content += f"\n{separator}\nEXTRA INFO\n{separator}\n\n"
            for key, value in extra.items():
                content += f"{key}:\n{value}\n\n"

        filepath.write_text(content, encoding='utf-8')

    def log_trace(
        self,
        instance_id: Any,
        calculator_name: str,
        steps: list,
        final_answer: Any = None,
        ground_truth: Any = None,
        is_correct: bool = False,
        error: Optional[str] = None,
    ):
        """
        Log a multi-step trace (for ReAct agents).

        Args:
            instance_id: Dataset instance ID
            calculator_name: Name of the medical calculator
            steps: List of trace steps (thought, action, observation)
            final_answer: The final answer
            ground_truth: Expected answer
            is_correct: Whether prediction was correct
            error: Any error that occurred
        """
        try:
            iid = int(instance_id)
            filename = f"instance_{iid:04d}.txt"
        except (ValueError, TypeError):
            filename = f"instance_{instance_id}.txt"
        filepath = self.log_dir / filename

        separator = "=" * 80
        step_sep = "-" * 40

        content = f"""{separator}
EXPERIMENT: {self.experiment_name}
INSTANCE: {instance_id}
CALCULATOR: {calculator_name}
TIMESTAMP: {datetime.now().isoformat()}
{separator}

GROUND TRUTH: {ground_truth}
FINAL ANSWER: {final_answer}
CORRECT: {is_correct}
{"ERROR: " + error if error else ""}
NUM STEPS: {len(steps)}

{separator}
TRACE
{separator}

"""

        for i, step in enumerate(steps, 1):
            content += f"{step_sep}\nSTEP {i}\n{step_sep}\n"

            if isinstance(step, dict):
                if 'thought' in step:
                    content += f"\nTHOUGHT:\n{step['thought']}\n"
                if 'action' in step:
                    content += f"\nACTION:\n{step['action']}\n"
                if 'observation' in step:
                    content += f"\nOBSERVATION:\n{step['observation']}\n"
            else:
                # Handle step objects
                if hasattr(step, 'thought') and step.thought:
                    content += f"\nTHOUGHT:\n{step.thought}\n"
                if hasattr(step, 'action') and step.action:
                    action_str = f"{step.action.ptool_name}({step.action.arguments})"
                    content += f"\nACTION:\n{action_str}\n"
                if hasattr(step, 'observation') and step.observation:
                    obs = step.observation
                    if hasattr(obs, 'result'):
                        content += f"\nOBSERVATION:\n{obs.result}\n"
                    else:
                        content += f"\nOBSERVATION:\n{obs}\n"

            content += "\n"

        filepath.write_text(content, encoding='utf-8')

    def get_log_path(self, instance_id: Any) -> Path:
        """Get the path where an instance would be logged."""
        try:
            iid = int(instance_id)
            return self.log_dir / f"instance_{iid:04d}.txt"
        except (ValueError, TypeError):
            return self.log_dir / f"instance_{instance_id}.txt"

    def log_audit_trace(
        self,
        instance_id: Any,
        calculator_name: str,
        steps: list,
        final_answer: Any = None,
        ground_truth: Any = None,
        is_correct: bool = False,
        error: Optional[str] = None,
        audit_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a multi-step trace with audit information (for L3+ with critic/repair).

        Args:
            instance_id: Dataset instance ID
            calculator_name: Name of the medical calculator
            steps: List of trace steps (thought, action, observation)
            final_answer: The final answer
            ground_truth: Expected answer
            is_correct: Whether prediction was correct
            error: Any error that occurred
            audit_info: Dictionary with critic evaluation and repair details
        """
        try:
            iid = int(instance_id)
            filename = f"instance_{iid:04d}.txt"
        except (ValueError, TypeError):
            filename = f"instance_{instance_id}.txt"
        filepath = self.log_dir / filename

        separator = "=" * 80
        step_sep = "-" * 40

        content = f"""{separator}
EXPERIMENT: {self.experiment_name}
INSTANCE: {instance_id}
CALCULATOR: {calculator_name}
TIMESTAMP: {datetime.now().isoformat()}
{separator}

GROUND TRUTH: {ground_truth}
FINAL ANSWER: {final_answer}
CORRECT: {is_correct}
{"ERROR: " + error if error else ""}
NUM STEPS: {len(steps)}

{separator}
TRACE
{separator}

"""

        for i, step in enumerate(steps, 1):
            content += f"{step_sep}\nSTEP {i}\n{step_sep}\n"

            if isinstance(step, dict):
                if 'thought' in step:
                    content += f"\nTHOUGHT:\n{step['thought']}\n"
                if 'action' in step:
                    content += f"\nACTION:\n{step['action']}\n"
                if 'observation' in step:
                    content += f"\nOBSERVATION:\n{step['observation']}\n"
            else:
                # Handle step objects
                if hasattr(step, 'thought') and step.thought:
                    content += f"\nTHOUGHT:\n{step.thought}\n"
                if hasattr(step, 'action') and step.action:
                    content += f"\nACTION:\n{step.action}\n"
                if hasattr(step, 'observation') and step.observation:
                    content += f"\nOBSERVATION:\n{step.observation}\n"

            content += "\n"

        # Add audit/critic/repair information
        if audit_info:
            content += f"\n{separator}\nAUDIT & REPAIR\n{separator}\n\n"

            # Critic evaluation
            if 'evaluation' in audit_info:
                eval_info = audit_info['evaluation']
                content += f"CRITIC VERDICT: {eval_info.get('verdict', 'N/A')}\n"
                content += f"CONFIDENCE: {eval_info.get('confidence', 'N/A')}\n"
                content += f"TOTAL VIOLATIONS: {eval_info.get('total_violations', 0)}\n"
                content += f"FAILED STEPS: {eval_info.get('failed_steps', [])}\n\n"

                # Audit reports
                if 'audit_reports' in eval_info:
                    content += "AUDIT REPORTS:\n"
                    for report in eval_info['audit_reports']:
                        content += f"  - {report.get('audit_name', 'unknown')}: {report.get('result', 'N/A')}\n"
                        if report.get('violations'):
                            for v in report['violations']:
                                content += f"      VIOLATION: {v.get('rule_name', '')} - {v.get('message', '')}\n"
                                if v.get('location'):
                                    content += f"        Location: step_index={v['location'].get('step_index')}\n"
                    content += "\n"

                # Repair suggestions
                if 'repair_suggestions' in eval_info:
                    content += "REPAIR SUGGESTIONS:\n"
                    for i, suggestion in enumerate(eval_info['repair_suggestions'], 1):
                        content += f"  {i}. {suggestion.get('action', 'unknown')}: step_index={suggestion.get('step_index')}\n"
                        content += f"     Reason: {suggestion.get('reason', 'N/A')}\n"
                    content += "\n"

            # Repair attempts
            if 'repair_attempts' in audit_info:
                content += f"REPAIR ATTEMPTS: {audit_info['repair_attempts']}\n"
            if 'repair_success' in audit_info:
                content += f"REPAIR SUCCESS: {audit_info['repair_success']}\n"
            if 'repair_actions' in audit_info:
                content += "REPAIR ACTIONS TAKEN:\n"
                for action in audit_info['repair_actions']:
                    content += f"  - {action}\n"
                content += "\n"

        filepath.write_text(content, encoding='utf-8')
