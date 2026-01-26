"""
Repair Agent: Automatically repair traces based on critic feedback.

This module implements the repair component of William's Approach 3:
using critic evaluations to fix problematic traces.

Key Classes:
    - RepairAction: Describes a repair action to take
    - RepairResult: Result of a repair attempt
    - RepairAgent: Main agent that performs repairs

The repair agent can:
- Regenerate failed steps
- Add missing steps
- Remove redundant steps
- Reorder steps
- Use ICL examples for better generation

Example:
    >>> from ptool_framework.repair import RepairAgent
    >>> from ptool_framework.critic import TraceCritic, CriticVerdict
    >>>
    >>> critic = TraceCritic(audits=[...])
    >>> repair_agent = RepairAgent(critic=critic)
    >>>
    >>> eval = critic.evaluate(trace, goal=goal)
    >>> if eval.verdict == CriticVerdict.REPAIR_NEEDED:
    ...     result = repair_agent.repair(trace, eval, goal)
    ...     if result.success:
    ...         print("Trace repaired successfully!")
    ...         fixed_trace = result.repaired_trace
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from .traces import WorkflowTrace, TraceStep
    from .react import ReActTrajectory, ReActAgent
    from .trace_store import TraceStore, ExecutionTrace

from .traces import WorkflowTrace, TraceStep, StepStatus
from .critic import TraceCritic, CriticEvaluation, CriticVerdict
from .audit.dataframe_converter import convert_trace


class RepairActionType(Enum):
    """Types of repair actions."""
    REGENERATE_STEP = "regenerate_step"
    ADD_STEP = "add_step"
    REMOVE_STEP = "remove_step"
    MODIFY_ARGS = "modify_args"
    REORDER_STEPS = "reorder_steps"


@dataclass
class RepairAction:
    """
    Describes a single repair action.

    Attributes:
        action_type: Type of repair
        step_index: Index of step to repair (if applicable)
        details: Additional details for the repair
        reason: Why this repair is needed

    Example:
        >>> action = RepairAction(
        ...     action_type=RepairActionType.REGENERATE_STEP,
        ...     step_index=2,
        ...     reason="Step failed during execution"
        ... )
    """

    action_type: RepairActionType
    step_index: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type.value,
            "step_index": self.step_index,
            "details": self.details,
            "reason": self.reason,
        }


@dataclass
class RepairResult:
    """
    Result of a repair attempt.

    Attributes:
        success: Whether repair was successful
        repaired_trace: The repaired trace (if successful)
        actions_taken: List of actions performed
        iterations: Number of repair iterations
        final_evaluation: Critic evaluation after repair
        original_verdict: Original verdict before repair
        error: Error message if repair failed

    Example:
        >>> result = repair_agent.repair(trace, eval, goal)
        >>> if result.success:
        ...     print(f"Fixed in {result.iterations} iterations")
        ...     print(f"Actions: {len(result.actions_taken)}")
    """

    success: bool
    repaired_trace: Optional[WorkflowTrace] = None
    actions_taken: List[RepairAction] = field(default_factory=list)
    iterations: int = 0
    final_evaluation: Optional[CriticEvaluation] = None
    original_verdict: Optional[CriticVerdict] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "repaired_trace": self.repaired_trace.to_dict() if self.repaired_trace else None,
            "actions_taken": [a.to_dict() for a in self.actions_taken],
            "iterations": self.iterations,
            "original_verdict": self.original_verdict.value if self.original_verdict else None,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Repair Result: {status}",
            f"  Iterations: {self.iterations}",
            f"  Actions taken: {len(self.actions_taken)}",
        ]
        if self.success and self.final_evaluation:
            lines.append(f"  Final verdict: {self.final_evaluation.verdict.value}")
        if self.error:
            lines.append(f"  Error: {self.error}")
        return "\n".join(lines)


class RepairAgent:
    """
    Agent that repairs traces based on critic feedback.

    The repair agent implements an iterative repair loop:
    1. Get repair suggestions from critic evaluation
    2. Apply repairs to trace
    3. Re-evaluate with critic
    4. Repeat until ACCEPT or max attempts

    Attributes:
        critic: TraceCritic for evaluation
        max_attempts: Maximum repair iterations
        model: LLM model for regeneration

    Example:
        >>> repair_agent = RepairAgent(
        ...     critic=critic,
        ...     max_attempts=3,
        ...     model="deepseek-v3-0324"
        ... )
        >>>
        >>> result = repair_agent.repair(trace, evaluation, goal)
    """

    def __init__(
        self,
        critic: Optional[TraceCritic] = None,
        react_agent: Optional["ReActAgent"] = None,
        trace_store: Optional["TraceStore"] = None,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
        max_attempts: int = 3,
        use_icl_examples: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the repair agent.

        Args:
            critic: TraceCritic for evaluating repairs
            react_agent: Optional ReActAgent for regeneration
            trace_store: TraceStore for ICL examples
            model: LLM model for regeneration
            llm_backend: Custom LLM backend
            max_attempts: Maximum repair iterations
            use_icl_examples: Whether to use ICL examples
        """
        self.critic = critic or TraceCritic()
        self.react_agent = react_agent
        self.trace_store = trace_store
        self.model = model
        self.llm_backend = llm_backend
        self.max_attempts = max_attempts
        self.use_icl_examples = use_icl_examples
        self.verbose = verbose

    # =========================================================================
    # NEW SIMPLIFIED REPAIR: Rewind and Regenerate
    # =========================================================================

    def rewind_and_regenerate(
        self,
        trajectory: "ReActTrajectory",
        failing_step_index: int,
        audit_feedback: str,
        goal: str,
    ) -> RepairResult:
        """
        Simple repair: rewind to failing step, inject feedback, regenerate.

        This is the new simplified repair mechanism that:
        1. Truncates trajectory to steps before the failure
        2. Injects audit feedback into the goal
        3. Uses ReActAgent.continue_from() to regenerate

        Args:
            trajectory: The full ReAct trajectory
            failing_step_index: Which step failed (from audit)
            audit_feedback: What went wrong (from audit violation)
            goal: Original goal

        Returns:
            RepairResult with new trajectory
        """
        if self.react_agent is None:
            return RepairResult(
                success=False,
                error="ReActAgent required for rewind_and_regenerate",
                original_verdict=CriticVerdict.REPAIR_NEEDED,
            )

        if self.verbose:
            print(f"\n[Repair] Rewind to step {failing_step_index}, regenerate with feedback")
            print(f"  Feedback: {audit_feedback[:100]}...")

        # 1. Truncate trajectory to steps before failure
        from .react import ReActTrajectory, ReActStep
        truncated_steps = list(trajectory.steps[:failing_step_index])

        # 2. Create enhanced goal with audit feedback
        enhanced_goal = (
            f"{goal}\n\n"
            f"IMPORTANT: Your previous attempt failed at step {failing_step_index + 1}.\n"
            f"Reason: {audit_feedback}\n"
            f"Please avoid this error and complete the task correctly."
        )

        # 3. Create new trajectory from truncated state
        new_trajectory = ReActTrajectory(
            trajectory_id=trajectory.trajectory_id or "repair",
            goal=enhanced_goal,
            steps=truncated_steps,
        )

        # 4. Use ReActAgent to continue from there
        try:
            result = self.react_agent.continue_from(new_trajectory)

            # 5. Evaluate the new trajectory
            new_trace = result.trace
            if new_trace:
                new_eval = self.critic.evaluate(new_trace, goal=goal)
                success = new_eval.verdict == CriticVerdict.ACCEPT
            else:
                new_eval = None
                success = result.success

            return RepairResult(
                success=success,
                repaired_trace=new_trace,
                actions_taken=[RepairAction(
                    action_type=RepairActionType.REGENERATE_STEP,
                    step_index=failing_step_index,
                    reason=audit_feedback,
                    details={"method": "rewind_and_regenerate"},
                )],
                iterations=1,
                final_evaluation=new_eval,
                original_verdict=CriticVerdict.REPAIR_NEEDED,
            )
        except Exception as e:
            return RepairResult(
                success=False,
                error=f"Rewind and regenerate failed: {e}",
                original_verdict=CriticVerdict.REPAIR_NEEDED,
            )

    def _find_failing_step(self, evaluation: CriticEvaluation) -> Optional[int]:
        """
        Find the first failing step from evaluation.

        Looks at audit_reports, failed_steps, and repair_suggestions.
        """
        # First check explicit failed_steps
        if evaluation.failed_steps:
            return min(evaluation.failed_steps)

        # Check audit_reports for violations with step locations
        for report in evaluation.audit_reports:
            if hasattr(report, 'violations'):
                for violation in report.violations:
                    if hasattr(violation, 'location') and violation.location:
                        step_idx = violation.location.get('step_index')
                        if step_idx is not None:
                            return step_idx

        # Try to extract from repair suggestions
        for suggestion in evaluation.repair_suggestions:
            step_idx = suggestion.get('step_index')
            if step_idx is not None:
                return step_idx

        # Default to step 0 if we can't find a specific step
        return 0

    def _format_audit_feedback(self, evaluation: CriticEvaluation) -> str:
        """
        Format audit violations into feedback for the LLM.
        """
        feedback_parts = []

        # Extract violations from audit_reports
        for report in evaluation.audit_reports:
            if hasattr(report, 'violations'):
                for v in report.violations[:2]:  # Limit per report
                    if hasattr(v, 'message'):
                        feedback_parts.append(v.message)

        # Also include repair suggestions as feedback
        for suggestion in evaluation.repair_suggestions[:2]:
            reason = suggestion.get('reason', '')
            if reason and reason not in feedback_parts:
                feedback_parts.append(reason)

        if not feedback_parts:
            return "Trace evaluation failed - please try a different approach"

        return "; ".join(feedback_parts[:4])  # Max 4 feedback items

    def _trajectory_from_trace(
        self,
        trace: WorkflowTrace,
        goal: str,
    ) -> "ReActTrajectory":
        """
        Convert a WorkflowTrace back to a ReActTrajectory for rewind.
        """
        from .react import ReActTrajectory, ReActStep, Thought, Action, Observation

        steps = []
        for i, step in enumerate(trace.steps):
            # Create minimal ReActStep from TraceStep
            thought = Thought(
                content=f"Execute {step.ptool_name}",
                step_number=i,
            )
            action = Action(
                ptool_name=step.ptool_name,
                args=step.args or {},
                step_number=i,
            )
            observation = Observation(
                result=step.result,
                success=step.status == StepStatus.COMPLETED,
                error=step.error,
                step_number=i,
            )
            steps.append(ReActStep(
                thought=thought,
                action=action,
                observation=observation,
            ))

        return ReActTrajectory(
            goal=goal,
            steps=steps,
        )

    # =========================================================================
    # END OF NEW SIMPLIFIED REPAIR
    # =========================================================================

    def repair(
        self,
        trace: Union[WorkflowTrace, "ReActTrajectory", List[Dict]],
        evaluation: CriticEvaluation,
        goal: str,
        use_rewind: bool = True,
    ) -> RepairResult:
        """
        Attempt to repair a trace using rewind-and-regenerate.

        This simplified repair mechanism:
        1. Finds the failing step from audit feedback
        2. Rewinds to that step (truncates trajectory)
        3. Injects audit feedback into the prompt
        4. Regenerates from that point with full context

        Args:
            trace: The trace to repair (WorkflowTrace, ReActTrajectory, or list)
            evaluation: Critic evaluation with violations
            goal: The goal the trace should achieve
            use_rewind: Use rewind-and-regenerate (default True)

        Returns:
            RepairResult with success status and repaired trace

        Example:
            >>> result = repair_agent.repair(trace, evaluation, goal)
            >>> if result.success:
            ...     use_trace(result.repaired_trace)
        """
        original_verdict = evaluation.verdict

        # Check if already acceptable
        if evaluation.verdict == CriticVerdict.ACCEPT:
            working_trace = self._to_workflow_trace(trace)
            return RepairResult(
                success=True,
                repaired_trace=working_trace,
                original_verdict=original_verdict,
            )

        # Check if rejected (unfixable)
        if evaluation.verdict == CriticVerdict.REJECT:
            return RepairResult(
                success=False,
                error="Trace rejected as unfixable",
                original_verdict=original_verdict,
            )

        # Use rewind-and-regenerate if we have a ReActAgent
        if use_rewind and self.react_agent is not None:
            # Convert trace to trajectory
            if hasattr(trace, 'steps') and hasattr(trace, 'goal'):
                # Already a trajectory-like object
                trajectory = trace
            else:
                # Convert WorkflowTrace to trajectory
                working_trace = self._to_workflow_trace(trace)
                if working_trace is None:
                    return RepairResult(
                        success=False,
                        error="Could not convert trace",
                        original_verdict=original_verdict,
                    )
                trajectory = self._trajectory_from_trace(working_trace, goal)

            # Find failing step and format feedback
            failing_step = self._find_failing_step(evaluation)
            feedback = self._format_audit_feedback(evaluation)

            if self.verbose:
                print(f"\n[Repair] Using rewind-and-regenerate")
                print(f"  Failing step: {failing_step}")
                print(f"  Feedback: {feedback[:80]}...")

            # Rewind and regenerate
            return self.rewind_and_regenerate(
                trajectory=trajectory,
                failing_step_index=failing_step,
                audit_feedback=feedback,
                goal=goal,
            )

        # Fallback: No ReActAgent available
        return RepairResult(
            success=False,
            error="ReActAgent required for repair (set use_rewind=True and provide react_agent)",
            original_verdict=original_verdict,
        )

    def _to_workflow_trace(
        self,
        trace: Union[WorkflowTrace, "ReActTrajectory", List[Dict]],
    ) -> Optional[WorkflowTrace]:
        """Convert various trace types to WorkflowTrace."""
        if isinstance(trace, WorkflowTrace):
            return trace

        # Handle ReActTrajectory
        if hasattr(trace, 'generated_trace') and trace.generated_trace:
            return trace.generated_trace

        # Handle list of dicts
        if isinstance(trace, list):
            wf_trace = WorkflowTrace(goal="Converted trace")
            for step_dict in trace:
                wf_trace.add_step(
                    ptool_name=step_dict.get("fn_name", step_dict.get("ptool_name", "unknown")),
                    args=step_dict.get("args", step_dict.get("inputs", {})),
                    goal=step_dict.get("goal"),
                )
            return wf_trace

        return None

    # =========================================================================
    # LEGACY REPAIR MECHANISM (WIP - CONCEPT IN DEVELOPMENT)
    # =========================================================================
    # The methods below implement a more complex repair approach with multiple
    # action types (regenerate, add, remove, reorder, modify). This approach
    # is preserved for future development but currently NOT USED.
    #
    # The active repair mechanism uses the simpler rewind_and_regenerate()
    # method above, which:
    # 1. Rewinds to the failing step
    # 2. Injects audit feedback into the prompt
    # 3. Uses ReActAgent.continue_from() to regenerate with full context
    #
    # To re-enable the legacy mechanism, set use_rewind=False in repair()
    # and uncomment the legacy code path in repair().
    # =========================================================================

    def _legacy_apply_suggestion(
        self,
        trace: WorkflowTrace,
        suggestion: Dict[str, Any],
        goal: str,
    ) -> Optional[RepairAction]:
        """[LEGACY] Apply a single repair suggestion."""
        action_type = suggestion.get("action", "")

        if action_type == "regenerate_step":
            return self._legacy_regenerate_step(
                trace,
                step_index=suggestion.get("step_index"),
                reason=suggestion.get("reason", ""),
                goal=goal,
            )
        elif action_type == "add_step":
            return self._legacy_add_missing_step(
                trace,
                suggestion=suggestion,
                goal=goal,
            )
        elif action_type == "remove_step":
            return self._legacy_remove_step(
                trace,
                step_index=suggestion.get("step_index"),
                reason=suggestion.get("reason", ""),
            )
        elif action_type == "reorder_steps":
            return self._legacy_reorder_steps(
                trace,
                suggestion=suggestion,
            )
        elif action_type == "modify_args":
            return self._legacy_modify_args(
                trace,
                suggestion=suggestion,
            )

        return None

    def _legacy_regenerate_step(
        self,
        trace: WorkflowTrace,
        step_index: Optional[int],
        reason: str,
        goal: str,
    ) -> Optional[RepairAction]:
        """Regenerate a failed step."""
        if step_index is None:
            if self.verbose:
                print(f"    [regenerate_step] step_index is None")
            return None

        if step_index >= len(trace.steps):
            if self.verbose:
                print(f"    [regenerate_step] step_index {step_index} >= trace length {len(trace.steps)}")
            return None

        step = trace.steps[step_index]
        if self.verbose:
            print(f"    [regenerate_step] Regenerating step {step_index}: {step.ptool_name}")

        # Special case: calculation errors should use Python calculator
        is_calculation_error = (
            "calculation" in step.ptool_name.lower() or
            "perform" in step.ptool_name.lower()
        ) and (
            "differs" in reason.lower() or
            "incorrect" in reason.lower() or
            "wrong" in reason.lower()
        )

        if is_calculation_error:
            if self.verbose:
                print(f"    [regenerate_step] Calculation error detected - using Python calculator")
            try:
                # Try to use Python calculator instead
                new_result = self._legacy_use_python_calculator(trace, step_index, goal)
                if new_result is not None:
                    step.result = new_result
                    step.status = StepStatus.COMPLETED
                    step.error = None
                    step.ptool_name = "python_calculate"  # Mark as Python-calculated
                    return RepairAction(
                        action_type=RepairActionType.REGENERATE_STEP,
                        step_index=step_index,
                        reason=reason + " (used Python calculator)",
                        details={"new_result": str(new_result)[:200], "method": "python_calculator"},
                    )
            except Exception as e:
                if self.verbose:
                    print(f"    [regenerate_step] Python calculator failed: {e}")
                # Fall through to LLM regeneration

        # Get ICL examples if available
        positive_examples, negative_examples = [], []
        if self.use_icl_examples and self.trace_store:
            positive_examples, negative_examples = self._legacy_get_icl_examples(
                step.ptool_name,
                error_type=step.error,
            )

        # Regenerate using LLM
        try:
            new_result = self._legacy_call_ptool_with_repair_context(
                ptool_name=step.ptool_name,
                args=step.args,
                goal=step.goal or goal,
                positive_examples=positive_examples,
                negative_examples=negative_examples,
                error_context=step.error,
            )

            # Update step
            step.result = new_result
            step.status = StepStatus.COMPLETED
            step.error = None

            return RepairAction(
                action_type=RepairActionType.REGENERATE_STEP,
                step_index=step_index,
                reason=reason,
                details={"new_result": str(new_result)[:200]},
            )

        except Exception as e:
            # Regeneration failed
            if self.verbose:
                print(f"    [regenerate_step] Exception: {e}")
            step.error = str(e)
            return None

    def _legacy_use_python_calculator(
        self,
        trace: WorkflowTrace,
        step_index: int,
        goal: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Use Python calculator to fix a calculation error.

        Extracts patient note and question from the goal, then uses
        the calculators module to get the correct answer.
        """
        try:
            # Import calculators lazily
            from benchmark.experiments import calculators

            # Extract patient note and question from goal
            # Goal format: "Patient Note:\n{note}\n\nTask: {question}"
            parts = goal.split("\n\nTask:")
            if len(parts) != 2:
                if self.verbose:
                    print(f"    [python_calc] Could not parse goal format")
                return None

            patient_note_part = parts[0]
            question = parts[1].strip()

            # Remove "Patient Note:" prefix
            patient_note = patient_note_part.replace("Patient Note:", "").strip()

            # Call the Python calculator
            calc_result = calculators.calculate(patient_note, question)
            if calc_result is None:
                if self.verbose:
                    print(f"    [python_calc] Calculator returned None")
                return None

            if self.verbose:
                print(f"    [python_calc] Got result: {calc_result.result} from {calc_result.calculator_name}")

            return {
                "calculator_name": calc_result.calculator_name,
                "result": calc_result.result,
                "extracted_values": calc_result.extracted_values,
                "formula_used": calc_result.formula_used,
            }

        except ImportError as e:
            if self.verbose:
                print(f"    [python_calc] Could not import calculators: {e}")
            return None
        except Exception as e:
            if self.verbose:
                print(f"    [python_calc] Error: {e}")
            return None

    def _legacy_add_missing_step(
        self,
        trace: WorkflowTrace,
        suggestion: Dict[str, Any],
        goal: str,
    ) -> Optional[RepairAction]:
        """Add a missing step to the trace."""
        details = suggestion.get("details", {})
        reason = suggestion.get("reason", "")

        # Determine what step to add
        step_to_add = self._legacy_determine_missing_step(trace, reason, goal)
        if not step_to_add:
            return None

        ptool_name, args, insert_index = step_to_add

        # Create new step
        new_step = TraceStep(
            ptool_name=ptool_name,
            args=args,
            goal=f"Added to fix: {reason[:100]}",
        )

        # Insert at appropriate position
        if insert_index is None:
            trace.steps.append(new_step)
            insert_index = len(trace.steps) - 1
        else:
            trace.steps.insert(insert_index, new_step)

        # Try to execute the new step
        try:
            result = self._legacy_call_ptool_with_repair_context(
                ptool_name=ptool_name,
                args=args,
                goal=goal,
            )
            new_step.result = result
            new_step.status = StepStatus.COMPLETED
        except Exception as e:
            new_step.error = str(e)
            new_step.status = StepStatus.FAILED

        return RepairAction(
            action_type=RepairActionType.ADD_STEP,
            step_index=insert_index,
            reason=reason,
            details={"ptool_name": ptool_name, "args": args},
        )

    def _legacy_remove_step(
        self,
        trace: WorkflowTrace,
        step_index: Optional[int],
        reason: str,
    ) -> Optional[RepairAction]:
        """Remove a redundant step."""
        if step_index is None or step_index >= len(trace.steps):
            return None

        removed_step = trace.steps.pop(step_index)

        return RepairAction(
            action_type=RepairActionType.REMOVE_STEP,
            step_index=step_index,
            reason=reason,
            details={"removed_ptool": removed_step.ptool_name},
        )

    def _legacy_reorder_steps(
        self,
        trace: WorkflowTrace,
        suggestion: Dict[str, Any],
    ) -> Optional[RepairAction]:
        """Reorder steps to fix order issues."""
        details = suggestion.get("details", {})
        reason = suggestion.get("reason", "")

        # Parse order constraint from reason
        # E.g., "'extract' must come before 'calculate'"
        import re
        match = re.search(r"'(\w+)'.*before.*'(\w+)'", reason)
        if not match:
            return None

        before_fn, after_fn = match.groups()

        # Find indices
        before_idx = None
        after_idx = None
        for i, step in enumerate(trace.steps):
            if before_fn in step.ptool_name.lower() and before_idx is None:
                before_idx = i
            if after_fn in step.ptool_name.lower():
                after_idx = i

        if before_idx is None or after_idx is None or before_idx < after_idx:
            return None  # Already in order or can't find

        # Swap
        trace.steps[before_idx], trace.steps[after_idx] = \
            trace.steps[after_idx], trace.steps[before_idx]

        return RepairAction(
            action_type=RepairActionType.REORDER_STEPS,
            details={"swapped": [before_idx, after_idx]},
            reason=reason,
        )

    def _legacy_modify_args(
        self,
        trace: WorkflowTrace,
        suggestion: Dict[str, Any],
    ) -> Optional[RepairAction]:
        """Modify arguments of a step."""
        step_index = suggestion.get("step_index")
        details = suggestion.get("details", {})
        new_args = details.get("new_args")

        if step_index is None or step_index >= len(trace.steps):
            return None
        if not new_args:
            return None

        step = trace.steps[step_index]
        old_args = step.args.copy()
        step.args.update(new_args)

        return RepairAction(
            action_type=RepairActionType.MODIFY_ARGS,
            step_index=step_index,
            reason=suggestion.get("reason", ""),
            details={"old_args": old_args, "new_args": new_args},
        )

    def _legacy_determine_missing_step(
        self,
        trace: WorkflowTrace,
        reason: str,
        goal: str,
    ) -> Optional[Tuple[str, Dict, Optional[int]]]:
        """Determine what step to add based on the reason."""
        from .llm_backend import call_llm

        # Build prompt to determine missing step
        current_steps = [
            f"{i+1}. {s.ptool_name}({s.args})"
            for i, s in enumerate(trace.steps)
        ]

        prompt = f"""A trace for goal "{goal}" is missing a step.

Current steps:
{chr(10).join(current_steps)}

Issue: {reason}

What step should be added to fix this? Format:
PTOOL: <ptool_name>
ARGS: {{"key": "value"}}
INSERT_AFTER: <step_number or 0 for beginning>

Available ptools: identify_calculator, extract_values, calculate, format_result, validate_input"""

        try:
            if self.llm_backend:
                response = self.llm_backend(prompt, self.model)
            else:
                response = call_llm(prompt, self.model)

            # Parse response
            import re
            ptool_match = re.search(r'PTOOL:\s*(\w+)', response)
            args_match = re.search(r'ARGS:\s*(\{.*?\})', response, re.DOTALL)
            insert_match = re.search(r'INSERT_AFTER:\s*(\d+)', response)

            if ptool_match:
                ptool_name = ptool_match.group(1)
                args = {}
                if args_match:
                    try:
                        import json
                        args = json.loads(args_match.group(1))
                    except:
                        pass
                insert_idx = int(insert_match.group(1)) if insert_match else None

                return (ptool_name, args, insert_idx)

        except Exception:
            pass

        return None

    def _legacy_get_icl_examples(
        self,
        ptool_name: str,
        error_type: Optional[str] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Get ICL examples from trace store."""
        positive = []
        negative = []

        if not self.trace_store:
            return positive, negative

        try:
            # Get successful traces
            good_traces = self.trace_store.get_traces(
                ptool_name=ptool_name,
                success_only=True,
                limit=5,
            )
            for trace in good_traces:
                positive.append({
                    "inputs": trace.inputs,
                    "output": trace.output,
                })

            # Get failed traces (if error_type known)
            if error_type:
                bad_traces = self.trace_store.get_traces(
                    ptool_name=ptool_name,
                    success_only=False,
                    limit=5,
                )
                for trace in bad_traces:
                    if not trace.success:
                        negative.append({
                            "inputs": trace.inputs,
                            "error": trace.error,
                        })

        except Exception:
            pass

        return positive, negative

    def _legacy_call_ptool_with_repair_context(
        self,
        ptool_name: str,
        args: Dict[str, Any],
        goal: str,
        positive_examples: Optional[List[Dict]] = None,
        negative_examples: Optional[List[Dict]] = None,
        error_context: Optional[str] = None,
    ) -> Any:
        """Call ptool with repair context for better results."""
        from .llm_backend import execute_ptool
        from .ptool import get_registry

        registry = get_registry()
        spec = registry.get(ptool_name)

        if spec is None:
            raise ValueError(f"Unknown ptool: {ptool_name}")

        # Build enhanced prompt with repair context
        repair_context = ""
        if error_context:
            repair_context += f"\nPrevious attempt failed with: {error_context}\n"
            repair_context += "Please avoid this error in your response.\n"

        if positive_examples:
            repair_context += "\nGood examples:\n"
            for ex in positive_examples[:2]:
                repair_context += f"  Input: {ex['inputs']} -> Output: {ex['output']}\n"

        if negative_examples:
            repair_context += "\nBad examples to avoid:\n"
            for ex in negative_examples[:2]:
                repair_context += f"  Input: {ex['inputs']} -> Error: {ex.get('error', 'failed')}\n"

        # Execute with context
        result = execute_ptool(
            spec,
            args,
            custom_backend=self.llm_backend,
            additional_context=repair_context if repair_context else None,
        )

        return result

    def _legacy_generate_basic_suggestions(
        self,
        trace: WorkflowTrace,
        evaluation: CriticEvaluation,
    ) -> List[Dict[str, Any]]:
        """Generate basic suggestions when none provided."""
        suggestions = []

        # Check for failed steps
        for i, step in enumerate(trace.steps):
            if step.status == StepStatus.FAILED:
                suggestions.append({
                    "action": "regenerate_step",
                    "step_index": i,
                    "reason": f"Step {i} failed: {step.error}",
                    "priority": "high",
                })

        # Check for empty trace
        if len(trace.steps) == 0:
            suggestions.append({
                "action": "add_step",
                "reason": "Trace is empty",
                "priority": "high",
            })

        return suggestions

    # =========================================================================
    # END OF LEGACY REPAIR MECHANISM
    # =========================================================================


# =============================================================================
# Convenience Functions
# =============================================================================

def repair_trace(
    trace: Union[WorkflowTrace, "ReActTrajectory", List[Dict]],
    goal: str,
    critic: Optional[TraceCritic] = None,
    max_attempts: int = 3,
) -> RepairResult:
    """
    Convenience function to repair a trace.

    Args:
        trace: Trace to repair
        goal: The goal
        critic: Optional critic (creates default if None)
        max_attempts: Max repair iterations

    Returns:
        RepairResult

    Example:
        >>> result = repair_trace(bad_trace, "Calculate BMI")
        >>> if result.success:
        ...     good_trace = result.repaired_trace
    """
    if critic is None:
        critic = TraceCritic()

    evaluation = critic.evaluate(trace, goal=goal)

    if evaluation.verdict == CriticVerdict.ACCEPT:
        return RepairResult(
            success=True,
            repaired_trace=trace if isinstance(trace, WorkflowTrace) else None,
            original_verdict=CriticVerdict.ACCEPT,
        )

    agent = RepairAgent(critic=critic, max_attempts=max_attempts)
    return agent.repair(trace, evaluation, goal)


def auto_repair(
    trace: Union[WorkflowTrace, "ReActTrajectory", List[Dict]],
    goal: str,
) -> Optional[WorkflowTrace]:
    """
    Automatically repair a trace, returning the fixed trace or None.

    Args:
        trace: Trace to repair
        goal: The goal

    Returns:
        Repaired WorkflowTrace or None if repair failed

    Example:
        >>> fixed = auto_repair(trace, "Calculate BMI")
        >>> if fixed:
        ...     execute(fixed)
    """
    result = repair_trace(trace, goal)
    return result.repaired_trace if result.success else None
