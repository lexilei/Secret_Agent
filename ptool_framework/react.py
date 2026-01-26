"""
ReAct Agent: Reasoning + Acting loop for ptool_framework.

This module implements a ReAct-style agent that:
1. Iteratively reasons (generates thoughts) about how to achieve a goal
2. Selects and executes ptools (actions) based on reasoning
3. Observes results and decides next steps
4. Generates WorkflowTraces that can be executed by TraceExecutor

Key insight (Hybrid Approach):
- ReAct loop generates instance-specific traces
- Traces are compatible with existing TraceExecutor
- Trajectories are stored for later distillation (William's approach 2)
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type

import logging

# Try to use loguru, fall back to standard logging
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .traces import WorkflowTrace, TraceStep, StepStatus, ExecutionResult
from .ptool import PToolSpec, get_registry
from .llm_backend import call_llm, execute_ptool, LLMError, ParseError


# ============================================================================
# Exceptions
# ============================================================================

class ActionParseError(Exception):
    """Error parsing action from thought."""
    pass


class ReActError(Exception):
    """Error during ReAct execution."""
    pass


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Thought:
    """A reasoning step in the ReAct loop."""
    content: str
    step_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "step_number": self.step_number,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Thought:
        return cls(
            content=data["content"],
            step_number=data["step_number"],
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class Action:
    """An action to take (a ptool call)."""
    ptool_name: str
    args: Dict[str, Any]
    step_number: int
    rationale: str = ""
    raw_action_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ptool_name": self.ptool_name,
            "args": self.args,
            "step_number": self.step_number,
            "rationale": self.rationale,
            "raw_action_text": self.raw_action_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Action:
        return cls(
            ptool_name=data["ptool_name"],
            args=data["args"],
            step_number=data["step_number"],
            rationale=data.get("rationale", ""),
            raw_action_text=data.get("raw_action_text", ""),
        )


@dataclass
class Observation:
    """Result of executing an action."""
    result: Any
    success: bool
    error: Optional[str] = None
    step_number: int = 0
    execution_time_ms: float = 0.0
    ptool_trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "step_number": self.step_number,
            "execution_time_ms": self.execution_time_ms,
            "ptool_trace_id": self.ptool_trace_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Observation:
        return cls(
            result=data["result"],
            success=data["success"],
            error=data.get("error"),
            step_number=data.get("step_number", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            ptool_trace_id=data.get("ptool_trace_id"),
        )


@dataclass
class ReActStep:
    """A complete thought-action-observation tuple."""
    thought: Thought
    action: Optional[Action] = None
    observation: Optional[Observation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought.to_dict(),
            "action": self.action.to_dict() if self.action else None,
            "observation": self.observation.to_dict() if self.observation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReActStep:
        return cls(
            thought=Thought.from_dict(data["thought"]),
            action=Action.from_dict(data["action"]) if data.get("action") else None,
            observation=Observation.from_dict(data["observation"]) if data.get("observation") else None,
        )


@dataclass
class ReActTrajectory:
    """Complete record of a ReAct episode for storage and distillation."""
    trajectory_id: str
    goal: str
    steps: List[ReActStep] = field(default_factory=list)

    # Outcome
    final_answer: Optional[str] = None
    success: bool = False
    termination_reason: str = ""  # "answer_found", "max_steps", "error"

    # The generated trace (key for hybrid approach)
    generated_trace: Optional[WorkflowTrace] = None

    # Metadata
    model_used: str = ""
    total_llm_calls: int = 0
    total_time_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_ptp_trace(self) -> str:
        """Convert to Program Trace Prompting format (text-based trace log)."""
        lines = []
        for step in self.steps:
            if step.action and step.observation:
                # Format args
                args_parts = []
                for k, v in step.action.args.items():
                    args_parts.append(f"{k}={repr(v)}")
                args_str = ", ".join(args_parts)

                lines.append(f"Calling {step.action.ptool_name}({args_str})...")
                lines.append(f"...{step.action.ptool_name} returned {repr(step.observation.result)}")
                lines.append("")

        if self.final_answer:
            lines.append(f"Final answer: {self.final_answer}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "generated_trace": self.generated_trace.to_dict() if self.generated_trace else None,
            "model_used": self.model_used,
            "total_llm_calls": self.total_llm_calls,
            "total_time_ms": self.total_time_ms,
            "created_at": self.created_at,
            "ptp_trace": self.to_ptp_trace(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReActTrajectory:
        traj = cls(
            trajectory_id=data["trajectory_id"],
            goal=data["goal"],
            final_answer=data.get("final_answer"),
            success=data.get("success", False),
            termination_reason=data.get("termination_reason", ""),
            model_used=data.get("model_used", ""),
            total_llm_calls=data.get("total_llm_calls", 0),
            total_time_ms=data.get("total_time_ms", 0.0),
            created_at=data.get("created_at", ""),
        )
        traj.steps = [ReActStep.from_dict(s) for s in data.get("steps", [])]
        if data.get("generated_trace"):
            traj.generated_trace = WorkflowTrace.from_dict(data["generated_trace"])
        return traj


@dataclass
class ReActResult:
    """Result of a ReAct run."""
    trajectory: ReActTrajectory
    success: bool
    answer: Optional[str] = None
    trace: Optional[WorkflowTrace] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory.to_dict(),
            "success": self.success,
            "answer": self.answer,
            "trace": self.trace.to_dict() if self.trace else None,
        }


# ============================================================================
# ReAct Agent
# ============================================================================

class ReActAgent:
    """
    ReAct agent that generates WorkflowTraces through iterative reasoning.

    Key insight (Hybrid Approach):
    1. Run the ReAct loop to generate a plan (WorkflowTrace)
    2. Execute the trace via TraceExecutor (optional)
    3. Store the trajectory for later distillation

    This enables William's behavior distillation approach 2:
    learning trace builders from ReAct runs.
    """

    def __init__(
        self,
        available_ptools: Optional[List[PToolSpec]] = None,
        model: str = "deepseek-v3-0324",
        max_steps: int = 10,
        llm_backend: Optional[Callable] = None,
        execution_mode: Literal["immediate", "deferred"] = "immediate",
        store_trajectories: bool = True,
        echo: bool = False,
    ):
        """
        Args:
            available_ptools: Which ptools the agent can use (default: all registered)
            model: LLM model for reasoning
            max_steps: Maximum ReAct iterations
            llm_backend: Custom LLM backend (for testing)
            execution_mode:
                "immediate" - execute each action as it's generated (classic ReAct)
                "deferred" - generate full plan first, then execute via TraceExecutor
            store_trajectories: Whether to store trajectories for distillation
            echo: Print debug output during execution
        """
        self.registry = get_registry()

        if available_ptools is not None:
            self.available_ptools = available_ptools
        else:
            self.available_ptools = self.registry.list_all()

        self.model = model
        self.max_steps = max_steps
        self.llm_backend = llm_backend
        self.execution_mode = execution_mode
        self.store_trajectories = store_trajectories
        self.echo = echo

        # For trajectory storage
        self._trajectory_store: Optional[ReActStore] = None
        if store_trajectories:
            self._trajectory_store = get_react_store()

    def run(self, goal: str) -> ReActResult:
        """
        Run the ReAct loop to achieve a goal.

        Returns:
            ReActResult containing the trajectory and execution result
        """
        trajectory_id = str(uuid.uuid4())[:8]
        steps: List[ReActStep] = []
        context: Dict[str, Any] = {}  # Accumulated results

        start_time = time.time()
        llm_calls = 0

        if self.echo:
            print(f"\n{'='*60}")
            print(f"ReAct Agent: {goal}")
            print(f"{'='*60}")

        for step_num in range(self.max_steps):
            if self.echo:
                print(f"\n--- Step {step_num + 1} ---")

            # 1. Generate thought
            thought = self._generate_thought(goal, steps, context)
            llm_calls += 1

            if self.echo:
                print(f"Thought: {thought.content[:200]}...")

            # 2. Check for termination signal in thought
            should_stop, reason, answer = self._check_termination_in_thought(thought)
            if should_stop:
                # Add final thought step (no action)
                steps.append(ReActStep(thought=thought, action=None, observation=None))

                if self.echo:
                    print(f"\nFinal Answer: {answer}")

                # Build and return trajectory
                trajectory = self._finalize_trajectory(
                    trajectory_id, goal, steps, answer, reason, llm_calls, start_time
                )
                return ReActResult(
                    trajectory=trajectory,
                    success=True,
                    answer=answer,
                    trace=trajectory.generated_trace,
                )

            # 3. Parse action from thought
            try:
                action = self._parse_action(thought)
                if self.echo:
                    print(f"Action: {action.ptool_name}({action.args})")
            except ActionParseError as e:
                # Could not parse action - record error and continue
                observation = Observation(
                    result=None,
                    success=False,
                    error=f"Could not parse action: {e}",
                    step_number=step_num,
                )
                steps.append(ReActStep(thought=thought, action=None, observation=observation))
                if self.echo:
                    print(f"Parse Error: {e}")
                continue

            # 4. Execute action
            observation = self._execute_action(action, context)

            if self.echo:
                if observation.success:
                    print(f"Observation: {repr(observation.result)[:200]}")
                else:
                    print(f"Error: {observation.error}")

            # 5. Update context with result
            if observation.success:
                context[f"step_{step_num}"] = observation.result
                context[action.ptool_name] = observation.result

            # 6. Record step
            steps.append(ReActStep(thought=thought, action=action, observation=observation))

        # Max steps reached
        if self.echo:
            print(f"\nMax steps ({self.max_steps}) reached")

        trajectory = self._finalize_trajectory(
            trajectory_id, goal, steps, None, "max_steps", llm_calls, start_time
        )
        return ReActResult(
            trajectory=trajectory,
            success=False,
            answer=None,
            trace=trajectory.generated_trace,
        )

    def continue_from(
        self,
        partial_trajectory: "ReActTrajectory",
        max_additional_steps: Optional[int] = None,
    ) -> "ReActResult":
        """
        Continue execution from a partial trajectory.

        This enables "rewind and regenerate" for repair:
        1. Truncate trajectory to step N
        2. Call continue_from() with the truncated trajectory
        3. Agent continues with full context of steps 0..N

        Args:
            partial_trajectory: Trajectory with existing steps to continue from
            max_additional_steps: Max new steps to generate (default: self.max_steps)

        Returns:
            ReActResult with completed trajectory
        """
        trajectory_id = partial_trajectory.trajectory_id or str(uuid.uuid4())[:8]
        goal = partial_trajectory.goal

        # Copy existing steps
        steps: List[ReActStep] = list(partial_trajectory.steps)

        # Rebuild context from existing steps
        context = self._rebuild_context(steps)

        start_time = time.time()
        llm_calls = 0

        if self.echo:
            print(f"\n{'='*60}")
            print(f"ReAct Agent (continuing from step {len(steps)}): {goal[:100]}...")
            print(f"{'='*60}")

        # Calculate how many more steps we can take
        max_total_steps = max_additional_steps or self.max_steps
        remaining_steps = max_total_steps

        for step_num in range(len(steps), len(steps) + remaining_steps):
            if self.echo:
                print(f"\n--- Step {step_num + 1} ---")

            # 1. Generate thought with full history
            thought = self._generate_thought(goal, steps, context)
            llm_calls += 1

            if self.echo:
                print(f"Thought: {thought.content[:200]}...")

            # 2. Check for termination
            should_stop, reason, answer = self._check_termination_in_thought(thought)
            if should_stop:
                steps.append(ReActStep(thought=thought, action=None, observation=None))

                if self.echo:
                    print(f"\nFinal Answer: {answer}")

                trajectory = self._finalize_trajectory(
                    trajectory_id, goal, steps, answer, reason, llm_calls, start_time
                )
                return ReActResult(
                    trajectory=trajectory,
                    success=True,
                    answer=answer,
                    trace=trajectory.generated_trace,
                )

            # 3. Parse action
            try:
                action = self._parse_action(thought)
                if self.echo:
                    print(f"Action: {action.ptool_name}({action.args})")
            except ActionParseError as e:
                observation = Observation(
                    result=None,
                    success=False,
                    error=f"Could not parse action: {e}",
                    step_number=step_num,
                )
                steps.append(ReActStep(thought=thought, action=None, observation=observation))
                if self.echo:
                    print(f"Parse Error: {e}")
                continue

            # 4. Execute action
            observation = self._execute_action(action, context)

            if self.echo:
                if observation.success:
                    print(f"Observation: {repr(observation.result)[:200]}")
                else:
                    print(f"Error: {observation.error}")

            # 5. Update context
            if observation.success:
                context[f"step_{step_num}"] = observation.result
                context[action.ptool_name] = observation.result

            # 6. Record step
            steps.append(ReActStep(thought=thought, action=action, observation=observation))

        # Max steps reached
        if self.echo:
            print(f"\nMax steps reached")

        trajectory = self._finalize_trajectory(
            trajectory_id, goal, steps, None, "max_steps", llm_calls, start_time
        )
        return ReActResult(
            trajectory=trajectory,
            success=False,
            answer=None,
            trace=trajectory.generated_trace,
        )

    def _rebuild_context(self, steps: List[ReActStep]) -> Dict[str, Any]:
        """
        Rebuild the context dictionary from existing steps.

        This allows continue_from() to resume with the same context
        that would have been built during original execution.
        """
        context: Dict[str, Any] = {}

        for i, step in enumerate(steps):
            if step.action and step.observation and step.observation.success:
                context[f"step_{i}"] = step.observation.result
                context[step.action.ptool_name] = step.observation.result

        return context

    def _generate_thought(
        self,
        goal: str,
        history: List[ReActStep],
        context: Dict[str, Any],
    ) -> Thought:
        """Generate a thought given the current state."""
        prompt = self._format_thought_prompt(goal, history, self.available_ptools)

        if self.llm_backend:
            response = self.llm_backend(prompt, self.model)
        else:
            response = call_llm(prompt, self.model)

        return Thought(
            content=response,
            step_number=len(history),
        )

    def _format_thought_prompt(
        self,
        goal: str,
        history: List[ReActStep],
        available_ptools: List[PToolSpec],
    ) -> str:
        """Format the prompt for generating a thought."""
        lines = [
            "You are a reasoning agent. Think step-by-step about how to achieve the goal.",
            "",
            "## Available Tools (ptools):",
            "",
        ]

        for spec in available_ptools:
            doc = spec.docstring.strip()[:150] if spec.docstring else "No description"
            lines.append(f"- **{spec.name}**: {doc}")

            # Format signature using the spec's method
            lines.append(f"  Signature: {spec.get_signature_str()}")
            # List required parameters explicitly
            param_names = list(spec.parameters.keys())
            if param_names:
                lines.append(f"  REQUIRED args: {param_names}")
            lines.append("")

        lines.extend([
            f"## Goal: {goal}",
            "",
        ])

        if history:
            lines.append("## Previous Steps:")
            for i, step in enumerate(history):
                lines.append(f"\n**Step {i + 1}:**")
                thought_preview = step.thought.content[:200]
                lines.append(f"  Thought: {thought_preview}{'...' if len(step.thought.content) > 200 else ''}")
                if step.action:
                    lines.append(f"  Action: {step.action.ptool_name}({step.action.args})")
                    if step.observation:
                        if step.observation.success:
                            result_str = repr(step.observation.result)[:200]
                            lines.append(f"  Result: {result_str}")
                        else:
                            lines.append(f"  Error: {step.observation.error}")
            lines.append("")

        lines.extend([
            "## Instructions:",
            "Think about what to do next. Then specify an action OR provide a final answer.",
            "",
            "Format your response as:",
            "",
            "<thought>Your reasoning about the current state and what to do next</thought>",
            "<action>ptool_name(arg1=value1, arg2=value2)</action>",
            "",
            "OR if you have the final answer:",
            "",
            "<thought>Your reasoning here</thought>",
            "<answer>Your final answer (just the number)</answer>",
            "",
            "## CRITICAL RULES:",
            "1. ALWAYS provide ALL required arguments - never call a ptool with empty args!",
            "2. Arguments must use Python syntax: strings in quotes, dicts with curly braces",
            "3. If you get 'Missing required parameters' error, re-read the signature and try again WITH all args",
            "",
            "## Correct Examples:",
            "<action>identify_calculator(clinical_text=\"Calculate BMI\")</action>",
            "<action>extract_clinical_values(patient_note=\"Patient is 70kg, 180cm\", required_values=[\"weight\", \"height\"])</action>",
            "<action>perform_calculation(calculator_name=\"BMI\", values={\"weight\": 70, \"height\": 1.8})</action>",
            "",
            "## WRONG (will cause error):",
            "<action>perform_calculation()</action>  ← MISSING ARGS!",
            "<action>perform_calculation({})</action>  ← WRONG FORMAT!",
        ])

        return "\n".join(lines)

    def _parse_action(self, thought: Thought) -> Action:
        """Parse an action from the thought content."""
        # Look for <action>ptool_name(args)</action>
        action_match = re.search(
            r'<action>\s*(\w+)\s*\((.*?)\)\s*</action>',
            thought.content,
            re.DOTALL
        )

        if not action_match:
            raise ActionParseError("No <action> tag found in thought")

        ptool_name = action_match.group(1)
        args_str = action_match.group(2)

        # Parse arguments
        args = self._parse_args(args_str)

        # Validate ptool exists
        valid_names = [p.name for p in self.available_ptools]
        if ptool_name not in valid_names:
            raise ActionParseError(f"Unknown ptool: {ptool_name}. Available: {valid_names}")

        return Action(
            ptool_name=ptool_name,
            args=args,
            step_number=thought.step_number,
            rationale=thought.content,
            raw_action_text=action_match.group(0),
        )

    def _parse_args(self, args_str: str) -> Dict[str, Any]:
        """Parse argument string into dict."""
        if not args_str.strip():
            return {}

        # Try JSON-style first
        try:
            # Wrap in braces if needed
            test_str = args_str.strip()
            if not test_str.startswith('{'):
                test_str = '{' + test_str + '}'
            return json.loads(test_str)
        except json.JSONDecodeError:
            pass

        # Try Python-style kwargs
        args = {}

        # Match: arg_name=value or arg_name="value" or arg_name='value'
        # Handle nested structures carefully
        pattern = r'(\w+)\s*=\s*'
        matches = list(re.finditer(pattern, args_str))

        for i, match in enumerate(matches):
            key = match.group(1)
            start = match.end()

            # Find the end of the value (next key or end of string)
            if i + 1 < len(matches):
                end = matches[i + 1].start()
                value_str = args_str[start:end].rstrip(', \t\n')
            else:
                value_str = args_str[start:].rstrip(', \t\n)')

            # Try to parse as Python literal
            try:
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # Fall back to string
                value = value_str.strip('"\'')

            args[key] = value

        return args

    def _execute_action(self, action: Action, context: Dict[str, Any]) -> Observation:
        """Execute an action and return observation."""
        start_time = time.time()

        try:
            # Get the ptool spec - first from available_ptools, then from registry
            spec = None
            for p in self.available_ptools:
                if p.name == action.ptool_name:
                    spec = p
                    break

            # Fall back to global registry if not in available_ptools
            if spec is None:
                spec = self.registry.get(action.ptool_name)

            if spec is None:
                return Observation(
                    result=None,
                    success=False,
                    error=f"Ptool not found: {action.ptool_name}",
                    step_number=action.step_number,
                )

            # Resolve any context references in args (e.g., $step_0)
            resolved_args = self._resolve_args(action.args, context)

            # Validate required parameters
            missing_params = []
            for param_name in spec.parameters.keys():
                if param_name not in resolved_args:
                    missing_params.append(param_name)

            if missing_params:
                # Return helpful error with signature
                sig = spec.get_signature_str()
                return Observation(
                    result=None,
                    success=False,
                    error=f"Missing required parameters: {missing_params}. Expected signature: {sig}",
                    step_number=action.step_number,
                )

            # Execute the ptool
            # Check if this is a Python-only function (no LLM call)
            if spec.model == "python" and spec.func is not None:
                # Call the Python function directly
                result = spec.func(**resolved_args)
            elif self.llm_backend:
                # Use custom backend for ptool execution
                result = execute_ptool(
                    spec, resolved_args,
                    custom_backend=self.llm_backend,
                    collect_traces=True,
                )
            else:
                result = execute_ptool(spec, resolved_args, collect_traces=True)

            execution_time = (time.time() - start_time) * 1000

            return Observation(
                result=result,
                success=True,
                step_number=action.step_number,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return Observation(
                result=None,
                success=False,
                error=str(e),
                step_number=action.step_number,
                execution_time_ms=execution_time,
            )

    def _resolve_args(self, args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve context references (like $step_0) in arguments."""
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith('$'):
                ref_key = value[1:]  # Remove $
                if ref_key in context:
                    resolved[key] = context[ref_key]
                else:
                    resolved[key] = value  # Keep as-is if not found
            else:
                resolved[key] = value
        return resolved

    def _check_termination_in_thought(
        self, thought: Thought
    ) -> Tuple[bool, str, Optional[str]]:
        """Check if thought indicates we have a final answer."""
        # Look for <answer> tag
        answer_match = re.search(
            r'<answer>(.*?)</answer>',
            thought.content,
            re.DOTALL
        )

        if answer_match:
            return True, "answer_found", answer_match.group(1).strip()

        return False, "", None

    def _build_trace(self, steps: List[ReActStep]) -> WorkflowTrace:
        """Convert successful ReAct steps into a WorkflowTrace."""
        trace = WorkflowTrace(
            goal=f"ReAct-generated trace from {len(steps)} steps",
            metadata={"source": "react_agent"}
        )

        for i, step in enumerate(steps):
            if step.action and step.observation and step.observation.success:
                # Get expected type from ptool spec
                spec = self.registry.get(step.action.ptool_name)
                expected_type = spec.return_type if spec else Any

                trace.add_step(
                    ptool_name=step.action.ptool_name,
                    args=step.action.args,
                    expected_type=expected_type,
                    goal=step.thought.content[:100],  # Truncated thought as goal
                )

                # Fill in execution results
                trace_step = trace.steps[-1]
                trace_step.status = StepStatus.COMPLETED
                trace_step.result = step.observation.result

        return trace

    def _finalize_trajectory(
        self,
        trajectory_id: str,
        goal: str,
        steps: List[ReActStep],
        answer: Optional[str],
        termination_reason: str,
        llm_calls: int,
        start_time: float,
    ) -> ReActTrajectory:
        """Finalize and store the trajectory."""
        total_time = (time.time() - start_time) * 1000

        # Build the WorkflowTrace
        generated_trace = self._build_trace(steps)
        generated_trace.metadata["react_trajectory_id"] = trajectory_id

        trajectory = ReActTrajectory(
            trajectory_id=trajectory_id,
            goal=goal,
            steps=steps,
            final_answer=answer,
            success=answer is not None,
            termination_reason=termination_reason,
            generated_trace=generated_trace,
            model_used=self.model,
            total_llm_calls=llm_calls,
            total_time_ms=total_time,
        )

        # Store trajectory
        if self.store_trajectories and self._trajectory_store:
            self._trajectory_store.store_trajectory(trajectory)

        return trajectory


# ============================================================================
# ReAct Store
# ============================================================================

class ReActStore:
    """
    Storage for ReAct trajectories to enable distillation.

    Storage structure:
    ~/.react_traces/
    ├── all_trajectories/
    │   └── trajectory_{id}.json
    ├── by_success/
    │   ├── successful.jsonl
    │   └── failed.jsonl
    └── ptp_traces/
        └── trajectory_{id}.txt  (PTP format)
    """

    def __init__(self, path: str = "~/.react_traces"):
        self.base_path = Path(os.path.expanduser(path))
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        (self.base_path / "all_trajectories").mkdir(parents=True, exist_ok=True)
        (self.base_path / "by_success").mkdir(parents=True, exist_ok=True)
        (self.base_path / "ptp_traces").mkdir(parents=True, exist_ok=True)

    def store_trajectory(self, trajectory: ReActTrajectory) -> None:
        """Store a trajectory for later analysis."""
        # Store full trajectory as JSON
        traj_file = self.base_path / "all_trajectories" / f"trajectory_{trajectory.trajectory_id}.json"
        with open(traj_file, 'w') as f:
            f.write(trajectory.to_json())

        # Store PTP format trace
        ptp_file = self.base_path / "ptp_traces" / f"trajectory_{trajectory.trajectory_id}.txt"
        with open(ptp_file, 'w') as f:
            f.write(f"Goal: {trajectory.goal}\n\n")
            f.write(trajectory.to_ptp_trace())

        # Append to success/failure index
        status_file = "successful.jsonl" if trajectory.success else "failed.jsonl"
        with open(self.base_path / "by_success" / status_file, 'a') as f:
            summary = {
                "trajectory_id": trajectory.trajectory_id,
                "goal": trajectory.goal,
                "steps": len(trajectory.steps),
                "success": trajectory.success,
                "termination_reason": trajectory.termination_reason,
                "created_at": trajectory.created_at,
                "model_used": trajectory.model_used,
            }
            f.write(json.dumps(summary) + "\n")

        logger.info(f"Stored trajectory {trajectory.trajectory_id} ({'success' if trajectory.success else 'failed'})")

    def get_trajectory(self, trajectory_id: str) -> Optional[ReActTrajectory]:
        """Load a trajectory by ID."""
        traj_file = self.base_path / "all_trajectories" / f"trajectory_{trajectory_id}.json"
        if not traj_file.exists():
            return None

        with open(traj_file) as f:
            data = json.load(f)
        return ReActTrajectory.from_dict(data)

    def get_ptp_trace(self, trajectory_id: str) -> Optional[str]:
        """Get the PTP format trace for a trajectory."""
        ptp_file = self.base_path / "ptp_traces" / f"trajectory_{trajectory_id}.txt"
        if not ptp_file.exists():
            return None

        with open(ptp_file) as f:
            return f.read()

    def list_trajectories(
        self,
        success_only: bool = False,
        failed_only: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List trajectory summaries."""
        results = []

        # Read from appropriate index file(s)
        files_to_read = []
        if success_only:
            files_to_read = [self.base_path / "by_success" / "successful.jsonl"]
        elif failed_only:
            files_to_read = [self.base_path / "by_success" / "failed.jsonl"]
        else:
            files_to_read = [
                self.base_path / "by_success" / "successful.jsonl",
                self.base_path / "by_success" / "failed.jsonl",
            ]

        for file_path in files_to_read:
            if file_path.exists():
                with open(file_path) as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
                            if len(results) >= limit:
                                break

        # Sort by created_at descending
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results[:limit]

    def get_successful_trajectories(
        self,
        limit: int = 100,
        goal_filter: Optional[str] = None,
    ) -> List[ReActTrajectory]:
        """Get successful trajectories for distillation."""
        summaries = self.list_trajectories(success_only=True, limit=limit * 2)

        if goal_filter:
            summaries = [s for s in summaries if goal_filter.lower() in s["goal"].lower()]

        trajectories = []
        for summary in summaries[:limit]:
            traj = self.get_trajectory(summary["trajectory_id"])
            if traj:
                trajectories.append(traj)

        return trajectories

    def get_distillation_candidates(
        self,
        min_trajectories: int = 5,
        min_success_rate: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Find goal patterns with enough successful trajectories for distillation."""
        # Group by goal similarity (simplified - could use embedding clustering)
        from collections import defaultdict

        goal_groups: Dict[str, List[Dict]] = defaultdict(list)

        all_summaries = self.list_trajectories(limit=1000)
        for summary in all_summaries:
            # Simple grouping by first few words of goal
            goal_key = " ".join(summary["goal"].lower().split()[:5])
            goal_groups[goal_key].append(summary)

        candidates = []
        for goal_pattern, summaries in goal_groups.items():
            if len(summaries) >= min_trajectories:
                success_count = sum(1 for s in summaries if s["success"])
                success_rate = success_count / len(summaries)

                if success_rate >= min_success_rate:
                    candidates.append({
                        "goal_pattern": goal_pattern,
                        "total_trajectories": len(summaries),
                        "success_count": success_count,
                        "success_rate": success_rate,
                        "trajectory_ids": [s["trajectory_id"] for s in summaries],
                    })

        # Sort by success count descending
        candidates.sort(key=lambda x: x["success_count"], reverse=True)
        return candidates


# Global store instance
_REACT_STORE: Optional[ReActStore] = None


def get_react_store() -> ReActStore:
    """Get the global ReAct trajectory store."""
    global _REACT_STORE
    if _REACT_STORE is None:
        _REACT_STORE = ReActStore()
    return _REACT_STORE


def set_react_store(store: ReActStore) -> None:
    """Set the global ReAct trajectory store."""
    global _REACT_STORE
    _REACT_STORE = store


# ============================================================================
# Convenience Functions
# ============================================================================

def react(goal: str, **kwargs) -> ReActResult:
    """
    Run a ReAct agent on a goal.

    Args:
        goal: The goal to achieve
        **kwargs: Arguments passed to ReActAgent

    Returns:
        ReActResult with trajectory, success status, and answer
    """
    agent = ReActAgent(**kwargs)
    return agent.run(goal)


def react_and_execute(goal: str, **kwargs) -> Tuple[ReActResult, Optional[ExecutionResult]]:
    """
    Run ReAct to generate a trace, then execute it with TraceExecutor.

    This is useful when you want to re-execute the generated trace
    with the full TraceExecutor features (retry, validation, etc.).

    Args:
        goal: The goal to achieve
        **kwargs: Arguments passed to ReActAgent

    Returns:
        Tuple of (ReActResult, ExecutionResult from TraceExecutor)
    """
    from .executor import TraceExecutor

    # Run ReAct
    result = react(goal, **kwargs)

    if result.trace and result.trace.steps:
        # Reset step statuses for re-execution
        for step in result.trace.steps:
            step.status = StepStatus.PENDING
            step.result = None

        # Execute with TraceExecutor
        executor = TraceExecutor(max_retries=2)
        exec_result = executor.execute(result.trace)
        return result, exec_result

    return result, None
