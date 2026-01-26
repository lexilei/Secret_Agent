"""
Trace Executor: Executes workflow traces with validation and error handling.

The executor runs through each step in a trace, calling the appropriate ptool,
validating outputs, and optionally retrying failed steps with reflection.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import logging

# Try to use loguru, fall back to standard logging
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .llm_backend import LLMError, ParseError, execute_ptool
from .ptool import PToolSpec, get_registry
from .traces import ExecutionResult, StepStatus, TraceStep, WorkflowTrace


class ExecutionError(Exception):
    """Error during trace execution."""
    pass


class TraceExecutor:
    """
    Executes workflow traces step by step.

    Features:
    - Validates outputs against expected types
    - Retries failed steps with reflection (Reflexion-style)
    - Tracks execution statistics
    - Supports custom LLM backends
    """

    def __init__(
        self,
        max_retries: int = 3,
        llm_backend: Optional[Callable] = None,
        stop_on_error: bool = True,
    ):
        """
        Initialize the executor.

        Args:
            max_retries: Maximum retries per step on failure
            llm_backend: Optional custom LLM backend
            stop_on_error: If True, stop execution on first error
        """
        self.max_retries = max_retries
        self.llm_backend = llm_backend
        self.stop_on_error = stop_on_error
        self._registry = get_registry()

    def execute(self, trace: WorkflowTrace) -> ExecutionResult:
        """
        Execute a workflow trace.

        Args:
            trace: The workflow trace to execute

        Returns:
            ExecutionResult with the executed trace and statistics
        """
        result = ExecutionResult(
            trace=trace,
            total_steps=len(trace.steps),
            start_time=time.time(),
        )

        logger.info(f"Executing trace {trace.trace_id}: {len(trace.steps)} steps")
        logger.debug(f"Goal: {trace.goal}")

        # Context for passing results between steps
        context: Dict[str, Any] = {}

        for i, step in enumerate(trace.steps):
            logger.info(f"Step {i + 1}/{len(trace.steps)}: {step.ptool_name}")

            try:
                step_result = self._execute_step(step, context)
                result.completed_steps += 1

                # Store result in context for subsequent steps
                context[step.step_id or f"step_{i}"] = step_result
                context[step.ptool_name] = step_result  # Also store by ptool name

            except ExecutionError as e:
                logger.error(f"Step {i + 1} failed: {e}")
                result.failed_steps += 1

                if self.stop_on_error:
                    result.error = str(e)
                    break

        # Finalize result
        result.end_time = time.time()
        result.success = result.failed_steps == 0 and result.completed_steps == result.total_steps

        if result.success and trace.steps:
            # Use the last step's result as the final result
            result.final_result = trace.steps[-1].result

        logger.info(f"Trace {trace.trace_id} finished: {result}")
        return result

    def _execute_step(
        self,
        step: TraceStep,
        context: Dict[str, Any],
    ) -> Any:
        """Execute a single step with retry logic."""
        step.status = StepStatus.RUNNING
        step.start_time = time.time()

        # Get ptool spec
        spec = self._registry.get(step.ptool_name)
        if spec is None:
            step.status = StepStatus.FAILED
            step.error = f"Unknown ptool: {step.ptool_name}"
            raise ExecutionError(step.error)

        # Resolve argument references (e.g., "$step_0" -> previous result)
        resolved_args = self._resolve_args(step.args, context)

        # Try to execute with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.debug(f"Retry {attempt}/{self.max_retries}")
                    # Add error context for reflection
                    resolved_args["_previous_error"] = str(last_error)

                result = execute_ptool(spec, resolved_args, self.llm_backend)

                # Success!
                step.status = StepStatus.COMPLETED
                step.result = result
                step.end_time = time.time()
                return result

            except (LLMError, ParseError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

        # All retries exhausted
        step.status = StepStatus.FAILED
        step.error = str(last_error)
        step.end_time = time.time()
        raise ExecutionError(f"Step {step.ptool_name} failed after {self.max_retries + 1} attempts: {last_error}")

    def _resolve_args(
        self,
        args: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve argument references in step arguments.

        References like "$step_0" or "$extract_options" are replaced with
        actual values from the context.
        """
        resolved = {}
        for key, value in args.items():
            if key.startswith("_"):
                continue  # Skip internal args
            resolved[key] = self._resolve_value(value, context)
        return resolved

    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        """Resolve a single value, handling references."""
        if isinstance(value, str) and value.startswith("$"):
            ref_name = value[1:]  # Remove $
            if ref_name in context:
                return context[ref_name]
            else:
                logger.warning(f"Unresolved reference: {value}")
                return value
        elif isinstance(value, list):
            return [self._resolve_value(v, context) for v in value]
        elif isinstance(value, dict):
            return {k: self._resolve_value(v, context) for k, v in value.items()}
        return value


# Convenience function for quick execution
def run_trace(
    trace: WorkflowTrace,
    max_retries: int = 3,
    llm_backend: Optional[Callable] = None,
) -> ExecutionResult:
    """
    Execute a workflow trace with default settings.

    Args:
        trace: The workflow trace to execute
        max_retries: Maximum retries per step
        llm_backend: Optional custom LLM backend

    Returns:
        ExecutionResult
    """
    executor = TraceExecutor(max_retries=max_retries, llm_backend=llm_backend)
    return executor.execute(trace)


# Simple execution without traces - just call ptools directly
def run_ptool(ptool_name: str, **kwargs) -> Any:
    """
    Execute a single ptool by name.

    Args:
        ptool_name: Name of the registered ptool
        **kwargs: Arguments to pass to the ptool

    Returns:
        The ptool result
    """
    registry = get_registry()
    spec = registry.get(ptool_name)
    if spec is None:
        raise ExecutionError(f"Unknown ptool: {ptool_name}")
    return execute_ptool(spec, kwargs)
