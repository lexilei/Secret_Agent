"""
Trace Builder: Generates workflow traces from natural language goals.

This is where "agentic" behavior lives - but it's contained:
1. Takes a goal (unstructured text)
2. Produces a structured WorkflowTrace
3. The trace can then be validated and executed deterministically

This is the key abstraction from William's document: the trace builder
is a special ptool that converts unstructured goals into verifiable plans.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Type

import logging

# Try to use loguru, fall back to standard logging
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .llm_backend import call_llm, ParseError
from .ptool import PToolSpec, get_registry
from .traces import TraceStep, WorkflowTrace


def build_trace(
    goal: str,
    available_ptools: Optional[List[PToolSpec]] = None,
    model: str = "deepseek-v3-0324",
    llm_backend: Optional[Callable] = None,
) -> WorkflowTrace:
    """
    Build a workflow trace from a natural language goal.

    This is the "trace builder" - a special ptool that takes a goal
    and produces a plan (WorkflowTrace) for achieving it.

    Args:
        goal: Natural language description of what to accomplish
        available_ptools: List of ptools to use (defaults to all registered)
        model: LLM model to use for planning
        llm_backend: Optional custom LLM backend

    Returns:
        WorkflowTrace with planned steps
    """
    registry = get_registry()

    if available_ptools is None:
        available_ptools = registry.list_all()

    if not available_ptools:
        raise ValueError("No ptools available for planning")

    logger.info(f"Building trace for goal: {goal[:100]}...")
    logger.debug(f"Available ptools: {[p.name for p in available_ptools]}")

    # Format the planning prompt
    prompt = _format_planning_prompt(goal, available_ptools)

    # Call LLM
    if llm_backend:
        response = llm_backend(prompt, model)
    else:
        response = call_llm(prompt, model)

    logger.trace(f"Planning response:\n{response}")

    # Parse the response into a WorkflowTrace
    trace = _parse_planning_response(goal, response, available_ptools)
    trace.model_used = model

    logger.info(f"Built trace with {len(trace.steps)} steps")
    return trace


def _format_planning_prompt(
    goal: str,
    available_ptools: List[PToolSpec],
) -> str:
    """Format the prompt for the planning LLM call."""
    lines = [
        "You are a planning agent. Given a goal and available tools (ptools),",
        "create a step-by-step plan to achieve the goal.",
        "",
        "## Available ptools:",
        "",
    ]

    for spec in available_ptools:
        lines.append(f"### {spec.name}")
        lines.append(f"Signature: {spec.get_signature_str()}")
        lines.append(f"Description: {spec.docstring.strip()}")
        lines.append("")

    lines.extend([
        "## Goal:",
        goal,
        "",
        "## Instructions:",
        "1. Analyze the goal and break it into steps",
        "2. For each step, identify which ptool to use",
        "3. Specify the arguments for each ptool call",
        "4. Steps can reference results from previous steps using $step_N notation",
        "",
        "## Output Format:",
        "Return a JSON array of steps. Each step should have:",
        '- "ptool": name of the ptool to call',
        '- "args": dictionary of arguments',
        '- "goal": brief description of what this step accomplishes',
        "",
        "Example:",
        "```json",
        "[",
        '  {"ptool": "extract_data", "args": {"text": "input text"}, "goal": "Extract relevant data"},',
        '  {"ptool": "process_data", "args": {"data": "$step_0"}, "goal": "Process extracted data"}',
        "]",
        "```",
        "",
        "Now create a plan for the goal. Return ONLY the JSON array.",
    ])

    return "\n".join(lines)


def _parse_planning_response(
    goal: str,
    response: str,
    available_ptools: List[PToolSpec],
) -> WorkflowTrace:
    """Parse the LLM response into a WorkflowTrace."""
    # Extract JSON from response
    json_match = re.search(r'\[[\s\S]*\]', response)
    if not json_match:
        raise ParseError(f"No JSON array found in planning response: {response[:200]}...")

    try:
        steps_data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON in planning response: {e}")

    if not isinstance(steps_data, list):
        raise ParseError(f"Expected JSON array, got: {type(steps_data)}")

    # Create trace
    trace = WorkflowTrace(goal=goal)

    # Build lookup for ptool specs
    ptool_lookup = {spec.name: spec for spec in available_ptools}

    for i, step_data in enumerate(steps_data):
        if not isinstance(step_data, dict):
            logger.warning(f"Skipping invalid step {i}: {step_data}")
            continue

        ptool_name = step_data.get("ptool")
        if not ptool_name:
            logger.warning(f"Step {i} missing ptool name")
            continue

        # Validate ptool exists
        if ptool_name not in ptool_lookup:
            logger.warning(f"Unknown ptool in step {i}: {ptool_name}")
            # Still add it - execution will fail if it's truly unknown
            expected_type = Any
        else:
            expected_type = ptool_lookup[ptool_name].return_type

        trace.add_step(
            ptool_name=ptool_name,
            args=step_data.get("args", {}),
            expected_type=expected_type,
            goal=step_data.get("goal"),
        )

    return trace


# Higher-level function: plan and execute
def plan_and_execute(
    goal: str,
    available_ptools: Optional[List[PToolSpec]] = None,
    model: str = "deepseek-v3-0324",
    max_retries: int = 3,
    llm_backend: Optional[Callable] = None,
):
    """
    Plan and execute a goal in one call.

    This combines trace building and execution:
    1. Build a trace from the goal
    2. Execute the trace
    3. Return the result

    Args:
        goal: Natural language goal
        available_ptools: Optional list of ptools to use
        model: LLM model for planning
        max_retries: Max retries per step during execution
        llm_backend: Optional custom LLM backend

    Returns:
        ExecutionResult
    """
    from .executor import TraceExecutor

    # Build trace
    trace = build_trace(
        goal=goal,
        available_ptools=available_ptools,
        model=model,
        llm_backend=llm_backend,
    )

    # Execute trace
    executor = TraceExecutor(
        max_retries=max_retries,
        llm_backend=llm_backend,
    )
    return executor.execute(trace)


# Manual trace building helpers

class TraceBuilder:
    """
    Fluent interface for manually building traces.

    Example:
        trace = (TraceBuilder("Calculate BMI")
            .step("parse_weight", {"text": input}, goal="Extract weight")
            .step("parse_height", {"text": input}, goal="Extract height")
            .step("compute_bmi", {"weight": "$step_0", "height": "$step_1"})
            .build())
    """

    def __init__(self, goal: str):
        self._goal = goal
        self._steps: List[Dict[str, Any]] = []

    def step(
        self,
        ptool_name: str,
        args: Dict[str, Any],
        goal: Optional[str] = None,
    ) -> TraceBuilder:
        """Add a step to the trace."""
        self._steps.append({
            "ptool_name": ptool_name,
            "args": args,
            "goal": goal,
        })
        return self

    def build(self) -> WorkflowTrace:
        """Build the WorkflowTrace."""
        registry = get_registry()
        trace = WorkflowTrace(goal=self._goal)

        for step_data in self._steps:
            ptool_name = step_data["ptool_name"]
            spec = registry.get(ptool_name)
            expected_type = spec.return_type if spec else Any

            trace.add_step(
                ptool_name=ptool_name,
                args=step_data["args"],
                expected_type=expected_type,
                goal=step_data.get("goal"),
            )

        return trace
