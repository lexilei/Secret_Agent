"""
Workflow Traces: Data structures for representing and executing ptool call sequences.

A WorkflowTrace represents a planned sequence of ptool calls to achieve a goal.
Each step in the trace can be executed and validated independently.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type


class StepStatus(Enum):
    """Status of a trace step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TraceStep:
    """
    A single step in a workflow trace.

    Represents a planned ptool call with its arguments and expected output type.
    """
    ptool_name: str
    args: Dict[str, Any]
    expected_type: Type = field(default=Any)

    # Optional metadata
    goal: Optional[str] = None  # Why this step is needed
    step_id: Optional[str] = None  # Unique identifier

    # Execution results (filled after execution)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        """Duration of step execution in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ptool_name": self.ptool_name,
            "args": self.args,
            "expected_type": str(self.expected_type),
            "goal": self.goal,
            "step_id": self.step_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceStep:
        """Create from dictionary."""
        step = cls(
            ptool_name=data["ptool_name"],
            args=data["args"],
            goal=data.get("goal"),
            step_id=data.get("step_id"),
        )
        if "status" in data:
            step.status = StepStatus(data["status"])
        if "result" in data:
            step.result = data["result"]
        if "error" in data:
            step.error = data["error"]
        return step


@dataclass
class WorkflowTrace:
    """
    A complete workflow trace: a sequence of ptool calls to achieve a goal.

    The trace represents a plan that can be executed step by step.
    """
    goal: str
    steps: List[TraceStep] = field(default_factory=list)

    # Metadata
    trace_id: Optional[str] = None
    created_at: Optional[str] = None
    model_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.trace_id is None:
            import uuid
            self.trace_id = str(uuid.uuid4())[:8]

    def add_step(
        self,
        ptool_name: str,
        args: Dict[str, Any],
        expected_type: Type = Any,
        goal: Optional[str] = None,
    ) -> TraceStep:
        """Add a step to the trace."""
        step = TraceStep(
            ptool_name=ptool_name,
            args=args,
            expected_type=expected_type,
            goal=goal,
            step_id=f"{self.trace_id}_{len(self.steps)}",
        )
        self.steps.append(step)
        return step

    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(s.status == StepStatus.COMPLETED for s in self.steps)

    @property
    def has_failed(self) -> bool:
        """Check if any step has failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)

    @property
    def pending_steps(self) -> List[TraceStep]:
        """Get all pending steps."""
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    @property
    def completed_steps(self) -> List[TraceStep]:
        """Get all completed steps."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    def get_results(self) -> Dict[str, Any]:
        """Get results from all completed steps."""
        return {
            step.step_id or f"step_{i}": step.result
            for i, step in enumerate(self.steps)
            if step.status == StepStatus.COMPLETED
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "goal": self.goal,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "model_used": self.model_used,
            "metadata": self.metadata,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowTrace:
        """Create from dictionary."""
        trace = cls(
            goal=data["goal"],
            trace_id=data.get("trace_id"),
            created_at=data.get("created_at"),
            model_used=data.get("model_used"),
            metadata=data.get("metadata", {}),
        )
        trace.steps = [TraceStep.from_dict(s) for s in data.get("steps", [])]
        return trace

    @classmethod
    def from_json(cls, json_str: str) -> WorkflowTrace:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else ("failed" if self.has_failed else "pending")
        return f"<WorkflowTrace {self.trace_id}: {len(self.steps)} steps, {status}>"


@dataclass
class ExecutionResult:
    """
    Result of executing a workflow trace.

    Contains the trace (with updated step statuses) and overall execution metadata.
    """
    trace: WorkflowTrace
    success: bool = False
    final_result: Optional[Any] = None
    error: Optional[str] = None

    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Statistics
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    retries: int = 0

    @property
    def duration(self) -> Optional[float]:
        """Total execution duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace": self.trace.to_dict(),
            "success": self.success,
            "final_result": self.final_result,
            "error": self.error,
            "duration": self.duration,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "retries": self.retries,
        }

    def __repr__(self) -> str:
        status = "success" if self.success else "failed"
        return f"<ExecutionResult {status}: {self.completed_steps}/{self.total_steps} steps>"


# Helper functions for building traces programmatically

def step(
    ptool_name: str,
    args: Dict[str, Any],
    expected_type: Type = Any,
    goal: Optional[str] = None,
) -> TraceStep:
    """Create a trace step."""
    return TraceStep(
        ptool_name=ptool_name,
        args=args,
        expected_type=expected_type,
        goal=goal,
    )


def trace(goal: str, *steps: TraceStep) -> WorkflowTrace:
    """Create a workflow trace from steps."""
    t = WorkflowTrace(goal=goal)
    for s in steps:
        t.steps.append(s)
        s.step_id = f"{t.trace_id}_{len(t.steps) - 1}"
    return t
