"""
Trace to DataFrame converter for SSRM-style auditing.

This module converts various trace types to pandas DataFrames with a consistent schema,
enabling structured audits to query and analyze trace data.

Supported trace types:
    - WorkflowTrace: From ptool_framework.traces
    - ReActTrajectory: From ptool_framework.react
    - ExecutionTrace: From ptool_framework.trace_store
    - List[Dict]: Generic list of step dictionaries

DataFrame Schema:
    - step_idx: int - Index of the step (0-based)
    - fn_name: str - Name of the ptool/function called
    - input: str - JSON serialized input (all args)
    - output: str - JSON serialized output
    - input_0, input_1, ...: Individual input arguments (extracted from args dict)
    - status: str - Step status (completed, failed, pending, etc.)
    - duration_ms: float - Execution time in milliseconds
    - goal: str - Step's goal/description (if available)
    - error: Optional[str] - Error message if step failed

Example:
    >>> from ptool_framework.audit.dataframe_converter import TraceDataFrameConverter
    >>> from ptool_framework.traces import WorkflowTrace
    >>>
    >>> trace = WorkflowTrace(goal="example")
    >>> trace.add_step("extract_data", {"text": "hello"})
    >>> trace.add_step("analyze", {"data": [1, 2, 3]})
    >>>
    >>> converter = TraceDataFrameConverter()
    >>> df, metadata = converter.convert(trace)
    >>> print(df.columns.tolist())
    ['step_idx', 'fn_name', 'input', 'output', 'status', 'duration_ms', 'goal', 'error', 'input_0']
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from ..traces import WorkflowTrace, TraceStep
    from ..react import ReActTrajectory, ReActStep
    from ..trace_store import ExecutionTrace


class TraceDataFrameConverter:
    """
    Converts various trace types to pandas DataFrames for auditing.

    The converter provides a unified interface for converting different trace
    formats into a consistent DataFrame schema that can be used by structured audits.

    Attributes:
        include_raw_data: Whether to include raw JSON in the output
        max_individual_inputs: Maximum number of input_N columns to create

    Example:
        >>> converter = TraceDataFrameConverter()
        >>> df, metadata = converter.convert(my_trace)
        >>>
        >>> # Query the DataFrame
        >>> extract_steps = df[df["fn_name"].str.contains("extract")]
        >>> failed_steps = df[df["status"] == "failed"]
    """

    def __init__(
        self,
        include_raw_data: bool = True,
        max_individual_inputs: int = 10,
    ):
        """
        Initialize the converter.

        Args:
            include_raw_data: Whether to include raw JSON in 'input' and 'output' columns
            max_individual_inputs: Maximum number of individual input columns (input_0, input_1, etc.)
        """
        self.include_raw_data = include_raw_data
        self.max_individual_inputs = max_individual_inputs

    def convert(
        self,
        trace: Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]],
    ) -> Tuple["pd.DataFrame", Dict[str, Any]]:
        """
        Convert a trace to DataFrame and metadata.

        This method auto-detects the trace type and calls the appropriate
        conversion method.

        Args:
            trace: The trace to convert. Can be:
                - WorkflowTrace: Workflow execution trace
                - ReActTrajectory: ReAct agent trajectory
                - ExecutionTrace: Single ptool execution
                - List[Dict]: List of step dictionaries

        Returns:
            Tuple of (DataFrame, metadata dict) where:
                - DataFrame has the standard schema
                - metadata contains trace_id, goal, success, final_answer, etc.

        Raises:
            TypeError: If trace type is not supported

        Example:
            >>> df, metadata = converter.convert(trace)
            >>> print(metadata["trace_id"])
            >>> print(df.head())
        """
        import pandas as pd

        # Import here to avoid circular imports
        from ..traces import WorkflowTrace
        from ..react import ReActTrajectory
        from ..trace_store import ExecutionTrace

        if isinstance(trace, WorkflowTrace):
            return self._convert_workflow_trace(trace)
        elif isinstance(trace, ReActTrajectory):
            return self._convert_react_trajectory(trace)
        elif isinstance(trace, ExecutionTrace):
            return self._convert_execution_trace(trace)
        elif isinstance(trace, list):
            return self._convert_dict_list(trace)
        else:
            raise TypeError(
                f"Unsupported trace type: {type(trace).__name__}. "
                "Supported types: WorkflowTrace, ReActTrajectory, ExecutionTrace, List[Dict]"
            )

    def _convert_workflow_trace(
        self,
        trace: "WorkflowTrace",
    ) -> Tuple["pd.DataFrame", Dict[str, Any]]:
        """
        Convert a WorkflowTrace to DataFrame.

        Args:
            trace: The workflow trace to convert

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        import pandas as pd

        rows = []
        all_input_keys: set = set()

        for idx, step in enumerate(trace.steps):
            row = self._step_to_row(
                step_idx=idx,
                fn_name=step.ptool_name,
                args=step.args,
                result=step.result,
                status=step.status.value if hasattr(step.status, 'value') else str(step.status),
                duration_ms=step.duration * 1000 if step.duration else None,
                goal=step.goal,
                error=step.error,
            )
            rows.append(row)

            # Track all input keys for column creation
            if step.args:
                all_input_keys.update(step.args.keys())

        # Create DataFrame
        df = pd.DataFrame(rows) if rows else self._empty_dataframe()

        # Add individual input columns
        df = self._add_individual_input_columns(df, all_input_keys)

        # Create metadata
        metadata = {
            "trace_id": trace.trace_id,
            "goal": trace.goal,
            "success": trace.is_complete and not trace.has_failed,
            "final_answer": trace.get_results() if trace.is_complete else None,
            "model_used": trace.model_used,
            "created_at": trace.created_at,
            "trace_type": "WorkflowTrace",
            "step_count": len(trace.steps),
            "extra_metadata": trace.metadata,
        }

        return df, metadata

    def _convert_react_trajectory(
        self,
        trajectory: "ReActTrajectory",
    ) -> Tuple["pd.DataFrame", Dict[str, Any]]:
        """
        Convert a ReActTrajectory to DataFrame.

        Only includes steps that have actions (skips pure thought steps).

        Args:
            trajectory: The ReAct trajectory to convert

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        import pandas as pd

        rows = []
        all_input_keys: set = set()

        for idx, step in enumerate(trajectory.steps):
            # Only include steps with actions
            if step.action is None:
                continue

            # Determine status from observation
            if step.observation:
                status = "completed" if step.observation.success else "failed"
                result = step.observation.result
                error = step.observation.error
                duration_ms = step.observation.execution_time_ms
            else:
                status = "pending"
                result = None
                error = None
                duration_ms = None

            row = self._step_to_row(
                step_idx=idx,
                fn_name=step.action.ptool_name,
                args=step.action.args,
                result=result,
                status=status,
                duration_ms=duration_ms,
                goal=step.thought.content[:200] if step.thought else None,  # Use thought as goal
                error=error,
            )

            # Add ReAct-specific fields
            row["thought"] = step.thought.content if step.thought else None
            row["rationale"] = step.action.rationale

            rows.append(row)

            # Track all input keys
            if step.action.args:
                all_input_keys.update(step.action.args.keys())

        # Create DataFrame
        df = pd.DataFrame(rows) if rows else self._empty_dataframe()

        # Add individual input columns
        df = self._add_individual_input_columns(df, all_input_keys)

        # Create metadata
        metadata = {
            "trace_id": trajectory.trajectory_id,
            "goal": trajectory.goal,
            "success": trajectory.success,
            "final_answer": trajectory.final_answer,
            "model_used": trajectory.model_used,
            "created_at": trajectory.created_at,
            "trace_type": "ReActTrajectory",
            "step_count": len([s for s in trajectory.steps if s.action]),
            "total_steps": len(trajectory.steps),
            "termination_reason": trajectory.termination_reason,
            "total_llm_calls": trajectory.total_llm_calls,
            "total_time_ms": trajectory.total_time_ms,
        }

        return df, metadata

    def _convert_execution_trace(
        self,
        trace: "ExecutionTrace",
    ) -> Tuple["pd.DataFrame", Dict[str, Any]]:
        """
        Convert a single ExecutionTrace to DataFrame.

        Note: ExecutionTrace represents a single ptool call, so the
        resulting DataFrame has only one row.

        Args:
            trace: The execution trace to convert

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        import pandas as pd

        row = self._step_to_row(
            step_idx=0,
            fn_name=trace.ptool_name,
            args=trace.inputs,
            result=trace.output,
            status="completed" if trace.success else "failed",
            duration_ms=trace.execution_time_ms,
            goal=None,
            error=trace.error,
        )

        df = pd.DataFrame([row])

        # Add individual input columns
        all_input_keys = set(trace.inputs.keys()) if trace.inputs else set()
        df = self._add_individual_input_columns(df, all_input_keys)

        metadata = {
            "trace_id": trace.trace_id,
            "goal": None,
            "success": trace.success,
            "final_answer": trace.output if trace.success else None,
            "model_used": trace.model_used,
            "created_at": trace.timestamp,
            "trace_type": "ExecutionTrace",
            "step_count": 1,
            "prompt": trace.prompt,
            "raw_response": trace.raw_response,
        }

        return df, metadata

    def _convert_dict_list(
        self,
        steps: List[Dict[str, Any]],
    ) -> Tuple["pd.DataFrame", Dict[str, Any]]:
        """
        Convert a list of step dictionaries to DataFrame.

        Expected dict format:
            {
                "fn_name": str,  # or "ptool_name" or "name"
                "args": Dict,    # or "inputs" or "input"
                "result": Any,   # or "output"
                "status": str,   # optional
                "duration_ms": float,  # optional
                "goal": str,     # optional
                "error": str,    # optional
            }

        Args:
            steps: List of step dictionaries

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        import pandas as pd

        rows = []
        all_input_keys: set = set()

        for idx, step in enumerate(steps):
            # Handle various field name conventions
            fn_name = step.get("fn_name") or step.get("ptool_name") or step.get("name", "unknown")
            args = step.get("args") or step.get("inputs") or step.get("input") or {}
            result = step.get("result") or step.get("output")
            status = step.get("status", "completed")
            duration_ms = step.get("duration_ms") or step.get("duration")
            goal = step.get("goal") or step.get("description")
            error = step.get("error")

            row = self._step_to_row(
                step_idx=idx,
                fn_name=fn_name,
                args=args if isinstance(args, dict) else {"value": args},
                result=result,
                status=status,
                duration_ms=duration_ms,
                goal=goal,
                error=error,
            )
            rows.append(row)

            # Track input keys
            if isinstance(args, dict):
                all_input_keys.update(args.keys())

        # Create DataFrame
        df = pd.DataFrame(rows) if rows else self._empty_dataframe()

        # Add individual input columns
        df = self._add_individual_input_columns(df, all_input_keys)

        # Extract trace-level metadata if available in first step
        first_step = steps[0] if steps else {}
        metadata = {
            "trace_id": first_step.get("trace_id", "unknown"),
            "goal": first_step.get("trace_goal"),
            "success": all(s.get("status") != "failed" for s in steps),
            "final_answer": steps[-1].get("result") if steps else None,
            "trace_type": "List[Dict]",
            "step_count": len(steps),
        }

        return df, metadata

    def _step_to_row(
        self,
        step_idx: int,
        fn_name: str,
        args: Dict[str, Any],
        result: Any,
        status: str,
        duration_ms: Optional[float],
        goal: Optional[str],
        error: Optional[str],
    ) -> Dict[str, Any]:
        """
        Convert step data to a row dictionary.

        Args:
            step_idx: Step index
            fn_name: Function/ptool name
            args: Input arguments
            result: Output result
            status: Step status
            duration_ms: Execution time
            goal: Step goal/description
            error: Error message

        Returns:
            Dictionary representing a DataFrame row
        """
        row = {
            "step_idx": step_idx,
            "fn_name": fn_name,
            "status": status,
            "duration_ms": duration_ms,
            "goal": goal,
            "error": error,
        }

        # Add serialized input/output if requested
        if self.include_raw_data:
            row["input"] = self._safe_json_serialize(args)
            row["output"] = self._safe_json_serialize(result)
        else:
            row["input"] = None
            row["output"] = None

        # Store raw args for individual column extraction
        row["_raw_args"] = args

        return row

    def _add_individual_input_columns(
        self,
        df: "pd.DataFrame",
        input_keys: set,
    ) -> "pd.DataFrame":
        """
        Add individual input columns (input_0, input_1, etc.) based on args.

        Args:
            df: DataFrame to modify
            input_keys: Set of all input keys found

        Returns:
            Modified DataFrame with input columns added
        """
        if df.empty or "_raw_args" not in df.columns:
            return df

        # Sort keys for consistent ordering
        sorted_keys = sorted(input_keys)[:self.max_individual_inputs]

        # Create input_N columns based on sorted keys
        for i, key in enumerate(sorted_keys):
            col_name = f"input_{i}"
            df[col_name] = df["_raw_args"].apply(
                lambda args: self._safe_json_serialize(args.get(key)) if isinstance(args, dict) else None
            )

        # Also create named columns for direct access
        for key in sorted_keys:
            col_name = f"input_{key}"
            df[col_name] = df["_raw_args"].apply(
                lambda args: self._safe_json_serialize(args.get(key)) if isinstance(args, dict) else None
            )

        # Remove temporary column
        df = df.drop(columns=["_raw_args"])

        return df

    def _empty_dataframe(self) -> "pd.DataFrame":
        """
        Create an empty DataFrame with the standard schema.

        Returns:
            Empty DataFrame with correct columns
        """
        import pandas as pd

        return pd.DataFrame(columns=[
            "step_idx", "fn_name", "input", "output",
            "status", "duration_ms", "goal", "error"
        ])

    def _safe_json_serialize(self, obj: Any) -> Optional[str]:
        """
        Safely serialize an object to JSON.

        Args:
            obj: Object to serialize

        Returns:
            JSON string or None if serialization fails
        """
        if obj is None:
            return None

        try:
            return json.dumps(obj, default=str)
        except (TypeError, ValueError):
            return str(obj)


def convert_trace(
    trace: Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]],
    **kwargs,
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """
    Convenience function to convert a trace to DataFrame.

    Args:
        trace: The trace to convert
        **kwargs: Additional arguments passed to TraceDataFrameConverter

    Returns:
        Tuple of (DataFrame, metadata dict)

    Example:
        >>> df, metadata = convert_trace(my_workflow_trace)
        >>> print(df["fn_name"].value_counts())
    """
    converter = TraceDataFrameConverter(**kwargs)
    return converter.convert(trace)


def batch_convert_traces(
    traces: List[Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]]],
    **kwargs,
) -> List[Tuple["pd.DataFrame", Dict[str, Any]]]:
    """
    Convert multiple traces to DataFrames.

    Args:
        traces: List of traces to convert
        **kwargs: Additional arguments passed to TraceDataFrameConverter

    Returns:
        List of (DataFrame, metadata dict) tuples

    Example:
        >>> results = batch_convert_traces([trace1, trace2, trace3])
        >>> for df, metadata in results:
        ...     print(f"Trace {metadata['trace_id']}: {len(df)} steps")
    """
    converter = TraceDataFrameConverter(**kwargs)
    return [converter.convert(trace) for trace in traces]


def traces_to_combined_dataframe(
    traces: List[Union["WorkflowTrace", "ReActTrajectory", "ExecutionTrace", List[Dict]]],
    include_trace_id: bool = True,
    **kwargs,
) -> "pd.DataFrame":
    """
    Convert multiple traces to a single combined DataFrame.

    Each row is tagged with its trace_id for grouping.

    Args:
        traces: List of traces to convert
        include_trace_id: Whether to add trace_id column to each row
        **kwargs: Additional arguments passed to TraceDataFrameConverter

    Returns:
        Combined DataFrame with all traces

    Example:
        >>> combined_df = traces_to_combined_dataframe([trace1, trace2, trace3])
        >>> # Group by trace and analyze
        >>> for trace_id, group in combined_df.groupby("trace_id"):
        ...     print(f"Trace {trace_id}: {len(group)} steps")
    """
    import pandas as pd

    converter = TraceDataFrameConverter(**kwargs)
    dfs = []

    for trace in traces:
        df, metadata = converter.convert(trace)
        if include_trace_id:
            df["trace_id"] = metadata.get("trace_id", "unknown")
        dfs.append(df)

    if not dfs:
        return converter._empty_dataframe()

    return pd.concat(dfs, ignore_index=True)
