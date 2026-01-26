"""
Trace Store: Persistent storage for execution traces.

This module provides trace collection and storage for:
1. Logging all ptool executions (inputs, outputs, timing)
2. Supporting behavior distillation (pattern analysis)
3. Feeding the observability dashboard with real-time events

Traces are stored as JSON files for easy inspection and analysis.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import threading
from queue import Queue

import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class ExecutionTrace:
    """A single ptool execution trace."""

    ptool_name: str
    inputs: Dict[str, Any]
    output: Any
    success: bool
    execution_time_ms: float
    model_used: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    error: Optional[str] = None
    prompt: Optional[str] = None  # The actual prompt sent to LLM
    raw_response: Optional[str] = None  # Raw LLM response before parsing

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionTrace:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TraceEvent:
    """Real-time event for dashboard streaming."""

    event_type: str  # "ptool_start", "llm_request", "llm_response", "ptool_complete", "error"
    timestamp: str
    data: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


class TraceStore:
    """
    Persistent storage for execution traces.

    Stores traces as JSON files in a directory structure:
    ~/.ptool_traces/
    ├── index.json              # Index of all traces
    ├── by_ptool/
    │   ├── extract_food_items.json
    │   ├── is_healthy.json
    │   └── ...
    └── by_session/
        ├── session_abc123.json
        └── ...
    """

    def __init__(
        self,
        path: str = "~/.ptool_traces",
        session_id: Optional[str] = None,
    ):
        self.base_path = Path(os.path.expanduser(path))
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._ensure_directories()

        # In-memory cache for current session
        self._session_traces: List[ExecutionTrace] = []

        # Event queue for real-time streaming
        self._event_queue: Queue[TraceEvent] = Queue()
        self._event_subscribers: List[Callable[[TraceEvent], None]] = []

        # Thread safety
        self._lock = threading.Lock()

    def _ensure_directories(self) -> None:
        """Create directory structure if it doesn't exist."""
        (self.base_path / "by_ptool").mkdir(parents=True, exist_ok=True)
        (self.base_path / "by_session").mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Trace Logging
    # =========================================================================

    def log_execution(self, trace: ExecutionTrace) -> None:
        """Log a ptool execution trace."""
        with self._lock:
            # Add to session cache
            self._session_traces.append(trace)

            # Append to ptool-specific file
            ptool_file = self.base_path / "by_ptool" / f"{trace.ptool_name}.jsonl"
            with open(ptool_file, "a") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")

            # Emit completion event
            self._emit_event(TraceEvent(
                event_type="ptool_complete",
                timestamp=datetime.now().isoformat(),
                data={
                    "trace_id": trace.trace_id,
                    "ptool_name": trace.ptool_name,
                    "success": trace.success,
                    "execution_time_ms": trace.execution_time_ms,
                    "output": trace.output,
                    "error": trace.error,
                },
            ))

        logger.debug(f"Logged trace: {trace.ptool_name} ({trace.trace_id})")

    def save_session(self) -> str:
        """Save current session traces to file."""
        session_file = self.base_path / "by_session" / f"session_{self.session_id}.json"

        session_data = {
            "session_id": self.session_id,
            "started_at": self._session_traces[0].timestamp if self._session_traces else None,
            "ended_at": datetime.now().isoformat(),
            "trace_count": len(self._session_traces),
            "traces": [t.to_dict() for t in self._session_traces],
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Saved session {self.session_id} with {len(self._session_traces)} traces")
        return str(session_file)

    # =========================================================================
    # Trace Retrieval
    # =========================================================================

    def get_traces(
        self,
        ptool_name: Optional[str] = None,
        limit: int = 100,
        success_only: bool = False,
    ) -> List[ExecutionTrace]:
        """Get traces, optionally filtered by ptool name."""
        traces = []

        if ptool_name:
            # Load from ptool-specific file
            ptool_file = self.base_path / "by_ptool" / f"{ptool_name}.jsonl"
            if ptool_file.exists():
                with open(ptool_file, "r") as f:
                    for line in f:
                        if line.strip():
                            trace = ExecutionTrace.from_dict(json.loads(line))
                            if success_only and not trace.success:
                                continue
                            traces.append(trace)
        else:
            # Load from all ptool files
            ptool_dir = self.base_path / "by_ptool"
            if ptool_dir.exists():
                for ptool_file in ptool_dir.glob("*.jsonl"):
                    with open(ptool_file, "r") as f:
                        for line in f:
                            if line.strip():
                                trace = ExecutionTrace.from_dict(json.loads(line))
                                if success_only and not trace.success:
                                    continue
                                traces.append(trace)

        # Sort by timestamp (newest first) and limit
        traces.sort(key=lambda t: t.timestamp, reverse=True)
        return traces[:limit]

    def get_session_traces(self) -> List[ExecutionTrace]:
        """Get all traces from current session."""
        return list(self._session_traces)

    def get_trace_count(self, ptool_name: str) -> int:
        """Get count of traces for a specific ptool."""
        ptool_file = self.base_path / "by_ptool" / f"{ptool_name}.jsonl"
        if not ptool_file.exists():
            return 0

        count = 0
        with open(ptool_file, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def get_all_ptools_with_traces(self) -> Dict[str, int]:
        """Get all ptools that have traces, with their counts."""
        ptool_dir = self.base_path / "by_ptool"
        result = {}

        if ptool_dir.exists():
            for ptool_file in ptool_dir.glob("*.jsonl"):
                ptool_name = ptool_file.stem
                result[ptool_name] = self.get_trace_count(ptool_name)

        return result

    # =========================================================================
    # Distillation Support
    # =========================================================================

    def get_distillation_candidates(self, min_traces: int = 10) -> List[str]:
        """Find ptools with enough successful traces to attempt distillation."""
        candidates = []
        ptool_counts = self.get_all_ptools_with_traces()

        for ptool_name, total_count in ptool_counts.items():
            if total_count < min_traces:
                continue

            # Check success rate
            traces = self.get_traces(ptool_name=ptool_name, limit=total_count)
            success_count = sum(1 for t in traces if t.success)
            success_rate = success_count / total_count if total_count > 0 else 0

            if success_rate >= 0.8:  # At least 80% success rate
                candidates.append(ptool_name)

        return candidates

    def get_input_output_pairs(
        self,
        ptool_name: str,
        limit: int = 1000,
    ) -> List[tuple]:
        """Get (input, output) pairs for distillation analysis."""
        traces = self.get_traces(ptool_name=ptool_name, success_only=True, limit=limit)
        return [(t.inputs, t.output) for t in traces]

    # =========================================================================
    # Real-time Events (for Dashboard)
    # =========================================================================

    def _emit_event(self, event: TraceEvent) -> None:
        """Emit an event to all subscribers."""
        self._event_queue.put(event)
        for subscriber in self._event_subscribers:
            try:
                subscriber(event)
            except Exception as e:
                logger.warning(f"Event subscriber error: {e}")

    def subscribe(self, callback: Callable[[TraceEvent], None]) -> None:
        """Subscribe to real-time trace events."""
        self._event_subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[TraceEvent], None]) -> None:
        """Unsubscribe from trace events."""
        if callback in self._event_subscribers:
            self._event_subscribers.remove(callback)

    def emit_ptool_start(
        self,
        ptool_name: str,
        args: Dict[str, Any],
        trace_id: str,
    ) -> None:
        """Emit event when a ptool execution starts."""
        self._emit_event(TraceEvent(
            event_type="ptool_start",
            timestamp=datetime.now().isoformat(),
            data={
                "trace_id": trace_id,
                "ptool_name": ptool_name,
                "args": args,
            },
        ))

    def emit_llm_request(
        self,
        trace_id: str,
        model: str,
        prompt: str,
    ) -> None:
        """Emit event when an LLM request is sent."""
        self._emit_event(TraceEvent(
            event_type="llm_request",
            timestamp=datetime.now().isoformat(),
            data={
                "trace_id": trace_id,
                "model": model,
                "prompt": prompt,
                "prompt_length": len(prompt),
            },
        ))

    def emit_llm_response(
        self,
        trace_id: str,
        response: str,
        latency_ms: float,
    ) -> None:
        """Emit event when an LLM response is received."""
        self._emit_event(TraceEvent(
            event_type="llm_response",
            timestamp=datetime.now().isoformat(),
            data={
                "trace_id": trace_id,
                "response": response,
                "response_length": len(response),
                "latency_ms": latency_ms,
            },
        ))

    def emit_error(
        self,
        trace_id: str,
        error: str,
        ptool_name: Optional[str] = None,
    ) -> None:
        """Emit error event."""
        self._emit_event(TraceEvent(
            event_type="error",
            timestamp=datetime.now().isoformat(),
            data={
                "trace_id": trace_id,
                "ptool_name": ptool_name,
                "error": error,
            },
        ))

    # =========================================================================
    # Utilities
    # =========================================================================

    def clear_ptool_traces(self, ptool_name: str) -> None:
        """Clear all traces for a specific ptool."""
        ptool_file = self.base_path / "by_ptool" / f"{ptool_name}.jsonl"
        if ptool_file.exists():
            ptool_file.unlink()
            logger.info(f"Cleared traces for {ptool_name}")

    def clear_all_traces(self) -> None:
        """Clear all traces. Use with caution!"""
        import shutil
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
        self._ensure_directories()
        self._session_traces.clear()
        logger.warning("Cleared all traces")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored traces."""
        ptool_counts = self.get_all_ptools_with_traces()
        total_traces = sum(ptool_counts.values())

        return {
            "total_traces": total_traces,
            "ptool_count": len(ptool_counts),
            "ptools": ptool_counts,
            "session_traces": len(self._session_traces),
            "storage_path": str(self.base_path),
        }


# Global trace store instance
_TRACE_STORE: Optional[TraceStore] = None


def get_trace_store() -> TraceStore:
    """Get the global trace store instance."""
    global _TRACE_STORE
    if _TRACE_STORE is None:
        _TRACE_STORE = TraceStore()
    return _TRACE_STORE


def set_trace_store(store: TraceStore) -> None:
    """Set the global trace store instance."""
    global _TRACE_STORE
    _TRACE_STORE = store
