"""
Dashboard: Web-based observability interface for ptool execution.

Provides real-time monitoring of:
- Code execution with highlighted recent edits
- LLM request/response logging
- Execution traces
- Timeline visualization
"""

from .events import EventEmitter, get_event_emitter

__all__ = ["EventEmitter", "get_event_emitter"]
