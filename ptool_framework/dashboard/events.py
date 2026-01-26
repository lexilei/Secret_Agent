"""
Event Emitter: Bridges trace events to dashboard WebSocket clients.

This module provides a centralized event system that:
1. Subscribes to TraceStore events
2. Maintains a list of WebSocket connections
3. Broadcasts events to all connected clients
4. Stores recent events for new clients joining
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
import threading
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class DashboardEvent:
    """Event structure for dashboard updates."""

    event_type: str
    timestamp: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_trace_event(cls, trace_event) -> DashboardEvent:
        """Convert a TraceEvent to DashboardEvent."""
        return cls(
            event_type=trace_event.event_type,
            timestamp=trace_event.timestamp,
            data=trace_event.data,
        )


class EventEmitter:
    """
    Central event emitter for dashboard updates.

    Maintains:
    - A set of WebSocket connections (clients)
    - A buffer of recent events (for new clients)
    - Subscription to TraceStore events
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._clients: Set[Any] = set()  # WebSocket connections
        self._event_history: deque[DashboardEvent] = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._subscribed = False

    def subscribe_to_trace_store(self) -> None:
        """Subscribe to TraceStore events."""
        if self._subscribed:
            return

        try:
            from ..trace_store import get_trace_store

            trace_store = get_trace_store()
            trace_store.subscribe(self._on_trace_event)
            self._subscribed = True
            logger.info("EventEmitter subscribed to TraceStore")
        except ImportError:
            logger.warning("TraceStore not available for subscription")

    def _on_trace_event(self, trace_event) -> None:
        """Handle incoming trace event."""
        event = DashboardEvent.from_trace_event(trace_event)
        self._add_event(event)

    def _add_event(self, event: DashboardEvent) -> None:
        """Add event to history and broadcast."""
        with self._lock:
            self._event_history.append(event)

        # Broadcast to all clients
        self._broadcast(event)

    def _broadcast(self, event: DashboardEvent) -> None:
        """Broadcast event to all connected clients."""
        if not self._clients:
            return

        message = event.to_json()

        # For sync clients (if any)
        disconnected = set()
        for client in list(self._clients):
            try:
                if hasattr(client, "send_text"):
                    # This is an async WebSocket - need to schedule
                    asyncio.create_task(self._async_send(client, message))
                elif callable(client):
                    # Callback-style client
                    client(event)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self._clients -= disconnected

    async def _async_send(self, websocket, message: str) -> None:
        """Send message to async WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.warning(f"WebSocket send error: {e}")
            self._clients.discard(websocket)

    def add_client(self, client: Any) -> None:
        """Add a new client (WebSocket or callback)."""
        with self._lock:
            self._clients.add(client)
            logger.info(f"Dashboard client connected. Total: {len(self._clients)}")

    def remove_client(self, client: Any) -> None:
        """Remove a client."""
        with self._lock:
            self._clients.discard(client)
            logger.info(f"Dashboard client disconnected. Total: {len(self._clients)}")

    def get_history(self, limit: int = 50) -> List[DashboardEvent]:
        """Get recent event history for new clients."""
        with self._lock:
            history = list(self._event_history)
        return history[-limit:]

    def emit(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a custom event."""
        event = DashboardEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data,
        )
        self._add_event(event)

    # Convenience methods for common events

    def emit_code_update(
        self,
        filename: str,
        code: str,
        highlighted_lines: List[int] = None,
    ) -> None:
        """Emit a code update event."""
        self.emit("code_update", {
            "filename": filename,
            "code": code,
            "highlighted_lines": highlighted_lines or [],
        })

    def emit_program_state(
        self,
        current_step: str,
        status: str,
        variables: Dict[str, Any] = None,
        duration_ms: float = 0,
    ) -> None:
        """Emit program state update."""
        self.emit("program_state", {
            "current_step": current_step,
            "status": status,
            "variables": variables or {},
            "duration_ms": duration_ms,
        })

    def emit_timeline_update(
        self,
        steps: List[Dict[str, Any]],
    ) -> None:
        """Emit timeline update with all steps."""
        self.emit("timeline_update", {
            "steps": steps,
        })

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)


# Global event emitter instance
_EVENT_EMITTER: Optional[EventEmitter] = None


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter instance."""
    global _EVENT_EMITTER
    if _EVENT_EMITTER is None:
        _EVENT_EMITTER = EventEmitter()
        _EVENT_EMITTER.subscribe_to_trace_store()
    return _EVENT_EMITTER


def set_event_emitter(emitter: EventEmitter) -> None:
    """Set the global event emitter instance."""
    global _EVENT_EMITTER
    _EVENT_EMITTER = emitter
