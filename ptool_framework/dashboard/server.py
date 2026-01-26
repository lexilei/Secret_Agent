"""
Dashboard Server: FastAPI backend with WebSocket support.

Provides:
1. REST API for traces, code, and analysis
2. WebSocket for real-time event streaming
3. Static file serving for the dashboard UI
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

# FastAPI imports - these are optional
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not installed. Dashboard server unavailable.")

from .events import get_event_emitter, DashboardEvent


# ============================================================================
# Request/Response Models
# ============================================================================

if HAS_FASTAPI:

    class RunProgramRequest(BaseModel):
        """Request to run a program."""
        program_path: str
        input_data: str
        collect_traces: bool = True

    class RefactorRequest(BaseModel):
        """Request to refactor a program."""
        source_path: str
        mode: str  # "distill" or "expand"
        output_path: Optional[str] = None


# ============================================================================
# Dashboard Application
# ============================================================================

def create_app() -> "FastAPI":
    """Create the FastAPI application."""
    if not HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="ptool Observatory",
        description="Observability dashboard for ptool framework",
        version="0.2.0",
    )

    # Get static files directory
    static_dir = Path(__file__).parent / "static"

    # Mount static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ========================================================================
    # Routes
    # ========================================================================

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        """Serve the main dashboard page."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return HTMLResponse(content=_get_fallback_html(), status_code=200)

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "service": "ptool-observatory"}

    @app.get("/api/traces")
    async def get_traces(
        ptool: Optional[str] = None,
        limit: int = 100,
        success_only: bool = False,
    ):
        """Get execution traces."""
        try:
            from ..trace_store import get_trace_store

            trace_store = get_trace_store()
            traces = trace_store.get_traces(
                ptool_name=ptool,
                limit=limit,
                success_only=success_only,
            )
            return {
                "traces": [t.to_dict() for t in traces],
                "count": len(traces),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/traces/stats")
    async def get_trace_stats():
        """Get trace statistics."""
        try:
            from ..trace_store import get_trace_store

            trace_store = get_trace_store()
            return trace_store.get_stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/ptools")
    async def get_ptools():
        """Get all registered ptools."""
        try:
            from ..ptool import get_registry

            registry = get_registry()
            ptools = []
            for spec in registry.list_all():
                ptools.append({
                    "name": spec.name,
                    "signature": spec.get_signature_str(),
                    "docstring": spec.docstring[:200] + "..." if len(spec.docstring) > 200 else spec.docstring,
                    "model": spec.model,
                    "output_mode": spec.output_mode,
                })
            return {"ptools": ptools, "count": len(ptools)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/code/{filename:path}")
    async def get_code(filename: str):
        """Get source code of a file."""
        try:
            # Security: only allow reading from certain directories
            filepath = Path(filename)
            if not filepath.exists():
                raise HTTPException(status_code=404, detail="File not found")

            with open(filepath, "r") as f:
                code = f.read()

            return {
                "filename": filename,
                "code": code,
                "line_count": len(code.splitlines()),
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/distillation/candidates")
    async def get_distillation_candidates(min_traces: int = 10):
        """Get ptools that are candidates for distillation."""
        try:
            from ..trace_store import get_trace_store

            trace_store = get_trace_store()
            candidates = trace_store.get_distillation_candidates(min_traces)

            results = []
            for ptool_name in candidates:
                traces = trace_store.get_traces(ptool_name=ptool_name, limit=1000)
                success_count = sum(1 for t in traces if t.success)
                results.append({
                    "ptool_name": ptool_name,
                    "trace_count": len(traces),
                    "success_rate": success_count / len(traces) if traces else 0,
                    "status": "DISTILLABLE" if success_count / len(traces) > 0.9 else "PARTIALLY_DISTILLABLE",
                })

            return {"candidates": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/fallbacks")
    async def get_fallback_analysis():
        """Get analysis of distilled function fallbacks."""
        try:
            from ..distilled import analyze_fallbacks, get_fallback_log

            return {
                "analysis": analyze_fallbacks(),
                "recent_fallbacks": [
                    {
                        "ptool_name": e.ptool_name,
                        "reason": e.reason,
                        "timestamp": e.timestamp,
                    }
                    for e in get_fallback_log()[-20:]
                ],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/history")
    async def get_event_history(limit: int = 50):
        """Get recent event history."""
        emitter = get_event_emitter()
        history = emitter.get_history(limit)
        return {
            "events": [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp,
                    "data": e.data,
                }
                for e in history
            ],
            "count": len(history),
        }

    # ========================================================================
    # WebSocket for Real-time Updates
    # ========================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time event streaming."""
        await websocket.accept()
        logger.info("WebSocket client connected")

        emitter = get_event_emitter()
        emitter.add_client(websocket)

        # Send recent history to new client
        history = emitter.get_history(30)
        for event in history:
            try:
                await websocket.send_text(event.to_json())
            except Exception:
                break

        try:
            # Keep connection alive and handle incoming messages
            while True:
                data = await websocket.receive_text()
                # Handle client messages (e.g., requests for specific data)
                try:
                    message = json.loads(data)
                    await _handle_client_message(websocket, message)
                except json.JSONDecodeError:
                    pass

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.warning(f"WebSocket error: {e}")
        finally:
            emitter.remove_client(websocket)

    async def _handle_client_message(websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming WebSocket messages from clients."""
        msg_type = message.get("type")

        if msg_type == "ping":
            await websocket.send_text(json.dumps({"type": "pong"}))

        elif msg_type == "get_traces":
            from ..trace_store import get_trace_store
            trace_store = get_trace_store()
            traces = trace_store.get_traces(limit=message.get("limit", 50))
            await websocket.send_text(json.dumps({
                "type": "traces",
                "data": [t.to_dict() for t in traces],
            }))

        elif msg_type == "get_ptools":
            from ..ptool import get_registry
            registry = get_registry()
            ptools = [
                {"name": s.name, "signature": s.get_signature_str()}
                for s in registry.list_all()
            ]
            await websocket.send_text(json.dumps({
                "type": "ptools",
                "data": ptools,
            }))

    return app


def _get_fallback_html() -> str:
    """Return fallback HTML if static file not found."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ptool Observatory</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #1a1a2e;
                color: #eee;
                padding: 40px;
                text-align: center;
            }
            h1 { color: #00d4ff; }
            .status { padding: 20px; background: #16213e; border-radius: 8px; margin: 20px auto; max-width: 600px; }
            pre { text-align: left; background: #0f0f23; padding: 15px; border-radius: 4px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>ptool Observatory</h1>
        <div class="status">
            <p>Dashboard server is running!</p>
            <p>Static files not found. Place index.html in the static directory.</p>
            <p>API endpoints are available at /api/*</p>
        </div>
        <div class="status">
            <h3>Quick API Test</h3>
            <pre id="api-result">Loading...</pre>
        </div>
        <script>
            fetch('/api/health')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('api-result').textContent = JSON.stringify(data, null, 2);
                })
                .catch(e => {
                    document.getElementById('api-result').textContent = 'Error: ' + e.message;
                });
        </script>
    </body>
    </html>
    """


# ============================================================================
# Server Runner
# ============================================================================

def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    reload: bool = False,
) -> None:
    """
    Run the dashboard server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload (for development)
    """
    if not HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed. Install with: pip install fastapi uvicorn"
        )

    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "uvicorn is not installed. Install with: pip install uvicorn"
        )

    print(f"\n  ptool Observatory")
    print(f"  ─────────────────")
    print(f"  Dashboard: http://{host}:{port}")
    print(f"  API docs:  http://{host}:{port}/docs")
    print(f"  WebSocket: ws://{host}:{port}/ws")
    print()

    uvicorn.run(
        "ptool_framework.dashboard.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


if __name__ == "__main__":
    run_server()
