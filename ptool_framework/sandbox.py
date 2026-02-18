"""
Sandbox: Secure Python execution for LLM-generated code.

Two interfaces:
    1. @sandbox decorator — sandbox any function
    2. PythonSandbox class — sandbox code strings, expressions, and files

Two backends:
    - "smolagents": AST-walking interpreter (pip install smolagents)
    - "restricted": exec() with restricted builtins (no extra deps)
    - "auto": tries smolagents first, falls back to restricted

Example (decorator):
    @sandbox(allowed_imports=["math"])
    def compute(x: float) -> float:
        import math
        return math.sqrt(x)

    compute(4.0)  # runs sandboxed, returns 2.0

Example (class):
    sb = PythonSandbox(allowed_imports=["math"])
    sb.safe_exec("y = x ** 2", state={"x": 5})
    sb.safe_eval("x > 0", context={"x": 5})

Example (module-level):
    from ptool_framework.sandbox import safe_exec, safe_eval
    safe_exec("y = 2 + 2", state={})
    safe_eval("len(items) > 0", context={"items": [1, 2]})
"""

from __future__ import annotations

import functools
import inspect
import re
import signal
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

T = TypeVar("T")

# Try to import smolagents
try:
    from smolagents.local_python_executor import LocalPythonExecutor
    from smolagents.local_python_executor import InterpreterError as _SmolInterpreterError
    HAS_SMOLAGENTS = True
except ImportError:
    HAS_SMOLAGENTS = False
    _SmolInterpreterError = None


# ============================================================================
# Exceptions
# ============================================================================

class SandboxError(Exception):
    """Raised when sandboxed code fails a security check or exceeds limits."""
    pass


class SandboxTimeoutError(SandboxError):
    """Raised when sandboxed code exceeds the time limit."""
    pass


class SandboxImportError(SandboxError):
    """Raised when sandboxed code attempts an unauthorized import."""
    pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SandboxConfig:
    """Configuration for a PythonSandbox instance.

    Attributes:
        allowed_imports: Modules the sandboxed code may import.
        max_ops: Maximum AST operations (smolagents backend only).
        timeout_seconds: Wall-clock limit per execution.
        backend: "auto", "smolagents", or "restricted".
    """
    allowed_imports: List[str] = field(default_factory=list)
    max_ops: int = 1_000_000
    timeout_seconds: int = 30
    backend: str = "auto"


# ============================================================================
# Restricted-exec backend helpers
# ============================================================================

# Safe builtins for the restricted backend — matches l2_coc.py's SAFE_BUILTINS
# plus a few extras useful for general sandboxing.
_RESTRICTED_BUILTINS: Dict[str, Any] = {
    # Constants
    "True": True,
    "False": False,
    "None": None,
    # Types
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "bytes": bytes,
    "bytearray": bytearray,
    "complex": complex,
    # Math / comparison
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "pow": pow,
    "divmod": divmod,
    # Iterables
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "any": any,
    "all": all,
    "next": next,
    "iter": iter,
    # Type introspection
    "isinstance": isinstance,
    "issubclass": issubclass,
    "type": type,
    "hasattr": hasattr,
    "getattr": getattr,
    "callable": callable,
    # String / repr
    "repr": repr,
    "chr": chr,
    "ord": ord,
    "hash": hash,
    "id": id,
    "format": format,
    # Printing (no-op by default to capture output)
    "print": lambda *a, **kw: None,
}

# Modules that are NEVER importable, regardless of allowed_imports.
_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "ftplib", "smtplib",
    "multiprocessing", "threading", "ctypes",
    "io", "pty", "signal", "code", "codeop",
    "importlib", "builtins", "__builtin__",
})


def _make_restricted_import(allowed: frozenset):
    """Return a restricted __import__ that only allows listed modules."""

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split(".")[0]
        if top_level in _BLOCKED_MODULES:
            raise SandboxImportError(
                f"Import of '{name}' is blocked. "
                f"Blocked modules: {', '.join(sorted(_BLOCKED_MODULES))}"
            )
        if top_level not in allowed:
            raise SandboxImportError(
                f"Import of '{name}' is not allowed. "
                f"Allowed imports: {', '.join(sorted(allowed)) or '(none)'}"
            )
        return __builtins__["__import__"](name, globals, locals, fromlist, level)

    return _restricted_import


class _TimeoutHandler:
    """Context manager that raises SandboxTimeoutError after *seconds*."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self._old_handler = None

    def _handle(self, signum, frame):
        raise SandboxTimeoutError(
            f"Execution exceeded {self.seconds}s timeout"
        )

    def __enter__(self):
        if self.seconds and hasattr(signal, "SIGALRM"):
            self._old_handler = signal.signal(signal.SIGALRM, self._handle)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        if self.seconds and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        return False


# ============================================================================
# PythonSandbox
# ============================================================================

class PythonSandbox:
    """Sandboxed Python execution engine.

    Two backends:
        - **smolagents**: AST-walking interpreter with operation counting,
          import control, and dangerous-function blocking.  Requires
          ``pip install smolagents``.
        - **restricted**: ``exec()`` with stripped ``__builtins__``, a custom
          ``__import__``, and ``signal.SIGALRM`` timeout (Unix only).
          No extra dependencies.

    ``backend="auto"`` (the default) picks smolagents when available.

    Example::

        sb = PythonSandbox(allowed_imports=["math"], timeout_seconds=5)
        result = sb.safe_exec("import math; y = math.sqrt(x)", state={"x": 9})
        assert result["y"] == 3.0
    """

    def __init__(
        self,
        allowed_imports: Optional[List[str]] = None,
        max_ops: int = 1_000_000,
        timeout_seconds: int = 30,
        backend: str = "auto",
        config: Optional[SandboxConfig] = None,
    ):
        if config is not None:
            allowed_imports = config.allowed_imports
            max_ops = config.max_ops
            timeout_seconds = config.timeout_seconds
            backend = config.backend

        self.allowed_imports = list(allowed_imports or [])
        self.max_ops = max_ops
        self.timeout_seconds = timeout_seconds

        # Resolve backend
        if backend == "auto":
            self.backend = "smolagents" if HAS_SMOLAGENTS else "restricted"
        else:
            if backend == "smolagents" and not HAS_SMOLAGENTS:
                raise SandboxError(
                    "smolagents backend requested but not installed. "
                    "Install with: pip install smolagents"
                )
            self.backend = backend

        logger.debug(
            f"PythonSandbox initialized: backend={self.backend}, "
            f"imports={self.allowed_imports}"
        )

    # ------------------------------------------------------------------
    # Core execution: safe_exec
    # ------------------------------------------------------------------

    def safe_exec(
        self,
        code: str,
        state: Optional[Dict[str, Any]] = None,
        allowed_imports: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute *code* inside the sandbox and return the resulting state.

        Args:
            code: Python source code to execute.
            state: Variables to inject before execution.
            allowed_imports: Override instance-level allowed imports for this
                call.  Merged with the instance's list.

        Returns:
            Dict of all variables in scope after execution (private names
            starting with ``_`` are excluded).

        Raises:
            SandboxError: On security violation or execution failure.
            SandboxTimeoutError: If execution exceeds the time limit.
            SandboxImportError: If an unauthorized import is attempted.
        """
        state = dict(state or {})
        imports = set(self.allowed_imports)
        if allowed_imports:
            imports.update(allowed_imports)

        if self.backend == "smolagents":
            return self._exec_smolagents(code, state, imports)
        else:
            return self._exec_restricted(code, state, imports)

    # ------------------------------------------------------------------
    # Internal: raw execution (returns unfiltered state)
    # ------------------------------------------------------------------

    def _exec_smolagents_raw(
        self,
        code: str,
        state: Dict[str, Any],
        imports: set,
    ) -> Dict[str, Any]:
        """Execute using smolagents' LocalPythonExecutor. Returns raw state."""
        try:
            executor = LocalPythonExecutor(
                additional_authorized_imports=list(imports),
                timeout_seconds=self.timeout_seconds or None,
            )
            if state:
                executor.send_variables(state)

            executor(code)
            return dict(executor.state)

        except _SmolInterpreterError as exc:
            msg = str(exc)
            if "import" in msg.lower():
                raise SandboxImportError(msg) from exc
            if "timeout" in msg.lower() or "time" in msg.lower():
                raise SandboxTimeoutError(msg) from exc
            raise SandboxError(msg) from exc

    def _exec_restricted_raw(
        self,
        code: str,
        state: Dict[str, Any],
        imports: set,
    ) -> Dict[str, Any]:
        """Execute using restricted exec(). Returns raw locals dict."""
        builtins = dict(_RESTRICTED_BUILTINS)
        builtins["__import__"] = _make_restricted_import(frozenset(imports))

        exec_globals: Dict[str, Any] = {"__builtins__": builtins}
        exec_locals: Dict[str, Any] = dict(state)

        try:
            with _TimeoutHandler(self.timeout_seconds):
                exec(code, exec_globals, exec_locals)  # noqa: S102
        except SandboxError:
            raise
        except TimeoutError as exc:
            raise SandboxTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise SandboxError(f"Execution failed: {type(exc).__name__}: {exc}") from exc

        return dict(exec_locals)

    # ------------------------------------------------------------------
    # Internal: filtered execution (public-facing state)
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_state(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Remove private/internal names from execution state."""
        return {
            k: v for k, v in raw.items()
            if not k.startswith("_") and k != "__name__"
        }

    def _exec_smolagents(
        self,
        code: str,
        state: Dict[str, Any],
        imports: set,
    ) -> Dict[str, Any]:
        """Execute using smolagents and return filtered state."""
        return self._filter_state(
            self._exec_smolagents_raw(code, state, imports)
        )

    def _exec_restricted(
        self,
        code: str,
        state: Dict[str, Any],
        imports: set,
    ) -> Dict[str, Any]:
        """Execute using restricted exec and return filtered state."""
        return self._filter_state(
            self._exec_restricted_raw(code, state, imports)
        )

    # ------------------------------------------------------------------
    # Core execution: safe_eval
    # ------------------------------------------------------------------

    def safe_eval(
        self,
        expr: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Evaluate a single expression inside the sandbox.

        Args:
            expr: A Python expression (not a statement).
            context: Variables available during evaluation.

        Returns:
            The result of evaluating *expr*.

        Raises:
            SandboxError: On security violation or evaluation failure.
        """
        result_key = "_sandbox_eval_result"
        wrapper_code = f"{result_key} = ({expr})"
        # Use the raw execution methods directly to avoid filtering
        state = dict(context or {})
        imports = set(self.allowed_imports)
        if self.backend == "smolagents":
            raw = self._exec_smolagents_raw(wrapper_code, state, imports)
        else:
            raw = self._exec_restricted_raw(wrapper_code, state, imports)
        return raw.get(result_key)

    # ------------------------------------------------------------------
    # File execution
    # ------------------------------------------------------------------

    def exec_file(
        self,
        path: str,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Execute a Python file in a subprocess with resource limits.

        This uses process-level isolation (subprocess) rather than
        AST-walking, since full programs may need the complete Python
        runtime.  The subprocess inherits a *stripped* environment
        (no API keys leaked).

        Args:
            path: Path to the Python file.
            args: Command-line arguments passed to the script.
            timeout: Override timeout for this call.

        Returns:
            ``subprocess.CompletedProcess`` with stdout/stderr.

        Raises:
            SandboxTimeoutError: If execution exceeds the time limit.
            SandboxError: If the file doesn't exist or can't be executed.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise SandboxError(f"File not found: {path}")

        timeout = timeout or self.timeout_seconds
        cmd = [sys.executable, str(file_path)] + (args or [])

        # Build a minimal environment — strip API keys and dangerous vars
        safe_env = {
            k: v for k, v in __import__("os").environ.items()
            if not any(secret in k.upper() for secret in (
                "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
            ))
        }

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=safe_env,
            )
            return result
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(
                f"File execution exceeded {timeout}s timeout"
            ) from exc
        except Exception as exc:
            raise SandboxError(f"File execution failed: {exc}") from exc


# ============================================================================
# @sandbox decorator
# ============================================================================

def sandbox(
    allowed_imports: Optional[List[str]] = None,
    max_ops: int = 1_000_000,
    timeout_seconds: int = 30,
    backend: str = "auto",
) -> Callable:
    """Decorator that executes a function's body inside a sandbox.

    The decorated function's source is extracted, and on each call the
    function is *defined and invoked* inside a ``PythonSandbox``.  This
    means the function body cannot use ``open()``, ``os.system()``, or
    any import not listed in *allowed_imports*.

    Args:
        allowed_imports: Modules the function body may import.
        max_ops: Max AST operations (smolagents backend).
        timeout_seconds: Wall-clock timeout per call.
        backend: "auto", "smolagents", or "restricted".

    Example::

        @sandbox(allowed_imports=["math"])
        def hypotenuse(a: float, b: float) -> float:
            import math
            return math.sqrt(a**2 + b**2)

        hypotenuse(3.0, 4.0)  # → 5.0  (executed in sandbox)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Extract and cache the function source at decoration time
        raw_source = textwrap.dedent(inspect.getsource(func))
        func_name = func.__name__

        # Strip the @sandbox(...) decorator line(s) from the source.
        # Handles both single-line and multi-line decorator syntax.
        clean_source = _strip_decorator(raw_source, "sandbox")

        # Build the sandbox instance (shared across calls)
        sb = PythonSandbox(
            allowed_imports=allowed_imports,
            max_ops=max_ops,
            timeout_seconds=timeout_seconds,
            backend=backend,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to parameter names
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Build the code to execute inside the sandbox:
            #   <function definition>
            #   __sandbox_result__ = func_name(arg1=val1, ...)
            result_key = "_sandbox_call_result"
            call_args = ", ".join(
                f"{name}=_sandbox_arg_{name}"
                for name in bound.arguments
            )
            call_code = (
                f"{clean_source}\n"
                f"{result_key} = {func_name}({call_args})\n"
            )

            # Inject argument values into the sandbox state
            state = {
                f"_sandbox_arg_{name}": value
                for name, value in bound.arguments.items()
            }

            # Use raw execution to preserve internal result key
            imports = set(sb.allowed_imports)
            if sb.backend == "smolagents":
                raw = sb._exec_smolagents_raw(call_code, state, imports)
            else:
                raw = sb._exec_restricted_raw(call_code, state, imports)
            return raw.get(result_key)

        wrapper._sandbox = sb  # expose for introspection
        wrapper._is_sandboxed = True
        return wrapper

    return decorator


def _strip_decorator(source: str, decorator_name: str) -> str:
    """Remove ``@decorator_name(...)`` lines from function source.

    Handles single-line ``@sandbox(...)`` and multi-line decorator calls
    that span several lines with parentheses.
    """
    lines = source.split("\n")
    result = []
    skip = False
    paren_depth = 0

    for line in lines:
        stripped = line.strip()

        # Start of decorator to remove
        if stripped.startswith(f"@{decorator_name}"):
            skip = True
            paren_depth += stripped.count("(") - stripped.count(")")
            if paren_depth <= 0:
                skip = False
                paren_depth = 0
            continue

        # Continuation of multi-line decorator
        if skip:
            paren_depth += stripped.count("(") - stripped.count(")")
            if paren_depth <= 0:
                skip = False
                paren_depth = 0
            continue

        result.append(line)

    return "\n".join(result)


# ============================================================================
# Global singleton and convenience functions
# ============================================================================

_global_sandbox: Optional[PythonSandbox] = None


def get_sandbox(**kwargs) -> PythonSandbox:
    """Get or create the global ``PythonSandbox`` singleton.

    Keyword arguments are forwarded to ``PythonSandbox()`` only when
    creating the instance for the first time.
    """
    global _global_sandbox
    if _global_sandbox is None:
        _global_sandbox = PythonSandbox(**kwargs)
    return _global_sandbox


def configure_sandbox(**kwargs) -> None:
    """(Re-)configure the global sandbox.

    Replaces the singleton so subsequent ``safe_exec`` / ``safe_eval``
    calls use the new settings.

    Example::

        configure_sandbox(allowed_imports=["math", "re"], timeout_seconds=10)
    """
    global _global_sandbox
    _global_sandbox = PythonSandbox(**kwargs)
    logger.info(f"Global sandbox configured: backend={_global_sandbox.backend}")


def safe_exec(
    code: str,
    state: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Execute *code* in the global sandbox.  See ``PythonSandbox.safe_exec``."""
    return get_sandbox().safe_exec(code, state=state, **kwargs)


def safe_eval(
    expr: str,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Evaluate *expr* in the global sandbox.  See ``PythonSandbox.safe_eval``."""
    return get_sandbox().safe_eval(expr, context=context)
