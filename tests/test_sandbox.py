"""
Tests for ptool_framework.sandbox.

Tests run against the "restricted" backend by default (no extra deps).
If smolagents is installed, tests also verify the smolagents backend.
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from ptool_framework.sandbox import (
    PythonSandbox,
    SandboxConfig,
    SandboxError,
    SandboxImportError,
    SandboxTimeoutError,
    configure_sandbox,
    get_sandbox,
    safe_eval,
    safe_exec,
    sandbox,
    HAS_SMOLAGENTS,
    _global_sandbox,
)
import ptool_framework.sandbox as sandbox_module


# ============================================================================
# Helpers
# ============================================================================

BACKENDS = ["restricted"]
if HAS_SMOLAGENTS:
    BACKENDS.append("smolagents")


@pytest.fixture(autouse=True)
def reset_global_sandbox():
    """Reset the global singleton between tests."""
    sandbox_module._global_sandbox = None
    yield
    sandbox_module._global_sandbox = None


# ============================================================================
# safe_exec tests
# ============================================================================

class TestSafeExec:
    """Tests for PythonSandbox.safe_exec and the module-level safe_exec."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_basic_assignment(self, backend):
        sb = PythonSandbox(backend=backend)
        result = sb.safe_exec("x = 2 + 3")
        assert result["x"] == 5

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_injected_state(self, backend):
        sb = PythonSandbox(backend=backend)
        result = sb.safe_exec("y = x * 2", state={"x": 7})
        assert result["y"] == 14

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_multiline_code(self, backend):
        sb = PythonSandbox(backend=backend)
        code = textwrap.dedent("""\
            items = [1, 2, 3, 4, 5]
            total = sum(items)
            avg = total / len(items)
        """)
        result = sb.safe_exec(code)
        assert result["total"] == 15
        assert result["avg"] == 3.0

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_conditionals_and_loops(self, backend):
        sb = PythonSandbox(backend=backend)
        code = textwrap.dedent("""\
            result = []
            for i in range(5):
                if i % 2 == 0:
                    result.append(i)
        """)
        result = sb.safe_exec(code)
        assert result["result"] == [0, 2, 4]

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_function_definition_and_call(self, backend):
        sb = PythonSandbox(backend=backend)
        code = textwrap.dedent("""\
            def square(n):
                return n * n
            answer = square(6)
        """)
        result = sb.safe_exec(code)
        assert result["answer"] == 36

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_private_names_excluded(self, backend):
        sb = PythonSandbox(backend=backend)
        result = sb.safe_exec("x = 1\n_private = 2")
        assert "x" in result
        assert "_private" not in result


# ============================================================================
# safe_eval tests
# ============================================================================

class TestSafeEval:
    """Tests for PythonSandbox.safe_eval and the module-level safe_eval."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_arithmetic(self, backend):
        sb = PythonSandbox(backend=backend)
        assert sb.safe_eval("2 + 3") == 5

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_with_context(self, backend):
        sb = PythonSandbox(backend=backend)
        assert sb.safe_eval("x > 0", context={"x": 5}) is True
        assert sb.safe_eval("x > 0", context={"x": -1}) is False

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_builtin_functions(self, backend):
        sb = PythonSandbox(backend=backend)
        assert sb.safe_eval("len(items)", context={"items": [1, 2, 3]}) == 3
        assert sb.safe_eval("max(a, b)", context={"a": 3, "b": 7}) == 7

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_boolean_expression(self, backend):
        sb = PythonSandbox(backend=backend)
        ctx = {"state": {"count": 3, "done": False}}
        assert sb.safe_eval("state['count'] > 0", context=ctx) is True


# ============================================================================
# Import control tests
# ============================================================================

class TestImportControl:
    """Tests for import whitelisting and blocking."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_unauthorized_import_blocked(self, backend):
        sb = PythonSandbox(backend=backend)
        with pytest.raises((SandboxImportError, SandboxError)):
            sb.safe_exec("import os")

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_subprocess_blocked(self, backend):
        sb = PythonSandbox(backend=backend)
        with pytest.raises((SandboxImportError, SandboxError)):
            sb.safe_exec("import subprocess")

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_allowed_import_works(self, backend):
        sb = PythonSandbox(allowed_imports=["math"], backend=backend)
        result = sb.safe_exec("import math\ny = math.sqrt(16)")
        assert result["y"] == 4.0

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_per_call_allowed_imports(self, backend):
        sb = PythonSandbox(backend=backend)
        # Not allowed by default
        with pytest.raises((SandboxImportError, SandboxError)):
            sb.safe_exec("import json")
        # Allowed via per-call override
        result = sb.safe_exec(
            "import json\nx = json.dumps({'a': 1})",
            allowed_imports=["json"],
        )
        assert result["x"] == '{"a": 1}'

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_blocked_module_never_importable(self, backend):
        """Even if listed in allowed_imports, os/subprocess/etc. stay blocked."""
        sb = PythonSandbox(allowed_imports=["os"], backend=backend)
        with pytest.raises((SandboxImportError, SandboxError)):
            sb.safe_exec("import os")


# ============================================================================
# Dangerous builtins tests
# ============================================================================

class TestDangerousBuiltins:
    """Test that dangerous built-in functions are blocked."""

    def test_exec_blocked_restricted(self):
        sb = PythonSandbox(backend="restricted")
        with pytest.raises(SandboxError):
            sb.safe_exec("exec('x = 1')")

    def test_eval_blocked_restricted(self):
        sb = PythonSandbox(backend="restricted")
        with pytest.raises(SandboxError):
            sb.safe_exec("y = eval('2 + 2')")

    def test_open_blocked_restricted(self):
        sb = PythonSandbox(backend="restricted")
        with pytest.raises(SandboxError):
            sb.safe_exec("f = open('/etc/passwd')")


# ============================================================================
# Timeout tests
# ============================================================================

class TestTimeout:
    """Test that infinite loops / long computations are killed."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_infinite_loop_killed(self, backend):
        sb = PythonSandbox(timeout_seconds=2, backend=backend)
        with pytest.raises((SandboxTimeoutError, SandboxError)):
            sb.safe_exec("while True: pass")


# ============================================================================
# @sandbox decorator tests
# ============================================================================

class TestSandboxDecorator:
    """Tests for the @sandbox decorator."""

    def test_basic_function(self):
        @sandbox()
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_with_allowed_imports(self):
        @sandbox(allowed_imports=["math"])
        def hypotenuse(a: float, b: float) -> float:
            import math
            return math.sqrt(a**2 + b**2)

        assert hypotenuse(3.0, 4.0) == 5.0

    def test_blocks_unsafe_import(self):
        @sandbox()
        def evil():
            import os
            return os.getcwd()

        with pytest.raises((SandboxImportError, SandboxError)):
            evil()

    def test_function_with_defaults(self):
        @sandbox()
        def greet(name: str, greeting: str = "Hello") -> str:
            return greeting + " " + name

        assert greet("World") == "Hello World"
        assert greet("World", greeting="Hi") == "Hi World"

    def test_complex_function(self):
        @sandbox(allowed_imports=["re"])
        def extract_numbers(text: str) -> list:
            import re
            return [int(x) for x in re.findall(r'\d+', text)]

        assert extract_numbers("I have 3 cats and 5 dogs") == [3, 5]

    def test_is_sandboxed_attribute(self):
        @sandbox()
        def noop():
            pass

        assert hasattr(noop, "_is_sandboxed")
        assert noop._is_sandboxed is True

    def test_preserves_function_name(self):
        @sandbox()
        def my_func():
            return 42

        assert my_func.__name__ == "my_func"


# ============================================================================
# exec_file tests
# ============================================================================

class TestExecFile:
    """Tests for PythonSandbox.exec_file."""

    def test_basic_file_execution(self, tmp_path):
        script = tmp_path / "hello.py"
        script.write_text("print('hello sandbox')")

        sb = PythonSandbox()
        result = sb.exec_file(str(script))
        assert result.returncode == 0
        assert "hello sandbox" in result.stdout

    def test_file_not_found(self):
        sb = PythonSandbox()
        with pytest.raises(SandboxError, match="File not found"):
            sb.exec_file("/nonexistent/path.py")

    def test_file_timeout(self, tmp_path):
        script = tmp_path / "slow.py"
        script.write_text("import time; time.sleep(60)")

        sb = PythonSandbox(timeout_seconds=2)
        with pytest.raises(SandboxTimeoutError):
            sb.exec_file(str(script))

    def test_api_keys_stripped(self, tmp_path):
        script = tmp_path / "env_check.py"
        script.write_text(textwrap.dedent("""\
            import os
            keys = [k for k in os.environ if 'API_KEY' in k or 'SECRET' in k]
            if keys:
                print('LEAKED: ' + ','.join(keys))
            else:
                print('CLEAN')
        """))

        # Temporarily set a fake API key
        import os
        os.environ["TEST_API_KEY"] = "secret123"
        try:
            sb = PythonSandbox()
            result = sb.exec_file(str(script))
            assert "CLEAN" in result.stdout
            assert "LEAKED" not in result.stdout
        finally:
            del os.environ["TEST_API_KEY"]


# ============================================================================
# Singleton / configure tests
# ============================================================================

class TestGlobalSandbox:
    """Tests for get_sandbox, configure_sandbox, module-level helpers."""

    def test_singleton_returns_same_instance(self):
        sb1 = get_sandbox()
        sb2 = get_sandbox()
        assert sb1 is sb2

    def test_configure_replaces_singleton(self):
        sb1 = get_sandbox()
        configure_sandbox(allowed_imports=["math"], timeout_seconds=10)
        sb2 = get_sandbox()
        assert sb1 is not sb2
        assert "math" in sb2.allowed_imports

    def test_module_safe_exec(self):
        result = safe_exec("x = 42")
        assert result["x"] == 42

    def test_module_safe_eval(self):
        assert safe_eval("2 ** 10") == 1024

    def test_config_dataclass(self):
        config = SandboxConfig(
            allowed_imports=["json", "re"],
            timeout_seconds=5,
            backend="restricted",
        )
        sb = PythonSandbox(config=config)
        assert sb.allowed_imports == ["json", "re"]
        assert sb.timeout_seconds == 5
        assert sb.backend == "restricted"


# ============================================================================
# Backend-specific: smolagents
# ============================================================================

@pytest.mark.skipif(not HAS_SMOLAGENTS, reason="smolagents not installed")
class TestSmolagentsBackend:
    """Tests that only run when smolagents is available."""

    def test_explicit_backend(self):
        sb = PythonSandbox(backend="smolagents")
        assert sb.backend == "smolagents"
        result = sb.safe_exec("x = 1 + 1")
        assert result["x"] == 2

    def test_smolagents_import_error_class(self):
        sb = PythonSandbox(backend="smolagents")
        with pytest.raises(SandboxImportError):
            sb.safe_exec("import os")

    def test_auto_selects_smolagents(self):
        sb = PythonSandbox(backend="auto")
        assert sb.backend == "smolagents"


class TestRestrictedBackend:
    """Tests specifically for the restricted backend."""

    def test_explicit_backend(self):
        sb = PythonSandbox(backend="restricted")
        assert sb.backend == "restricted"
        result = sb.safe_exec("x = 1 + 1")
        assert result["x"] == 2

    def test_unavailable_smolagents_raises(self):
        if HAS_SMOLAGENTS:
            pytest.skip("smolagents is installed")
        sb = PythonSandbox(backend="auto")
        assert sb.backend == "restricted"
