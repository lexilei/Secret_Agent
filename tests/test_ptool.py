"""
Tests for the ptool framework.

Run with: pytest tests/test_ptool.py -v
"""

import sys
from pathlib import Path
from typing import List, Literal

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptool_framework import ptool, PToolSpec, get_registry, TraceExecutor
from ptool_framework.ptool import example, PToolExample
from ptool_framework.traces import TraceStep, WorkflowTrace, step, trace
from ptool_framework.llm_backend import (
    MockLLMBackend,
    parse_structured_response,
    parse_freeform_response,
    _coerce_type,
)


class TestPToolDecorator:
    """Tests for the @ptool decorator."""

    def test_basic_ptool(self):
        """Test basic ptool registration."""
        @ptool()
        def test_func(x: str) -> str:
            """Test function."""
            ...

        registry = get_registry()
        assert "test_func" in registry
        spec = registry.get("test_func")
        assert spec.name == "test_func"
        assert spec.docstring == "Test function."
        assert "x" in spec.parameters

    def test_ptool_with_literal_return(self):
        """Test ptool with Literal return type."""
        @ptool()
        def classify(text: str) -> Literal["a", "b", "c"]:
            """Classify text."""
            ...

        registry = get_registry()
        spec = registry.get("classify")
        sig = spec.get_signature_str()
        assert "Literal" in sig

    def test_ptool_with_examples(self):
        """Test ptool with ICL examples."""
        @ptool(
            examples=[
                example({"x": "hello"}, "HELLO", "Uppercase the input"),
            ]
        )
        def upper(x: str) -> str:
            """Uppercase."""
            ...

        registry = get_registry()
        spec = registry.get("upper")
        assert len(spec.examples) == 1
        assert spec.examples[0].inputs == {"x": "hello"}


class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_structured_simple(self):
        """Test parsing simple structured response."""
        response = '{"result": "hello"}'
        result = parse_structured_response(response, str)
        assert result == "hello"

    def test_parse_structured_with_extra_text(self):
        """Test parsing with extra text around JSON."""
        response = 'Here is the answer:\n{"result": 42}\nDone.'
        result = parse_structured_response(response, int)
        assert result == 42

    def test_parse_structured_list(self):
        """Test parsing list result."""
        response = '{"result": ["a", "b", "c"]}'
        result = parse_structured_response(response, List[str])
        assert result == ["a", "b", "c"]

    def test_parse_freeform(self):
        """Test parsing freeform response."""
        response = "Let me think...\nANSWER: 42"
        result = parse_freeform_response(response, int)
        assert result == 42

    def test_coerce_literal(self):
        """Test coercing to Literal type."""
        result = _coerce_type("positive", Literal["positive", "negative"])
        assert result == "positive"

    def test_coerce_literal_case_insensitive(self):
        """Test case-insensitive Literal matching."""
        result = _coerce_type("POSITIVE", Literal["positive", "negative"])
        assert result == "positive"


class TestTraces:
    """Tests for workflow traces."""

    def test_create_trace(self):
        """Test creating a workflow trace."""
        t = WorkflowTrace(goal="Test goal")
        assert t.goal == "Test goal"
        assert len(t.steps) == 0

    def test_add_step(self):
        """Test adding steps to a trace."""
        t = WorkflowTrace(goal="Test")
        t.add_step("func1", {"x": 1})
        t.add_step("func2", {"y": "$step_0"})

        assert len(t.steps) == 2
        assert t.steps[0].ptool_name == "func1"
        assert t.steps[1].args["y"] == "$step_0"

    def test_trace_helper(self):
        """Test trace() helper function."""
        t = trace(
            "Test goal",
            step("func1", {"x": 1}),
            step("func2", {"y": 2}),
        )

        assert t.goal == "Test goal"
        assert len(t.steps) == 2

    def test_trace_serialization(self):
        """Test trace JSON serialization."""
        t = trace("Test", step("func", {"x": 1}))
        json_str = t.to_json()
        t2 = WorkflowTrace.from_json(json_str)

        assert t2.goal == t.goal
        assert len(t2.steps) == len(t.steps)


class TestMockBackend:
    """Tests for the mock LLM backend."""

    def test_mock_response(self):
        """Test mock backend returns configured responses."""
        mock = MockLLMBackend()
        mock.add_response("test_key", '{"result": "mock_value"}')

        response = mock("prompt with test_key", "model")
        assert "mock_value" in response

    def test_mock_logs_calls(self):
        """Test mock backend logs all calls."""
        mock = MockLLMBackend()
        mock("prompt1", "model1")
        mock("prompt2", "model2")

        assert len(mock.call_log) == 2
        assert mock.call_log[0]["prompt"] == "prompt1"


class TestExecutor:
    """Tests for the trace executor."""

    def test_execute_with_mock(self):
        """Test executing a trace with mock backend."""
        # Register a test ptool
        @ptool()
        def add_one(x: int) -> int:
            """Add one to x."""
            ...

        # Create mock
        mock = MockLLMBackend()
        mock.add_response("add_one", '{"result": 2}')

        # Create and execute trace
        t = trace("Add one to 1", step("add_one", {"x": 1}))

        executor = TraceExecutor(llm_backend=mock)
        result = executor.execute(t)

        assert result.success
        assert result.completed_steps == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
