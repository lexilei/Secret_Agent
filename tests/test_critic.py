"""Tests for the critic system."""

import pytest
import pandas as pd
from ptool_framework.critic import (
    CriticVerdict,
    CriticEvaluation,
    TraceCritic,
    evaluate_trace,
    quick_evaluate,
)
from ptool_framework.audit.base import StructuredAudit, AuditResult
from ptool_framework.traces import WorkflowTrace, TraceStep, StepStatus


@pytest.fixture
def good_trace():
    """Create a good trace for testing."""
    trace = WorkflowTrace(goal="Calculate BMI")
    trace.add_step("extract_values", {"text": "weight=70, height=1.75"})
    trace.add_step("calculate_bmi", {"weight": 70, "height": 1.75})

    # Mark as completed
    for step in trace.steps:
        step.status = StepStatus.COMPLETED
        step.result = {"value": 42}

    return trace


@pytest.fixture
def bad_trace():
    """Create a bad trace with failures."""
    trace = WorkflowTrace(goal="Calculate BMI")
    trace.add_step("extract_values", {"text": "weight=70"})
    trace.add_step("calculate_bmi", {"weight": 70})

    # Mark first as completed, second as failed
    trace.steps[0].status = StepStatus.COMPLETED
    trace.steps[0].result = {"weight": 70}
    trace.steps[1].status = StepStatus.FAILED
    trace.steps[1].error = "Missing height parameter"

    return trace


@pytest.fixture
def sample_audit():
    """Create a sample audit."""
    audit = StructuredAudit("test_audit", "Test")
    audit.add_rule(
        "has_steps",
        lambda df, _: len(df) > 0,
        "Must have steps"
    )
    audit.add_rule(
        "no_failures",
        lambda df, _: (df["status"] != "failed").all() if len(df) > 0 else True,
        "No failures allowed"
    )
    return audit


class TestCriticVerdict:
    """Tests for CriticVerdict enum."""

    def test_values(self):
        """Test enum values."""
        assert CriticVerdict.ACCEPT.value == "accept"
        assert CriticVerdict.REPAIR_NEEDED.value == "repair_needed"
        assert CriticVerdict.REJECT.value == "reject"


class TestCriticEvaluation:
    """Tests for CriticEvaluation dataclass."""

    def test_creation(self):
        """Test creating evaluation."""
        eval_result = CriticEvaluation(
            verdict=CriticVerdict.ACCEPT,
            confidence=0.9,
            trace_id="test123",
        )
        assert eval_result.verdict == CriticVerdict.ACCEPT
        assert eval_result.confidence == 0.9
        assert eval_result.is_acceptable

    def test_properties(self):
        """Test evaluation properties."""
        eval_result = CriticEvaluation(
            verdict=CriticVerdict.REPAIR_NEEDED,
            confidence=0.6,
        )
        assert eval_result.needs_repair
        assert not eval_result.is_acceptable
        assert not eval_result.should_reject

    def test_to_dict(self):
        """Test serialization."""
        eval_result = CriticEvaluation(
            verdict=CriticVerdict.ACCEPT,
            confidence=0.8,
            trace_id="test",
        )
        d = eval_result.to_dict()
        assert d["verdict"] == "accept"
        assert d["confidence"] == 0.8

    def test_summary(self):
        """Test summary generation."""
        eval_result = CriticEvaluation(
            verdict=CriticVerdict.ACCEPT,
            confidence=0.9,
            trace_id="test123",
        )
        summary = eval_result.summary()
        assert "ACCEPT" in summary
        assert "test123" in summary


class TestTraceCritic:
    """Tests for TraceCritic class."""

    def test_creation(self, sample_audit):
        """Test creating critic."""
        critic = TraceCritic(audits=[sample_audit])
        assert len(critic.audits) == 1

    def test_add_audit(self, sample_audit):
        """Test adding audits."""
        critic = TraceCritic()
        critic.add_audit(sample_audit)
        assert len(critic.audits) == 1

    def test_evaluate_good_trace(self, good_trace, sample_audit):
        """Test evaluating a good trace."""
        critic = TraceCritic(audits=[sample_audit])
        evaluation = critic.evaluate(good_trace, goal="Calculate BMI")

        assert evaluation.verdict == CriticVerdict.ACCEPT
        assert evaluation.confidence > 0.5
        assert len(evaluation.failed_steps) == 0

    def test_evaluate_bad_trace(self, bad_trace, sample_audit):
        """Test evaluating a bad trace."""
        critic = TraceCritic(audits=[sample_audit])
        evaluation = critic.evaluate(bad_trace, goal="Calculate BMI")

        assert evaluation.verdict in (CriticVerdict.REPAIR_NEEDED, CriticVerdict.REJECT)
        assert len(evaluation.failed_steps) > 0
        assert len(evaluation.repair_suggestions) > 0

    def test_evaluate_empty_trace(self, sample_audit):
        """Test evaluating empty trace."""
        trace = WorkflowTrace(goal="Empty")
        critic = TraceCritic(audits=[sample_audit])
        evaluation = critic.evaluate(trace, goal="Empty")

        assert evaluation.verdict in (CriticVerdict.REPAIR_NEEDED, CriticVerdict.REJECT)

    def test_thresholds(self, good_trace, sample_audit):
        """Test custom thresholds."""
        # Very strict thresholds
        critic = TraceCritic(
            audits=[sample_audit],
            accept_threshold=0.99,
            repair_threshold=0.8,
        )
        evaluation = critic.evaluate(good_trace)

        # May not reach very high threshold
        assert evaluation.verdict in (CriticVerdict.ACCEPT, CriticVerdict.REPAIR_NEEDED)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_evaluate_trace(self, good_trace, sample_audit):
        """Test evaluate_trace function."""
        evaluation = evaluate_trace(good_trace, audits=[sample_audit])
        assert isinstance(evaluation, CriticEvaluation)

    def test_quick_evaluate(self, good_trace):
        """Test quick_evaluate function."""
        verdict = quick_evaluate(good_trace)
        assert isinstance(verdict, CriticVerdict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
