"""Tests for the repair agent system."""

import pytest
from ptool_framework.repair import (
    RepairActionType,
    RepairAction,
    RepairResult,
    RepairAgent,
    repair_trace,
    auto_repair,
)
from ptool_framework.critic import TraceCritic, CriticVerdict, CriticEvaluation
from ptool_framework.traces import WorkflowTrace, TraceStep, StepStatus
from ptool_framework.audit.base import StructuredAudit, AuditResult


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


@pytest.fixture
def sample_critic(sample_audit):
    """Create a sample critic."""
    return TraceCritic(audits=[sample_audit])


class TestRepairActionType:
    """Tests for RepairActionType enum."""

    def test_values(self):
        """Test enum values."""
        assert RepairActionType.REGENERATE_STEP.value == "regenerate_step"
        assert RepairActionType.ADD_STEP.value == "add_step"
        assert RepairActionType.REMOVE_STEP.value == "remove_step"
        assert RepairActionType.MODIFY_ARGS.value == "modify_args"
        assert RepairActionType.REORDER_STEPS.value == "reorder_steps"


class TestRepairAction:
    """Tests for RepairAction dataclass."""

    def test_creation(self):
        """Test creating repair action."""
        action = RepairAction(
            action_type=RepairActionType.REGENERATE_STEP,
            step_index=2,
            reason="Step failed during execution"
        )
        assert action.action_type == RepairActionType.REGENERATE_STEP
        assert action.step_index == 2
        assert action.reason == "Step failed during execution"

    def test_to_dict(self):
        """Test serialization."""
        action = RepairAction(
            action_type=RepairActionType.ADD_STEP,
            step_index=1,
            details={"ptool_name": "validate"},
            reason="Missing validation step"
        )
        d = action.to_dict()
        assert d["action_type"] == "add_step"
        assert d["step_index"] == 1
        assert d["details"]["ptool_name"] == "validate"


class TestRepairResult:
    """Tests for RepairResult dataclass."""

    def test_creation_success(self, good_trace):
        """Test creating successful repair result."""
        result = RepairResult(
            success=True,
            repaired_trace=good_trace,
            iterations=2,
        )
        assert result.success
        assert result.repaired_trace is good_trace
        assert result.iterations == 2

    def test_creation_failure(self):
        """Test creating failed repair result."""
        result = RepairResult(
            success=False,
            error="Max repair attempts reached",
            iterations=3,
        )
        assert not result.success
        assert result.error == "Max repair attempts reached"

    def test_to_dict(self, good_trace):
        """Test serialization."""
        result = RepairResult(
            success=True,
            repaired_trace=good_trace,
            actions_taken=[
                RepairAction(
                    action_type=RepairActionType.REGENERATE_STEP,
                    step_index=0,
                    reason="Test"
                )
            ],
            iterations=1,
            original_verdict=CriticVerdict.REPAIR_NEEDED,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["iterations"] == 1
        assert d["original_verdict"] == "repair_needed"
        assert len(d["actions_taken"]) == 1

    def test_summary(self):
        """Test summary generation."""
        result = RepairResult(
            success=True,
            iterations=2,
            actions_taken=[
                RepairAction(RepairActionType.REGENERATE_STEP, 0, {}, "fix")
            ],
        )
        summary = result.summary()
        assert "SUCCESS" in summary
        assert "2" in summary


class TestRepairAgent:
    """Tests for RepairAgent class."""

    def test_creation(self, sample_critic):
        """Test creating repair agent."""
        agent = RepairAgent(
            critic=sample_critic,
            max_attempts=3,
            model="deepseek-v3"
        )
        assert agent.critic is sample_critic
        assert agent.max_attempts == 3
        assert agent.model == "deepseek-v3"

    def test_creation_default(self):
        """Test creating with defaults."""
        agent = RepairAgent()
        assert agent.critic is not None
        assert agent.max_attempts == 3

    def test_repair_already_good(self, good_trace, sample_critic):
        """Test repairing an already good trace."""
        agent = RepairAgent(critic=sample_critic, max_attempts=3)
        evaluation = sample_critic.evaluate(good_trace, goal="Calculate BMI")

        # Should already be acceptable
        assert evaluation.verdict == CriticVerdict.ACCEPT

        result = agent.repair(good_trace, evaluation, "Calculate BMI")
        assert result.success
        assert result.iterations == 0  # No repairs needed

    def test_repair_bad_trace(self, bad_trace, sample_critic):
        """Test repairing a bad trace."""
        agent = RepairAgent(critic=sample_critic, max_attempts=3)
        evaluation = sample_critic.evaluate(bad_trace, goal="Calculate BMI")

        # Should need repair or be rejected
        assert evaluation.verdict in (CriticVerdict.REPAIR_NEEDED, CriticVerdict.REJECT)

        result = agent.repair(bad_trace, evaluation, "Calculate BMI")
        # Result depends on actual repair (may succeed or fail)
        assert result.iterations <= 3
        assert result.original_verdict in (CriticVerdict.REPAIR_NEEDED, CriticVerdict.REJECT)

    def test_repair_empty_trace(self, sample_critic):
        """Test repairing an empty trace."""
        trace = WorkflowTrace(goal="Test")
        agent = RepairAgent(critic=sample_critic, max_attempts=3)
        evaluation = sample_critic.evaluate(trace, goal="Test")

        result = agent.repair(trace, evaluation, "Test")
        assert result.iterations <= 3

    def test_to_workflow_trace(self, good_trace):
        """Test trace conversion."""
        agent = RepairAgent()

        # WorkflowTrace passes through
        converted = agent._to_workflow_trace(good_trace)
        assert converted is good_trace

        # List of dicts gets converted
        steps = [
            {"fn_name": "step1", "args": {"x": 1}},
            {"fn_name": "step2", "args": {"y": 2}},
        ]
        converted = agent._to_workflow_trace(steps)
        assert isinstance(converted, WorkflowTrace)
        assert len(converted.steps) == 2

    def test_generate_basic_suggestions(self, bad_trace):
        """Test basic suggestion generation."""
        agent = RepairAgent()
        evaluation = CriticEvaluation(
            verdict=CriticVerdict.REPAIR_NEEDED,
            confidence=0.5,
        )

        suggestions = agent._generate_basic_suggestions(bad_trace, evaluation)
        # Should suggest regenerating failed step
        assert len(suggestions) > 0
        assert any(s["action"] == "regenerate_step" for s in suggestions)

    def test_remove_step(self, good_trace):
        """Test step removal."""
        agent = RepairAgent()
        original_len = len(good_trace.steps)

        action = agent._remove_step(good_trace, step_index=0, reason="Test removal")
        assert action is not None
        assert action.action_type == RepairActionType.REMOVE_STEP
        assert len(good_trace.steps) == original_len - 1

    def test_remove_step_invalid_index(self, good_trace):
        """Test step removal with invalid index."""
        agent = RepairAgent()

        action = agent._remove_step(good_trace, step_index=100, reason="Invalid")
        assert action is None

    def test_modify_args(self, good_trace):
        """Test argument modification."""
        agent = RepairAgent()
        original_args = good_trace.steps[0].args.copy()

        suggestion = {
            "step_index": 0,
            "details": {"new_args": {"extra_key": "extra_value"}},
            "reason": "Add extra arg"
        }

        action = agent._modify_args(good_trace, suggestion)
        assert action is not None
        assert action.action_type == RepairActionType.MODIFY_ARGS
        assert "extra_key" in good_trace.steps[0].args


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_repair_trace(self, good_trace, sample_audit):
        """Test repair_trace function."""
        critic = TraceCritic(audits=[sample_audit])
        result = repair_trace(good_trace, "Calculate BMI", critic=critic)
        assert isinstance(result, RepairResult)

    def test_repair_trace_default_critic(self, good_trace):
        """Test repair_trace with default critic."""
        result = repair_trace(good_trace, "Calculate BMI")
        assert isinstance(result, RepairResult)

    def test_auto_repair(self, good_trace):
        """Test auto_repair function."""
        result = auto_repair(good_trace, "Calculate BMI")
        # Good trace should pass or be repaired
        # Result is trace or None
        if result is not None:
            assert isinstance(result, WorkflowTrace)


class TestRepairIntegration:
    """Integration tests for repair system."""

    def test_repair_flow(self, sample_audit):
        """Test complete repair flow."""
        # Create trace with mixed results
        trace = WorkflowTrace(goal="Test")
        trace.add_step("step1", {"input": "data"})
        trace.add_step("step2", {"input": "more"})
        trace.steps[0].status = StepStatus.COMPLETED
        trace.steps[0].result = {"output": "result1"}
        trace.steps[1].status = StepStatus.COMPLETED
        trace.steps[1].result = {"output": "result2"}

        critic = TraceCritic(audits=[sample_audit])
        agent = RepairAgent(critic=critic, max_attempts=2)

        evaluation = critic.evaluate(trace, goal="Test")
        result = agent.repair(trace, evaluation, "Test")

        # Should complete without error
        assert result.iterations <= 2
        assert result.original_verdict in (CriticVerdict.ACCEPT, CriticVerdict.REPAIR_NEEDED, CriticVerdict.REJECT)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
