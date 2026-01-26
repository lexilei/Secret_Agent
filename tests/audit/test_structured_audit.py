"""Tests for structured audit DSL and pre-built audits."""

import pytest
import pandas as pd
from ptool_framework.audit.structured_audit import (
    AuditDSL,
    audit_rule,
    DeclarativeAudit,
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    AllCompletedAudit,
    RequiredStepsAudit,
    StepOrderAudit,
    PatternAudit,
    PerformanceAudit,
    create_basic_audit,
    create_workflow_audit,
)
from ptool_framework.audit.base import AuditResult


@pytest.fixture
def sample_df():
    """Create a sample trace DataFrame."""
    return pd.DataFrame({
        "step_idx": [0, 1, 2],
        "fn_name": ["extract_data", "analyze", "format_output"],
        "input": ['{"text": "hello"}', '{"data": [1,2]}', '{"result": 42}'],
        "output": ['[1, 2, 3]', '42', '"formatted: 42"'],
        "status": ["completed", "completed", "completed"],
        "duration_ms": [100.0, 200.0, 50.0],
        "goal": ["Extract", "Analyze", "Format"],
        "error": [None, None, None],
    })


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return {
        "trace_id": "test123",
        "goal": "Process data",
        "success": True,
    }


class TestAuditDSL:
    """Tests for AuditDSL class."""

    def test_creation(self, sample_df, sample_metadata):
        """Test creating DSL instance."""
        dsl = AuditDSL(sample_df, sample_metadata)
        assert dsl.df is sample_df
        assert dsl.metadata == sample_metadata

    def test_steps_named(self, sample_df, sample_metadata):
        """Test steps_named query."""
        dsl = AuditDSL(sample_df, sample_metadata)

        # Partial match
        extract_steps = dsl.steps_named("extract")
        assert len(extract_steps) == 1

        # Exact match
        exact = dsl.steps_named("extract_data", exact=True)
        assert len(exact) == 1

        # No match
        no_match = dsl.steps_named("nonexistent")
        assert len(no_match) == 0

    def test_steps_with_status(self, sample_df, sample_metadata):
        """Test steps_with_status query."""
        dsl = AuditDSL(sample_df, sample_metadata)

        completed = dsl.steps_with_status("completed")
        assert len(completed) == 3

        failed = dsl.steps_with_status("failed")
        assert len(failed) == 0

    def test_get_step(self, sample_df, sample_metadata):
        """Test get_step."""
        dsl = AuditDSL(sample_df, sample_metadata)

        first = dsl.get_step(0)
        assert first["fn_name"] == "extract_data"

        last = dsl.get_step(-1)
        assert last["fn_name"] == "format_output"

        out_of_bounds = dsl.get_step(100)
        assert out_of_bounds is None

    def test_get_pattern(self, sample_df, sample_metadata):
        """Test get_pattern."""
        dsl = AuditDSL(sample_df, sample_metadata)
        pattern = dsl.get_pattern()
        assert pattern == ["extract_data", "analyze", "format_output"]

    def test_has_exactly(self, sample_df, sample_metadata):
        """Test has_exactly."""
        dsl = AuditDSL(sample_df, sample_metadata)

        assert dsl.has_exactly("extract", 1)
        assert dsl.has_exactly("analyze", 1)
        assert not dsl.has_exactly("extract", 2)

    def test_has_at_least(self, sample_df, sample_metadata):
        """Test has_at_least."""
        dsl = AuditDSL(sample_df, sample_metadata)

        assert dsl.has_at_least("extract", 1)
        assert dsl.has_at_least("extract", 0)
        assert not dsl.has_at_least("extract", 2)

    def test_has_at_most(self, sample_df, sample_metadata):
        """Test has_at_most."""
        dsl = AuditDSL(sample_df, sample_metadata)

        assert dsl.has_at_most("extract", 1)
        assert dsl.has_at_most("extract", 5)
        assert not dsl.has_at_most("extract", 0)

    def test_is_non_empty(self, sample_df, sample_metadata):
        """Test is_non_empty."""
        dsl = AuditDSL(sample_df, sample_metadata)
        assert dsl.is_non_empty()

        empty_dsl = AuditDSL(pd.DataFrame(), sample_metadata)
        assert not empty_dsl.is_non_empty()

    def test_comes_before(self, sample_df, sample_metadata):
        """Test comes_before."""
        dsl = AuditDSL(sample_df, sample_metadata)

        assert dsl.comes_before("extract", "analyze")
        assert dsl.comes_before("extract", "format")
        assert not dsl.comes_before("format", "extract")

    def test_comes_after(self, sample_df, sample_metadata):
        """Test comes_after."""
        dsl = AuditDSL(sample_df, sample_metadata)

        assert dsl.comes_after("analyze", "extract")
        assert dsl.comes_after("format", "extract")
        assert not dsl.comes_after("extract", "format")

    def test_pattern_exists(self, sample_df, sample_metadata):
        """Test pattern_exists."""
        dsl = AuditDSL(sample_df, sample_metadata)

        # Subsequence
        assert dsl.pattern_exists(["extract", "analyze"])
        assert dsl.pattern_exists(["extract", "format"])

        # Non-existent
        assert not dsl.pattern_exists(["format", "extract"])

    def test_pattern_exists_contiguous(self, sample_df, sample_metadata):
        """Test contiguous pattern matching."""
        dsl = AuditDSL(sample_df, sample_metadata)

        # Contiguous
        assert dsl.pattern_exists(["extract", "analyze"], contiguous=True)

        # Not contiguous
        assert not dsl.pattern_exists(["extract", "format"], contiguous=True)

    def test_all_completed(self, sample_df, sample_metadata):
        """Test all_completed."""
        dsl = AuditDSL(sample_df, sample_metadata)
        assert dsl.all_completed()

        # With a failed step
        df_with_fail = sample_df.copy()
        df_with_fail.loc[0, "status"] = "failed"
        dsl2 = AuditDSL(df_with_fail, sample_metadata)
        assert not dsl2.all_completed()

    def test_no_failures(self, sample_df, sample_metadata):
        """Test no_failures."""
        dsl = AuditDSL(sample_df, sample_metadata)
        assert dsl.no_failures()

    def test_success_rate(self, sample_df, sample_metadata):
        """Test success_rate."""
        dsl = AuditDSL(sample_df, sample_metadata)
        assert dsl.success_rate() == 1.0

        # With a failed step
        df_with_fail = sample_df.copy()
        df_with_fail.loc[0, "status"] = "failed"
        dsl2 = AuditDSL(df_with_fail, sample_metadata)
        assert dsl2.success_rate() == pytest.approx(2/3)


class TestDeclarativeAudit:
    """Tests for DeclarativeAudit with @audit_rule decorator."""

    def test_declarative_audit(self, sample_df, sample_metadata):
        """Test creating a declarative audit."""

        class TestAudit(DeclarativeAudit):
            def __init__(self):
                super().__init__("test_declarative", "Test")

            @audit_rule("non_empty", "Must have steps")
            def check_non_empty(self, dsl):
                return dsl.is_non_empty()

            @audit_rule("has_extract", "Must have extraction")
            def check_extract(self, dsl):
                return dsl.has_at_least("extract", 1)

        audit = TestAudit()
        report = audit.run(sample_df, sample_metadata)

        assert report.result == AuditResult.PASS
        assert "non_empty" in report.passed_checks
        assert "has_extract" in report.passed_checks


class TestPrebuiltAudits:
    """Tests for pre-built audit classes."""

    def test_non_empty_trace_audit(self, sample_df, sample_metadata):
        """Test NonEmptyTraceAudit."""
        audit = NonEmptyTraceAudit()
        report = audit.run(sample_df, sample_metadata)
        assert report.result == AuditResult.PASS

        # Empty trace
        report_empty = audit.run(pd.DataFrame(), sample_metadata)
        assert report_empty.result == AuditResult.FAIL

    def test_no_failed_steps_audit(self, sample_df, sample_metadata):
        """Test NoFailedStepsAudit."""
        audit = NoFailedStepsAudit()
        report = audit.run(sample_df, sample_metadata)
        assert report.result == AuditResult.PASS

        # With failed step
        df_fail = sample_df.copy()
        df_fail.loc[0, "status"] = "failed"
        report_fail = audit.run(df_fail, sample_metadata)
        assert report_fail.result == AuditResult.FAIL

    def test_all_completed_audit(self, sample_df, sample_metadata):
        """Test AllCompletedAudit."""
        audit = AllCompletedAudit()
        report = audit.run(sample_df, sample_metadata)
        assert report.result == AuditResult.PASS

    def test_required_steps_audit(self, sample_df, sample_metadata):
        """Test RequiredStepsAudit."""
        audit = RequiredStepsAudit(["extract", "analyze"])
        report = audit.run(sample_df, sample_metadata)
        assert report.result == AuditResult.PASS

        # Missing required step
        audit_missing = RequiredStepsAudit(["nonexistent"])
        report_missing = audit_missing.run(sample_df, sample_metadata)
        assert report_missing.result == AuditResult.FAIL

    def test_step_order_audit(self, sample_df, sample_metadata):
        """Test StepOrderAudit."""
        audit = StepOrderAudit([("extract", "analyze")])
        report = audit.run(sample_df, sample_metadata)
        assert report.result == AuditResult.PASS

        # Wrong order
        df_wrong = sample_df.iloc[::-1].reset_index(drop=True)
        df_wrong["step_idx"] = range(len(df_wrong))
        report_wrong = audit.run(df_wrong, sample_metadata)
        assert report_wrong.result == AuditResult.FAIL

    def test_pattern_audit(self, sample_df, sample_metadata):
        """Test PatternAudit."""
        audit = PatternAudit(
            required_patterns=[["extract", "analyze"]],
        )
        report = audit.run(sample_df, sample_metadata)
        assert report.result == AuditResult.PASS

    def test_performance_audit(self, sample_df, sample_metadata):
        """Test PerformanceAudit."""
        audit = PerformanceAudit(
            max_step_duration_ms=500,
            max_step_count=10,
        )
        report = audit.run(sample_df, sample_metadata)
        assert report.result == AuditResult.PASS

        # Too slow
        audit_strict = PerformanceAudit(max_step_duration_ms=50)
        report_slow = audit_strict.run(sample_df, sample_metadata)
        assert report_slow.result == AuditResult.FAIL


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_basic_audit(self, sample_df, sample_metadata):
        """Test create_basic_audit."""
        audit = create_basic_audit()
        report = audit.run(sample_df, sample_metadata)
        assert report.result == AuditResult.PASS

    def test_create_workflow_audit(self, sample_df, sample_metadata):
        """Test create_workflow_audit."""
        audit = create_workflow_audit(
            required_steps=["extract", "analyze"],
            step_order=[("extract", "analyze")],
        )
        report = audit.run(sample_df, sample_metadata)
        assert report.result == AuditResult.PASS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
