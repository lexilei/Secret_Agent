"""Tests for audit base classes."""

import pytest
import pandas as pd
from ptool_framework.audit.base import (
    AuditResult,
    AuditViolation,
    AuditReport,
    StructuredAudit,
    TypicalityAudit,
    AuditRegistry,
    get_audit_registry,
    register_audit,
)


class TestAuditResult:
    """Tests for AuditResult enum."""

    def test_values(self):
        """Test enum values."""
        assert AuditResult.PASS.value == "pass"
        assert AuditResult.FAIL.value == "fail"
        assert AuditResult.ABSTAIN.value == "abstain"


class TestAuditViolation:
    """Tests for AuditViolation dataclass."""

    def test_creation(self):
        """Test creating a violation."""
        violation = AuditViolation(
            audit_name="test_audit",
            rule_name="test_rule",
            message="Test message",
            severity="error",
        )
        assert violation.audit_name == "test_audit"
        assert violation.rule_name == "test_rule"
        assert violation.severity == "error"

    def test_to_dict(self):
        """Test serialization."""
        violation = AuditViolation(
            audit_name="test",
            rule_name="rule",
            message="msg",
        )
        d = violation.to_dict()
        assert d["audit_name"] == "test"
        assert d["rule_name"] == "rule"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "audit_name": "test",
            "rule_name": "rule",
            "message": "msg",
            "severity": "warning",
        }
        violation = AuditViolation.from_dict(data)
        assert violation.audit_name == "test"
        assert violation.severity == "warning"


class TestAuditReport:
    """Tests for AuditReport dataclass."""

    def test_creation(self):
        """Test creating a report."""
        report = AuditReport(
            trace_id="test123",
            result=AuditResult.PASS,
        )
        assert report.trace_id == "test123"
        assert report.is_pass
        assert not report.is_fail

    def test_with_violations(self):
        """Test report with violations."""
        violations = [
            AuditViolation("a1", "r1", "msg1", "error"),
            AuditViolation("a2", "r2", "msg2", "warning"),
        ]
        report = AuditReport(
            trace_id="test",
            result=AuditResult.FAIL,
            violations=violations,
        )
        assert report.is_fail
        assert report.error_count == 1
        assert report.warning_count == 1

    def test_serialization(self):
        """Test to_dict and from_dict."""
        report = AuditReport(
            trace_id="test",
            result=AuditResult.PASS,
            passed_checks=["check1", "check2"],
        )
        d = report.to_dict()
        restored = AuditReport.from_dict(d)
        assert restored.trace_id == "test"
        assert restored.result == AuditResult.PASS
        assert len(restored.passed_checks) == 2


class TestStructuredAudit:
    """Tests for StructuredAudit class."""

    def test_creation(self):
        """Test creating a structured audit."""
        audit = StructuredAudit("test_audit", "Test description")
        assert audit.name == "test_audit"
        assert audit.description == "Test description"

    def test_add_rule(self):
        """Test adding rules."""
        audit = StructuredAudit("test")
        audit.add_rule(
            "non_empty",
            lambda df, _: len(df) > 0,
            "DataFrame must not be empty"
        )
        assert "non_empty" in audit.rules

    def test_method_chaining(self):
        """Test method chaining for add_rule."""
        audit = (
            StructuredAudit("test")
            .add_rule("r1", lambda df, _: True, "m1")
            .add_rule("r2", lambda df, _: True, "m2")
        )
        assert len(audit.rules) == 2

    def test_run_passing(self):
        """Test running audit that passes."""
        audit = StructuredAudit("test")
        audit.add_rule(
            "has_rows",
            lambda df, _: len(df) > 0,
            "Must have rows"
        )

        df = pd.DataFrame({"fn_name": ["test"], "status": ["completed"]})
        report = audit.run(df, {"trace_id": "t1"})

        assert report.result == AuditResult.PASS
        assert "has_rows" in report.passed_checks

    def test_run_failing(self):
        """Test running audit that fails."""
        audit = StructuredAudit("test")
        audit.add_rule(
            "has_rows",
            lambda df, _: len(df) > 0,
            "Must have rows"
        )

        df = pd.DataFrame()
        report = audit.run(df, {"trace_id": "t1"})

        assert report.result == AuditResult.FAIL
        assert len(report.violations) == 1

    def test_rule_exception(self):
        """Test handling of rule exceptions."""
        audit = StructuredAudit("test")
        audit.add_rule(
            "bad_rule",
            lambda df, _: 1 / 0,  # Will raise ZeroDivisionError
            "Error message"
        )

        df = pd.DataFrame({"fn_name": ["test"]})
        report = audit.run(df, {"trace_id": "t1"})

        assert report.result == AuditResult.FAIL
        assert "exception" in report.violations[0].message.lower()


class TestAuditRegistry:
    """Tests for AuditRegistry class."""

    def test_creation(self):
        """Test creating registry."""
        registry = AuditRegistry()
        assert len(registry) == 0

    def test_register(self):
        """Test registering audits."""
        registry = AuditRegistry()
        audit = StructuredAudit("test", "Test audit")
        registry.register(audit)

        assert "test" in registry
        assert len(registry) == 1

    def test_register_with_domain(self):
        """Test registering with domain."""
        registry = AuditRegistry()
        audit = StructuredAudit("medcalc_test", "MedCalc audit")
        registry.register(audit, domain="medcalc")

        assert "medcalc" in registry.list_domains()
        medcalc_audits = registry.list_by_domain("medcalc")
        assert len(medcalc_audits) == 1

    def test_get(self):
        """Test getting audit by name."""
        registry = AuditRegistry()
        audit = StructuredAudit("test", "")
        registry.register(audit)

        retrieved = registry.get("test")
        assert retrieved is audit
        assert registry.get("nonexistent") is None

    def test_unregister(self):
        """Test unregistering audits."""
        registry = AuditRegistry()
        audit = StructuredAudit("test", "")
        registry.register(audit, domain="test_domain")

        removed = registry.unregister("test")
        assert removed is audit
        assert "test" not in registry

    def test_list_all(self):
        """Test listing all audits."""
        registry = AuditRegistry()
        a1 = StructuredAudit("a1", "")
        a2 = StructuredAudit("a2", "")
        registry.register(a1)
        registry.register(a2)

        all_audits = registry.list_all()
        assert len(all_audits) == 2


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_audit_registry(self):
        """Test getting global registry."""
        registry = get_audit_registry()
        assert isinstance(registry, AuditRegistry)

    def test_register_audit(self):
        """Test global register function."""
        audit = StructuredAudit("global_test_" + str(id(self)), "")
        register_audit(audit)

        registry = get_audit_registry()
        assert audit.name in registry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
