"""
SSRM-style Audit System for ptool_framework.

This package provides Semi-Structured Reasoning Model (SSRM) audits as described in
"Semi-structured LLM Reasoners Can Be Rigorously Audited" (Cohen et al.).

The audit system enables:
- **Structured audits**: Python unit tests using assertions on trace DataFrames
- **Typicality audits**: Probabilistic models over reasoning patterns

Subpackages:
    - typicality: Probabilistic models (Unigram, Bigram, Trigram, HMM)
    - domains: Domain-specific audits (e.g., MedCalc)

Core Classes:
    - AuditResult: Enum for audit outcomes (PASS, FAIL, ABSTAIN)
    - AuditViolation: Details about a specific audit failure
    - AuditReport: Complete report from running audits on a trace
    - BaseAudit: Abstract base class for all audit types
    - StructuredAudit: Base for audits using DataFrame assertions
    - TypicalityAudit: Base for audits using probabilistic models
    - BaseTypicalityModel: Abstract base for probabilistic models
    - AuditRegistry: Registry for managing available audits
    - TraceDataFrameConverter: Convert traces to DataFrames

Example:
    >>> from ptool_framework.audit import (
    ...     StructuredAudit,
    ...     AuditRunner,
    ...     convert_trace,
    ... )
    >>>
    >>> # Create a structured audit
    >>> audit = StructuredAudit("my_audit", "Check trace validity")
    >>> audit.add_rule(
    ...     "has_steps",
    ...     lambda df, _: len(df) > 0,
    ...     "Trace must have at least one step"
    ... )
    >>>
    >>> # Convert trace and run audit
    >>> df, metadata = convert_trace(my_trace)
    >>> report = audit.run(df, metadata)
    >>> print(report.is_pass)
"""

from .base import (
    # Enums and data classes
    AuditResult,
    AuditViolation,
    AuditReport,
    # Base classes
    BaseAudit,
    StructuredAudit,
    BaseTypicalityModel,
    TypicalityAudit,
    # Registry
    AuditRegistry,
    get_audit_registry,
    set_audit_registry,
    register_audit,
)

from .dataframe_converter import (
    TraceDataFrameConverter,
    convert_trace,
    batch_convert_traces,
    traces_to_combined_dataframe,
)

from .structured_audit import (
    # DSL
    AuditDSL,
    # Decorators and declarative
    audit_rule,
    AuditRuleInfo,
    DeclarativeAudit,
    # Pre-built audits
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    AllCompletedAudit,
    RequiredStepsAudit,
    StepOrderAudit,
    PatternAudit,
    PerformanceAudit,
    # Factory functions
    create_basic_audit,
    create_workflow_audit,
    register_prebuilt_audits,
)

from .runner import (
    # Result classes
    BatchAuditResult,
    # Runner
    AuditRunner,
    TypicalityTrainer,
    # Convenience functions
    audit_trace,
    batch_audit,
    train_typicality_model,
)

from .llm_audits import (
    # LLM-based audits
    LLMAudit,
    AuditGenerator,
    GeneratedRule,
    GeneratedAudit,
    create_reasoning_audit,
)

__all__ = [
    # Enums and data classes
    "AuditResult",
    "AuditViolation",
    "AuditReport",
    # Base classes
    "BaseAudit",
    "StructuredAudit",
    "BaseTypicalityModel",
    "TypicalityAudit",
    # Registry
    "AuditRegistry",
    "get_audit_registry",
    "set_audit_registry",
    "register_audit",
    # DataFrame conversion
    "TraceDataFrameConverter",
    "convert_trace",
    "batch_convert_traces",
    "traces_to_combined_dataframe",
    # Structured audit DSL
    "AuditDSL",
    "audit_rule",
    "AuditRuleInfo",
    "DeclarativeAudit",
    # Pre-built audits
    "NonEmptyTraceAudit",
    "NoFailedStepsAudit",
    "AllCompletedAudit",
    "RequiredStepsAudit",
    "StepOrderAudit",
    "PatternAudit",
    "PerformanceAudit",
    # Factory functions
    "create_basic_audit",
    "create_workflow_audit",
    "register_prebuilt_audits",
    # Runner
    "BatchAuditResult",
    "AuditRunner",
    "TypicalityTrainer",
    "audit_trace",
    "batch_audit",
    "train_typicality_model",
    # LLM-based audits
    "LLMAudit",
    "AuditGenerator",
    "GeneratedRule",
    "GeneratedAudit",
    "create_reasoning_audit",
]
