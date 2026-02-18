"""
ptool_framework: Python-Calling-LLMs Agent Framework

A framework where Python programs call LLMs (ptools) rather than LLMs calling tools.
This enables predictable, testable, verifiable agent workflows.

Three modes of use:
- Mode 1: @ptool decorator - Add LLM functions to existing code
- Mode 2: ProgramGenerator - Generate programs from task descriptions
- Mode 3: ReActAgent - Goal-driven reasoning with ptool execution

Core concepts:
- ptool: A "pseudo-tool" - a function stub executed by LLM prompting
- distilled: A pure Python function with automatic LLM fallback
- WorkflowTrace: A sequence of ptool calls to achieve a goal
- ReActAgent: Reasoning + Acting loop for dynamic ptool execution
- TraceStore: Persistent storage for execution traces (for distillation)
- BehaviorDistiller: Converts ptools to pure Python via trace analysis
"""

from .ptool import ptool, PToolSpec, PToolRegistry, get_registry
from .traces import TraceStep, WorkflowTrace, ExecutionResult
from .executor import TraceExecutor
from .trace_store import (
    TraceStore,
    ExecutionTrace,
    TraceEvent,
    get_trace_store,
    set_trace_store,
)
from .distilled import (
    distilled,
    distilled_ptool,
    DistillationFallback,
    get_distilled_stats,
    is_distilled,
    get_fallback_log,
    analyze_fallbacks,
)
from .llm_backend import enable_tracing
from .distiller import (
    BehaviorDistiller,
    DistillationResult,
    DistillationAnalysis,
    distill_ptool,
    analyze_ptool,
    distill_all_candidates,
)
from .refactorer import (
    CodeRefactorer,
    RefactorResult,
    InteractiveRefactorer,
    refactor_program,
    analyze_program,
    distill_program,
    expand_program,
)
from .program_generator import (
    ProgramGenerator,
    GeneratedProgram,
    generate_program,
    generate_from_template,
)
from .react import (
    # Data structures
    Thought,
    Action,
    Observation,
    ReActStep,
    ReActTrajectory,
    ReActResult,
    # Agent
    ReActAgent,
    # Storage
    ReActStore,
    get_react_store,
    set_react_store,
    # Convenience functions
    react,
    react_and_execute,
    # Exceptions
    ActionParseError,
    ReActError,
)

# New: Audit System
from .audit import (
    # Base classes
    AuditResult,
    AuditViolation,
    AuditReport,
    BaseAudit,
    StructuredAudit,
    TypicalityAudit,
    BaseTypicalityModel,
    AuditRegistry,
    get_audit_registry,
    # DSL and decorators
    AuditDSL,
    audit_rule,
    DeclarativeAudit,
    # Pre-built audits
    NonEmptyTraceAudit,
    NoFailedStepsAudit,
    RequiredStepsAudit,
    # Runner
    AuditRunner,
    TypicalityTrainer,
    audit_trace,
    batch_audit,
    # Typicality models
)

# New: Critic System
from .critic import (
    CriticVerdict,
    CriticEvaluation,
    TraceCritic,
    evaluate_trace,
    quick_evaluate,
)

# New: Repair Agent
from .repair import (
    RepairActionType,
    RepairAction,
    RepairResult,
    RepairAgent,
    repair_trace,
    auto_repair,
)

# New: Model Selector
from .model_selector import (
    TaskComplexity,
    ModelConfig,
    SelectionResult,
    FallbackChain,
    ExperienceStore,
    ModelSelector,
    select_model,
    get_model_for_complexity,
    # LLMS.json utilities
    load_llms_config,
    get_default_model,
    get_enabled_models,
    set_model_enabled,
    add_model_tag,
    remove_model_tag,
)

# New: LLM Config Builder
from .llm_config_builder import (
    LLMConfigBuilder,
    validate_llms_json,
    validate_llms_file,
    build_llms_config,
    update_llms_config,
    enable_models_by_tag,
    disable_all_models,
    enable_only_tag,
)

# New: Multi-Agent Orchestration (L4)
from .orchestrator import (
    # Data structures
    ExecutionMode,
    AgentSpec,
    RoutingDecision,
    OrchestrationStep,
    OrchestrationTrace,
    OrchestrationResult,
    # Routers
    BaseRouter,
    RuleBasedRouter,
    LLMRouter,
    ExperienceBasedRouter,
    HybridRouter,
    # Main orchestrator
    AgentOrchestrator,
    # Storage
    OrchestrationStore,
    get_orchestration_store,
    set_orchestration_store,
    # Convenience
    orchestrate,
)

# New: L2 StateGraph
from .stategraph import (
    StateGraph,
    StateNode,
    StateEdge,
    StateCondition,
    StateGraphResult,
    StateGraphError,
    run_stategraph,
)

# New: Python Sandbox
from .sandbox import (
    PythonSandbox,
    SandboxConfig,
    SandboxError,
    SandboxImportError,
    SandboxTimeoutError,
    sandbox,
    safe_exec,
    safe_eval,
    get_sandbox,
    configure_sandbox,
)

# New: Self-Improving Agents (L5)
from .self_improving import (
    # Data structures
    PatternType,
    LearnedPattern,
    ICLExample,
    LearningEvent,
    SelfImprovementMetrics,
    # Pattern extraction
    PatternExtractor,
    # Memory
    PatternMemory,
    get_pattern_memory,
    # Main agent
    SelfImprovingAgent,
    # Convenience
    self_improving_react,
)

__all__ = [
    # Core ptool
    "ptool",
    "PToolSpec",
    "PToolRegistry",
    "get_registry",
    # Traces
    "TraceStep",
    "WorkflowTrace",
    "ExecutionResult",
    "TraceExecutor",
    # Trace storage
    "TraceStore",
    "ExecutionTrace",
    "TraceEvent",
    "get_trace_store",
    "set_trace_store",
    "enable_tracing",
    # Distilled decorator
    "distilled",
    "distilled_ptool",
    "DistillationFallback",
    "get_distilled_stats",
    "is_distilled",
    "get_fallback_log",
    "analyze_fallbacks",
    # Behavior Distiller
    "BehaviorDistiller",
    "DistillationResult",
    "DistillationAnalysis",
    "distill_ptool",
    "analyze_ptool",
    "distill_all_candidates",
    # Code Refactorer
    "CodeRefactorer",
    "RefactorResult",
    "InteractiveRefactorer",
    "refactor_program",
    "analyze_program",
    "distill_program",
    "expand_program",
    # Program Generator
    "ProgramGenerator",
    "GeneratedProgram",
    "generate_program",
    "generate_from_template",
    # ReAct Agent
    "Thought",
    "Action",
    "Observation",
    "ReActStep",
    "ReActTrajectory",
    "ReActResult",
    "ReActAgent",
    "ReActStore",
    "get_react_store",
    "set_react_store",
    "react",
    "react_and_execute",
    "ActionParseError",
    "ReActError",
    # Audit System
    "AuditResult",
    "AuditViolation",
    "AuditReport",
    "BaseAudit",
    "StructuredAudit",
    "TypicalityAudit",
    "BaseTypicalityModel",
    "AuditRegistry",
    "get_audit_registry",
    "AuditDSL",
    "audit_rule",
    "DeclarativeAudit",
    "NonEmptyTraceAudit",
    "NoFailedStepsAudit",
    "RequiredStepsAudit",
    "AuditRunner",
    "TypicalityTrainer",
    "audit_trace",
    "batch_audit",
    # Critic System
    "CriticVerdict",
    "CriticEvaluation",
    "TraceCritic",
    "evaluate_trace",
    "quick_evaluate",
    # Repair Agent
    "RepairActionType",
    "RepairAction",
    "RepairResult",
    "RepairAgent",
    "repair_trace",
    "auto_repair",
    # Model Selector
    "TaskComplexity",
    "ModelConfig",
    "SelectionResult",
    "FallbackChain",
    "ExperienceStore",
    "ModelSelector",
    "select_model",
    "get_model_for_complexity",
    # LLMS.json utilities
    "load_llms_config",
    "get_default_model",
    "get_enabled_models",
    "set_model_enabled",
    "add_model_tag",
    "remove_model_tag",
    # LLM Config Builder
    "LLMConfigBuilder",
    "validate_llms_json",
    "validate_llms_file",
    "build_llms_config",
    "update_llms_config",
    "enable_models_by_tag",
    "disable_all_models",
    "enable_only_tag",
    # Multi-Agent Orchestration (L4)
    "ExecutionMode",
    "AgentSpec",
    "RoutingDecision",
    "OrchestrationStep",
    "OrchestrationTrace",
    "OrchestrationResult",
    "BaseRouter",
    "RuleBasedRouter",
    "LLMRouter",
    "ExperienceBasedRouter",
    "HybridRouter",
    "AgentOrchestrator",
    "OrchestrationStore",
    "get_orchestration_store",
    "set_orchestration_store",
    "orchestrate",
    # L2 StateGraph
    "StateGraph",
    "StateNode",
    "StateEdge",
    "StateCondition",
    "StateGraphResult",
    "StateGraphError",
    "run_stategraph",
    # Self-Improving Agents (L5)
    "PatternType",
    "LearnedPattern",
    "ICLExample",
    "LearningEvent",
    "SelfImprovementMetrics",
    "PatternExtractor",
    "PatternMemory",
    "get_pattern_memory",
    "SelfImprovingAgent",
    "self_improving_react",
    # Python Sandbox
    "PythonSandbox",
    "SandboxConfig",
    "SandboxError",
    "SandboxImportError",
    "SandboxTimeoutError",
    "sandbox",
    "safe_exec",
    "safe_eval",
    "get_sandbox",
    "configure_sandbox",
]

__version__ = "0.5.0"  # L4 + L5 features
