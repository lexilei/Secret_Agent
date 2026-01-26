# API Reference

Complete API reference for the audit, critic, repair, and model selection systems.

## Audit System

### ptool_framework.audit.base

#### AuditResult (Enum)
```python
class AuditResult(Enum):
    PASS = "pass"      # Audit passed
    FAIL = "fail"      # Audit failed
    ABSTAIN = "abstain" # Audit could not make a determination
```

#### AuditViolation
```python
@dataclass
class AuditViolation:
    audit_name: str           # Name of the audit
    rule_name: str            # Name of the failed rule
    message: str              # Error message
    severity: str = "error"   # "error", "warning", "info"
    step_index: Optional[int] = None  # Affected step
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict) -> "AuditViolation": ...
```

#### AuditReport
```python
@dataclass
class AuditReport:
    trace_id: Optional[str] = None
    result: AuditResult = AuditResult.PASS
    violations: List[AuditViolation] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    typicality_score: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_pass(self) -> bool: ...
    @property
    def is_fail(self) -> bool: ...
    @property
    def error_count(self) -> int: ...
    @property
    def warning_count(self) -> int: ...

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict) -> "AuditReport": ...
```

#### StructuredAudit
```python
class StructuredAudit:
    def __init__(self, name: str, description: str = ""): ...

    def add_rule(
        self,
        rule_name: str,
        check_fn: Callable[[pd.DataFrame, Dict], bool],
        error_message: str,
        severity: str = "error"
    ) -> "StructuredAudit": ...

    def run(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> AuditReport: ...
```

#### TypicalityAudit
```python
class TypicalityAudit:
    def __init__(
        self,
        name: str,
        model: "BaseTypicalityModel",
        fail_threshold: float = 0.1,
        pass_threshold: float = 0.5
    ): ...

    def run(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> AuditReport: ...
```

#### AuditRegistry
```python
class AuditRegistry:
    def register(
        self,
        audit: StructuredAudit,
        domain: Optional[str] = None
    ) -> None: ...

    def get(self, name: str) -> Optional[StructuredAudit]: ...
    def unregister(self, name: str) -> Optional[StructuredAudit]: ...
    def list_all(self) -> List[StructuredAudit]: ...
    def list_by_domain(self, domain: str) -> List[StructuredAudit]: ...
    def list_domains(self) -> List[str]: ...

    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...

# Global registry functions
def get_audit_registry() -> AuditRegistry: ...
def register_audit(audit: StructuredAudit, domain: Optional[str] = None) -> None: ...
```

### ptool_framework.audit.structured_audit

#### AuditDSL
```python
class AuditDSL:
    def __init__(self, df: pd.DataFrame, metadata: Dict[str, Any]): ...

    # Query methods
    def steps_named(self, fn_name: str, exact: bool = False) -> pd.DataFrame: ...
    def steps_with_status(self, status: str) -> pd.DataFrame: ...
    def get_step(self, index: int) -> Optional[pd.Series]: ...
    def get_pattern(self) -> List[str]: ...

    # Count assertions
    def has_exactly(self, fn_name: str, count: int) -> bool: ...
    def has_at_least(self, fn_name: str, count: int) -> bool: ...
    def has_at_most(self, fn_name: str, count: int) -> bool: ...
    def is_non_empty(self) -> bool: ...

    # Order assertions
    def comes_before(self, fn_a: str, fn_b: str) -> bool: ...
    def comes_after(self, fn_a: str, fn_b: str) -> bool: ...
    def pattern_exists(
        self,
        pattern: List[str],
        contiguous: bool = False
    ) -> bool: ...

    # Status checks
    def all_completed(self) -> bool: ...
    def no_failures(self) -> bool: ...
    def success_rate(self) -> float: ...
```

#### audit_rule (Decorator)
```python
def audit_rule(rule_name: str, error_message: str, severity: str = "error"):
    """Decorator to mark a method as an audit rule."""
    ...
```

#### DeclarativeAudit
```python
class DeclarativeAudit(StructuredAudit):
    """Base class for declarative audits using @audit_rule decorator."""
    ...
```

#### Pre-built Audits
```python
class NonEmptyTraceAudit(DeclarativeAudit): ...
class NoFailedStepsAudit(DeclarativeAudit): ...
class AllCompletedAudit(DeclarativeAudit): ...
class RequiredStepsAudit(DeclarativeAudit):
    def __init__(self, required_steps: List[str]): ...
class StepOrderAudit(DeclarativeAudit):
    def __init__(self, order_constraints: List[Tuple[str, str]]): ...
class PatternAudit(DeclarativeAudit):
    def __init__(
        self,
        required_patterns: Optional[List[List[str]]] = None,
        forbidden_patterns: Optional[List[List[str]]] = None
    ): ...
class PerformanceAudit(DeclarativeAudit):
    def __init__(
        self,
        max_step_duration_ms: float = 5000,
        max_step_count: int = 20
    ): ...

# Factory functions
def create_basic_audit() -> StructuredAudit: ...
def create_workflow_audit(
    required_steps: Optional[List[str]] = None,
    step_order: Optional[List[Tuple[str, str]]] = None
) -> StructuredAudit: ...
```

### ptool_framework.audit.typicality.models

#### BaseTypicalityModel
```python
class BaseTypicalityModel(ABC):
    @abstractmethod
    def fit(self, patterns: List[List[str]]) -> "BaseTypicalityModel": ...
    @abstractmethod
    def score(self, pattern: List[str]) -> float: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "BaseTypicalityModel": ...
```

#### UnigramModel
```python
class UnigramModel(BaseTypicalityModel):
    def __init__(self, smoothing_alpha: float = 1.0): ...
    def fit(self, patterns: List[List[str]]) -> "UnigramModel": ...
    def score(self, pattern: List[str]) -> float: ...
    def step_probability(self, step: str) -> float: ...
```

#### BigramModel / TrigramModel
```python
class BigramModel(NGramModel):
    def __init__(self, smoothing_alpha: float = 0.1): ...

class TrigramModel(NGramModel):
    def __init__(self, smoothing_alpha: float = 0.1): ...
```

#### InterpolatedModel
```python
class InterpolatedModel(BaseTypicalityModel):
    def __init__(self, lambdas: Tuple[float, float, float] = (0.1, 0.3, 0.6)): ...
    def fit(self, patterns: List[List[str]]) -> "InterpolatedModel": ...
    def score(self, pattern: List[str]) -> float: ...
```

#### HMMModel
```python
class HMMModel(BaseTypicalityModel):
    def __init__(self, n_states: int = 5): ...
    def fit(self, patterns: List[List[str]]) -> "HMMModel": ...
    def score(self, pattern: List[str]) -> float: ...
```

#### Factory Functions
```python
def create_typicality_model(model_type: str) -> BaseTypicalityModel: ...
def train_ensemble_models(
    patterns: List[List[str]]
) -> Dict[str, BaseTypicalityModel]: ...
def ensemble_score(
    models: Dict[str, BaseTypicalityModel],
    pattern: List[str],
    weights: Optional[Dict[str, float]] = None
) -> float: ...
```

### ptool_framework.audit.runner

#### AuditRunner
```python
class AuditRunner:
    def __init__(
        self,
        audits: Optional[List[StructuredAudit]] = None,
        typicality_model: Optional[BaseTypicalityModel] = None
    ): ...

    def add_audit(self, audit: StructuredAudit) -> None: ...

    def audit_trace(
        self,
        trace: Union[WorkflowTrace, List[Dict]],
        metadata: Optional[Dict] = None
    ) -> AuditReport: ...

    def run_batch(
        self,
        traces: List[Union[WorkflowTrace, List[Dict]]],
        metadata_list: Optional[List[Dict]] = None
    ) -> BatchAuditResult: ...
```

#### BatchAuditResult
```python
@dataclass
class BatchAuditResult:
    total: int
    passed: int
    failed: int
    reports: List[AuditReport]
    failed_traces: List[int]

    @property
    def pass_rate(self) -> float: ...
```

## Critic System

### ptool_framework.critic

#### CriticVerdict
```python
class CriticVerdict(Enum):
    ACCEPT = "accept"            # Trace is acceptable
    REPAIR_NEEDED = "repair_needed"  # Trace needs repair
    REJECT = "reject"            # Trace is unfixable
```

#### CriticEvaluation
```python
@dataclass
class CriticEvaluation:
    verdict: CriticVerdict
    confidence: float
    trace_id: Optional[str] = None
    audit_reports: List[AuditReport] = field(default_factory=list)
    failed_steps: List[int] = field(default_factory=list)
    repair_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    completeness_score: float = 0.0
    correctness_score: float = 0.0
    efficiency_score: float = 0.0
    reasoning_issues: List[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool: ...
    @property
    def needs_repair(self) -> bool: ...
    @property
    def should_reject(self) -> bool: ...

    def to_dict(self) -> Dict[str, Any]: ...
    def summary(self) -> str: ...
```

#### TraceCritic
```python
class TraceCritic:
    def __init__(
        self,
        audits: Optional[List[StructuredAudit]] = None,
        typicality_model: Optional[BaseTypicalityModel] = None,
        accept_threshold: float = 0.8,
        repair_threshold: float = 0.4
    ): ...

    def add_audit(self, audit: StructuredAudit) -> None: ...

    def evaluate(
        self,
        trace: Union[WorkflowTrace, List[Dict]],
        goal: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> CriticEvaluation: ...

# Convenience functions
def evaluate_trace(
    trace: Union[WorkflowTrace, List[Dict]],
    audits: Optional[List[StructuredAudit]] = None
) -> CriticEvaluation: ...

def quick_evaluate(trace: Union[WorkflowTrace, List[Dict]]) -> CriticVerdict: ...
```

## Repair System

### ptool_framework.repair

#### RepairActionType
```python
class RepairActionType(Enum):
    REGENERATE_STEP = "regenerate_step"
    ADD_STEP = "add_step"
    REMOVE_STEP = "remove_step"
    MODIFY_ARGS = "modify_args"
    REORDER_STEPS = "reorder_steps"
```

#### RepairAction
```python
@dataclass
class RepairAction:
    action_type: RepairActionType
    step_index: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]: ...
```

#### RepairResult
```python
@dataclass
class RepairResult:
    success: bool
    repaired_trace: Optional[WorkflowTrace] = None
    actions_taken: List[RepairAction] = field(default_factory=list)
    iterations: int = 0
    final_evaluation: Optional[CriticEvaluation] = None
    original_verdict: Optional[CriticVerdict] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]: ...
    def summary(self) -> str: ...
```

#### RepairAgent
```python
class RepairAgent:
    def __init__(
        self,
        critic: Optional[TraceCritic] = None,
        react_agent: Optional[ReActAgent] = None,
        trace_store: Optional[TraceStore] = None,
        model: str = "deepseek-v3",
        llm_backend: Optional[Callable] = None,
        max_attempts: int = 3,
        use_icl_examples: bool = True
    ): ...

    def repair(
        self,
        trace: Union[WorkflowTrace, List[Dict]],
        evaluation: CriticEvaluation,
        goal: str
    ) -> RepairResult: ...

# Convenience functions
def repair_trace(
    trace: Union[WorkflowTrace, List[Dict]],
    goal: str,
    critic: Optional[TraceCritic] = None,
    max_attempts: int = 3
) -> RepairResult: ...

def auto_repair(
    trace: Union[WorkflowTrace, List[Dict]],
    goal: str
) -> Optional[WorkflowTrace]: ...
```

## Model Selection

### ptool_framework.model_selector

#### TaskComplexity
```python
class TaskComplexity(Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"
```

#### ModelConfig
```python
@dataclass
class ModelConfig:
    name: str
    provider: str = "together"
    cost_per_1k_input: float = 0.001
    cost_per_1k_output: float = 0.002
    max_tokens: int = 32000
    capabilities: List[str] = field(default_factory=list)
    latency_ms_avg: float = 1000.0
    quality_score: float = 0.7

    def to_dict(self) -> Dict[str, Any]: ...
```

#### SelectionCriteria
```python
@dataclass
class SelectionCriteria:
    required_capabilities: List[str] = field(default_factory=list)
    max_cost_per_1k_tokens: float = 0.01
    max_latency_ms: int = 5000
    min_quality_threshold: float = 0.5
```

#### SelectionResult
```python
@dataclass
class SelectionResult:
    selected_model: str
    config: Optional[ModelConfig] = None
    reason: str = ""
    confidence: float = 0.8
    fallback_chain: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]: ...
```

#### FallbackChain
```python
@dataclass
class FallbackChain:
    models: List[str] = field(default_factory=list)
    reason: str = ""

    def __iter__(self): ...
    def __len__(self): ...
```

#### ModelPerformance
```python
@dataclass
class ModelPerformance:
    model_name: str
    ptool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    avg_latency_ms: float = 0.0
    total_cost: float = 0.0

    @property
    def success_rate(self) -> float: ...

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelPerformance": ...
```

#### ExperienceStore
```python
class ExperienceStore:
    def __init__(self, path: str = "~/.ptool_experiences"): ...

    def record_execution(
        self,
        model_name: str,
        ptool_name: str,
        success: bool,
        latency_ms: float,
        cost: float = 0.0
    ) -> None: ...

    def get_performance(
        self,
        model_name: str,
        ptool_name: str
    ) -> Optional[ModelPerformance]: ...

    def get_all_performance(
        self,
        model_name: str
    ) -> Dict[str, ModelPerformance]: ...

    def get_best_model_for_ptool(
        self,
        ptool_name: str,
        min_calls: int = 5
    ) -> Optional[str]: ...

    def get_model_ranking(
        self,
        ptool_name: str,
        min_calls: int = 3
    ) -> List[Tuple[str, float]]: ...
```

#### ModelSelector
```python
class ModelSelector:
    def __init__(
        self,
        experience_store: Optional[ExperienceStore] = None,
        models: Optional[Dict[str, ModelConfig]] = None,
        default_model: str = "deepseek-v3",
        enable_learning: bool = True,
        complexity_estimator: Optional[Callable] = None
    ): ...

    def select(
        self,
        ptool_spec: PToolSpec,
        inputs: Dict[str, Any],
        criteria: Optional[SelectionCriteria] = None
    ) -> str: ...

    def select_with_details(
        self,
        ptool_spec: PToolSpec,
        inputs: Dict[str, Any],
        criteria: Optional[SelectionCriteria] = None
    ) -> SelectionResult: ...

    def get_fallback_chain(
        self,
        ptool_spec: PToolSpec,
        inputs: Dict[str, Any],
        max_length: int = 3
    ) -> FallbackChain: ...

    def record_execution(
        self,
        ptool_name: str,
        model: str,
        inputs: Dict[str, Any],
        success: bool,
        latency_ms: float,
        cost: float = 0.0
    ) -> None: ...

# Convenience functions
def heuristic_complexity_estimator(
    ptool_spec: PToolSpec,
    inputs: Dict[str, Any]
) -> TaskComplexity: ...

def select_model(
    ptool_name: str,
    inputs: Dict[str, Any],
    default: str = "deepseek-v3"
) -> str: ...

def get_model_for_complexity(
    complexity: TaskComplexity,
    default: str = "deepseek-v3"
) -> str: ...
```

## DataFrame Converter

### ptool_framework.audit.dataframe_converter

```python
class TraceDataFrameConverter:
    """Convert various trace types to pandas DataFrames."""

    @classmethod
    def convert(
        cls,
        trace: Union[WorkflowTrace, ReActTrajectory, List[Dict]]
    ) -> pd.DataFrame: ...

    @classmethod
    def extract_metadata(
        cls,
        trace: Union[WorkflowTrace, ReActTrajectory, List[Dict]]
    ) -> Dict[str, Any]: ...

# Convenience functions
def convert_trace(
    trace: Union[WorkflowTrace, ReActTrajectory, List[Dict]]
) -> pd.DataFrame: ...

def batch_convert_traces(
    traces: List[Union[WorkflowTrace, ReActTrajectory, List[Dict]]]
) -> List[pd.DataFrame]: ...
```

## Typicality Patterns

### ptool_framework.audit.typicality.patterns

```python
# Constants
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"

# Functions
def extract_pattern(
    df: pd.DataFrame,
    add_special_tokens: bool = True
) -> List[str]: ...

def get_ngrams(
    pattern: List[str],
    n: int
) -> List[Tuple[str, ...]]: ...

def get_bigrams(pattern: List[str]) -> List[Tuple[str, str]]: ...
def get_trigrams(pattern: List[str]) -> List[Tuple[str, str, str]]: ...

def normalize_step_names(
    patterns: List[List[str]],
    method: str = "lowercase"
) -> List[List[str]]: ...

class PatternStats:
    @classmethod
    def from_patterns(cls, patterns: List[List[str]]) -> "PatternStats": ...

    def most_common_steps(self, n: int = 10) -> List[Tuple[str, int]]: ...
    def transition_probability(self, from_step: str, to_step: str) -> float: ...
    def summary(self) -> str: ...
```

## See Also

- [Audit System](audit-system.md)
- [Critic and Repair](critic-repair.md)
- [Model Selection](model-selection.md)
- [MedCalc Domain](medcalc-domain.md)
