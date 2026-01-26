# TODO: Completing William's Vision

A roadmap to fully realize the ptool_framework research agenda.

**Last Updated**: 2025-01-08
**Audit Status**: Comprehensive codebase audit completed

---

## Current Status (v0.4.0)

### Fully Implemented ✓

- [x] `@ptool` decorator - LLM-executed function stubs (`ptool.py`)
- [x] `@distilled` decorator - Python with LLM fallback (`distilled.py`)
- [x] `WorkflowTrace` - Structured execution traces (`traces.py`)
- [x] `TraceExecutor` - Execute pre-built traces (`executor.py`)
- [x] `TraceStore` - Persistent trace storage (`trace_store.py`)
- [x] `ReActAgent` - Reasoning + Acting loop (`react.py`)
- [x] `ReActStore` - Trajectory storage for distillation (`react.py`)
- [x] `ProgramGenerator` - Task description → program (`program_generator.py`)
- [x] `BehaviorDistiller` - Trace analysis (`distiller.py`)
- [x] `CodeRefactorer` - AST-based code transformation (`refactorer.py`)
- [x] Multi-provider LLM backend - Together, OpenAI, Anthropic, Groq, Gemini (`llm_backend.py`)
- [x] PTP trace format output (`react.py:to_ptp_trace()`)
- [x] Basic CLI commands (`cli.py`)
- [x] **Critic/Repair System** - `critic.py` (735 lines), `repair.py` (790 lines)
- [x] **Audit Generation** - Complete SSRM-style system (`audit/` directory)
  - Structured audits with AuditDSL (`audit/structured_audit.py`)
  - Typicality audits: Unigram, Bigram, Trigram, HMM (`audit/typicality/`)
  - LLM-based audit generation (`audit/llm_audits.py`)
  - Domain-specific audits (`audit/domains/medcalc.py`)
- [x] **Intelligent LLM Selection** - `model_selector.py` (1263 lines)
  - Task complexity estimation (heuristic + LLM)
  - Experience-based model tracking
  - Cost/latency optimization
  - Fallback chains
- [x] **Multi-Agent Orchestration** - `orchestrator.py` (500+ lines)
  - 4 router types: Rule-based, LLM, Experience-based, Hybrid
  - 3 execution modes: Sequential, Parallel, Pipeline
  - Agent handoff protocols
- [x] **Self-Improving Agents** - `self_improving.py` (600+ lines)
  - Pattern extraction from trajectories
  - ICLExample generation at L5 level
  - PatternMemory with persistence and decay
  - William's requirements: good→ICL demos, bad→negative examples
- [x] **Benchmark Suite** - `benchmark/` directory
  - L1-L5 experiments implemented
  - MedCalc dataset support
  - Metrics: accuracy, cost, latency

---

## Integration Gaps (CURRENT PRIORITY)

These components exist but **aren't connected**. Total effort: ~530 lines.

### Gap 1: Auto-ICL from TraceStore [HIGH PRIORITY]
**Status**: NOT INTEGRATED
**Impact**: HIGH - Completes the feedback loop

**Problem**:
```
@ptool execution → trace_store.log_execution() → traces saved to ~/.ptool_traces/
                                                          ↓
                                              (NEVER READ BACK FOR ICL)
```

**Solution**: Add `trace_path` and `write_to_memories` parameters to `@ptool`
```python
@ptool(
    model="deepseek-v3",
    trace_path="~/.ptool_traces",      # Path to read traces from
    write_to_memories=True             # Whether to save executions
)
def my_func(text: str) -> str:
    """..."""
```

**Files to Modify**:
- `ptool.py` - Add parameters to PToolSpec
- `llm_backend.py` - Read traces and inject as examples
- `trace_store.py` - Add `get_icl_examples()` method

**Effort**: ~100 lines

---

### Gap 2: ModelSelector in execute_ptool [HIGH PRIORITY]
**Status**: NOT INTEGRATED
**Impact**: HIGH - Enables automatic cost optimization

**Problem**: `model_selector.py` exists with full implementation but `execute_ptool()` uses hardcoded model:
```python
# llm_backend.py:609
model = model_override or spec.model  # Uses hardcoded model, ignores selector
```

**Solution**: Support `model="auto"` to trigger ModelSelector
```python
@ptool(model="auto")  # Triggers automatic model selection
def complex_task(text: str) -> Dict:
    """..."""
```

**Files to Modify**:
- `llm_backend.py:609` - Check for "auto" and call ModelSelector
- `ptool.py` - Document "auto" option

**Effort**: ~30 lines

---

### Gap 3: SelfImproving → Distiller Connection [MEDIUM PRIORITY]
**Status**: NOT INTEGRATED
**Impact**: MEDIUM - Automation of distillation

**Problem**: `SelfImprovingAgent` extracts patterns but never triggers `BehaviorDistiller`:
```python
# self_improving.py:1137-1161
# Note: In a full implementation, we would execute the repaired trace
# For now, we create a synthetic result
```

**Solution**: Add `trigger_distillation()` method
```python
# In SelfImprovingAgent
def trigger_distillation(self, min_patterns: int = 10) -> Optional[str]:
    """Convert learned patterns to @distilled code if we have enough."""
    positive_patterns = self.memory.get_patterns(PatternType.POSITIVE)
    if len(positive_patterns) < min_patterns:
        return None
    from .distiller import BehaviorDistiller
    return BehaviorDistiller().distill_from_patterns(positive_patterns)
```

**Files to Modify**:
- `self_improving.py` - Add distillation trigger
- `distiller.py` - Add `distill_from_patterns()` method

**Effort**: ~50 lines

---

### Gap 4: Missing CLI Commands [MEDIUM PRIORITY]
**Status**: NOT IMPLEMENTED
**Impact**: MEDIUM - User workflow improvements

| Command | Purpose | File | Effort |
|---------|---------|------|--------|
| `ptool learn` | Run SelfImprovingAgent and persist patterns | `cli.py` | Low |
| `ptool audit <ptool>` | Run audits on a ptool's traces | `cli.py` | Low |
| `ptool repair <trace>` | Run RepairAgent on a failed trace | `cli.py` | Low |
| `ptool orchestrate` | Launch AgentOrchestrator | `cli.py` | Medium |
| `ptool export-patterns` | Export learned patterns | `cli.py` | Low |

**Effort**: ~150 lines

---

### Gap 5: Algorithmic Trace Clustering [OPTIONAL]
**Status**: NOT IMPLEMENTED
**Impact**: MEDIUM - Reduces LLM dependency in distillation

**Problem**: `BehaviorDistiller` uses LLM to identify patterns (LLM-dependent)

**Solution**: Add ML-based clustering:
```python
class AlgorithmicDistiller:
    """ML-based trace clustering without LLM dependency."""

    def embed_traces(self, traces: List[ExecutionTrace]) -> np.ndarray:
        """Embed traces using sentence-transformers."""

    def cluster_traces(self, embeddings: np.ndarray) -> List[TraceCluster]:
        """Hierarchical clustering of trace embeddings."""
```

**Dependencies**:
```
sentence-transformers>=2.2.0
scikit-learn>=1.0.0
```

**Effort**: ~200 lines

---

## L3 Audit Improvements

Investigation items for improving the L3+ (ReAct + Audit + Repair) system:

### Investigation Items

1. **Examine actual traces** - Look at the LLM's reasoning to understand exactly what went wrong (e.g., why it returned BMI instead of CrCl)

2. **Detect wrong output type** - Improve audit system to detect when the LLM returns a completely wrong output type (e.g., BMI=15.23 instead of CrCl=25.0). Current audits check numeric tolerance but not semantic correctness of output.

3. **Investigate failure patterns** - Run more failed instances to identify systematic issues:
   - Are failures due to ambiguous patient notes (multiple time points)?
   - Calculator confusion (wrong formula selected)?
   - Value extraction errors (wrong units, wrong field)?
   - Repair mechanism limitations (feedback not specific enough)?

### Known Issues

- **Ambiguous time points**: Patient notes may have values from different dates (e.g., weight at admission vs discharge). Ground truth may use one time point while LLM uses another.
- **BMI vs CrCl confusion**: Instance 3 repair returned BMI (15.23) instead of creatinine clearance. Suggests fundamental misunderstanding, not arithmetic error.
- **Rewind-and-regenerate limitations**: When LLM fundamentally misunderstands the problem, simple regeneration may not fix it.

---

## Phase 1: Core Completeness ✓ DONE

All items in this phase are **fully implemented**:

### 1.1 Critic System ✓
- [x] `TraceCritic` class with ACCEPT/REPAIR_NEEDED/REJECT verdicts
- [x] `CriticEvaluation` with detailed scores
- [x] LLM-based reasoning analysis
- [x] Audit integration with typicality checking
- **Files**: `critic.py` (735 lines)

### 1.2 Audit Generation ✓
- [x] SSRM-style structured audits
- [x] Typicality audits (Unigram, Bigram, Trigram, HMM)
- [x] `AuditRunner` for batch execution
- [x] `AuditGenerator` from traces
- **Files**: `audit/` directory (complete subsystem)

### 1.3 Intelligent LLM Selection ✓
- [x] `ModelSelector` with complexity estimation
- [x] `ExperienceStore` for performance tracking
- [x] Cost/latency requirements
- [x] Fallback chains
- **Files**: `model_selector.py` (1263 lines)

---

## Phase 2: Distillation Pipeline ◐ PARTIAL

### 2.1 Automated Distillation Loop
- [x] `BehaviorDistiller` basic implementation
- [x] `@distilled` decorator with fallback
- [x] Trace storage and retrieval
- [ ] **Gap**: Trace clustering not connected to execute_ptool
- [ ] **Gap**: Automatic ICL injection not implemented

### 2.2 Incremental Distillation
- [x] `DistillationFallback` exception
- [x] Fallback tracking
- [ ] Pattern coverage tracking
- [ ] Auto-pattern generation from clusters

---

## Phase 3: Advanced Agents ✓ DONE

### 3.1 Multi-Agent Orchestration ✓
- [x] `AgentOrchestrator` class
- [x] 4 router types (Rule, LLM, Experience, Hybrid)
- [x] Parallel agent execution
- [x] Agent handoff protocols
- **Files**: `orchestrator.py` (500+ lines)

### 3.2 Self-Improving Agents ✓
- [x] `SelfImprovingAgent` wrapper
- [x] `PatternExtractor` for trajectories
- [x] `PatternMemory` with persistence
- [x] Forgetting mechanism (decay)
- [ ] **Gap**: No trigger for BehaviorDistiller
- **Files**: `self_improving.py` (600+ lines)

---

## Phase 4: Evaluation & Benchmarks ✓ DONE

### 4.1 Benchmark Suite ✓
- [x] `BenchmarkRunner` class
- [x] L1-L5 experiment framework
- [x] MedCalc benchmark adapter
- [x] Metrics: accuracy, cost, latency
- **Files**: `benchmark/` directory

### 4.2 Distillation Metrics ◐ PARTIAL
- [x] Cost tracking
- [x] Accuracy metrics
- [ ] Python coverage tracking
- [ ] Dashboard with distillation graphs

---

## Phase 5: Production Readiness ◐ PARTIAL

### 5.1 Dashboard Improvements
- [x] Basic dashboard framework
- [ ] Real-time execution visualization
- [ ] Trace explorer with filtering
- [ ] Distillation progress graphs

### 5.2 CLI Completeness
- [x] `ptool generate` - Program generation
- [x] `ptool run` - Execution with traces
- [x] `ptool refactor` - Code transformation
- [x] `ptool models` - List models
- [x] `ptool traces` - View traces
- [x] `ptool dashboard` - Launch dashboard
- [ ] **Gap**: Missing 5 commands (see Gap 4)

---

## Phase 6: Research Extensions ○ NOT STARTED

### 6.1 Trace Generalization
- [ ] Trace alignment algorithm
- [ ] `WorkflowTemplate` data structure

### 6.2 Hierarchical Distillation
- [ ] Workflow-level distillation
- [ ] Task family clustering

### 6.3 Counterfactual Reasoning
- [ ] `CounterfactualAnalyzer`
- [ ] Alternative action generation

### 6.4 Advanced Experiment Layers (Future Work)

Experiments requiring further research and implementation:

#### L5 Full Implementation
- [ ] Complete L5 ICL experiments with full trace integration
- [ ] L5 self-improving agent with automatic pattern learning
- [ ] Benchmark L5 against L3/L4 for accuracy vs cost tradeoff
- [ ] Implement trace-to-ICL example pipeline
- [ ] Add pattern decay and refresh mechanisms

#### L4 Adaptive Refinements
- [ ] Fine-tune difficulty classification thresholds
- [ ] Add model-per-difficulty routing (cheap models for simple, expensive for hard)
- [ ] Implement hybrid rule+LLM difficulty estimation
- [ ] Experiment with adaptive routing based on confidence scores
- [ ] Add dynamic difficulty reclassification based on failure patterns

#### L4 Orchestrator Experiments
- [ ] Multi-agent pipeline with specialist agents per stage
- [ ] Agent handoff optimization for medical domain
- [ ] Parallel agent execution for independent extraction tasks
- [ ] Compare orchestrator overhead vs accuracy gains
- [ ] Implement agent specialization (extractor agent, validator agent, calculator agent)

---

## Priority Order

### Immediate (This Week)
1. **Gap 1**: Auto-ICL from TraceStore (~100 lines)
2. **Gap 2**: ModelSelector integration (~30 lines)

### Short-term (Next 2 Weeks)
3. **Gap 4**: Missing CLI commands (~150 lines)
4. **Gap 3**: SelfImproving → Distiller (~50 lines)

### Medium-term (Month)
5. **Gap 5**: Algorithmic clustering (~200 lines)
6. Dashboard improvements
7. Documentation completion

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Core components implemented | 90% | 100% |
| Integration gaps closed | 0/5 | 5/5 |
| Auto-ICL working | No | Yes |
| ModelSelector integrated | No | Yes |
| CLI commands | 7 | 12 |
| Benchmark accuracy (MedCalc) | TBD | 85% |
| Cost reduction from auto-select | 0% | 30% |

---

## References

- William Cohen's 2026 Agent Research Plan
- Program Trace Prompting (PTP) paper
- SSRM: Auditing Reasoners paper
- ReAct: Synergizing Reasoning and Acting
- Reflexion: Language Agents with Verbal Reinforcement Learning
