# L2 StateGraph Design (Draft)

## Goal
Provide an L2 autonomy tier between L1 routing and L3 ReAct: a deterministic, inspectable state graph that chooses the next step based on explicit state and guard conditions, not free-form reasoning.

This should:
- Keep control flow explicit and testable.
- Allow limited adaptivity (branching, retries, backoff) without L3 unpredictability.
- Reuse existing ptool and trace infrastructure.

## Non-Goals
- Full L3 reasoning or tool selection by the LLM.
- Unstructured natural-language policy for transitions.
- A separate runtime that bypasses existing tracing, audits, or distillation.

## Core Abstractions

### StateGraph
A directed graph of nodes and edges.

```python
@dataclass
class StateGraph:
    name: str
    nodes: Dict[str, StateNode]
    edges: List[StateEdge]
    start: str
    terminal: Set[str]
```

### StateNode
A node executes a ptool or Python function and updates the state.

```python
@dataclass
class StateNode:
    node_id: str
    kind: Literal["ptool", "python"]
    callable_id: str  # ptool name or python handler name
    input_map: Dict[str, str]  # state key -> arg name
    output_map: Dict[str, str]  # result key -> state key
    goal: Optional[str] = None
```

### StateEdge
A transition conditioned on the current state.

```python
@dataclass
class StateEdge:
    src: str
    dst: str
    condition: "StateCondition"
    on_failure: bool = False
```

### StateCondition
A small, explicit predicate language compiled to Python for evaluation.

```python
@dataclass
class StateCondition:
    expr: str  # e.g. "state['score'] >= 10 and state['ok'] == true"
```

## Execution Semantics
1. Initialize `state` with inputs and metadata.
2. Start at `graph.start`.
3. Execute the node callable.
4. Write outputs to `state`.
5. Evaluate outgoing edges in priority order (first match wins).
6. Stop on terminal nodes, or on error if no edge matches.

Execution is deterministic given the same ptool outputs.

## Integration with Existing Components

### Tracing
- Each node execution emits a `TraceStep`.
- The entire graph run can be represented as a `WorkflowTrace`.
- Optionally record graph metadata in `WorkflowTrace.meta` (graph name, node path).

### Audits
- Audits can run on the resulting `WorkflowTrace`.
- Add graph-level audits such as "no illegal transitions" and "required nodes executed".

### Distillation
- A StateGraph can be distilled into a fixed Python workflow when edges are static.
- A WorkflowTrace from L3 can be clustered into an L2 StateGraph (future work).

## API Sketch

### Define a graph

```python
from ptool_framework.stategraph import StateGraph, StateNode, StateEdge, StateCondition

graph = StateGraph(
    name="medcalc_l2",
    start="identify",
    terminal={"done"},
    nodes={
        "identify": StateNode(
            node_id="identify",
            kind="ptool",
            callable_id="identify_calculator",
            input_map={"task": "task_description"},
            output_map={"calculator": "calculator"},
            goal="Choose calculator",
        ),
        "extract": StateNode(
            node_id="extract",
            kind="ptool",
            callable_id="extract_clinical_values",
            input_map={"note": "note", "required": "required_fields"},
            output_map={"values": "values"},
            goal="Extract values",
        ),
        "compute": StateNode(
            node_id="compute",
            kind="python",
            callable_id="compute_score",
            input_map={"calculator": "calculator", "values": "values"},
            output_map={"score": "score"},
        ),
        "done": StateNode(
            node_id="done",
            kind="python",
            callable_id="identity",
            input_map={"score": "score"},
            output_map={"answer": "answer"},
        ),
    },
    edges=[
        StateEdge("identify", "extract", StateCondition("state['calculator'] is not None")),
        StateEdge("extract", "compute", StateCondition("state['values'] is not None")),
        StateEdge("compute", "done", StateCondition("state['score'] is not None")),
    ],
)
```

### Run a graph

```python
from ptool_framework.stategraph import run_stategraph

result = run_stategraph(graph, inputs={
    "task_description": "Compute CHA2DS2-VASc",
    "note": note_text,
    "required_fields": ["age", "sex", "history"],
})
print(result.state["answer"])
```

## Data Storage
- Graphs should be serializable to JSON for reproducibility.
- Add `StateGraph.to_dict()` and `StateGraph.from_dict()`.

## Error Handling
- Each node execution yields a success/failure status.
- Failed nodes can be routed via `StateEdge(..., on_failure=True)`.
- If no edge matches, raise `StateGraphError` with the node id and state snapshot.

## CLI (Optional)
- `ptool graph run path/to/graph.json --inputs inputs.json`
- `ptool graph validate path/to/graph.json`

## Testing Strategy
- Unit tests: condition evaluation, edge selection, node output mapping.
- Integration tests: use a small ptool set with deterministic outputs.
- Audit tests: verify illegal transition detection.

## Open Questions
- Condition language: restrict to a safe subset or reuse Python eval with sandboxing.
- How to express retries and backoff (edge metadata vs node wrapper).
- Should graph nodes support batch execution for throughput.

