"""
L2 StateGraph: deterministic, inspectable workflow graph.

This module provides a minimal state graph runtime that executes
ptools or Python handlers with explicit, testable control flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Literal

from .ptool import get_registry
from .llm_backend import execute_ptool
from .traces import WorkflowTrace, StepStatus


class StateGraphError(RuntimeError):
    """Raised when a state graph fails to execute."""


@dataclass
class StateCondition:
    """Boolean predicate over the current state."""

    expr: str

    def evaluate(self, state: Dict[str, Any]) -> bool:
        """Evaluate condition against the current state."""
        safe_globals = {"__builtins__": {}}
        safe_locals = {"state": state, "True": True, "False": False, "None": None}
        return bool(eval(self.expr, safe_globals, safe_locals))


@dataclass
class StateEdge:
    src: str
    dst: str
    condition: StateCondition
    on_failure: bool = False


@dataclass
class StateNode:
    node_id: str
    kind: Literal["ptool", "python"]
    callable_id: str
    input_map: Dict[str, str] = field(default_factory=dict)  # state_key -> arg_name
    output_map: Dict[str, str] = field(default_factory=dict)  # result_key -> state_key
    goal: Optional[str] = None


@dataclass
class StateGraph:
    name: str
    nodes: Dict[str, StateNode]
    edges: List[StateEdge]
    start: str
    terminal: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "terminal": sorted(self.terminal),
            "nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "callable_id": node.callable_id,
                    "input_map": dict(node.input_map),
                    "output_map": dict(node.output_map),
                    "goal": node.goal,
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "condition": edge.condition.expr,
                    "on_failure": edge.on_failure,
                }
                for edge in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateGraph:
        nodes = {
            node_id: StateNode(
                node_id=node["node_id"],
                kind=node["kind"],
                callable_id=node["callable_id"],
                input_map=dict(node.get("input_map", {})),
                output_map=dict(node.get("output_map", {})),
                goal=node.get("goal"),
            )
            for node_id, node in data["nodes"].items()
        }
        edges = [
            StateEdge(
                src=edge["src"],
                dst=edge["dst"],
                condition=StateCondition(edge["condition"]),
                on_failure=edge.get("on_failure", False),
            )
            for edge in data.get("edges", [])
        ]
        return cls(
            name=data["name"],
            nodes=nodes,
            edges=edges,
            start=data["start"],
            terminal=set(data.get("terminal", [])),
        )


@dataclass
class StateGraphResult:
    graph: StateGraph
    state: Dict[str, Any]
    trace: WorkflowTrace
    path: List[str]
    success: bool
    error: Optional[str] = None


def _map_inputs(state: Dict[str, Any], input_map: Dict[str, str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    for state_key, arg_name in input_map.items():
        if state_key not in state:
            raise StateGraphError(f"Missing state key: {state_key}")
        args[arg_name] = state[state_key]
    return args


def _map_outputs(state: Dict[str, Any], output_map: Dict[str, str], result: Any) -> None:
    if not output_map:
        return
    if len(output_map) == 1 and "result" in output_map:
        state[output_map["result"]] = result
        return
    if not isinstance(result, dict):
        raise StateGraphError("Expected dict result for output mapping")
    for result_key, state_key in output_map.items():
        if result_key not in result:
            raise StateGraphError(f"Missing result key: {result_key}")
        state[state_key] = result[result_key]


def run_stategraph(
    graph: StateGraph,
    inputs: Dict[str, Any],
    python_handlers: Optional[Dict[str, Callable[..., Any]]] = None,
    max_steps: int = 100,
) -> StateGraphResult:
    """Execute a StateGraph deterministically with explicit state transitions."""
    python_handlers = python_handlers or {}
    state: Dict[str, Any] = dict(inputs)
    trace = WorkflowTrace(goal=state.get("goal", graph.name))
    path: List[str] = []

    current = graph.start
    for _ in range(max_steps):
        if current not in graph.nodes:
            raise StateGraphError(f"Unknown node: {current}")
        node = graph.nodes[current]
        path.append(current)

        args = _map_inputs(state, node.input_map)
        step = trace.add_step(
            ptool_name=f"{node.kind}:{node.callable_id}",
            args=args,
            goal=node.goal,
        )
        step.status = StepStatus.RUNNING

        ok = True
        try:
            if node.kind == "ptool":
                spec = get_registry().get(node.callable_id)
                if spec is None:
                    raise StateGraphError(f"Unknown ptool: {node.callable_id}")
                result = execute_ptool(spec, args)
            else:
                handler = python_handlers.get(node.callable_id)
                if handler is None:
                    raise StateGraphError(f"Unknown python handler: {node.callable_id}")
                result = handler(**args)
            _map_outputs(state, node.output_map, result)
            step.status = StepStatus.COMPLETED
            step.result = result
        except Exception as exc:
            ok = False
            step.status = StepStatus.FAILED
            step.error = str(exc)

        if ok and current in graph.terminal:
            return StateGraphResult(
                graph=graph,
                state=state,
                trace=trace,
                path=path,
                success=True,
            )

        candidates = [
            edge
            for edge in graph.edges
            if edge.src == current and edge.on_failure == (not ok)
        ]
        next_node = None
        for edge in candidates:
            if edge.condition.evaluate(state):
                next_node = edge.dst
                break
        if next_node is None:
            if ok and current in graph.terminal:
                return StateGraphResult(
                    graph=graph,
                    state=state,
                    trace=trace,
                    path=path,
                    success=True,
                )
            return StateGraphResult(
                graph=graph,
                state=state,
                trace=trace,
                path=path,
                success=False,
                error=f"No valid transition from node: {current}",
            )
        current = next_node

    return StateGraphResult(
        graph=graph,
        state=state,
        trace=trace,
        path=path,
        success=False,
        error="Max steps exceeded",
    )
