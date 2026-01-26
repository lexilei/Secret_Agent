"""
Multi-Agent Orchestration (L4) for ptool_framework.

This module implements William's L4 vision: Multi-agent systems with routing.
It allows multiple specialized ReActAgents to be orchestrated together,
with intelligent routing to select the best agent for each task.

Key components:
- AgentSpec: Specification for a specialized agent
- Routers: RuleBasedRouter, LLMRouter, ExperienceBasedRouter, HybridRouter
- AgentOrchestrator: Main orchestration class
- OrchestrationStore: Persistent storage for orchestration traces

Example:
    >>> orchestrator = AgentOrchestrator()
    >>> orchestrator.register_agent(AgentSpec(
    ...     name="extractor",
    ...     description="Extracts structured data from text",
    ...     domains=["extraction", "parsing"],
    ...     available_ptools=["extract_values", "parse_json"]
    ... ))
    >>> orchestrator.register_agent(AgentSpec(
    ...     name="calculator",
    ...     description="Performs medical calculations",
    ...     domains=["medcalc", "math"],
    ...     available_ptools=["calculate", "unit_convert"]
    ... ))
    >>> result = orchestrator.run("Extract patient weight and calculate BMI")
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import json
import os
import re
import uuid

from .react import ReActAgent, ReActResult, ReActTrajectory
from .ptool import PToolSpec, get_registry
from .llm_backend import call_llm
from .model_selector import ExperienceStore


# =============================================================================
# Enums
# =============================================================================

class ExecutionMode(Enum):
    """How to execute agent tasks."""
    SEQUENTIAL = "sequential"  # One agent at a time
    PARALLEL = "parallel"      # Multiple agents in parallel
    PIPELINE = "pipeline"      # Output of one feeds into next


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AgentSpec:
    """
    Specification for a specialized agent.

    Defines what ptools an agent can use and what domains it handles.
    """
    name: str
    description: str
    domains: List[str]  # Which domains this agent handles (e.g., ["extraction", "medcalc"])
    available_ptools: List[str]  # Ptool names this agent can use
    model: str = "deepseek-v3-0324"
    capabilities: List[str] = field(default_factory=list)  # e.g., ["extraction", "calculation"]
    priority: int = 0  # Higher = preferred when multiple match
    max_steps: int = 10  # Max ReAct steps for this agent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "domains": self.domains,
            "available_ptools": self.available_ptools,
            "model": self.model,
            "capabilities": self.capabilities,
            "priority": self.priority,
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpec":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            domains=data["domains"],
            available_ptools=data["available_ptools"],
            model=data.get("model", "deepseek-v3-0324"),
            capabilities=data.get("capabilities", []),
            priority=data.get("priority", 0),
            max_steps=data.get("max_steps", 10),
        )

    def format_for_prompt(self) -> str:
        """Format agent info for LLM routing prompt."""
        return (
            f"Agent: {self.name}\n"
            f"  Description: {self.description}\n"
            f"  Domains: {', '.join(self.domains)}\n"
            f"  Capabilities: {', '.join(self.capabilities) if self.capabilities else 'general'}\n"
            f"  Tools: {', '.join(self.available_ptools)}"
        )


@dataclass
class RoutingDecision:
    """Result of the router deciding which agent to use."""
    selected_agent: str
    confidence: float  # 0.0 to 1.0
    reason: str
    alternative_agents: List[str] = field(default_factory=list)
    routing_method: Literal["rules", "llm", "experience"] = "rules"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected_agent": self.selected_agent,
            "confidence": self.confidence,
            "reason": self.reason,
            "alternative_agents": self.alternative_agents,
            "routing_method": self.routing_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingDecision":
        """Create from dictionary."""
        return cls(
            selected_agent=data["selected_agent"],
            confidence=data["confidence"],
            reason=data["reason"],
            alternative_agents=data.get("alternative_agents", []),
            routing_method=data.get("routing_method", "rules"),
        )


@dataclass
class OrchestrationStep:
    """A step in the orchestration trace."""
    step_id: str
    agent_name: str
    task: str
    routing_decision: RoutingDecision
    agent_result: Optional[ReActResult] = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "agent_name": self.agent_name,
            "task": self.task,
            "routing_decision": self.routing_decision.to_dict(),
            "agent_result": self.agent_result.trajectory.to_dict() if self.agent_result else None,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
        }


@dataclass
class OrchestrationTrace:
    """Complete trace of an orchestration session."""
    trace_id: str
    goal: str
    steps: List[OrchestrationStep] = field(default_factory=list)
    final_result: Optional[Any] = None
    success: bool = False
    total_time_ms: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    execution_mode: str = "sequential"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "final_result": self.final_result,
            "success": self.success,
            "total_time_ms": self.total_time_ms,
            "agents_used": self.agents_used,
            "execution_mode": self.execution_mode,
            "created_at": self.created_at,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class OrchestrationResult:
    """Result of orchestration."""
    trace: OrchestrationTrace
    success: bool
    final_answer: Optional[Any] = None
    agents_used: List[str] = field(default_factory=list)
    routing_decisions: List[RoutingDecision] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace": self.trace.to_dict(),
            "success": self.success,
            "final_answer": self.final_answer,
            "agents_used": self.agents_used,
            "routing_decisions": [r.to_dict() for r in self.routing_decisions],
        }


# =============================================================================
# Router Classes
# =============================================================================

class BaseRouter(ABC):
    """Abstract base for routing mechanisms."""

    @abstractmethod
    def route(
        self,
        task: str,
        available_agents: List[AgentSpec],
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Decide which agent should handle the task.

        Args:
            task: The task/goal to route
            available_agents: List of available agent specs
            context: Optional context from previous steps

        Returns:
            RoutingDecision with selected agent and confidence
        """
        pass


class RuleBasedRouter(BaseRouter):
    """
    Route based on keyword/domain matching rules.

    Fast, deterministic, no LLM calls.

    Example:
        >>> router = RuleBasedRouter()
        >>> router.add_domain_keywords("extraction", ["extract", "parse", "get"])
        >>> router.add_domain_keywords("calculation", ["calculate", "compute", "bmi"])
        >>> decision = router.route("Extract patient weight", agents)
    """

    def __init__(self):
        self._domain_keywords: Dict[str, List[str]] = {}
        self._capability_keywords: Dict[str, List[str]] = {}

    def add_domain_keywords(self, domain: str, keywords: List[str]) -> "RuleBasedRouter":
        """Associate keywords with a domain."""
        self._domain_keywords[domain.lower()] = [k.lower() for k in keywords]
        return self

    def add_capability_keywords(self, capability: str, keywords: List[str]) -> "RuleBasedRouter":
        """Associate keywords with a capability."""
        self._capability_keywords[capability.lower()] = [k.lower() for k in keywords]
        return self

    def route(
        self,
        task: str,
        available_agents: List[AgentSpec],
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Match task keywords to agent domains/capabilities."""
        task_lower = task.lower()
        task_words = set(re.findall(r'\w+', task_lower))

        # Score each agent based on keyword matches
        scores: List[Tuple[AgentSpec, float, str]] = []

        for agent in available_agents:
            score = 0.0
            match_reasons = []

            # Check domain keywords
            for domain in agent.domains:
                domain_lower = domain.lower()
                if domain_lower in task_lower:
                    score += 0.5
                    match_reasons.append(f"domain '{domain}' in task")

                # Check configured keywords for this domain
                if domain_lower in self._domain_keywords:
                    for keyword in self._domain_keywords[domain_lower]:
                        if keyword in task_lower:
                            score += 0.3
                            match_reasons.append(f"keyword '{keyword}' for domain '{domain}'")

            # Check capability keywords
            for cap in agent.capabilities:
                cap_lower = cap.lower()
                if cap_lower in task_lower:
                    score += 0.4
                    match_reasons.append(f"capability '{cap}' in task")

                if cap_lower in self._capability_keywords:
                    for keyword in self._capability_keywords[cap_lower]:
                        if keyword in task_lower:
                            score += 0.2
                            match_reasons.append(f"keyword '{keyword}' for capability '{cap}'")

            # Check ptool names in task
            for ptool in agent.available_ptools:
                ptool_words = set(re.findall(r'\w+', ptool.lower()))
                if ptool_words & task_words:
                    score += 0.3
                    match_reasons.append(f"ptool '{ptool}' mentioned")

            # Add priority bonus
            score += agent.priority * 0.1

            if score > 0:
                reason = "; ".join(match_reasons[:3])  # Top 3 reasons
                scores.append((agent, score, reason))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        if not scores:
            # No matches, pick first agent
            return RoutingDecision(
                selected_agent=available_agents[0].name if available_agents else "",
                confidence=0.1,
                reason="No keyword matches, defaulting to first agent",
                alternative_agents=[a.name for a in available_agents[1:3]],
                routing_method="rules",
            )

        best_agent, best_score, best_reason = scores[0]
        max_possible_score = 3.0  # Rough max
        confidence = min(best_score / max_possible_score, 1.0)

        return RoutingDecision(
            selected_agent=best_agent.name,
            confidence=confidence,
            reason=best_reason,
            alternative_agents=[s[0].name for s in scores[1:3]],
            routing_method="rules",
        )


class LLMRouter(BaseRouter):
    """
    Use an LLM to analyze the task and select the best agent.

    More flexible but slower and costs tokens.

    Example:
        >>> router = LLMRouter(model="deepseek-v3-0324")
        >>> decision = router.route("Complex multi-step task...", agents)
    """

    def __init__(
        self,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
    ):
        self.model = model
        self.llm_backend = llm_backend or call_llm

    def route(
        self,
        task: str,
        available_agents: List[AgentSpec],
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Use LLM to analyze task and select agent."""
        # Build prompt
        agent_descriptions = "\n\n".join(
            a.format_for_prompt() for a in available_agents
        )

        prompt = f"""You are a task router. Given a task and a list of specialized agents, select the best agent to handle the task.

## Available Agents:
{agent_descriptions}

## Task:
{task}

## Context from previous steps:
{json.dumps(context, default=str) if context else "None"}

## Instructions:
1. Analyze the task requirements
2. Match requirements to agent capabilities
3. Select the single best agent

Respond in this exact JSON format:
{{
    "selected_agent": "agent_name",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation",
    "alternative_agents": ["second_best", "third_best"]
}}

Your response (JSON only):"""

        # Call LLM
        response = self.llm_backend(prompt, model=self.model)

        # Parse response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return RoutingDecision(
                    selected_agent=data.get("selected_agent", available_agents[0].name),
                    confidence=float(data.get("confidence", 0.7)),
                    reason=data.get("reason", "LLM routing"),
                    alternative_agents=data.get("alternative_agents", []),
                    routing_method="llm",
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: try to extract agent name from response
        for agent in available_agents:
            if agent.name.lower() in response.lower():
                return RoutingDecision(
                    selected_agent=agent.name,
                    confidence=0.5,
                    reason="LLM mentioned agent name",
                    alternative_agents=[],
                    routing_method="llm",
                )

        # Final fallback
        return RoutingDecision(
            selected_agent=available_agents[0].name if available_agents else "",
            confidence=0.3,
            reason="LLM response parsing failed, defaulting to first agent",
            alternative_agents=[],
            routing_method="llm",
        )


class ExperienceBasedRouter(BaseRouter):
    """
    Route based on historical performance (integrates with ExperienceStore).

    Learns which agents succeed on which types of tasks.

    Example:
        >>> router = ExperienceBasedRouter()
        >>> # After running tasks, record outcomes:
        >>> router.record_outcome("calculate bmi", "calculator", success=True)
        >>> # Future routing will prefer successful agents:
        >>> decision = router.route("calculate bmi for patient", agents)
    """

    def __init__(
        self,
        experience_store: Optional[ExperienceStore] = None,
        min_samples: int = 3,  # Minimum samples before trusting history
    ):
        self.experience_store = experience_store or ExperienceStore()
        self.min_samples = min_samples
        self._task_agent_history: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._load_history()

    def _get_history_path(self) -> Path:
        """Get path to history file."""
        return Path(os.path.expanduser("~/.orchestration_history/routing_history.json"))

    def _load_history(self) -> None:
        """Load history from disk."""
        path = self._get_history_path()
        if path.exists():
            try:
                with open(path) as f:
                    self._task_agent_history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._task_agent_history = {}

    def _save_history(self) -> None:
        """Save history to disk."""
        path = self._get_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._task_agent_history, f, indent=2)

    def _normalize_task(self, task: str) -> str:
        """Normalize task to key for history lookup."""
        # Use first 5 words as pattern key
        words = task.lower().split()[:5]
        return " ".join(words)

    def record_outcome(self, task: str, agent_name: str, success: bool) -> None:
        """Record outcome for learning."""
        task_key = self._normalize_task(task)

        if task_key not in self._task_agent_history:
            self._task_agent_history[task_key] = {}

        if agent_name not in self._task_agent_history[task_key]:
            self._task_agent_history[task_key][agent_name] = {
                "total": 0,
                "successes": 0,
            }

        self._task_agent_history[task_key][agent_name]["total"] += 1
        if success:
            self._task_agent_history[task_key][agent_name]["successes"] += 1

        self._save_history()

    def route(
        self,
        task: str,
        available_agents: List[AgentSpec],
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Select agent based on historical success rates for similar tasks."""
        task_key = self._normalize_task(task)

        if task_key not in self._task_agent_history:
            # No history, low confidence
            return RoutingDecision(
                selected_agent=available_agents[0].name if available_agents else "",
                confidence=0.2,
                reason="No historical data for this task pattern",
                alternative_agents=[a.name for a in available_agents[1:3]],
                routing_method="experience",
            )

        # Score agents by success rate
        agent_scores: List[Tuple[str, float, int]] = []
        history = self._task_agent_history[task_key]

        for agent in available_agents:
            if agent.name in history:
                stats = history[agent.name]
                total = stats["total"]
                successes = stats["successes"]

                if total >= self.min_samples:
                    success_rate = successes / total
                    agent_scores.append((agent.name, success_rate, total))

        if not agent_scores:
            # Not enough history
            return RoutingDecision(
                selected_agent=available_agents[0].name if available_agents else "",
                confidence=0.2,
                reason="Not enough historical data (need min {} samples)".format(self.min_samples),
                alternative_agents=[a.name for a in available_agents[1:3]],
                routing_method="experience",
            )

        # Sort by success rate descending
        agent_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

        best_agent, best_rate, sample_count = agent_scores[0]

        return RoutingDecision(
            selected_agent=best_agent,
            confidence=best_rate * 0.9,  # Slight discount
            reason=f"Historical success rate: {best_rate:.1%} ({sample_count} samples)",
            alternative_agents=[s[0] for s in agent_scores[1:3]],
            routing_method="experience",
        )


class HybridRouter(BaseRouter):
    """
    Combines multiple routing strategies.

    1. Try rule-based first (fast, deterministic)
    2. Fall back to experience-based if uncertain
    3. Use LLM as final fallback

    Example:
        >>> router = HybridRouter(confidence_threshold=0.7)
        >>> decision = router.route("Complex task", agents)
    """

    def __init__(
        self,
        rule_router: Optional[RuleBasedRouter] = None,
        experience_router: Optional[ExperienceBasedRouter] = None,
        llm_router: Optional[LLMRouter] = None,
        confidence_threshold: float = 0.6,
    ):
        self.rule_router = rule_router or RuleBasedRouter()
        self.experience_router = experience_router or ExperienceBasedRouter()
        self.llm_router = llm_router or LLMRouter()
        self.confidence_threshold = confidence_threshold

        # Set up default keywords
        self._setup_default_keywords()

    def _setup_default_keywords(self) -> None:
        """Set up common keyword mappings."""
        # Domain keywords
        self.rule_router.add_domain_keywords("extraction", [
            "extract", "parse", "get", "find", "identify", "detect", "recognize"
        ])
        self.rule_router.add_domain_keywords("calculation", [
            "calculate", "compute", "bmi", "score", "formula", "math"
        ])
        self.rule_router.add_domain_keywords("medcalc", [
            "bmi", "egfr", "cha2ds2", "wells", "apache", "sofa", "meld",
            "creatinine", "clearance", "dosage", "clinical"
        ])
        self.rule_router.add_domain_keywords("analysis", [
            "analyze", "review", "assess", "evaluate", "examine"
        ])
        self.rule_router.add_domain_keywords("generation", [
            "generate", "create", "write", "compose", "draft"
        ])

        # Capability keywords
        self.rule_router.add_capability_keywords("reasoning", [
            "think", "reason", "deduce", "infer", "conclude"
        ])
        self.rule_router.add_capability_keywords("planning", [
            "plan", "schedule", "organize", "prioritize"
        ])

    def route(
        self,
        task: str,
        available_agents: List[AgentSpec],
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Route using cascading strategy: rules -> experience -> LLM."""
        # Try rule-based first
        rule_decision = self.rule_router.route(task, available_agents, context)
        if rule_decision.confidence >= self.confidence_threshold:
            return rule_decision

        # Try experience-based
        exp_decision = self.experience_router.route(task, available_agents, context)
        if exp_decision.confidence >= self.confidence_threshold:
            return exp_decision

        # Fall back to LLM
        llm_decision = self.llm_router.route(task, available_agents, context)

        # Return the decision with highest confidence
        decisions = [rule_decision, exp_decision, llm_decision]
        best_decision = max(decisions, key=lambda d: d.confidence)

        # Enhance reason with routing path info
        if best_decision == llm_decision:
            best_decision.reason = f"[LLM fallback] {best_decision.reason}"
        elif best_decision == exp_decision:
            best_decision.reason = f"[Experience] {best_decision.reason}"
        else:
            best_decision.reason = f"[Rules] {best_decision.reason}"

        return best_decision

    def record_outcome(self, task: str, agent_name: str, success: bool) -> None:
        """Record outcome for experience-based learning."""
        self.experience_router.record_outcome(task, agent_name, success)


# =============================================================================
# Orchestration Store
# =============================================================================

class OrchestrationStore:
    """
    Persistent storage for orchestration traces.

    Storage structure:
    ~/.orchestration_traces/
    ├── traces/
    │   └── trace_{id}.json
    ├── by_agent/
    │   └── {agent_name}.jsonl
    └── routing_history.jsonl
    """

    def __init__(self, path: str = "~/.orchestration_traces"):
        self.base_path = Path(os.path.expanduser(path))
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        (self.base_path / "traces").mkdir(parents=True, exist_ok=True)
        (self.base_path / "by_agent").mkdir(parents=True, exist_ok=True)

    def store_trace(self, trace: OrchestrationTrace) -> None:
        """Store an orchestration trace."""
        # Store full trace
        trace_path = self.base_path / "traces" / f"trace_{trace.trace_id}.json"
        with open(trace_path, "w") as f:
            f.write(trace.to_json())

        # Index by agent
        for agent_name in trace.agents_used:
            agent_path = self.base_path / "by_agent" / f"{agent_name}.jsonl"
            with open(agent_path, "a") as f:
                summary = {
                    "trace_id": trace.trace_id,
                    "goal": trace.goal,
                    "success": trace.success,
                    "created_at": trace.created_at,
                }
                f.write(json.dumps(summary) + "\n")

        # Store routing decisions
        routing_path = self.base_path / "routing_history.jsonl"
        for step in trace.steps:
            with open(routing_path, "a") as f:
                entry = {
                    "trace_id": trace.trace_id,
                    "task": step.task,
                    "routing_decision": step.routing_decision.to_dict(),
                    "success": step.status == "completed",
                    "timestamp": trace.created_at,
                }
                f.write(json.dumps(entry) + "\n")

    def get_trace(self, trace_id: str) -> Optional[OrchestrationTrace]:
        """Retrieve a trace by ID."""
        trace_path = self.base_path / "traces" / f"trace_{trace_id}.json"
        if not trace_path.exists():
            return None

        with open(trace_path) as f:
            data = json.load(f)

        # Reconstruct trace
        trace = OrchestrationTrace(
            trace_id=data["trace_id"],
            goal=data["goal"],
            final_result=data.get("final_result"),
            success=data["success"],
            total_time_ms=data.get("total_time_ms", 0.0),
            agents_used=data.get("agents_used", []),
            execution_mode=data.get("execution_mode", "sequential"),
            created_at=data.get("created_at", ""),
        )

        # Reconstruct steps
        for step_data in data.get("steps", []):
            step = OrchestrationStep(
                step_id=step_data["step_id"],
                agent_name=step_data["agent_name"],
                task=step_data["task"],
                routing_decision=RoutingDecision.from_dict(step_data["routing_decision"]),
                status=step_data["status"],
                start_time=step_data.get("start_time"),
                end_time=step_data.get("end_time"),
                error=step_data.get("error"),
            )
            trace.steps.append(step)

        return trace

    def get_agent_history(self, agent_name: str, limit: int = 100) -> List[Dict]:
        """Get execution history for a specific agent."""
        agent_path = self.base_path / "by_agent" / f"{agent_name}.jsonl"
        if not agent_path.exists():
            return []

        history = []
        with open(agent_path) as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))

        return history[-limit:]

    def get_routing_history(self, limit: int = 100) -> List[Dict]:
        """Get history of routing decisions for analysis."""
        routing_path = self.base_path / "routing_history.jsonl"
        if not routing_path.exists():
            return []

        history = []
        with open(routing_path) as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))

        return history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        traces_dir = self.base_path / "traces"
        trace_files = list(traces_dir.glob("trace_*.json"))

        total_traces = len(trace_files)
        successful = 0
        agents_used = {}

        for trace_file in trace_files:
            with open(trace_file) as f:
                data = json.load(f)
            if data.get("success"):
                successful += 1
            for agent in data.get("agents_used", []):
                agents_used[agent] = agents_used.get(agent, 0) + 1

        return {
            "total_traces": total_traces,
            "successful_traces": successful,
            "success_rate": successful / total_traces if total_traces > 0 else 0.0,
            "agents_used": agents_used,
        }


# Global store
_ORCHESTRATION_STORE: Optional[OrchestrationStore] = None


def get_orchestration_store() -> OrchestrationStore:
    """Get the global orchestration store."""
    global _ORCHESTRATION_STORE
    if _ORCHESTRATION_STORE is None:
        _ORCHESTRATION_STORE = OrchestrationStore()
    return _ORCHESTRATION_STORE


def set_orchestration_store(store: OrchestrationStore) -> None:
    """Set the global orchestration store."""
    global _ORCHESTRATION_STORE
    _ORCHESTRATION_STORE = store


# =============================================================================
# Agent Orchestrator
# =============================================================================

class AgentOrchestrator:
    """
    Orchestrates multiple specialized ReActAgents.

    Features:
    - Agent registry with domain specializations
    - Configurable routing (rules, LLM, experience-based)
    - Sequential, parallel, and pipeline execution modes
    - Orchestration trace collection for analysis
    - Integration with existing TraceStore and ExperienceStore

    Example:
        >>> orchestrator = AgentOrchestrator()
        >>>
        >>> # Register specialized agents
        >>> orchestrator.register_agent(AgentSpec(
        ...     name="extractor",
        ...     description="Extracts structured data from text",
        ...     domains=["extraction", "parsing"],
        ...     available_ptools=["extract_values", "parse_json", "extract_entities"]
        ... ))
        >>> orchestrator.register_agent(AgentSpec(
        ...     name="calculator",
        ...     description="Performs medical calculations",
        ...     domains=["medcalc", "math"],
        ...     available_ptools=["calculate", "unit_convert", "validate_range"]
        ... ))
        >>>
        >>> # Run a task
        >>> result = orchestrator.run("Extract patient weight and calculate BMI")
    """

    def __init__(
        self,
        router: Optional[BaseRouter] = None,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        max_agents_per_task: int = 3,
        store_traces: bool = True,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
        echo: bool = False,
    ):
        self.router = router or HybridRouter()
        self.execution_mode = execution_mode
        self.max_agents_per_task = max_agents_per_task
        self.store_traces = store_traces
        self.model = model
        self.llm_backend = llm_backend
        self.echo = echo

        # Agent registry
        self._agents: Dict[str, AgentSpec] = {}
        self._agent_instances: Dict[str, ReActAgent] = {}  # Lazily created

        # Trace storage
        self._store: Optional[OrchestrationStore] = None
        if store_traces:
            self._store = get_orchestration_store()

    # =========================================================================
    # Agent Management
    # =========================================================================

    def register_agent(self, spec: AgentSpec) -> "AgentOrchestrator":
        """Register a specialized agent."""
        self._agents[spec.name] = spec
        # Clear cached instance if it exists
        self._agent_instances.pop(spec.name, None)
        return self

    def unregister_agent(self, name: str) -> Optional[AgentSpec]:
        """Remove an agent from the registry."""
        spec = self._agents.pop(name, None)
        self._agent_instances.pop(name, None)
        return spec

    def get_agent(self, name: str) -> Optional[ReActAgent]:
        """Get or create a ReActAgent instance for the spec."""
        if name not in self._agents:
            return None

        if name not in self._agent_instances:
            spec = self._agents[name]

            # Get ptool specs for this agent
            registry = get_registry()
            available_ptools = []
            for ptool_name in spec.available_ptools:
                ptool_spec = registry.get(ptool_name)
                if ptool_spec:
                    available_ptools.append(ptool_spec)

            # Create agent instance
            self._agent_instances[name] = ReActAgent(
                available_ptools=available_ptools if available_ptools else None,
                model=spec.model,
                max_steps=spec.max_steps,
                llm_backend=self.llm_backend,
                echo=self.echo,
            )

        return self._agent_instances[name]

    def list_agents(self) -> List[AgentSpec]:
        """List all registered agents."""
        return list(self._agents.values())

    def get_agents_for_domain(self, domain: str) -> List[AgentSpec]:
        """Get agents that handle a specific domain."""
        domain_lower = domain.lower()
        return [
            spec for spec in self._agents.values()
            if any(d.lower() == domain_lower for d in spec.domains)
        ]

    # =========================================================================
    # Execution
    # =========================================================================

    def run(self, goal: str, context: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """
        Run the orchestrator on a goal.

        1. Route to appropriate agent(s)
        2. Execute agent(s) based on execution_mode
        3. Collect and aggregate results
        4. Store orchestration trace
        """
        import time
        start_time = time.time()

        trace_id = str(uuid.uuid4())[:8]
        trace = OrchestrationTrace(
            trace_id=trace_id,
            goal=goal,
            execution_mode=self.execution_mode.value,
        )

        available_agents = list(self._agents.values())
        if not available_agents:
            return OrchestrationResult(
                trace=trace,
                success=False,
                final_answer="No agents registered",
                agents_used=[],
                routing_decisions=[],
            )

        # Route to agent
        routing_decision = self.router.route(goal, available_agents, context)

        if self.echo:
            print(f"[Orchestrator] Routing to: {routing_decision.selected_agent}")
            print(f"[Orchestrator] Confidence: {routing_decision.confidence:.2f}")
            print(f"[Orchestrator] Reason: {routing_decision.reason}")

        # Create step
        step = OrchestrationStep(
            step_id=str(uuid.uuid4())[:8],
            agent_name=routing_decision.selected_agent,
            task=goal,
            routing_decision=routing_decision,
            status="running",
            start_time=datetime.now().isoformat(),
        )
        trace.steps.append(step)

        # Execute agent
        agent = self.get_agent(routing_decision.selected_agent)
        if agent is None:
            step.status = "failed"
            step.error = f"Agent '{routing_decision.selected_agent}' not found"
            step.end_time = datetime.now().isoformat()
            return OrchestrationResult(
                trace=trace,
                success=False,
                final_answer=step.error,
                agents_used=[],
                routing_decisions=[routing_decision],
            )

        try:
            result = agent.run(goal)
            step.agent_result = result
            step.status = "completed" if result.success else "failed"
            step.end_time = datetime.now().isoformat()

            trace.success = result.success
            trace.final_result = result.answer
            trace.agents_used = [routing_decision.selected_agent]

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.end_time = datetime.now().isoformat()
            trace.success = False

        # Record outcome for learning
        if isinstance(self.router, HybridRouter):
            self.router.record_outcome(goal, routing_decision.selected_agent, trace.success)

        # Calculate total time
        trace.total_time_ms = (time.time() - start_time) * 1000

        # Store trace
        if self.store_traces and self._store:
            self._store.store_trace(trace)

        return OrchestrationResult(
            trace=trace,
            success=trace.success,
            final_answer=trace.final_result,
            agents_used=trace.agents_used,
            routing_decisions=[routing_decision],
        )

    def run_sequential(
        self,
        tasks: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Run multiple tasks sequentially, passing context between them."""
        import time
        start_time = time.time()

        trace_id = str(uuid.uuid4())[:8]
        trace = OrchestrationTrace(
            trace_id=trace_id,
            goal="; ".join(tasks),
            execution_mode="sequential",
        )

        context = context or {}
        routing_decisions = []
        all_agents_used = set()
        final_answer = None
        all_success = True

        for i, task in enumerate(tasks):
            available_agents = list(self._agents.values())
            routing_decision = self.router.route(task, available_agents, context)
            routing_decisions.append(routing_decision)

            step = OrchestrationStep(
                step_id=str(uuid.uuid4())[:8],
                agent_name=routing_decision.selected_agent,
                task=task,
                routing_decision=routing_decision,
                status="running",
                start_time=datetime.now().isoformat(),
            )
            trace.steps.append(step)

            agent = self.get_agent(routing_decision.selected_agent)
            if agent is None:
                step.status = "failed"
                step.error = f"Agent '{routing_decision.selected_agent}' not found"
                step.end_time = datetime.now().isoformat()
                all_success = False
                continue

            try:
                result = agent.run(task)
                step.agent_result = result
                step.status = "completed" if result.success else "failed"
                step.end_time = datetime.now().isoformat()

                all_agents_used.add(routing_decision.selected_agent)

                # Pass result to next task's context
                context[f"step_{i}"] = result.answer
                context[f"step_{i}_success"] = result.success
                final_answer = result.answer

                if not result.success:
                    all_success = False

                # Record for learning
                if isinstance(self.router, HybridRouter):
                    self.router.record_outcome(task, routing_decision.selected_agent, result.success)

            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                step.end_time = datetime.now().isoformat()
                all_success = False

        trace.success = all_success
        trace.final_result = final_answer
        trace.agents_used = list(all_agents_used)
        trace.total_time_ms = (time.time() - start_time) * 1000

        if self.store_traces and self._store:
            self._store.store_trace(trace)

        return OrchestrationResult(
            trace=trace,
            success=all_success,
            final_answer=final_answer,
            agents_used=list(all_agents_used),
            routing_decisions=routing_decisions,
        )

    def run_parallel(
        self,
        tasks: List[str],
        context: Optional[Dict[str, Any]] = None,
        max_workers: int = 3,
    ) -> OrchestrationResult:
        """Run multiple tasks in parallel (using ThreadPoolExecutor)."""
        import time
        start_time = time.time()

        trace_id = str(uuid.uuid4())[:8]
        trace = OrchestrationTrace(
            trace_id=trace_id,
            goal="; ".join(tasks),
            execution_mode="parallel",
        )

        context = context or {}
        routing_decisions = []
        all_agents_used = set()
        results = []

        def run_task(task: str) -> Tuple[str, OrchestrationStep, Optional[ReActResult]]:
            """Run a single task and return results."""
            available_agents = list(self._agents.values())
            routing_decision = self.router.route(task, available_agents, context)

            step = OrchestrationStep(
                step_id=str(uuid.uuid4())[:8],
                agent_name=routing_decision.selected_agent,
                task=task,
                routing_decision=routing_decision,
                status="running",
                start_time=datetime.now().isoformat(),
            )

            agent = self.get_agent(routing_decision.selected_agent)
            if agent is None:
                step.status = "failed"
                step.error = f"Agent '{routing_decision.selected_agent}' not found"
                step.end_time = datetime.now().isoformat()
                return task, step, None

            try:
                result = agent.run(task)
                step.agent_result = result
                step.status = "completed" if result.success else "failed"
                step.end_time = datetime.now().isoformat()
                return task, step, result
            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                step.end_time = datetime.now().isoformat()
                return task, step, None

        # Run tasks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}

            for future in as_completed(futures):
                task, step, result = future.result()
                trace.steps.append(step)
                routing_decisions.append(step.routing_decision)

                if result:
                    all_agents_used.add(step.agent_name)
                    results.append(result)

                    if isinstance(self.router, HybridRouter):
                        self.router.record_outcome(task, step.agent_name, result.success)

        # Aggregate results
        all_success = all(r.success for r in results) if results else False
        final_answers = [r.answer for r in results if r.answer]

        trace.success = all_success
        trace.final_result = final_answers if len(final_answers) > 1 else (final_answers[0] if final_answers else None)
        trace.agents_used = list(all_agents_used)
        trace.total_time_ms = (time.time() - start_time) * 1000

        if self.store_traces and self._store:
            self._store.store_trace(trace)

        return OrchestrationResult(
            trace=trace,
            success=all_success,
            final_answer=trace.final_result,
            agents_used=list(all_agents_used),
            routing_decisions=routing_decisions,
        )

    def run_pipeline(
        self,
        stages: List[Tuple[str, str]],  # List of (agent_name, task_template)
        initial_input: Any = None,
    ) -> OrchestrationResult:
        """
        Run a predefined pipeline of agent stages.

        Output of each stage becomes input to the next.
        Task templates can use {prev_result} placeholder.

        Example:
            >>> result = orchestrator.run_pipeline([
            ...     ("extractor", "Extract values from: {input}"),
            ...     ("calculator", "Calculate BMI using: {prev_result}"),
            ... ], initial_input="Patient weighs 70kg...")
        """
        import time
        start_time = time.time()

        trace_id = str(uuid.uuid4())[:8]
        trace = OrchestrationTrace(
            trace_id=trace_id,
            goal=f"Pipeline: {' -> '.join(s[0] for s in stages)}",
            execution_mode="pipeline",
        )

        routing_decisions = []
        all_agents_used = set()
        current_input = initial_input
        final_answer = None
        all_success = True

        for i, (agent_name, task_template) in enumerate(stages):
            # Format task with previous result
            task = task_template.format(
                input=initial_input,
                prev_result=current_input,
            )

            # Create routing decision (predetermined agent)
            routing_decision = RoutingDecision(
                selected_agent=agent_name,
                confidence=1.0,
                reason=f"Pipeline stage {i+1}",
                routing_method="rules",
            )
            routing_decisions.append(routing_decision)

            step = OrchestrationStep(
                step_id=str(uuid.uuid4())[:8],
                agent_name=agent_name,
                task=task,
                routing_decision=routing_decision,
                status="running",
                start_time=datetime.now().isoformat(),
            )
            trace.steps.append(step)

            agent = self.get_agent(agent_name)
            if agent is None:
                step.status = "failed"
                step.error = f"Agent '{agent_name}' not found"
                step.end_time = datetime.now().isoformat()
                all_success = False
                break

            try:
                result = agent.run(task)
                step.agent_result = result
                step.status = "completed" if result.success else "failed"
                step.end_time = datetime.now().isoformat()

                all_agents_used.add(agent_name)
                current_input = result.answer
                final_answer = result.answer

                if not result.success:
                    all_success = False
                    break

            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                step.end_time = datetime.now().isoformat()
                all_success = False
                break

        trace.success = all_success
        trace.final_result = final_answer
        trace.agents_used = list(all_agents_used)
        trace.total_time_ms = (time.time() - start_time) * 1000

        if self.store_traces and self._store:
            self._store.store_trace(trace)

        return OrchestrationResult(
            trace=trace,
            success=all_success,
            final_answer=final_answer,
            agents_used=list(all_agents_used),
            routing_decisions=routing_decisions,
        )

    # =========================================================================
    # Advanced Features
    # =========================================================================

    def decompose_task(self, complex_goal: str) -> List[str]:
        """
        Use LLM to decompose a complex goal into subtasks.

        Returns list of subtasks that can be routed to different agents.
        """
        # Get agent descriptions for context
        agent_info = "\n".join(
            f"- {spec.name}: {spec.description} (domains: {', '.join(spec.domains)})"
            for spec in self._agents.values()
        )

        prompt = f"""Decompose this complex task into smaller subtasks that can be handled by specialized agents.

## Available Agents:
{agent_info}

## Complex Task:
{complex_goal}

## Instructions:
1. Break down the task into 2-5 sequential subtasks
2. Each subtask should be handleable by one of the available agents
3. Order matters - later tasks may depend on earlier results

Respond with a JSON array of subtask strings:
["subtask 1", "subtask 2", "subtask 3"]

Your response (JSON array only):"""

        backend = self.llm_backend or call_llm
        response = backend(prompt, model=self.model)

        # Parse response
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                subtasks = json.loads(json_match.group())
                if isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks):
                    return subtasks
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: return original task
        return [complex_goal]


# =============================================================================
# Convenience Functions
# =============================================================================

def orchestrate(
    goal: str,
    agents: Optional[List[AgentSpec]] = None,
    router: Optional[BaseRouter] = None,
    **kwargs,
) -> OrchestrationResult:
    """
    Quick orchestration with default settings.

    Example:
        >>> result = orchestrate("Calculate BMI", agents=[extractor_spec, calculator_spec])
    """
    orchestrator = AgentOrchestrator(router=router, **kwargs)

    if agents:
        for spec in agents:
            orchestrator.register_agent(spec)

    return orchestrator.run(goal)
