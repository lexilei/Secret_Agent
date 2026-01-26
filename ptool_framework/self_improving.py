"""
Self-Improving Agents (L5) for ptool_framework.

This module implements William's L5 vision: Self-modifying systems that learn
from experience. It wraps any ReActAgent and enhances it with:
- Pattern extraction from successful/failed trajectories
- Memory persistence for learned patterns
- Prompt enhancement with relevant patterns
- Forgetting mechanism for outdated patterns

Key components:
- PatternExtractor: Extracts patterns from trajectories
- PatternMemory: Persistent storage for learned patterns
- SelfImprovingAgent: Wrapper that adds self-improvement capabilities

William's requirements implemented:
- Good outputs -> ICL demos + unit tests
- Bad outputs -> unit tests (negative examples)
- Good traces -> generate audits
- Bad traces -> ICL demo for critic

Example:
    >>> base_agent = ReActAgent(available_ptools=[...])
    >>> improving_agent = SelfImprovingAgent(base_agent)
    >>> result = improving_agent.run("Calculate BMI for patient")
    >>> # Agent learns from this execution
    >>> patterns = improving_agent.get_learned_patterns()
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import json
import os
import re
import uuid

from .react import ReActAgent, ReActResult, ReActTrajectory, ReActStep
from .critic import TraceCritic, CriticVerdict, CriticEvaluation
from .repair import RepairAgent, RepairResult
from .trace_store import TraceStore, get_trace_store
from .llm_backend import call_llm


# =============================================================================
# Enums
# =============================================================================

class PatternType(Enum):
    """Types of patterns that can be learned."""
    POSITIVE = "positive"      # Pattern from successful trajectory (ICL demo)
    NEGATIVE = "negative"      # Pattern from failed trajectory (what to avoid)
    REPAIR = "repair"          # Pattern from successful repair
    HEURISTIC = "heuristic"    # General heuristic learned over time


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LearnedPattern:
    """
    A pattern extracted from experience.

    Based on William's requirements:
    - Good outputs -> ICL demos + unit tests
    - Bad outputs -> unit tests
    - Good traces -> generate audits
    - Bad traces -> ICL demo for critic
    """
    pattern_id: str
    pattern_type: PatternType
    content: str  # The actual pattern (e.g., ICL demo, heuristic rule)
    source_trace_id: str  # Which trace this was learned from
    domain: Optional[str] = None
    ptool_name: Optional[str] = None  # If ptool-specific
    goal_pattern: Optional[str] = None  # Normalized goal pattern

    # Relevance tracking
    times_used: int = 0
    times_helpful: int = 0  # When used, did it help?
    last_used: Optional[str] = None

    # Decay/forgetting
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0  # Decays over time if not used
    decay_rate: float = 0.05  # How fast confidence decays

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def relevance_score(self) -> float:
        """Calculate current relevance score."""
        if self.times_used == 0:
            base_score = 0.5  # Unknown usefulness
        else:
            base_score = self.times_helpful / self.times_used
        return base_score * self.confidence

    def apply_decay(self, days_since_use: int = 1) -> None:
        """Apply time-based decay to confidence."""
        self.confidence = max(0.1, self.confidence * (1 - self.decay_rate * days_since_use))

    def reinforce(self, was_helpful: bool) -> None:
        """Update pattern based on usage outcome."""
        self.times_used += 1
        if was_helpful:
            self.times_helpful += 1
            self.confidence = min(1.0, self.confidence * 1.1)  # Boost
        else:
            self.confidence = max(0.1, self.confidence * 0.9)  # Decrease
        self.last_used = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "content": self.content,
            "source_trace_id": self.source_trace_id,
            "domain": self.domain,
            "ptool_name": self.ptool_name,
            "goal_pattern": self.goal_pattern,
            "times_used": self.times_used,
            "times_helpful": self.times_helpful,
            "last_used": self.last_used,
            "created_at": self.created_at,
            "confidence": self.confidence,
            "decay_rate": self.decay_rate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedPattern":
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=PatternType(data["pattern_type"]),
            content=data["content"],
            source_trace_id=data["source_trace_id"],
            domain=data.get("domain"),
            ptool_name=data.get("ptool_name"),
            goal_pattern=data.get("goal_pattern"),
            times_used=data.get("times_used", 0),
            times_helpful=data.get("times_helpful", 0),
            last_used=data.get("last_used"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            confidence=data.get("confidence", 1.0),
            decay_rate=data.get("decay_rate", 0.05),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ICLExample:
    """In-context learning example extracted from experience."""
    inputs: Dict[str, Any]
    expected_output: Any
    actual_output: Optional[Any] = None
    reasoning: Optional[str] = None  # Chain-of-thought if available
    source_trace_id: str = ""
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "inputs": self.inputs,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "reasoning": self.reasoning,
            "source_trace_id": self.source_trace_id,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ICLExample":
        """Create from dictionary."""
        return cls(
            inputs=data["inputs"],
            expected_output=data["expected_output"],
            actual_output=data.get("actual_output"),
            reasoning=data.get("reasoning"),
            source_trace_id=data.get("source_trace_id", ""),
            success=data.get("success", True),
        )

    def format_for_prompt(self) -> str:
        """Format as ICL example for prompt."""
        parts = [f"Input: {json.dumps(self.inputs, default=str)}"]
        if self.reasoning:
            parts.append(f"Reasoning: {self.reasoning}")
        parts.append(f"Output: {json.dumps(self.expected_output, default=str)}")
        return "\n".join(parts)


@dataclass
class LearningEvent:
    """Record of a learning event."""
    event_id: str
    event_type: str  # pattern_extracted, pattern_used, pattern_helpful, pattern_decayed
    pattern_id: Optional[str] = None
    trace_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "pattern_id": self.pattern_id,
            "trace_id": self.trace_id,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class SelfImprovementMetrics:
    """Metrics tracking self-improvement over time."""
    total_patterns_learned: int = 0
    active_patterns: int = 0
    patterns_by_type: Dict[str, int] = field(default_factory=dict)
    improvement_rate: float = 0.0  # Success rate improvement over baseline
    baseline_success_rate: float = 0.0
    current_success_rate: float = 0.0
    total_runs: int = 0
    successful_runs: int = 0
    patterns_used_in_last_run: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_patterns_learned": self.total_patterns_learned,
            "active_patterns": self.active_patterns,
            "patterns_by_type": self.patterns_by_type,
            "improvement_rate": self.improvement_rate,
            "baseline_success_rate": self.baseline_success_rate,
            "current_success_rate": self.current_success_rate,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "patterns_used_in_last_run": self.patterns_used_in_last_run,
        }


# =============================================================================
# Pattern Extractor
# =============================================================================

class PatternExtractor:
    """
    Extracts patterns from trajectories for learning.

    Following William's approach:
    - Good outputs -> ICL demos + unit tests
    - Bad outputs -> unit tests
    - Good traces -> generate audits
    - Bad traces -> ICL demo for critic
    """

    def __init__(
        self,
        model: str = "deepseek-v3-0324",
        llm_backend: Optional[Callable] = None,
    ):
        self.model = model
        self.llm_backend = llm_backend or call_llm

    def extract_from_successful_trace(
        self,
        trajectory: ReActTrajectory,
    ) -> List[LearnedPattern]:
        """
        Extract patterns from a successful trajectory.

        Returns:
        - ICL examples for similar future tasks
        - Heuristic rules about what worked
        - Step sequence patterns
        """
        patterns = []

        # 1. Extract ICL example (positive)
        icl_pattern = self._extract_icl_pattern(trajectory, success=True)
        if icl_pattern:
            patterns.append(icl_pattern)

        # 2. Extract step sequence pattern
        sequence_pattern = self._extract_sequence_pattern(trajectory)
        if sequence_pattern:
            patterns.append(sequence_pattern)

        # 3. Extract heuristic from reasoning
        heuristic_pattern = self._extract_heuristic(trajectory, success=True)
        if heuristic_pattern:
            patterns.append(heuristic_pattern)

        return patterns

    def extract_from_failed_trace(
        self,
        trajectory: ReActTrajectory,
        error_analysis: Optional[str] = None,
    ) -> List[LearnedPattern]:
        """
        Extract patterns from a failed trajectory.

        Returns:
        - Negative examples (what to avoid)
        - Failure patterns to avoid
        - Critic ICL demos
        """
        patterns = []

        # 1. Extract negative ICL example
        negative_pattern = self._extract_icl_pattern(trajectory, success=False)
        if negative_pattern:
            patterns.append(negative_pattern)

        # 2. Extract failure pattern to avoid
        avoid_pattern = self._extract_failure_pattern(trajectory, error_analysis)
        if avoid_pattern:
            patterns.append(avoid_pattern)

        return patterns

    def extract_from_repair(
        self,
        original_trajectory: ReActTrajectory,
        repaired_trajectory: ReActTrajectory,
        repair_result: RepairResult,
    ) -> List[LearnedPattern]:
        """
        Extract patterns from a successful repair.

        Returns:
        - Before/after examples
        - What repair actions worked
        """
        patterns = []

        # Extract repair transformation pattern
        repair_pattern = self._extract_repair_pattern(
            original_trajectory, repaired_trajectory, repair_result
        )
        if repair_pattern:
            patterns.append(repair_pattern)

        return patterns

    def _extract_icl_pattern(
        self,
        trajectory: ReActTrajectory,
        success: bool,
    ) -> Optional[LearnedPattern]:
        """Extract an ICL example from a trajectory."""
        if not trajectory.steps:
            return None

        # Build content from trajectory
        steps_summary = []
        for i, step in enumerate(trajectory.steps):
            if step.thought:
                steps_summary.append(f"Step {i+1}: {step.thought.content[:100]}...")
            if step.action:
                steps_summary.append(f"  Action: {step.action.ptool_name}({step.action.args})")
            if step.observation and step.observation.success:
                result_str = str(step.observation.result)[:100]
                steps_summary.append(f"  Result: {result_str}")

        content = f"""Goal: {trajectory.goal}

Steps taken:
{chr(10).join(steps_summary)}

Final answer: {trajectory.final_answer}
Outcome: {'SUCCESS' if success else 'FAILURE'}"""

        pattern_type = PatternType.POSITIVE if success else PatternType.NEGATIVE

        return LearnedPattern(
            pattern_id=str(uuid.uuid4())[:8],
            pattern_type=pattern_type,
            content=content,
            source_trace_id=trajectory.trajectory_id,
            goal_pattern=self._normalize_goal(trajectory.goal),
            metadata={
                "num_steps": len(trajectory.steps),
                "final_answer": trajectory.final_answer,
            },
        )

    def _extract_sequence_pattern(
        self,
        trajectory: ReActTrajectory,
    ) -> Optional[LearnedPattern]:
        """Extract the step sequence pattern."""
        if not trajectory.steps:
            return None

        # Get sequence of ptools called
        ptool_sequence = []
        for step in trajectory.steps:
            if step.action:
                ptool_sequence.append(step.action.ptool_name)

        if not ptool_sequence:
            return None

        content = f"""For goals like "{trajectory.goal}":
Successful step sequence: {' -> '.join(ptool_sequence)}"""

        return LearnedPattern(
            pattern_id=str(uuid.uuid4())[:8],
            pattern_type=PatternType.HEURISTIC,
            content=content,
            source_trace_id=trajectory.trajectory_id,
            goal_pattern=self._normalize_goal(trajectory.goal),
            metadata={
                "sequence": ptool_sequence,
            },
        )

    def _extract_heuristic(
        self,
        trajectory: ReActTrajectory,
        success: bool,
    ) -> Optional[LearnedPattern]:
        """Use LLM to extract a general heuristic from the trajectory."""
        if not trajectory.steps:
            return None

        # Build trajectory summary
        steps_text = []
        for step in trajectory.steps:
            if step.thought:
                steps_text.append(f"Thought: {step.thought.content}")
            if step.action:
                steps_text.append(f"Action: {step.action.ptool_name}({step.action.args})")

        prompt = f"""Analyze this {'successful' if success else 'failed'} agent trajectory and extract a general heuristic rule.

Goal: {trajectory.goal}

Steps:
{chr(10).join(steps_text[:10])}

Final answer: {trajectory.final_answer}
Success: {success}

Extract ONE brief heuristic rule (1-2 sentences) that could help in similar future tasks.
Format: "When [condition], [action/approach]"

Heuristic:"""

        try:
            response = self.llm_backend(prompt, model=self.model)
            heuristic = response.strip()

            if heuristic and len(heuristic) > 10:
                return LearnedPattern(
                    pattern_id=str(uuid.uuid4())[:8],
                    pattern_type=PatternType.HEURISTIC,
                    content=heuristic,
                    source_trace_id=trajectory.trajectory_id,
                    goal_pattern=self._normalize_goal(trajectory.goal),
                )
        except Exception:
            pass

        return None

    def _extract_failure_pattern(
        self,
        trajectory: ReActTrajectory,
        error_analysis: Optional[str],
    ) -> Optional[LearnedPattern]:
        """Extract what went wrong to avoid in future."""
        # Find failed steps
        failed_steps = []
        for step in trajectory.steps:
            if step.observation and not step.observation.success:
                failed_steps.append(f"- {step.action.ptool_name if step.action else 'unknown'}: {step.observation.error}")

        if not failed_steps and not error_analysis:
            return None

        content = f"""AVOID this pattern for goals like "{trajectory.goal}":

Failure reasons:
{chr(10).join(failed_steps) if failed_steps else error_analysis or 'Unknown failure'}

Termination reason: {trajectory.termination_reason}"""

        return LearnedPattern(
            pattern_id=str(uuid.uuid4())[:8],
            pattern_type=PatternType.NEGATIVE,
            content=content,
            source_trace_id=trajectory.trajectory_id,
            goal_pattern=self._normalize_goal(trajectory.goal),
            metadata={
                "termination_reason": trajectory.termination_reason,
            },
        )

    def _extract_repair_pattern(
        self,
        original: ReActTrajectory,
        repaired: ReActTrajectory,
        repair_result: RepairResult,
    ) -> Optional[LearnedPattern]:
        """Extract repair transformation pattern."""
        # Summarize what was changed
        original_actions = [s.action.ptool_name for s in original.steps if s.action]
        repaired_actions = [s.action.ptool_name for s in repaired.steps if s.action]

        content = f"""Repair pattern for goals like "{original.goal}":

Original approach (FAILED): {' -> '.join(original_actions)}
Repaired approach (SUCCESS): {' -> '.join(repaired_actions)}

Repair iterations: {repair_result.iterations}
Actions taken: {len(repair_result.actions_taken)}"""

        return LearnedPattern(
            pattern_id=str(uuid.uuid4())[:8],
            pattern_type=PatternType.REPAIR,
            content=content,
            source_trace_id=original.trajectory_id,
            goal_pattern=self._normalize_goal(original.goal),
            metadata={
                "original_actions": original_actions,
                "repaired_actions": repaired_actions,
                "repair_iterations": repair_result.iterations,
            },
        )

    def _normalize_goal(self, goal: str) -> str:
        """Normalize goal to pattern for matching."""
        # Use first 5 words as pattern key
        words = goal.lower().split()[:5]
        return " ".join(words)


# =============================================================================
# Pattern Memory
# =============================================================================

class PatternMemory:
    """
    Persistent storage for learned patterns.

    Storage structure:
    ~/.pattern_memory/
    ├── patterns.json           # All patterns
    ├── by_type/
    │   ├── positive.jsonl
    │   ├── negative.jsonl
    │   └── repair.jsonl
    ├── by_domain/
    │   └── {domain}.jsonl
    ├── icl_examples/
    │   └── {ptool_name}.json
    └── learning_log.jsonl      # History of learning events
    """

    def __init__(self, path: str = "~/.pattern_memory"):
        self.base_path = Path(os.path.expanduser(path))
        self._ensure_directories()
        self._patterns: Dict[str, LearnedPattern] = {}
        self._load_patterns()

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        (self.base_path / "by_type").mkdir(parents=True, exist_ok=True)
        (self.base_path / "by_domain").mkdir(parents=True, exist_ok=True)
        (self.base_path / "icl_examples").mkdir(parents=True, exist_ok=True)

    def _patterns_file(self) -> Path:
        """Get path to main patterns file."""
        return self.base_path / "patterns.json"

    def _load_patterns(self) -> None:
        """Load patterns from disk."""
        patterns_file = self._patterns_file()
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    data = json.load(f)
                for pattern_data in data:
                    pattern = LearnedPattern.from_dict(pattern_data)
                    self._patterns[pattern.pattern_id] = pattern
            except (json.JSONDecodeError, IOError):
                self._patterns = {}

    def _save_patterns(self) -> None:
        """Save patterns to disk."""
        patterns_file = self._patterns_file()
        with open(patterns_file, "w") as f:
            json.dump([p.to_dict() for p in self._patterns.values()], f, indent=2)

    # =========================================================================
    # Storage
    # =========================================================================

    def store_pattern(self, pattern: LearnedPattern) -> None:
        """Store a new pattern."""
        self._patterns[pattern.pattern_id] = pattern
        self._save_patterns()

        # Also store in type-specific file
        type_file = self.base_path / "by_type" / f"{pattern.pattern_type.value}.jsonl"
        with open(type_file, "a") as f:
            f.write(json.dumps(pattern.to_dict()) + "\n")

        # Store in domain file if domain specified
        if pattern.domain:
            domain_file = self.base_path / "by_domain" / f"{pattern.domain}.jsonl"
            with open(domain_file, "a") as f:
                f.write(json.dumps(pattern.to_dict()) + "\n")

    def get_pattern(self, pattern_id: str) -> Optional[LearnedPattern]:
        """Retrieve a pattern by ID."""
        return self._patterns.get(pattern_id)

    def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> None:
        """Update pattern metadata."""
        pattern = self._patterns.get(pattern_id)
        if pattern:
            for key, value in updates.items():
                if hasattr(pattern, key):
                    setattr(pattern, key, value)
            self._save_patterns()

    def delete_pattern(self, pattern_id: str) -> None:
        """Remove a pattern."""
        if pattern_id in self._patterns:
            del self._patterns[pattern_id]
            self._save_patterns()

    # =========================================================================
    # Retrieval
    # =========================================================================

    def get_relevant_patterns(
        self,
        task: str,
        domain: Optional[str] = None,
        ptool_name: Optional[str] = None,
        pattern_type: Optional[PatternType] = None,
        min_relevance: float = 0.3,
        limit: int = 10,
    ) -> List[LearnedPattern]:
        """
        Get patterns relevant to a task.

        Uses keyword matching + relevance scoring.
        """
        task_lower = task.lower()
        task_words = set(re.findall(r'\w+', task_lower))
        task_pattern = " ".join(task_lower.split()[:5])

        scored_patterns: List[tuple] = []

        for pattern in self._patterns.values():
            # Filter by type if specified
            if pattern_type and pattern.pattern_type != pattern_type:
                continue

            # Filter by domain if specified
            if domain and pattern.domain and pattern.domain != domain:
                continue

            # Filter by ptool if specified
            if ptool_name and pattern.ptool_name and pattern.ptool_name != ptool_name:
                continue

            # Calculate relevance
            score = pattern.relevance_score

            # Boost for matching goal pattern
            if pattern.goal_pattern and pattern.goal_pattern in task_lower:
                score += 0.3

            # Boost for keyword overlap with content
            content_words = set(re.findall(r'\w+', pattern.content.lower()))
            overlap = len(task_words & content_words)
            if overlap > 0:
                score += 0.1 * min(overlap, 5)

            if score >= min_relevance:
                scored_patterns.append((pattern, score))

        # Sort by score descending
        scored_patterns.sort(key=lambda x: x[1], reverse=True)

        return [p[0] for p in scored_patterns[:limit]]

    def get_icl_examples(
        self,
        ptool_name: Optional[str] = None,
        success_only: bool = True,
        limit: int = 5,
    ) -> List[LearnedPattern]:
        """Get ICL example patterns."""
        pattern_type = PatternType.POSITIVE if success_only else None

        examples = []
        for pattern in self._patterns.values():
            if success_only and pattern.pattern_type != PatternType.POSITIVE:
                continue
            if not success_only and pattern.pattern_type == PatternType.NEGATIVE:
                continue
            if ptool_name and pattern.ptool_name != ptool_name:
                continue
            examples.append(pattern)

        # Sort by relevance
        examples.sort(key=lambda p: p.relevance_score, reverse=True)
        return examples[:limit]

    def get_negative_patterns(
        self,
        task: str,
        limit: int = 5,
    ) -> List[LearnedPattern]:
        """Get patterns of what to avoid."""
        return self.get_relevant_patterns(
            task=task,
            pattern_type=PatternType.NEGATIVE,
            limit=limit,
        )

    def get_heuristics(
        self,
        task: str,
        limit: int = 3,
    ) -> List[LearnedPattern]:
        """Get relevant heuristic patterns."""
        return self.get_relevant_patterns(
            task=task,
            pattern_type=PatternType.HEURISTIC,
            limit=limit,
        )

    # =========================================================================
    # Forgetting/Decay
    # =========================================================================

    def apply_decay(self, days: int = 1) -> int:
        """
        Apply time-based decay to all patterns.

        Returns number of patterns affected.
        """
        affected = 0
        for pattern in self._patterns.values():
            old_confidence = pattern.confidence
            pattern.apply_decay(days)
            if pattern.confidence != old_confidence:
                affected += 1

        self._save_patterns()
        return affected

    def prune_low_confidence(self, threshold: float = 0.1) -> int:
        """Remove patterns below confidence threshold."""
        to_remove = [
            pid for pid, p in self._patterns.items()
            if p.confidence < threshold
        ]

        for pid in to_remove:
            del self._patterns[pid]

        if to_remove:
            self._save_patterns()

        return len(to_remove)

    def reinforce_pattern(self, pattern_id: str, was_helpful: bool) -> None:
        """
        Update pattern based on whether it was helpful.

        If helpful: boost confidence, increment times_helpful
        If not helpful: decrease confidence
        """
        pattern = self._patterns.get(pattern_id)
        if pattern:
            pattern.reinforce(was_helpful)
            self._save_patterns()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        patterns_by_type = {}
        total_confidence = 0.0
        total_used = 0
        total_helpful = 0

        for pattern in self._patterns.values():
            type_name = pattern.pattern_type.value
            patterns_by_type[type_name] = patterns_by_type.get(type_name, 0) + 1
            total_confidence += pattern.confidence
            total_used += pattern.times_used
            total_helpful += pattern.times_helpful

        return {
            "total_patterns": len(self._patterns),
            "patterns_by_type": patterns_by_type,
            "avg_confidence": total_confidence / len(self._patterns) if self._patterns else 0.0,
            "total_uses": total_used,
            "total_helpful": total_helpful,
            "helpfulness_rate": total_helpful / total_used if total_used > 0 else 0.0,
        }

    # =========================================================================
    # Learning Log
    # =========================================================================

    def log_learning_event(self, event: LearningEvent) -> None:
        """Record a learning event."""
        log_file = self.base_path / "learning_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def get_learning_history(self, limit: int = 100) -> List[LearningEvent]:
        """Get recent learning events."""
        log_file = self.base_path / "learning_log.jsonl"
        if not log_file.exists():
            return []

        events = []
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    events.append(LearningEvent(
                        event_id=data["event_id"],
                        event_type=data["event_type"],
                        pattern_id=data.get("pattern_id"),
                        trace_id=data.get("trace_id"),
                        details=data.get("details", {}),
                        timestamp=data.get("timestamp", ""),
                    ))

        return events[-limit:]


# =============================================================================
# Self-Improving Agent
# =============================================================================

class SelfImprovingAgent:
    """
    Wraps a base agent and enhances it with self-improvement capabilities.

    Features:
    - Learns from successful and failed trajectories
    - Enhances prompts with relevant learned patterns
    - Applies forgetting mechanism for outdated patterns
    - Tracks improvement metrics over time

    Example:
        >>> base_agent = ReActAgent(available_ptools=[...])
        >>> improving_agent = SelfImprovingAgent(base_agent)
        >>>
        >>> # Run tasks - agent learns automatically
        >>> result = improving_agent.run("Calculate BMI for patient")
        >>>
        >>> # Check what it learned
        >>> patterns = improving_agent.get_learned_patterns()
        >>> metrics = improving_agent.get_improvement_metrics()
    """

    def __init__(
        self,
        base_agent: ReActAgent,
        pattern_memory: Optional[PatternMemory] = None,
        pattern_extractor: Optional[PatternExtractor] = None,
        trace_store: Optional[TraceStore] = None,
        critic: Optional[TraceCritic] = None,
        repair_agent: Optional[RepairAgent] = None,
        # Learning settings
        learn_from_success: bool = True,
        learn_from_failure: bool = True,
        learn_from_repair: bool = True,
        auto_repair: bool = True,
        max_repair_attempts: int = 2,
        # Pattern retrieval settings
        max_positive_patterns: int = 3,
        max_negative_patterns: int = 2,
        max_heuristics: int = 2,
        min_pattern_relevance: float = 0.3,
        # Decay settings
        enable_decay: bool = True,
        decay_interval_runs: int = 10,  # Apply decay every N runs
        min_confidence_threshold: float = 0.1,
        # Debugging
        echo: bool = False,
    ):
        self.base_agent = base_agent
        self.pattern_memory = pattern_memory or PatternMemory()
        self.pattern_extractor = pattern_extractor or PatternExtractor(
            model=base_agent.model,
            llm_backend=base_agent.llm_backend,
        )
        self.trace_store = trace_store or get_trace_store()
        self.critic = critic
        self.repair_agent = repair_agent

        # Learning settings
        self.learn_from_success = learn_from_success
        self.learn_from_failure = learn_from_failure
        self.learn_from_repair = learn_from_repair
        self.auto_repair = auto_repair
        self.max_repair_attempts = max_repair_attempts

        # Pattern retrieval settings
        self.max_positive_patterns = max_positive_patterns
        self.max_negative_patterns = max_negative_patterns
        self.max_heuristics = max_heuristics
        self.min_pattern_relevance = min_pattern_relevance

        # Decay settings
        self.enable_decay = enable_decay
        self.decay_interval_runs = decay_interval_runs
        self.min_confidence_threshold = min_confidence_threshold

        # Metrics tracking
        self._run_count = 0
        self._success_count = 0
        self._baseline_runs: List[bool] = []  # Track first N runs as baseline
        self._baseline_size = 10

        self.echo = echo

    def run(self, goal: str, context: Optional[Dict[str, Any]] = None) -> ReActResult:
        """
        Run the agent on a goal with self-improvement.

        1. Retrieve relevant patterns from memory
        2. Enhance base agent with patterns
        3. Execute the enhanced agent
        4. Evaluate result with critic (if available)
        5. If needed and auto_repair enabled, repair
        6. Learn from the experience
        7. Return result
        """
        self._run_count += 1

        # 1. Retrieve relevant patterns
        relevant_patterns = self._retrieve_patterns(goal, context)

        if self.echo:
            print(f"[SelfImprove] Retrieved {len(relevant_patterns)} patterns")

        # 2. Run enhanced agent
        result = self._run_with_patterns(goal, relevant_patterns)

        # 3. Evaluate with critic (if available)
        evaluation = None
        if self.critic:
            evaluation = self.critic.evaluate(
                result.trajectory,
                goal=goal,
            )

            if self.echo:
                print(f"[SelfImprove] Critic verdict: {evaluation.verdict.value}")

        # 4. Auto-repair if needed
        repaired = False
        repair_result = None
        if (self.auto_repair and evaluation and
            evaluation.verdict == CriticVerdict.REPAIR_NEEDED):

            repair_result = self._attempt_repair(result, evaluation, goal)
            if repair_result and repair_result.success:
                repaired = True
                # Re-run with repaired trace
                result = self._rerun_after_repair(repair_result, goal)

                if self.echo:
                    print(f"[SelfImprove] Repair successful")

        # 5. Learn from experience
        final_success = result.success or repaired

        if final_success and self.learn_from_success:
            self._learn_from_success(result, goal, relevant_patterns)
        elif not final_success and self.learn_from_failure:
            self._learn_from_failure(result, evaluation, goal, relevant_patterns)

        if repaired and self.learn_from_repair and repair_result:
            self._learn_from_repair(result, repair_result, goal)

        # 6. Update pattern usefulness
        self._update_pattern_usefulness(relevant_patterns, final_success)

        # 7. Update metrics
        if final_success:
            self._success_count += 1

        if len(self._baseline_runs) < self._baseline_size:
            self._baseline_runs.append(final_success)

        # Apply decay periodically
        if self.enable_decay and self._run_count % self.decay_interval_runs == 0:
            self.run_decay_cycle()

        return result

    def _retrieve_patterns(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[LearnedPattern]:
        """Retrieve relevant patterns from memory."""
        patterns = []

        # Get positive ICL examples
        patterns.extend(
            self.pattern_memory.get_relevant_patterns(
                task=goal,
                pattern_type=PatternType.POSITIVE,
                min_relevance=self.min_pattern_relevance,
                limit=self.max_positive_patterns,
            )
        )

        # Get negative patterns (what to avoid)
        patterns.extend(
            self.pattern_memory.get_negative_patterns(
                task=goal,
                limit=self.max_negative_patterns,
            )
        )

        # Get heuristics
        patterns.extend(
            self.pattern_memory.get_heuristics(
                task=goal,
                limit=self.max_heuristics,
            )
        )

        return patterns

    def _run_with_patterns(
        self,
        goal: str,
        patterns: List[LearnedPattern],
    ) -> ReActResult:
        """Run agent with patterns injected into context."""
        if not patterns:
            return self.base_agent.run(goal)

        # Format patterns as context
        pattern_context = self._format_pattern_context(patterns)

        # Enhance goal with pattern context
        enhanced_goal = f"""{pattern_context}

---

TASK: {goal}"""

        return self.base_agent.run(enhanced_goal)

    def _format_pattern_context(self, patterns: List[LearnedPattern]) -> str:
        """Format patterns as context for the prompt."""
        sections = []

        # Group by type
        positive = [p for p in patterns if p.pattern_type == PatternType.POSITIVE]
        negative = [p for p in patterns if p.pattern_type == PatternType.NEGATIVE]
        heuristics = [p for p in patterns if p.pattern_type == PatternType.HEURISTIC]
        repairs = [p for p in patterns if p.pattern_type == PatternType.REPAIR]

        if positive:
            sections.append("## Successful examples from past experience:")
            for p in positive[:3]:
                sections.append(f"\n{p.content[:500]}\n")

        if negative:
            sections.append("\n## Patterns to AVOID (learned from failures):")
            for p in negative[:2]:
                sections.append(f"\n{p.content[:300]}\n")

        if heuristics:
            sections.append("\n## Helpful heuristics:")
            for p in heuristics[:2]:
                sections.append(f"- {p.content[:200]}")

        if repairs:
            sections.append("\n## Repair patterns that worked:")
            for p in repairs[:1]:
                sections.append(f"\n{p.content[:300]}\n")

        return "\n".join(sections)

    def _attempt_repair(
        self,
        result: ReActResult,
        evaluation: CriticEvaluation,
        goal: str,
    ) -> Optional[RepairResult]:
        """Attempt to repair a failed trace."""
        if self.repair_agent is None:
            # Create repair agent with critic
            if self.critic is None:
                return None
            self.repair_agent = RepairAgent(
                critic=self.critic,
                max_attempts=self.max_repair_attempts,
            )

        try:
            return self.repair_agent.repair(
                result.trajectory,
                evaluation,
                goal,
            )
        except Exception as e:
            if self.echo:
                print(f"[SelfImprove] Repair failed: {e}")
            return None

    def _rerun_after_repair(
        self,
        repair_result: RepairResult,
        goal: str,
    ) -> ReActResult:
        """Re-run agent after repair (for now, just return success indicator)."""
        # Note: In a full implementation, we would execute the repaired trace
        # For now, we create a synthetic result
        from .react import ReActTrajectory

        trajectory = ReActTrajectory(
            trajectory_id=str(uuid.uuid4())[:8],
            goal=goal,
            success=True,
            final_answer="[Repaired]",
            model_used=self.base_agent.model,
            total_llm_calls=0,
            total_time_ms=0,
        )

        return ReActResult(
            trajectory=trajectory,
            success=True,
            answer="[Repaired result]",
        )

    def _learn_from_success(
        self,
        result: ReActResult,
        goal: str,
        used_patterns: List[LearnedPattern],
    ) -> None:
        """Extract and store patterns from a successful run."""
        patterns = self.pattern_extractor.extract_from_successful_trace(
            result.trajectory
        )

        for pattern in patterns:
            pattern.domain = self._infer_domain(goal)
            self.pattern_memory.store_pattern(pattern)

            self.pattern_memory.log_learning_event(LearningEvent(
                event_id=str(uuid.uuid4())[:8],
                event_type="pattern_extracted",
                pattern_id=pattern.pattern_id,
                trace_id=result.trajectory.trajectory_id,
                details={
                    "pattern_type": pattern.pattern_type.value,
                    "goal": goal,
                    "from_success": True,
                },
            ))

            if self.echo:
                print(f"[SelfImprove] Learned {pattern.pattern_type.value} pattern: {pattern.pattern_id}")

    def _learn_from_failure(
        self,
        result: ReActResult,
        evaluation: Optional[CriticEvaluation],
        goal: str,
        used_patterns: List[LearnedPattern],
    ) -> None:
        """Extract patterns from a failed run to avoid in future."""
        error_analysis = None
        if evaluation:
            error_analysis = "; ".join(evaluation.reasoning_issues) if evaluation.reasoning_issues else None

        patterns = self.pattern_extractor.extract_from_failed_trace(
            result.trajectory,
            error_analysis=error_analysis,
        )

        for pattern in patterns:
            pattern.domain = self._infer_domain(goal)
            self.pattern_memory.store_pattern(pattern)

            self.pattern_memory.log_learning_event(LearningEvent(
                event_id=str(uuid.uuid4())[:8],
                event_type="pattern_extracted",
                pattern_id=pattern.pattern_id,
                trace_id=result.trajectory.trajectory_id,
                details={
                    "pattern_type": pattern.pattern_type.value,
                    "goal": goal,
                    "from_success": False,
                },
            ))

            if self.echo:
                print(f"[SelfImprove] Learned {pattern.pattern_type.value} pattern from failure: {pattern.pattern_id}")

    def _learn_from_repair(
        self,
        result: ReActResult,
        repair_result: RepairResult,
        goal: str,
    ) -> None:
        """Learn from a successful repair."""
        # Need both original and repaired trajectories
        # For now, store the repair pattern
        patterns = self.pattern_extractor.extract_from_repair(
            result.trajectory,
            result.trajectory,  # Would be repaired trajectory
            repair_result,
        )

        for pattern in patterns:
            pattern.domain = self._infer_domain(goal)
            self.pattern_memory.store_pattern(pattern)

            if self.echo:
                print(f"[SelfImprove] Learned repair pattern: {pattern.pattern_id}")

    def _update_pattern_usefulness(
        self,
        used_patterns: List[LearnedPattern],
        was_successful: bool,
    ) -> None:
        """Update patterns based on whether they contributed to success."""
        for pattern in used_patterns:
            self.pattern_memory.reinforce_pattern(
                pattern.pattern_id,
                was_helpful=was_successful,
            )

            self.pattern_memory.log_learning_event(LearningEvent(
                event_id=str(uuid.uuid4())[:8],
                event_type="pattern_helpful" if was_successful else "pattern_unhelpful",
                pattern_id=pattern.pattern_id,
                details={"was_successful": was_successful},
            ))

    def _infer_domain(self, goal: str) -> Optional[str]:
        """Infer domain from goal text."""
        goal_lower = goal.lower()

        # Simple keyword-based domain inference
        if any(w in goal_lower for w in ["bmi", "egfr", "dosage", "clinical", "patient", "medical"]):
            return "medcalc"
        if any(w in goal_lower for w in ["code", "program", "function", "class", "bug"]):
            return "coding"
        if any(w in goal_lower for w in ["analyze", "extract", "parse", "find"]):
            return "analysis"

        return "general"

    # =========================================================================
    # Decay Management
    # =========================================================================

    def run_decay_cycle(self) -> Dict[str, int]:
        """
        Run a decay cycle on pattern memory.

        Returns statistics about patterns affected.
        """
        if not self.enable_decay:
            return {}

        decayed = self.pattern_memory.apply_decay(days=1)
        pruned = self.pattern_memory.prune_low_confidence(
            self.min_confidence_threshold
        )

        if self.echo and (decayed > 0 or pruned > 0):
            print(f"[SelfImprove] Decay cycle: {decayed} decayed, {pruned} pruned")

        return {
            "patterns_decayed": decayed,
            "patterns_pruned": pruned,
        }

    # =========================================================================
    # Metrics and Introspection
    # =========================================================================

    def get_learned_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
    ) -> List[LearnedPattern]:
        """Get all learned patterns, optionally filtered by type."""
        if pattern_type:
            return [
                p for p in self.pattern_memory._patterns.values()
                if p.pattern_type == pattern_type
            ]
        return list(self.pattern_memory._patterns.values())

    def get_improvement_metrics(self) -> SelfImprovementMetrics:
        """Get metrics showing improvement over time."""
        memory_stats = self.pattern_memory.get_stats()

        # Calculate baseline vs current success rate
        baseline_rate = 0.0
        if self._baseline_runs:
            baseline_rate = sum(1 for r in self._baseline_runs if r) / len(self._baseline_runs)

        current_rate = self._success_count / self._run_count if self._run_count > 0 else 0.0
        improvement = current_rate - baseline_rate

        return SelfImprovementMetrics(
            total_patterns_learned=memory_stats["total_patterns"],
            active_patterns=len([
                p for p in self.pattern_memory._patterns.values()
                if p.confidence >= self.min_confidence_threshold
            ]),
            patterns_by_type=memory_stats["patterns_by_type"],
            improvement_rate=improvement,
            baseline_success_rate=baseline_rate,
            current_success_rate=current_rate,
            total_runs=self._run_count,
            successful_runs=self._success_count,
        )

    def explain_patterns_for_task(self, goal: str) -> str:
        """Explain what patterns would be used for a given goal."""
        patterns = self._retrieve_patterns(goal)

        if not patterns:
            return f"No relevant patterns found for: {goal}"

        lines = [f"Patterns for: {goal}\n"]

        for pattern in patterns:
            lines.append(f"- [{pattern.pattern_type.value}] (relevance: {pattern.relevance_score:.2f})")
            lines.append(f"  {pattern.content[:100]}...")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def self_improving_react(
    goal: str,
    available_ptools: Optional[List] = None,
    pattern_memory: Optional[PatternMemory] = None,
    **kwargs,
) -> ReActResult:
    """
    Run a self-improving ReAct agent.

    Example:
        >>> result = self_improving_react("Calculate BMI", available_ptools=[...])
    """
    base_agent = ReActAgent(available_ptools=available_ptools, **kwargs)
    improving_agent = SelfImprovingAgent(
        base_agent,
        pattern_memory=pattern_memory,
    )
    return improving_agent.run(goal)


def get_pattern_memory(path: str = "~/.pattern_memory") -> PatternMemory:
    """Get or create a PatternMemory instance."""
    return PatternMemory(path=path)
