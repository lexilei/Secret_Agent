"""
Intelligent LLM Model Selection.

This module provides intelligent routing of ptool calls to appropriate LLM models
based on task complexity, historical performance, and cost considerations.

Key Classes:
    - TaskComplexity: Enum for complexity levels
    - ModelConfig: Configuration for an LLM model
    - SelectionResult: Result of model selection
    - FallbackChain: Chain of models to try
    - ExperienceStore: Store for performance history
    - ModelSelector: Main selector class

Features:
- Automatic complexity estimation
- Experience-based learning from past executions
- Fallback chains for robustness
- Cost/latency optimization

Example:
    >>> from ptool_framework.model_selector import ModelSelector, TaskComplexity
    >>>
    >>> selector = ModelSelector()
    >>>
    >>> # Select model for a ptool call
    >>> model = selector.select(ptool_spec, inputs)
    >>>
    >>> # Get fallback chain
    >>> chain = selector.get_fallback_chain(ptool_spec, inputs)
    >>> for model in chain.models:
    ...     try:
    ...         result = execute_with_model(model)
    ...         break
    ...     except Exception:
    ...         continue
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .ptool import PToolSpec


# =============================================================================
# LLMS.json Loading
# =============================================================================

def _find_llms_json() -> Optional[Path]:
    """Find LLMS.json in standard locations."""
    search_paths = [
        Path.cwd() / "LLMS.json",
        Path.cwd().parent / "LLMS.json",
        Path(__file__).parent.parent / "LLMS.json",
    ]
    for path in search_paths:
        if path.exists():
            return path
    return None


def load_llms_config(
    config_path: Optional[str] = None,
    enabled_only: bool = True,
    required_tags: Optional[List[str]] = None,
) -> Dict[str, "ModelConfig"]:
    """
    Load model configurations from LLMS.json.

    Args:
        config_path: Path to LLMS.json, or None to auto-find
        enabled_only: If True, only return models with enabled=True
        required_tags: If provided, only return models with ALL these tags

    Returns:
        Dictionary mapping model names to ModelConfig objects

    Example:
        >>> # Load all enabled models
        >>> models = load_llms_config()
        >>>
        >>> # Load only models tagged for testing
        >>> models = load_llms_config(required_tags=["testing"])
        >>>
        >>> # Load all models (including disabled)
        >>> models = load_llms_config(enabled_only=False)
    """
    if config_path:
        path = Path(config_path)
    else:
        path = _find_llms_json()

    if not path or not path.exists():
        # Return minimal default
        return {
            "deepseek-v3-0324": ModelConfig(
                name="deepseek-v3-0324",
                provider="together",
                model_id="deepseek-ai/DeepSeek-V3",
                quality_score=0.95,
            )
        }

    with open(path) as f:
        data = json.load(f)

    models = {}
    for name, model_data in data.get("models", {}).items():
        # Check enabled flag
        if enabled_only and not model_data.get("enabled", True):
            continue

        # Check required tags
        if required_tags:
            model_tags = set(model_data.get("tags", []))
            if not all(tag in model_tags for tag in required_tags):
                continue

        # Parse cost
        cost_data = model_data.get("cost", {})
        if isinstance(cost_data, dict):
            cost_input = cost_data.get("input", 0.001)
            cost_output = cost_data.get("output", 0.002)
        else:
            cost_input = 0.001
            cost_output = 0.002

        # Use manual quality_score if provided, otherwise estimate from capabilities
        quality = model_data.get("quality_score")
        if quality is None:
            quality = _estimate_quality_from_capabilities(
                model_data.get("capabilities", [])
            )

        # Get ELO score if available
        elo_score = model_data.get("elo_score")

        models[name] = ModelConfig(
            name=name,
            provider=model_data.get("provider", "unknown"),
            model_id=model_data.get("model_id", name),
            cost_per_1k_input=cost_input,
            cost_per_1k_output=cost_output,
            max_tokens=model_data.get("context_window", 32000),
            capabilities=model_data.get("capabilities", []),
            latency_ms_avg=_estimate_latency(model_data.get("provider", "")),
            quality_score=quality,
            elo_score=elo_score,
            enabled=model_data.get("enabled", True),
            tags=model_data.get("tags", []),
            description=model_data.get("description", ""),
        )

    return models


def _estimate_quality_from_capabilities(capabilities: List[str]) -> float:
    """Estimate quality score from capabilities."""
    base = 0.7
    boosts = {
        "reasoning": 0.1,
        "extended-thinking": 0.1,
        "chain-of-thought": 0.05,
        "coding": 0.05,
        "agent": 0.03,
    }
    penalties = {
        "fast": -0.05,  # Fast models trade quality for speed
    }
    for cap in capabilities:
        base += boosts.get(cap, 0)
        base += penalties.get(cap, 0)
    return min(1.0, base)


def _estimate_latency(provider: str) -> float:
    """Estimate latency based on provider."""
    latencies = {
        "together": 800,
        "anthropic": 1200,
        "openai": 1000,
        "groq": 300,
        "google": 1000,
        "local": 500,
    }
    return latencies.get(provider.lower(), 1000)


def get_default_model(config_path: Optional[str] = None) -> str:
    """Get the default model from LLMS.json."""
    if config_path:
        path = Path(config_path)
    else:
        path = _find_llms_json()

    if not path or not path.exists():
        return "deepseek-v3-0324"

    with open(path) as f:
        data = json.load(f)
    return data.get("default_model", "deepseek-v3-0324")


def get_enabled_models(required_tags: Optional[List[str]] = None) -> List[str]:
    """Get list of enabled model names."""
    models = load_llms_config(enabled_only=True, required_tags=required_tags)
    return list(models.keys())


def set_model_enabled(
    model_name: str,
    enabled: bool,
    config_path: Optional[str] = None,
) -> bool:
    """
    Enable or disable a model in LLMS.json.

    Args:
        model_name: Name of the model
        enabled: Whether to enable or disable
        config_path: Path to LLMS.json

    Returns:
        True if successful, False if model not found
    """
    if config_path:
        path = Path(config_path)
    else:
        path = _find_llms_json()

    if not path or not path.exists():
        return False

    with open(path) as f:
        data = json.load(f)

    if model_name not in data.get("models", {}):
        return False

    data["models"][model_name]["enabled"] = enabled

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return True


def add_model_tag(
    model_name: str,
    tag: str,
    config_path: Optional[str] = None,
) -> bool:
    """Add a tag to a model."""
    if config_path:
        path = Path(config_path)
    else:
        path = _find_llms_json()

    if not path or not path.exists():
        return False

    with open(path) as f:
        data = json.load(f)

    if model_name not in data.get("models", {}):
        return False

    tags = data["models"][model_name].get("tags", [])
    if tag not in tags:
        tags.append(tag)
        data["models"][model_name]["tags"] = tags

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return True


def remove_model_tag(
    model_name: str,
    tag: str,
    config_path: Optional[str] = None,
) -> bool:
    """Remove a tag from a model."""
    if config_path:
        path = Path(config_path)
    else:
        path = _find_llms_json()

    if not path or not path.exists():
        return False

    with open(path) as f:
        data = json.load(f)

    if model_name not in data.get("models", {}):
        return False

    tags = data["models"][model_name].get("tags", [])
    if tag in tags:
        tags.remove(tag)
        data["models"][model_name]["tags"] = tags

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return True


class TaskComplexity(Enum):
    """
    Complexity levels for tasks.

    Used to route to appropriate models:
    - TRIVIAL: Very simple, any model works
    - SIMPLE: Basic reasoning, fast models work
    - MODERATE: Some complexity, mid-tier models
    - COMPLEX: Significant reasoning, capable models
    - EXPERT: Maximum difficulty, best models only
    """
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class ModelConfig:
    """
    Configuration for an LLM model.

    Attributes:
        name: Model identifier (e.g., "deepseek-v3")
        provider: API provider (e.g., "together", "openai")
        model_id: The actual model ID for API calls
        cost_per_1k_input: Cost per 1k input tokens
        cost_per_1k_output: Cost per 1k output tokens
        max_tokens: Maximum context length
        capabilities: List of capability tags
        latency_ms_avg: Average latency in ms
        quality_score: Estimated quality (0-1)
        enabled: Whether this model is enabled for use
        tags: Environment tags (e.g., "testing", "production")
        description: Human-readable description

    Example:
        >>> config = ModelConfig(
        ...     name="deepseek-v3",
        ...     provider="together",
        ...     cost_per_1k_input=0.0005,
        ...     quality_score=0.85
        ... )
    """

    name: str
    provider: str = "together"
    model_id: str = ""
    cost_per_1k_input: float = 0.001
    cost_per_1k_output: float = 0.002
    max_tokens: int = 32000
    capabilities: List[str] = field(default_factory=list)
    latency_ms_avg: float = 1000.0
    quality_score: float = 0.7
    elo_score: Optional[int] = None  # From LM Arena leaderboard
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        if not self.model_id:
            self.model_id = self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "max_tokens": self.max_tokens,
            "capabilities": self.capabilities,
            "latency_ms_avg": self.latency_ms_avg,
            "quality_score": self.quality_score,
            "enabled": self.enabled,
            "tags": self.tags,
            "description": self.description,
        }


# Models are now loaded from LLMS.json - no hardcoded defaults
# Use load_llms_config() to get available models


@dataclass
class SelectionCriteria:
    """
    Criteria for model selection.

    Attributes:
        required_capabilities: Capabilities the model must have
        max_cost_per_1k_tokens: Maximum acceptable cost
        max_latency_ms: Maximum acceptable latency
        min_quality_threshold: Minimum quality score
    """

    required_capabilities: List[str] = field(default_factory=list)
    max_cost_per_1k_tokens: float = 0.01
    max_latency_ms: int = 5000
    min_quality_threshold: float = 0.5


@dataclass
class SelectionResult:
    """
    Result of model selection.

    Attributes:
        selected_model: Name of selected model
        config: Full model configuration
        reason: Why this model was selected
        confidence: Confidence in selection (0-1)
        fallback_chain: Backup models to try
        estimated_cost: Estimated cost for this call
    """

    selected_model: str
    config: Optional[ModelConfig] = None
    reason: str = ""
    confidence: float = 0.8
    fallback_chain: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected_model": self.selected_model,
            "config": self.config.to_dict() if self.config else None,
            "reason": self.reason,
            "confidence": self.confidence,
            "fallback_chain": self.fallback_chain,
            "estimated_cost": self.estimated_cost,
        }


@dataclass
class FallbackChain:
    """
    Chain of models to try in order.

    Attributes:
        models: List of model names in order
        reason: Why this chain was created
    """

    models: List[str] = field(default_factory=list)
    reason: str = ""

    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)


@dataclass
class ModelPerformance:
    """
    Performance statistics for a model on a specific ptool.

    Attributes:
        model_name: Name of the model
        ptool_name: Name of the ptool
        total_calls: Total number of calls
        successful_calls: Number of successful calls
        avg_latency_ms: Average latency
        total_cost: Total cost
        last_updated: When last updated
    """

    model_name: str
    ptool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    avg_latency_ms: float = 0.0
    total_cost: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "ptool_name": self.ptool_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "avg_latency_ms": self.avg_latency_ms,
            "total_cost": self.total_cost,
            "success_rate": self.success_rate,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelPerformance":
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            ptool_name=data["ptool_name"],
            total_calls=data.get("total_calls", 0),
            successful_calls=data.get("successful_calls", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            total_cost=data.get("total_cost", 0.0),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
        )


class ExperienceStore:
    """
    Persistent storage for model performance history.

    Stores performance data for each model-ptool combination
    to enable experience-based routing decisions.

    Storage:
        ~/.ptool_experiences/
        ├── performance/
        │   ├── deepseek-v3.json
        │   └── llama-3.1-70b.json
        └── best_models.json

    Example:
        >>> store = ExperienceStore()
        >>> store.record_execution("deepseek-v3", "extract_values", True, 500, 0.001)
        >>> perf = store.get_performance("deepseek-v3", "extract_values")
        >>> print(f"Success rate: {perf.success_rate:.1%}")
    """

    def __init__(self, path: str = "~/.ptool_experiences"):
        """
        Initialize the experience store.

        Args:
            path: Base path for storage
        """
        self.base_path = Path(os.path.expanduser(path))
        self._ensure_directories()
        self._performance_cache: Dict[str, Dict[str, ModelPerformance]] = {}
        self._load_cache()

    def _ensure_directories(self) -> None:
        """Create storage directories."""
        (self.base_path / "performance").mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> None:
        """Load performance data into cache."""
        perf_dir = self.base_path / "performance"
        if not perf_dir.exists():
            return

        for model_file in perf_dir.glob("*.json"):
            try:
                model_name = model_file.stem
                with open(model_file) as f:
                    data = json.load(f)
                self._performance_cache[model_name] = {
                    ptool: ModelPerformance.from_dict(perf_data)
                    for ptool, perf_data in data.items()
                }
            except Exception:
                pass

    def _save_model_performance(self, model_name: str) -> None:
        """Save performance data for a model."""
        if model_name not in self._performance_cache:
            return

        model_file = self.base_path / "performance" / f"{model_name}.json"
        data = {
            ptool: perf.to_dict()
            for ptool, perf in self._performance_cache[model_name].items()
        }
        with open(model_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_execution(
        self,
        model_name: str,
        ptool_name: str,
        success: bool,
        latency_ms: float,
        cost: float = 0.0,
    ) -> None:
        """
        Record an execution for learning.

        Args:
            model_name: Name of the model used
            ptool_name: Name of the ptool called
            success: Whether execution succeeded
            latency_ms: Execution latency
            cost: Cost of the call
        """
        if model_name not in self._performance_cache:
            self._performance_cache[model_name] = {}

        if ptool_name not in self._performance_cache[model_name]:
            self._performance_cache[model_name][ptool_name] = ModelPerformance(
                model_name=model_name,
                ptool_name=ptool_name,
            )

        perf = self._performance_cache[model_name][ptool_name]

        # Update statistics
        old_total = perf.total_calls
        perf.total_calls += 1

        if success:
            perf.successful_calls += 1

        # Running average for latency
        perf.avg_latency_ms = (
            (perf.avg_latency_ms * old_total + latency_ms) / perf.total_calls
        )

        perf.total_cost += cost
        perf.last_updated = datetime.now().isoformat()

        # Save to disk
        self._save_model_performance(model_name)

    def get_performance(
        self,
        model_name: str,
        ptool_name: str,
    ) -> Optional[ModelPerformance]:
        """
        Get performance statistics for a model-ptool combination.

        Args:
            model_name: Model name
            ptool_name: Ptool name

        Returns:
            ModelPerformance or None if no data
        """
        if model_name not in self._performance_cache:
            return None
        return self._performance_cache[model_name].get(ptool_name)

    def get_all_performance(self, model_name: str) -> Dict[str, ModelPerformance]:
        """Get all performance data for a model."""
        return self._performance_cache.get(model_name, {})

    def get_best_model_for_ptool(
        self,
        ptool_name: str,
        min_calls: int = 5,
    ) -> Optional[str]:
        """
        Get the best performing model for a ptool.

        Args:
            ptool_name: Name of the ptool
            min_calls: Minimum calls required for consideration

        Returns:
            Best model name or None
        """
        best_model = None
        best_score = -1.0

        for model_name, ptools in self._performance_cache.items():
            if ptool_name in ptools:
                perf = ptools[ptool_name]
                if perf.total_calls >= min_calls:
                    score = perf.success_rate
                    if score > best_score:
                        best_score = score
                        best_model = model_name

        return best_model

    def get_model_ranking(
        self,
        ptool_name: str,
        min_calls: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Get ranked list of models for a ptool.

        Args:
            ptool_name: Name of the ptool
            min_calls: Minimum calls required

        Returns:
            List of (model_name, success_rate) tuples
        """
        rankings = []

        for model_name, ptools in self._performance_cache.items():
            if ptool_name in ptools:
                perf = ptools[ptool_name]
                if perf.total_calls >= min_calls:
                    rankings.append((model_name, perf.success_rate))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class ModelSelector:
    """
    Intelligent model selector for ptool calls.

    Selects the most appropriate model based on:
    - Task complexity estimation
    - Historical performance data
    - Cost/latency requirements
    - Required capabilities

    Example:
        >>> selector = ModelSelector()
        >>>
        >>> # Simple selection
        >>> model = selector.select(ptool_spec, inputs)
        >>>
        >>> # Detailed selection
        >>> result = selector.select_with_details(ptool_spec, inputs)
        >>> print(f"Selected: {result.selected_model}")
        >>> print(f"Reason: {result.reason}")
        >>>
        >>> # Get fallback chain
        >>> chain = selector.get_fallback_chain(ptool_spec, inputs)
    """

    def __init__(
        self,
        experience_store: Optional[ExperienceStore] = None,
        models: Optional[Dict[str, ModelConfig]] = None,
        default_model: Optional[str] = None,
        enable_learning: bool = True,
        complexity_estimator: Optional[Callable] = None,
        required_tags: Optional[List[str]] = None,
    ):
        """
        Initialize the model selector.

        Args:
            experience_store: Store for performance history
            models: Model configurations (if None, loads from LLMS.json)
            default_model: Fallback model (if None, uses LLMS.json default)
            enable_learning: Whether to learn from executions
            complexity_estimator: Custom complexity estimator
            required_tags: Only use models with these tags (e.g., ["testing"])

        Example:
            >>> # For testing - only use models tagged for testing
            >>> selector = ModelSelector(required_tags=["testing"])
            >>>
            >>> # For production
            >>> selector = ModelSelector(required_tags=["production"])
        """
        self.experience_store = experience_store or ExperienceStore()
        self.required_tags = required_tags

        # Load models from LLMS.json if not provided
        if models is not None:
            self.models = models
        else:
            self.models = load_llms_config(
                enabled_only=True,
                required_tags=required_tags,
            )

        # Get default model from LLMS.json if not provided
        self.default_model = default_model or get_default_model()

        self.enable_learning = enable_learning
        self.complexity_estimator = complexity_estimator or heuristic_complexity_estimator

    def select(
        self,
        ptool_spec: "PToolSpec",
        inputs: Dict[str, Any],
        criteria: Optional[SelectionCriteria] = None,
    ) -> str:
        """
        Select the best model for a ptool call.

        Args:
            ptool_spec: Specification of the ptool
            inputs: Input arguments
            criteria: Optional selection criteria

        Returns:
            Model name to use

        Example:
            >>> model = selector.select(spec, {"text": "Hello world"})
        """
        result = self.select_with_details(ptool_spec, inputs, criteria)
        return result.selected_model

    def select_with_details(
        self,
        ptool_spec: "PToolSpec",
        inputs: Dict[str, Any],
        criteria: Optional[SelectionCriteria] = None,
    ) -> SelectionResult:
        """
        Select model with detailed reasoning.

        Args:
            ptool_spec: Specification of the ptool
            inputs: Input arguments
            criteria: Optional selection criteria

        Returns:
            SelectionResult with model and reasoning

        Example:
            >>> result = selector.select_with_details(spec, inputs)
            >>> print(f"Reason: {result.reason}")
        """
        criteria = criteria or SelectionCriteria()

        # 1. Estimate task complexity
        complexity = self._estimate_complexity(ptool_spec, inputs)

        # 2. Get candidates based on complexity
        candidates = self._get_candidates_for_complexity(complexity)

        # 3. Filter by criteria
        candidates = self._filter_by_criteria(candidates, criteria)

        # 4. Rank by experience
        ranked = self._rank_by_experience(candidates, ptool_spec.name)

        # 5. Select best
        if ranked:
            best_model, score = ranked[0]
            config = self.models.get(best_model)

            # Build fallback chain
            fallbacks = [m for m, _ in ranked[1:4]]

            return SelectionResult(
                selected_model=best_model,
                config=config,
                reason=f"Selected for {complexity.value} task (score: {score:.2f})",
                confidence=min(1.0, score + 0.2),
                fallback_chain=fallbacks,
            )

        # Fallback to default
        return SelectionResult(
            selected_model=self.default_model,
            config=self.models.get(self.default_model),
            reason="Fallback to default model",
            confidence=0.5,
        )

    def get_fallback_chain(
        self,
        ptool_spec: "PToolSpec",
        inputs: Dict[str, Any],
        max_length: int = 3,
    ) -> FallbackChain:
        """
        Get a chain of models to try in order.

        Args:
            ptool_spec: Specification of the ptool
            inputs: Input arguments
            max_length: Maximum chain length

        Returns:
            FallbackChain with models to try

        Example:
            >>> chain = selector.get_fallback_chain(spec, inputs)
            >>> for model in chain:
            ...     try:
            ...         result = execute(model, inputs)
            ...         break
            ...     except:
            ...         continue
        """
        result = self.select_with_details(ptool_spec, inputs)

        models = [result.selected_model] + result.fallback_chain
        models = models[:max_length]

        # Ensure default is in chain
        if self.default_model not in models and len(models) < max_length:
            models.append(self.default_model)

        return FallbackChain(
            models=models,
            reason=f"Chain for {ptool_spec.name}",
        )

    def record_execution(
        self,
        ptool_name: str,
        model: str,
        inputs: Dict[str, Any],
        success: bool,
        latency_ms: float,
        cost: float = 0.0,
    ) -> None:
        """
        Record execution result for learning.

        Args:
            ptool_name: Name of the ptool
            model: Model used
            inputs: Input arguments
            success: Whether execution succeeded
            latency_ms: Execution latency
            cost: Cost of the call
        """
        if self.enable_learning:
            self.experience_store.record_execution(
                model_name=model,
                ptool_name=ptool_name,
                success=success,
                latency_ms=latency_ms,
                cost=cost,
            )

    def _estimate_complexity(
        self,
        ptool_spec: "PToolSpec",
        inputs: Dict[str, Any],
    ) -> TaskComplexity:
        """Estimate task complexity."""
        return self.complexity_estimator(ptool_spec, inputs)

    def _get_candidates_for_complexity(
        self,
        complexity: TaskComplexity,
    ) -> List[str]:
        """Get candidate models for a complexity level."""
        complexity_thresholds = {
            TaskComplexity.TRIVIAL: 0.5,
            TaskComplexity.SIMPLE: 0.6,
            TaskComplexity.MODERATE: 0.7,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.EXPERT: 0.9,
        }

        min_quality = complexity_thresholds.get(complexity, 0.6)

        candidates = []
        for name, config in self.models.items():
            if config.quality_score >= min_quality:
                candidates.append(name)

        # For trivial/simple, also include lower quality but faster models
        if complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE):
            for name, config in self.models.items():
                if config.latency_ms_avg < 500 and name not in candidates:
                    candidates.append(name)

        return candidates

    def _filter_by_criteria(
        self,
        candidates: List[str],
        criteria: SelectionCriteria,
    ) -> List[str]:
        """Filter candidates by selection criteria."""
        filtered = []

        for name in candidates:
            config = self.models.get(name)
            if not config:
                continue

            # Check capabilities
            if criteria.required_capabilities:
                has_caps = all(
                    cap in config.capabilities
                    for cap in criteria.required_capabilities
                )
                if not has_caps:
                    continue

            # Check cost
            avg_cost = (config.cost_per_1k_input + config.cost_per_1k_output) / 2
            if avg_cost > criteria.max_cost_per_1k_tokens:
                continue

            # Check latency
            if config.latency_ms_avg > criteria.max_latency_ms:
                continue

            # Check quality
            if config.quality_score < criteria.min_quality_threshold:
                continue

            filtered.append(name)

        return filtered or candidates  # Return all if none pass

    def _rank_by_experience(
        self,
        candidates: List[str],
        ptool_name: str,
    ) -> List[Tuple[str, float]]:
        """Rank candidates by historical performance."""
        rankings = []

        for model_name in candidates:
            # Get experience-based score
            perf = self.experience_store.get_performance(model_name, ptool_name)
            if perf and perf.total_calls >= 3:
                exp_score = perf.success_rate
            else:
                # Use model's base quality score
                config = self.models.get(model_name)
                exp_score = config.quality_score if config else 0.5

            rankings.append((model_name, exp_score))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


# =============================================================================
# Complexity Estimators
# =============================================================================

def heuristic_complexity_estimator(
    ptool_spec: "PToolSpec",
    inputs: Dict[str, Any],
) -> TaskComplexity:
    """
    Estimate task complexity using heuristics.

    This is a fast, no-LLM estimator based on:
    - Input size
    - Ptool name patterns
    - Number of inputs

    Args:
        ptool_spec: Specification of the ptool
        inputs: Input arguments

    Returns:
        Estimated TaskComplexity
    """
    score = 0

    # Check input size
    total_chars = sum(
        len(str(v)) for v in inputs.values()
    )
    if total_chars > 5000:
        score += 2
    elif total_chars > 1000:
        score += 1

    # Check number of inputs
    if len(inputs) > 5:
        score += 1

    # Check ptool name for complexity hints
    name_lower = ptool_spec.name.lower()
    complex_keywords = ["analyze", "reason", "synthesize", "complex", "multi", "chain"]
    simple_keywords = ["extract", "format", "validate", "check", "get"]

    if any(kw in name_lower for kw in complex_keywords):
        score += 2
    if any(kw in name_lower for kw in simple_keywords):
        score -= 1

    # Check docstring for complexity hints
    if ptool_spec.docstring:
        doc_lower = ptool_spec.docstring.lower()
        if any(kw in doc_lower for kw in complex_keywords):
            score += 1

    # Map score to complexity
    if score <= 0:
        return TaskComplexity.TRIVIAL
    elif score == 1:
        return TaskComplexity.SIMPLE
    elif score == 2:
        return TaskComplexity.MODERATE
    elif score == 3:
        return TaskComplexity.COMPLEX
    else:
        return TaskComplexity.EXPERT


def estimate_task_complexity_llm(
    ptool_spec: "PToolSpec",
    inputs: Dict[str, Any],
    model: str = "deepseek-v3-0324",
    llm_backend: Optional[Callable] = None,
) -> TaskComplexity:
    """
    Estimate task complexity using an LLM.

    This is slower but more accurate for edge cases.

    Args:
        ptool_spec: Specification of the ptool
        inputs: Input arguments
        model: LLM to use for estimation
        llm_backend: Custom LLM backend

    Returns:
        Estimated TaskComplexity
    """
    from .llm_backend import call_llm

    prompt = f"""Estimate the complexity of this task:

PTOOL: {ptool_spec.name}
DESCRIPTION: {ptool_spec.docstring or 'No description'}
INPUTS: {json.dumps(inputs, default=str)[:500]}

Rate the complexity as one of:
- TRIVIAL: Very simple, one-step operation
- SIMPLE: Basic reasoning or extraction
- MODERATE: Some multi-step reasoning
- COMPLEX: Significant reasoning required
- EXPERT: Maximum difficulty

Respond with just the complexity level."""

    try:
        if llm_backend:
            response = llm_backend(prompt, model)
        else:
            response = call_llm(prompt, model)

        response_upper = response.strip().upper()
        for level in TaskComplexity:
            if level.value.upper() in response_upper:
                return level

    except Exception:
        pass

    # Fallback to heuristic
    return heuristic_complexity_estimator(ptool_spec, inputs)


# =============================================================================
# Convenience Functions
# =============================================================================

def select_model(
    ptool_name: str,
    inputs: Dict[str, Any],
    default: Optional[str] = None,
    required_tags: Optional[List[str]] = None,
) -> str:
    """
    Quick model selection for a ptool call.

    Args:
        ptool_name: Name of the ptool
        inputs: Input arguments
        default: Default model (uses LLMS.json default if None)
        required_tags: Only consider models with these tags

    Returns:
        Selected model name

    Example:
        >>> model = select_model("extract_values", {"text": "..."})
        >>> # For testing only
        >>> model = select_model("extract_values", {"text": "..."}, required_tags=["testing"])
    """
    # Create minimal spec
    from dataclasses import dataclass as dc

    @dc
    class MinimalSpec:
        name: str
        docstring: str = ""

    spec = MinimalSpec(name=ptool_name)
    selector = ModelSelector(default_model=default, required_tags=required_tags)
    return selector.select(spec, inputs)


def get_model_for_complexity(
    complexity: TaskComplexity,
    default: Optional[str] = None,
    required_tags: Optional[List[str]] = None,
) -> str:
    """
    Get recommended model for a complexity level.

    Uses enabled models from LLMS.json, selecting based on quality scores.

    Args:
        complexity: Task complexity
        default: Default model (uses LLMS.json default if None)
        required_tags: Only consider models with these tags

    Returns:
        Recommended model name

    Example:
        >>> model = get_model_for_complexity(TaskComplexity.COMPLEX)
        >>> # For testing only
        >>> model = get_model_for_complexity(TaskComplexity.COMPLEX, required_tags=["testing"])
    """
    # Load enabled models from LLMS.json
    models = load_llms_config(enabled_only=True, required_tags=required_tags)

    if not models:
        return default or get_default_model()

    # Quality thresholds for complexity
    thresholds = {
        TaskComplexity.TRIVIAL: 0.7,
        TaskComplexity.SIMPLE: 0.75,
        TaskComplexity.MODERATE: 0.85,
        TaskComplexity.COMPLEX: 0.90,
        TaskComplexity.EXPERT: 0.95,
    }

    min_quality = thresholds.get(complexity, 0.8)

    # Find models meeting the threshold, sorted by quality
    candidates = [
        (name, config)
        for name, config in models.items()
        if config.quality_score >= min_quality
    ]

    if candidates:
        # Sort by quality descending, then by cost ascending
        candidates.sort(key=lambda x: (-x[1].quality_score, x[1].cost_per_1k_input))
        return candidates[0][0]

    # Fallback to highest quality model available
    best = max(models.items(), key=lambda x: x[1].quality_score)
    return best[0]
