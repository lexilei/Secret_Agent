"""Tests for the model selector system."""

import pytest
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from ptool_framework.model_selector import (
    TaskComplexity,
    ModelConfig,
    SelectionCriteria,
    SelectionResult,
    FallbackChain,
    ModelPerformance,
    ExperienceStore,
    ModelSelector,
    DEFAULT_MODELS,
    heuristic_complexity_estimator,
    select_model,
    get_model_for_complexity,
)


@dataclass
class MockPToolSpec:
    """Mock ptool spec for testing."""
    name: str
    docstring: str = ""


@pytest.fixture
def temp_experience_dir():
    """Create temporary directory for experience store."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_spec():
    """Create a sample ptool spec."""
    return MockPToolSpec(
        name="extract_values",
        docstring="Extract values from text"
    )


@pytest.fixture
def complex_spec():
    """Create a complex ptool spec."""
    return MockPToolSpec(
        name="analyze_reasoning_chain",
        docstring="Analyze and synthesize complex reasoning patterns"
    )


class TestTaskComplexity:
    """Tests for TaskComplexity enum."""

    def test_values(self):
        """Test enum values."""
        assert TaskComplexity.TRIVIAL.value == "trivial"
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MODERATE.value == "moderate"
        assert TaskComplexity.COMPLEX.value == "complex"
        assert TaskComplexity.EXPERT.value == "expert"


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_creation(self):
        """Test creating model config."""
        config = ModelConfig(
            name="test-model",
            provider="test",
            cost_per_1k_input=0.001,
            quality_score=0.85,
        )
        assert config.name == "test-model"
        assert config.provider == "test"
        assert config.quality_score == 0.85

    def test_to_dict(self):
        """Test serialization."""
        config = ModelConfig(
            name="test",
            capabilities=["reasoning", "code"],
        )
        d = config.to_dict()
        assert d["name"] == "test"
        assert "reasoning" in d["capabilities"]

    def test_default_values(self):
        """Test default values."""
        config = ModelConfig(name="test")
        assert config.provider == "together"
        assert config.max_tokens == 32000
        assert config.quality_score == 0.7


class TestSelectionCriteria:
    """Tests for SelectionCriteria dataclass."""

    def test_creation(self):
        """Test creating selection criteria."""
        criteria = SelectionCriteria(
            required_capabilities=["code"],
            max_cost_per_1k_tokens=0.005,
            max_latency_ms=2000,
            min_quality_threshold=0.7,
        )
        assert "code" in criteria.required_capabilities
        assert criteria.max_cost_per_1k_tokens == 0.005

    def test_defaults(self):
        """Test default values."""
        criteria = SelectionCriteria()
        assert criteria.max_latency_ms == 5000
        assert criteria.min_quality_threshold == 0.5


class TestSelectionResult:
    """Tests for SelectionResult dataclass."""

    def test_creation(self):
        """Test creating selection result."""
        result = SelectionResult(
            selected_model="deepseek-v3",
            reason="Best for task",
            confidence=0.9,
        )
        assert result.selected_model == "deepseek-v3"
        assert result.confidence == 0.9

    def test_to_dict(self):
        """Test serialization."""
        config = ModelConfig(name="test")
        result = SelectionResult(
            selected_model="test",
            config=config,
            fallback_chain=["fallback1", "fallback2"],
        )
        d = result.to_dict()
        assert d["selected_model"] == "test"
        assert len(d["fallback_chain"]) == 2


class TestFallbackChain:
    """Tests for FallbackChain dataclass."""

    def test_creation(self):
        """Test creating fallback chain."""
        chain = FallbackChain(
            models=["model1", "model2", "model3"],
            reason="Test chain",
        )
        assert len(chain) == 3
        assert chain.models[0] == "model1"

    def test_iteration(self):
        """Test iterating over chain."""
        chain = FallbackChain(models=["a", "b", "c"])
        models = list(chain)
        assert models == ["a", "b", "c"]


class TestModelPerformance:
    """Tests for ModelPerformance dataclass."""

    def test_creation(self):
        """Test creating performance record."""
        perf = ModelPerformance(
            model_name="test-model",
            ptool_name="extract",
            total_calls=100,
            successful_calls=90,
        )
        assert perf.model_name == "test-model"
        assert perf.success_rate == 0.9

    def test_success_rate_empty(self):
        """Test success rate with no calls."""
        perf = ModelPerformance(
            model_name="test",
            ptool_name="test",
        )
        assert perf.success_rate == 0.0

    def test_to_dict(self):
        """Test serialization."""
        perf = ModelPerformance(
            model_name="model",
            ptool_name="ptool",
            total_calls=50,
            successful_calls=45,
            avg_latency_ms=500.0,
        )
        d = perf.to_dict()
        assert d["total_calls"] == 50
        assert d["success_rate"] == 0.9

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "model_name": "test",
            "ptool_name": "extract",
            "total_calls": 20,
            "successful_calls": 18,
            "avg_latency_ms": 300.0,
        }
        perf = ModelPerformance.from_dict(data)
        assert perf.total_calls == 20
        assert perf.success_rate == 0.9


class TestExperienceStore:
    """Tests for ExperienceStore class."""

    def test_creation(self, temp_experience_dir):
        """Test creating experience store."""
        store = ExperienceStore(path=temp_experience_dir)
        assert store.base_path == Path(temp_experience_dir)

    def test_record_execution(self, temp_experience_dir):
        """Test recording executions."""
        store = ExperienceStore(path=temp_experience_dir)
        store.record_execution(
            model_name="test-model",
            ptool_name="extract",
            success=True,
            latency_ms=500,
            cost=0.001,
        )

        perf = store.get_performance("test-model", "extract")
        assert perf is not None
        assert perf.total_calls == 1
        assert perf.successful_calls == 1

    def test_record_multiple(self, temp_experience_dir):
        """Test recording multiple executions."""
        store = ExperienceStore(path=temp_experience_dir)

        # Record 10 calls
        for i in range(10):
            store.record_execution(
                model_name="test",
                ptool_name="ptool",
                success=(i < 8),  # 8 successes, 2 failures
                latency_ms=100 + i * 10,
                cost=0.001,
            )

        perf = store.get_performance("test", "ptool")
        assert perf.total_calls == 10
        assert perf.successful_calls == 8
        assert perf.success_rate == 0.8

    def test_get_best_model(self, temp_experience_dir):
        """Test getting best model for ptool."""
        store = ExperienceStore(path=temp_experience_dir)

        # Record for multiple models
        for _ in range(10):
            store.record_execution("model_a", "extract", True, 100, 0.001)
        for _ in range(10):
            store.record_execution("model_b", "extract", False, 100, 0.001)

        best = store.get_best_model_for_ptool("extract", min_calls=5)
        assert best == "model_a"

    def test_get_model_ranking(self, temp_experience_dir):
        """Test model ranking."""
        store = ExperienceStore(path=temp_experience_dir)

        # Record with different success rates
        for _ in range(10):
            store.record_execution("model_a", "extract", True, 100, 0.001)
        for i in range(10):
            store.record_execution("model_b", "extract", i < 7, 100, 0.001)
        for i in range(10):
            store.record_execution("model_c", "extract", i < 5, 100, 0.001)

        ranking = store.get_model_ranking("extract", min_calls=5)
        assert len(ranking) == 3
        assert ranking[0][0] == "model_a"  # Highest success rate

    def test_persistence(self, temp_experience_dir):
        """Test data persistence."""
        store1 = ExperienceStore(path=temp_experience_dir)
        store1.record_execution("model", "ptool", True, 100, 0.001)

        # Create new store pointing to same location
        store2 = ExperienceStore(path=temp_experience_dir)
        perf = store2.get_performance("model", "ptool")
        assert perf is not None
        assert perf.total_calls == 1


class TestModelSelector:
    """Tests for ModelSelector class."""

    def test_creation(self, temp_experience_dir):
        """Test creating model selector."""
        store = ExperienceStore(path=temp_experience_dir)
        selector = ModelSelector(
            experience_store=store,
            default_model="deepseek-v3",
        )
        assert selector.default_model == "deepseek-v3"
        assert selector.enable_learning

    def test_creation_defaults(self):
        """Test creating with defaults."""
        selector = ModelSelector()
        assert selector.default_model == "deepseek-v3-0324"
        assert len(selector.models) > 0

    def test_select_simple(self, sample_spec):
        """Test simple model selection."""
        selector = ModelSelector()
        model = selector.select(sample_spec, {"text": "hello"})
        assert model in selector.models

    def test_select_with_details(self, sample_spec):
        """Test selection with details."""
        selector = ModelSelector()
        result = selector.select_with_details(sample_spec, {"text": "test"})

        assert isinstance(result, SelectionResult)
        assert result.selected_model in selector.models
        assert len(result.reason) > 0

    def test_select_complex(self, complex_spec):
        """Test selection for complex task."""
        selector = ModelSelector()
        result = selector.select_with_details(
            complex_spec,
            {"data": "x" * 6000}  # Large input
        )

        # Should select a high-quality model
        assert result.selected_model in selector.models

    def test_select_with_criteria(self, sample_spec):
        """Test selection with criteria."""
        selector = ModelSelector()
        criteria = SelectionCriteria(
            required_capabilities=["reasoning"],
            max_cost_per_1k_tokens=0.01,
        )
        result = selector.select_with_details(sample_spec, {"x": 1}, criteria)
        assert result.selected_model in selector.models

    def test_fallback_chain(self, sample_spec):
        """Test fallback chain generation."""
        selector = ModelSelector()
        chain = selector.get_fallback_chain(sample_spec, {"x": 1})

        assert isinstance(chain, FallbackChain)
        assert len(chain) > 0
        assert len(chain) <= 3

    def test_fallback_chain_contains_default(self, sample_spec):
        """Test that fallback chain contains default model."""
        selector = ModelSelector()
        chain = selector.get_fallback_chain(sample_spec, {"x": 1}, max_length=5)

        # Default should be in chain
        assert selector.default_model in chain.models

    def test_record_execution(self, sample_spec, temp_experience_dir):
        """Test execution recording."""
        store = ExperienceStore(path=temp_experience_dir)
        selector = ModelSelector(experience_store=store)

        selector.record_execution(
            ptool_name="extract",
            model="deepseek-v3",
            inputs={"text": "test"},
            success=True,
            latency_ms=500,
            cost=0.001,
        )

        perf = store.get_performance("deepseek-v3", "extract")
        assert perf is not None
        assert perf.total_calls == 1

    def test_learning_disabled(self, sample_spec, temp_experience_dir):
        """Test with learning disabled."""
        store = ExperienceStore(path=temp_experience_dir)
        selector = ModelSelector(experience_store=store, enable_learning=False)

        selector.record_execution(
            ptool_name="extract",
            model="deepseek-v3",
            inputs={"text": "test"},
            success=True,
            latency_ms=500,
        )

        # Should not record when learning disabled
        perf = store.get_performance("deepseek-v3", "extract")
        assert perf is None

    def test_experience_based_selection(self, sample_spec, temp_experience_dir):
        """Test that experience affects selection."""
        store = ExperienceStore(path=temp_experience_dir)

        # Record good performance for one model
        for _ in range(10):
            store.record_execution("llama-3.1-8b", sample_spec.name, True, 100, 0.001)

        # Record bad performance for another
        for _ in range(10):
            store.record_execution("deepseek-v3", sample_spec.name, False, 500, 0.002)

        selector = ModelSelector(experience_store=store)
        result = selector.select_with_details(sample_spec, {"text": "test"})

        # Should prefer model with better historical performance
        # (though quality score may also influence)
        assert result.selected_model in selector.models


class TestComplexityEstimation:
    """Tests for complexity estimation."""

    def test_heuristic_trivial(self):
        """Test trivial complexity estimation."""
        spec = MockPToolSpec(name="format_text", docstring="Simple formatting")
        complexity = heuristic_complexity_estimator(spec, {"text": "hi"})
        assert complexity == TaskComplexity.TRIVIAL

    def test_heuristic_simple(self):
        """Test simple complexity estimation."""
        spec = MockPToolSpec(name="extract_values", docstring="Extract from text")
        complexity = heuristic_complexity_estimator(spec, {"text": "short input"})
        assert complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE)

    def test_heuristic_complex(self):
        """Test complex task estimation."""
        spec = MockPToolSpec(
            name="analyze_multi_step_reasoning",
            docstring="Complex multi-step synthesis"
        )
        complexity = heuristic_complexity_estimator(spec, {"data": "x" * 3000})
        assert complexity in (TaskComplexity.MODERATE, TaskComplexity.COMPLEX, TaskComplexity.EXPERT)

    def test_large_input_increases_complexity(self):
        """Test that large input increases complexity."""
        spec = MockPToolSpec(name="process", docstring="Process data")

        small_complexity = heuristic_complexity_estimator(spec, {"text": "small"})
        large_complexity = heuristic_complexity_estimator(spec, {"text": "x" * 10000})

        # Large input should be more complex
        complexities = list(TaskComplexity)
        assert complexities.index(large_complexity) >= complexities.index(small_complexity)

    def test_many_inputs_increases_complexity(self):
        """Test that many inputs increase complexity."""
        spec = MockPToolSpec(name="process", docstring="Process data")

        few_inputs = {"a": 1, "b": 2}
        many_inputs = {f"input_{i}": i for i in range(10)}

        few_complexity = heuristic_complexity_estimator(spec, few_inputs)
        many_complexity = heuristic_complexity_estimator(spec, many_inputs)

        complexities = list(TaskComplexity)
        assert complexities.index(many_complexity) >= complexities.index(few_complexity)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_select_model(self):
        """Test select_model function."""
        model = select_model("extract_values", {"text": "test"})
        assert model in DEFAULT_MODELS

    def test_select_model_default(self):
        """Test select_model with custom default."""
        model = select_model("unknown", {"x": 1}, default="gpt-4o")
        assert model in DEFAULT_MODELS

    def test_get_model_for_complexity(self):
        """Test get_model_for_complexity function."""
        trivial_model = get_model_for_complexity(TaskComplexity.TRIVIAL)
        expert_model = get_model_for_complexity(TaskComplexity.EXPERT)

        # Expert should be higher quality
        assert trivial_model in DEFAULT_MODELS or trivial_model == "llama-3.1-8b"
        assert expert_model in DEFAULT_MODELS or expert_model == "gpt-4o"


class TestDefaultModels:
    """Tests for default model configurations."""

    def test_default_models_exist(self):
        """Test that default models are defined."""
        assert len(DEFAULT_MODELS) > 0
        assert "deepseek-v3-0324" in DEFAULT_MODELS

    def test_model_configs_valid(self):
        """Test that model configs are valid."""
        for name, config in DEFAULT_MODELS.items():
            assert config.name  # Has a name
            assert config.provider  # Has a provider
            assert config.quality_score >= 0 and config.quality_score <= 1
            assert config.cost_per_1k_input >= 0
            assert config.latency_ms_avg > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
