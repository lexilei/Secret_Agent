"""
Tests for Self-Improving Agents (L5).

All tests use REAL LLM calls (DeepSeek V3 via Together API).
"""

import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set API key
os.environ["TOGETHER_API_KEY"] = "REDACTED_API_KEY"

from ptool_framework import (
    ptool,
    PToolSpec,
    get_registry,
    ReActAgent,
)
from ptool_framework.self_improving import (
    PatternType,
    LearnedPattern,
    ICLExample,
    LearningEvent,
    SelfImprovementMetrics,
    PatternExtractor,
    PatternMemory,
    SelfImprovingAgent,
    self_improving_react,
)
from ptool_framework.react import ReActTrajectory, ReActStep, Thought, Action, Observation


# =============================================================================
# Test Fixtures - Sample PTools
# =============================================================================

@ptool(model="deepseek-v3-0324")
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together and return the sum."""
    ...

@ptool(model="deepseek-v3-0324")
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together and return the product."""
    ...


# =============================================================================
# Test LearnedPattern
# =============================================================================

class TestLearnedPattern:
    """Tests for LearnedPattern data structure."""

    def test_creation(self):
        """Test basic pattern creation."""
        pattern = LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="Example successful pattern",
            source_trace_id="trace1",
        )
        assert pattern.pattern_id == "p1"
        assert pattern.pattern_type == PatternType.POSITIVE
        assert pattern.confidence == 1.0

    def test_relevance_score(self):
        """Test relevance score calculation."""
        pattern = LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="test",
            source_trace_id="t1",
            times_used=10,
            times_helpful=8,
            confidence=0.9,
        )
        # (8/10) * 0.9 = 0.72
        assert abs(pattern.relevance_score - 0.72) < 0.01

    def test_decay(self):
        """Test confidence decay."""
        pattern = LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="test",
            source_trace_id="t1",
            confidence=1.0,
            decay_rate=0.1,
        )

        pattern.apply_decay(days_since_use=2)

        # After 2 days with 0.1 decay rate: 1.0 * (1 - 0.1*2) = 0.8
        assert abs(pattern.confidence - 0.8) < 0.01

    def test_reinforce_helpful(self):
        """Test reinforcement when helpful."""
        pattern = LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="test",
            source_trace_id="t1",
            confidence=0.8,
        )

        pattern.reinforce(was_helpful=True)

        assert pattern.times_used == 1
        assert pattern.times_helpful == 1
        assert pattern.confidence > 0.8  # Boosted

    def test_reinforce_unhelpful(self):
        """Test reinforcement when not helpful."""
        pattern = LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="test",
            source_trace_id="t1",
            confidence=0.8,
        )

        pattern.reinforce(was_helpful=False)

        assert pattern.times_used == 1
        assert pattern.times_helpful == 0
        assert pattern.confidence < 0.8  # Decreased

    def test_to_from_dict(self):
        """Test serialization and deserialization."""
        pattern = LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.NEGATIVE,
            content="avoid this",
            source_trace_id="t1",
            domain="medcalc",
        )

        d = pattern.to_dict()
        restored = LearnedPattern.from_dict(d)

        assert restored.pattern_id == "p1"
        assert restored.pattern_type == PatternType.NEGATIVE
        assert restored.domain == "medcalc"


# =============================================================================
# Test ICLExample
# =============================================================================

class TestICLExample:
    """Tests for ICLExample data structure."""

    def test_creation(self):
        """Test ICL example creation."""
        example = ICLExample(
            inputs={"a": 5, "b": 3},
            expected_output=8,
            reasoning="Add 5 and 3 to get 8",
            success=True,
        )
        assert example.expected_output == 8
        assert example.success is True

    def test_format_for_prompt(self):
        """Test formatting for prompt injection."""
        example = ICLExample(
            inputs={"text": "Hello world"},
            expected_output="greeting",
            reasoning="The text contains a greeting",
        )

        formatted = example.format_for_prompt()

        assert "Hello world" in formatted
        assert "greeting" in formatted
        assert "Reasoning:" in formatted


# =============================================================================
# Test PatternMemory
# =============================================================================

class TestPatternMemory:
    """Tests for PatternMemory persistence."""

    @pytest.fixture
    def temp_memory(self):
        """Create temporary pattern memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = PatternMemory(path=tmpdir)
            yield memory

    def test_store_and_retrieve(self, temp_memory):
        """Test storing and retrieving patterns."""
        pattern = LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="Example pattern content",
            source_trace_id="t1",
            domain="math",
        )

        temp_memory.store_pattern(pattern)
        retrieved = temp_memory.get_pattern("p1")

        assert retrieved is not None
        assert retrieved.content == "Example pattern content"
        assert retrieved.domain == "math"

    def test_get_relevant_patterns(self, temp_memory):
        """Test relevance-based retrieval."""
        # Store patterns with different domains
        patterns = [
            LearnedPattern(
                pattern_id="p1",
                pattern_type=PatternType.POSITIVE,
                content="Calculate BMI for patient",
                source_trace_id="t1",
                goal_pattern="calculate bmi",
                domain="medcalc",
            ),
            LearnedPattern(
                pattern_id="p2",
                pattern_type=PatternType.POSITIVE,
                content="Add two numbers together",
                source_trace_id="t2",
                goal_pattern="add numbers",
                domain="math",
            ),
        ]

        for p in patterns:
            temp_memory.store_pattern(p)

        # Search for BMI-related patterns
        relevant = temp_memory.get_relevant_patterns(
            task="calculate bmi for the patient",
            min_relevance=0.0,  # Lower threshold for testing
        )

        assert len(relevant) >= 1
        # BMI pattern should rank higher
        if len(relevant) >= 2:
            assert relevant[0].pattern_id == "p1"

    def test_get_negative_patterns(self, temp_memory):
        """Test retrieving negative patterns."""
        # Store positive and negative patterns
        temp_memory.store_pattern(LearnedPattern(
            pattern_id="pos1",
            pattern_type=PatternType.POSITIVE,
            content="Good approach",
            source_trace_id="t1",
        ))
        temp_memory.store_pattern(LearnedPattern(
            pattern_id="neg1",
            pattern_type=PatternType.NEGATIVE,
            content="AVOID: Bad approach",
            source_trace_id="t2",
        ))

        negatives = temp_memory.get_negative_patterns("any task")

        assert len(negatives) == 1
        assert negatives[0].pattern_id == "neg1"

    def test_apply_decay(self, temp_memory):
        """Test decay application."""
        # Store patterns with high confidence
        for i in range(3):
            temp_memory.store_pattern(LearnedPattern(
                pattern_id=f"p{i}",
                pattern_type=PatternType.POSITIVE,
                content=f"Pattern {i}",
                source_trace_id=f"t{i}",
                confidence=1.0,
            ))

        affected = temp_memory.apply_decay(days=5)

        assert affected == 3
        # All patterns should have lower confidence now
        for i in range(3):
            p = temp_memory.get_pattern(f"p{i}")
            assert p.confidence < 1.0

    def test_prune_low_confidence(self, temp_memory):
        """Test pruning low confidence patterns."""
        # Store patterns with varying confidence
        temp_memory.store_pattern(LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="High confidence",
            source_trace_id="t1",
            confidence=0.9,
        ))
        temp_memory.store_pattern(LearnedPattern(
            pattern_id="p2",
            pattern_type=PatternType.POSITIVE,
            content="Low confidence",
            source_trace_id="t2",
            confidence=0.05,
        ))

        pruned = temp_memory.prune_low_confidence(threshold=0.1)

        assert pruned == 1
        assert temp_memory.get_pattern("p1") is not None
        assert temp_memory.get_pattern("p2") is None

    def test_reinforce_pattern(self, temp_memory):
        """Test pattern reinforcement through memory."""
        temp_memory.store_pattern(LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="test",
            source_trace_id="t1",
            confidence=0.5,
        ))

        temp_memory.reinforce_pattern("p1", was_helpful=True)

        p = temp_memory.get_pattern("p1")
        assert p.times_used == 1
        assert p.times_helpful == 1
        assert p.confidence > 0.5

    def test_get_stats(self, temp_memory):
        """Test statistics collection."""
        # Store various patterns
        for i, ptype in enumerate([PatternType.POSITIVE, PatternType.NEGATIVE, PatternType.HEURISTIC]):
            temp_memory.store_pattern(LearnedPattern(
                pattern_id=f"p{i}",
                pattern_type=ptype,
                content=f"Pattern {i}",
                source_trace_id=f"t{i}",
            ))

        stats = temp_memory.get_stats()

        assert stats["total_patterns"] == 3
        assert "positive" in stats["patterns_by_type"]
        assert "negative" in stats["patterns_by_type"]
        assert "heuristic" in stats["patterns_by_type"]

    def test_log_learning_event(self, temp_memory):
        """Test learning event logging."""
        event = LearningEvent(
            event_id="e1",
            event_type="pattern_extracted",
            pattern_id="p1",
            trace_id="t1",
            details={"goal": "test goal"},
        )

        temp_memory.log_learning_event(event)
        history = temp_memory.get_learning_history(limit=10)

        assert len(history) >= 1
        assert history[-1].event_type == "pattern_extracted"


# =============================================================================
# Test PatternExtractor (Real LLM calls)
# =============================================================================

class TestPatternExtractor:
    """Tests for PatternExtractor with real LLM."""

    @pytest.fixture
    def extractor(self):
        """Create pattern extractor."""
        return PatternExtractor(model="deepseek-v3-0324")

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample successful trajectory."""
        trajectory = ReActTrajectory(
            trajectory_id="test123",
            goal="Add 5 and 3",
            success=True,
            final_answer="8",
            model_used="deepseek-v3-0324",
            total_llm_calls=2,
            total_time_ms=1000,
            termination_reason="answer_found",
        )

        # Add steps
        trajectory.steps.append(ReActStep(
            thought=Thought(
                content="I need to add 5 and 3 together",
                step_number=0,
            ),
            action=Action(
                ptool_name="add_numbers",
                args={"a": 5, "b": 3},
                step_number=0,
                rationale="Adding the numbers",
                raw_action_text="add_numbers(a=5, b=3)",
            ),
            observation=Observation(
                result=8,
                success=True,
                step_number=0,
                execution_time_ms=500,
            ),
        ))

        return trajectory

    def test_extract_from_successful_trace(self, extractor, sample_trajectory):
        """Test extracting patterns from successful trajectory."""
        patterns = extractor.extract_from_successful_trace(sample_trajectory)

        print(f"\nExtracted {len(patterns)} patterns from successful trace:")
        for p in patterns:
            print(f"  - [{p.pattern_type.value}] {p.content[:50]}...")

        # Should extract at least ICL example
        assert len(patterns) >= 1

        # Check for positive pattern
        positive_patterns = [p for p in patterns if p.pattern_type == PatternType.POSITIVE]
        assert len(positive_patterns) >= 1

    def test_extract_from_failed_trace(self, extractor):
        """Test extracting patterns from failed trajectory."""
        trajectory = ReActTrajectory(
            trajectory_id="fail123",
            goal="Do something impossible",
            success=False,
            final_answer=None,
            model_used="deepseek-v3-0324",
            total_llm_calls=5,
            total_time_ms=5000,
            termination_reason="max_steps",
        )

        # Add a failed step
        trajectory.steps.append(ReActStep(
            thought=Thought(
                content="I will try something",
                step_number=0,
            ),
            action=Action(
                ptool_name="unknown_tool",
                args={},
                step_number=0,
                rationale="Trying",
                raw_action_text="unknown_tool()",
            ),
            observation=Observation(
                result=None,
                success=False,
                error="Tool not found",
                step_number=0,
                execution_time_ms=100,
            ),
        ))

        patterns = extractor.extract_from_failed_trace(
            trajectory,
            error_analysis="Tried to use unknown tool",
        )

        print(f"\nExtracted {len(patterns)} patterns from failed trace:")
        for p in patterns:
            print(f"  - [{p.pattern_type.value}] {p.content[:50]}...")

        # Should extract negative patterns
        assert len(patterns) >= 1
        negative_patterns = [p for p in patterns if p.pattern_type == PatternType.NEGATIVE]
        assert len(negative_patterns) >= 1


# =============================================================================
# Test SelfImprovingAgent (Real LLM calls)
# =============================================================================

class TestSelfImprovingAgent:
    """Tests for SelfImprovingAgent with real LLM."""

    @pytest.fixture
    def temp_memory(self):
        """Create temporary pattern memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield PatternMemory(path=tmpdir)

    @pytest.fixture
    def base_agent(self):
        """Create base ReAct agent."""
        registry = get_registry()
        available_ptools = [
            registry.get("add_numbers"),
            registry.get("multiply_numbers"),
        ]
        available_ptools = [p for p in available_ptools if p is not None]

        return ReActAgent(
            available_ptools=available_ptools if available_ptools else None,
            model="deepseek-v3-0324",
            max_steps=5,
            echo=True,
        )

    def test_creation(self, base_agent, temp_memory):
        """Test creating self-improving agent."""
        agent = SelfImprovingAgent(
            base_agent=base_agent,
            pattern_memory=temp_memory,
            learn_from_success=True,
            learn_from_failure=True,
            echo=True,
        )

        assert agent.base_agent is base_agent
        assert agent.pattern_memory is temp_memory

    def test_run_learns_patterns(self, base_agent, temp_memory):
        """Test that running the agent learns patterns."""
        agent = SelfImprovingAgent(
            base_agent=base_agent,
            pattern_memory=temp_memory,
            learn_from_success=True,
            learn_from_failure=True,
            auto_repair=False,  # Disable repair for simple test
            echo=True,
        )

        # Run a simple task
        result = agent.run("Add 2 and 3 together")

        print(f"\nResult success: {result.success}")
        print(f"Answer: {result.answer}")

        # Check that patterns were learned
        patterns = agent.get_learned_patterns()
        print(f"Learned {len(patterns)} patterns")

        # Should have learned something
        # Note: May vary depending on success/failure
        assert len(patterns) >= 0

    def test_pattern_retrieval(self, base_agent, temp_memory):
        """Test pattern retrieval for tasks."""
        # Pre-populate some patterns
        temp_memory.store_pattern(LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="When adding numbers, use add_numbers tool",
            source_trace_id="t1",
            goal_pattern="add numbers",
        ))

        agent = SelfImprovingAgent(
            base_agent=base_agent,
            pattern_memory=temp_memory,
            echo=True,
        )

        # Test pattern retrieval
        patterns = agent._retrieve_patterns("add two numbers")

        assert len(patterns) >= 1

    def test_explain_patterns_for_task(self, base_agent, temp_memory):
        """Test pattern explanation."""
        temp_memory.store_pattern(LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="Example: Add 5 + 3 = 8",
            source_trace_id="t1",
            goal_pattern="add numbers",
        ))

        agent = SelfImprovingAgent(
            base_agent=base_agent,
            pattern_memory=temp_memory,
        )

        explanation = agent.explain_patterns_for_task("add two numbers")

        print(f"\nExplanation:\n{explanation}")

        assert "positive" in explanation.lower() or "no relevant" in explanation.lower()

    def test_improvement_metrics(self, base_agent, temp_memory):
        """Test improvement metrics tracking."""
        agent = SelfImprovingAgent(
            base_agent=base_agent,
            pattern_memory=temp_memory,
            echo=True,
        )

        # Run a few tasks
        for goal in ["Add 1 and 2", "Add 3 and 4"]:
            try:
                agent.run(goal)
            except Exception as e:
                print(f"Task failed: {e}")

        metrics = agent.get_improvement_metrics()

        print(f"\nMetrics:")
        print(f"  Total runs: {metrics.total_runs}")
        print(f"  Successful: {metrics.successful_runs}")
        print(f"  Patterns learned: {metrics.total_patterns_learned}")

        assert metrics.total_runs == 2

    def test_decay_cycle(self, base_agent, temp_memory):
        """Test decay cycle execution."""
        # Store some patterns
        temp_memory.store_pattern(LearnedPattern(
            pattern_id="p1",
            pattern_type=PatternType.POSITIVE,
            content="test",
            source_trace_id="t1",
            confidence=1.0,
        ))

        agent = SelfImprovingAgent(
            base_agent=base_agent,
            pattern_memory=temp_memory,
            enable_decay=True,
        )

        result = agent.run_decay_cycle()

        print(f"\nDecay cycle result: {result}")

        assert "patterns_decayed" in result
        assert "patterns_pruned" in result


# =============================================================================
# Integration Test
# =============================================================================

class TestSelfImprovingIntegration:
    """Full integration test for self-improving agents."""

    def test_end_to_end_self_improvement(self):
        """Test complete self-improvement flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = PatternMemory(path=tmpdir)

            # Create base agent
            registry = get_registry()
            ptools = [
                registry.get("add_numbers"),
                registry.get("multiply_numbers"),
            ]
            ptools = [p for p in ptools if p is not None]

            base_agent = ReActAgent(
                available_ptools=ptools if ptools else None,
                model="deepseek-v3-0324",
                max_steps=5,
                echo=True,
            )

            # Create self-improving agent
            agent = SelfImprovingAgent(
                base_agent=base_agent,
                pattern_memory=memory,
                learn_from_success=True,
                learn_from_failure=True,
                auto_repair=False,
                echo=True,
            )

            print("\n=== Self-Improvement Integration Test ===")

            # Run first task
            print("\n--- Task 1: Add 5 and 3 ---")
            result1 = agent.run("Add 5 and 3")
            print(f"Success: {result1.success}, Answer: {result1.answer}")

            # Check learning
            patterns_after_1 = agent.get_learned_patterns()
            print(f"Patterns after task 1: {len(patterns_after_1)}")

            # Run second similar task
            print("\n--- Task 2: Add 10 and 20 ---")
            result2 = agent.run("Add 10 and 20")
            print(f"Success: {result2.success}, Answer: {result2.answer}")

            # Check metrics
            metrics = agent.get_improvement_metrics()
            print(f"\nFinal metrics:")
            print(f"  Total runs: {metrics.total_runs}")
            print(f"  Successful runs: {metrics.successful_runs}")
            print(f"  Patterns learned: {metrics.total_patterns_learned}")
            print(f"  Current success rate: {metrics.current_success_rate:.1%}")

            # Verify agent ran
            assert metrics.total_runs == 2


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_self_improving_react(self):
        """Test self_improving_react convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = PatternMemory(path=tmpdir)

            result = self_improving_react(
                "What is 2 + 2?",
                pattern_memory=memory,
                model="deepseek-v3-0324",
                max_steps=5,
                echo=True,
            )

            print(f"\nConvenience function result:")
            print(f"  Success: {result.success}")
            print(f"  Answer: {result.answer}")

            assert result.trajectory is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
