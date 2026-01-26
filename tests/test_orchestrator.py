"""
Tests for Multi-Agent Orchestration (L4).

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
)
from ptool_framework.orchestrator import (
    ExecutionMode,
    AgentSpec,
    RoutingDecision,
    OrchestrationStep,
    OrchestrationTrace,
    OrchestrationResult,
    RuleBasedRouter,
    LLMRouter,
    ExperienceBasedRouter,
    HybridRouter,
    AgentOrchestrator,
    OrchestrationStore,
)


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

@ptool(model="deepseek-v3-0324")
def extract_value(text: str, key: str) -> str:
    """Extract a value from text based on the given key."""
    ...


# =============================================================================
# Test AgentSpec
# =============================================================================

class TestAgentSpec:
    """Tests for AgentSpec data structure."""

    def test_creation(self):
        """Test basic agent spec creation."""
        spec = AgentSpec(
            name="calculator",
            description="Performs mathematical calculations",
            domains=["math", "calculation"],
            available_ptools=["add_numbers", "multiply_numbers"],
        )
        assert spec.name == "calculator"
        assert "math" in spec.domains
        assert spec.model == "deepseek-v3-0324"

    def test_to_dict(self):
        """Test serialization to dict."""
        spec = AgentSpec(
            name="test",
            description="Test agent",
            domains=["test"],
            available_ptools=["ptool1"],
            capabilities=["reasoning"],
        )
        d = spec.to_dict()
        assert d["name"] == "test"
        assert d["capabilities"] == ["reasoning"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "name": "test",
            "description": "Test agent",
            "domains": ["test"],
            "available_ptools": ["ptool1"],
        }
        spec = AgentSpec.from_dict(data)
        assert spec.name == "test"

    def test_format_for_prompt(self):
        """Test prompt formatting."""
        spec = AgentSpec(
            name="calculator",
            description="Math operations",
            domains=["math"],
            available_ptools=["add_numbers"],
            capabilities=["arithmetic"],
        )
        formatted = spec.format_for_prompt()
        assert "calculator" in formatted
        assert "Math operations" in formatted
        assert "math" in formatted


# =============================================================================
# Test RuleBasedRouter
# =============================================================================

class TestRuleBasedRouter:
    """Tests for RuleBasedRouter."""

    @pytest.fixture
    def sample_agents(self):
        return [
            AgentSpec(
                name="calculator",
                description="Performs calculations",
                domains=["math", "calculation"],
                available_ptools=["add_numbers", "multiply_numbers"],
            ),
            AgentSpec(
                name="extractor",
                description="Extracts information",
                domains=["extraction", "parsing"],
                available_ptools=["extract_value"],
            ),
        ]

    def test_keyword_matching(self, sample_agents):
        """Test routing based on keywords."""
        router = RuleBasedRouter()
        router.add_domain_keywords("math", ["add", "multiply", "calculate", "sum"])
        router.add_domain_keywords("extraction", ["extract", "parse", "get"])

        # Should route to calculator
        decision = router.route("add two numbers together", sample_agents)
        assert decision.selected_agent == "calculator"
        assert decision.confidence > 0.0
        assert decision.routing_method == "rules"

    def test_extraction_routing(self, sample_agents):
        """Test routing to extractor agent."""
        router = RuleBasedRouter()
        router.add_domain_keywords("extraction", ["extract", "parse", "get", "find"])

        decision = router.route("extract the patient name from text", sample_agents)
        assert decision.selected_agent == "extractor"

    def test_no_match_fallback(self, sample_agents):
        """Test fallback when no keywords match."""
        router = RuleBasedRouter()
        # No keywords added

        decision = router.route("do something random", sample_agents)
        assert decision.selected_agent in ["calculator", "extractor"]
        assert decision.confidence < 0.5


# =============================================================================
# Test LLMRouter (Real LLM calls)
# =============================================================================

class TestLLMRouter:
    """Tests for LLMRouter using real DeepSeek V3 calls."""

    @pytest.fixture
    def sample_agents(self):
        return [
            AgentSpec(
                name="calculator",
                description="Performs mathematical calculations like addition, multiplication",
                domains=["math", "calculation"],
                available_ptools=["add_numbers", "multiply_numbers"],
                capabilities=["arithmetic"],
            ),
            AgentSpec(
                name="extractor",
                description="Extracts specific values and information from text",
                domains=["extraction", "parsing"],
                available_ptools=["extract_value"],
                capabilities=["text_processing"],
            ),
        ]

    def test_llm_routing_math(self, sample_agents):
        """Test LLM routes math tasks to calculator."""
        router = LLMRouter(model="deepseek-v3-0324")

        decision = router.route(
            "Calculate the sum of 5 and 10",
            sample_agents
        )

        print(f"LLM Decision: {decision.selected_agent}")
        print(f"Reason: {decision.reason}")

        assert decision.selected_agent == "calculator"
        assert decision.routing_method == "llm"

    def test_llm_routing_extraction(self, sample_agents):
        """Test LLM routes extraction tasks correctly."""
        router = LLMRouter(model="deepseek-v3-0324")

        decision = router.route(
            "Extract the patient's age from the medical record",
            sample_agents
        )

        print(f"LLM Decision: {decision.selected_agent}")
        print(f"Reason: {decision.reason}")

        assert decision.selected_agent == "extractor"


# =============================================================================
# Test HybridRouter
# =============================================================================

class TestHybridRouter:
    """Tests for HybridRouter combining strategies."""

    @pytest.fixture
    def sample_agents(self):
        return [
            AgentSpec(
                name="calculator",
                description="Performs mathematical calculations",
                domains=["math", "calculation"],
                available_ptools=["add_numbers", "multiply_numbers"],
            ),
            AgentSpec(
                name="extractor",
                description="Extracts information from text",
                domains=["extraction", "parsing"],
                available_ptools=["extract_value"],
            ),
        ]

    def test_hybrid_uses_rules_first(self, sample_agents):
        """Test that hybrid router tries rules first."""
        router = HybridRouter(confidence_threshold=0.5)

        decision = router.route("calculate the sum", sample_agents)

        # Should use rules because "calculate" matches math domain
        assert decision.selected_agent == "calculator"
        assert "[Rules]" in decision.reason or decision.confidence >= 0.5

    def test_hybrid_falls_back_to_llm(self, sample_agents):
        """Test that hybrid router falls back to LLM for ambiguous tasks."""
        router = HybridRouter(confidence_threshold=0.9)  # High threshold

        # Ambiguous task - rules won't be confident
        decision = router.route(
            "Help me with this complex task that requires intelligence",
            sample_agents
        )

        print(f"Hybrid Decision: {decision.selected_agent}")
        print(f"Method: {decision.reason}")

        # Should get a decision from one of the methods
        assert decision.selected_agent in ["calculator", "extractor"]


# =============================================================================
# Test AgentOrchestrator (Real LLM calls)
# =============================================================================

class TestAgentOrchestrator:
    """Tests for AgentOrchestrator with real LLM calls."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with sample agents."""
        orch = AgentOrchestrator(
            execution_mode=ExecutionMode.SEQUENTIAL,
            store_traces=False,  # Don't persist for tests
            echo=True,
        )

        # Register calculator agent
        orch.register_agent(AgentSpec(
            name="calculator",
            description="Performs mathematical calculations",
            domains=["math", "calculation"],
            available_ptools=["add_numbers", "multiply_numbers"],
            max_steps=5,
        ))

        # Register extractor agent
        orch.register_agent(AgentSpec(
            name="extractor",
            description="Extracts information from text",
            domains=["extraction", "parsing"],
            available_ptools=["extract_value"],
            max_steps=5,
        ))

        return orch

    def test_register_agents(self, orchestrator):
        """Test agent registration."""
        agents = orchestrator.list_agents()
        assert len(agents) == 2
        assert any(a.name == "calculator" for a in agents)
        assert any(a.name == "extractor" for a in agents)

    def test_get_agents_for_domain(self, orchestrator):
        """Test domain-based agent lookup."""
        math_agents = orchestrator.get_agents_for_domain("math")
        assert len(math_agents) == 1
        assert math_agents[0].name == "calculator"

    def test_run_simple_task(self, orchestrator):
        """Test running a simple task through orchestrator."""
        result = orchestrator.run("Add 5 and 3 together")

        print(f"Success: {result.success}")
        print(f"Answer: {result.final_answer}")
        print(f"Agents used: {result.agents_used}")

        assert result.trace is not None
        assert len(result.agents_used) > 0
        # Note: May or may not succeed depending on available ptools

    def test_run_sequential_tasks(self, orchestrator):
        """Test running multiple tasks sequentially."""
        tasks = [
            "Calculate 2 + 3",
            "Calculate 4 times 5",
        ]

        result = orchestrator.run_sequential(tasks)

        print(f"Success: {result.success}")
        print(f"Steps: {len(result.trace.steps)}")

        assert result.trace is not None
        assert len(result.trace.steps) == len(tasks)

    def test_decompose_task(self, orchestrator):
        """Test task decomposition with LLM."""
        complex_goal = "Extract the numbers from the text and then calculate their sum"

        subtasks = orchestrator.decompose_task(complex_goal)

        print(f"Decomposed into: {subtasks}")

        assert len(subtasks) >= 1
        assert isinstance(subtasks[0], str)


# =============================================================================
# Test OrchestrationStore
# =============================================================================

class TestOrchestrationStore:
    """Tests for OrchestrationStore persistence."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OrchestrationStore(path=tmpdir)
            yield store

    def test_store_and_retrieve(self, temp_store):
        """Test storing and retrieving traces."""
        trace = OrchestrationTrace(
            trace_id="test123",
            goal="Test goal",
            success=True,
            agents_used=["calculator"],
        )

        # Add a step
        trace.steps.append(OrchestrationStep(
            step_id="step1",
            agent_name="calculator",
            task="Add numbers",
            routing_decision=RoutingDecision(
                selected_agent="calculator",
                confidence=0.9,
                reason="Test routing",
            ),
            status="completed",
        ))

        temp_store.store_trace(trace)

        # Retrieve
        retrieved = temp_store.get_trace("test123")
        assert retrieved is not None
        assert retrieved.goal == "Test goal"
        assert retrieved.success is True

    def test_get_stats(self, temp_store):
        """Test statistics collection."""
        # Store some traces
        for i in range(3):
            trace = OrchestrationTrace(
                trace_id=f"trace{i}",
                goal=f"Goal {i}",
                success=(i % 2 == 0),
                agents_used=["calculator"],
            )
            temp_store.store_trace(trace)

        stats = temp_store.get_stats()
        assert stats["total_traces"] == 3
        assert stats["successful_traces"] == 2  # 0 and 2 are successful


# =============================================================================
# Integration Test
# =============================================================================

class TestOrchestratorIntegration:
    """Full integration test with real LLM."""

    def test_end_to_end_orchestration(self):
        """Test complete orchestration flow."""
        # Create orchestrator
        orchestrator = AgentOrchestrator(
            execution_mode=ExecutionMode.SEQUENTIAL,
            store_traces=False,
            echo=True,
        )

        # Register a math agent
        orchestrator.register_agent(AgentSpec(
            name="math_agent",
            description="Performs mathematical operations like addition and multiplication",
            domains=["math", "arithmetic", "calculation"],
            available_ptools=["add_numbers", "multiply_numbers"],
            capabilities=["addition", "multiplication"],
            max_steps=5,
        ))

        # Run a task
        result = orchestrator.run("What is 2 plus 3?")

        print("\n=== Integration Test Results ===")
        print(f"Success: {result.success}")
        print(f"Answer: {result.final_answer}")
        print(f"Agents used: {result.agents_used}")
        print(f"Routing decisions: {len(result.routing_decisions)}")

        if result.routing_decisions:
            rd = result.routing_decisions[0]
            print(f"  Selected: {rd.selected_agent}")
            print(f"  Confidence: {rd.confidence}")
            print(f"  Method: {rd.routing_method}")

        # Basic assertions
        assert result.trace is not None
        assert result.trace.trace_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
