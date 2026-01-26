"""
Comprehensive tests for the ReAct agent implementation.

Run with: python -m pytest tests/test_react.py -v
Or directly: python tests/test_react.py
"""

import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptool_framework import (
    ptool,
    get_registry,
    Thought,
    Action,
    Observation,
    ReActStep,
    ReActTrajectory,
    ReActResult,
    ReActAgent,
    ReActStore,
    get_react_store,
    set_react_store,
    react,
    ActionParseError,
)
from ptool_framework.traces import WorkflowTrace, TraceStep, StepStatus


# ============================================================================
# Test Utilities
# ============================================================================

class MockLLMBackend:
    """Mock LLM backend for testing ReAct without actual API calls."""

    def __init__(self):
        self.responses: List[str] = []
        self.call_log: List[Dict[str, Any]] = []
        self.response_index = 0

    def add_response(self, response: str) -> None:
        """Add a response to the queue."""
        self.responses.append(response)

    def __call__(self, prompt: str, model: str) -> str:
        """Return next mock response."""
        self.call_log.append({"prompt": prompt, "model": model})

        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response

        # Default response
        return '{"result": "mock_default"}'

    def reset(self):
        """Reset the mock."""
        self.response_index = 0
        self.call_log = []


def create_temp_store() -> ReActStore:
    """Create a ReActStore with a temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="react_test_")
    return ReActStore(path=temp_dir)


# ============================================================================
# Test Ptools for ReAct
# ============================================================================

# Define some test ptools
@ptool(model="test-model")
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together and return the sum."""
    ...

@ptool(model="test-model")
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together and return the product."""
    ...

@ptool(model="test-model")
def extract_name(text: str) -> str:
    """Extract a person's name from the given text."""
    ...

@ptool(model="test-model")
def is_positive(number: int) -> bool:
    """Check if a number is positive (greater than zero)."""
    ...


# ============================================================================
# Test Cases
# ============================================================================

def test_thought_dataclass():
    """Test Thought dataclass creation and serialization."""
    print("\n=== Test: Thought Dataclass ===")

    thought = Thought(
        content="I need to add two numbers",
        step_number=0,
    )

    assert thought.content == "I need to add two numbers"
    assert thought.step_number == 0
    assert thought.timestamp is not None

    # Test serialization
    d = thought.to_dict()
    assert d["content"] == "I need to add two numbers"
    assert d["step_number"] == 0

    # Test deserialization
    thought2 = Thought.from_dict(d)
    assert thought2.content == thought.content
    assert thought2.step_number == thought.step_number

    print("  - Thought creation: OK")
    print("  - Thought serialization: OK")
    print("  - Thought deserialization: OK")
    return True


def test_action_dataclass():
    """Test Action dataclass creation and serialization."""
    print("\n=== Test: Action Dataclass ===")

    action = Action(
        ptool_name="add_numbers",
        args={"a": 5, "b": 3},
        step_number=0,
        rationale="Need to add 5 and 3",
        raw_action_text="<action>add_numbers(a=5, b=3)</action>",
    )

    assert action.ptool_name == "add_numbers"
    assert action.args == {"a": 5, "b": 3}

    # Test serialization roundtrip
    d = action.to_dict()
    action2 = Action.from_dict(d)
    assert action2.ptool_name == action.ptool_name
    assert action2.args == action.args

    print("  - Action creation: OK")
    print("  - Action serialization: OK")
    return True


def test_observation_dataclass():
    """Test Observation dataclass creation and serialization."""
    print("\n=== Test: Observation Dataclass ===")

    obs = Observation(
        result=8,
        success=True,
        step_number=0,
        execution_time_ms=150.5,
    )

    assert obs.result == 8
    assert obs.success is True
    assert obs.error is None

    # Test error case
    obs_error = Observation(
        result=None,
        success=False,
        error="Division by zero",
        step_number=1,
    )
    assert obs_error.success is False
    assert obs_error.error == "Division by zero"

    print("  - Observation success case: OK")
    print("  - Observation error case: OK")
    return True


def test_react_step():
    """Test ReActStep composite dataclass."""
    print("\n=== Test: ReActStep Dataclass ===")

    thought = Thought(content="Let me add numbers", step_number=0)
    action = Action(ptool_name="add_numbers", args={"a": 1, "b": 2}, step_number=0)
    obs = Observation(result=3, success=True, step_number=0)

    step = ReActStep(thought=thought, action=action, observation=obs)

    # Test serialization roundtrip
    d = step.to_dict()
    step2 = ReActStep.from_dict(d)

    assert step2.thought.content == step.thought.content
    assert step2.action.ptool_name == step.action.ptool_name
    assert step2.observation.result == step.observation.result

    print("  - ReActStep creation: OK")
    print("  - ReActStep serialization: OK")
    return True


def test_react_trajectory():
    """Test ReActTrajectory with PTP trace generation."""
    print("\n=== Test: ReActTrajectory ===")

    trajectory = ReActTrajectory(
        trajectory_id="test123",
        goal="Calculate 2 + 3",
    )

    # Add some steps
    step1 = ReActStep(
        thought=Thought(content="I'll add 2 and 3", step_number=0),
        action=Action(ptool_name="add_numbers", args={"a": 2, "b": 3}, step_number=0),
        observation=Observation(result=5, success=True, step_number=0),
    )
    step2 = ReActStep(
        thought=Thought(content="The answer is 5\n<answer>5</answer>", step_number=1),
        action=None,
        observation=None,
    )

    trajectory.steps = [step1, step2]
    trajectory.final_answer = "5"
    trajectory.success = True
    trajectory.termination_reason = "answer_found"

    # Test PTP trace generation
    ptp_trace = trajectory.to_ptp_trace()
    print(f"  - PTP Trace:\n{ptp_trace}")

    assert "Calling add_numbers" in ptp_trace
    assert "returned 5" in ptp_trace
    assert "Final answer: 5" in ptp_trace

    # Test JSON serialization
    json_str = trajectory.to_json()
    data = json.loads(json_str)
    assert data["trajectory_id"] == "test123"
    assert data["goal"] == "Calculate 2 + 3"
    assert data["ptp_trace"] == ptp_trace

    print("  - PTP trace generation: OK")
    print("  - JSON serialization: OK")
    return True


def test_action_parsing():
    """Test action parsing from thought content."""
    print("\n=== Test: Action Parsing ===")

    # Get test ptools
    registry = get_registry()
    ptools = [
        registry.get("add_numbers"),
        registry.get("multiply_numbers"),
    ]
    ptools = [p for p in ptools if p is not None]

    if not ptools:
        print("  - WARNING: No test ptools registered, skipping parsing test")
        return True

    agent = ReActAgent(
        available_ptools=ptools,
        llm_backend=MockLLMBackend(),
    )

    # Test basic parsing
    thought = Thought(
        content='<thought>I need to add</thought><action>add_numbers(a=5, b=3)</action>',
        step_number=0,
    )
    action = agent._parse_action(thought)
    assert action.ptool_name == "add_numbers"
    assert action.args == {"a": 5, "b": 3}
    print("  - Basic parsing: OK")

    # Test with string arguments
    agent2 = ReActAgent(
        available_ptools=[registry.get("extract_name")] if registry.get("extract_name") else ptools,
        llm_backend=MockLLMBackend(),
    )
    thought2 = Thought(
        content='<action>extract_name(text="Hello John")</action>',
        step_number=0,
    )
    if registry.get("extract_name"):
        action2 = agent2._parse_action(thought2)
        assert action2.args["text"] == "Hello John"
        print("  - String argument parsing: OK")

    # Test missing action tag
    thought_no_action = Thought(content="Just some thinking", step_number=0)
    try:
        agent._parse_action(thought_no_action)
        assert False, "Should have raised ActionParseError"
    except ActionParseError:
        print("  - Missing action tag detection: OK")

    # Test unknown ptool
    thought_unknown = Thought(
        content='<action>unknown_ptool(x=1)</action>',
        step_number=0,
    )
    try:
        agent._parse_action(thought_unknown)
        assert False, "Should have raised ActionParseError for unknown ptool"
    except ActionParseError as e:
        assert "unknown_ptool" in str(e).lower() or "Unknown" in str(e)
        print("  - Unknown ptool detection: OK")

    return True


def test_react_agent_with_mock():
    """Test ReActAgent with mock LLM backend."""
    print("\n=== Test: ReActAgent with Mock Backend ===")

    mock = MockLLMBackend()

    # Set up mock responses for a simple 2-step task
    # Step 1: Think and add
    mock.add_response(
        """<thought>I need to add 2 and 3 to get the sum.</thought>
<action>add_numbers(a=2, b=3)</action>"""
    )

    # Mock ptool execution response
    mock.add_response('{"result": 5}')

    # Step 2: Report answer
    mock.add_response(
        """<thought>The sum is 5. Task complete.</thought>
<answer>5</answer>"""
    )

    # Get test ptools
    registry = get_registry()
    ptools = [p for p in [registry.get("add_numbers")] if p is not None]

    if not ptools:
        print("  - WARNING: add_numbers ptool not found, skipping")
        return True

    agent = ReActAgent(
        available_ptools=ptools,
        model="test-model",
        max_steps=5,
        llm_backend=mock,
        store_trajectories=False,  # Don't store during test
        echo=True,
    )

    result = agent.run("Calculate 2 + 3")

    print(f"\n  - Success: {result.success}")
    print(f"  - Answer: {result.answer}")
    print(f"  - Steps: {len(result.trajectory.steps)}")

    assert result.success is True
    assert result.answer == "5"
    assert len(result.trajectory.steps) >= 1

    # Check trajectory
    assert result.trajectory.goal == "Calculate 2 + 3"
    assert result.trajectory.termination_reason == "answer_found"

    # Check trace was generated
    assert result.trace is not None
    print(f"  - Generated trace steps: {len(result.trace.steps)}")

    print("  - ReActAgent execution: OK")
    return True


def test_react_store():
    """Test ReActStore storage and retrieval."""
    print("\n=== Test: ReActStore ===")

    # Create temp store
    store = create_temp_store()

    try:
        # Create a trajectory
        trajectory = ReActTrajectory(
            trajectory_id="store_test_001",
            goal="Test storage",
            steps=[
                ReActStep(
                    thought=Thought(content="Testing storage", step_number=0),
                    action=Action(ptool_name="test", args={"x": 1}, step_number=0),
                    observation=Observation(result="ok", success=True, step_number=0),
                )
            ],
            final_answer="stored",
            success=True,
            termination_reason="answer_found",
            model_used="test-model",
        )

        # Store it
        store.store_trajectory(trajectory)
        print("  - Store trajectory: OK")

        # Retrieve it
        loaded = store.get_trajectory("store_test_001")
        assert loaded is not None
        assert loaded.trajectory_id == "store_test_001"
        assert loaded.goal == "Test storage"
        assert loaded.success is True
        print("  - Retrieve trajectory: OK")

        # Get PTP trace
        ptp = store.get_ptp_trace("store_test_001")
        assert ptp is not None
        assert "Goal: Test storage" in ptp
        print("  - Retrieve PTP trace: OK")

        # List trajectories
        summaries = store.list_trajectories()
        assert len(summaries) >= 1
        assert any(s["trajectory_id"] == "store_test_001" for s in summaries)
        print("  - List trajectories: OK")

        # Test successful-only filtering
        success_summaries = store.list_trajectories(success_only=True)
        assert all(s["success"] for s in success_summaries)
        print("  - Filter successful: OK")

        # Add a failed trajectory
        failed_traj = ReActTrajectory(
            trajectory_id="store_test_002",
            goal="Test failure",
            steps=[],
            success=False,
            termination_reason="max_steps",
        )
        store.store_trajectory(failed_traj)

        failed_summaries = store.list_trajectories(failed_only=True)
        assert len(failed_summaries) >= 1
        print("  - Store and filter failed: OK")

    finally:
        # Cleanup
        shutil.rmtree(store.base_path, ignore_errors=True)

    return True


def test_ptp_trace_format():
    """Test PTP trace format matches expected output."""
    print("\n=== Test: PTP Trace Format ===")

    trajectory = ReActTrajectory(
        trajectory_id="ptp_test",
        goal="Check sports consistency",
        steps=[
            ReActStep(
                thought=Thought(content="Extract player info", step_number=0),
                action=Action(
                    ptool_name="analyze_sentence",
                    args={"sentence": "DeMar DeRozan scored a goal."},
                    step_number=0,
                ),
                observation=Observation(
                    result=("DeMar DeRozan", "scored a goal", ""),
                    success=True,
                    step_number=0,
                ),
            ),
            ReActStep(
                thought=Thought(content="Check sport for player", step_number=1),
                action=Action(
                    ptool_name="sport_for",
                    args={"x": "DeMar DeRozan"},
                    step_number=1,
                ),
                observation=Observation(
                    result="basketball",
                    success=True,
                    step_number=1,
                ),
            ),
        ],
        final_answer="no",
        success=True,
    )

    ptp = trajectory.to_ptp_trace()
    print(f"Generated PTP trace:\n{ptp}\n")

    # Check format
    assert 'Calling analyze_sentence(sentence=' in ptp
    assert "...analyze_sentence returned" in ptp
    assert 'Calling sport_for(x=' in ptp
    assert "...sport_for returned 'basketball'" in ptp
    assert "Final answer: no" in ptp

    print("  - PTP format validation: OK")
    return True


def test_workflow_trace_generation():
    """Test that ReAct generates valid WorkflowTraces."""
    print("\n=== Test: WorkflowTrace Generation ===")

    mock = MockLLMBackend()

    # Simulate a multi-step task
    mock.add_response('<thought>Add first</thought><action>add_numbers(a=1, b=2)</action>')
    mock.add_response('{"result": 3}')
    mock.add_response('<thought>Multiply result</thought><action>multiply_numbers(a=3, b=4)</action>')
    mock.add_response('{"result": 12}')
    mock.add_response('<thought>Done</thought><answer>12</answer>')

    registry = get_registry()
    ptools = [
        p for p in [registry.get("add_numbers"), registry.get("multiply_numbers")]
        if p is not None
    ]

    if len(ptools) < 2:
        print("  - WARNING: Need add_numbers and multiply_numbers ptools")
        return True

    agent = ReActAgent(
        available_ptools=ptools,
        llm_backend=mock,
        store_trajectories=False,
    )

    result = agent.run("Calculate (1+2)*4")

    # Check the generated trace
    trace = result.trace
    assert trace is not None
    print(f"  - Trace steps: {len(trace.steps)}")

    for i, step in enumerate(trace.steps):
        print(f"    Step {i}: {step.ptool_name}({step.args}) -> {step.result}")
        assert step.status == StepStatus.COMPLETED
        assert step.result is not None

    # Verify trace is compatible with TraceExecutor format
    trace_dict = trace.to_dict()
    assert "goal" in trace_dict
    assert "steps" in trace_dict
    assert trace_dict["metadata"]["source"] == "react_agent"

    print("  - WorkflowTrace generation: OK")
    return True


def test_max_steps_termination():
    """Test that ReAct terminates at max_steps."""
    print("\n=== Test: Max Steps Termination ===")

    mock = MockLLMBackend()

    # Always return an action, never an answer
    for i in range(10):
        mock.add_response(f'<thought>Step {i}</thought><action>add_numbers(a=1, b=1)</action>')
        mock.add_response('{"result": 2}')

    registry = get_registry()
    ptools = [p for p in [registry.get("add_numbers")] if p is not None]

    if not ptools:
        print("  - WARNING: add_numbers ptool not found")
        return True

    agent = ReActAgent(
        available_ptools=ptools,
        llm_backend=mock,
        max_steps=3,  # Should stop after 3 steps
        store_trajectories=False,
    )

    result = agent.run("Loop forever")

    assert result.success is False
    assert result.trajectory.termination_reason == "max_steps"
    assert len(result.trajectory.steps) == 3

    print(f"  - Terminated at step {len(result.trajectory.steps)}")
    print("  - Max steps termination: OK")
    return True


def test_error_handling():
    """Test ReAct handles ptool execution errors gracefully."""
    print("\n=== Test: Error Handling ===")

    mock = MockLLMBackend()

    # First step: valid action
    mock.add_response('<thought>Add</thought><action>add_numbers(a=1, b=2)</action>')

    # Mock ptool throws error
    def error_backend(prompt, model):
        if "add_numbers" in str(mock.call_log):
            raise Exception("Simulated ptool error")
        return mock(prompt, model)

    # Actually let's use a simpler approach - mock returns error
    mock.add_response('ERROR: something went wrong')  # This will cause parse error

    # Then recover with answer
    mock.add_response('<thought>Had an error, giving up</thought><answer>error</answer>')

    registry = get_registry()
    ptools = [p for p in [registry.get("add_numbers")] if p is not None]

    if not ptools:
        print("  - WARNING: add_numbers ptool not found")
        return True

    agent = ReActAgent(
        available_ptools=ptools,
        llm_backend=mock,
        max_steps=5,
        store_trajectories=False,
    )

    # This should not crash
    result = agent.run("Handle errors")

    print(f"  - Completed with success={result.success}")
    print(f"  - Steps executed: {len(result.trajectory.steps)}")

    # Should have handled the error gracefully
    print("  - Error handling: OK")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("ReAct Agent Test Suite")
    print("=" * 60)

    tests = [
        ("Thought Dataclass", test_thought_dataclass),
        ("Action Dataclass", test_action_dataclass),
        ("Observation Dataclass", test_observation_dataclass),
        ("ReActStep", test_react_step),
        ("ReActTrajectory", test_react_trajectory),
        ("Action Parsing", test_action_parsing),
        ("ReActAgent with Mock", test_react_agent_with_mock),
        ("ReActStore", test_react_store),
        ("PTP Trace Format", test_ptp_trace_format),
        ("WorkflowTrace Generation", test_workflow_trace_generation),
        ("Max Steps Termination", test_max_steps_termination),
        ("Error Handling", test_error_handling),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            print(f"\n  ERROR: {e}")
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, success, error in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")
        if error:
            print(f"         Error: {error}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
