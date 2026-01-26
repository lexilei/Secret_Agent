"""
Real LLM integration test for the ReAct agent.

This test uses actual LLM calls to verify end-to-end functionality.
Requires TOGETHER_API_KEY environment variable.

Run with: python tests/test_react_integration.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptool_framework import (
    ptool,
    get_registry,
    react,
    ReActAgent,
    enable_tracing,
)


# ============================================================================
# Test Ptools - Simple math operations
# ============================================================================

@ptool(model="deepseek-v3")
def add(a: int, b: int) -> int:
    """Add two integers and return the sum.

    Examples:
    >>> add(2, 3)
    5
    >>> add(10, 20)
    30
    """
    ...


@ptool(model="deepseek-v3")
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product.

    Examples:
    >>> multiply(3, 4)
    12
    >>> multiply(5, 5)
    25
    """
    ...


@ptool(model="deepseek-v3")
def is_even(n: int) -> bool:
    """Check if an integer is even.

    Examples:
    >>> is_even(4)
    True
    >>> is_even(7)
    False
    """
    ...


# ============================================================================
# Integration Test
# ============================================================================

def test_real_react():
    """Test ReAct with real LLM calls."""
    print("=" * 60)
    print("ReAct Real LLM Integration Test")
    print("=" * 60)

    # Enable tracing
    enable_tracing(True)

    # Get available ptools
    registry = get_registry()
    ptools = [
        registry.get("add"),
        registry.get("multiply"),
        registry.get("is_even"),
    ]
    ptools = [p for p in ptools if p is not None]

    print(f"\nAvailable ptools: {[p.name for p in ptools]}")

    # Create agent
    agent = ReActAgent(
        available_ptools=ptools,
        model="deepseek-v3",
        max_steps=5,
        echo=True,
        store_trajectories=True,
    )

    # Simple test: Calculate (2 + 3) * 4
    print("\n" + "=" * 60)
    print("Test 1: Calculate (2 + 3) * 4")
    print("=" * 60)

    result = agent.run("Calculate (2 + 3) * 4. First add 2 and 3, then multiply the result by 4.")

    print(f"\n--- Result ---")
    print(f"Success: {result.success}")
    print(f"Answer: {result.answer}")
    print(f"Steps: {len(result.trajectory.steps)}")

    print(f"\n--- PTP Trace ---")
    print(result.trajectory.to_ptp_trace())

    if result.trace:
        print(f"\n--- Generated WorkflowTrace ---")
        for step in result.trace.steps:
            print(f"  {step.ptool_name}({step.args}) -> {step.result}")

    # Test 2: Check if sum is even
    print("\n" + "=" * 60)
    print("Test 2: Add 3 + 5 and check if the result is even")
    print("=" * 60)

    result2 = agent.run("Add 3 and 5 together, then check if the result is even.")

    print(f"\n--- Result ---")
    print(f"Success: {result2.success}")
    print(f"Answer: {result2.answer}")

    print(f"\n--- PTP Trace ---")
    print(result2.trajectory.to_ptp_trace())

    print("\n" + "=" * 60)
    print("Integration Test Complete")
    print("=" * 60)

    return result.success and result2.success


if __name__ == "__main__":
    success = test_real_react()
    sys.exit(0 if success else 1)
