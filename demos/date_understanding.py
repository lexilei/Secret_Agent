"""
Date Understanding Demo: BBH task using the ptool framework.

This demo shows how to convert the existing mock-based date_understanding
implementation to use real LLM-executed ptools.

The workflow is:
1. Extract options (multiple choice answers)
2. Extract date facts from the question
3. Make inferences from each fact
4. Extract the question
5. Answer the question using inferences
6. Match the answer to an option

Python controls the flow. LLMs (via ptools) do the reasoning.
"""

import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptool_framework import ptool, PToolSpec, get_registry, TraceExecutor
from ptool_framework.ptool import example
from ptool_framework.traces import WorkflowTrace, step, trace
from ptool_framework.trace_builder import TraceBuilder, build_trace

# Type alias for options
Option = Tuple[str, str]  # (letter, date)


# ============================================================================
# Define ptools - these are the "reasoning units" executed by LLM
# ============================================================================

@ptool(
    model="deepseek-v3.1",
    output_mode="structured",
    examples=[
        example(
            inputs={"input_str": "Q: What is 2+2?\nOptions:\n(A) 3\n(B) 4\n(C) 5"},
            output=[["A", "3"], ["B", "4"], ["C", "5"]],
        ),
    ],
)
def extract_options(input_str: str) -> List[List[str]]:
    """Extract the multiple-choice options from the input string.

    Parse the input and return a list of [letter, value] pairs.
    For date questions, the value is typically a date in MM/DD/YYYY format.

    Example output: [["A", "12/14/2026"], ["B", "12/14/1950"], ...]
    """
    ...


@ptool(
    model="deepseek-v3.1",
    output_mode="structured",
    examples=[
        example(
            inputs={"input_str": "Q: Today is Monday. Tomorrow is Tuesday. What day was yesterday?"},
            output=["Today is Monday.", "Tomorrow is Tuesday."],
        ),
    ],
)
def extract_date_facts(input_str: str) -> List[str]:
    """Extract the background facts about dates from the input string.

    These are statements that provide context about dates, times, or temporal
    relationships needed to answer the question.

    Return a list of fact strings.
    """
    ...


@ptool(
    model="deepseek-v3.1",
    output_mode="structured",
)
def extract_question(input_str: str) -> str:
    """Extract the question to be answered from the input string.

    Return just the question part, without the options.
    """
    ...


@ptool(
    model="deepseek-v3.1",
    output_mode="freeform",  # Use freeform for reasoning
)
def make_inference(date_fact: str, context: List[str]) -> str:
    """Given a background fact about dates and previous context, make a relevant inference.

    Reason about what the fact implies and produce a natural language inference.
    Consider temporal relationships, date calculations, and logical deductions.

    Return a single inference statement.
    """
    ...


@ptool(
    model="deepseek-v3.1",
    output_mode="freeform",
)
def answer_question(question: str, inferences: List[str]) -> str:
    """Given a question and a list of inferences about dates, construct an answer.

    Use the inferences to reason about the question and compute the answer.
    Return a natural language answer that includes the computed date.
    """
    ...


@ptool(
    model="deepseek-v3.1",
    output_mode="structured",
)
def match_option(answer: str, option: List[str]) -> bool:
    """Check if a multiple-choice option matches the answer.

    The option is [letter, value]. Check if the value appears in or matches the answer.
    Return True if it matches, False otherwise.
    """
    ...


# ============================================================================
# The workflow: Python controls flow, ptools do the reasoning
# ============================================================================

def date_understanding(input_str: str) -> Optional[Tuple[str, str]]:
    """
    Solve a date understanding problem.

    This is the main workflow - pure Python calling ptools.
    Python controls the flow. LLMs (ptools) do the reasoning.

    Args:
        input_str: The problem including question and options

    Returns:
        The best matching option as (letter, date) or None
    """
    # Step 1: Extract the multiple choice options
    options = extract_options(input_str)

    # Step 2: Extract date facts from the problem
    date_facts = extract_date_facts(input_str)

    # Step 3: Make inferences from each fact (building on previous)
    inferences = []
    for i, fact in enumerate(date_facts):
        inference = make_inference(fact, inferences)
        inferences.append(inference)

    # Step 4: Extract the question
    question = extract_question(input_str)

    # Step 5: Answer the question using inferences
    answer = answer_question(question, inferences)

    # Step 6: Find the matching option (pure Python control flow)
    best_opt = None
    for opt in options:
        if match_option(answer, opt):
            best_opt = tuple(opt)
            break

    if best_opt:
        print(f"Final answer: {best_opt[0]}")

    return best_opt


# ============================================================================
# Alternative: Build and execute as a trace
# ============================================================================

def date_understanding_via_trace(input_str: str) -> Optional[Tuple[str, str]]:
    """
    Solve using explicit trace building.

    This version builds a WorkflowTrace first, then executes it.
    Useful for inspection, debugging, and serialization.
    """
    # Build trace manually
    trace = (TraceBuilder(f"Answer date question: {input_str[:50]}...")
        .step("extract_options", {"input_str": input_str}, goal="Get options")
        .step("extract_date_facts", {"input_str": input_str}, goal="Get facts")
        .step("extract_question", {"input_str": input_str}, goal="Get question")
        .build()
    )

    # Execute
    executor = TraceExecutor()
    result = executor.execute(trace)

    if result.success:
        print(f"Trace completed: {result}")
        return result.final_result
    else:
        print(f"Trace failed: {result.error}")
        return None


# ============================================================================
# Test inputs (from original BBH task)
# ============================================================================

TEST_INPUTS = [
    """Q: Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?
Options:
(A) 12/14/2026
(B) 12/14/1950
(C) 12/14/2007
(D) 12/14/1937
(E) 07/14/1938
(F) 12/14/1988
""",
    """Q: Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?
Options:
(A) 09/04/2018
(B) 11/11/2018
(C) 08/25/2018
(D) 11/02/2018
(E) 11/04/2018
""",
    """Q: Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date tomorrow in MM/DD/YYYY?
Options:
(A) 01/11/1961
(B) 01/03/1963
(C) 01/18/1961
(D) 10/14/1960
(E) 01/03/1982
(F) 12/03/1960
""",
]

EXPECTED_ANSWERS = ["D", "B", "B"]


# ============================================================================
# Demo runner
# ============================================================================

def run_demo(use_mock: bool = True):
    """
    Run the date understanding demo.

    Args:
        use_mock: If True, use mock LLM backend (no API calls)
    """
    print("=" * 60)
    print("Date Understanding Demo - ptool Framework")
    print("=" * 60)
    print()

    # Show registered ptools
    registry = get_registry()
    print(f"Registered ptools: {len(registry)}")
    for spec in registry.list_all():
        print(f"  - {spec.get_signature_str()}")
    print()

    if use_mock:
        print("Using MOCK backend (no LLM calls)")
        print("To use real LLM: run_demo(use_mock=False)")
        print()

        # Set up mock backend
        from ptool_framework.llm_backend import MockLLMBackend
        mock = MockLLMBackend()

        # Add mock responses
        mock.add_response(
            "extract_options",
            '{"result": [["A", "12/14/2026"], ["B", "12/14/1950"], ["C", "12/14/2007"], ["D", "12/14/1937"], ["E", "07/14/1938"], ["F", "12/14/1988"]]}'
        )
        mock.add_response(
            "extract_date_facts",
            '{"result": ["Today is Christmas Eve of 1937."]}'
        )
        mock.add_response(
            "extract_question",
            '{"result": "What is the date 10 days ago in MM/DD/YYYY?"}'
        )
        mock.add_response(
            "make_inference",
            'Christmas Eve of 1937 means today is December 24, 1937.\nANSWER: If today is Christmas Eve of 1937, then today\'s date is 12/24/1937.'
        )
        mock.add_response(
            "answer_question",
            '10 days before December 24, 1937 is December 14, 1937.\nANSWER: 10 days before 12/24/1937 is 12/14/1937.'
        )
        mock.add_response(
            "match_option",
            '{"result": true}'
        )

        # Patch the LLM backend
        import ptool_framework.llm_backend as backend
        original_call = backend.call_llm
        backend.call_llm = mock

        try:
            # Run on first test input
            print("Test Input 1:")
            print(TEST_INPUTS[0][:100] + "...")
            print()

            result = date_understanding(TEST_INPUTS[0])
            print(f"Result: {result}")
            print(f"Expected: {EXPECTED_ANSWERS[0]}")
            print()
        finally:
            backend.call_llm = original_call

    else:
        print("Using REAL LLM backend")
        print("Make sure ANTHROPIC_API_KEY is set")
        print()

        # Run on first test input
        print("Test Input 1:")
        print(TEST_INPUTS[0])
        print()

        result = date_understanding(TEST_INPUTS[0])
        print(f"\nResult: {result}")
        print(f"Expected: {EXPECTED_ANSWERS[0]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Date Understanding Demo")
    parser.add_argument("--real", action="store_true", help="Use real LLM (requires API key)")
    args = parser.parse_args()

    run_demo(use_mock=not args.real)
