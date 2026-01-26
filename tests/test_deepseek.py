#!/usr/bin/env python3
"""
Test script for ptool framework with DeepSeek V3 via Together.ai.

This script tests:
1. LLMS.json configuration loading
2. LLM backend routing to Together.ai
3. Basic ptool execution
4. Response parsing

Run with: python tests/test_deepseek.py
"""

import os
import sys
from pathlib import Path
from typing import Literal, List

# Add the framework to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import the framework
from ptool_framework import ptool, get_registry
from ptool_framework.llm_backend import (
    get_config,
    reload_config,
    call_llm,
    list_available_models,
    get_model_info,
)


def test_config_loading():
    """Test that LLMS.json is loaded correctly."""
    print("=" * 60)
    print("Test 1: Configuration Loading")
    print("=" * 60)

    config = reload_config()
    print(f"Default model: {config.default_model}")
    print(f"Available models: {list(config.models.keys())}")

    # Check deepseek-v3 config
    ds_config = config.get_model("deepseek-v3")
    print(f"\nDeepSeek V3 config:")
    print(f"  Provider: {ds_config.provider}")
    print(f"  Model ID: {ds_config.model_id}")
    print(f"  API Key Env: {ds_config.api_key_env}")
    print(f"  Capabilities: {ds_config.capabilities}")

    print("\nConfig loading: PASSED")
    return True


def test_list_models():
    """Test listing available models."""
    print("\n" + "=" * 60)
    print("Test 2: List Available Models")
    print("=" * 60)

    models = list_available_models()
    print(f"Found {len(models)} models:")
    for model in models:
        info = get_model_info(model)
        cost_str = info['cost'] if info['cost'] == 'local' else f"${info['cost'].get('input', '?')}/{info['cost'].get('output', '?')} per 1K tokens"
        print(f"  - {model}: {info['description'][:50]}... [{cost_str}]")

    print("\nList models: PASSED")
    return True


def test_simple_llm_call():
    """Test a simple LLM call to DeepSeek V3."""
    print("\n" + "=" * 60)
    print("Test 3: Simple LLM Call (DeepSeek V3)")
    print("=" * 60)

    prompt = """You are a helpful assistant. Answer concisely.

Question: What is 2 + 2?

Return your answer as JSON: {"result": <number>}"""

    print(f"Sending prompt to DeepSeek V3...")
    print(f"Prompt: {prompt[:100]}...")

    try:
        response = call_llm(prompt, model="deepseek-v3")
        print(f"\nResponse:\n{response}")
        print("\nSimple LLM call: PASSED")
        return True
    except Exception as e:
        print(f"\nError: {e}")
        print("Simple LLM call: FAILED")
        return False


def test_ptool_execution():
    """Test ptool execution with DeepSeek V3."""
    print("\n" + "=" * 60)
    print("Test 4: ptool Execution")
    print("=" * 60)

    # Define a simple ptool
    @ptool(model="deepseek-v3", output_mode="structured")
    def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
        """Classify the emotional tone of the given text.

        Analyze the text and determine if the overall sentiment is:
        - positive: expressing happiness, satisfaction, or approval
        - negative: expressing sadness, frustration, or disapproval
        - neutral: factual or without clear emotional tone

        Return exactly one of: positive, negative, neutral
        """
        ...

    # Check it's registered
    registry = get_registry()
    print(f"ptool registered: {'classify_sentiment' in registry}")

    # Test execution
    test_texts = [
        "I absolutely love this product! It's amazing!",
        "This is terrible. I'm very disappointed.",
        "The meeting is scheduled for 3pm tomorrow.",
    ]

    print("\nTesting sentiment classification:")
    for text in test_texts:
        print(f"\nText: {text[:50]}...")
        try:
            result = classify_sentiment(text)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
            return False

    print("\nptool execution: PASSED")
    return True


def test_ptool_list_output():
    """Test ptool with list output."""
    print("\n" + "=" * 60)
    print("Test 5: ptool with List Output")
    print("=" * 60)

    @ptool(model="deepseek-v3", output_mode="structured")
    def extract_keywords(text: str) -> List[str]:
        """Extract the main keywords from the given text.

        Identify 3-5 important keywords or phrases that capture
        the main topics discussed in the text.

        Return a JSON array of strings.
        """
        ...

    text = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."

    print(f"Text: {text}")
    try:
        keywords = extract_keywords(text)
        print(f"Keywords: {keywords}")
        print("\nList output: PASSED")
        return True
    except Exception as e:
        print(f"Error: {e}")
        print("List output: FAILED")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ptool Framework Test Suite - DeepSeek V3")
    print("=" * 60)
    print()

    results = []

    # Run tests
    results.append(("Config Loading", test_config_loading()))
    results.append(("List Models", test_list_models()))
    results.append(("Simple LLM Call", test_simple_llm_call()))
    results.append(("ptool Execution", test_ptool_execution()))
    results.append(("List Output", test_ptool_list_output()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
