#!/usr/bin/env python3
"""
Simple Example: A Python program with unimplemented parts handled by LLMs.

This shows the core idea:
- You write the Python structure (control flow, logic)
- You leave some functions unimplemented (just `...`)
- You wrap them with @ptool and specify types
- LLMs fill in the "thinking" at runtime
"""

import os
import sys
from pathlib import Path
from typing import List, Literal

# Setup
sys.path.insert(0, str(Path(__file__).parent))
os.environ["TOGETHER_API_KEY"] = "REDACTED_API_KEY"

from ptool_framework import ptool

# ============================================================================
# UNIMPLEMENTED FUNCTIONS - Just specify WHAT, not HOW
# ============================================================================

@ptool()
def extract_food_items(text: str) -> List[str]:
    """Extract all food items mentioned in the text.

    Return a list of food item names.
    Example: "I had pizza and salad" -> ["pizza", "salad"]
    """
    ...  # NOT IMPLEMENTED - LLM handles this


@ptool()
def is_healthy(food: str) -> bool:
    """Determine if a food item is generally considered healthy.

    Return True if healthy (vegetables, fruits, lean proteins, whole grains).
    Return False if unhealthy (fast food, sugary items, fried foods).
    """
    ...  # NOT IMPLEMENTED - LLM handles this


@ptool()
def suggest_alternative(unhealthy_food: str) -> str:
    """Suggest a healthier alternative to an unhealthy food.

    Return a single healthier food suggestion.
    Example: "french fries" -> "baked sweet potato"
    """
    ...  # NOT IMPLEMENTED - LLM handles this


# ============================================================================
# THE ACTUAL PROGRAM - You write this part (the logic/flow)
# ============================================================================

def analyze_meal(description: str) -> dict:
    """
    Analyze a meal description and suggest improvements.

    This is regular Python - YOU control the flow.
    The @ptool functions handle the "thinking" parts.
    """
    print(f"\n📝 Analyzing: '{description}'")

    # Step 1: Extract food items (LLM does this)
    foods = extract_food_items(description)
    print(f"🍽️  Found foods: {foods}")

    # Step 2: Check each food (LLM does the health判断)
    healthy_foods = []
    unhealthy_foods = []

    for food in foods:
        if is_healthy(food):  # LLM decides
            healthy_foods.append(food)
        else:
            unhealthy_foods.append(food)

    print(f"✅ Healthy: {healthy_foods}")
    print(f"❌ Unhealthy: {unhealthy_foods}")

    # Step 3: Suggest alternatives for unhealthy items (LLM does this)
    suggestions = {}
    for food in unhealthy_foods:
        alternative = suggest_alternative(food)  # LLM suggests
        suggestions[food] = alternative

    if suggestions:
        print(f"💡 Suggestions:")
        for bad, good in suggestions.items():
            print(f"   Replace '{bad}' with '{good}'")

    # Return structured result
    return {
        "foods_found": foods,
        "healthy": healthy_foods,
        "unhealthy": unhealthy_foods,
        "suggestions": suggestions,
        "health_score": len(healthy_foods) / len(foods) if foods else 0
    }


# ============================================================================
# RUN IT
# ============================================================================

if __name__ == "__main__":
    # Test meals
    test_meals = [
        "For lunch I had a burger with fries and a large soda",
        "I made a salad with grilled chicken, spinach, and quinoa",
        "Breakfast was donuts and coffee with extra sugar",
    ]

    print("=" * 60)
    print("MEAL ANALYZER - ptools in action!")
    print("=" * 60)

    for meal in test_meals:
        result = analyze_meal(meal)
        print(f"📊 Health Score: {result['health_score']:.0%}")
        print("-" * 60)
