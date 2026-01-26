#!/usr/bin/env python3
"""
Distilled Version: Same program but with @distilled functions.

This shows what the program would look like AFTER distillation:
- @distilled functions try pure Python first
- Fall back to LLM only when Python can't handle it
- Tracks success/fallback rates

Generated from trace analysis of 17 executions.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Literal

# Setup
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TOGETHER_API_KEY"] = "REDACTED_API_KEY"

from ptool_framework import ptool, distilled, DistillationFallback

# ============================================================================
# ORIGINAL PTOOLS (kept for fallback)
# ============================================================================

@ptool()
def extract_food_items(text: str) -> List[str]:
    """Extract all food items mentioned in the text."""
    ...


@ptool()
def is_healthy(food: str) -> bool:
    """Determine if a food item is generally considered healthy."""
    ...


@ptool()
def suggest_alternative(unhealthy_food: str) -> str:
    """Suggest a healthier alternative to an unhealthy food."""
    ...


# ============================================================================
# DISTILLED VERSIONS - Pure Python with LLM fallback
# ============================================================================

# Common food words learned from traces
KNOWN_FOODS = {
    'pizza', 'salad', 'eggs', 'bacon', 'toast', 'butter',
    'grilled chicken', 'chicken', 'quinoa', 'vegetables',
    'burger', 'fries', 'soda', 'apple', 'banana', 'rice',
    'pasta', 'steak', 'fish', 'salmon', 'broccoli', 'spinach'
}

HEALTHY_FOODS = {
    'salad', 'eggs', 'grilled chicken', 'chicken', 'quinoa',
    'vegetables', 'apple', 'banana', 'fish', 'salmon',
    'broccoli', 'spinach', 'toast', 'rice'
}

UNHEALTHY_FOODS = {
    'pizza', 'bacon', 'butter', 'burger', 'fries', 'soda',
    'fried chicken', 'hot dog', 'chips', 'candy'
}

ALTERNATIVES = {
    'pizza': 'whole wheat veggie pizza',
    'bacon': 'turkey bacon',
    'butter': 'avocado',
    'burger': 'veggie burger',
    'fries': 'baked sweet potato',
    'soda': 'sparkling water',
    'fried chicken': 'grilled chicken',
    'chips': 'mixed nuts',
    'candy': 'fresh fruit',
}


@distilled(fallback_ptool="extract_food_items")
def extract_food_items_distilled(text: str) -> List[str]:
    """
    Distilled implementation of extract_food_items.

    Tries regex matching against known foods first.
    Falls back to LLM for unknown patterns.

    Distilled from 3 successful traces.
    Estimated coverage: 70%
    """
    text_lower = text.lower()
    found = []

    # Try to find known foods
    for food in KNOWN_FOODS:
        if food in text_lower:
            found.append(food)

    if found:
        return found

    # Can't find any known foods - fall back to LLM
    raise DistillationFallback("No known food patterns matched")


@distilled(fallback_ptool="is_healthy")
def is_healthy_distilled(food: str) -> bool:
    """
    Distilled implementation of is_healthy.

    Uses lookup table for known foods.
    Falls back to LLM for unknown foods.

    Distilled from 9 successful traces.
    Estimated coverage: 85%
    """
    food_lower = food.lower()

    if food_lower in HEALTHY_FOODS:
        return True
    if food_lower in UNHEALTHY_FOODS:
        return False

    # Unknown food - fall back to LLM
    raise DistillationFallback(f"Unknown food: {food}")


@distilled(fallback_ptool="suggest_alternative")
def suggest_alternative_distilled(unhealthy_food: str) -> str:
    """
    Distilled implementation of suggest_alternative.

    Uses lookup table for common substitutions.
    Falls back to LLM for unknown foods.

    Distilled from 3 successful traces.
    Estimated coverage: 60%
    """
    food_lower = unhealthy_food.lower()

    if food_lower in ALTERNATIVES:
        return ALTERNATIVES[food_lower]

    # Unknown food - fall back to LLM
    raise DistillationFallback(f"No known alternative for: {unhealthy_food}")


# ============================================================================
# THE ACTUAL PROGRAM - Uses distilled versions
# ============================================================================

def analyze_meal(description: str) -> dict:
    """
    Analyze a meal description and suggest improvements.

    Now uses distilled functions for faster execution.
    Falls back to LLM for edge cases.
    """
    print(f"\n📝 Analyzing: '{description}'")

    # Step 1: Extract food items (tries Python first, falls back to LLM)
    foods = extract_food_items_distilled(description)
    print(f"🍽️  Found foods: {foods}")

    # Step 2: Categorize each food
    healthy = []
    unhealthy = []

    for food in foods:
        if is_healthy_distilled(food):
            healthy.append(food)
        else:
            unhealthy.append(food)

    print(f"✅ Healthy: {healthy}")
    print(f"❌ Unhealthy: {unhealthy}")

    # Step 3: Suggest alternatives for unhealthy foods
    suggestions = {}
    if unhealthy:
        print("💡 Suggestions:")
        for food in unhealthy:
            alt = suggest_alternative_distilled(food)
            suggestions[food] = alt
            print(f"   Replace '{food}' with '{alt}'")

    # Step 4: Calculate health score (pure Python)
    total = len(foods)
    health_score = (len(healthy) / total * 100) if total > 0 else 0

    return {
        "foods_found": foods,
        "healthy": healthy,
        "unhealthy": unhealthy,
        "suggestions": suggestions,
        "health_score": round(health_score, 1),
    }


# ============================================================================
# STATS DISPLAY
# ============================================================================

def show_distillation_stats():
    """Show how often each function used Python vs LLM."""
    from ptool_framework import get_distilled_stats

    print("\n=== Distillation Statistics ===")

    for func in [extract_food_items_distilled, is_healthy_distilled, suggest_alternative_distilled]:
        stats = get_distilled_stats(func)
        if stats:
            print(f"\n{func.__name__}:")
            print(f"  Total calls: {stats['total_calls']}")
            print(f"  Python success: {stats['python_success']} ({stats['success_rate']:.0%})")
            print(f"  LLM fallbacks: {stats['fallback_count']} ({stats['fallback_rate']:.0%})")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    # Run the same test cases
    test_meals = [
        "I had pizza and salad for lunch",
        "Breakfast was eggs, bacon, and toast with butter",
        "For dinner I ate grilled chicken with quinoa and vegetables",
        "I snacked on chips and candy",  # New case - should fallback
    ]

    for meal in test_meals:
        try:
            result = analyze_meal(meal)
            print(f"   Score: {result['health_score']}%")
        except Exception as e:
            print(f"   Error: {e}")

    # Show stats
    show_distillation_stats()
