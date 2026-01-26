"""
Distilled Functions: Pure Python implementations with LLM fallback.

The @distilled decorator marks a function that:
1. Has a pure Python implementation (distilled from LLM traces)
2. Falls back to the original @ptool LLM call when the Python implementation fails
3. Logs fallbacks for future improvement analysis

This is William's "backoff" strategy - less agentic systems fall back to
more general agentic systems when they encounter unknown cases.

Example:
    @distilled(fallback_ptool="extract_food_items")
    def extract_food_items(text: str) -> List[str]:
        '''Distilled from 47 traces. Falls back to LLM on failure.'''
        pattern = r'\\b(pizza|salad|burger|fries)\\b'
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches
        raise DistillationFallback("No known patterns matched")
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

T = TypeVar("T")


class DistillationFallback(Exception):
    """
    Raised when a distilled function needs to fall back to LLM.

    This is a signal exception, not an error. The @distilled decorator
    catches this and invokes the original @ptool.
    """

    def __init__(self, reason: str = "Unknown pattern"):
        self.reason = reason
        super().__init__(reason)


@dataclass
class FallbackEvent:
    """Record of a fallback event for analysis."""

    ptool_name: str
    inputs: Dict[str, Any]
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    python_error: Optional[str] = None
    llm_result: Optional[Any] = None


# Storage for fallback events (for analysis)
_FALLBACK_LOG: List[FallbackEvent] = []


def get_fallback_log() -> List[FallbackEvent]:
    """Get the log of all fallback events."""
    return list(_FALLBACK_LOG)


def clear_fallback_log() -> None:
    """Clear the fallback log."""
    global _FALLBACK_LOG
    _FALLBACK_LOG = []


def distilled(
    fallback_ptool: str,
    log_fallbacks: bool = True,
    max_fallback_ratio: Optional[float] = None,  # Alert if fallbacks exceed this ratio
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for distilled functions with LLM fallback.

    A distilled function is a pure Python implementation that was generated
    from analyzing LLM traces. If the Python implementation fails or raises
    DistillationFallback, the original @ptool is called via LLM.

    Args:
        fallback_ptool: Name of the ptool to call on fallback
        log_fallbacks: Whether to log fallback events for analysis
        max_fallback_ratio: Optional threshold to warn if fallbacks are too frequent

    Example:
        # Original ptool (still needed as fallback)
        @ptool()
        def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
            '''Classify sentiment of text.'''
            ...

        # Distilled version with fallback
        @distilled(fallback_ptool="classify_sentiment")
        def classify_sentiment_distilled(text: str) -> Literal["positive", "negative", "neutral"]:
            '''Distilled from 150 traces. 92% coverage.'''
            text_lower = text.lower()
            positive_words = {'love', 'great', 'amazing', 'excellent', 'happy'}
            negative_words = {'hate', 'terrible', 'awful', 'bad', 'sad'}

            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)

            if pos_count > neg_count:
                return "positive"
            elif neg_count > pos_count:
                return "negative"
            elif pos_count == 0 and neg_count == 0:
                raise DistillationFallback("No sentiment words found")

            return "neutral"
    """

    # Track statistics for this distilled function
    stats = {
        "total_calls": 0,
        "python_success": 0,
        "fallback_count": 0,
    }

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            nonlocal stats
            stats["total_calls"] += 1

            # Convert args to kwargs for logging
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = dict(bound.arguments)

            # Try the distilled Python implementation first
            try:
                result = func(*args, **kwargs)
                stats["python_success"] += 1
                return result

            except DistillationFallback as e:
                # Expected signal - fall back to LLM
                reason = e.reason
                python_error = None
                logger.debug(f"Distilled {func.__name__} falling back: {reason}")

            except Exception as e:
                # Unexpected error in Python code - also fall back
                reason = f"Python error: {type(e).__name__}"
                python_error = str(e)
                logger.warning(f"Distilled {func.__name__} error: {e}")

            # Fall back to the original ptool via LLM
            stats["fallback_count"] += 1

            # Get the original ptool from registry
            from .ptool import get_registry
            registry = get_registry()
            spec = registry.get(fallback_ptool)

            if spec is None:
                raise RuntimeError(
                    f"Fallback ptool '{fallback_ptool}' not found in registry. "
                    f"Make sure the @ptool decorated function is defined."
                )

            # Execute via LLM
            from .llm_backend import execute_ptool
            llm_result = execute_ptool(spec, inputs)

            # Log the fallback event
            if log_fallbacks:
                event = FallbackEvent(
                    ptool_name=fallback_ptool,
                    inputs=inputs,
                    reason=reason,
                    python_error=python_error,
                    llm_result=llm_result,
                )
                _FALLBACK_LOG.append(event)

                # Also log to trace store
                try:
                    from .trace_store import get_trace_store
                    trace_store = get_trace_store()
                    trace_store.emit_error(
                        trace_id="fallback",
                        error=f"Distilled fallback: {reason}",
                        ptool_name=fallback_ptool,
                    )
                except ImportError:
                    pass

            # Check fallback ratio
            if max_fallback_ratio is not None:
                ratio = stats["fallback_count"] / stats["total_calls"]
                if ratio > max_fallback_ratio:
                    logger.warning(
                        f"Distilled {func.__name__} fallback ratio {ratio:.1%} "
                        f"exceeds threshold {max_fallback_ratio:.1%}"
                    )

            return llm_result

        # Attach stats and metadata to the wrapper
        wrapper._distilled_stats = stats  # type: ignore
        wrapper._fallback_ptool = fallback_ptool  # type: ignore
        wrapper._is_distilled = True  # type: ignore

        return wrapper

    return decorator


def get_distilled_stats(func: Callable) -> Dict[str, Any]:
    """Get statistics for a distilled function."""
    if not hasattr(func, "_distilled_stats"):
        return {"error": "Not a distilled function"}

    stats = func._distilled_stats
    total = stats["total_calls"]
    return {
        "total_calls": total,
        "python_success": stats["python_success"],
        "fallback_count": stats["fallback_count"],
        "success_rate": stats["python_success"] / total if total > 0 else 0,
        "fallback_rate": stats["fallback_count"] / total if total > 0 else 0,
    }


def is_distilled(func: Callable) -> bool:
    """Check if a function is decorated with @distilled."""
    return getattr(func, "_is_distilled", False)


# ============================================================================
# Convenience: Combined @distilled_ptool decorator
# ============================================================================

def distilled_ptool(
    implementation: Callable[..., T],
    model: str = "deepseek-v3-0324",
    output_mode: str = "structured",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Create a ptool with a distilled implementation in one decorator.

    This combines @ptool (as fallback) and @distilled (as primary) in one step.

    Args:
        implementation: The distilled Python implementation
        model: LLM model for fallback
        output_mode: Output mode for fallback ("structured" or "freeform")

    Example:
        def _extract_numbers_impl(text: str) -> List[int]:
            '''Pure Python implementation.'''
            import re
            return [int(x) for x in re.findall(r'\\d+', text)]

        @distilled_ptool(_extract_numbers_impl)
        def extract_numbers(text: str) -> List[int]:
            '''Extract all numbers from text. Falls back to LLM if needed.'''
            ...  # This docstring is used as the LLM prompt
    """
    from .ptool import ptool

    def decorator(fallback_func: Callable[..., T]) -> Callable[..., T]:
        # First, create the ptool (this registers it)
        ptool_wrapper = ptool(model=model, output_mode=output_mode)(fallback_func)

        # Then, wrap the implementation with distilled
        distilled_wrapper = distilled(
            fallback_ptool=fallback_func.__name__,
            log_fallbacks=True,
        )(implementation)

        # Copy metadata
        distilled_wrapper.__doc__ = (
            f"{implementation.__doc__ or ''}\n\n"
            f"Falls back to LLM ({model}) on failure."
        )

        return distilled_wrapper

    return decorator


# ============================================================================
# Analysis utilities
# ============================================================================

def analyze_fallbacks(
    min_count: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze fallback events to find patterns for improvement.

    Returns a dict mapping ptool names to analysis results.
    """
    from collections import Counter

    # Group fallbacks by ptool
    by_ptool: Dict[str, List[FallbackEvent]] = {}
    for event in _FALLBACK_LOG:
        by_ptool.setdefault(event.ptool_name, []).append(event)

    results = {}
    for ptool_name, events in by_ptool.items():
        if len(events) < min_count:
            continue

        # Analyze reasons
        reason_counts = Counter(e.reason for e in events)

        # Find common input patterns
        # (This could be extended with clustering)

        results[ptool_name] = {
            "total_fallbacks": len(events),
            "top_reasons": reason_counts.most_common(5),
            "has_python_errors": any(e.python_error for e in events),
            "recommendation": _get_recommendation(reason_counts),
        }

    return results


def _get_recommendation(reason_counts: Dict[str, int]) -> str:
    """Generate a recommendation based on fallback reasons."""
    top_reason = reason_counts.most_common(1)[0][0] if reason_counts else ""

    if "pattern" in top_reason.lower():
        return "Consider adding more patterns to the distilled implementation"
    elif "error" in top_reason.lower():
        return "Fix bugs in the distilled implementation"
    elif "confidence" in top_reason.lower():
        return "Lower the confidence threshold or improve pattern matching"
    else:
        return "Analyze fallback inputs to identify missing cases"
