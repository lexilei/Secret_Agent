"""
Accuracy metrics for MedCalc-Bench evaluation.

Follows the evaluation protocol from the MedCalc-Bench paper:
- Rule-based calculators: Exact match required
- Equation-based calculators: ±5% tolerance for numeric outputs
- Date outputs: Exact match required
- Weeks/days outputs: Exact match required
- Range validation using lower/upper limits
"""

from dataclasses import dataclass
from typing import Any, Optional, Union
import re


@dataclass
class AccuracyResult:
    """Result of accuracy evaluation."""
    is_exact_match: bool
    is_within_tolerance: bool
    is_within_limits: bool
    absolute_error: float
    relative_error: Optional[float]  # None if ground_truth is 0
    output_type: str = "numeric"     # numeric, date, weeks_days


def normalize_date(date_str: str) -> str:
    """Normalize date string for comparison."""
    # Remove extra whitespace
    date_str = date_str.strip()
    # Parse MM/DD/YYYY format and normalize
    match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match:
        month, day, year = match.groups()
        return f"{int(month):02d}/{int(day):02d}/{year}"
    return date_str


def normalize_weeks_days(weeks_days_str: str) -> tuple:
    """
    Normalize weeks/days string for comparison.
    Returns tuple of (weeks, days) as integers.
    """
    weeks_days_str = str(weeks_days_str).strip()

    # Pattern for ("X weeks", "Y days") format
    match = re.search(r"(\d+)\s*weeks?.*?(\d+)\s*days?", weeks_days_str, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    # Pattern for just weeks
    match = re.search(r"(\d+)\s*weeks?", weeks_days_str, re.IGNORECASE)
    if match:
        return (int(match.group(1)), 0)

    # Pattern for just days
    match = re.search(r"(\d+)\s*days?", weeks_days_str, re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))

    return None


def calculate_accuracy(
    predicted: Optional[Any],
    ground_truth: Any,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    tolerance_pct: float = 5.0,
    output_type: str = "numeric",
    category: str = "equation-based",
) -> AccuracyResult:
    """
    Calculate accuracy metrics for a prediction.

    Args:
        predicted: Predicted value (None if extraction failed)
        ground_truth: True value from dataset
        lower_limit: Optional lower bound from dataset
        upper_limit: Optional upper bound from dataset
        tolerance_pct: Percentage tolerance (default 5% per MedCalc-Bench)
        output_type: Type of output (numeric, date, weeks_days)
        category: Calculator category ("equation-based" or "rule-based")
                  Rule-based requires exact match, equation-based allows tolerance

    Returns:
        AccuracyResult with all accuracy metrics
    """
    # Handle None predictions
    if predicted is None:
        return AccuracyResult(
            is_exact_match=False,
            is_within_tolerance=False,
            is_within_limits=False,
            absolute_error=float('inf'),
            relative_error=None,
            output_type=output_type,
        )

    # Handle date outputs
    output_type_lower = output_type.lower()
    if output_type_lower == "date":
        pred_normalized = normalize_date(str(predicted))
        gt_normalized = normalize_date(str(ground_truth))
        is_exact = pred_normalized == gt_normalized
        return AccuracyResult(
            is_exact_match=is_exact,
            is_within_tolerance=is_exact,  # No tolerance for dates
            is_within_limits=is_exact,
            absolute_error=0.0 if is_exact else 1.0,
            relative_error=None,
            output_type="date",
        )

    # Handle weeks/days outputs
    if "week" in output_type_lower or "day" in output_type_lower:
        pred_normalized = normalize_weeks_days(str(predicted))
        gt_normalized = normalize_weeks_days(str(ground_truth))
        if pred_normalized is None or gt_normalized is None:
            # Fall back to string comparison
            is_exact = str(predicted).strip() == str(ground_truth).strip()
        else:
            is_exact = pred_normalized == gt_normalized
        return AccuracyResult(
            is_exact_match=is_exact,
            is_within_tolerance=is_exact,  # No tolerance for weeks/days
            is_within_limits=is_exact,
            absolute_error=0.0 if is_exact else 1.0,
            relative_error=None,
            output_type="weeks_days",
        )

    # Numeric output handling
    try:
        predicted = float(predicted)
        ground_truth = float(ground_truth)
    except (ValueError, TypeError):
        # Can't compare as numbers
        return AccuracyResult(
            is_exact_match=False,
            is_within_tolerance=False,
            is_within_limits=False,
            absolute_error=float('inf'),
            relative_error=None,
            output_type="numeric",
        )

    # Calculate errors
    absolute_error = abs(predicted - ground_truth)

    if ground_truth != 0:
        relative_error = absolute_error / abs(ground_truth)
    else:
        relative_error = None

    # Exact match (allowing for floating point epsilon)
    is_exact = absolute_error < 0.01

    # Within percentage tolerance
    # Rule-based calculators require exact match per MedCalc-Bench protocol
    if category == "rule-based":
        is_within_tolerance = is_exact
    else:
        # Equation-based: apply ±5% tolerance
        if ground_truth != 0:
            tolerance_value = abs(ground_truth) * (tolerance_pct / 100)
        else:
            # For zero ground truth, use small absolute tolerance
            tolerance_value = 0.05
        is_within_tolerance = absolute_error <= tolerance_value

    # Within defined limits (if provided)
    is_within_limits = True
    if lower_limit is not None and upper_limit is not None:
        is_within_limits = lower_limit <= predicted <= upper_limit
    elif lower_limit is not None:
        is_within_limits = predicted >= lower_limit
    elif upper_limit is not None:
        is_within_limits = predicted <= upper_limit

    return AccuracyResult(
        is_exact_match=is_exact,
        is_within_tolerance=is_within_tolerance,
        is_within_limits=is_within_limits,
        absolute_error=absolute_error,
        relative_error=relative_error,
        output_type="numeric",
    )


def is_correct(
    predicted: Optional[float],
    ground_truth: float,
    tolerance_pct: float = 5.0,
    category: str = "equation-based",
) -> bool:
    """
    Simple correctness check (within tolerance).

    Args:
        predicted: Predicted value
        ground_truth: True value
        tolerance_pct: Percentage tolerance
        category: Calculator category ("equation-based" or "rule-based")

    Returns:
        True if prediction is correct within tolerance
    """
    result = calculate_accuracy(
        predicted, ground_truth, tolerance_pct=tolerance_pct, category=category
    )
    return result.is_within_tolerance
