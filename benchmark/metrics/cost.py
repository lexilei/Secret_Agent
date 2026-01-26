"""
Cost calculation for LLM API usage.

Loads pricing from LLMS.json and calculates costs per request.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class CostMetrics:
    """Token and cost metrics for an LLM call."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model: str


# Cache for LLMS.json config
_llms_config_cache: Optional[Dict[str, Any]] = None


def load_llms_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load LLM configuration from LLMS.json.

    Args:
        config_path: Optional path to LLMS.json. If None, searches common locations.

    Returns:
        Parsed LLMS.json config
    """
    global _llms_config_cache

    if _llms_config_cache is not None:
        return _llms_config_cache

    # Search paths
    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))

    # Add common locations
    search_paths.extend([
        Path(__file__).parent.parent.parent / "LLMS.json",  # Project root
        Path.cwd() / "LLMS.json",
        Path.home() / ".config" / "ptool_framework" / "LLMS.json",
    ])

    for path in search_paths:
        if path.exists():
            with open(path) as f:
                _llms_config_cache = json.load(f)
            return _llms_config_cache

    # Return minimal fallback config
    _llms_config_cache = {
        "models": {
            "deepseek-v3-0324": {
                "cost": {"input": 0.00125, "output": 0.00125}
            },
            "deepseek-r1": {
                "cost": {"input": 0.003, "output": 0.007}
            },
        }
    }
    return _llms_config_cache


def get_model_pricing(model: str, config: Optional[Dict] = None) -> Dict[str, float]:
    """
    Get pricing for a specific model.

    Args:
        model: Model name
        config: Optional LLMS config (loads default if None)

    Returns:
        Dict with 'input' and 'output' cost per 1K tokens
    """
    if config is None:
        config = load_llms_config()

    model_config = config.get("models", {}).get(model, {})
    cost_info = model_config.get("cost", {})

    return {
        "input": cost_info.get("input", 0.001),   # Default $0.001/1K
        "output": cost_info.get("output", 0.002),  # Default $0.002/1K
    }


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    config: Optional[Dict] = None,
) -> CostMetrics:
    """
    Calculate cost for an LLM call.

    Args:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        model: Model name (must be in LLMS.json)
        config: Optional LLMS config

    Returns:
        CostMetrics with token counts and USD cost
    """
    pricing = get_model_pricing(model, config)

    # Cost is per 1K tokens
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost

    return CostMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=total_cost,
        model=model,
    )


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count for text.

    Uses simple heuristic: ~4 chars per token for English.
    For accurate counts, use the actual tokenizer.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Rough heuristic: 4 characters per token
    return len(text) // 4 + 1
