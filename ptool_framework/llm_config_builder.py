"""
LLM Configuration Builder.

Uses an LLM to procedurally generate and update LLMS.json based on
natural language model descriptions like:
- "latest OpenAI models"
- "deepseek-v3"
- "Claude 4 family"
- "fast inference models from Groq"

Features:
- Schema validation for LLMS.json
- LLM-powered model lookup and configuration
- Preserves existing format and structure
- Supports tagging and enabling models

Example:
    >>> from ptool_framework.llm_config_builder import LLMConfigBuilder
    >>>
    >>> builder = LLMConfigBuilder()
    >>>
    >>> # Add models from natural language descriptions
    >>> builder.add_models([
    ...     "deepseek-v3",
    ...     "latest GPT-4 offerings",
    ...     "Claude Sonnet 4",
    ... ])
    >>>
    >>> # Save to LLMS.json
    >>> builder.save()
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .llm_backend import call_llm
from .model_selector import (
    _find_llms_json,
    load_llms_config,
    get_default_model,
    ModelConfig,
)

# Try to import web search capability
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# LM Arena leaderboard URL
LM_ARENA_URL = "https://huggingface.co/spaces/lmarena-ai/lmarena-leaderboard"


# =============================================================================
# Schema Definition
# =============================================================================

LLMS_JSON_SCHEMA = {
    "type": "object",
    "required": ["version", "default_model", "models"],
    "properties": {
        "version": {"type": "string"},
        "default_model": {"type": "string"},
        "description": {"type": "string"},
        "models": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["provider", "model_id"],
                "properties": {
                    "provider": {
                        "type": "string",
                        "enum": ["together", "anthropic", "openai", "groq", "google", "local"],
                    },
                    "model_id": {"type": "string"},
                    "api_key_env": {"type": ["string", "null"]},
                    "endpoint": {"type": ["string", "null"]},
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": {"type": "string"},
                    "cost": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "number"},
                            "output": {"type": "number"},
                            "currency": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                    },
                    "context_window": {"type": "integer"},
                    "max_output": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "provider_defaults": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "base_url": {"type": "string"},
                },
            },
        },
        "capability_descriptions": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "tag_descriptions": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
}

# Known providers and their API key env vars
PROVIDER_INFO = {
    "together": {
        "api_key_env": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com",
    },
    "local": {
        "api_key_env": None,
        "base_url": "http://localhost:8000/v1",
    },
}

# Standard capabilities
KNOWN_CAPABILITIES = [
    "reasoning",
    "coding",
    "math",
    "general",
    "fast",
    "long-context",
    "vision",
    "chain-of-thought",
    "extended-thinking",
    "agent",
    "function-calling",
    "json-mode",
]


# =============================================================================
# Validation
# =============================================================================

@dataclass
class ValidationError:
    """A validation error."""
    path: str
    message: str


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)

    def __bool__(self):
        return self.valid


def validate_llms_json(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate LLMS.json data against the schema.

    Args:
        data: Parsed LLMS.json content

    Returns:
        ValidationResult with any errors

    Example:
        >>> with open("LLMS.json") as f:
        ...     data = json.load(f)
        >>> result = validate_llms_json(data)
        >>> if not result:
        ...     for error in result.errors:
        ...         print(f"{error.path}: {error.message}")
    """
    errors = []

    # Check required fields
    for field in ["version", "default_model", "models"]:
        if field not in data:
            errors.append(ValidationError("", f"Missing required field: {field}"))

    if errors:
        return ValidationResult(valid=False, errors=errors)

    # Check version
    if not isinstance(data.get("version"), str):
        errors.append(ValidationError("version", "Must be a string"))

    # Check default_model
    default = data.get("default_model")
    if not isinstance(default, str):
        errors.append(ValidationError("default_model", "Must be a string"))
    elif default not in data.get("models", {}):
        errors.append(ValidationError(
            "default_model",
            f"Model '{default}' not found in models"
        ))

    # Check models
    models = data.get("models", {})
    if not isinstance(models, dict):
        errors.append(ValidationError("models", "Must be an object"))
    else:
        for name, model in models.items():
            path = f"models.{name}"

            if not isinstance(model, dict):
                errors.append(ValidationError(path, "Must be an object"))
                continue

            # Required model fields
            if "provider" not in model:
                errors.append(ValidationError(path, "Missing required field: provider"))
            elif model["provider"] not in PROVIDER_INFO:
                errors.append(ValidationError(
                    f"{path}.provider",
                    f"Unknown provider: {model['provider']}"
                ))

            if "model_id" not in model:
                errors.append(ValidationError(path, "Missing required field: model_id"))

            # Check capabilities
            caps = model.get("capabilities", [])
            if not isinstance(caps, list):
                errors.append(ValidationError(f"{path}.capabilities", "Must be an array"))

            # Check cost structure
            cost = model.get("cost")
            if cost is not None and not isinstance(cost, dict):
                errors.append(ValidationError(f"{path}.cost", "Must be an object"))
            elif cost:
                if "input" in cost and not isinstance(cost["input"], (int, float)):
                    errors.append(ValidationError(f"{path}.cost.input", "Must be a number"))
                if "output" in cost and not isinstance(cost["output"], (int, float)):
                    errors.append(ValidationError(f"{path}.cost.output", "Must be a number"))

            # Check enabled
            enabled = model.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                errors.append(ValidationError(f"{path}.enabled", "Must be a boolean"))

            # Check tags
            tags = model.get("tags")
            if tags is not None and not isinstance(tags, list):
                errors.append(ValidationError(f"{path}.tags", "Must be an array"))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_llms_file(path: Optional[str] = None) -> ValidationResult:
    """
    Validate an LLMS.json file.

    Args:
        path: Path to LLMS.json, or None to auto-find

    Returns:
        ValidationResult
    """
    if path:
        file_path = Path(path)
    else:
        file_path = _find_llms_json()

    if not file_path or not file_path.exists():
        return ValidationResult(
            valid=False,
            errors=[ValidationError("", "LLMS.json file not found")]
        )

    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            errors=[ValidationError("", f"Invalid JSON: {e}")]
        )

    return validate_llms_json(data)


# =============================================================================
# LLM Config Builder
# =============================================================================

def fetch_lm_arena_elo(model_name: str) -> Optional[int]:
    """
    Fetch ELO score from LM Arena leaderboard.

    Args:
        model_name: Model name to look up

    Returns:
        ELO score or None if not found
    """
    if not HAS_REQUESTS:
        return None

    try:
        # Search for LM Arena data for this model
        search_url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.post(
            search_url,
            data={"q": f"lmarena chatbot arena {model_name} elo score"},
            headers=headers,
            timeout=10,
        )

        if response.status_code != 200:
            return None

        # Look for ELO scores in results (typically 4-digit numbers around 1000-1400)
        import re
        # Find patterns like "ELO: 1234" or "1234 ELO" or just 4-digit numbers near "elo"
        text = response.text.lower()

        # Find ELO mentions
        elo_patterns = [
            rf'{model_name.lower()}[^0-9]*(\d{{4}})',  # model name followed by 4 digits
            rf'(\d{{4}})[^0-9]*{model_name.lower()}',  # 4 digits followed by model name
            r'elo[:\s]*(\d{4})',  # elo: NNNN
            r'(\d{4})\s*elo',  # NNNN elo
        ]

        for pattern in elo_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                elo = int(match)
                if 900 <= elo <= 1500:  # Reasonable ELO range
                    return elo

        return None

    except Exception:
        return None


def fetch_lm_arena_leaderboard() -> Dict[str, int]:
    """
    Fetch the full LM Arena leaderboard.

    Returns:
        Dict mapping model names to ELO scores
    """
    if not HAS_REQUESTS:
        return {}

    try:
        # Try to get leaderboard data via web search
        search_url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.post(
            search_url,
            data={"q": "lmarena chatbot arena leaderboard elo scores 2024"},
            headers=headers,
            timeout=10,
        )

        if response.status_code != 200:
            return {}

        # This is a simplified approach - in production you'd want to
        # scrape the actual leaderboard or use an API
        return {}

    except Exception:
        return {}


@dataclass
class GeneratedModelConfig:
    """A model configuration generated by the LLM."""
    name: str
    provider: str
    model_id: str
    description: str
    capabilities: List[str]
    cost_input: float
    cost_output: float
    context_window: int
    max_output: int
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    quality_score: Optional[float] = None  # Manual quality score
    elo_score: Optional[int] = None  # From LM Arena

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LLMS.json model format."""
        result = {
            "provider": self.provider,
            "model_id": self.model_id,
            "api_key_env": PROVIDER_INFO.get(self.provider, {}).get("api_key_env"),
            "endpoint": None,
            "capabilities": self.capabilities,
            "description": self.description,
            "cost": {
                "input": self.cost_input,
                "output": self.cost_output,
                "currency": "USD",
                "unit": "1K tokens",
            },
            "context_window": self.context_window,
            "max_output": self.max_output,
            "enabled": self.enabled,
            "tags": self.tags,
        }
        if self.quality_score is not None:
            result["quality_score"] = self.quality_score
        if self.elo_score is not None:
            result["elo_score"] = self.elo_score
        return result


class LLMConfigBuilder:
    """
    Builder for LLMS.json using LLM-powered model lookup.

    Uses an enabled LLM from the current LLMS.json to look up
    information about new models and generate configurations.

    Example:
        >>> builder = LLMConfigBuilder()
        >>>
        >>> # Add models using natural language
        >>> builder.add_models([
        ...     "gpt-4o",
        ...     "gpt-4o-mini",
        ...     "latest Claude models",
        ... ])
        >>>
        >>> # Preview changes
        >>> print(builder.preview())
        >>>
        >>> # Save
        >>> builder.save()
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model: Optional[str] = None,
        default_tags: Optional[List[str]] = None,
        default_enabled: bool = False,
        fetch_elo: bool = True,
    ):
        """
        Initialize the builder.

        Args:
            config_path: Path to LLMS.json
            model: LLM to use for generation (uses default if None)
            default_tags: Default tags for new models
            default_enabled: Default enabled state for new models
            fetch_elo: Whether to fetch ELO scores from LM Arena
        """
        self.config_path = config_path
        self.model = model or get_default_model()
        self.default_tags = default_tags or []
        self.default_enabled = default_enabled
        self.fetch_elo = fetch_elo

        # Load existing config
        self._load_config()

        # Track new models to add
        self._pending_models: Dict[str, GeneratedModelConfig] = {}

    def _load_config(self) -> None:
        """Load existing LLMS.json."""
        if self.config_path:
            path = Path(self.config_path)
        else:
            path = _find_llms_json()

        if path and path.exists():
            with open(path) as f:
                self._config = json.load(f)
            self._config_path = path
        else:
            # Create default config
            self._config = {
                "version": "3.0",
                "default_model": "deepseek-v3-0324",
                "description": "LLM configuration for ptool framework.",
                "models": {},
                "provider_defaults": PROVIDER_INFO.copy(),
                "capability_descriptions": {
                    cap: f"Model has {cap} capability"
                    for cap in KNOWN_CAPABILITIES
                },
                "tag_descriptions": {
                    "testing": "Enabled for testing/development use",
                    "development": "Enabled for local development",
                    "production": "Approved for production use",
                },
            }
            self._config_path = Path.cwd() / "LLMS.json"

    def add_models(
        self,
        descriptions: List[str],
        tags: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
        quality_score: Optional[float] = None,
        echo: bool = True,
    ) -> List[str]:
        """
        Add models from natural language descriptions.

        Args:
            descriptions: List of model descriptions like:
                - "gpt-4o"
                - "latest OpenAI models"
                - "Claude 4 Sonnet"
                - "fast models from Groq"
            tags: Tags to apply to new models
            enabled: Whether to enable new models
            quality_score: Manual quality score to set (0.0-1.0)
            echo: Print progress

        Returns:
            List of model names that were added

        Example:
            >>> builder.add_models(["gpt-4o", "gpt-4o-mini"])
            ['gpt-4o', 'gpt-4o-mini']
            >>> # With manual quality score
            >>> builder.add_models(["gpt-4o"], quality_score=0.95)
        """
        added = []
        tags = tags or self.default_tags
        enabled = enabled if enabled is not None else self.default_enabled

        for desc in descriptions:
            if echo:
                print(f"Looking up: {desc}...")

            try:
                models = self._lookup_models(desc)
                for model in models:
                    model.tags = list(set(model.tags + tags))
                    model.enabled = enabled

                    # Set manual quality score if provided
                    if quality_score is not None:
                        model.quality_score = quality_score

                    # Fetch ELO score from LM Arena if enabled
                    if self.fetch_elo:
                        elo = fetch_lm_arena_elo(model.name)
                        if elo:
                            model.elo_score = elo
                            if echo:
                                print(f"    ELO: {elo}")

                    self._pending_models[model.name] = model
                    added.append(model.name)
                    if echo:
                        print(f"  + {model.name} ({model.provider})")
            except Exception as e:
                if echo:
                    print(f"  Error: {e}")

        return added

    def _search_web(self, query: str) -> str:
        """
        Search the web for model information.

        Uses DuckDuckGo HTML search (no API key needed).
        """
        if not HAS_REQUESTS:
            return ""

        try:
            # Use DuckDuckGo HTML search
            url = "https://html.duckduckgo.com/html/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.post(
                url,
                data={"q": query},
                headers=headers,
                timeout=10,
            )

            if response.status_code != 200:
                return ""

            # Extract text snippets from results
            html = response.text
            snippets = []

            # Simple extraction of result snippets
            import re
            # Find result snippets
            matches = re.findall(r'class="result__snippet"[^>]*>([^<]+)<', html)
            snippets.extend(matches[:5])

            # Find result titles
            titles = re.findall(r'class="result__a"[^>]*>([^<]+)<', html)
            snippets.extend(titles[:5])

            return "\n".join(snippets)

        except Exception:
            return ""

    def _lookup_models(self, description: str) -> List[GeneratedModelConfig]:
        """Use web search + LLM to look up current model information."""

        # First, search the web for up-to-date information
        search_queries = [
            f"{description} API model ID pricing 2024 2025",
            f"{description} LLM context window capabilities",
        ]

        web_context = ""
        for query in search_queries:
            result = self._search_web(query)
            if result:
                web_context += f"\n{result}\n"

        # Build prompt with web context
        web_section = ""
        if web_context:
            web_section = f"""
RECENT WEB SEARCH RESULTS (use this for accurate, up-to-date information):
{web_context[:2000]}
"""

        prompt = f"""You are an expert on LLM APIs. Generate configuration for the requested model(s).

MODEL REQUEST: {description}
{web_section}
Generate a JSON array with model configuration(s). Each object must have:
- name: Short identifier for the model
- provider: One of: together, anthropic, openai, groq, google, local
- model_id: The EXACT API model ID as used in API calls (get this from the web search results or official docs)
- description: One-line description
- capabilities: Array from {KNOWN_CAPABILITIES}
- cost_input: USD per 1K input tokens (from web search or estimate)
- cost_output: USD per 1K output tokens (from web search or estimate)
- context_window: Max context tokens
- max_output: Max output tokens

Use the web search results above to get accurate, current information about model IDs and pricing.
If the web search has pricing info, use it. Otherwise, estimate based on similar models.

Respond with ONLY the JSON array (no markdown, no code blocks, no explanation):
[
  {{
    "name": "model-name",
    "provider": "provider",
    "model_id": "exact-api-model-id",
    "description": "Description",
    "capabilities": ["reasoning", "coding"],
    "cost_input": 0.001,
    "cost_output": 0.002,
    "context_window": 128000,
    "max_output": 4096
  }}
]"""

        response = call_llm(prompt, self.model)

        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not parse LLM response: {response[:200]}")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}")

        models = []
        for item in data:
            models.append(GeneratedModelConfig(
                name=item["name"],
                provider=item["provider"],
                model_id=item["model_id"],
                description=item.get("description", ""),
                capabilities=item.get("capabilities", []),
                cost_input=float(item.get("cost_input", 0)),
                cost_output=float(item.get("cost_output", 0)),
                context_window=int(item.get("context_window", 8192)),
                max_output=int(item.get("max_output", 4096)),
            ))

        return models

    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the config.

        Args:
            name: Model name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self._pending_models:
            del self._pending_models[name]
            return True
        if name in self._config.get("models", {}):
            del self._config["models"][name]
            return True
        return False

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Set enabled state for a model."""
        if name in self._pending_models:
            self._pending_models[name].enabled = enabled
            return True
        if name in self._config.get("models", {}):
            self._config["models"][name]["enabled"] = enabled
            return True
        return False

    def set_quality_score(self, name: str, score: float) -> bool:
        """
        Set manual quality score for a model.

        Args:
            name: Model name
            score: Quality score (0.0-1.0)

        Returns:
            True if successful
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")

        if name in self._pending_models:
            self._pending_models[name].quality_score = score
            return True
        if name in self._config.get("models", {}):
            self._config["models"][name]["quality_score"] = score
            return True
        return False

    def set_elo_score(self, name: str, elo: int) -> bool:
        """
        Set ELO score for a model.

        Args:
            name: Model name
            elo: ELO score (typically 900-1500)

        Returns:
            True if successful
        """
        if name in self._pending_models:
            self._pending_models[name].elo_score = elo
            return True
        if name in self._config.get("models", {}):
            self._config["models"][name]["elo_score"] = elo
            return True
        return False

    def add_tag(self, name: str, tag: str) -> bool:
        """Add a tag to a model."""
        if name in self._pending_models:
            if tag not in self._pending_models[name].tags:
                self._pending_models[name].tags.append(tag)
            return True
        if name in self._config.get("models", {}):
            tags = self._config["models"][name].get("tags", [])
            if tag not in tags:
                tags.append(tag)
                self._config["models"][name]["tags"] = tags
            return True
        return False

    def set_default_model(self, name: str) -> bool:
        """Set the default model."""
        all_models = set(self._config.get("models", {}).keys())
        all_models.update(self._pending_models.keys())

        if name in all_models:
            self._config["default_model"] = name
            return True
        return False

    def preview(self) -> str:
        """
        Preview the final LLMS.json content.

        Returns:
            Formatted JSON string
        """
        final = self._build_final_config()
        return json.dumps(final, indent=2)

    def _build_final_config(self) -> Dict[str, Any]:
        """Build the final config with pending changes."""
        final = self._config.copy()
        final["models"] = final.get("models", {}).copy()

        # Add pending models
        for name, model in self._pending_models.items():
            final["models"][name] = model.to_dict()

        # Update provider defaults
        for provider, info in PROVIDER_INFO.items():
            if provider not in final.get("provider_defaults", {}):
                if "provider_defaults" not in final:
                    final["provider_defaults"] = {}
                final["provider_defaults"][provider] = {"base_url": info["base_url"]}

        return final

    def validate(self) -> ValidationResult:
        """
        Validate the final configuration.

        Returns:
            ValidationResult
        """
        final = self._build_final_config()
        return validate_llms_json(final)

    def save(
        self,
        path: Optional[str] = None,
        backup: bool = True,
    ) -> Path:
        """
        Save the configuration to LLMS.json.

        Args:
            path: Output path (uses original path if None)
            backup: Create backup of existing file

        Returns:
            Path to saved file

        Raises:
            ValueError: If validation fails
        """
        # Validate first
        result = self.validate()
        if not result:
            errors = "\n".join(f"  {e.path}: {e.message}" for e in result.errors)
            raise ValueError(f"Validation failed:\n{errors}")

        # Determine output path
        out_path = Path(path) if path else self._config_path

        # Backup existing
        if backup and out_path.exists():
            backup_path = out_path.with_suffix(
                f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(out_path) as f:
                backup_data = f.read()
            with open(backup_path, "w") as f:
                f.write(backup_data)

        # Save
        final = self._build_final_config()
        with open(out_path, "w") as f:
            json.dump(final, f, indent=2)

        # Clear pending
        self._pending_models.clear()

        return out_path

    def list_models(self) -> List[str]:
        """List all models (existing + pending)."""
        models = list(self._config.get("models", {}).keys())
        models.extend(self._pending_models.keys())
        return sorted(set(models))

    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model configuration by name."""
        if name in self._pending_models:
            return self._pending_models[name].to_dict()
        return self._config.get("models", {}).get(name)


# =============================================================================
# Convenience Functions
# =============================================================================

def build_llms_config(
    model_descriptions: List[str],
    output_path: Optional[str] = None,
    tags: Optional[List[str]] = None,
    enabled: bool = False,
    echo: bool = True,
) -> Path:
    """
    Build a new LLMS.json from model descriptions.

    Args:
        model_descriptions: List of model names or descriptions
        output_path: Output path for LLMS.json
        tags: Tags to apply to all models
        enabled: Whether to enable all models
        echo: Print progress

    Returns:
        Path to generated file

    Example:
        >>> build_llms_config([
        ...     "gpt-4o",
        ...     "gpt-4o-mini",
        ...     "claude-sonnet-4",
        ... ], tags=["production"])
    """
    builder = LLMConfigBuilder(default_tags=tags or [], default_enabled=enabled)
    builder.add_models(model_descriptions, echo=echo)
    return builder.save(output_path)


def update_llms_config(
    model_descriptions: List[str],
    tags: Optional[List[str]] = None,
    enabled: bool = False,
    echo: bool = True,
) -> Path:
    """
    Update existing LLMS.json with new models.

    Args:
        model_descriptions: List of model names or descriptions to add
        tags: Tags to apply to new models
        enabled: Whether to enable new models
        echo: Print progress

    Returns:
        Path to updated file

    Example:
        >>> update_llms_config(["gpt-4o-mini"], tags=["testing"], enabled=True)
    """
    builder = LLMConfigBuilder(default_tags=tags or [], default_enabled=enabled)
    builder.add_models(model_descriptions, echo=echo)
    return builder.save()


def enable_models_by_tag(tag: str, config_path: Optional[str] = None) -> int:
    """
    Enable all models with a specific tag.

    Args:
        tag: Tag to match
        config_path: Path to LLMS.json

    Returns:
        Number of models enabled
    """
    if config_path:
        path = Path(config_path)
    else:
        path = _find_llms_json()

    if not path or not path.exists():
        return 0

    with open(path) as f:
        data = json.load(f)

    count = 0
    for name, model in data.get("models", {}).items():
        if tag in model.get("tags", []):
            model["enabled"] = True
            count += 1

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return count


def disable_all_models(config_path: Optional[str] = None) -> int:
    """
    Disable all models in LLMS.json.

    Args:
        config_path: Path to LLMS.json

    Returns:
        Number of models disabled
    """
    if config_path:
        path = Path(config_path)
    else:
        path = _find_llms_json()

    if not path or not path.exists():
        return 0

    with open(path) as f:
        data = json.load(f)

    count = 0
    for model in data.get("models", {}).values():
        model["enabled"] = False
        count += 1

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return count


def enable_only_tag(tag: str, config_path: Optional[str] = None) -> int:
    """
    Enable only models with a specific tag, disable all others.

    Args:
        tag: Tag to match
        config_path: Path to LLMS.json

    Returns:
        Number of models enabled

    Example:
        >>> # Enable only testing models
        >>> enable_only_tag("testing")
    """
    disable_all_models(config_path)
    return enable_models_by_tag(tag, config_path)
