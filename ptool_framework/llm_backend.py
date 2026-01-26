"""
LLM Backend: Handles actual LLM calls and response parsing for ptools.

This module:
1. Loads LLM configuration from LLMS.json
2. Routes requests to appropriate providers
3. Handles response parsing (structured JSON or freeform)
4. Supports local and cloud-hosted models
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeVar, Union, get_args, get_origin

import logging

# Try to use loguru, fall back to standard logging
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

# Import LLM providers - these are optional
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import together
    HAS_TOGETHER = True
except ImportError:
    HAS_TOGETHER = False

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

from .ptool import PToolSpec

T = TypeVar("T")

# Flag to enable/disable trace collection globally
TRACE_ENABLED = True


# ============================================================================
# Exceptions
# ============================================================================

class LLMError(Exception):
    """Error during LLM execution."""
    pass


class ParseError(Exception):
    """Error parsing LLM response."""
    pass


class ConfigError(Exception):
    """Error in LLM configuration."""
    pass


# ============================================================================
# LLM Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    name: str
    provider: str
    model_id: str
    api_key_env: Optional[str] = None
    endpoint: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    description: str = ""
    cost: Union[Dict[str, Any], str] = field(default_factory=dict)
    context_window: int = 8192
    max_output: int = 4096

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment variable."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    @property
    def is_local(self) -> bool:
        """Check if this is a local model."""
        return self.provider == "local" or self.cost == "local"


@dataclass
class LLMConfig:
    """Configuration for all available LLMs."""
    default_model: str
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    provider_defaults: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> LLMConfig:
        """
        Load LLM configuration from LLMS.json.

        Args:
            config_path: Path to LLMS.json. If None, searches in:
                1. Current directory
                2. Parent directory
                3. Package directory
        """
        if config_path is None:
            # Search for LLMS.json
            search_paths = [
                Path.cwd() / "LLMS.json",
                Path.cwd().parent / "LLMS.json",
                Path(__file__).parent.parent / "LLMS.json",
            ]
            for path in search_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path is None or not Path(config_path).exists():
            logger.warning("LLMS.json not found, using default configuration")
            return cls._default_config()

        logger.info(f"Loading LLM config from {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        # Parse models
        models = {}
        for name, model_data in data.get("models", {}).items():
            models[name] = ModelConfig(
                name=name,
                provider=model_data.get("provider", "unknown"),
                model_id=model_data.get("model_id", name),
                api_key_env=model_data.get("api_key_env"),
                endpoint=model_data.get("endpoint"),
                capabilities=model_data.get("capabilities", []),
                description=model_data.get("description", ""),
                cost=model_data.get("cost", {}),
                context_window=model_data.get("context_window", 8192),
                max_output=model_data.get("max_output", 4096),
            )

        return cls(
            default_model=data.get("default_model", "deepseek-v3-0324"),
            models=models,
            provider_defaults=data.get("provider_defaults", {}),
        )

    @classmethod
    def _default_config(cls) -> LLMConfig:
        """Return default configuration when LLMS.json is not found."""
        return cls(
            default_model="deepseek-v3-0324",
            models={
                "deepseek-v3-0324": ModelConfig(
                    name="deepseek-v3-0324",
                    provider="together",
                    model_id="deepseek-ai/DeepSeek-V3",
                    api_key_env="TOGETHER_API_KEY",
                    capabilities=["reasoning", "coding"],
                    description="DeepSeek V3 via Together.ai",
                ),
            },
        )

    def get_model(self, name: Optional[str] = None) -> ModelConfig:
        """Get model configuration by name, or default if not specified."""
        if name is None:
            name = self.default_model

        # Check if name is in our models
        if name in self.models:
            return self.models[name]

        # Try to find by model_id
        for model in self.models.values():
            if model.model_id == name:
                return model

        raise ConfigError(f"Unknown model: {name}. Available: {list(self.models.keys())}")


# Global config instance
_CONFIG: Optional[LLMConfig] = None


def get_config() -> LLMConfig:
    """Get the global LLM configuration."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = LLMConfig.load()
    return _CONFIG


def reload_config(config_path: Optional[str] = None) -> LLMConfig:
    """Reload the LLM configuration."""
    global _CONFIG
    _CONFIG = LLMConfig.load(config_path)
    return _CONFIG


# ============================================================================
# Provider-Specific Implementations
# ============================================================================

def _call_together(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call Together.ai API."""
    if not HAS_TOGETHER:
        # Fall back to OpenAI client with Together endpoint
        if not HAS_OPENAI:
            raise LLMError("Neither 'together' nor 'openai' package installed")

        api_key = model_config.get_api_key()
        if not api_key:
            raise LLMError(f"API key not found in {model_config.api_key_env}")

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )

        try:
            completion = client.chat.completions.create(
                model=model_config.model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise LLMError(f"Together API error: {e}")

    # Use native together client
    api_key = model_config.get_api_key()
    if not api_key:
        raise LLMError(f"API key not found in {model_config.api_key_env}")

    client = together.Together(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise LLMError(f"Together API error: {e}")


def _call_anthropic(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call Anthropic Claude API."""
    if not HAS_ANTHROPIC:
        raise LLMError("anthropic package not installed")

    api_key = model_config.get_api_key()
    if not api_key:
        raise LLMError(f"API key not found in {model_config.api_key_env}")

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        raise LLMError(f"Anthropic API error: {e}")


def _call_openai(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call OpenAI API."""
    if not HAS_OPENAI:
        raise LLMError("openai package not installed")

    api_key = model_config.get_api_key()
    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    try:
        completion = client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise LLMError(f"OpenAI API error: {e}")


def _call_groq(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call Groq API."""
    if not HAS_GROQ:
        raise LLMError("groq package not installed")

    api_key = model_config.get_api_key()
    if not api_key:
        raise LLMError(f"API key not found in {model_config.api_key_env}")

    client = groq.Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise LLMError(f"Groq API error: {e}")


def _call_google(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call Google Gemini API."""
    if not HAS_GOOGLE:
        raise LLMError("google-generativeai package not installed")

    api_key = model_config.get_api_key()
    if not api_key:
        raise LLMError(f"API key not found in {model_config.api_key_env}")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_config.model_id)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise LLMError(f"Google API error: {e}")


def _call_local(prompt: str, model_config: ModelConfig, max_tokens: int = 4096) -> str:
    """Call local LLM endpoint (Ollama, vLLM, etc.)."""
    if not HAS_OPENAI:
        raise LLMError("openai package not installed (needed for local endpoint)")

    endpoint = model_config.endpoint
    if not endpoint:
        raise LLMError("No endpoint specified for local model")

    # Use OpenAI client with custom endpoint
    client = openai.OpenAI(
        api_key="not-needed",  # Local endpoints often don't need auth
        base_url=endpoint,
    )

    try:
        completion = client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise LLMError(f"Local LLM error ({endpoint}): {e}")


# ============================================================================
# Main LLM Call Function
# ============================================================================

def call_llm(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """
    Call an LLM with the given prompt.

    Routes to the appropriate provider based on LLMS.json configuration.

    Args:
        prompt: The prompt to send to the LLM
        model: Model name (from LLMS.json) or None for default
        max_tokens: Maximum tokens in response

    Returns:
        The LLM's response text
    """
    config = get_config()
    model_config = config.get_model(model)

    logger.info(f"Calling LLM: {model_config.name} ({model_config.provider})")
    logger.debug(f"Model ID: {model_config.model_id}, prompt_len={len(prompt)}")

    # Route to provider
    provider = model_config.provider.lower()

    if provider == "together":
        return _call_together(prompt, model_config, max_tokens)
    elif provider == "anthropic":
        return _call_anthropic(prompt, model_config, max_tokens)
    elif provider == "openai":
        return _call_openai(prompt, model_config, max_tokens)
    elif provider == "groq":
        return _call_groq(prompt, model_config, max_tokens)
    elif provider == "google":
        return _call_google(prompt, model_config, max_tokens)
    elif provider == "local":
        return _call_local(prompt, model_config, max_tokens)
    else:
        raise LLMError(f"Unknown provider: {provider}")


# ============================================================================
# Response Parsing
# ============================================================================

def parse_structured_response(response: str, return_type: Type[T]) -> T:
    """
    Parse a structured (JSON) response from an LLM.

    Expects format: {"result": <value>}
    """
    # Try to extract JSON from the response
    # Handle cases where LLM adds extra text
    json_match = re.search(r'\{[^{}]*"result"[^{}]*\}', response, re.DOTALL)

    if json_match:
        json_str = json_match.group()
    else:
        # Try to find any JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            raise ParseError(f"No JSON found in response: {response[:200]}...")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}\nResponse: {json_str[:200]}...")

    # Extract result based on return type
    origin = get_origin(return_type)

    # If return type is Dict, use the entire parsed object (not just "result" key)
    if origin is dict or return_type is dict:
        result = data
    elif "result" in data:
        result = data["result"]
    else:
        # Use the entire parsed object
        result = data

    # Validate against return type
    result = _coerce_type(result, return_type)
    return result


def parse_freeform_response(response: str, return_type: Type[T]) -> T:
    """
    Parse a freeform response from an LLM.

    Expects format with "ANSWER: <value>" on the last line.
    """
    # Look for ANSWER: pattern
    answer_match = re.search(r'ANSWER:\s*(.+?)$', response, re.MULTILINE | re.IGNORECASE)

    if answer_match:
        answer_str = answer_match.group(1).strip()
    else:
        # Use the last non-empty line
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            answer_str = lines[-1]
        else:
            raise ParseError(f"Could not extract answer from response: {response[:200]}...")

    # Try to parse as JSON first, then as literal
    try:
        result = json.loads(answer_str)
    except json.JSONDecodeError:
        result = answer_str

    # Coerce to expected type
    result = _coerce_type(result, return_type)
    return result


def _coerce_type(value: Any, target_type: Type[T]) -> T:
    """
    Attempt to coerce a value to the target type.

    Handles common cases like Literal types, Lists, etc.
    """
    origin = get_origin(target_type)
    args = get_args(target_type)

    # Handle Literal types
    if origin is Literal:
        if value in args:
            return value
        # Try case-insensitive match for strings
        if isinstance(value, str):
            for arg in args:
                if isinstance(arg, str) and value.lower() == arg.lower():
                    return arg
        raise ParseError(f"Value {value!r} not in Literal{args}")

    # Handle List types
    if origin is list:
        if not isinstance(value, list):
            raise ParseError(f"Expected list, got {type(value).__name__}")
        if args:
            item_type = args[0]
            return [_coerce_type(item, item_type) for item in value]
        return value

    # Handle Dict types
    if origin is dict:
        if not isinstance(value, dict):
            raise ParseError(f"Expected dict, got {type(value).__name__}")
        return value

    # Handle Tuple types
    if origin is tuple:
        if isinstance(value, (list, tuple)):
            if args:
                return tuple(_coerce_type(v, t) for v, t in zip(value, args))
            return tuple(value)
        raise ParseError(f"Expected tuple, got {type(value).__name__}")

    # Handle basic types
    if target_type is str:
        return str(value)
    if target_type is int:
        if isinstance(value, int):
            return value
        # Extract integer from string like "42 points" or "Score: 3"
        match = re.search(r'[-+]?\d+', str(value))
        if match:
            return int(match.group())
        raise ParseError(f"Could not extract int from: {value}")
    if target_type is float:
        if isinstance(value, (int, float)):
            return float(value)
        # Extract number from string like "23.5 kg/m²" or "BMI: 26.12"
        match = re.search(r'[-+]?\d*\.?\d+', str(value))
        if match:
            return float(match.group())
        raise ParseError(f"Could not extract float from: {value}")
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1')
        return bool(value)

    # Default: return as-is
    return value


# ============================================================================
# ptool Execution
# ============================================================================

def execute_ptool(
    spec: PToolSpec,
    inputs: Dict[str, Any],
    custom_backend: Optional[Callable] = None,
    model_override: Optional[str] = None,
    collect_traces: bool = True,
    additional_context: Optional[str] = None,
) -> Any:
    """
    Execute a ptool with the given inputs.

    Args:
        spec: The ptool specification
        inputs: Dictionary of input arguments
        custom_backend: Optional custom LLM backend function
        model_override: Override the model specified in the ptool
        collect_traces: Whether to log this execution to the trace store
        additional_context: Optional additional context to append to the prompt
            (used for repair, ICL examples, etc.)

    Returns:
        The parsed result from the LLM
    """
    import time
    import uuid

    # Generate trace ID for this execution
    trace_id = str(uuid.uuid4())[:8]

    # Format the prompt
    prompt = spec.format_prompt(**inputs)

    # Add additional context if provided (for repair, ICL, etc.)
    if additional_context:
        prompt = f"{prompt}\n\n{additional_context}"

    model = model_override or spec.model

    logger.debug(f"Executing ptool {spec.name} with model {model}")

    # Get trace store if tracing is enabled
    trace_store = None
    if TRACE_ENABLED and collect_traces:
        try:
            from .trace_store import get_trace_store, ExecutionTrace
            trace_store = get_trace_store()

            # Emit start event
            trace_store.emit_ptool_start(spec.name, inputs, trace_id)
        except ImportError:
            pass  # trace_store not available

    # Track timing
    start_time = time.time()
    response = None
    result = None
    error_msg = None
    success = False

    try:
        # Emit LLM request event
        if trace_store:
            trace_store.emit_llm_request(trace_id, model, prompt)

        # Call LLM
        llm_start = time.time()
        if custom_backend:
            response = custom_backend(prompt, model)
        else:
            response = call_llm(prompt, model)
        llm_duration = (time.time() - llm_start) * 1000  # ms

        # Emit LLM response event
        if trace_store:
            trace_store.emit_llm_response(trace_id, response, llm_duration)

        logger.debug(f"Response length: {len(response)}")

        # Parse response based on output mode
        if spec.output_mode == "structured":
            result = parse_structured_response(response, spec.return_type)
        else:
            result = parse_freeform_response(response, spec.return_type)

        success = True
        logger.debug(f"ptool {spec.name} returned: {result!r}")

    except Exception as e:
        error_msg = str(e)
        if trace_store:
            trace_store.emit_error(trace_id, error_msg, spec.name)
        raise

    finally:
        # Log execution trace
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        if trace_store and TRACE_ENABLED and collect_traces:
            from .trace_store import ExecutionTrace
            trace = ExecutionTrace(
                ptool_name=spec.name,
                inputs=inputs,
                output=result,
                success=success,
                execution_time_ms=execution_time_ms,
                model_used=model,
                trace_id=trace_id,
                error=error_msg,
                prompt=prompt,
                raw_response=response,
            )
            trace_store.log_execution(trace)

    return result


def enable_tracing(enabled: bool = True) -> None:
    """Enable or disable global trace collection."""
    global TRACE_ENABLED
    TRACE_ENABLED = enabled
    logger.info(f"Trace collection {'enabled' if enabled else 'disabled'}")


# ============================================================================
# Mock Backend for Testing
# ============================================================================

class MockLLMBackend:
    """
    Mock LLM backend for testing.

    Stores predefined responses for specific inputs.
    """

    def __init__(self):
        self.responses: Dict[str, str] = {}
        self.call_log: list = []

    def add_response(self, prompt_contains: str, response: str) -> None:
        """Add a mock response for prompts containing the given string."""
        self.responses[prompt_contains] = response

    def __call__(self, prompt: str, model: str) -> str:
        """Return a mock response."""
        self.call_log.append({"prompt": prompt, "model": model})

        for key, response in self.responses.items():
            if key in prompt:
                return response

        # Default: return a generic JSON response
        return '{"result": "mock_response"}'


# ============================================================================
# Utility Functions
# ============================================================================

def list_available_models() -> List[str]:
    """List all available models from LLMS.json."""
    config = get_config()
    return list(config.models.keys())


def get_model_info(model: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    config = get_config()
    model_config = config.get_model(model)
    return {
        "name": model_config.name,
        "provider": model_config.provider,
        "model_id": model_config.model_id,
        "capabilities": model_config.capabilities,
        "description": model_config.description,
        "cost": model_config.cost,
        "context_window": model_config.context_window,
        "max_output": model_config.max_output,
        "is_local": model_config.is_local,
    }


def select_model_by_capability(capability: str) -> str:
    """Select the cheapest model that has a given capability."""
    config = get_config()

    matching = []
    for name, model in config.models.items():
        if capability in model.capabilities:
            matching.append(model)

    if not matching:
        raise ConfigError(f"No model found with capability: {capability}")

    # Sort by cost (local models are "free")
    def get_cost(m: ModelConfig) -> float:
        if m.cost == "local":
            return 0.0
        if isinstance(m.cost, dict):
            return m.cost.get("input", 0) + m.cost.get("output", 0)
        return float('inf')

    matching.sort(key=get_cost)
    return matching[0].name
