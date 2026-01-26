"""
Core ptool (pseudo-tool) decorator and registry.

A ptool is a function that:
1. Has a typed signature (input types, return type)
2. Has a docstring that serves as the prompt template
3. Is executed by calling an LLM, not by running Python code

Example:
    @ptool(model="claude-3-5-sonnet")
    def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
        '''Classify the emotional tone of the given text.'''
        ...
"""

from __future__ import annotations

import functools
import inspect
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, ValidationError

# Type variable for generic ptool return types
T = TypeVar("T")


@dataclass
class PToolExample:
    """An in-context learning example for a ptool."""
    inputs: Dict[str, Any]
    output: Any
    reasoning: Optional[str] = None  # Optional chain-of-thought


@dataclass
class PToolSpec:
    """Specification for a ptool - all metadata needed to execute it."""
    name: str
    docstring: str
    parameters: Dict[str, Type]  # param_name -> type
    return_type: Type
    model: str = "deepseek-v3-0324"
    output_mode: Literal["structured", "freeform"] = "structured"
    examples: List[PToolExample] = field(default_factory=list)
    func: Optional[Callable] = None  # Original function (for reference)

    def get_signature_str(self) -> str:
        """Return a string representation of the function signature."""
        params = ", ".join(
            f"{name}: {self._type_to_str(typ)}"
            for name, typ in self.parameters.items()
        )
        return_str = self._type_to_str(self.return_type)
        return f"{self.name}({params}) -> {return_str}"

    def _type_to_str(self, typ: Type) -> str:
        """Convert a type to its string representation."""
        origin = get_origin(typ)
        if origin is not None:
            args = get_args(typ)
            if origin is Literal:
                return f"Literal[{', '.join(repr(a) for a in args)}]"
            elif origin is list or origin is List:
                return f"List[{self._type_to_str(args[0])}]"
            elif origin is dict or origin is Dict:
                return f"Dict[{self._type_to_str(args[0])}, {self._type_to_str(args[1])}]"
            elif origin is tuple or origin is Tuple:
                return f"Tuple[{', '.join(self._type_to_str(a) for a in args)}]"
            elif origin is Union:
                return f"Union[{', '.join(self._type_to_str(a) for a in args)}]"
        if hasattr(typ, "__name__"):
            return typ.__name__
        return str(typ)

    def format_prompt(self, **kwargs) -> str:
        """Format the prompt for LLM execution."""
        lines = [
            f"You are executing the function `{self.name}`.",
            "",
            f"Function signature: {self.get_signature_str()}",
            "",
            "Description:",
            self.docstring.strip(),
            "",
        ]

        # Add examples if present
        if self.examples:
            lines.append("Examples:")
            for i, ex in enumerate(self.examples, 1):
                lines.append(f"  Example {i}:")
                lines.append(f"    Input: {json.dumps(ex.inputs)}")
                if ex.reasoning:
                    lines.append(f"    Reasoning: {ex.reasoning}")
                lines.append(f"    Output: {json.dumps(ex.output)}")
            lines.append("")

        # Add current inputs
        lines.append("Current inputs:")
        for param_name, param_value in kwargs.items():
            lines.append(f"  {param_name} = {json.dumps(param_value) if not isinstance(param_value, str) else repr(param_value)}")
        lines.append("")

        # Add output format instructions
        if self.output_mode == "structured":
            lines.append("Return ONLY a valid JSON object with the result.")
            lines.append('Format: {"result": <your answer>}')
        else:
            lines.append("Provide your reasoning, then give the final answer.")
            lines.append("Format your final answer on the last line as: ANSWER: <your answer>")

        return "\n".join(lines)


class PToolRegistry:
    """Registry of all defined ptools."""

    def __init__(self):
        self._ptools: Dict[str, PToolSpec] = {}

    def register(self, spec: PToolSpec) -> None:
        """Register a ptool specification."""
        self._ptools[spec.name] = spec

    def get(self, name: str) -> Optional[PToolSpec]:
        """Get a ptool by name."""
        return self._ptools.get(name)

    def list_all(self) -> List[PToolSpec]:
        """List all registered ptools."""
        return list(self._ptools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._ptools

    def __len__(self) -> int:
        return len(self._ptools)


# Global registry
_REGISTRY = PToolRegistry()


def get_registry() -> PToolRegistry:
    """Get the global ptool registry."""
    return _REGISTRY


class PToolWrapper(Generic[T]):
    """Wrapper that makes a ptool callable and tracks its spec."""

    def __init__(
        self,
        spec: PToolSpec,
        llm_backend: Optional[Callable] = None,
    ):
        self.spec = spec
        self._llm_backend = llm_backend
        # Copy function metadata
        if spec.func:
            functools.update_wrapper(self, spec.func)

    def __call__(self, *args, **kwargs) -> T:
        """Execute the ptool via LLM."""
        # Convert positional args to kwargs
        if args:
            param_names = list(self.spec.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(param_names):
                    kwargs[param_names[i]] = arg

        # Import here to avoid circular dependency
        from .llm_backend import execute_ptool

        return execute_ptool(self.spec, kwargs, self._llm_backend)

    def __repr__(self) -> str:
        return f"<ptool {self.spec.get_signature_str()}>"


def ptool(
    model: str = "deepseek-v3-0324",
    output_mode: Literal["structured", "freeform"] = "structured",
    examples: Optional[List[PToolExample]] = None,
    llm_backend: Optional[Callable] = None,
) -> Callable[[Callable[..., T]], PToolWrapper[T]]:
    """
    Decorator to mark a function as a ptool (pseudo-tool).

    The decorated function's body is ignored - execution happens via LLM.
    The function signature and docstring define the ptool's interface.

    Args:
        model: LLM model to use for execution
        output_mode: "structured" for JSON output, "freeform" for text
        examples: In-context learning examples
        llm_backend: Optional custom LLM backend function

    Example:
        @ptool(model="claude-3-5-sonnet")
        def classify_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
            '''Classify the emotional tone of the given text.'''
            ...
    """
    def decorator(func: Callable[..., T]) -> PToolWrapper[T]:
        # Extract type hints
        hints = get_type_hints(func)
        return_type = hints.pop("return", Any)

        # Create spec
        spec = PToolSpec(
            name=func.__name__,
            docstring=func.__doc__ or "",
            parameters=hints,
            return_type=return_type,
            model=model,
            output_mode=output_mode,
            examples=examples or [],
            func=func,
        )

        # Register
        _REGISTRY.register(spec)

        # Return wrapper
        wrapper = PToolWrapper(spec, llm_backend)
        return wrapper

    return decorator


# Convenience function for creating examples
def example(
    inputs: Dict[str, Any],
    output: Any,
    reasoning: Optional[str] = None,
) -> PToolExample:
    """Create an in-context learning example for a ptool."""
    return PToolExample(inputs=inputs, output=output, reasoning=reasoning)
