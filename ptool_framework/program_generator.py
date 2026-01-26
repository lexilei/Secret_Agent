"""
Program Generator: Creates Python programs from task descriptions.

This is the core of the pipeline:
1. Takes a natural language task description
2. Analyzes what capabilities are needed
3. Generates @ptool definitions for LLM-handled parts
4. Generates Python workflow code for control flow
5. Outputs a complete, runnable .py file

The generated program follows the philosophy:
"Python does maximum heavy lifting before handing to LLMs"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .ptool import ptool, PToolSpec, get_registry
from .llm_backend import call_llm, get_config


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PToolDefinition:
    """A ptool definition to be generated."""
    name: str
    parameters: List[Tuple[str, str]]  # [(name, type_str), ...]
    return_type: str
    docstring: str
    model: str = "deepseek-v3-0324"
    output_mode: str = "structured"


@dataclass
class WorkflowStep:
    """A step in the generated workflow."""
    description: str
    code: str
    is_ptool_call: bool = False
    ptool_name: Optional[str] = None


@dataclass
class GeneratedProgram:
    """Result of program generation."""
    task_description: str
    ptools: List[PToolDefinition]
    workflow_code: str
    full_code: str
    output_path: Optional[str] = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Internal ptools for generation (meta-level)
# ============================================================================

@ptool(model="deepseek-v3-0324", output_mode="structured")
def analyze_task(task_description: str) -> Dict[str, Any]:
    """Analyze a task description and identify what capabilities are needed.

    Given a natural language task description, analyze what the task requires:
    - What are the inputs?
    - What are the outputs?
    - What sub-tasks can be done in pure Python?
    - What sub-tasks require LLM reasoning?

    Return a JSON object with:
    {
        "task_summary": "Brief summary of the task",
        "inputs": [{"name": "input_name", "type": "str", "description": "..."}],
        "output": {"type": "dict", "description": "..."},
        "python_tasks": ["task1", "task2"],  // Things pure Python can handle
        "llm_tasks": ["task3", "task4"]      // Things needing LLM reasoning
    }
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="structured")
def design_ptools(
    task_description: str,
    llm_tasks: List[str],
) -> List[Dict[str, Any]]:
    """Design ptool definitions for tasks requiring LLM reasoning.

    For each LLM task, design a ptool with:
    - Clear function name
    - Typed parameters
    - Typed return value
    - Clear docstring explaining what it does

    Return a list of ptool definitions:
    [
        {
            "name": "extract_food_items",
            "parameters": [["text", "str"]],
            "return_type": "List[str]",
            "docstring": "Extract all food items mentioned...",
            "output_mode": "structured"
        },
        ...
    ]
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="freeform")
def generate_workflow(
    task_description: str,
    ptools: List[Dict[str, Any]],
    python_tasks: List[str],
) -> str:
    """Generate the main workflow function that orchestrates ptools and Python code.

    Given the task description, available ptools, and pure Python tasks:
    1. Design a main function that accomplishes the task
    2. Use ptools for LLM reasoning steps
    3. Use pure Python for everything else (loops, conditions, data manipulation)
    4. Return well-formatted Python code

    The workflow should follow this pattern:
    ```python
    def main_workflow(input_data: str) -> dict:
        '''Main workflow - Python controls flow, ptools do the thinking.'''
        # Step 1: Parse or validate input (Python)
        # Step 2: Call ptool for reasoning
        # Step 3: Process result (Python)
        # ...
        return result
    ```

    ANSWER: [the complete Python function code]
    """
    ...


# ============================================================================
# Program Generator
# ============================================================================

class ProgramGenerator:
    """
    Generates Python programs from task descriptions.

    Usage:
        generator = ProgramGenerator()
        result = generator.generate(
            task_description="Analyze customer reviews and extract sentiment",
            output_path="review_analyzer.py"
        )
    """

    def __init__(
        self,
        available_ptools: Optional[List[PToolSpec]] = None,
        available_tools: Optional[List[str]] = None,
        available_llms: Optional[List[str]] = None,
        trace_store=None,
        default_model: str = "deepseek-v3-0324",
    ):
        """
        Initialize the program generator.

        Args:
            available_ptools: Existing ptools to potentially reuse
            available_tools: External tools (MCP servers, etc.) - future use
            available_llms: Available LLMs from LLMS.json
            trace_store: TraceStore for checking distilled versions
            default_model: Default LLM model for generated ptools
        """
        self.available_ptools = available_ptools or []
        self.available_tools = available_tools or []
        self.available_llms = available_llms or self._get_available_llms()
        self.trace_store = trace_store
        self.default_model = default_model

    def _get_available_llms(self) -> List[str]:
        """Get list of available LLMs from LLMS.json."""
        try:
            config = get_config()
            return list(config.models.keys())
        except Exception:
            return ["deepseek-v3-0324"]

    def generate(
        self,
        task_description: str,
        output_path: Optional[str] = None,
        prefer_distilled: bool = True,
    ) -> GeneratedProgram:
        """
        Generate a complete Python program from a task description.

        Args:
            task_description: Natural language description of the task
            output_path: Path to write the generated .py file
            prefer_distilled: Whether to use distilled versions of ptools if available

        Returns:
            GeneratedProgram with the generated code
        """
        logger.info(f"Generating program for: {task_description[:100]}...")

        # Step 1: Analyze the task
        logger.info("Step 1: Analyzing task requirements...")
        analysis = analyze_task(task_description)

        # Step 2: Design ptools for LLM tasks
        logger.info("Step 2: Designing ptools...")
        llm_tasks = analysis.get("llm_tasks", [])
        if llm_tasks:
            ptool_designs = design_ptools(task_description, llm_tasks)
        else:
            ptool_designs = []

        # Convert to PToolDefinition objects
        ptools = []
        for design in ptool_designs:
            ptools.append(PToolDefinition(
                name=design["name"],
                parameters=[(p[0], p[1]) for p in design.get("parameters", [])],
                return_type=design.get("return_type", "Any"),
                docstring=design.get("docstring", ""),
                model=self.default_model,
                output_mode=design.get("output_mode", "structured"),
            ))

        # Step 3: Generate workflow
        logger.info("Step 3: Generating workflow...")
        python_tasks = analysis.get("python_tasks", [])
        workflow_code = generate_workflow(
            task_description,
            ptool_designs,
            python_tasks,
        )

        # Step 4: Assemble full program
        logger.info("Step 4: Assembling program...")
        full_code = self._assemble_program(
            task_description=task_description,
            analysis=analysis,
            ptools=ptools,
            workflow_code=workflow_code,
            prefer_distilled=prefer_distilled,
        )

        # Step 5: Write to file if requested
        if output_path:
            logger.info(f"Writing to {output_path}...")
            with open(output_path, "w") as f:
                f.write(full_code)

        result = GeneratedProgram(
            task_description=task_description,
            ptools=ptools,
            workflow_code=workflow_code,
            full_code=full_code,
            output_path=output_path,
        )

        logger.info(f"Generated program with {len(ptools)} ptools")
        return result

    def _assemble_program(
        self,
        task_description: str,
        analysis: Dict[str, Any],
        ptools: List[PToolDefinition],
        workflow_code: str,
        prefer_distilled: bool,
    ) -> str:
        """Assemble all parts into a complete Python program."""

        # Header
        header = f'''#!/usr/bin/env python3
"""
Generated Agent Program
========================
Task: {task_description}

Generated at: {datetime.now().isoformat()}
Generator: ptool_framework v0.2.0

This program follows the "Python calling LLMs" paradigm:
- Python handles control flow, data manipulation, and validation
- @ptool functions delegate reasoning to LLMs
- @distilled functions try pure Python first, fall back to LLM

Usage:
    python {Path(task_description).stem if task_description else 'program'}.py <input>
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Add ptool_framework to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up environment
os.environ.setdefault("TOGETHER_API_KEY", "")  # Set your API key

from ptool_framework import ptool, distilled, DistillationFallback, enable_tracing

# Enable trace collection for observability
enable_tracing(True)
'''

        # Configuration section
        config_section = f'''
# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "{self.default_model}"
TRACE_LOGGING = True

'''

        # Generate ptool definitions
        ptool_section = '''
# ============================================================================
# PTOOLS (LLM-executed functions)
# ============================================================================

'''
        for pt in ptools:
            ptool_section += self._generate_ptool_code(pt)
            ptool_section += "\n\n"

        # Workflow section
        workflow_section = f'''
# ============================================================================
# WORKFLOW
# ============================================================================

{workflow_code}
'''

        # Main entry point
        main_section = '''
# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\\n")[2])
    parser.add_argument("input", nargs="?", help="Input data or file path")
    parser.add_argument("--file", "-f", help="Read input from file")
    parser.add_argument("--trace", action="store_true", help="Enable detailed tracing")

    args = parser.parse_args()

    # Get input
    if args.file:
        with open(args.file) as f:
            input_data = f.read()
    elif args.input:
        input_data = args.input
    else:
        print("Reading from stdin...")
        input_data = sys.stdin.read()

    # Run workflow
    try:
        result = main_workflow(input_data)
        print("\\n=== Result ===")
        if isinstance(result, dict):
            import json
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
'''

        return header + config_section + ptool_section + workflow_section + main_section

    def _generate_ptool_code(self, pt: PToolDefinition) -> str:
        """Generate Python code for a ptool definition."""
        # Format parameters
        params = ", ".join(f"{name}: {typ}" for name, typ in pt.parameters)

        # Format docstring
        docstring = pt.docstring.replace('"""', "'''")

        return f'''@ptool(model=DEFAULT_MODEL, output_mode="{pt.output_mode}")
def {pt.name}({params}) -> {pt.return_type}:
    """{docstring}"""
    ...  # LLM handles this'''


# ============================================================================
# Convenience functions
# ============================================================================

def generate_program(
    task_description: str,
    output_path: Optional[str] = None,
    **kwargs,
) -> GeneratedProgram:
    """
    Convenience function to generate a program.

    Args:
        task_description: What the program should do
        output_path: Where to save the .py file

    Returns:
        GeneratedProgram with the generated code
    """
    generator = ProgramGenerator(**kwargs)
    return generator.generate(task_description, output_path)


def generate_from_template(
    template: str,
    task_description: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a program using a predefined template.

    Templates provide common patterns that can be customized.
    """
    templates = {
        "analyzer": _ANALYZER_TEMPLATE,
        "classifier": _CLASSIFIER_TEMPLATE,
        "extractor": _EXTRACTOR_TEMPLATE,
    }

    if template not in templates:
        raise ValueError(f"Unknown template: {template}. Available: {list(templates.keys())}")

    code = templates[template].format(
        task_description=task_description,
        timestamp=datetime.now().isoformat(),
    )

    if output_path:
        with open(output_path, "w") as f:
            f.write(code)

    return code


# ============================================================================
# Templates
# ============================================================================

_ANALYZER_TEMPLATE = '''#!/usr/bin/env python3
"""
Analyzer: {task_description}
Generated at: {timestamp}
"""

import sys
from typing import List, Dict, Any
from ptool_framework import ptool

@ptool()
def extract_items(text: str) -> List[str]:
    """Extract relevant items from the text."""
    ...

@ptool()
def analyze_item(item: str) -> Dict[str, Any]:
    """Analyze a single item and return structured data."""
    ...

@ptool()
def summarize_analysis(items: List[Dict[str, Any]]) -> str:
    """Summarize the analysis results."""
    ...

def main_workflow(input_text: str) -> Dict[str, Any]:
    """Main analysis workflow."""
    # Extract items
    items = extract_items(input_text)
    print(f"Found {{len(items)}} items")

    # Analyze each item
    analyses = []
    for item in items:
        analysis = analyze_item(item)
        analyses.append(analysis)

    # Summarize
    summary = summarize_analysis(analyses)

    return {{
        "items": items,
        "analyses": analyses,
        "summary": summary,
    }}

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else input("Enter text: ")
    result = main_workflow(text)
    print(result)
'''

_CLASSIFIER_TEMPLATE = '''#!/usr/bin/env python3
"""
Classifier: {task_description}
Generated at: {timestamp}
"""

import sys
from typing import List, Literal
from ptool_framework import ptool

Categories = Literal["category_a", "category_b", "category_c", "other"]

@ptool()
def classify(text: str) -> Categories:
    """Classify the input text into a category."""
    ...

@ptool()
def explain_classification(text: str, category: str) -> str:
    """Explain why the text was classified into the given category."""
    ...

def main_workflow(input_text: str) -> dict:
    """Classification workflow."""
    category = classify(input_text)
    explanation = explain_classification(input_text, category)

    return {{
        "input": input_text,
        "category": category,
        "explanation": explanation,
    }}

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else input("Enter text: ")
    result = main_workflow(text)
    print(f"Category: {{result['category']}}")
    print(f"Explanation: {{result['explanation']}}")
'''

_EXTRACTOR_TEMPLATE = '''#!/usr/bin/env python3
"""
Extractor: {task_description}
Generated at: {timestamp}
"""

import sys
from typing import List, Dict, Any
from ptool_framework import ptool

@ptool()
def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract named entities from text.

    Return list of dicts with 'text', 'type', and 'context' keys.
    """
    ...

@ptool()
def extract_relationships(text: str, entities: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Extract relationships between entities.

    Return list of dicts with 'subject', 'predicate', 'object' keys.
    """
    ...

def main_workflow(input_text: str) -> dict:
    """Entity extraction workflow."""
    entities = extract_entities(input_text)
    relationships = extract_relationships(input_text, entities)

    return {{
        "entities": entities,
        "relationships": relationships,
        "entity_count": len(entities),
        "relationship_count": len(relationships),
    }}

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else input("Enter text: ")
    result = main_workflow(text)
    print(f"Found {{result['entity_count']}} entities and {{result['relationship_count']}} relationships")
'''
