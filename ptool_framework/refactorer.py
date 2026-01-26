"""
Code Refactorer: AST-based program transformation for distillation and expansion.

This module handles bidirectional refactoring:
- Distill: Replace @ptool functions with @distilled (LLM → Python)
- Expand: Replace pure Python functions with @ptool (Python → LLM)

The refactorer uses AST manipulation to:
1. Parse existing Python code
2. Identify refactoring candidates
3. Apply transformations
4. Validate the result compiles and runs

This is an LLM-heavy module - it uses LLMs to analyze and generate code.
"""

from __future__ import annotations

import ast
import copy
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .ptool import ptool, PToolSpec, get_registry
from .trace_store import TraceStore, get_trace_store
from .distiller import BehaviorDistiller, DistillationResult, DistillationAnalysis
from .llm_backend import call_llm


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FunctionInfo:
    """Information about a function in the source code."""
    name: str
    lineno: int
    end_lineno: int
    decorators: List[str]
    parameters: List[Tuple[str, Optional[str]]]  # (name, type_annotation)
    return_type: Optional[str]
    docstring: Optional[str]
    body_source: str
    is_ptool: bool = False
    is_distilled: bool = False
    is_pure_python: bool = False


@dataclass
class RefactorCandidate:
    """A function that can be refactored."""
    function: FunctionInfo
    mode: Literal["distill", "expand"]
    reason: str
    confidence: float
    trace_count: int = 0
    estimated_coverage: float = 0


@dataclass
class RefactorChange:
    """A change to be applied during refactoring."""
    function_name: str
    mode: Literal["distill", "expand"]
    original_source: str
    new_source: str
    start_line: int
    end_line: int
    success: bool
    error: Optional[str] = None


@dataclass
class RefactorResult:
    """Result of refactoring a program."""
    source_path: str
    output_path: str
    mode: Literal["distill", "expand"]
    changes: List[RefactorChange]
    original_code: str
    refactored_code: str
    compile_success: bool
    validation_passed: bool
    error: Optional[str] = None


# ============================================================================
# LLM-Powered Analysis ptools
# ============================================================================

@ptool(model="deepseek-v3-0324", output_mode="structured")
def analyze_function_for_expansion(
    function_name: str,
    function_source: str,
    docstring: Optional[str],
    parameter_types: List[Tuple[str, str]],
    return_type: str,
) -> Dict[str, Any]:
    """Analyze a pure Python function to determine if it should be expanded to @ptool.

    Consider expanding to @ptool if:
    - The function involves complex decision-making that could benefit from LLM reasoning
    - The function handles natural language or unstructured data
    - The function has many edge cases that are hard to enumerate
    - The function would benefit from being more flexible/robust

    Do NOT expand if:
    - The function is purely computational (math, string formatting, etc.)
    - The function is a simple data transformation
    - The function is deterministic and handles all cases well

    Return:
    {
        "should_expand": true/false,
        "reason": "Why this should/shouldn't be expanded",
        "complexity_score": 0.0-1.0,  // How complex is the decision-making
        "nlp_score": 0.0-1.0,         // How much does it deal with NL/unstructured data
        "edge_case_score": 0.0-1.0,   // How many edge cases might exist
        "confidence": 0.0-1.0
    }
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="freeform")
def generate_ptool_from_function(
    function_name: str,
    function_source: str,
    docstring: Optional[str],
    parameter_types: List[Tuple[str, str]],
    return_type: str,
) -> str:
    """Convert a pure Python function into a @ptool definition.

    Create a @ptool version that:
    1. Has the same signature
    2. Has a comprehensive docstring explaining what it does
    3. Has ... as the body (LLM will execute)
    4. Uses appropriate output_mode (structured for typed returns, freeform for strings)

    ANSWER:
    ```python
    @ptool(model="deepseek-v3-0324", output_mode="structured")
    def {function_name}(...) -> ...:
        '''Detailed docstring explaining the function's purpose and expected behavior.'''
        ...
    ```
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="structured")
def suggest_refactoring_strategy(
    source_code: str,
    ptool_functions: List[str],
    pure_functions: List[str],
    distilled_functions: List[str],
    trace_info: Dict[str, int],
) -> Dict[str, Any]:
    """Suggest an overall refactoring strategy for a program.

    Analyze the program structure and suggest which functions to:
    - Distill (convert @ptool to @distilled with Python implementation)
    - Expand (convert pure Python to @ptool for more flexibility)
    - Keep unchanged

    Return:
    {
        "distill_candidates": [
            {"name": "func1", "reason": "Has 50+ traces, simple patterns", "priority": "high"}
        ],
        "expand_candidates": [
            {"name": "func2", "reason": "Complex NLP logic, hard to maintain", "priority": "medium"}
        ],
        "keep_unchanged": ["func3"],
        "overall_recommendation": "Summary of recommended changes",
        "risk_assessment": "Potential risks of refactoring"
    }
    """
    ...


# ============================================================================
# AST Utilities
# ============================================================================

class FunctionVisitor(ast.NodeVisitor):
    """AST visitor to extract function information."""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.functions: List[FunctionInfo] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)
        self.generic_visit(node)

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        # Extract decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)

        # Determine function type
        is_ptool = "ptool" in decorators
        is_distilled = "distilled" in decorators
        is_pure_python = not is_ptool and not is_distilled

        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = None
            if arg.annotation:
                param_type = ast.unparse(arg.annotation)
            parameters.append((param_name, param_type))

        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract body source
        start_line = node.lineno - 1
        end_line = node.end_lineno if node.end_lineno else node.lineno
        body_source = "\n".join(self.source_lines[start_line:end_line])

        self.functions.append(FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=end_line,
            decorators=decorators,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            body_source=body_source,
            is_ptool=is_ptool,
            is_distilled=is_distilled,
            is_pure_python=is_pure_python,
        ))


def extract_functions(source_code: str) -> List[FunctionInfo]:
    """Extract all function definitions from source code."""
    tree = ast.parse(source_code)
    source_lines = source_code.split('\n')
    visitor = FunctionVisitor(source_lines)
    visitor.visit(tree)
    return visitor.functions


def get_function_source_range(source_code: str, func: FunctionInfo) -> Tuple[int, int]:
    """Get the exact line range for a function including decorators."""
    lines = source_code.split('\n')

    # Find the start (including decorators)
    start_line = func.lineno - 1
    # Look backwards for decorators
    while start_line > 0:
        prev_line = lines[start_line - 1].strip()
        if prev_line.startswith('@') or prev_line == '':
            start_line -= 1
        else:
            break

    # Skip leading empty lines
    while start_line < len(lines) and lines[start_line].strip() == '':
        start_line += 1

    return start_line, func.end_lineno


# ============================================================================
# Code Refactorer
# ============================================================================

class CodeRefactorer:
    """
    Refactors Python programs for distillation or expansion.

    Usage:
        refactorer = CodeRefactorer()

        # Distill: Replace @ptool with @distilled
        result = refactorer.refactor(
            source_path="program.py",
            output_path="program_distilled.py",
            mode="distill"
        )

        # Expand: Replace pure Python with @ptool
        result = refactorer.refactor(
            source_path="program.py",
            output_path="program_expanded.py",
            mode="expand"
        )
    """

    def __init__(
        self,
        trace_store: Optional[TraceStore] = None,
        distiller: Optional[BehaviorDistiller] = None,
        min_traces_for_distill: int = 10,
        min_confidence_for_expand: float = 0.7,
        validate_output: bool = True,
    ):
        """
        Initialize the refactorer.

        Args:
            trace_store: TraceStore for checking distillation candidates
            distiller: BehaviorDistiller for generating distilled implementations
            min_traces_for_distill: Minimum traces needed to distill a ptool
            min_confidence_for_expand: Minimum confidence to expand a function
            validate_output: Whether to compile-test the output
        """
        self.trace_store = trace_store or get_trace_store()
        self.distiller = distiller or BehaviorDistiller(trace_store=self.trace_store)
        self.min_traces_for_distill = min_traces_for_distill
        self.min_confidence_for_expand = min_confidence_for_expand
        self.validate_output = validate_output

    def analyze(self, source_path: str) -> Dict[str, Any]:
        """
        Analyze a source file for refactoring opportunities.

        Args:
            source_path: Path to the Python file to analyze

        Returns:
            Analysis with distill and expand candidates
        """
        logger.info(f"Analyzing {source_path} for refactoring opportunities...")

        with open(source_path) as f:
            source_code = f.read()

        functions = extract_functions(source_code)

        # Categorize functions
        ptool_functions = [f for f in functions if f.is_ptool]
        distilled_functions = [f for f in functions if f.is_distilled]
        pure_functions = [f for f in functions if f.is_pure_python]

        # Get trace info for ptools
        trace_info = {}
        for func in ptool_functions:
            count = self.trace_store.get_trace_count(func.name)
            trace_info[func.name] = count

        # Analyze distill candidates
        distill_candidates = []
        for func in ptool_functions:
            trace_count = trace_info.get(func.name, 0)
            if trace_count >= self.min_traces_for_distill:
                analysis = self.distiller.analyze(func.name)
                distill_candidates.append({
                    "name": func.name,
                    "trace_count": trace_count,
                    "estimated_coverage": analysis.estimated_coverage,
                    "recommendation": analysis.recommendation,
                    "reason": analysis.notes or "Sufficient traces for distillation",
                })

        # Analyze expand candidates (use LLM)
        expand_candidates = []
        for func in pure_functions:
            if func.name.startswith('_'):
                continue  # Skip private functions

            param_types = [(n, t or "Any") for n, t in func.parameters]
            analysis = analyze_function_for_expansion(
                function_name=func.name,
                function_source=func.body_source,
                docstring=func.docstring,
                parameter_types=param_types,
                return_type=func.return_type or "Any",
            )

            if analysis.get("should_expand"):
                expand_candidates.append({
                    "name": func.name,
                    "reason": analysis.get("reason", ""),
                    "complexity_score": analysis.get("complexity_score", 0),
                    "nlp_score": analysis.get("nlp_score", 0),
                    "confidence": analysis.get("confidence", 0),
                })

        return {
            "source_path": source_path,
            "total_functions": len(functions),
            "ptool_functions": [f.name for f in ptool_functions],
            "distilled_functions": [f.name for f in distilled_functions],
            "pure_functions": [f.name for f in pure_functions],
            "trace_info": trace_info,
            "distill_candidates": distill_candidates,
            "expand_candidates": expand_candidates,
        }

    def refactor(
        self,
        source_path: str,
        output_path: str,
        mode: Literal["distill", "expand"],
        functions: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> RefactorResult:
        """
        Refactor a program.

        Args:
            source_path: Path to the source Python file
            output_path: Path to write the refactored code
            mode: "distill" to convert @ptool to @distilled, "expand" for reverse
            functions: Specific functions to refactor (None = all candidates)
            dry_run: If True, don't write output file

        Returns:
            RefactorResult with the changes made
        """
        logger.info(f"Refactoring {source_path} in {mode} mode...")

        with open(source_path) as f:
            source_code = f.read()

        all_functions = extract_functions(source_code)
        changes: List[RefactorChange] = []

        if mode == "distill":
            changes = self._distill_functions(source_code, all_functions, functions)
        else:  # expand
            changes = self._expand_functions(source_code, all_functions, functions)

        # Apply changes to source
        refactored_code = self._apply_changes(source_code, changes)

        # Validate
        compile_success = True
        validation_passed = True
        error = None

        if self.validate_output and changes:
            compile_result = self._compile_test(refactored_code)
            compile_success = compile_result["success"]
            if not compile_success:
                error = compile_result["error"]
                validation_passed = False

        # Write output
        if not dry_run and compile_success:
            with open(output_path, 'w') as f:
                f.write(refactored_code)
            logger.info(f"Wrote refactored code to {output_path}")

        return RefactorResult(
            source_path=source_path,
            output_path=output_path,
            mode=mode,
            changes=changes,
            original_code=source_code,
            refactored_code=refactored_code,
            compile_success=compile_success,
            validation_passed=validation_passed,
            error=error,
        )

    def _distill_functions(
        self,
        source_code: str,
        all_functions: List[FunctionInfo],
        target_functions: Optional[List[str]],
    ) -> List[RefactorChange]:
        """Distill @ptool functions to @distilled."""
        changes = []

        ptool_functions = [f for f in all_functions if f.is_ptool]
        if target_functions:
            ptool_functions = [f for f in ptool_functions if f.name in target_functions]

        for func in ptool_functions:
            logger.info(f"Distilling {func.name}...")

            # Check if we have enough traces
            trace_count = self.trace_store.get_trace_count(func.name)
            if trace_count < self.min_traces_for_distill:
                logger.warning(f"  Skipping {func.name}: only {trace_count} traces")
                continue

            # Run distillation
            result = self.distiller.distill(func.name)

            if result.success and result.distilled_code:
                start_line, end_line = get_function_source_range(source_code, func)
                changes.append(RefactorChange(
                    function_name=func.name,
                    mode="distill",
                    original_source=func.body_source,
                    new_source=result.distilled_code,
                    start_line=start_line,
                    end_line=end_line,
                    success=True,
                ))
                logger.info(f"  Successfully distilled {func.name}")
            else:
                changes.append(RefactorChange(
                    function_name=func.name,
                    mode="distill",
                    original_source=func.body_source,
                    new_source="",
                    start_line=func.lineno,
                    end_line=func.end_lineno,
                    success=False,
                    error=result.error,
                ))
                logger.warning(f"  Failed to distill {func.name}: {result.error}")

        return changes

    def _expand_functions(
        self,
        source_code: str,
        all_functions: List[FunctionInfo],
        target_functions: Optional[List[str]],
    ) -> List[RefactorChange]:
        """Expand pure Python functions to @ptool."""
        changes = []

        pure_functions = [f for f in all_functions if f.is_pure_python]
        if target_functions:
            pure_functions = [f for f in pure_functions if f.name in target_functions]

        for func in pure_functions:
            if func.name.startswith('_'):
                continue  # Skip private functions

            logger.info(f"Analyzing {func.name} for expansion...")

            # Analyze if should expand
            param_types = [(n, t or "Any") for n, t in func.parameters]
            analysis = analyze_function_for_expansion(
                function_name=func.name,
                function_source=func.body_source,
                docstring=func.docstring,
                parameter_types=param_types,
                return_type=func.return_type or "Any",
            )

            if not analysis.get("should_expand"):
                logger.info(f"  Skipping {func.name}: {analysis.get('reason', 'not suitable')}")
                continue

            confidence = analysis.get("confidence", 0)
            if confidence < self.min_confidence_for_expand:
                logger.info(f"  Skipping {func.name}: confidence {confidence:.2f} too low")
                continue

            # Generate @ptool version
            logger.info(f"  Generating @ptool for {func.name}...")
            ptool_code = generate_ptool_from_function(
                function_name=func.name,
                function_source=func.body_source,
                docstring=func.docstring,
                parameter_types=param_types,
                return_type=func.return_type or "Any",
            )

            # Extract code from response
            new_code = self._extract_code(ptool_code)
            if not new_code:
                changes.append(RefactorChange(
                    function_name=func.name,
                    mode="expand",
                    original_source=func.body_source,
                    new_source="",
                    start_line=func.lineno,
                    end_line=func.end_lineno,
                    success=False,
                    error="Could not extract code from LLM response",
                ))
                continue

            # Also keep original as a fallback (comment it out)
            original_commented = self._comment_out_function(func.body_source, func.name)
            new_code = new_code + "\n\n" + original_commented

            start_line, end_line = get_function_source_range(source_code, func)
            changes.append(RefactorChange(
                function_name=func.name,
                mode="expand",
                original_source=func.body_source,
                new_source=new_code,
                start_line=start_line,
                end_line=end_line,
                success=True,
            ))
            logger.info(f"  Successfully expanded {func.name}")

        return changes

    def _apply_changes(
        self,
        source_code: str,
        changes: List[RefactorChange],
    ) -> str:
        """Apply refactoring changes to source code."""
        if not changes:
            return source_code

        # Sort changes by line number in reverse order (apply from bottom up)
        successful_changes = [c for c in changes if c.success]
        sorted_changes = sorted(successful_changes, key=lambda c: c.start_line, reverse=True)

        lines = source_code.split('\n')

        for change in sorted_changes:
            # Replace lines
            new_lines = change.new_source.split('\n')
            lines = lines[:change.start_line] + new_lines + lines[change.end_line:]

        return '\n'.join(lines)

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        import re

        # Try to find code in ```python ... ``` blocks
        code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try to find code in ``` ... ``` blocks
        code_match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Look for ANSWER: prefix
        if "ANSWER:" in response:
            after_answer = response.split("ANSWER:", 1)[1]
            code_match = re.search(r'```(?:python)?\s*(.*?)```', after_answer, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

        # If response looks like pure code, use it
        if response.strip().startswith(("@ptool", "from ", "import ", "def ")):
            return response.strip()

        return None

    def _comment_out_function(self, source: str, func_name: str) -> str:
        """Comment out a function as a backup."""
        lines = source.split('\n')
        commented = [f"# Original {func_name} (backup):"]
        for line in lines:
            commented.append(f"# {line}")
        return '\n'.join(commented)

    def _compile_test(self, code: str) -> Dict[str, Any]:
        """Test if code compiles successfully."""
        try:
            ast.parse(code)
            return {"success": True, "error": None}
        except SyntaxError as e:
            return {"success": False, "error": f"SyntaxError at line {e.lineno}: {e.msg}"}


# ============================================================================
# Convenience Functions
# ============================================================================

def refactor_program(
    source_path: str,
    output_path: str,
    mode: Literal["distill", "expand"],
    **kwargs,
) -> RefactorResult:
    """Convenience function to refactor a program."""
    refactorer = CodeRefactorer(**kwargs)
    return refactorer.refactor(source_path, output_path, mode)


def analyze_program(source_path: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to analyze a program for refactoring."""
    refactorer = CodeRefactorer(**kwargs)
    return refactorer.analyze(source_path)


def distill_program(
    source_path: str,
    output_path: Optional[str] = None,
    **kwargs,
) -> RefactorResult:
    """Convenience function to distill all eligible ptools in a program."""
    if output_path is None:
        output_path = source_path.replace(".py", "_distilled.py")
    return refactor_program(source_path, output_path, "distill", **kwargs)


def expand_program(
    source_path: str,
    output_path: Optional[str] = None,
    **kwargs,
) -> RefactorResult:
    """Convenience function to expand eligible functions in a program."""
    if output_path is None:
        output_path = source_path.replace(".py", "_expanded.py")
    return refactor_program(source_path, output_path, "expand", **kwargs)


# ============================================================================
# Interactive Refactoring
# ============================================================================

class InteractiveRefactorer:
    """
    Interactive refactoring session with user confirmation.

    Usage:
        ir = InteractiveRefactorer("program.py")
        ir.show_analysis()
        ir.preview_change("extract_food_items", mode="distill")
        ir.apply_change("extract_food_items")
        ir.save("program_v2.py")
    """

    def __init__(self, source_path: str):
        self.source_path = source_path
        self.refactorer = CodeRefactorer()
        self.pending_changes: Dict[str, RefactorChange] = {}
        self._analysis: Optional[Dict[str, Any]] = None

        with open(source_path) as f:
            self.original_code = f.read()
        self.current_code = self.original_code

    def show_analysis(self) -> Dict[str, Any]:
        """Show refactoring analysis."""
        if self._analysis is None:
            self._analysis = self.refactorer.analyze(self.source_path)
        return self._analysis

    def preview_change(
        self,
        function_name: str,
        mode: Literal["distill", "expand"],
    ) -> Optional[str]:
        """Preview what a refactoring change would look like."""
        functions = extract_functions(self.current_code)
        func = next((f for f in functions if f.name == function_name), None)

        if func is None:
            logger.error(f"Function {function_name} not found")
            return None

        if mode == "distill":
            if not func.is_ptool:
                logger.error(f"{function_name} is not a @ptool")
                return None
            result = self.refactorer.distiller.distill(function_name)
            if result.success:
                return result.distilled_code
            else:
                logger.error(f"Distillation failed: {result.error}")
                return None

        else:  # expand
            if not func.is_pure_python:
                logger.error(f"{function_name} is not pure Python")
                return None

            param_types = [(n, t or "Any") for n, t in func.parameters]
            ptool_code = generate_ptool_from_function(
                function_name=func.name,
                function_source=func.body_source,
                docstring=func.docstring,
                parameter_types=param_types,
                return_type=func.return_type or "Any",
            )
            return self.refactorer._extract_code(ptool_code)

    def stage_change(
        self,
        function_name: str,
        mode: Literal["distill", "expand"],
    ) -> bool:
        """Stage a change for later application."""
        new_source = self.preview_change(function_name, mode)
        if new_source is None:
            return False

        functions = extract_functions(self.current_code)
        func = next((f for f in functions if f.name == function_name), None)
        start_line, end_line = get_function_source_range(self.current_code, func)

        self.pending_changes[function_name] = RefactorChange(
            function_name=function_name,
            mode=mode,
            original_source=func.body_source,
            new_source=new_source,
            start_line=start_line,
            end_line=end_line,
            success=True,
        )
        return True

    def apply_staged_changes(self) -> bool:
        """Apply all staged changes."""
        if not self.pending_changes:
            logger.warning("No pending changes to apply")
            return False

        self.current_code = self.refactorer._apply_changes(
            self.current_code,
            list(self.pending_changes.values()),
        )

        # Verify compile
        result = self.refactorer._compile_test(self.current_code)
        if not result["success"]:
            logger.error(f"Compile error after applying changes: {result['error']}")
            return False

        self.pending_changes.clear()
        return True

    def save(self, output_path: str) -> bool:
        """Save the current code to a file."""
        try:
            with open(output_path, 'w') as f:
                f.write(self.current_code)
            logger.info(f"Saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return False

    def reset(self):
        """Reset to original code."""
        self.current_code = self.original_code
        self.pending_changes.clear()
