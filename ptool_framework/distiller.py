"""
Behavior Distiller: Converts LLM-executed ptools into pure Python code.

This is William's "behavior distillation" - analyzing execution traces to
identify patterns that can be implemented in deterministic Python code.

The process:
1. Load traces for a ptool
2. Cluster inputs by similarity (LLM-assisted)
3. Identify deterministic patterns (LLM-assisted)
4. Generate Python code that handles those patterns (LLM-assisted)
5. Compile-test the generated code
6. Validate against held-out traces
7. Wrap in @distilled decorator with fallback

This is an LLM-heavy module - it uses LLMs to analyze patterns and generate code.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .ptool import ptool, PToolSpec, get_registry
from .trace_store import TraceStore, ExecutionTrace, get_trace_store
from .llm_backend import call_llm


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PatternCluster:
    """A cluster of similar input/output patterns."""
    cluster_id: str
    pattern_description: str
    example_inputs: List[Dict[str, Any]]
    example_outputs: List[Any]
    coverage_count: int  # How many traces this pattern covers
    confidence: float  # How confident we are in this pattern


@dataclass
class DistillationAnalysis:
    """Analysis of a ptool's distillation potential."""
    ptool_name: str
    total_traces: int
    success_rate: float
    clusters: List[PatternCluster]
    estimated_coverage: float  # What % of cases can Python handle
    recommendation: Literal["distillable", "partial", "not_ready"]
    notes: str


@dataclass
class GeneratedImplementation:
    """A generated Python implementation."""
    ptool_name: str
    python_code: str
    imports: List[str]
    patterns_covered: List[str]
    validation_results: Dict[str, Any]
    compile_success: bool
    test_success_rate: float


@dataclass
class DistillationResult:
    """Result of the distillation process."""
    ptool_name: str
    success: bool
    implementation: Optional[GeneratedImplementation]
    distilled_code: Optional[str]  # Full @distilled function code
    error: Optional[str]
    iterations: int  # How many LLM iterations needed


# ============================================================================
# LLM-Powered Analysis ptools
# ============================================================================

@ptool(model="deepseek-v3-0324", output_mode="structured")
def analyze_trace_patterns(
    ptool_name: str,
    ptool_signature: str,
    ptool_docstring: str,
    sample_traces: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze execution traces to identify patterns for distillation.

    Given a ptool's signature, docstring, and sample traces (input/output pairs),
    identify patterns that could be implemented in pure Python.

    For each pattern found:
    - Describe what the pattern matches
    - Explain the transformation logic
    - Estimate how many traces match this pattern

    Return:
    {
        "patterns": [
            {
                "id": "pattern_1",
                "description": "When input contains food words like pizza/salad, extract them",
                "logic": "regex match for known food words",
                "example_inputs": [...],
                "example_outputs": [...],
                "estimated_coverage": 0.7
            }
        ],
        "uncovered_cases": "Cases with unusual phrasing or new foods",
        "overall_distillability": 0.85,
        "recommendation": "distillable"
    }
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="freeform")
def generate_python_implementation(
    ptool_name: str,
    ptool_signature: str,
    ptool_docstring: str,
    patterns: List[Dict[str, Any]],
    sample_traces: List[Dict[str, Any]],
) -> str:
    """Generate pure Python code that implements the identified patterns.

    Given the ptool specification and identified patterns, generate Python code
    that handles as many cases as possible. The code should:

    1. Be pure Python (no LLM calls)
    2. Handle all identified patterns
    3. Raise DistillationFallback for cases it can't handle
    4. Be well-documented with comments
    5. Include necessary imports

    Format your response as valid Python code. Start with imports, then the function.

    ANSWER:
    ```python
    import re
    from typing import List
    from ptool_framework import DistillationFallback

    def {ptool_name}_impl(...) -> ...:
        '''Distilled implementation. Falls back to LLM for unknown patterns.'''
        # Implementation here
        ...
    ```
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="freeform")
def fix_code_error(
    code: str,
    error_message: str,
    error_type: str,
) -> str:
    """Fix a Python code error.

    Given code that failed to compile or run, and the error message,
    generate corrected code.

    Return ONLY the corrected Python code, no explanations.

    ANSWER:
    ```python
    <corrected code>
    ```
    """
    ...


@ptool(model="deepseek-v3-0324", output_mode="structured")
def validate_implementation(
    ptool_name: str,
    implementation_code: str,
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze whether an implementation correctly handles test cases.

    For each test case, determine if the implementation would produce
    the expected output.

    Return:
    {
        "analysis": [
            {
                "input": {...},
                "expected_output": ...,
                "would_handle": true/false,
                "reason": "..."
            }
        ],
        "estimated_success_rate": 0.85,
        "failure_patterns": ["unusual input format", "..."]
    }
    """
    ...


# ============================================================================
# Behavior Distiller
# ============================================================================

class BehaviorDistiller:
    """
    Distills LLM-executed ptools into pure Python implementations.

    Usage:
        distiller = BehaviorDistiller()

        # Analyze a ptool
        analysis = distiller.analyze("extract_food_items")

        # If distillable, generate implementation
        if analysis.recommendation == "distillable":
            result = distiller.distill("extract_food_items")
            if result.success:
                print(result.distilled_code)
    """

    def __init__(
        self,
        trace_store: Optional[TraceStore] = None,
        llm_model: str = "deepseek-v3-0324",
        max_iterations: int = 3,
        min_traces: int = 10,
        min_success_rate: float = 0.8,
        validation_split: float = 0.2,
    ):
        """
        Initialize the distiller.

        Args:
            trace_store: TraceStore to load traces from
            llm_model: LLM model for code generation
            max_iterations: Max attempts to fix code errors
            min_traces: Minimum traces needed for distillation
            min_success_rate: Minimum success rate for distillation
            validation_split: Fraction of traces to hold out for validation
        """
        self.trace_store = trace_store or get_trace_store()
        self.llm_model = llm_model
        self.max_iterations = max_iterations
        self.min_traces = min_traces
        self.min_success_rate = min_success_rate
        self.validation_split = validation_split

    def analyze(self, ptool_name: str) -> DistillationAnalysis:
        """
        Analyze a ptool's traces to determine distillation potential.

        Args:
            ptool_name: Name of the ptool to analyze

        Returns:
            DistillationAnalysis with patterns and recommendations
        """
        logger.info(f"Analyzing {ptool_name} for distillation...")

        # Get ptool spec
        registry = get_registry()
        spec = registry.get(ptool_name)
        if spec is None:
            return DistillationAnalysis(
                ptool_name=ptool_name,
                total_traces=0,
                success_rate=0,
                clusters=[],
                estimated_coverage=0,
                recommendation="not_ready",
                notes=f"ptool '{ptool_name}' not found in registry",
            )

        # Get traces
        traces = self.trace_store.get_traces(ptool_name=ptool_name, limit=1000)
        if len(traces) < self.min_traces:
            return DistillationAnalysis(
                ptool_name=ptool_name,
                total_traces=len(traces),
                success_rate=0,
                clusters=[],
                estimated_coverage=0,
                recommendation="not_ready",
                notes=f"Need at least {self.min_traces} traces, have {len(traces)}",
            )

        # Calculate success rate
        successful = [t for t in traces if t.success]
        success_rate = len(successful) / len(traces)

        if success_rate < self.min_success_rate:
            return DistillationAnalysis(
                ptool_name=ptool_name,
                total_traces=len(traces),
                success_rate=success_rate,
                clusters=[],
                estimated_coverage=0,
                recommendation="not_ready",
                notes=f"Success rate {success_rate:.1%} below threshold {self.min_success_rate:.1%}",
            )

        # Prepare sample traces for LLM analysis
        sample_traces = [
            {"inputs": t.inputs, "output": t.output}
            for t in successful[:50]  # Sample up to 50
        ]

        # Use LLM to analyze patterns
        logger.info("Calling LLM to analyze patterns...")
        analysis_result = analyze_trace_patterns(
            ptool_name=ptool_name,
            ptool_signature=spec.get_signature_str(),
            ptool_docstring=spec.docstring,
            sample_traces=sample_traces,
        )

        # Convert to PatternCluster objects
        clusters = []
        for p in analysis_result.get("patterns", []):
            clusters.append(PatternCluster(
                cluster_id=p.get("id", f"pattern_{len(clusters)}"),
                pattern_description=p.get("description", ""),
                example_inputs=p.get("example_inputs", []),
                example_outputs=p.get("example_outputs", []),
                coverage_count=int(p.get("estimated_coverage", 0) * len(successful)),
                confidence=p.get("estimated_coverage", 0),
            ))

        estimated_coverage = analysis_result.get("overall_distillability", 0)
        recommendation = analysis_result.get("recommendation", "partial")

        # Map recommendation
        if recommendation == "distillable" or estimated_coverage >= 0.9:
            rec = "distillable"
        elif estimated_coverage >= 0.7:
            rec = "partial"
        else:
            rec = "not_ready"

        return DistillationAnalysis(
            ptool_name=ptool_name,
            total_traces=len(traces),
            success_rate=success_rate,
            clusters=clusters,
            estimated_coverage=estimated_coverage,
            recommendation=rec,
            notes=analysis_result.get("uncovered_cases", ""),
        )

    def distill(self, ptool_name: str) -> DistillationResult:
        """
        Distill a ptool into pure Python code.

        This is the main distillation process:
        1. Analyze traces for patterns
        2. Generate Python implementation
        3. Compile-test the code
        4. Validate against held-out traces
        5. Iterate if needed

        Args:
            ptool_name: Name of the ptool to distill

        Returns:
            DistillationResult with the generated code
        """
        logger.info(f"Starting distillation for {ptool_name}...")

        # First, analyze
        analysis = self.analyze(ptool_name)
        if analysis.recommendation == "not_ready":
            return DistillationResult(
                ptool_name=ptool_name,
                success=False,
                implementation=None,
                distilled_code=None,
                error=analysis.notes,
                iterations=0,
            )

        # Get ptool spec
        registry = get_registry()
        spec = registry.get(ptool_name)

        # Get traces and split into train/validation
        all_traces = self.trace_store.get_traces(
            ptool_name=ptool_name, success_only=True, limit=1000
        )
        split_idx = int(len(all_traces) * (1 - self.validation_split))
        train_traces = all_traces[:split_idx]
        val_traces = all_traces[split_idx:]

        # Prepare data for code generation
        patterns = [
            {
                "id": c.cluster_id,
                "description": c.pattern_description,
                "example_inputs": c.example_inputs,
                "example_outputs": c.example_outputs,
            }
            for c in analysis.clusters
        ]

        sample_traces = [
            {"inputs": t.inputs, "output": t.output}
            for t in train_traces[:30]
        ]

        # Iterative code generation with compile testing
        implementation = None
        iterations = 0
        last_error = None

        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            logger.info(f"Distillation iteration {iterations}/{self.max_iterations}")

            if iteration == 0:
                # First attempt: generate from patterns
                logger.info("Generating Python implementation...")
                code_response = generate_python_implementation(
                    ptool_name=ptool_name,
                    ptool_signature=spec.get_signature_str(),
                    ptool_docstring=spec.docstring,
                    patterns=patterns,
                    sample_traces=sample_traces,
                )
            else:
                # Subsequent attempts: fix errors
                logger.info(f"Fixing error: {last_error[:100]}...")
                code_response = fix_code_error(
                    code=python_code,
                    error_message=last_error,
                    error_type=error_type,
                )

            # Extract code from response
            python_code = self._extract_code(code_response)
            if not python_code:
                last_error = "Could not extract Python code from LLM response"
                error_type = "extraction"
                continue

            # Compile test
            compile_result = self._compile_test(python_code)
            if not compile_result["success"]:
                last_error = compile_result["error"]
                error_type = "compile"
                logger.warning(f"Compile error: {last_error}")
                continue

            # Runtime test with sample inputs
            runtime_result = self._runtime_test(python_code, ptool_name, train_traces[:5])
            if not runtime_result["success"]:
                last_error = runtime_result["error"]
                error_type = "runtime"
                logger.warning(f"Runtime error: {last_error}")
                continue

            # Validation against held-out traces
            val_result = self._validate_implementation(
                python_code, ptool_name, val_traces
            )

            implementation = GeneratedImplementation(
                ptool_name=ptool_name,
                python_code=python_code,
                imports=self._extract_imports(python_code),
                patterns_covered=[c.cluster_id for c in analysis.clusters],
                validation_results=val_result,
                compile_success=True,
                test_success_rate=val_result.get("success_rate", 0),
            )

            logger.info(f"Validation success rate: {val_result.get('success_rate', 0):.1%}")

            # If validation passes, we're done
            if val_result.get("success_rate", 0) >= 0.8:
                break

            # Otherwise, try to improve
            last_error = f"Low validation success rate: {val_result.get('success_rate', 0):.1%}"
            error_type = "validation"

        if implementation is None:
            return DistillationResult(
                ptool_name=ptool_name,
                success=False,
                implementation=None,
                distilled_code=None,
                error=last_error,
                iterations=iterations,
            )

        # Generate final @distilled wrapper
        distilled_code = self._generate_distilled_wrapper(
            ptool_name=ptool_name,
            implementation=implementation,
            analysis=analysis,
        )

        return DistillationResult(
            ptool_name=ptool_name,
            success=True,
            implementation=implementation,
            distilled_code=distilled_code,
            error=None,
            iterations=iterations,
        )

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
        if response.strip().startswith(("import ", "from ", "def ")):
            return response.strip()

        return None

    def _compile_test(self, code: str) -> Dict[str, Any]:
        """Test if code compiles successfully."""
        try:
            ast.parse(code)
            return {"success": True, "error": None}
        except SyntaxError as e:
            return {"success": False, "error": f"SyntaxError: {e}"}

    def _runtime_test(
        self,
        code: str,
        ptool_name: str,
        test_traces: List[ExecutionTrace],
    ) -> Dict[str, Any]:
        """Test if code runs without errors on sample inputs."""
        # Create a temporary module with the code
        test_code = f"""
{code}

# Test harness
import json
import sys

test_cases = json.loads(sys.argv[1])
func_name = sys.argv[2]

# Find the implementation function
impl_func = None
for name in dir():
    if name.startswith(func_name) or name.endswith('_impl'):
        obj = eval(name)
        if callable(obj) and not name.startswith('_'):
            impl_func = obj
            break

if impl_func is None:
    print("ERROR: Could not find implementation function")
    sys.exit(1)

errors = []
for i, tc in enumerate(test_cases):
    try:
        result = impl_func(**tc['inputs'])
    except Exception as e:
        if 'DistillationFallback' not in str(type(e)):
            errors.append(f"Case {{i}}: {{type(e).__name__}}: {{e}}")

if errors:
    print("ERRORS:\\n" + "\\n".join(errors))
    sys.exit(1)
else:
    print("OK")
"""

        # Prepare test cases
        test_cases = [
            {"inputs": t.inputs, "output": t.output}
            for t in test_traces
        ]

        # Write to temp file and run
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path, json.dumps(test_cases), ptool_name],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return {"success": True, "error": None}
            else:
                error = result.stderr or result.stdout
                return {"success": False, "error": error[:500]}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _validate_implementation(
        self,
        code: str,
        ptool_name: str,
        val_traces: List[ExecutionTrace],
    ) -> Dict[str, Any]:
        """Validate implementation against held-out traces."""
        if not val_traces:
            return {"success_rate": 1.0, "tested": 0, "passed": 0, "failed": 0}

        # Run implementation on validation traces
        test_code = f"""
{code}

import json
import sys

test_cases = json.loads(sys.argv[1])
func_name = sys.argv[2]

# Find the implementation function
impl_func = None
for name in dir():
    if name.startswith(func_name) or name.endswith('_impl'):
        obj = eval(name)
        if callable(obj) and not name.startswith('_'):
            impl_func = obj
            break

if impl_func is None:
    print(json.dumps({{"error": "Could not find function"}}))
    sys.exit(0)

results = []
for tc in test_cases:
    try:
        result = impl_func(**tc['inputs'])
        # Check if result matches expected
        matches = result == tc['expected']
        results.append({{"passed": matches, "fallback": False}})
    except Exception as e:
        if 'DistillationFallback' in str(type(e)):
            results.append({{"passed": True, "fallback": True}})  # Fallback is OK
        else:
            results.append({{"passed": False, "fallback": False, "error": str(e)}})

print(json.dumps(results))
"""

        test_cases = [
            {"inputs": t.inputs, "expected": t.output}
            for t in val_traces
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path, json.dumps(test_cases), ptool_name],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return {"success_rate": 0, "error": result.stderr}

            results = json.loads(result.stdout)
            passed = sum(1 for r in results if r.get("passed"))
            fallbacks = sum(1 for r in results if r.get("fallback"))

            return {
                "success_rate": passed / len(results) if results else 0,
                "tested": len(results),
                "passed": passed,
                "fallbacks": fallbacks,
                "failed": len(results) - passed,
            }

        except Exception as e:
            return {"success_rate": 0, "error": str(e)}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports

    def _generate_distilled_wrapper(
        self,
        ptool_name: str,
        implementation: GeneratedImplementation,
        analysis: DistillationAnalysis,
    ) -> str:
        """Generate the final @distilled decorated function."""
        registry = get_registry()
        spec = registry.get(ptool_name)

        # Get function signature
        params = ", ".join(
            f"{name}: {spec._type_to_str(typ)}"
            for name, typ in spec.parameters.items()
        )
        return_type = spec._type_to_str(spec.return_type)

        # Generate the wrapper
        code = f'''"""
Distilled implementation of {ptool_name}

Generated: {datetime.now().isoformat()}
Traces analyzed: {analysis.total_traces}
Estimated coverage: {analysis.estimated_coverage:.1%}
Validation success rate: {implementation.test_success_rate:.1%}
"""

{chr(10).join(implementation.imports)}
from ptool_framework import distilled, DistillationFallback

# Original implementation generated by distillation
{implementation.python_code}

# Wrapped with @distilled for auto-fallback
@distilled(fallback_ptool="{ptool_name}")
def {ptool_name}_distilled({params}) -> {return_type}:
    """Distilled implementation with LLM fallback.

    Handles {analysis.estimated_coverage:.0%} of cases in pure Python.
    Falls back to LLM for: {analysis.notes or "edge cases"}
    """
    return {ptool_name}_impl({", ".join(spec.parameters.keys())})
'''
        return code


# ============================================================================
# Convenience Functions
# ============================================================================

def distill_ptool(ptool_name: str, **kwargs) -> DistillationResult:
    """Convenience function to distill a single ptool."""
    distiller = BehaviorDistiller(**kwargs)
    return distiller.distill(ptool_name)


def analyze_ptool(ptool_name: str, **kwargs) -> DistillationAnalysis:
    """Convenience function to analyze a ptool's distillation potential."""
    distiller = BehaviorDistiller(**kwargs)
    return distiller.analyze(ptool_name)


def distill_all_candidates(min_traces: int = 10, **kwargs) -> List[DistillationResult]:
    """Distill all ptools that are candidates for distillation."""
    trace_store = kwargs.get("trace_store") or get_trace_store()
    candidates = trace_store.get_distillation_candidates(min_traces)

    results = []
    for ptool_name in candidates:
        logger.info(f"Distilling {ptool_name}...")
        result = distill_ptool(ptool_name, **kwargs)
        results.append(result)

        if result.success:
            logger.info(f"  Success! {result.iterations} iterations")
        else:
            logger.warning(f"  Failed: {result.error}")

    return results
