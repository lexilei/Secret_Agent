"""
L2 Chain-of-Code (CoC) for StrategyQA.

Based on the Chain-of-Code paper: LLM generates Python-style code mixing
executable operations with semantic (non-executable) function calls.

L1-CoC: LLM generates code + simulates trace in a single pass (no execution).
L2-CoC: LLM generates code -> Python actually executes it with ptool fallback.

The key insight: executable Python lines run via exec(), while semantic
function calls (knowledge lookups, reasoning) are routed to ptools (LLM calls).
This maps CoC's "LMulator" to the ptool framework.
"""

import ast
import re
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost
from .trace_builder import ParagraphRetriever, get_paragraph_retriever

from ptool_framework.llm_backend import call_llm


# ============================================================================
# Safe exec() sandbox
# ============================================================================

SAFE_BUILTINS = {
    'True': True, 'False': False, 'None': None,
    'abs': abs, 'min': min, 'max': max, 'len': len,
    'int': int, 'float': float, 'str': str, 'bool': bool,
    'round': round, 'sum': sum, 'sorted': sorted,
    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
    'range': range, 'enumerate': enumerate, 'zip': zip,
    'isinstance': isinstance, 'type': type,
    'print': lambda *a, **kw: None,  # no-op
}


# ============================================================================
# CoC Trace Data Structures
# ============================================================================

@dataclass
class CoCStep:
    """A single step in CoC execution."""
    line: str                       # The code line executed
    method: str                     # "python" or "ptool"
    explanation: str                # What happened
    state_delta: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoCTrace:
    """Complete CoC execution trace."""
    question: str
    generated_code: str
    steps: List[CoCStep] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "generated_code": self.generated_code,
            "steps": [s.to_dict() for s in self.steps],
            "final_state": {k: repr(v) for k, v in self.final_state.items()},
            "answer": self.answer,
            "success": self.success,
            "metadata": self.metadata,
        }

    def to_readable(self) -> str:
        """Format as human-readable trace (like the CoC paper)."""
        lines = [f'# Question: {self.question}', '', '# Generated Code:', self.generated_code, '', '# Execution Trace:']
        for i, step in enumerate(self.steps):
            lines.append(f'state: {step.state_delta}')
            lines.append(f'line: {step.line}')
            lines.append(f'explanation: {step.explanation}')
            lines.append(f'delta state: {step.state_delta}')
            lines.append('')
        lines.append(f'# Answer: {self.answer}')
        return '\n'.join(lines)


# ============================================================================
# Code Generator
# ============================================================================

class CoCCodeGenerator:
    """Generate Python-style code for StrategyQA questions."""

    PROMPT = """Write Python code to answer this yes/no question.

Rules:
- Use real Python syntax (variables, if/else, comparisons, math, boolean logic)
- For knowledge lookups, use descriptive function calls with ret_type hints
- Semantic functions will be executed by an AI, so give them clear names and arguments
- The final line MUST set: answer = "Yes" or answer = "No"
- Keep code concise (4-8 lines)
- Do NOT import anything or use print

Examples:

Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?
```python
question = "Could a llama birth twice during War in Vietnam (1945-46)?"
war_duration_months = get_duration("War in Vietnam 1945-46", unit="months", ret_type=int)
llama_gestation_months = get_gestation_period("llama", unit="months", ret_type=int)
can_birth_twice = war_duration_months >= llama_gestation_months * 2
answer = "Yes" if can_birth_twice else "No"
```

Q: Yes or no: Would a pear sink in water?
```python
question = "Would a pear sink in water?"
pear_density = get_density("pear", unit="g/cm3", ret_type=float)
water_density = 1.0
sinks = pear_density > water_density
answer = "Yes" if sinks else "No"
```

Q: Yes or no: Is it common to see frost during some college commencements?
```python
question = "Is it common to see frost during some college commencements?"
commencement_months = get_typical_months("college commencements", ret_type=str)
has_winter_month = check_condition(commencement_months, "includes a winter month (Nov-Feb)", ret_type=bool)
frost_in_winter = check_fact("Can frost occur during winter months?", ret_type=bool)
answer = "Yes" if has_winter_month and frost_in_winter else "No"
```

Q: Yes or no: Are more people today related to Genghis Khan than Julius Caesar?
```python
question = "Are more people today related to Genghis Khan than Julius Caesar?"
genghis_descendants = get_estimated_descendants("Genghis Khan", ret_type=int)
caesar_descendants = get_estimated_descendants("Julius Caesar", ret_type=int)
answer = "Yes" if genghis_descendants > caesar_descendants else "No"
```

Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?
```python
question = "Hydrogen's atomic number squared exceeds number of Spice Girls?"
hydrogen_atomic_number = get_atomic_number("Hydrogen", ret_type=int)
num_spice_girls = get_count("Spice Girls members", ret_type=int)
exceeds = hydrogen_atomic_number ** 2 > num_spice_girls
answer = "Yes" if exceeds else "No"
```

Q: Yes or no: {question}
```python
"""

    def __init__(self, model: str = "deepseek-v3-0324"):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate(self, question: str) -> str:
        """Generate Python-style code for the question."""
        prompt = self.PROMPT.format(question=question)
        response = call_llm(prompt=prompt, model=self.model)

        self.total_input_tokens += len(prompt) // 4
        self.total_output_tokens += len(response) // 4 if response else 0

        return self._extract_code(response)

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if not response:
            return ""

        # Try to find code between ```python and ```
        match = re.search(r'```python\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find code between ``` and ```
        match = re.search(r'```\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Take everything up to ``` or end
        code = response.split('```')[0].strip()

        # Remove any leading text before first assignment/variable
        lines = code.split('\n')
        code_lines = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not started:
                # Look for first line that looks like code
                if stripped and (
                    '=' in stripped
                    or stripped.startswith('if ')
                    or stripped.startswith('answer')
                    or stripped.startswith('#')
                    or stripped.startswith('question')
                ):
                    started = True
            if started:
                code_lines.append(line)

        return '\n'.join(code_lines).strip() if code_lines else code


# ============================================================================
# Hybrid Executor (Python + PTools)
# ============================================================================

class CoCHybridExecutor:
    """Execute CoC code line by line: Python where possible, ptools where needed."""

    def __init__(self, model: str = "deepseek-v3-0324", use_rag: bool = False):
        self.model = model
        self.use_rag = use_rag
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._retriever: Optional[ParagraphRetriever] = None

    @property
    def retriever(self) -> ParagraphRetriever:
        if self._retriever is None:
            self._retriever = get_paragraph_retriever()
            self._retriever.load()
        return self._retriever

    def execute(self, code: str, question: str) -> CoCTrace:
        """Execute generated code with hybrid Python/ptool dispatch."""
        trace = CoCTrace(question=question, generated_code=code)
        state: Dict[str, Any] = {}

        lines = self._prepare_lines(code)

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            prev_state = dict(state)

            try:
                # Try Python execution first
                exec(line, {"__builtins__": SAFE_BUILTINS}, state)
                delta = self._compute_delta(prev_state, state)
                trace.steps.append(CoCStep(
                    line=stripped,
                    method="python",
                    explanation="Python execution.",
                    state_delta=delta,
                ))
            except NameError as e:
                # Semantic function - route to ptool
                step = self._handle_semantic_call(stripped, state, question)
                trace.steps.append(step)
                if not step.success:
                    trace.success = False
            except SyntaxError:
                # Might be a multi-line construct - try to handle
                step = self._handle_syntax_error(stripped, state, question)
                trace.steps.append(step)
            except Exception as e:
                trace.steps.append(CoCStep(
                    line=stripped,
                    method="python",
                    explanation=f"Error: {e}",
                    success=False,
                    error=str(e),
                ))

        # Extract final answer
        trace.final_state = {k: v for k, v in state.items() if not k.startswith('_')}
        trace.answer = state.get("answer")
        trace.metadata = {
            "model": self.model,
            "num_python_steps": sum(1 for s in trace.steps if s.method == "python"),
            "num_ptool_steps": sum(1 for s in trace.steps if s.method == "ptool"),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }

        return trace

    def _prepare_lines(self, code: str) -> List[str]:
        """Split code into executable lines, handling multi-line constructs."""
        lines = code.split('\n')
        prepared = []
        buffer = ""

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                if buffer:
                    prepared.append(buffer)
                    buffer = ""
                continue

            # Handle if/else blocks - combine with following lines
            if buffer:
                # Check if this is a continuation (indented or else/elif)
                if line.startswith((' ', '\t')) or stripped.startswith(('else:', 'elif ')):
                    buffer += '\n' + line
                    continue
                else:
                    prepared.append(buffer)
                    buffer = ""

            if stripped.startswith(('if ', 'for ', 'while ')):
                buffer = line
            else:
                prepared.append(line)

        if buffer:
            prepared.append(buffer)

        return prepared

    def _compute_delta(self, prev: Dict, curr: Dict) -> Dict[str, Any]:
        """Compute state changes between prev and curr."""
        delta = {}
        for k, v in curr.items():
            if k.startswith('_'):
                continue
            if k not in prev or prev[k] != v:
                # Serialize the value for the trace
                try:
                    json.dumps(v)
                    delta[k] = v
                except (TypeError, ValueError):
                    delta[k] = repr(v)
        return delta

    def _handle_semantic_call(self, line: str, state: Dict, question: str) -> CoCStep:
        """Handle a line containing a semantic function call via ptool."""
        try:
            # Parse the assignment and function call
            var_name, func_name, args, kwargs = self._parse_function_call(line)

            # Extract ret_type hint
            ret_type = kwargs.pop('ret_type', 'str')
            if isinstance(ret_type, type):
                ret_type = ret_type.__name__

            # Resolve any variable references in args
            resolved_args = []
            for arg in args:
                if isinstance(arg, str) and arg in state:
                    resolved_args.append(state[arg])
                else:
                    resolved_args.append(arg)

            resolved_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, str) and v in state:
                    resolved_kwargs[k] = state[v]
                else:
                    resolved_kwargs[k] = v

            # Build ptool prompt
            result = self._call_ptool(func_name, resolved_args, resolved_kwargs, ret_type, question)

            # Assign to state
            if var_name:
                state[var_name] = result

            return CoCStep(
                line=line,
                method="ptool",
                explanation=f"LLM executed {func_name}() -> {repr(result)}",
                state_delta={var_name: result} if var_name else {},
            )
        except Exception as e:
            return CoCStep(
                line=line,
                method="ptool",
                explanation=f"Failed to parse/execute semantic call: {e}",
                success=False,
                error=str(e),
            )

    def _handle_syntax_error(self, line: str, state: Dict, question: str) -> CoCStep:
        """Handle lines that cause SyntaxError - try as ptool."""
        # Might be a semantic call that also has syntax issues
        try:
            return self._handle_semantic_call(line, state, question)
        except Exception:
            return CoCStep(
                line=line,
                method="python",
                explanation=f"Syntax error, could not execute.",
                success=False,
                error="SyntaxError",
            )

    def _parse_function_call(self, line: str):
        """
        Parse a line like: var_name = func_name(arg1, arg2, kwarg=val)
        Returns: (var_name, func_name, positional_args, keyword_args)
        """
        # Try AST parsing first
        try:
            tree = ast.parse(line.strip())
            if tree.body and isinstance(tree.body[0], ast.Assign):
                assign = tree.body[0]
                var_name = assign.targets[0].id if isinstance(assign.targets[0], ast.Name) else None
                if isinstance(assign.value, ast.Call):
                    call = assign.value
                    func_name = self._get_func_name(call.func)
                    args = [self._eval_ast_node(a) for a in call.args]
                    kwargs = {kw.arg: self._eval_ast_node(kw.value) for kw in call.keywords}
                    return var_name, func_name, args, kwargs
        except (SyntaxError, AttributeError):
            pass

        # Fallback: regex parsing
        match = re.match(r'(\w+)\s*=\s*(\w+)\s*\((.*)\)', line.strip(), re.DOTALL)
        if match:
            var_name = match.group(1)
            func_name = match.group(2)
            args_str = match.group(3)
            args, kwargs = self._parse_args_string(args_str)
            return var_name, func_name, args, kwargs

        raise ValueError(f"Cannot parse function call from: {line}")

    def _get_func_name(self, node) -> str:
        """Extract function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._get_func_name(node.value)}.{node.attr}"
        return "unknown"

    def _eval_ast_node(self, node) -> Any:
        """Safely evaluate an AST node to get a Python value."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            # Return the name as a string reference (will be resolved later)
            builtin_types = {'int': int, 'float': float, 'str': str, 'bool': bool}
            if node.id in builtin_types:
                return builtin_types[node.id]
            return node.id
        if isinstance(node, ast.List):
            return [self._eval_ast_node(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {self._eval_ast_node(k): self._eval_ast_node(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval_ast_node(node.operand)
        return repr(ast.dump(node))

    def _parse_args_string(self, args_str: str):
        """Parse a comma-separated args string into positional and keyword args."""
        args = []
        kwargs = {}

        if not args_str.strip():
            return args, kwargs

        # Simple split on commas (not inside quotes/parens)
        parts = self._smart_split(args_str)

        for part in parts:
            part = part.strip()
            if '=' in part and not part.startswith('"') and not part.startswith("'"):
                key, _, val = part.partition('=')
                key = key.strip()
                val = val.strip()
                kwargs[key] = self._parse_value(val)
            else:
                args.append(self._parse_value(part))

        return args, kwargs

    def _smart_split(self, s: str) -> List[str]:
        """Split on commas, respecting quotes and parentheses."""
        parts = []
        current = ""
        depth = 0
        in_str = None

        for ch in s:
            if ch in ('"', "'") and in_str is None:
                in_str = ch
            elif ch == in_str:
                in_str = None
            elif in_str is None:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == ',' and depth == 0:
                    parts.append(current)
                    current = ""
                    continue
            current += ch

        if current.strip():
            parts.append(current)

        return parts

    def _parse_value(self, val: str) -> Any:
        """Parse a string value into a Python value."""
        val = val.strip()
        # String literal
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            return val[1:-1]
        # Boolean
        if val == 'True':
            return True
        if val == 'False':
            return False
        if val == 'None':
            return None
        # Type references
        if val in ('int', 'float', 'str', 'bool'):
            return val
        # Number
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        # Return as string (variable reference)
        return val

    def _call_ptool(
        self,
        func_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        ret_type: str,
        question: str,
    ) -> Any:
        """Call LLM to execute a semantic function (ptool)."""
        # Format arguments
        args_parts = [repr(a) for a in args]
        args_parts += [f'{k}={repr(v)}' for k, v in kwargs.items()]
        formatted_call = f"{func_name}({', '.join(args_parts)})"

        # Build context from RAG if enabled
        rag_context = ""
        if self.use_rag:
            query = f"{func_name} {' '.join(str(a) for a in args)}"
            try:
                paragraphs = self.retriever.retrieve(query, top_k=3)
                if paragraphs:
                    rag_parts = [f"[{p['title']}]: {p['content']}" for p in paragraphs]
                    rag_context = f"\n\nRelevant knowledge:\n" + "\n".join(rag_parts)
            except Exception:
                pass

        # Build prompt
        type_instruction = {
            'int': "Return only an integer number.",
            'float': "Return only a decimal number.",
            'bool': "Return only True or False.",
            'str': "Return only a brief text answer.",
        }.get(str(ret_type), "Return only the value.")

        prompt = f"""Execute this function and return only the result.

Function call: {formatted_call}
Context: Answering the question "{question}"{rag_context}

{type_instruction}

Result:"""

        response = call_llm(prompt=prompt, model=self.model)
        self.total_input_tokens += len(prompt) // 4
        self.total_output_tokens += len(response) // 4 if response else 0

        return self._parse_result(response, ret_type)

    def _parse_result(self, response: str, ret_type: str) -> Any:
        """Parse LLM response according to expected return type."""
        if not response:
            return None

        text = response.strip()

        if str(ret_type) in ('int', "<class 'int'>"):
            # Extract first integer
            match = re.search(r'-?\d[\d,]*', text)
            if match:
                return int(match.group().replace(',', ''))
            return 0

        if str(ret_type) in ('float', "<class 'float'>"):
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                return float(match.group())
            return 0.0

        if str(ret_type) in ('bool', "<class 'bool'>"):
            lower = text.lower()
            if 'true' in lower or 'yes' in lower:
                return True
            return False

        # Default: return as string
        # Clean up common LLM artifacts
        text = re.sub(r'^(Result|Answer|Output)\s*[:=]\s*', '', text, flags=re.IGNORECASE)
        text = text.strip().strip('"').strip("'")
        return text


# ============================================================================
# L2-CoC Experiment
# ============================================================================

class L2_CoC(StrategyQAExperiment):
    """
    L2-CoC: Chain-of-Code with hybrid Python/ptool execution.

    Workflow:
    1. LLM generates Python-style code (mix of executable + semantic functions)
    2. Python executes code line by line:
       - Executable lines → exec() (real Python)
       - Semantic functions → ptool calls (LLM execution)
    3. Extract final answer from execution state

    This is the L2 upgrade of L1-CoC: actual execution replaces LLM simulation.
    The CoC "LMulator" maps to the ptool executor.
    """

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)
        self.code_generator = CoCCodeGenerator(model)
        self.executor = CoCHybridExecutor(model, use_rag=False)
        self.traces: List[CoCTrace] = []

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        start_time = time.time()

        # Reset token counts
        self.code_generator.total_input_tokens = 0
        self.code_generator.total_output_tokens = 0
        self.executor.total_input_tokens = 0
        self.executor.total_output_tokens = 0

        try:
            # Step 1: Generate code
            code = self.code_generator.generate(instance.question)

            if not code:
                raise ValueError("Code generation returned empty result")

            # Step 2: Execute with hybrid dispatch
            trace = self.executor.execute(code, instance.question)
            self.traces.append(trace)

            latency_ms = (time.time() - start_time) * 1000

            # Sum tokens from both phases
            input_tokens = (
                self.code_generator.total_input_tokens
                + self.executor.total_input_tokens
            )
            output_tokens = (
                self.code_generator.total_output_tokens
                + self.executor.total_output_tokens
            )

            cost = calculate_cost(input_tokens, output_tokens, self.model)

            # Extract boolean answer
            predicted = self._extract_answer(trace.answer)
            is_correct = predicted == instance.answer if predicted is not None else False

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=is_correct,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost.cost_usd,
                num_steps=len(trace.steps),
                decomposition_used=False,
                raw_response=trace.to_readable(),
                trace=trace.to_dict(),
            )

        except Exception as e:
            return self._error_result(
                instance, str(e), type(e).__name__, time.time() - start_time
            )

    def _extract_answer(self, answer_val: Any) -> Optional[bool]:
        """Convert answer value to boolean."""
        if answer_val is None:
            return None
        if isinstance(answer_val, bool):
            return answer_val
        if isinstance(answer_val, str):
            lower = answer_val.strip().lower()
            if lower in ('yes', 'true'):
                return True
            if lower in ('no', 'false'):
                return False
            return self.extract_boolean(answer_val)
        return None

    def get_summary(self) -> Dict[str, Any]:
        summary = super().get_summary()

        if self.traces:
            total_steps = sum(len(t.steps) for t in self.traces)
            python_steps = sum(
                sum(1 for s in t.steps if s.method == "python") for t in self.traces
            )
            ptool_steps = sum(
                sum(1 for s in t.steps if s.method == "ptool") for t in self.traces
            )
            successful = sum(1 for t in self.traces if t.success)

            summary["coc_stats"] = {
                "total_traces": len(self.traces),
                "successful_traces": successful,
                "avg_steps_per_trace": total_steps / len(self.traces),
                "total_python_steps": python_steps,
                "total_ptool_steps": ptool_steps,
                "python_ratio": python_steps / total_steps if total_steps > 0 else 0,
            }

        return summary


# ============================================================================
# L2-CoC with RAG
# ============================================================================

class L2_CoCRAG(L2_CoC):
    """
    L2-CoC-RAG: Chain-of-Code with retrieval-augmented ptool execution.

    Same as L2_CoC but retrieves relevant Wikipedia paragraphs before
    each ptool call to provide grounded context for semantic functions.
    """

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)
        self.executor = CoCHybridExecutor(model, use_rag=True)
