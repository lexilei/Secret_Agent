"""
L2 Program Generator for StrategyQA.

Uses ptool_framework's ProgramGenerator to dynamically generate a
task-specific program for each StrategyQA question. The generated program
contains @ptool definitions and a workflow function, which are then
executed to produce an answer.

The key idea: instead of a fixed decompose→answer→combine trace,
let the ProgramGenerator analyze each question and design a custom
workflow with custom ptools tailored to that specific question.

Each run produces:
- The generated program source (for inspection)
- A workflow trace recording each step
- The final yes/no answer

All traces are saved to a JSON file for later analysis / ICL / distillation.
"""

import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost

from ptool_framework.program_generator import ProgramGenerator, GeneratedProgram
from ptool_framework.llm_backend import call_llm


# ============================================================================
# Trace Data Structures
# ============================================================================

@dataclass
class ProgramGenStep:
    """A single step in the program generation + execution trace."""
    phase: str                          # "analyze", "design_ptools", "generate_workflow", "execute"
    description: str
    input_summary: str
    output_summary: str
    raw_output: Any = None
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProgramGenTrace:
    """Complete trace for a single StrategyQA instance processed by ProgramGenerator."""
    qid: str
    question: str
    generated_program: Optional[str] = None     # Full generated source code
    ptools_designed: List[Dict[str, Any]] = field(default_factory=list)
    workflow_code: Optional[str] = None
    steps: List[ProgramGenStep] = field(default_factory=list)
    final_answer: Optional[bool] = None
    execution_response: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "generated_program": self.generated_program,
            "ptools_designed": self.ptools_designed,
            "workflow_code": self.workflow_code,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "execution_response": self.execution_response,
            "success": self.success,
            "metadata": self.metadata,
        }


# ============================================================================
# L2 Program Generator Experiment
# ============================================================================

class L2_ProgramGen(StrategyQAExperiment):
    """
    L2-ProgramGen: Use ProgramGenerator to create question-specific workflows.

    For each StrategyQA question:
    1. ProgramGenerator.analyze_task() identifies what reasoning is needed
    2. ProgramGenerator.design_ptools() creates custom @ptool definitions
    3. ProgramGenerator.generate_workflow() writes the orchestration code
    4. The generated workflow description is sent to the LLM for execution

    This differs from L2-TraceBuilder (fixed decompose→answer→combine) by
    letting the LLM design a custom workflow per question.

    All traces (generated programs + execution results) are collected and
    can be exported for analysis, ICL, or distillation.
    """

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)
        self.generator: Optional[ProgramGenerator] = None
        self.traces: List[ProgramGenTrace] = []

    def setup(self) -> None:
        self.generator = ProgramGenerator(default_model=self.model)
        self._setup_complete = True

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        start_time = time.time()
        trace = ProgramGenTrace(qid=instance.qid, question=instance.question)
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            # Phase 1: Generate a task-specific program
            task_description = (
                f"Answer this yes/no question by reasoning step-by-step: "
                f"{instance.question}\n\n"
                f"The program should decompose the question, gather relevant facts, "
                f"and combine them to produce a final yes or no answer."
            )

            gen_start = time.time()
            program = self.generator.generate(task_description=task_description)
            gen_latency = (time.time() - gen_start) * 1000

            # Record the generated program
            trace.generated_program = program.full_code
            trace.workflow_code = program.workflow_code
            trace.ptools_designed = [
                {
                    "name": pt.name,
                    "parameters": pt.parameters,
                    "return_type": pt.return_type,
                    "docstring": pt.docstring,
                    "output_mode": pt.output_mode,
                }
                for pt in program.ptools
            ]

            # Estimate tokens used during generation (3 LLM calls: analyze, design, workflow)
            gen_input_est = len(task_description) // 4 * 3  # rough: 3 calls
            gen_output_est = len(program.full_code) // 4
            total_input_tokens += gen_input_est
            total_output_tokens += gen_output_est

            trace.steps.append(ProgramGenStep(
                phase="generate",
                description="Generated task-specific program with ProgramGenerator",
                input_summary=task_description[:200],
                output_summary=f"{len(program.ptools)} ptools designed, workflow generated",
                raw_output={
                    "num_ptools": len(program.ptools),
                    "ptool_names": [pt.name for pt in program.ptools],
                    "workflow_length": len(program.workflow_code) if program.workflow_code else 0,
                },
                latency_ms=gen_latency,
            ))

            # Phase 2: Execute the generated workflow by asking the LLM to follow it
            exec_start = time.time()

            execution_prompt = self._build_execution_prompt(
                instance.question, program
            )

            response = call_llm(prompt=execution_prompt, model=self.model)
            exec_latency = (time.time() - exec_start) * 1000

            exec_input_tokens = len(execution_prompt) // 4
            exec_output_tokens = len(response) // 4 if response else 0
            total_input_tokens += exec_input_tokens
            total_output_tokens += exec_output_tokens

            trace.execution_response = response

            trace.steps.append(ProgramGenStep(
                phase="execute",
                description="Executed generated workflow via LLM",
                input_summary=execution_prompt[:200],
                output_summary=response[:200] if response else "No response",
                latency_ms=exec_latency,
            ))

            # Phase 3: Extract answer
            predicted = self.extract_boolean(response)
            trace.final_answer = predicted

            latency_ms = (time.time() - start_time) * 1000
            cost = calculate_cost(total_input_tokens, total_output_tokens, self.model)

            trace.metadata = {
                "model": self.model,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "generation_latency_ms": gen_latency,
                "execution_latency_ms": exec_latency,
            }

            self.traces.append(trace)

            is_correct = predicted == instance.answer if predicted is not None else False

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=is_correct,
                latency_ms=latency_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=cost.cost_usd,
                num_steps=len(trace.steps),
                decomposition_used=True,
                raw_response=response,
                trace=trace.to_dict(),
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            trace.success = False
            trace.steps.append(ProgramGenStep(
                phase="error",
                description=f"Error: {type(e).__name__}",
                input_summary=instance.question[:200],
                output_summary=str(e)[:200],
                success=False,
                error=str(e),
            ))
            self.traces.append(trace)

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=None,
                ground_truth=instance.answer,
                is_correct=False,
                latency_ms=latency_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=0.0,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _build_execution_prompt(self, question: str, program: GeneratedProgram) -> str:
        """Build a prompt that asks the LLM to execute the generated workflow."""
        # Describe the ptools and workflow so the LLM can follow the plan
        ptool_descriptions = []
        for pt in program.ptools:
            params = ", ".join(f"{name}: {typ}" for name, typ in pt.parameters)
            ptool_descriptions.append(
                f"  - {pt.name}({params}) -> {pt.return_type}: {pt.docstring}"
            )
        ptool_section = "\n".join(ptool_descriptions) if ptool_descriptions else "  (none)"

        return f"""You are executing a generated workflow to answer a yes/no question.

Question: {question}

The workflow has designed these reasoning tools (ptools):
{ptool_section}

Generated workflow:
{program.workflow_code}

Now execute this workflow step by step for the given question.
For each ptool call, provide the reasoning and result.
After completing all steps, give your final answer.

Answer with "Yes" or "No" at the end, on its own line, like:
Final Answer: Yes
or
Final Answer: No
"""

    def get_summary(self) -> Dict[str, Any]:
        summary = super().get_summary()

        if self.traces:
            successful = sum(1 for t in self.traces if t.success)
            total_ptools = sum(len(t.ptools_designed) for t in self.traces)
            total_steps = sum(len(t.steps) for t in self.traces)

            summary["program_gen_stats"] = {
                "total_traces": len(self.traces),
                "successful_traces": successful,
                "avg_ptools_per_question": total_ptools / len(self.traces),
                "avg_steps_per_trace": total_steps / len(self.traces),
            }

        return summary

    def save_traces(self, output_path: str) -> str:
        """
        Save all collected traces to a JSON file.

        Args:
            output_path: Path to save the traces JSON file.

        Returns:
            The path where traces were saved.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trace_data = {
            "experiment": "L2-program-gen",
            "model": self.model,
            "generated_at": datetime.now().isoformat(),
            "num_traces": len(self.traces),
            "summary": self.get_summary(),
            "traces": [t.to_dict() for t in self.traces],
        }

        with open(path, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)

        return str(path)
