"""
L2 Trace Builder for StrategyQA.

Based on William Cohen's research plan:
- Trace Builder is a PoT-like ptool that produces a workflow trace
- Trace = instance-specific sequence of ptool calls with goals
- Traces are executable, auditable, and usable for ICL/distillation

Key concepts:
- ptool: A Python stub with type signature + documentation (executed by LLM)
- trace: Sequence of (ptool_name, input, output, goal) tuples
- trace builder: Generates instance-specific traces
"""

import time
import json
import re
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal
from collections import Counter

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost

from ptool_framework.llm_backend import call_llm


# ============================================================================
# RAG: Paragraph Retriever
# ============================================================================

class ParagraphRetriever:
    """
    TF-IDF based retriever for StrategyQA paragraphs.

    Uses simple TF-IDF similarity without external dependencies.
    Loads paragraphs from strategyqa_train_paragraphs.json.
    """

    def __init__(self, paragraphs_path: Optional[str] = None):
        if paragraphs_path is None:
            # Default path relative to this file
            base_dir = Path(__file__).parent.parent.parent.parent
            paragraphs_path = base_dir / "data" / "strategyqa" / "strategyqa_train_paragraphs.json"

        self.paragraphs_path = Path(paragraphs_path)
        self.paragraphs: Dict[str, Dict] = {}
        self.doc_freqs: Dict[str, int] = {}  # Document frequency for IDF
        self.doc_vectors: Dict[str, Dict[str, float]] = {}  # TF-IDF vectors
        self._loaded = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency (normalized)."""
        counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {term: count / total for term, count in counts.items()}

    def load(self):
        """Load paragraphs and build index."""
        if self._loaded:
            return

        if not self.paragraphs_path.exists():
            raise FileNotFoundError(f"Paragraphs file not found: {self.paragraphs_path}")

        with open(self.paragraphs_path, 'r') as f:
            self.paragraphs = json.load(f)

        # Build document frequency
        for para_id, para_data in self.paragraphs.items():
            content = para_data.get("content", "")
            title = para_data.get("title", "")
            tokens = set(self._tokenize(f"{title} {content}"))
            for token in tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Pre-compute TF-IDF vectors
        num_docs = len(self.paragraphs)
        for para_id, para_data in self.paragraphs.items():
            content = para_data.get("content", "")
            title = para_data.get("title", "")
            tokens = self._tokenize(f"{title} {content}")
            tf = self._compute_tf(tokens)

            # TF-IDF
            tfidf = {}
            for term, tf_val in tf.items():
                df = self.doc_freqs.get(term, 1)
                idf = math.log(num_docs / df) if df > 0 else 0
                tfidf[term] = tf_val * idf
            self.doc_vectors[para_id] = tfidf

        self._loaded = True

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant paragraphs for a query.

        Args:
            query: The search query
            top_k: Number of paragraphs to retrieve

        Returns:
            List of {"para_id": str, "title": str, "content": str, "score": float}
        """
        if not self._loaded:
            self.load()

        # Compute query vector
        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)

        num_docs = len(self.paragraphs)
        query_tfidf = {}
        for term, tf_val in query_tf.items():
            df = self.doc_freqs.get(term, 1)
            idf = math.log(num_docs / df) if df > 0 else 0
            query_tfidf[term] = tf_val * idf

        # Compute cosine similarity with all documents
        scores = []
        query_norm = math.sqrt(sum(v ** 2 for v in query_tfidf.values())) or 1

        for para_id, doc_tfidf in self.doc_vectors.items():
            # Dot product
            dot = sum(query_tfidf.get(term, 0) * weight for term, weight in doc_tfidf.items())
            doc_norm = math.sqrt(sum(v ** 2 for v in doc_tfidf.values())) or 1
            similarity = dot / (query_norm * doc_norm)
            scores.append((para_id, similarity))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        results = []
        for para_id, score in scores[:top_k]:
            para_data = self.paragraphs[para_id]
            results.append({
                "para_id": para_id,
                "title": para_data.get("title", ""),
                "content": para_data.get("content", ""),
                "score": round(score, 4),
            })

        return results


# Global retriever instance (lazy loaded)
_paragraph_retriever: Optional[ParagraphRetriever] = None

def get_paragraph_retriever() -> ParagraphRetriever:
    """Get or create the global paragraph retriever."""
    global _paragraph_retriever
    if _paragraph_retriever is None:
        _paragraph_retriever = ParagraphRetriever()
    return _paragraph_retriever


# ============================================================================
# Trace Data Structures
# ============================================================================

@dataclass
class TraceStep:
    """A single step in a workflow trace."""
    ptool: str                      # Name of the ptool executed
    goal: str                       # What this step aims to achieve
    input: Dict[str, Any]           # Input parameters to the ptool
    output: Any                     # Output from the ptool
    success: bool = True            # Whether the step succeeded
    error: Optional[str] = None     # Error message if failed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_icl_example(self) -> str:
        """Format as ICL example for teaching other models."""
        return f"""# Goal: {self.goal}
>>> {self.ptool}({json.dumps(self.input, ensure_ascii=False)})
{json.dumps(self.output, ensure_ascii=False)}"""


@dataclass
class WorkflowTrace:
    """A complete workflow trace for solving a question."""
    question: str
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: Optional[bool] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "success": self.success,
            "metadata": self.metadata,
        }

    def to_icl_demo(self) -> str:
        """Format entire trace as ICL demonstration."""
        lines = [f'# Question: {self.question}', '']
        for i, step in enumerate(self.steps):
            lines.append(f'# Step {i+1}: {step.goal}')
            lines.append(f'>>> {step.ptool}({json.dumps(step.input, ensure_ascii=False)})')
            lines.append(json.dumps(step.output, ensure_ascii=False))
            lines.append('')
        lines.append(f'# Final Answer: {self.final_answer}')
        return '\n'.join(lines)

    def get_audit_info(self) -> Dict[str, Any]:
        """Extract info for auditing the trace."""
        return {
            "num_steps": len(self.steps),
            "ptools_used": [s.ptool for s in self.steps],
            "all_succeeded": all(s.success for s in self.steps),
            "has_decomposition": any(s.ptool == "decompose" for s in self.steps),
            "has_final_answer": self.final_answer is not None,
        }


# ============================================================================
# PTools (executed by LLM)
# ============================================================================

class PToolExecutor:
    """Executor for ptools - calls LLM to execute each ptool."""

    def __init__(self, model: str = "deepseek-v3-0324", use_rag: bool = False):
        self.model = model
        self.use_rag = use_rag
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._retriever: Optional[ParagraphRetriever] = None

    @property
    def retriever(self) -> ParagraphRetriever:
        """Lazy load retriever only when needed."""
        if self._retriever is None:
            self._retriever = get_paragraph_retriever()
            self._retriever.load()
        return self._retriever

    def execute(self, ptool_name: str, goal: str, inputs: Dict[str, Any]) -> TraceStep:
        """Execute a ptool and return a TraceStep."""
        # Dispatch to appropriate ptool
        ptool_map = {
            "decompose": self._decompose,
            "answer_factual": self._answer_factual,
            "retrieve": self._retrieve,
            "evaluate_condition": self._evaluate_condition,
            "combine_facts": self._combine_facts,
        }

        if ptool_name not in ptool_map:
            return TraceStep(
                ptool=ptool_name,
                goal=goal,
                input=inputs,
                output=None,
                success=False,
                error=f"Unknown ptool: {ptool_name}"
            )

        try:
            output = ptool_map[ptool_name](**inputs)
            return TraceStep(
                ptool=ptool_name,
                goal=goal,
                input=inputs,
                output=output,
                success=True,
            )
        except Exception as e:
            return TraceStep(
                ptool=ptool_name,
                goal=goal,
                input=inputs,
                output=None,
                success=False,
                error=str(e),
            )

    def _call_llm(self, prompt: str) -> str:
        """Call LLM and track tokens."""
        response = call_llm(prompt=prompt, model=self.model)
        self.total_input_tokens += len(prompt) // 4
        self.total_output_tokens += len(response) // 4 if response else 0
        return response

    def _decompose(self, question: str) -> List[Dict[str, str]]:
        """
        ptool: decompose
        Decompose a yes/no question into sub-questions.

        Returns list of {type: "factual"|"comparison", question: str}
        """
        prompt = f"""Decompose this yes/no question into 2-3 simpler sub-questions.

Question: {question}

For each sub-question, specify:
- type: "factual" (needs fact lookup) or "comparison" (compares previous answers)
- question: the sub-question text

Use #1, #2 etc to reference answers from previous sub-questions.

Return as JSON array:
[{{"type": "factual", "question": "..."}}, {{"type": "comparison", "question": "Is #1 > #2?"}}]

Sub-questions:"""

        response = self._call_llm(prompt)

        # Parse JSON
        try:
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: parse lines
        subs = []
        for line in response.split('\n'):
            line = line.strip()
            if line and ('?' in line or line.startswith(('-', '•', '*'))):
                q = re.sub(r'^[\d\.\)\-\•\*\s]+', '', line).strip()
                if q:
                    subs.append({"type": "factual", "question": q})
        return subs[:4] if subs else [{"type": "factual", "question": question}]

    def _answer_factual(self, question: str, context: Optional[str] = None) -> str:
        """
        ptool: answer_factual
        Answer a factual question with a brief response.
        """
        ctx = f"\nContext: {context}" if context else ""
        prompt = f"""Answer this factual question briefly (1-2 sentences).
{ctx}
Question: {question}

Answer:"""

        response = self._call_llm(prompt)
        return response.strip() if response else "Unknown"

    def _retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        ptool: retrieve
        Retrieve relevant paragraphs from the knowledge base using TF-IDF.

        Args:
            query: Search query (usually a sub-question)
            top_k: Number of paragraphs to retrieve

        Returns:
            List of {"title": str, "content": str, "score": float}
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        # Return simplified format
        return [
            {"title": r["title"], "content": r["content"], "score": r["score"]}
            for r in results
        ]

    def _evaluate_condition(self, condition: str, facts: Dict[str, str]) -> bool:
        """
        ptool: evaluate_condition
        Evaluate a condition based on given facts.
        Returns True or False.
        """
        facts_str = "\n".join(f"#{i+1}: {v}" for i, v in enumerate(facts.values()))
        prompt = f"""Evaluate this condition based on the facts.

Facts:
{facts_str}

Condition: {condition}

Is this condition True or False? Answer with only "True" or "False":"""

        response = self._call_llm(prompt)
        return "true" in response.lower()

    def _combine_facts(self, question: str, facts: List[str]) -> bool:
        """
        ptool: combine_facts
        Combine facts to answer the original yes/no question.
        """
        facts_str = "\n".join(f"- {f}" for f in facts)
        prompt = f"""Based on these facts, answer the yes/no question.

Question: {question}

Facts:
{facts_str}

Is the answer Yes or No? Answer with only "Yes" or "No":"""

        response = self._call_llm(prompt)
        return "yes" in response.lower()


# ============================================================================
# Trace Builder
# ============================================================================

class StrategyQATraceBuilder:
    """
    Trace Builder for StrategyQA.

    Generates instance-specific workflow traces by:
    1. Decomposing the question into sub-questions
    2. Answering each sub-question (resolving #N references)
    3. Combining facts to get final answer

    The trace is:
    - Executable (can re-run each step)
    - Auditable (can check correctness of each step)
    - ICL-ready (can format as demonstration)
    """

    def __init__(self, model: str = "deepseek-v3-0324"):
        self.model = model
        self.executor = PToolExecutor(model)

    def build_trace(self, question: str) -> WorkflowTrace:
        """Build a workflow trace for answering the question."""
        trace = WorkflowTrace(question=question)

        # Step 1: Decompose
        decompose_step = self.executor.execute(
            ptool_name="decompose",
            goal="Break down the question into simpler sub-questions",
            inputs={"question": question},
        )
        trace.steps.append(decompose_step)

        if not decompose_step.success:
            trace.success = False
            return trace

        sub_questions = decompose_step.output
        answers = {}  # Store answers by index for #N reference resolution

        # Step 2: Answer each sub-question
        for i, sub_q in enumerate(sub_questions):
            sub_type = sub_q.get("type", "factual")
            sub_text = sub_q.get("question", str(sub_q))

            # Resolve #N references
            resolved_text = self._resolve_references(sub_text, answers)

            if sub_type == "comparison" and answers:
                # Use evaluate_condition for comparisons
                step = self.executor.execute(
                    ptool_name="evaluate_condition",
                    goal=f"Evaluate: {resolved_text}",
                    inputs={"condition": resolved_text, "facts": answers},
                )
                if step.success:
                    answers[i] = "True" if step.output else "False"
            else:
                # Use answer_factual for facts
                step = self.executor.execute(
                    ptool_name="answer_factual",
                    goal=f"Find fact: {resolved_text}",
                    inputs={"question": resolved_text},
                )
                if step.success:
                    answers[i] = step.output

            trace.steps.append(step)

        # Step 3: Combine to get final answer
        facts_list = list(answers.values())
        combine_step = self.executor.execute(
            ptool_name="combine_facts",
            goal="Combine facts to determine final yes/no answer",
            inputs={"question": question, "facts": facts_list},
        )
        trace.steps.append(combine_step)

        if combine_step.success:
            trace.final_answer = combine_step.output
        else:
            trace.success = False

        # Store metadata
        trace.metadata = {
            "model": self.model,
            "num_sub_questions": len(sub_questions),
            "input_tokens": self.executor.total_input_tokens,
            "output_tokens": self.executor.total_output_tokens,
        }

        return trace

    def _resolve_references(self, text: str, answers: Dict[int, str]) -> str:
        """Replace #N references with actual answers."""
        result = text
        for match in re.finditer(r'#(\d+)', text):
            idx = int(match.group(1)) - 1
            if idx in answers:
                result = result.replace(match.group(0), f'"{answers[idx]}"')
        return result


# ============================================================================
# L2 Experiment using Trace Builder
# ============================================================================

class L2_TraceBuilder(StrategyQAExperiment):
    """
    L2: Trace Builder experiment.

    Uses StrategyQATraceBuilder to generate workflow traces.
    Each trace is a sequence of ptool calls that can be:
    - Executed step by step
    - Audited for correctness
    - Used as ICL demonstrations
    - Distilled into simpler workflows
    """

    def __init__(self, model: str = "deepseek-v3-0324"):
        super().__init__(model)
        self.trace_builder = StrategyQATraceBuilder(model)
        self.traces: List[WorkflowTrace] = []  # Store all traces for later analysis

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run trace builder on a single instance."""
        start_time = time.time()

        # Reset executor token counts
        self.trace_builder.executor.total_input_tokens = 0
        self.trace_builder.executor.total_output_tokens = 0

        try:
            # Build trace
            trace = self.trace_builder.build_trace(instance.question)
            self.traces.append(trace)

            latency_ms = (time.time() - start_time) * 1000

            # Get token counts
            input_tokens = trace.metadata.get("input_tokens", 0)
            output_tokens = trace.metadata.get("output_tokens", 0)

            cost = calculate_cost(input_tokens, output_tokens, self.model)

            # Determine correctness
            predicted = trace.final_answer
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
                decomposition_used=True,
                raw_response=trace.to_icl_demo(),
                trace=trace.to_dict(),
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=None,
                ground_truth=instance.answer,
                is_correct=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
                error_type=type(e).__name__,
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with trace statistics."""
        summary = super().get_summary()

        if self.traces:
            # Analyze traces
            total_steps = sum(len(t.steps) for t in self.traces)
            successful_traces = sum(1 for t in self.traces if t.success)

            summary["trace_stats"] = {
                "total_traces": len(self.traces),
                "successful_traces": successful_traces,
                "avg_steps_per_trace": total_steps / len(self.traces),
                "ptools_used": self._count_ptools(),
            }

        return summary

    def _count_ptools(self) -> Dict[str, int]:
        """Count ptool usage across all traces."""
        counts = {}
        for trace in self.traces:
            for step in trace.steps:
                counts[step.ptool] = counts.get(step.ptool, 0) + 1
        return counts

    def export_traces_for_icl(self, n: int = 5) -> str:
        """Export successful traces as ICL demonstrations."""
        successful = [t for t in self.traces if t.success and t.final_answer is not None]
        demos = []
        for trace in successful[:n]:
            demos.append(trace.to_icl_demo())
        return "\n\n---\n\n".join(demos)

    def export_traces_for_audit(self) -> List[Dict[str, Any]]:
        """Export trace audit info for analysis."""
        return [t.get_audit_info() for t in self.traces]
