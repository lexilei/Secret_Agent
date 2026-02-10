"""
StrategyQA experiments package.

Contains experiment implementations for StrategyQA benchmark.

Levels (based on William Cohen's research plan):
- L0: Baseline prompt
- L1: Structured prompts (CoT, CoC, PTP)
- L2: Trace Builder (generates workflow traces)
- L3: ReAct Agent
"""

from .base import StrategyQAExperiment, StrategyQAResult
from .baseline import StrategyQABaseline, StrategyQABaselineCoT
from .decompose import StrategyQADecomposeOracle, StrategyQADecomposeLLM
from .react_exp import StrategyQAReAct, StrategyQAReActSimple
from .trace_builder import L2_TraceBuilder, L2_TraceBuilderRAG, WorkflowTrace, TraceStep
from .levels import (
    L0_Baseline,
    L1_CoT, L1_CoC, L1_PTP,
    L2_Decompose,  # Alias for L2_TraceBuilder
    L3_ReAct, L3_ReActSimple, L3_ReActRAG,
    L4_Adaptive, L4_Pipeline,
    L5_Improving, L5_ICL,
    EXPERIMENTS,
)

__all__ = [
    # Base
    "StrategyQAExperiment",
    "StrategyQAResult",
    # Legacy
    "StrategyQABaseline",
    "StrategyQABaselineCoT",
    "StrategyQADecomposeOracle",
    "StrategyQADecomposeLLM",
    "StrategyQAReAct",
    "StrategyQAReActSimple",
    # Levels
    "L0_Baseline",
    "L1_CoT", "L1_CoC", "L1_PTP",
    "L2_TraceBuilder", "L2_TraceBuilderRAG", "L2_Decompose",
    "WorkflowTrace", "TraceStep",
    "L3_ReAct", "L3_ReActSimple", "L3_ReActRAG",
    "L4_Adaptive", "L4_Pipeline",
    "L5_Improving", "L5_ICL",
    "EXPERIMENTS",
]
