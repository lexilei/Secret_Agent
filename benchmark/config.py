"""
Experiment configurations for MedCalc-Bench benchmark.

Defines all experiment configurations for the ablation study:
- L0: Baseline (direct LLM)
- L1: @ptool only
- L2: @distilled (Python-first)
- L3: ReActAgent
- L3+: ReAct + Critic + Repair
- L4: AgentOrchestrator
- L5: SelfImprovingAgent
- L5+: Self-improving with ICL from training
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ExperimentLevel(Enum):
    """Framework level being tested."""
    L0 = "L0"      # Baseline - no framework
    L1 = "L1"      # @ptool only
    L1O = "L1O"    # @ptool extraction + official Python calculators
    L2 = "L2"      # @distilled
    L2O = "L2O"    # @distilled with official calculators
    L3 = "L3"      # ReActAgent
    L3_AUDIT = "L3+"  # ReAct + Critic + Repair
    L4 = "L4"      # AgentOrchestrator
    L4_PIPELINE = "L4P"  # Python-orchestrated pipeline
    L5 = "L5"      # SelfImprovingAgent
    L5_ICL = "L5+"  # Self-improving with ICL


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    level: ExperimentLevel

    # Model settings
    model: str = "deepseek-v3-0324"

    # ReAct settings
    max_steps: int = 15

    # Audit settings
    enable_audit: bool = False
    enable_critic: bool = False
    enable_repair: bool = False
    max_repair_attempts: int = 2

    # Orchestrator settings (L4)
    use_orchestrator: bool = False
    execution_mode: str = "sequential"  # sequential, parallel, pipeline

    # Self-improving settings (L5)
    enable_learning: bool = False
    learn_from_success: bool = True
    learn_from_failure: bool = True
    decay_enabled: bool = True

    # ICL settings (L5+)
    use_icl_from_train: bool = False
    icl_examples_per_calculator: int = 3

    # Timeout
    timeout_seconds: float = 60.0

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "level": self.level.value,
            "model": self.model,
            "max_steps": self.max_steps,
            "enable_audit": self.enable_audit,
            "enable_critic": self.enable_critic,
            "enable_repair": self.enable_repair,
            "max_repair_attempts": self.max_repair_attempts,
            "use_orchestrator": self.use_orchestrator,
            "execution_mode": self.execution_mode,
            "enable_learning": self.enable_learning,
            "learn_from_success": self.learn_from_success,
            "learn_from_failure": self.learn_from_failure,
            "decay_enabled": self.decay_enabled,
            "use_icl_from_train": self.use_icl_from_train,
            "icl_examples_per_calculator": self.icl_examples_per_calculator,
            "timeout_seconds": self.timeout_seconds,
        }


# Pre-defined experiment configurations for ablation study
ABLATION_CONFIGS: Dict[str, ExperimentConfig] = {
    "baseline": ExperimentConfig(
        name="baseline",
        description="Direct LLM call without any framework (vanilla baseline)",
        level=ExperimentLevel.L0,
    ),

    "l1_ptool": ExperimentConfig(
        name="l1_ptool",
        description="Single @ptool call with structured JSON output",
        level=ExperimentLevel.L1,
    ),

    "l2_distilled": ExperimentConfig(
        name="l2_distilled",
        description="@distilled decorator: Python-first with LLM fallback",
        level=ExperimentLevel.L2,
    ),

    "l3_react": ExperimentConfig(
        name="l3_react",
        description="ReActAgent with multi-step reasoning and tool use",
        level=ExperimentLevel.L3,
        max_steps=15,
    ),

    "l3_audit": ExperimentConfig(
        name="l3_audit",
        description="ReAct + TraceCritic + RepairAgent for error detection and correction",
        level=ExperimentLevel.L3_AUDIT,
        max_steps=15,
        enable_audit=True,
        enable_critic=True,
        enable_repair=True,
        max_repair_attempts=2,
    ),

    "l4_orchestrator": ExperimentConfig(
        name="l4_orchestrator",
        description="AgentOrchestrator with specialized agents per calculator category",
        level=ExperimentLevel.L4,
        use_orchestrator=True,
        execution_mode="sequential",
    ),

    "l5_improving": ExperimentConfig(
        name="l5_improving",
        description="SelfImprovingAgent that learns from execution experience",
        level=ExperimentLevel.L5,
        enable_learning=True,
        learn_from_success=True,
        learn_from_failure=True,
        decay_enabled=True,
    ),

    "l5_icl": ExperimentConfig(
        name="l5_icl",
        description="Self-improving agent with ICL examples from training set",
        level=ExperimentLevel.L5_ICL,
        enable_learning=True,
        learn_from_success=True,
        learn_from_failure=True,
        decay_enabled=True,
        use_icl_from_train=True,
        icl_examples_per_calculator=3,
    ),

    "l1o_official": ExperimentConfig(
        name="l1o_official",
        description="LLM extraction + official Python calculator implementations",
        level=ExperimentLevel.L1O,
    ),

    "l2o_official": ExperimentConfig(
        name="l2o_official",
        description="@distilled with official calculator implementations",
        level=ExperimentLevel.L2O,
    ),

    "l4_pipeline": ExperimentConfig(
        name="l4_pipeline",
        description="Python-orchestrated pipeline with specialist LLM agents",
        level=ExperimentLevel.L4_PIPELINE,
    ),
}


def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment configuration by name."""
    if name not in ABLATION_CONFIGS:
        available = ", ".join(ABLATION_CONFIGS.keys())
        raise ValueError(f"Unknown experiment: {name}. Available: {available}")
    return ABLATION_CONFIGS[name]


def get_all_experiment_names() -> List[str]:
    """Get all available experiment names."""
    return list(ABLATION_CONFIGS.keys())
