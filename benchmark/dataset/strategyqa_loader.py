"""
StrategyQA dataset loader.

Loads StrategyQA from local JSON files and provides structured access
with train/validation splitting.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StrategyQAInstance:
    """A single instance from the StrategyQA dataset."""
    qid: str
    term: str
    description: str
    question: str
    answer: bool                        # Ground truth boolean answer
    facts: List[str]                    # Supporting facts
    decomposition: List[str]            # Sub-questions for multi-hop reasoning
    evidence: List[Any] = field(default_factory=list)  # Evidence structure

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "qid": self.qid,
            "term": self.term,
            "description": self.description,
            "question": self.question,
            "answer": self.answer,
            "facts": self.facts,
            "decomposition": self.decomposition,
            "evidence": self.evidence,
        }

    def get_prompt(self, include_facts: bool = False, include_decomposition: bool = False) -> str:
        """Format as a prompt for the LLM."""
        prompt = f"Question: {self.question}"

        if include_facts:
            facts_str = "\n".join(f"- {fact}" for fact in self.facts)
            prompt += f"\n\nRelevant Facts:\n{facts_str}"

        if include_decomposition:
            decomp_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(self.decomposition))
            prompt += f"\n\nSub-questions to consider:\n{decomp_str}"

        return prompt


class StrategyQADataset:
    """
    StrategyQA dataset wrapper.

    Loads from local JSON files and provides filtered access with
    train/validation splitting.

    Usage:
        dataset = StrategyQADataset()
        train, val = dataset.load_with_split(val_ratio=0.2)
        debug_data = dataset.get_debug_subset(10)
    """

    def __init__(self, data_dir: str = "data/strategyqa"):
        """
        Initialize dataset loader.

        Args:
            data_dir: Directory containing StrategyQA JSON files
        """
        self.data_dir = Path(data_dir)
        self._train_data: Optional[List[StrategyQAInstance]] = None
        self._dev_data: Optional[List[StrategyQAInstance]] = None
        self._val_split: Optional[List[StrategyQAInstance]] = None
        self._train_split: Optional[List[StrategyQAInstance]] = None

    def load(self, split: str = "train") -> List[StrategyQAInstance]:
        """
        Load specified split from local JSON files.

        Args:
            split: "train" or "dev"

        Returns:
            List of StrategyQAInstance objects
        """
        if split == "train" and self._train_data is not None:
            return self._train_data
        if split == "dev" and self._dev_data is not None:
            return self._dev_data

        file_path = self.data_dir / f"{split}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"StrategyQA {split} file not found at {file_path}")

        print(f"Loading StrategyQA {split} split from {file_path}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        instances = []
        for item in raw_data:
            instance = self._parse_item(item)
            if instance is not None:
                instances.append(instance)

        print(f"Loaded {len(instances)} instances from {split} split")

        if split == "train":
            self._train_data = instances
        else:
            self._dev_data = instances

        return instances

    def _parse_item(self, item: Dict[str, Any]) -> Optional[StrategyQAInstance]:
        """Parse a JSON item into StrategyQAInstance."""
        try:
            return StrategyQAInstance(
                qid=item.get("qid", ""),
                term=item.get("term", ""),
                description=item.get("description", ""),
                question=item.get("question", ""),
                answer=bool(item.get("answer", False)),
                facts=item.get("facts", []),
                decomposition=item.get("decomposition", []),
                evidence=item.get("evidence", []),
            )
        except Exception as e:
            print(f"Warning: Failed to parse item: {e}")
            return None

    def load_with_split(
        self,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[List[StrategyQAInstance], List[StrategyQAInstance]]:
        """
        Load train data and split into train/validation sets.

        Args:
            val_ratio: Fraction to use for validation (default 20%)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_split, val_split)
        """
        if self._train_split is not None and self._val_split is not None:
            return self._train_split, self._val_split

        # Load full train data
        full_train = self.load("train")

        # Shuffle with seed for reproducibility
        rng = random.Random(seed)
        shuffled = full_train.copy()
        rng.shuffle(shuffled)

        # Split
        val_size = int(len(shuffled) * val_ratio)
        self._val_split = shuffled[:val_size]
        self._train_split = shuffled[val_size:]

        print(f"Split: {len(self._train_split)} train, {len(self._val_split)} validation")

        return self._train_split, self._val_split

    def get_debug_subset(self, n: int = 10, from_split: str = "val") -> List[StrategyQAInstance]:
        """
        Get first n instances for debugging.

        Args:
            n: Number of instances to return
            from_split: "train", "val", or "dev"

        Returns:
            First n instances from specified split
        """
        if from_split == "val":
            _, val_data = self.load_with_split()
            return val_data[:n]
        elif from_split == "train":
            train_data, _ = self.load_with_split()
            return train_data[:n]
        else:
            dev_data = self.load("dev")
            return dev_data[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with counts and answer distribution
        """
        train_data = self.load("train")

        # Count answers
        true_count = sum(1 for i in train_data if i.answer)
        false_count = len(train_data) - true_count

        # Average decomposition length
        avg_decomp = sum(len(i.decomposition) for i in train_data) / len(train_data)

        # Average facts count
        avg_facts = sum(len(i.facts) for i in train_data) / len(train_data)

        return {
            "total_instances": len(train_data),
            "answer_distribution": {
                "true": true_count,
                "false": false_count,
            },
            "avg_decomposition_steps": round(avg_decomp, 2),
            "avg_facts_per_question": round(avg_facts, 2),
        }


def load_strategyqa_dataset(
    split: str = "train",
    debug: bool = False,
    debug_n: int = 10,
    val_ratio: float = 0.2,
) -> List[StrategyQAInstance]:
    """
    Convenience function to load StrategyQA.

    Args:
        split: "train", "val", or "dev"
        debug: If True, return only first debug_n instances
        debug_n: Number of debug instances
        val_ratio: Validation split ratio (only used if split="val")

    Returns:
        List of StrategyQAInstance objects
    """
    dataset = StrategyQADataset()

    if debug:
        return dataset.get_debug_subset(debug_n, from_split=split)

    if split == "val":
        _, val_data = dataset.load_with_split(val_ratio=val_ratio)
        return val_data
    elif split == "train":
        train_data, _ = dataset.load_with_split(val_ratio=val_ratio)
        return train_data
    else:
        return dataset.load(split)
