"""
MedCalc-Bench dataset loader.

Loads the MedCalc-Bench dataset from HuggingFace and provides
structured access to instances.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import re


@dataclass
class MedCalcInstance:
    """A single instance from the MedCalc-Bench dataset."""
    row_number: int
    calculator_id: int
    calculator_name: str
    category: str                       # equation-based or rule-based
    output_type: str                    # decimal, integer, date, weeks/days
    note_id: str
    note_type: str                      # LLM-generated, template-based, extracted
    patient_note: str
    question: str
    relevant_entities: Dict[str, Any]   # Extracted parameters and values
    ground_truth_answer: Any            # float for numeric, str for date/weeks
    lower_limit: Optional[float]
    upper_limit: Optional[float]
    ground_truth_explanation: str
    ground_truth_raw: str = ""          # Original string representation

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "row_number": self.row_number,
            "calculator_id": self.calculator_id,
            "calculator_name": self.calculator_name,
            "category": self.category,
            "output_type": self.output_type,
            "note_id": self.note_id,
            "note_type": self.note_type,
            "patient_note": self.patient_note,
            "question": self.question,
            "relevant_entities": self.relevant_entities,
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_raw": self.ground_truth_raw,
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
            "ground_truth_explanation": self.ground_truth_explanation,
        }

    def is_numeric_output(self) -> bool:
        """Check if this instance expects a numeric answer."""
        return self.output_type.lower() in ["decimal", "integer"]

    def is_date_output(self) -> bool:
        """Check if this instance expects a date answer."""
        return self.output_type.lower() == "date"

    def is_weeks_days_output(self) -> bool:
        """Check if this instance expects a weeks/days answer."""
        return "week" in self.output_type.lower() or "day" in self.output_type.lower()

    def get_prompt(self, include_explanation: bool = False) -> str:
        """Format as a prompt for the LLM."""
        prompt = f"""Patient Note:
{self.patient_note}

Question: {self.question}"""

        if include_explanation:
            prompt += f"\n\nExpected Answer: {self.ground_truth_answer}"
            prompt += f"\nExplanation: {self.ground_truth_explanation}"

        return prompt


class MedCalcDataset:
    """
    MedCalc-Bench dataset wrapper.

    Loads from HuggingFace and provides filtered access.

    Usage:
        dataset = MedCalcDataset()
        test_data = dataset.load("test")
        debug_data = dataset.get_debug_subset(10)
    """

    def __init__(self, dataset_name: str = "ncbi/MedCalc-Bench-v1.2"):
        """
        Initialize dataset loader.

        Args:
            dataset_name: HuggingFace dataset identifier
        """
        self.dataset_name = dataset_name
        self._train_data: Optional[List[MedCalcInstance]] = None
        self._test_data: Optional[List[MedCalcInstance]] = None
        self._calculator_names: Optional[List[str]] = None

    def load(self, split: str = "test") -> List[MedCalcInstance]:
        """
        Load specified split from HuggingFace.

        Args:
            split: "train" or "test"

        Returns:
            List of MedCalcInstance objects
        """
        from datasets import load_dataset

        if split == "train" and self._train_data is not None:
            return self._train_data
        if split == "test" and self._test_data is not None:
            return self._test_data

        print(f"Loading MedCalc-Bench {split} split from HuggingFace...")
        ds = load_dataset(self.dataset_name, split=split)

        instances = []
        for idx, row in enumerate(ds):
            instance = self._parse_row(row, idx)
            if instance is not None:
                instances.append(instance)

        print(f"Loaded {len(instances)} instances from {split} split")

        if split == "train":
            self._train_data = instances
        else:
            self._test_data = instances

        return instances

    def _parse_row(self, row: Dict[str, Any], idx: int) -> Optional[MedCalcInstance]:
        """Parse a dataset row into MedCalcInstance."""
        try:
            # Parse relevant entities (may be JSON string or dict)
            relevant_entities = row.get("Relevant Entities", {})
            if isinstance(relevant_entities, str):
                try:
                    relevant_entities = json.loads(relevant_entities)
                except json.JSONDecodeError:
                    relevant_entities = {}

            # Get output type to determine how to parse answer
            output_type = row.get("Output Type", "decimal").lower()
            gt_raw = str(row.get("Ground Truth Answer", ""))

            # Parse ground truth answer based on output type
            gt_answer: Any
            if output_type == "date":
                # Keep dates as strings (format: MM/DD/YYYY)
                gt_answer = gt_raw.strip()
            elif "week" in output_type or "day" in output_type:
                # Keep weeks/days as strings (format: "('X weeks', 'Y days')" or similar)
                gt_answer = gt_raw.strip()
            else:
                # Numeric output (decimal, integer)
                gt_answer_val = row.get("Ground Truth Answer", 0)
                if isinstance(gt_answer_val, str):
                    # Extract number from string
                    match = re.search(r'-?\d+\.?\d*', gt_answer_val)
                    gt_answer = float(match.group()) if match else 0.0
                else:
                    gt_answer = float(gt_answer_val)

            # Parse limits (only relevant for numeric outputs)
            lower_limit = row.get("Lower Limit")
            upper_limit = row.get("Upper Limit")
            if lower_limit == "" or lower_limit is None:
                lower_limit = None
            else:
                try:
                    lower_limit = float(lower_limit)
                except (ValueError, TypeError):
                    lower_limit = None
            if upper_limit == "" or upper_limit is None:
                upper_limit = None
            else:
                try:
                    upper_limit = float(upper_limit)
                except (ValueError, TypeError):
                    upper_limit = None

            return MedCalcInstance(
                row_number=row.get("Row Number", idx),
                calculator_id=row.get("Calculator ID", 0),
                calculator_name=row.get("Calculator Name", "unknown"),
                category=row.get("Category", "unknown"),
                output_type=row.get("Output Type", "decimal"),
                note_id=str(row.get("Note ID", "")),
                note_type=row.get("Note Type", "unknown"),
                patient_note=row.get("Patient Note", ""),
                question=row.get("Question", ""),
                relevant_entities=relevant_entities,
                ground_truth_answer=gt_answer,
                ground_truth_raw=gt_raw,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                ground_truth_explanation=row.get("Ground Truth Explanation", ""),
            )
        except Exception as e:
            print(f"Warning: Failed to parse row {idx}: {e}")
            return None

    def get_debug_subset(self, n: int = 10) -> List[MedCalcInstance]:
        """
        Get first n instances for debugging.

        Args:
            n: Number of instances to return

        Returns:
            First n test instances
        """
        test_data = self.load("test")
        return test_data[:n]

    def get_by_calculator(self, calculator_name: str) -> List[MedCalcInstance]:
        """
        Filter instances by calculator type.

        Args:
            calculator_name: Name of the calculator

        Returns:
            Instances using that calculator
        """
        test_data = self.load("test")
        return [i for i in test_data if i.calculator_name.lower() == calculator_name.lower()]

    def get_by_category(self, category: str) -> List[MedCalcInstance]:
        """
        Filter instances by category.

        Args:
            category: "equation-based" or "rule-based"

        Returns:
            Instances in that category
        """
        test_data = self.load("test")
        return [i for i in test_data if category.lower() in i.category.lower()]

    def get_calculator_names(self) -> List[str]:
        """
        Get unique calculator names in the dataset.

        Returns:
            List of 55 calculator names
        """
        if self._calculator_names is not None:
            return self._calculator_names

        test_data = self.load("test")
        self._calculator_names = sorted(set(i.calculator_name for i in test_data))
        return self._calculator_names

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with counts by category, calculator, etc.
        """
        test_data = self.load("test")

        # Count by category
        category_counts = {}
        calculator_counts = {}
        output_type_counts = {}

        for instance in test_data:
            category_counts[instance.category] = category_counts.get(instance.category, 0) + 1
            calculator_counts[instance.calculator_name] = calculator_counts.get(instance.calculator_name, 0) + 1
            output_type_counts[instance.output_type] = output_type_counts.get(instance.output_type, 0) + 1

        return {
            "total_instances": len(test_data),
            "num_calculators": len(calculator_counts),
            "by_category": category_counts,
            "by_calculator": calculator_counts,
            "by_output_type": output_type_counts,
        }


def load_medcalc_dataset(
    split: str = "test",
    debug: bool = False,
    debug_n: int = 10,
) -> List[MedCalcInstance]:
    """
    Convenience function to load MedCalc-Bench.

    Args:
        split: "train" or "test"
        debug: If True, return only first debug_n instances
        debug_n: Number of debug instances

    Returns:
        List of MedCalcInstance objects
    """
    dataset = MedCalcDataset()

    if debug:
        return dataset.get_debug_subset(debug_n)

    return dataset.load(split)
