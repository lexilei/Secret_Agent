"""
Pattern extraction utilities for typicality audits.

This module provides utilities for extracting reasoning patterns from traces
and computing pattern statistics for probabilistic modeling.

Key functions:
    - extract_pattern: Extract function name sequence from trace
    - get_ngrams: Get n-grams from a pattern
    - PatternStats: Statistical analysis of pattern corpus

Example:
    >>> from ptool_framework.audit.typicality.patterns import extract_pattern, get_ngrams
    >>>
    >>> # Extract pattern from trace DataFrame
    >>> pattern = extract_pattern(df)  # ['<START>', 'extract', 'analyze', '<END>']
    >>>
    >>> # Get bigrams
    >>> bigrams = get_ngrams(pattern, n=2)
    >>> # [('<START>', 'extract'), ('extract', 'analyze'), ('analyze', '<END>')]
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# Special tokens
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"


def extract_pattern(
    df: "pd.DataFrame",
    fn_name_column: str = "fn_name",
    add_special_tokens: bool = True,
) -> List[str]:
    """
    Extract a reasoning pattern (sequence of function names) from a trace DataFrame.

    Args:
        df: Trace DataFrame
        fn_name_column: Column containing function names
        add_special_tokens: Whether to add <START> and <END> tokens

    Returns:
        List of function names (pattern)

    Example:
        >>> pattern = extract_pattern(df)
        >>> print(pattern)  # ['<START>', 'extract_data', 'analyze', 'format', '<END>']
    """
    if df.empty:
        pattern = []
    else:
        pattern = df[fn_name_column].tolist()

    if add_special_tokens:
        pattern = [START_TOKEN] + pattern + [END_TOKEN]

    return pattern


def extract_patterns_from_traces(
    traces: List["pd.DataFrame"],
    fn_name_column: str = "fn_name",
    add_special_tokens: bool = True,
) -> List[List[str]]:
    """
    Extract patterns from multiple trace DataFrames.

    Args:
        traces: List of trace DataFrames
        fn_name_column: Column containing function names
        add_special_tokens: Whether to add special tokens

    Returns:
        List of patterns

    Example:
        >>> patterns = extract_patterns_from_traces([df1, df2, df3])
    """
    return [
        extract_pattern(df, fn_name_column, add_special_tokens)
        for df in traces
    ]


def get_ngrams(pattern: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from a pattern.

    Args:
        pattern: Sequence of step names
        n: Size of n-grams

    Returns:
        List of n-gram tuples

    Example:
        >>> pattern = ['<START>', 'extract', 'analyze', '<END>']
        >>> bigrams = get_ngrams(pattern, 2)
        >>> # [('<START>', 'extract'), ('extract', 'analyze'), ('analyze', '<END>')]
    """
    if len(pattern) < n:
        return []

    return [tuple(pattern[i:i + n]) for i in range(len(pattern) - n + 1)]


def get_unigrams(pattern: List[str]) -> List[Tuple[str]]:
    """
    Extract unigrams from a pattern.

    Args:
        pattern: Sequence of step names

    Returns:
        List of unigram tuples

    Example:
        >>> unigrams = get_unigrams(['extract', 'analyze'])
        >>> # [('extract',), ('analyze',)]
    """
    return [(step,) for step in pattern]


def get_bigrams(pattern: List[str]) -> List[Tuple[str, str]]:
    """
    Extract bigrams from a pattern.

    Args:
        pattern: Sequence of step names

    Returns:
        List of bigram tuples

    Example:
        >>> bigrams = get_bigrams(['<START>', 'extract', 'analyze', '<END>'])
        >>> # [('<START>', 'extract'), ('extract', 'analyze'), ('analyze', '<END>')]
    """
    return get_ngrams(pattern, 2)


def get_trigrams(pattern: List[str]) -> List[Tuple[str, str, str]]:
    """
    Extract trigrams from a pattern.

    Args:
        pattern: Sequence of step names

    Returns:
        List of trigram tuples
    """
    return get_ngrams(pattern, 3)


@dataclass
class PatternStats:
    """
    Statistical analysis of a pattern corpus.

    Computes various statistics useful for probabilistic modeling:
    - Vocabulary (unique step names)
    - N-gram counts
    - Transition probabilities
    - Pattern length distribution

    Attributes:
        patterns: The corpus of patterns
        vocabulary: Set of unique step names
        unigram_counts: Counts of each step
        bigram_counts: Counts of each bigram
        trigram_counts: Counts of each trigram
        pattern_lengths: Distribution of pattern lengths

    Example:
        >>> stats = PatternStats.from_patterns(training_patterns)
        >>> print(f"Vocabulary size: {len(stats.vocabulary)}")
        >>> print(f"Most common step: {stats.most_common_steps(1)}")
    """

    patterns: List[List[str]] = field(default_factory=list)
    vocabulary: Set[str] = field(default_factory=set)

    # N-gram counts
    unigram_counts: Counter = field(default_factory=Counter)
    bigram_counts: Counter = field(default_factory=Counter)
    trigram_counts: Counter = field(default_factory=Counter)

    # Length distribution
    pattern_lengths: List[int] = field(default_factory=list)

    # Transition counts (for computing probabilities)
    transition_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    @classmethod
    def from_patterns(cls, patterns: List[List[str]]) -> "PatternStats":
        """
        Create PatternStats from a corpus of patterns.

        Args:
            patterns: List of patterns (each pattern is a list of step names)

        Returns:
            PatternStats instance with computed statistics

        Example:
            >>> patterns = [
            ...     ['<START>', 'extract', 'analyze', '<END>'],
            ...     ['<START>', 'extract', 'validate', 'analyze', '<END>'],
            ... ]
            >>> stats = PatternStats.from_patterns(patterns)
        """
        stats = cls()
        stats.patterns = patterns

        for pattern in patterns:
            # Track vocabulary
            stats.vocabulary.update(pattern)

            # Track pattern length (excluding special tokens)
            length = sum(1 for step in pattern if step not in (START_TOKEN, END_TOKEN))
            stats.pattern_lengths.append(length)

            # Count n-grams
            for unigram in get_unigrams(pattern):
                stats.unigram_counts[unigram[0]] += 1

            for bigram in get_bigrams(pattern):
                stats.bigram_counts[bigram] += 1
                # Track transitions
                stats.transition_counts[bigram[0]][bigram[1]] += 1

            for trigram in get_trigrams(pattern):
                stats.trigram_counts[trigram] += 1

        return stats

    @property
    def total_patterns(self) -> int:
        """Number of patterns in the corpus."""
        return len(self.patterns)

    @property
    def total_steps(self) -> int:
        """Total number of steps across all patterns."""
        return sum(self.unigram_counts.values())

    @property
    def vocab_size(self) -> int:
        """Size of vocabulary (unique step names)."""
        return len(self.vocabulary)

    @property
    def avg_pattern_length(self) -> float:
        """Average pattern length (excluding special tokens)."""
        if not self.pattern_lengths:
            return 0.0
        return sum(self.pattern_lengths) / len(self.pattern_lengths)

    @property
    def min_pattern_length(self) -> int:
        """Minimum pattern length."""
        return min(self.pattern_lengths) if self.pattern_lengths else 0

    @property
    def max_pattern_length(self) -> int:
        """Maximum pattern length."""
        return max(self.pattern_lengths) if self.pattern_lengths else 0

    def most_common_steps(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most common step names.

        Args:
            n: Number of steps to return

        Returns:
            List of (step_name, count) tuples
        """
        return self.unigram_counts.most_common(n)

    def most_common_bigrams(self, n: int = 10) -> List[Tuple[Tuple[str, str], int]]:
        """
        Get the most common bigrams.

        Args:
            n: Number of bigrams to return

        Returns:
            List of (bigram, count) tuples
        """
        return self.bigram_counts.most_common(n)

    def step_probability(self, step: str) -> float:
        """
        Get the unigram probability of a step.

        Args:
            step: Step name

        Returns:
            Probability (0 to 1)
        """
        total = self.total_steps
        if total == 0:
            return 0.0
        return self.unigram_counts.get(step, 0) / total

    def transition_probability(self, from_step: str, to_step: str) -> float:
        """
        Get the transition probability P(to_step | from_step).

        Args:
            from_step: Source step
            to_step: Target step

        Returns:
            Conditional probability (0 to 1)
        """
        from_counts = self.transition_counts.get(from_step, {})
        total_from = sum(from_counts.values())
        if total_from == 0:
            return 0.0
        return from_counts.get(to_step, 0) / total_from

    def get_transition_matrix(self) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        """
        Get the full transition probability matrix.

        Returns:
            Tuple of (vocabulary_list, transition_probs_dict)
            where transition_probs_dict[from_step][to_step] = probability
        """
        vocab = sorted(self.vocabulary)
        matrix = {}

        for from_step in vocab:
            matrix[from_step] = {}
            for to_step in vocab:
                matrix[from_step][to_step] = self.transition_probability(from_step, to_step)

        return vocab, matrix

    def summary(self) -> str:
        """Get a human-readable summary of the statistics."""
        lines = [
            f"Pattern Statistics",
            f"  Total patterns: {self.total_patterns}",
            f"  Total steps: {self.total_steps}",
            f"  Vocabulary size: {self.vocab_size}",
            f"  Pattern length: {self.avg_pattern_length:.1f} avg ({self.min_pattern_length}-{self.max_pattern_length})",
            f"  Most common steps: {self.most_common_steps(3)}",
        ]
        return "\n".join(lines)


def normalize_step_names(
    patterns: List[List[str]],
    normalization: str = "lowercase",
) -> List[List[str]]:
    """
    Normalize step names in patterns.

    Args:
        patterns: List of patterns
        normalization: Type of normalization:
            - "lowercase": Convert to lowercase
            - "strip_prefix": Remove common prefixes (e.g., "ptool_")
            - "stem": Basic stemming (remove _data, _result suffixes)

    Returns:
        Normalized patterns

    Example:
        >>> patterns = [['Extract_Data', 'Analyze_Result']]
        >>> normalized = normalize_step_names(patterns, "lowercase")
        >>> # [['extract_data', 'analyze_result']]
    """
    result = []

    for pattern in patterns:
        normalized_pattern = []
        for step in pattern:
            if step in (START_TOKEN, END_TOKEN, UNK_TOKEN):
                normalized_pattern.append(step)
                continue

            if normalization == "lowercase":
                step = step.lower()
            elif normalization == "strip_prefix":
                # Remove common prefixes
                for prefix in ["ptool_", "tool_", "fn_"]:
                    if step.lower().startswith(prefix):
                        step = step[len(prefix):]
                        break
            elif normalization == "stem":
                # Basic stemming
                for suffix in ["_data", "_result", "_output", "_input"]:
                    if step.lower().endswith(suffix):
                        step = step[:-len(suffix)]
                        break

            normalized_pattern.append(step)

        result.append(normalized_pattern)

    return result


def cluster_similar_steps(
    patterns: List[List[str]],
    similarity_threshold: float = 0.8,
) -> Tuple[List[List[str]], Dict[str, str]]:
    """
    Cluster similar step names to reduce vocabulary.

    Uses simple string similarity to group similar steps.

    Args:
        patterns: List of patterns
        similarity_threshold: Minimum similarity to cluster (0-1)

    Returns:
        Tuple of (clustered_patterns, mapping) where mapping shows
        original -> canonical name

    Example:
        >>> patterns = [['extract_data', 'extract_values', 'analyze']]
        >>> clustered, mapping = cluster_similar_steps(patterns)
        >>> # If extract_data and extract_values are similar enough,
        >>> # they'll be mapped to the same canonical name
    """
    # Collect all unique steps
    all_steps = set()
    for pattern in patterns:
        all_steps.update(pattern)

    # Remove special tokens from clustering
    all_steps -= {START_TOKEN, END_TOKEN, UNK_TOKEN}

    # Simple clustering based on common prefix
    clusters: Dict[str, List[str]] = defaultdict(list)
    mapping: Dict[str, str] = {}

    for step in sorted(all_steps):
        # Find existing cluster or create new one
        found_cluster = None
        for canonical, members in clusters.items():
            # Check similarity with canonical
            if _string_similarity(step, canonical) >= similarity_threshold:
                found_cluster = canonical
                break

        if found_cluster:
            clusters[found_cluster].append(step)
            mapping[step] = found_cluster
        else:
            clusters[step].append(step)
            mapping[step] = step

    # Apply mapping to patterns
    clustered_patterns = []
    for pattern in patterns:
        clustered = [
            mapping.get(step, step) if step not in (START_TOKEN, END_TOKEN, UNK_TOKEN) else step
            for step in pattern
        ]
        clustered_patterns.append(clustered)

    return clustered_patterns, mapping


def _string_similarity(s1: str, s2: str) -> float:
    """Simple string similarity based on common prefix."""
    if not s1 or not s2:
        return 0.0

    s1_lower = s1.lower()
    s2_lower = s2.lower()

    # Find common prefix length
    common_prefix = 0
    for c1, c2 in zip(s1_lower, s2_lower):
        if c1 == c2:
            common_prefix += 1
        else:
            break

    # Similarity is common prefix / max length
    max_len = max(len(s1), len(s2))
    return common_prefix / max_len if max_len > 0 else 0.0


def patterns_to_sequences(
    patterns: List[List[str]],
    vocab: Optional[List[str]] = None,
) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
    """
    Convert patterns to integer sequences for HMM training.

    Args:
        patterns: List of patterns
        vocab: Optional pre-defined vocabulary (if None, built from patterns)

    Returns:
        Tuple of:
            - sequences: List of integer sequences
            - step_to_idx: Mapping from step name to index
            - idx_to_step: Mapping from index to step name

    Example:
        >>> sequences, s2i, i2s = patterns_to_sequences(patterns)
        >>> print(sequences[0])  # [0, 3, 2, 1]  (integer IDs)
    """
    # Build vocabulary if not provided
    if vocab is None:
        all_steps = set()
        for pattern in patterns:
            all_steps.update(pattern)
        vocab = [UNK_TOKEN] + sorted(all_steps - {UNK_TOKEN})

    # Create mappings
    step_to_idx = {step: idx for idx, step in enumerate(vocab)}
    idx_to_step = {idx: step for idx, step in enumerate(vocab)}

    # Convert patterns to sequences
    sequences = []
    for pattern in patterns:
        seq = [step_to_idx.get(step, step_to_idx.get(UNK_TOKEN, 0)) for step in pattern]
        sequences.append(seq)

    return sequences, step_to_idx, idx_to_step
