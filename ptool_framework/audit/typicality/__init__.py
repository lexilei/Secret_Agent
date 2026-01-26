"""
Typicality models for SSRM-style auditing.

This package provides probabilistic models over reasoning patterns for
detecting atypical (potentially erroneous) traces.

Models:
    - UnigramModel: Bag-of-steps with Dirichlet smoothing
    - BigramModel: P(step_i | step_{i-1})
    - TrigramModel: P(step_i | step_{i-2}, step_{i-1})
    - InterpolatedModel: Linear combination of unigram/bigram/trigram
    - HMMModel: Hidden Markov Model (requires hmmlearn)

Utilities:
    - Pattern extraction: extract_pattern, get_ngrams
    - Pattern statistics: PatternStats
    - Normalization: normalize_step_names

Example:
    >>> from ptool_framework.audit.typicality import (
    ...     BigramModel,
    ...     extract_pattern,
    ...     PatternStats,
    ... )
    >>>
    >>> # Extract patterns from DataFrames
    >>> patterns = [extract_pattern(df) for df in trace_dfs]
    >>>
    >>> # Analyze pattern corpus
    >>> stats = PatternStats.from_patterns(patterns)
    >>> print(stats.summary())
    >>>
    >>> # Train model
    >>> model = BigramModel()
    >>> model.fit(patterns)
    >>>
    >>> # Score new pattern
    >>> score = model.score(['<START>', 'extract', 'analyze', '<END>'])
    >>> print(f"Typicality: {score:.4f}")
"""

from .patterns import (
    # Constants
    START_TOKEN,
    END_TOKEN,
    UNK_TOKEN,
    # Pattern extraction
    extract_pattern,
    extract_patterns_from_traces,
    get_ngrams,
    get_unigrams,
    get_bigrams,
    get_trigrams,
    # Statistics
    PatternStats,
    # Normalization
    normalize_step_names,
    cluster_similar_steps,
    patterns_to_sequences,
)

from .models import (
    # Base n-gram models
    UnigramModel,
    NGramModel,
    BigramModel,
    TrigramModel,
    # Advanced models
    InterpolatedModel,
    HMMModel,
    HMMGridSearch,
    HMMGridSearchResult,
    # Factory functions
    create_typicality_model,
    train_ensemble_models,
    ensemble_score,
)

__all__ = [
    # Constants
    "START_TOKEN",
    "END_TOKEN",
    "UNK_TOKEN",
    # Pattern extraction
    "extract_pattern",
    "extract_patterns_from_traces",
    "get_ngrams",
    "get_unigrams",
    "get_bigrams",
    "get_trigrams",
    # Statistics
    "PatternStats",
    # Normalization
    "normalize_step_names",
    "cluster_similar_steps",
    "patterns_to_sequences",
    # Models
    "UnigramModel",
    "NGramModel",
    "BigramModel",
    "TrigramModel",
    "InterpolatedModel",
    "HMMModel",
    "HMMGridSearch",
    "HMMGridSearchResult",
    # Factory functions
    "create_typicality_model",
    "train_ensemble_models",
    "ensemble_score",
]
