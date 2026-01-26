"""
Probabilistic models for typicality audits.

This module provides probabilistic models over reasoning patterns:
- UnigramModel: Multinomial with Dirichlet smoothing
- BigramModel: Conditional probabilities P(step_i | step_{i-1})
- TrigramModel: Conditional probabilities P(step_i | step_{i-2}, step_{i-1})
- HMMModel: Hidden Markov Model (requires hmmlearn)

These models assign probability scores to patterns. Atypical patterns
(low probability) may indicate errors in reasoning.

Example:
    >>> from ptool_framework.audit.typicality.models import BigramModel
    >>>
    >>> # Train on good traces
    >>> model = BigramModel()
    >>> model.fit(training_patterns)
    >>>
    >>> # Score a new pattern
    >>> score = model.score(['<START>', 'extract', 'analyze', '<END>'])
    >>> print(f"Pattern typicality: {score:.4f}")
"""

from __future__ import annotations

import json
import math
from abc import abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..base import BaseTypicalityModel
from .patterns import (
    START_TOKEN,
    END_TOKEN,
    UNK_TOKEN,
    get_ngrams,
    get_unigrams,
    get_bigrams,
    get_trigrams,
    PatternStats,
)


class UnigramModel(BaseTypicalityModel):
    """
    Unigram (bag-of-steps) model with Dirichlet smoothing.

    Assigns probability to patterns based on individual step frequencies:
    P(pattern) = Product of P(step_i) for each step

    Uses Dirichlet (add-k) smoothing to handle unseen steps.

    Attributes:
        name: Model name ("unigram")
        smoothing_alpha: Smoothing parameter (default 1.0 for Laplace smoothing)
        vocabulary: Set of known step names
        counts: Step counts

    Example:
        >>> model = UnigramModel(smoothing_alpha=0.1)
        >>> model.fit(training_patterns)
        >>> score = model.score(['<START>', 'extract', 'analyze', '<END>'])
    """

    name = "unigram"

    def __init__(self, smoothing_alpha: float = 1.0):
        """
        Initialize the unigram model.

        Args:
            smoothing_alpha: Dirichlet smoothing parameter (default 1.0 = Laplace)
        """
        self.smoothing_alpha = smoothing_alpha
        self.vocabulary: set = set()
        self.counts: Counter = Counter()
        self.total_count: int = 0

    def fit(self, patterns: List[List[str]]) -> "UnigramModel":
        """
        Train the model on a corpus of patterns.

        Args:
            patterns: List of patterns (each pattern is a list of step names)

        Returns:
            self for method chaining
        """
        self.counts = Counter()
        self.vocabulary = set()

        for pattern in patterns:
            for step in pattern:
                self.counts[step] += 1
                self.vocabulary.add(step)

        self.total_count = sum(self.counts.values())
        return self

    def step_probability(self, step: str) -> float:
        """
        Get smoothed probability of a step.

        Args:
            step: Step name

        Returns:
            Smoothed probability
        """
        vocab_size = len(self.vocabulary)
        if vocab_size == 0:
            return 0.0

        count = self.counts.get(step, 0)
        # Add-k smoothing
        prob = (count + self.smoothing_alpha) / (
            self.total_count + self.smoothing_alpha * vocab_size
        )
        return prob

    def score(self, pattern: List[str]) -> float:
        """
        Score a pattern's probability.

        Uses log probability to avoid underflow, then converts back.

        Args:
            pattern: Sequence of step names

        Returns:
            Probability score in [0, 1] range
        """
        if not pattern:
            return 1.0

        log_prob = 0.0
        for step in pattern:
            prob = self.step_probability(step)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                return 0.0

        # Normalize by pattern length to get per-step probability
        avg_log_prob = log_prob / len(pattern)
        return math.exp(avg_log_prob)

    def save(self, path: str) -> None:
        """Save model to disk."""
        data = {
            "model_type": "unigram",
            "smoothing_alpha": self.smoothing_alpha,
            "vocabulary": list(self.vocabulary),
            "counts": dict(self.counts),
            "total_count": self.total_count,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "UnigramModel":
        """Load model from disk."""
        data = json.loads(Path(path).read_text())
        model = cls(smoothing_alpha=data["smoothing_alpha"])
        model.vocabulary = set(data["vocabulary"])
        model.counts = Counter(data["counts"])
        model.total_count = data["total_count"]
        return model


class NGramModel(BaseTypicalityModel):
    """
    Base class for n-gram models (bigram, trigram).

    Uses Kneser-Ney inspired backoff smoothing.

    Attributes:
        name: Model name
        n: N-gram size
        smoothing_alpha: Smoothing parameter
    """

    def __init__(self, n: int, smoothing_alpha: float = 0.1):
        """
        Initialize the n-gram model.

        Args:
            n: Size of n-grams
            smoothing_alpha: Smoothing parameter
        """
        self.n = n
        self.smoothing_alpha = smoothing_alpha
        self.vocabulary: set = set()
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()  # Counts of (n-1)-gram contexts

    def fit(self, patterns: List[List[str]]) -> "NGramModel":
        """
        Train the model on a corpus of patterns.

        Args:
            patterns: List of patterns

        Returns:
            self for method chaining
        """
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocabulary = set()

        for pattern in patterns:
            self.vocabulary.update(pattern)

            # Count n-grams
            ngrams = get_ngrams(pattern, self.n)
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                # Context is everything but the last element
                context = ngram[:-1]
                self.context_counts[context] += 1

        return self

    def ngram_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Get smoothed probability of an n-gram.

        P(w_n | w_1, ..., w_{n-1}) with backoff smoothing.

        Args:
            ngram: N-gram tuple

        Returns:
            Conditional probability
        """
        if len(ngram) != self.n:
            return 0.0

        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)
        ngram_count = self.ngram_counts.get(ngram, 0)

        vocab_size = len(self.vocabulary)
        if vocab_size == 0:
            return 0.0

        # Add-k smoothing
        prob = (ngram_count + self.smoothing_alpha) / (
            context_count + self.smoothing_alpha * vocab_size
        )
        return prob

    def score(self, pattern: List[str]) -> float:
        """
        Score a pattern's probability.

        Args:
            pattern: Sequence of step names

        Returns:
            Probability score in [0, 1] range
        """
        ngrams = get_ngrams(pattern, self.n)
        if not ngrams:
            return 1.0

        log_prob = 0.0
        for ngram in ngrams:
            prob = self.ngram_probability(ngram)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                return 0.0

        # Normalize by number of n-grams
        avg_log_prob = log_prob / len(ngrams)
        return math.exp(avg_log_prob)

    def save(self, path: str) -> None:
        """Save model to disk."""
        data = {
            "model_type": f"{self.n}gram",
            "n": self.n,
            "smoothing_alpha": self.smoothing_alpha,
            "vocabulary": list(self.vocabulary),
            "ngram_counts": {str(k): v for k, v in self.ngram_counts.items()},
            "context_counts": {str(k): v for k, v in self.context_counts.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "NGramModel":
        """Load model from disk."""
        data = json.loads(Path(path).read_text())
        # BigramModel and TrigramModel don't accept n parameter
        if cls is NGramModel:
            model = cls(n=data["n"], smoothing_alpha=data["smoothing_alpha"])
        else:
            model = cls(smoothing_alpha=data["smoothing_alpha"])
        model.vocabulary = set(data["vocabulary"])

        # Parse string keys back to tuples
        model.ngram_counts = Counter()
        for k, v in data["ngram_counts"].items():
            # Parse tuple from string representation
            key = eval(k) if k.startswith("(") else (k,)
            model.ngram_counts[key] = v

        model.context_counts = Counter()
        for k, v in data["context_counts"].items():
            key = eval(k) if k.startswith("(") else (k,)
            model.context_counts[key] = v

        return model


class BigramModel(NGramModel):
    """
    Bigram model: P(step_i | step_{i-1}).

    Example:
        >>> model = BigramModel()
        >>> model.fit(training_patterns)
        >>> score = model.score(['<START>', 'extract', 'analyze', '<END>'])
    """

    name = "bigram"

    def __init__(self, smoothing_alpha: float = 0.1):
        super().__init__(n=2, smoothing_alpha=smoothing_alpha)


class TrigramModel(NGramModel):
    """
    Trigram model: P(step_i | step_{i-2}, step_{i-1}).

    Example:
        >>> model = TrigramModel()
        >>> model.fit(training_patterns)
        >>> score = model.score(['<START>', 'extract', 'analyze', '<END>'])
    """

    name = "trigram"

    def __init__(self, smoothing_alpha: float = 0.1):
        super().__init__(n=3, smoothing_alpha=smoothing_alpha)


class InterpolatedModel(BaseTypicalityModel):
    """
    Interpolated n-gram model combining unigram, bigram, and trigram.

    Uses linear interpolation:
    P(w | context) = λ1 * P_unigram(w) + λ2 * P_bigram(w|w-1) + λ3 * P_trigram(w|w-2,w-1)

    Example:
        >>> model = InterpolatedModel(lambdas=(0.1, 0.3, 0.6))
        >>> model.fit(training_patterns)
        >>> score = model.score(pattern)
    """

    name = "interpolated"

    def __init__(self, lambdas: Tuple[float, float, float] = (0.1, 0.3, 0.6)):
        """
        Initialize interpolated model.

        Args:
            lambdas: Interpolation weights (unigram, bigram, trigram)
                     Must sum to 1.0
        """
        assert abs(sum(lambdas) - 1.0) < 1e-6, "Lambdas must sum to 1.0"
        self.lambdas = lambdas
        self.unigram = UnigramModel()
        self.bigram = BigramModel()
        self.trigram = TrigramModel()

    def fit(self, patterns: List[List[str]]) -> "InterpolatedModel":
        """Train all component models."""
        self.unigram.fit(patterns)
        self.bigram.fit(patterns)
        self.trigram.fit(patterns)
        return self

    def score(self, pattern: List[str]) -> float:
        """
        Score using interpolated probability.

        Args:
            pattern: Sequence of step names

        Returns:
            Interpolated probability score
        """
        if len(pattern) < 3:
            # Fall back to bigram or unigram for short patterns
            if len(pattern) < 2:
                return self.unigram.score(pattern)
            return self.bigram.score(pattern)

        log_prob = 0.0
        count = 0

        for i in range(2, len(pattern)):
            # Get context
            w = pattern[i]
            w1 = pattern[i - 1]
            w2 = pattern[i - 2]

            # Interpolate probabilities
            p_uni = self.unigram.step_probability(w)
            p_bi = self.bigram.ngram_probability((w1, w))
            p_tri = self.trigram.ngram_probability((w2, w1, w))

            p_interp = (
                self.lambdas[0] * p_uni +
                self.lambdas[1] * p_bi +
                self.lambdas[2] * p_tri
            )

            if p_interp > 0:
                log_prob += math.log(p_interp)
                count += 1
            else:
                return 0.0

        if count == 0:
            return 1.0

        avg_log_prob = log_prob / count
        return math.exp(avg_log_prob)

    def save(self, path: str) -> None:
        """Save model to disk."""
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {"lambdas": self.lambdas}
        (base_path / "config.json").write_text(json.dumps(config))

        # Save component models
        self.unigram.save(str(base_path / "unigram.json"))
        self.bigram.save(str(base_path / "bigram.json"))
        self.trigram.save(str(base_path / "trigram.json"))

    @classmethod
    def load(cls, path: str) -> "InterpolatedModel":
        """Load model from disk."""
        base_path = Path(path)

        config = json.loads((base_path / "config.json").read_text())
        model = cls(lambdas=tuple(config["lambdas"]))

        model.unigram = UnigramModel.load(str(base_path / "unigram.json"))
        model.bigram = BigramModel.load(str(base_path / "bigram.json"))
        model.trigram = TrigramModel.load(str(base_path / "trigram.json"))

        return model


class HMMModel(BaseTypicalityModel):
    """
    Hidden Markov Model over reasoning patterns.

    Uses the hmmlearn library if available, otherwise falls back to
    a simplified implementation.

    The HMM models sequences where:
    - Hidden states represent abstract reasoning states
    - Observations are the step names

    Example:
        >>> model = HMMModel(n_states=5)
        >>> model.fit(training_patterns)
        >>> score = model.score(pattern)
    """

    name = "hmm"

    def __init__(
        self,
        n_states: int = 5,
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize HMM model.

        Args:
            n_states: Number of hidden states
            n_iter: Number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.vocabulary: List[str] = []
        self.step_to_idx: Dict[str, int] = {}
        self.idx_to_step: Dict[int, str] = {}
        self._hmm = None
        self._use_hmmlearn = False

        # Try to import hmmlearn
        try:
            from hmmlearn import hmm
            self._use_hmmlearn = True
        except ImportError:
            pass

    def fit(self, patterns: List[List[str]]) -> "HMMModel":
        """
        Train the HMM on a corpus of patterns.

        Args:
            patterns: List of patterns

        Returns:
            self for method chaining
        """
        # Build vocabulary
        all_steps = set()
        for pattern in patterns:
            all_steps.update(pattern)

        self.vocabulary = sorted(all_steps)
        self.step_to_idx = {step: idx for idx, step in enumerate(self.vocabulary)}
        self.idx_to_step = {idx: step for idx, step in enumerate(self.vocabulary)}

        # Convert patterns to integer sequences
        sequences = []
        lengths = []
        for pattern in patterns:
            seq = [self.step_to_idx[step] for step in pattern]
            sequences.extend(seq)
            lengths.append(len(seq))

        X = np.array(sequences).reshape(-1, 1)
        lengths = np.array(lengths)

        if self._use_hmmlearn:
            from hmmlearn import hmm
            self._hmm = hmm.CategoricalHMM(
                n_components=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self._hmm.fit(X, lengths)
        else:
            # Simplified HMM implementation
            self._fit_simple_hmm(patterns)

        return self

    def _fit_simple_hmm(self, patterns: List[List[str]]) -> None:
        """Simplified HMM training using transition counting."""
        # Use transition counts as a simple approximation
        self._transition_probs = defaultdict(lambda: defaultdict(float))
        self._start_probs = defaultdict(float)

        total_patterns = len(patterns)

        for pattern in patterns:
            if pattern:
                # Start probability
                self._start_probs[pattern[0]] += 1.0 / total_patterns

                # Transition probabilities
                for i in range(len(pattern) - 1):
                    self._transition_probs[pattern[i]][pattern[i + 1]] += 1

        # Normalize transition probabilities
        for from_step in self._transition_probs:
            total = sum(self._transition_probs[from_step].values())
            for to_step in self._transition_probs[from_step]:
                self._transition_probs[from_step][to_step] /= total

    def score(self, pattern: List[str]) -> float:
        """
        Score a pattern using the HMM.

        Args:
            pattern: Sequence of step names

        Returns:
            Probability score in [0, 1] range
        """
        if not pattern:
            return 1.0

        # Handle unseen steps
        for step in pattern:
            if step not in self.step_to_idx:
                return 0.0  # Unseen step = unlikely pattern

        if self._use_hmmlearn and self._hmm is not None:
            # Use hmmlearn's score
            seq = [self.step_to_idx[step] for step in pattern]
            X = np.array(seq).reshape(-1, 1)
            log_prob = self._hmm.score(X)

            # Normalize by sequence length
            normalized_log_prob = log_prob / len(pattern)

            # Convert to probability (clamp to avoid overflow)
            return min(1.0, max(0.0, math.exp(normalized_log_prob)))
        else:
            # Use simple transition-based scoring
            return self._score_simple(pattern)

    def _score_simple(self, pattern: List[str]) -> float:
        """Score using simplified transition model."""
        if not pattern:
            return 1.0

        log_prob = 0.0

        # Start probability
        start_prob = self._start_probs.get(pattern[0], 1e-10)
        log_prob += math.log(start_prob)

        # Transition probabilities
        for i in range(len(pattern) - 1):
            trans_prob = self._transition_probs[pattern[i]].get(pattern[i + 1], 1e-10)
            log_prob += math.log(trans_prob)

        # Normalize
        avg_log_prob = log_prob / len(pattern)
        return math.exp(avg_log_prob)

    def save(self, path: str) -> None:
        """Save model to disk."""
        data = {
            "model_type": "hmm",
            "n_states": self.n_states,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "vocabulary": self.vocabulary,
            "use_hmmlearn": self._use_hmmlearn,
        }

        if self._use_hmmlearn and self._hmm is not None:
            # Save hmmlearn model parameters
            data["startprob"] = self._hmm.startprob_.tolist()
            data["transmat"] = self._hmm.transmat_.tolist()
            data["emissionprob"] = self._hmm.emissionprob_.tolist()
        else:
            # Save simple model parameters
            data["start_probs"] = dict(self._start_probs)
            data["transition_probs"] = {
                k: dict(v) for k, v in self._transition_probs.items()
            }

        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "HMMModel":
        """Load model from disk."""
        data = json.loads(Path(path).read_text())

        model = cls(
            n_states=data["n_states"],
            n_iter=data["n_iter"],
            random_state=data["random_state"],
        )
        model.vocabulary = data["vocabulary"]
        model.step_to_idx = {step: idx for idx, step in enumerate(model.vocabulary)}
        model.idx_to_step = {idx: step for idx, step in enumerate(model.vocabulary)}

        if data.get("use_hmmlearn") and model._use_hmmlearn:
            from hmmlearn import hmm
            model._hmm = hmm.CategoricalHMM(
                n_components=model.n_states,
                n_iter=model.n_iter,
                random_state=model.random_state,
            )
            model._hmm.startprob_ = np.array(data["startprob"])
            model._hmm.transmat_ = np.array(data["transmat"])
            model._hmm.emissionprob_ = np.array(data["emissionprob"])
        else:
            model._start_probs = defaultdict(float, data.get("start_probs", {}))
            model._transition_probs = defaultdict(lambda: defaultdict(float))
            for k, v in data.get("transition_probs", {}).items():
                model._transition_probs[k] = defaultdict(float, v)

        return model


@dataclass
class HMMGridSearchResult:
    """Result from HMM grid search."""
    best_n_states: int
    best_score: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)


class HMMGridSearch:
    """
    Grid search for optimal HMM parameters using BIC.

    Example:
        >>> search = HMMGridSearch(n_states_range=range(2, 10))
        >>> result = search.fit(patterns)
        >>> print(f"Best n_states: {result.best_n_states}")
        >>> best_model = HMMModel(n_states=result.best_n_states)
        >>> best_model.fit(patterns)
    """

    def __init__(
        self,
        n_states_range: range = range(2, 10),
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize grid search.

        Args:
            n_states_range: Range of n_states values to try
            n_iter: Number of EM iterations
            random_state: Random seed
        """
        self.n_states_range = n_states_range
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, patterns: List[List[str]]) -> HMMGridSearchResult:
        """
        Run grid search to find optimal n_states.

        Uses BIC (Bayesian Information Criterion) for model selection.

        Args:
            patterns: Training patterns

        Returns:
            HMMGridSearchResult with best parameters
        """
        results = []
        best_score = float('-inf')
        best_n_states = 2

        # Build vocabulary once
        all_steps = set()
        for pattern in patterns:
            all_steps.update(pattern)
        vocab = sorted(all_steps)
        step_to_idx = {step: idx for idx, step in enumerate(vocab)}

        # Convert to sequences
        sequences = []
        lengths = []
        for pattern in patterns:
            seq = [step_to_idx[step] for step in pattern]
            sequences.extend(seq)
            lengths.append(len(seq))

        X = np.array(sequences).reshape(-1, 1)
        lengths = np.array(lengths)
        n_samples = len(sequences)

        for n_states in self.n_states_range:
            try:
                model = HMMModel(
                    n_states=n_states,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                )
                model.fit(patterns)

                if model._use_hmmlearn and model._hmm is not None:
                    # Calculate log-likelihood
                    log_likelihood = model._hmm.score(X, lengths)

                    # Calculate BIC
                    n_params = n_states * (n_states - 1) + n_states * (len(vocab) - 1)
                    bic = -2 * log_likelihood + n_params * math.log(n_samples)

                    # We want to maximize log_likelihood (or minimize BIC)
                    score = -bic  # Negative BIC for maximization

                    results.append({
                        "n_states": n_states,
                        "log_likelihood": log_likelihood,
                        "bic": bic,
                        "score": score,
                    })

                    if score > best_score:
                        best_score = score
                        best_n_states = n_states
                else:
                    # Simple fallback - use held-out scoring
                    total_score = 0.0
                    for pattern in patterns[:10]:  # Sample of patterns
                        total_score += model.score(pattern)
                    avg_score = total_score / min(10, len(patterns))

                    results.append({
                        "n_states": n_states,
                        "avg_score": avg_score,
                        "score": avg_score,
                    })

                    if avg_score > best_score:
                        best_score = avg_score
                        best_n_states = n_states

            except Exception as e:
                results.append({
                    "n_states": n_states,
                    "error": str(e),
                    "score": float('-inf'),
                })

        return HMMGridSearchResult(
            best_n_states=best_n_states,
            best_score=best_score,
            all_results=results,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_typicality_model(
    model_type: str = "bigram",
    **kwargs,
) -> BaseTypicalityModel:
    """
    Create a typicality model by type name.

    Args:
        model_type: Type of model ("unigram", "bigram", "trigram", "hmm", "interpolated")
        **kwargs: Additional arguments for the model

    Returns:
        Instantiated model

    Example:
        >>> model = create_typicality_model("bigram", smoothing_alpha=0.1)
        >>> model.fit(patterns)
    """
    models = {
        "unigram": UnigramModel,
        "bigram": BigramModel,
        "trigram": TrigramModel,
        "hmm": HMMModel,
        "interpolated": InterpolatedModel,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](**kwargs)


def train_ensemble_models(
    patterns: List[List[str]],
) -> Dict[str, BaseTypicalityModel]:
    """
    Train all available typicality models on a corpus.

    Args:
        patterns: Training patterns

    Returns:
        Dictionary of model_name -> trained_model

    Example:
        >>> models = train_ensemble_models(patterns)
        >>> for name, model in models.items():
        ...     score = model.score(test_pattern)
        ...     print(f"{name}: {score:.4f}")
    """
    models = {}

    models["unigram"] = UnigramModel().fit(patterns)
    models["bigram"] = BigramModel().fit(patterns)
    models["trigram"] = TrigramModel().fit(patterns)

    # Try HMM if we have enough data
    if len(patterns) >= 10:
        try:
            models["hmm"] = HMMModel(n_states=5).fit(patterns)
        except Exception:
            pass  # HMM may fail if hmmlearn not available

    return models


def ensemble_score(
    models: Dict[str, BaseTypicalityModel],
    pattern: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Get ensemble score from multiple models.

    Args:
        models: Dictionary of model_name -> model
        pattern: Pattern to score
        weights: Optional weights for each model (default: equal weights)

    Returns:
        Weighted average score

    Example:
        >>> models = train_ensemble_models(patterns)
        >>> score = ensemble_score(models, test_pattern)
    """
    if not models:
        return 1.0

    if weights is None:
        weights = {name: 1.0 / len(models) for name in models}

    total_weight = sum(weights.get(name, 0) for name in models)
    if total_weight == 0:
        return 1.0

    weighted_sum = 0.0
    for name, model in models.items():
        weight = weights.get(name, 0)
        if weight > 0:
            weighted_sum += weight * model.score(pattern)

    return weighted_sum / total_weight
