"""Tests for typicality models."""

import pytest
import tempfile
from pathlib import Path
from ptool_framework.audit.typicality.patterns import (
    START_TOKEN,
    END_TOKEN,
    extract_pattern,
    get_ngrams,
    get_bigrams,
    get_trigrams,
    PatternStats,
    normalize_step_names,
)
from ptool_framework.audit.typicality.models import (
    UnigramModel,
    BigramModel,
    TrigramModel,
    InterpolatedModel,
    HMMModel,
    create_typicality_model,
    train_ensemble_models,
    ensemble_score,
)
import pandas as pd


@pytest.fixture
def sample_patterns():
    """Sample training patterns."""
    return [
        [START_TOKEN, "extract", "analyze", "format", END_TOKEN],
        [START_TOKEN, "extract", "validate", "analyze", "format", END_TOKEN],
        [START_TOKEN, "extract", "analyze", "format", END_TOKEN],
        [START_TOKEN, "extract", "analyze", "report", END_TOKEN],
        [START_TOKEN, "extract", "validate", "analyze", "report", END_TOKEN],
    ]


class TestPatternExtraction:
    """Tests for pattern extraction utilities."""

    def test_extract_pattern(self):
        """Test extracting pattern from DataFrame."""
        df = pd.DataFrame({
            "fn_name": ["extract", "analyze", "format"]
        })
        pattern = extract_pattern(df)
        assert pattern == [START_TOKEN, "extract", "analyze", "format", END_TOKEN]

    def test_extract_pattern_no_special_tokens(self):
        """Test extracting without special tokens."""
        df = pd.DataFrame({"fn_name": ["a", "b", "c"]})
        pattern = extract_pattern(df, add_special_tokens=False)
        assert pattern == ["a", "b", "c"]

    def test_get_ngrams(self):
        """Test n-gram extraction."""
        pattern = ["a", "b", "c", "d"]

        bigrams = get_ngrams(pattern, 2)
        assert bigrams == [("a", "b"), ("b", "c"), ("c", "d")]

        trigrams = get_ngrams(pattern, 3)
        assert trigrams == [("a", "b", "c"), ("b", "c", "d")]

    def test_get_bigrams(self):
        """Test bigram helper."""
        pattern = [START_TOKEN, "extract", "analyze", END_TOKEN]
        bigrams = get_bigrams(pattern)
        assert (START_TOKEN, "extract") in bigrams
        assert ("extract", "analyze") in bigrams

    def test_get_trigrams(self):
        """Test trigram helper."""
        pattern = [START_TOKEN, "a", "b", "c", END_TOKEN]
        trigrams = get_trigrams(pattern)
        assert len(trigrams) == 3


class TestPatternStats:
    """Tests for PatternStats class."""

    def test_from_patterns(self, sample_patterns):
        """Test creating stats from patterns."""
        stats = PatternStats.from_patterns(sample_patterns)

        assert stats.total_patterns == 5
        assert stats.vocab_size > 0
        assert "extract" in stats.vocabulary
        assert "analyze" in stats.vocabulary

    def test_most_common_steps(self, sample_patterns):
        """Test most common steps."""
        stats = PatternStats.from_patterns(sample_patterns)
        common = stats.most_common_steps(3)

        # extract and analyze should be common
        step_names = [name for name, _ in common]
        assert "extract" in step_names

    def test_transition_probability(self, sample_patterns):
        """Test transition probability."""
        stats = PatternStats.from_patterns(sample_patterns)

        # After START, extract should have high probability
        prob = stats.transition_probability(START_TOKEN, "extract")
        assert prob > 0

    def test_summary(self, sample_patterns):
        """Test summary generation."""
        stats = PatternStats.from_patterns(sample_patterns)
        summary = stats.summary()
        assert "Pattern Statistics" in summary
        assert str(stats.total_patterns) in summary


class TestNormalization:
    """Tests for pattern normalization."""

    def test_lowercase(self):
        """Test lowercase normalization."""
        patterns = [[START_TOKEN, "Extract_Data", "Analyze", END_TOKEN]]
        normalized = normalize_step_names(patterns, "lowercase")
        assert normalized[0][1] == "extract_data"

    def test_strip_prefix(self):
        """Test prefix stripping."""
        patterns = [[START_TOKEN, "ptool_extract", "tool_analyze", END_TOKEN]]
        normalized = normalize_step_names(patterns, "strip_prefix")
        assert normalized[0][1] == "extract"


class TestUnigramModel:
    """Tests for UnigramModel."""

    def test_fit(self, sample_patterns):
        """Test model fitting."""
        model = UnigramModel()
        model.fit(sample_patterns)

        assert len(model.vocabulary) > 0
        assert model.total_count > 0

    def test_score(self, sample_patterns):
        """Test scoring patterns."""
        model = UnigramModel()
        model.fit(sample_patterns)

        # Typical pattern should have reasonable score
        typical = [START_TOKEN, "extract", "analyze", END_TOKEN]
        score = model.score(typical)
        assert 0 < score <= 1

        # Empty pattern
        empty_score = model.score([])
        assert empty_score == 1.0

    def test_step_probability(self, sample_patterns):
        """Test step probability."""
        model = UnigramModel()
        model.fit(sample_patterns)

        prob = model.step_probability("extract")
        assert prob > 0

    def test_save_load(self, sample_patterns, tmp_path):
        """Test saving and loading."""
        model = UnigramModel()
        model.fit(sample_patterns)

        path = str(tmp_path / "unigram.json")
        model.save(path)

        loaded = UnigramModel.load(path)
        assert len(loaded.vocabulary) == len(model.vocabulary)

        # Scores should match
        pattern = [START_TOKEN, "extract", END_TOKEN]
        assert model.score(pattern) == loaded.score(pattern)


class TestBigramModel:
    """Tests for BigramModel."""

    def test_fit(self, sample_patterns):
        """Test model fitting."""
        model = BigramModel()
        model.fit(sample_patterns)

        assert len(model.ngram_counts) > 0
        assert len(model.context_counts) > 0

    def test_score(self, sample_patterns):
        """Test scoring."""
        model = BigramModel()
        model.fit(sample_patterns)

        typical = [START_TOKEN, "extract", "analyze", END_TOKEN]
        score = model.score(typical)
        assert 0 < score <= 1

    def test_ngram_probability(self, sample_patterns):
        """Test n-gram probability."""
        model = BigramModel()
        model.fit(sample_patterns)

        prob = model.ngram_probability((START_TOKEN, "extract"))
        assert prob > 0


class TestTrigramModel:
    """Tests for TrigramModel."""

    def test_fit_and_score(self, sample_patterns):
        """Test fitting and scoring."""
        model = TrigramModel()
        model.fit(sample_patterns)

        typical = [START_TOKEN, "extract", "analyze", "format", END_TOKEN]
        score = model.score(typical)
        assert 0 <= score <= 1


class TestInterpolatedModel:
    """Tests for InterpolatedModel."""

    def test_fit(self, sample_patterns):
        """Test fitting."""
        model = InterpolatedModel(lambdas=(0.2, 0.3, 0.5))
        model.fit(sample_patterns)

        # All component models should be fitted
        assert model.unigram.total_count > 0
        assert len(model.bigram.ngram_counts) > 0

    def test_score(self, sample_patterns):
        """Test scoring."""
        model = InterpolatedModel()
        model.fit(sample_patterns)

        pattern = [START_TOKEN, "extract", "analyze", "format", END_TOKEN]
        score = model.score(pattern)
        assert 0 < score <= 1

    def test_save_load(self, sample_patterns, tmp_path):
        """Test saving and loading."""
        model = InterpolatedModel()
        model.fit(sample_patterns)

        path = str(tmp_path / "interpolated")
        model.save(path)

        loaded = InterpolatedModel.load(path)
        pattern = [START_TOKEN, "extract", "analyze", END_TOKEN]
        assert abs(model.score(pattern) - loaded.score(pattern)) < 0.01


class TestHMMModel:
    """Tests for HMMModel."""

    def test_fit(self, sample_patterns):
        """Test fitting HMM."""
        model = HMMModel(n_states=3)
        model.fit(sample_patterns)

        assert len(model.vocabulary) > 0

    def test_score(self, sample_patterns):
        """Test scoring with HMM."""
        model = HMMModel(n_states=3)
        model.fit(sample_patterns)

        pattern = [START_TOKEN, "extract", "analyze", END_TOKEN]
        score = model.score(pattern)
        assert 0 <= score <= 1

    def test_unseen_step(self, sample_patterns):
        """Test handling unseen steps."""
        model = HMMModel(n_states=3)
        model.fit(sample_patterns)

        # Pattern with unseen step should get 0 score
        unseen = [START_TOKEN, "never_seen_before", END_TOKEN]
        score = model.score(unseen)
        assert score == 0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_typicality_model(self, sample_patterns):
        """Test model creation factory."""
        unigram = create_typicality_model("unigram")
        assert isinstance(unigram, UnigramModel)

        bigram = create_typicality_model("bigram")
        assert isinstance(bigram, BigramModel)

        with pytest.raises(ValueError):
            create_typicality_model("nonexistent")

    def test_train_ensemble_models(self, sample_patterns):
        """Test ensemble training."""
        models = train_ensemble_models(sample_patterns)

        assert "unigram" in models
        assert "bigram" in models
        assert "trigram" in models

    def test_ensemble_score(self, sample_patterns):
        """Test ensemble scoring."""
        models = train_ensemble_models(sample_patterns)
        pattern = [START_TOKEN, "extract", "analyze", END_TOKEN]

        score = ensemble_score(models, pattern)
        assert 0 <= score <= 1

        # With custom weights
        weights = {"unigram": 0.5, "bigram": 0.5}
        weighted_score = ensemble_score(models, pattern, weights=weights)
        assert 0 <= weighted_score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
