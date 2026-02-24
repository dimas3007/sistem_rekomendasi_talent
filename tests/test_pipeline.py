"""Integration tests for the full pipeline."""

import pytest
import pandas as pd
import numpy as np

from config.settings import FEATURE_COLUMNS


class TestFeatureColumns:
    def test_feature_column_count(self):
        assert len(FEATURE_COLUMNS) == 15

    def test_cosine_similarity_first(self):
        assert FEATURE_COLUMNS[0] == "cosine_similarity"

    def test_all_columns_unique(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


class TestEvaluatorMetrics:
    """Test evaluator functions with synthetic data."""

    def test_precision_at_k(self):
        from src.model.evaluator import precision_at_k

        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        p5 = precision_at_k(y_true, y_scores, k=5)
        # Top 5 by score: indices 0,1,2,3,4 => labels 1,0,1,0,1 => 3/5
        assert p5 == 0.6

    def test_ndcg_at_k_perfect(self):
        from src.model.evaluator import ndcg_at_k

        # Perfect ranking: all positives ranked first
        y_true = np.array([1, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])

        ndcg = ndcg_at_k(y_true, y_scores, k=5)
        assert ndcg == 1.0

    def test_ndcg_at_k_zero(self):
        from src.model.evaluator import ndcg_at_k

        # All zeros
        y_true = np.array([0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])

        ndcg = ndcg_at_k(y_true, y_scores, k=5)
        assert ndcg == 0.0


class TestLabeling:
    def test_cohen_kappa_perfect(self):
        from src.utils.labeling import compute_cohen_kappa
        a1 = [1, 0, 1, 0, 1]
        a2 = [1, 0, 1, 0, 1]
        kappa = compute_cohen_kappa(a1, a2)
        assert kappa == 1.0

    def test_resolve_disagreements(self):
        from src.utils.labeling import resolve_disagreements
        a1 = [1, 0, 1, 0]
        a2 = [1, 1, 1, 0]
        resolved = resolve_disagreements(a1, a2)
        assert resolved == [1, 0, 1, 0]  # Disagreements default to 0
