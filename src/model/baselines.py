"""4 baseline models for comparison with the proposed S-BERT + XGBoost model."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from xgboost import XGBClassifier
from scipy import stats


def baseline_1_tfidf_cosine(cv_texts, jd_texts):
    """
    Baseline 1: TF-IDF + Cosine Similarity.

    Args:
        cv_texts: List of CV narrative texts.
        jd_texts: List of JD narrative texts.

    Returns:
        Cosine similarity matrix (n_cv x n_jd).
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    all_texts = cv_texts + jd_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    cv_vectors = tfidf_matrix[:len(cv_texts)]
    jd_vectors = tfidf_matrix[len(cv_texts):]

    scores = sklearn_cosine(cv_vectors, jd_vectors)
    return scores, vectorizer


def baseline_2_tfidf_lr(X_tfidf_train, y_train, X_tfidf_test):
    """
    Baseline 2: TF-IDF features + Logistic Regression.

    Returns:
        (y_pred, y_proba)
    """
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tfidf_train, y_train)
    return lr.predict(X_tfidf_test), lr.predict_proba(X_tfidf_test)[:, 1], lr


def baseline_3_sbert_cosine(cosine_scores, threshold=0.6):
    """
    Baseline 3: S-BERT Cosine Similarity Only (tanpa XGBoost).

    Returns:
        Binary predictions based on threshold.
    """
    return (np.array(cosine_scores) >= threshold).astype(int)


def baseline_4_structured_xgb(X_structured_train, y_train, X_structured_test):
    """
    Baseline 4: Structured Features + XGBoost (tanpa fitur semantik).

    Returns:
        (y_pred, y_proba, model)
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=n_neg / max(n_pos, 1),
    )
    model.fit(X_structured_train, y_train)
    return model.predict(X_structured_test), model.predict_proba(X_structured_test)[:, 1], model


def paired_t_test(proposed_scores, baseline_scores, alpha=0.05):
    """
    Paired t-test antara model usulan vs baseline (5-Fold CV scores).

    Args:
        proposed_scores: Array of proposed model scores per fold.
        baseline_scores: Array of baseline model scores per fold.
        alpha: Significance level.

    Returns:
        (t_stat, p_value, is_significant)
    """
    t_stat, p_value = stats.ttest_rel(proposed_scores, baseline_scores)

    is_significant = p_value < alpha
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Significant at alpha={alpha}: {'Yes' if is_significant else 'No'}")

    return t_stat, p_value, is_significant
