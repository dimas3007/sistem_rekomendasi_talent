"""Model evaluation: classification metrics and ranking metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)


def evaluate_classification(model, X_test, y_test):
    """
    Evaluasi metrik klasifikasi biner pada test set.

    Returns:
        (metrics_dict, confusion_matrix, y_proba)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No Match", "Match"]))
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nAUC-ROC: {metrics['auc_roc']:.4f}")

    return metrics, cm, y_proba


def get_roc_curve(y_test, y_proba):
    """Return FPR, TPR, thresholds for ROC curve plotting."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    return fpr, tpr, thresholds


def precision_at_k(y_true, y_scores, k=5):
    """Hitung Precision@K."""
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    top_k_labels = y_true[top_k_indices]
    return np.sum(top_k_labels) / k


def dcg_at_k(y_true, y_scores, k=5):
    """Hitung DCG@K."""
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    top_k_labels = y_true[top_k_indices]
    gains = top_k_labels / np.log2(np.arange(2, k + 2))
    return np.sum(gains)


def ndcg_at_k(y_true, y_scores, k=5):
    """Hitung NDCG@K."""
    dcg = dcg_at_k(y_true, y_scores, k)
    ideal_labels = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal_labels / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking_per_jd(jd_groups, model):
    """
    Evaluasi ranking metrics per JD, lalu rata-ratakan.

    Args:
        jd_groups: Iterable of (jd_id, group_dataframe) tuples.
        model: Trained XGBoost model.

    Returns:
        Dictionary of averaged ranking metrics.
    """
    all_p5, all_p10, all_ndcg5, all_ndcg10 = [], [], [], []

    for jd_id, group in jd_groups:
        # Keep only numeric feature columns + label
        non_feature_cols = [c for c in group.columns if c in ("label", "jd_id", "cv_id")]
        X_group = group.drop(columns=non_feature_cols)
        y_group = group["label"].values
        scores = model.predict_proba(X_group)[:, 1]

        if len(y_group) >= 5:
            all_p5.append(precision_at_k(y_group, scores, k=5))
            all_ndcg5.append(ndcg_at_k(y_group, scores, k=5))
        if len(y_group) >= 10:
            all_p10.append(precision_at_k(y_group, scores, k=10))
            all_ndcg10.append(ndcg_at_k(y_group, scores, k=10))

    return {
        "P@5": np.mean(all_p5) if all_p5 else 0.0,
        "P@10": np.mean(all_p10) if all_p10 else 0.0,
        "NDCG@5": np.mean(all_ndcg5) if all_ndcg5 else 0.0,
        "NDCG@10": np.mean(all_ndcg10) if all_ndcg10 else 0.0,
    }


def analyze_feature_importance(model, feature_names):
    """
    Analisis feature importance dari model XGBoost.

    Returns:
        Dictionary mapping feature_name -> importance_score.
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    print("=== Feature Importance Ranking ===")
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"#{rank:2d} {feature_names[idx]:30s} {importance[idx]:.4f} ({importance[idx]*100:.2f}%)")

    # Semantic vs Structured contribution
    semantic_imp = importance[0]  # cosine_similarity
    structured_imp = sum(importance[1:])
    print(f"\nSemantik (1 fitur):     {semantic_imp*100:.2f}%")
    print(f"Terstruktur (14 fitur): {structured_imp*100:.2f}%")

    return dict(zip(feature_names, importance))
