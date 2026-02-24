"""Visualization utilities for model evaluation and data analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix heatmap."""
    if labels is None:
        labels = ["No Match", "Match"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curve(y_test, y_proba, title="ROC Curve", save_path=None):
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_feature_importance(importance_dict, top_n=15, title="Feature Importance", save_path=None):
    """Plot horizontal bar chart of feature importance."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(features))
    ax.barh(y_pos, scores, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_curves(evals_result, title="Training & Validation Loss", save_path=None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    train_loss = evals_result["validation_0"]["logloss"]
    val_loss = evals_result["validation_1"]["logloss"]
    epochs = range(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, label="Training Loss", color="blue")
    ax.plot(epochs, val_loss, label="Validation Loss", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_comparison(model_names, f1_scores, title="Model Comparison (F1-Score)", save_path=None):
    """Plot bar chart comparing F1-scores across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2ecc71" if i == len(model_names) - 1 else "#3498db" for i in range(len(model_names))]
    bars = ax.bar(model_names, f1_scores, color=colors)

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{score:.4f}", ha="center", va="bottom", fontsize=10
        )

    ax.set_ylabel("F1-Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_correlation_heatmap(df, title="Feature Correlation Matrix", save_path=None):
    """Plot correlation heatmap of features."""
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, ax=ax, linewidths=0.5
    )
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_score_distribution(scores, labels, title="Score Distribution by Class", save_path=None):
    """Plot histogram/boxplot of scores per class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    match_scores = scores[labels == 1]
    no_match_scores = scores[labels == 0]

    axes[0].hist(no_match_scores, bins=30, alpha=0.7, label="No Match", color="#e74c3c")
    axes[0].hist(match_scores, bins=30, alpha=0.7, label="Match", color="#2ecc71")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distribution")
    axes[0].legend()

    # Boxplot
    data = pd.DataFrame({"Score": scores, "Label": ["Match" if l == 1 else "No Match" for l in labels]})
    sns.boxplot(x="Label", y="Score", data=data, ax=axes[1], palette={"Match": "#2ecc71", "No Match": "#e74c3c"})
    axes[1].set_title("Score Boxplot by Class")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_radar_chart_plotly(candidate_name, feature_values, feature_names):
    """
    Create a radar/spider chart for a single candidate using Plotly.

    Returns:
        Plotly figure object.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=feature_values,
        theta=feature_names,
        fill="toself",
        name=candidate_name,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f"Profil Kandidat: {candidate_name}",
    )
    return fig


def plot_top_candidates_plotly(results_df, top_n=10):
    """
    Create bar chart of top N candidates using Plotly.

    Returns:
        Plotly figure object.
    """
    top_df = results_df.head(top_n)

    fig = px.bar(
        top_df,
        x="Skor XGBoost",
        y="CV",
        orientation="h",
        title=f"Top {top_n} Kandidat",
        color="Skor XGBoost",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))

    return fig
