"""
Train XGBoost model and evaluate against baselines.

Usage:
    python scripts/train_model.py [--tune]

Options:
    --tune    Run full grid search (slow, ~30 min). Default uses preset params.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import FEATURE_COLUMNS, MODEL_PATH
from src.model.trainer import split_dataset, tune_hyperparameters, train_final_model, train_with_defaults
from src.model.evaluator import evaluate_classification, evaluate_ranking_per_jd, analyze_feature_importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run grid search tuning")
    args = parser.parse_args()

    # Load dataset
    dataset_path = Path("data/features/dataset.csv")
    if not dataset_path.exists():
        print("ERROR: dataset.csv not found. Run resolve_labels.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Prepare features (drop cv_id, jd_id, keep only features + label)
    feature_df = df[FEATURE_COLUMNS + ["label"]].copy()

    # Keep jd_id for ranking evaluation
    jd_ids = df["jd_id"].copy()

    # Split
    print("\n=== Splitting Data ===")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(feature_df)

    # Train
    print("\n=== Training Model ===")
    if args.tune:
        print("Running grid search (this may take 15-30 minutes)...")
        best_model, best_params = tune_hyperparameters(X_train, y_train)
        model = train_final_model(best_model, X_train, y_train, X_val, y_val)
    else:
        print("Training with default parameters...")
        model = train_with_defaults(X_train, y_train, X_val, y_val)

    # Evaluate classification
    print("\n=== Classification Evaluation (Test Set) ===")
    metrics, cm, y_proba = evaluate_classification(model, X_test, y_test)

    # Evaluate ranking per JD
    print("\n=== Ranking Evaluation ===")
    test_indices = X_test.index
    test_df_with_jd = feature_df.loc[test_indices].copy()
    test_df_with_jd["jd_id"] = jd_ids.loc[test_indices]
    jd_groups = list(test_df_with_jd.groupby("jd_id"))
    ranking_metrics = evaluate_ranking_per_jd(jd_groups, model)

    print(f"\nRanking Metrics:")
    for k, v in ranking_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Feature importance
    print("\n=== Feature Importance ===")
    importance = analyze_feature_importance(model, FEATURE_COLUMNS)

    # Save results
    results = {
        **metrics,
        **ranking_metrics,
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv("data/features/model_results.csv", index=False)

    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Model saved to: {MODEL_PATH}/xgboost_model.json")
    print(f"Results saved to: data/features/model_results.csv")
    print(f"\nKey Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
