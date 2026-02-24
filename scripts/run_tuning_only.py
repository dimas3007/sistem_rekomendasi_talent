"""
run_tuning_only.py
==================
Standalone hyperparameter tuning script.
Runs GridSearchCV with 5-fold CV and saves top-5 configurations.
"""

import sys, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE, VAL_SIZE

DATA_DIR = ROOT / "data" / "features"


def main():
    print("=" * 60)
    print("  HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / "dataset.csv")
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + VAL_SIZE),
        stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        stratify=y_temp, random_state=RANDOM_STATE
    )

    # Combine train+val for tuning
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])

    n_neg = (y_tv == 0).sum()
    n_pos = (y_tv == 1).sum()
    spw = n_neg / max(n_pos, 1)

    print(f"\nDataset: {len(df)} rows")
    print(f"Train+Val for tuning: {len(X_tv)} rows")
    print(f"scale_pos_weight: {spw:.4f}")

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"Total combinations: {total_combos}")
    print(f"With 5-fold CV: {total_combos * 5} fits")
    print(f"\nStarting grid search... (this may take 10-30 minutes)")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
    )

    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    grid.fit(X_tv, y_tv)

    print(f"\nFinished: {datetime.now().strftime('%H:%M:%S')}")

    # Extract top-5
    results = pd.DataFrame(grid.cv_results_)
    top5 = results.sort_values("rank_test_score").head(5)

    rows = []
    print(f"\n{'='*60}")
    print(f"  TOP-5 HYPERPARAMETER CONFIGURATIONS")
    print(f"{'='*60}")

    for _, row in top5.iterrows():
        config = {
            "rank": int(row["rank_test_score"]),
            "n_estimators": int(row["param_n_estimators"]),
            "max_depth": int(row["param_max_depth"]),
            "learning_rate": row["param_learning_rate"],
            "subsample": row["param_subsample"],
            "colsample_bytree": row["param_colsample_bytree"],
            "f1_cv_mean": row["mean_test_score"],
            "f1_cv_std": row["std_test_score"],
        }
        rows.append(config)
        print(f"\n  Rank {config['rank']}:")
        print(f"    n_estimators={config['n_estimators']}, "
              f"max_depth={config['max_depth']}, "
              f"lr={config['learning_rate']}")
        print(f"    subsample={config['subsample']}, "
              f"colsample_bytree={config['colsample_bytree']}")
        print(f"    F1-CV: {config['f1_cv_mean']:.4f} +/- {config['f1_cv_std']:.4f}")

    # Save results
    top5_df = pd.DataFrame(rows)
    out_path = DATA_DIR / "tuning_top5.csv"
    top5_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also save full grid search results
    full_path = DATA_DIR / "tuning_full_results.csv"
    results_slim = results[
        ["rank_test_score", "mean_test_score", "std_test_score",
         "param_n_estimators", "param_max_depth", "param_learning_rate",
         "param_subsample", "param_colsample_bytree", "mean_fit_time"]
    ].sort_values("rank_test_score")
    results_slim.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")

    # Print thesis-ready table
    print(f"\n{'='*60}")
    print("  THESIS TABLE (Tabel 4.8 / 4.11)")
    print(f"{'='*60}")
    print("| Rank | n_est | max_depth | lr | subsample | colsample | F1-CV Mean | F1-CV Std |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['rank']} | {r['n_estimators']} | {r['max_depth']} | "
              f"{r['learning_rate']} | {r['subsample']} | {r['colsample_bytree']} | "
              f"{r['f1_cv_mean']:.4f} | +/-{r['f1_cv_std']:.4f} |")

    # Compare best config vs default
    best = rows[0]
    print(f"\n  Best config: n_est={best['n_estimators']}, depth={best['max_depth']}, "
          f"lr={best['learning_rate']}")
    print(f"  Default config: n_est=300, depth=5, lr=0.1")
    print(f"  Best F1-CV: {best['f1_cv_mean']:.4f}")


if __name__ == "__main__":
    main()
