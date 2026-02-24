"""XGBoost model training, hyperparameter tuning, and data splitting."""

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

from config.settings import (
    TEST_SIZE, VAL_SIZE, RANDOM_STATE, XGBOOST_PARAMS, MODEL_PATH
)


def split_dataset(df: pd.DataFrame):
    """
    Split dataset: 70% train / 15% validation / 15% test (stratified).

    Args:
        df: DataFrame with feature columns + 'label' column.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=["label"])
    y = df["label"]

    # First: split 70% train vs 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + VAL_SIZE),
        stratify=y, random_state=RANDOM_STATE
    )

    # Second: split 30% temp into 15% val + 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train pos ratio: {y_train.mean():.3f}")
    print(f"Val pos ratio:   {y_val.mean():.3f}")
    print(f"Test pos ratio:  {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_hyperparameters(X_train, y_train):
    """
    Grid Search 5-Fold CV untuk optimasi hyperparameter.

    Returns:
        (best_model, best_params)
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

    # imbalance ratio: negatif / positif
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    print(f"Best F1-CV: {grid_search.best_score_:.4f}")
    print(f"Best Params: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_


def train_final_model(best_model, X_train, y_train, X_val, y_val):
    """
    Train model final dengan monitoring training/validation loss.

    Returns:
        Trained model.
    """
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    # Save model
    model_path = os.path.join(MODEL_PATH, "xgboost_model.json")
    best_model.save_model(model_path)
    print(f"Model saved to {model_path}")

    return best_model


def train_with_defaults(X_train, y_train, X_val, y_val):
    """
    Train with default hyperparameters from settings (skip grid search).

    Returns:
        Trained model.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    params = XGBOOST_PARAMS.copy()
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)

    model = XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    model_path = os.path.join(MODEL_PATH, "xgboost_model.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    return model
