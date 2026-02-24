"""Assemble the 15-dimensional feature vector (1 semantic + 14 structured)."""

import pandas as pd

from config.settings import FEATURE_COLUMNS
from src.features.semantic import json_to_narrative, get_semantic_score
from src.features.structured import compute_structured_features


def build_feature_vector(cv_entities: dict, jd_data: dict) -> dict:
    """
    Return 15-dim feature vector dari pasangan CV-JD.
    """
    # Path A: Semantic similarity
    cv_narrative = json_to_narrative(cv_entities, is_cv=True)
    jd_narrative = json_to_narrative(jd_data, is_cv=False)
    cosine_sim = get_semantic_score(cv_narrative, jd_narrative)

    # Path B: Structured features
    jd_req = jd_data.get("requirements", jd_data)
    structured = compute_structured_features(cv_entities, jd_req)

    # Combine
    features = {"cosine_similarity": cosine_sim}
    features.update(structured)

    return features


def build_feature_dataframe(cv_entities: dict, jd_data: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame with proper column ordering.

    Args:
        cv_entities: NER output from CV.
        jd_data: Full JD data dictionary.

    Returns:
        DataFrame with shape (1, 15).
    """
    features = build_feature_vector(cv_entities, jd_data)
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def build_dataset(cv_ner_list: list, jd_list: list, pairs: list) -> pd.DataFrame:
    """
    Build the full feature matrix from a list of (cv_index, jd_index) pairs.

    Args:
        cv_ner_list: List of CV NER dictionaries.
        jd_list: List of JD data dictionaries.
        pairs: List of (cv_idx, jd_idx) tuples.

    Returns:
        DataFrame with shape (n_pairs, 15).
    """
    rows = []
    for cv_idx, jd_idx in pairs:
        features = build_feature_vector(cv_ner_list[cv_idx], jd_list[jd_idx])
        features["cv_idx"] = cv_idx
        features["jd_idx"] = jd_idx
        rows.append(features)

    df = pd.DataFrame(rows)
    return df
