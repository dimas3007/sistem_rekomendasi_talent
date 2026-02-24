"""Data labeling helper tools and inter-annotator agreement."""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def compute_cohen_kappa(annotator_1: list, annotator_2: list) -> float:
    """
    Hitung Cohen's Kappa untuk inter-annotator agreement.

    Args:
        annotator_1: Labels from annotator 1.
        annotator_2: Labels from annotator 2.

    Returns:
        Kappa score.
    """
    kappa = cohen_kappa_score(annotator_1, annotator_2)
    print(f"Cohen's Kappa: {kappa:.4f}")

    if kappa >= 0.80:
        print("Interpretation: Almost perfect agreement")
    elif kappa >= 0.60:
        print("Interpretation: Substantial agreement")
    elif kappa >= 0.40:
        print("Interpretation: Moderate agreement")
    else:
        print("Interpretation: Fair or poor agreement")

    return kappa


def resolve_disagreements(annotator_1: list, annotator_2: list) -> list:
    """
    Kalau dua annotator tidak sepakat, default ke 0 (no match)

    Returns:
        List label final.
    """
    resolved = []
    disagreements = 0

    for a1, a2 in zip(annotator_1, annotator_2):
        if a1 == a2:
            resolved.append(a1)
        else:
            disagreements += 1
            resolved.append(0)  # Conservative: no match when disagreement

    print(f"Total samples: {len(resolved)}")
    print(f"Disagreements: {disagreements} ({disagreements/len(resolved)*100:.1f}%)")

    return resolved


def create_labeling_template(cv_ids: list, jd_ids: list, output_path: str):
    """
    Create a CSV template for manual labeling.

    Args:
        cv_ids: List of CV identifiers.
        jd_ids: List of JD identifiers.
        output_path: Path to save the CSV template.
    """
    rows = []
    for jd_id in jd_ids:
        for cv_id in cv_ids:
            rows.append({
                "cv_id": cv_id,
                "jd_id": jd_id,
                "annotator_1": "",
                "annotator_2": "",
                "final_label": "",
                "notes": "",
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Labeling template saved to {output_path} ({len(df)} pairs)")

    return df
