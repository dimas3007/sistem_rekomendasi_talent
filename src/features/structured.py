"""14 structured numerical features for CV-JD matching."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Education level encoding map
EDU_MAP = {"SD": 1, "SMP": 2, "SMA": 3, "D3": 4, "S1": 5, "S2": 6, "S3": 7}

# Thresholds matching resolve_labels.py / auto_annotate.py
HARD_SKILL_THRESHOLD = 0.55
SOFT_SKILL_THRESHOLD = 0.50


def _semantic_skill_overlap(cv_skills: list, jd_skills: list, threshold: float) -> tuple:
    """
    Hitung jumlah JD skill yang memiliki pasangan semantik dari CV skill.

    Menggunakan S-BERT cosine similarity — konsisten dengan resolve_labels.py.

    Args:
        cv_skills: List of skill strings from CV.
        jd_skills: List of skill strings from JD.
        threshold: Minimum cosine similarity to count as a match.

    Returns:
        (match_count, mean_max_similarity)
    """
    if not cv_skills or not jd_skills:
        return 0, 0.0

    from src.features.semantic import _get_model
    model = _get_model()

    cv_embs = model.encode(cv_skills, normalize_embeddings=True, show_progress_bar=False)
    jd_embs = model.encode(jd_skills, normalize_embeddings=True, show_progress_bar=False)

    # sim shape: (n_jd, n_cv) — best CV match for each JD skill
    sim = sklearn_cosine(jd_embs, cv_embs)
    max_sims = sim.max(axis=1)

    match_count = int((max_sims >= threshold).sum())
    mean_sim = float(max_sims.mean())
    return match_count, mean_sim


def compute_structured_features(cv_entities: dict, jd_requirements: dict) -> dict:
    """
    Hitung 14 fitur numerik dari perbandingan CV vs JD.

    Skill matching menggunakan S-BERT semantic similarity (threshold-based),
    konsisten dengan cara dataset.csv dibangun di resolve_labels.py.

    Args:
        cv_entities: NER output from CV.
        jd_requirements: Requirements section from JD (or full JD dict).

    Returns:
        Dictionary of 14 numerical features.
    """
    # Extract skill lists (preserve original case for S-BERT encoding)
    cv_hard = list(cv_entities.get("hard_skills") or [])
    jd_hard = list(jd_requirements.get("hard_skills") or [])
    cv_soft = list(cv_entities.get("soft_skills") or [])
    jd_soft = list(jd_requirements.get("soft_skills") or [])

    # Semantic skill matching (mirrors resolve_labels.py)
    h_count, _ = _semantic_skill_overlap(cv_hard, jd_hard, HARD_SKILL_THRESHOLD)
    s_count, _ = _semantic_skill_overlap(cv_soft, jd_soft, SOFT_SKILL_THRESHOLD)

    # Language and industry: still use exact matching (short tokens, no benefit from semantics)
    cv_lang = set(s.lower() for s in (cv_entities.get("languages") or []))
    jd_lang = set(s.lower() for s in (jd_requirements.get("languages") or []))
    cv_ind = set(s.lower() for s in (cv_entities.get("industries") or []))
    jd_ind = set(s.lower() for s in (jd_requirements.get("industries") or []))

    # Experience
    cv_exp = cv_entities.get("years_experience") or 0
    jd_exp = jd_requirements.get("min_experience_years") or 0

    # Education level
    cv_edu = EDU_MAP.get(cv_entities.get("education_level"), 0)
    jd_edu = EDU_MAP.get(jd_requirements.get("education_level"), 0)

    # Certifications
    cv_certs = cv_entities.get("certifications") or []
    jd_certs = jd_requirements.get("certifications") or []

    return {
        # Hard Skills (2 features) — semantic matching
        "hard_skill_match_ratio": h_count / max(len(jd_hard), 1),
        "hard_skill_match_count": h_count,

        # Soft Skills (2 features) — semantic matching
        "soft_skill_match_ratio": s_count / max(len(jd_soft), 1),
        "soft_skill_match_count": s_count,

        # Experience (2 features)
        "experience_gap": cv_exp - jd_exp,
        "experience_meets_req": 1 if cv_exp >= jd_exp else 0,

        # Education (2 features)
        "education_match": 1 if cv_edu >= jd_edu else 0,
        "education_level_numeric": cv_edu,

        # Certifications (2 features)
        "certification_count": len(cv_certs),
        "certification_relevance": (
            len(set(c.lower() for c in cv_certs) & set(c.lower() for c in jd_certs))
            / max(len(jd_certs), 1)
        ),

        # Language (1 feature) — exact match (language names are standardized)
        "language_match_ratio": len(cv_lang & jd_lang) / max(len(jd_lang), 1),

        # Industry (1 feature) — exact match
        "industry_overlap": len(cv_ind & jd_ind) / max(len(jd_ind), 1),

        # Location (1 feature)
        "location_match": 1 if (
            (cv_entities.get("location") or "").lower()
            == (jd_requirements.get("location") or "").lower()
            and cv_entities.get("location") is not None
            and jd_requirements.get("location") is not None
        ) else 0,

        # Total Skills (1 feature)
        "total_skill_count": len(cv_hard) + len(cv_soft),
    }
