"""14 structured numerical features for CV-JD matching."""

# Education level encoding map
EDU_MAP = {"SD": 1, "SMP": 2, "SMA": 3, "D3": 4, "S1": 5, "S2": 6, "S3": 7}


def compute_structured_features(cv_entities: dict, jd_requirements: dict) -> dict:
    """
    Hitung 14 fitur numerik dari perbandingan CV vs JD.

    Args:
        cv_entities: NER output from CV.
        jd_requirements: Requirements section from JD (or full JD dict).

    Returns:
        Dictionary of 14 numerical features.
    """
    # Extract and normalize skill sets
    cv_hard = set(s.lower() for s in (cv_entities.get("hard_skills") or []))
    jd_hard = set(s.lower() for s in (jd_requirements.get("hard_skills") or []))
    cv_soft = set(s.lower() for s in (cv_entities.get("soft_skills") or []))
    jd_soft = set(s.lower() for s in (jd_requirements.get("soft_skills") or []))
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
        # Hard Skills (2 features)
        "hard_skill_match_ratio": len(cv_hard & jd_hard) / max(len(jd_hard), 1),
        "hard_skill_match_count": len(cv_hard & jd_hard),

        # Soft Skills (2 features)
        "soft_skill_match_ratio": len(cv_soft & jd_soft) / max(len(jd_soft), 1),
        "soft_skill_match_count": len(cv_soft & jd_soft),

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

        # Language (1 feature)
        "language_match_ratio": len(cv_lang & jd_lang) / max(len(jd_lang), 1),

        # Industry (1 feature)
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
