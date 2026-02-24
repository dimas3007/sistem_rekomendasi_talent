import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import SBERT_MODEL_NAME

# Load model once at import time
_model = None


def _get_model():
    """Load model waktu pertama dipakai, bukan saat import."""
    global _model
    if _model is None:
        _model = SentenceTransformer(SBERT_MODEL_NAME)
    return _model


def json_to_narrative(entities: dict, is_cv: bool = True) -> str:
    """
    Konversi JSON NER/JD menjadi teks naratif untuk S-BERT encoding.

    Args:
        entities: Dictionary of extracted entities (CV) or JD data.
        is_cv: True for CV, False for JD.

    Returns:
        Narrative text string.
    """
    parts = []

    if is_cv:
        if entities.get("education_level"):
            parts.append(f"Pendidikan {entities['education_level']}")
        if entities.get("education_major"):
            parts.append(f"jurusan {entities['education_major']}")
        if entities.get("university"):
            parts.append(f"dari {entities['university']}")
        if entities.get("hard_skills"):
            parts.append(f"Keahlian teknis: {', '.join(entities['hard_skills'])}")
        if entities.get("soft_skills"):
            parts.append(f"Kemampuan interpersonal: {', '.join(entities['soft_skills'])}")
        if entities.get("years_experience"):
            parts.append(f"Pengalaman kerja {entities['years_experience']} tahun")
        if entities.get("job_titles"):
            parts.append(f"Posisi sebelumnya: {', '.join(entities['job_titles'])}")
        if entities.get("certifications"):
            parts.append(f"Sertifikasi: {', '.join(entities['certifications'])}")
    else:
        # For JD, use requirements fields
        req = entities.get("requirements", entities)
        if entities.get("title"):
            parts.append(f"Posisi {entities['title']}")
        if entities.get("description"):
            parts.append(entities["description"][:500])
        if req.get("hard_skills"):
            parts.append(f"Dibutuhkan keahlian: {', '.join(req['hard_skills'])}")
        if req.get("soft_skills"):
            parts.append(f"Kemampuan: {', '.join(req['soft_skills'])}")
        if req.get("education_level"):
            parts.append(f"Pendidikan minimal {req['education_level']}")
        if req.get("min_experience_years"):
            parts.append(f"Pengalaman minimal {req['min_experience_years']} tahun")

    return ". ".join(parts)


def get_embedding(text: str) -> np.ndarray:
    """Get S-BERT embedding for a single text."""
    model = _get_model()
    return model.encode(text, normalize_embeddings=True, show_progress_bar=False)


def get_semantic_score(cv_text: str, jd_text: str) -> float:
    """
    Hitung cosine similarity antara embedding CV dan JD.

    Args:
        cv_text: Narrative text for CV.
        jd_text: Narrative text for JD.

    Returns:
        Cosine similarity score clipped to [0, 1].
    """
    model = _get_model()
    embeddings = model.encode(
        [cv_text, jd_text],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(score, 0.0, 1.0))
