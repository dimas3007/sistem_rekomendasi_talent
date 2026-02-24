"""
Auto-annotate semua pairs di labeling sheet.

Pakai S-BERT buat semantic skill matching, jadi bisa handle variasi nama skill
seperti "Event Management" vs "Event planning and coordination" — yang exact match
biasanya miss.

Dua "annotator" simulasi dengan kriteria berbeda (ketat vs longgar), hasilnya
di-resolve di resolve_labels.py.

Usage: python scripts/auto_annotate.py
"""

import json
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_NER_CV, DATA_RAW_JD, SBERT_MODEL_NAME

random.seed(42)

# Education level map
EDU_MAP = {"SD": 1, "SMP": 2, "SMA": 3, "D3": 4, "S1": 5, "S2": 6, "S3": 7}

# Major-to-department relevance mapping
MAJOR_DEPT_RELEVANCE = {
    "IT": [
        "informatika", "komputer", "ilmu komputer", "computer science", "sistem informasi",
        "information system", "teknik informatika", "software", "data science",
        "information technology", "teknologi informasi", "elektro", "multimedia",
    ],
    "Marketing": [
        "marketing", "komunikasi", "communication", "jurnalistik", "desain",
        "bisnis digital", "manajemen", "management", "dkv", "desain komunikasi visual",
        "advertising", "public relation",
    ],
    "Finance": [
        "akuntansi", "accounting", "keuangan", "finance", "ekonomi", "economics",
        "perpajakan", "taxation", "manajemen keuangan",
    ],
    "Operations": [
        "teknik industri", "industrial engineering", "manajemen", "management",
        "administrasi", "logistik", "supply chain", "bisnis", "komunikasi",
    ],
    "HR": [
        "psikologi", "psychology", "manajemen sdm", "human resource", "hukum", "law",
        "pendidikan", "education", "komunikasi",
    ],
    "Design": [
        "dkv", "desain komunikasi visual", "desain", "design", "seni", "art",
        "multimedia", "arsitektur", "informatika",
    ],
    "Product": [
        "informatika", "sistem informasi", "manajemen", "bisnis", "teknik industri",
        "komputer", "information system",
    ],
    "Business": [
        "manajemen", "management", "bisnis", "business", "ekonomi", "economics",
        "akuntansi", "accounting", "administrasi", "informatika", "sistem informasi",
    ],
    "Business Development": [
        "manajemen", "management", "bisnis", "business", "komunikasi", "marketing",
        "hubungan internasional", "public relation",
    ],
    "Creative": [
        "dkv", "desain komunikasi visual", "multimedia", "broadcasting",
        "komunikasi", "film", "seni", "jurnalistik",
    ],
}


def is_major_relevant(cv_major: str, jd_department: str) -> bool:
    """Check if CV major is relevant to JD department."""
    if not cv_major or not jd_department:
        return False
    cv_major_lower = cv_major.lower()
    keywords = MAJOR_DEPT_RELEVANCE.get(jd_department, [])
    return any(kw in cv_major_lower for kw in keywords)


def load_cv_jd_data():
    """Load all CV NER and JD data."""
    ner_dir = Path(DATA_NER_CV)
    jd_dir = Path(DATA_RAW_JD)

    cv_cache = {}
    for f in sorted(ner_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            cv_cache[f.stem] = json.load(fp)

    jd_cache = {}
    for f in sorted(jd_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            jd_cache[data["jd_id"]] = data

    return cv_cache, jd_cache


def build_skill_embeddings(cv_cache, jd_cache):
    """Collect all unique skill texts and encode with S-BERT."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print("  Loading S-BERT model...")
    model = SentenceTransformer(SBERT_MODEL_NAME)

    # Collect all unique skills
    all_skills = set()
    for cv in cv_cache.values():
        for s in (cv.get("hard_skills") or []):
            all_skills.add(s.strip())
        for s in (cv.get("soft_skills") or []):
            all_skills.add(s.strip())
    for jd in jd_cache.values():
        req = jd.get("requirements", jd)
        for s in (req.get("hard_skills") or []):
            all_skills.add(s.strip())
        for s in (req.get("soft_skills") or []):
            all_skills.add(s.strip())

    skill_list = sorted(all_skills)
    print(f"  Encoding {len(skill_list)} unique skill texts...")
    embeddings = model.encode(skill_list, normalize_embeddings=True,
                               show_progress_bar=True, batch_size=128)

    skill_embeddings = {}
    for skill, emb in zip(skill_list, embeddings):
        skill_embeddings[skill] = emb

    return skill_embeddings


def compute_semantic_skill_overlap(cv_skills, jd_skills, skill_embeddings, threshold=0.55):
    """
    Count JD skills that semantically match at least one CV skill.

    Returns:
        (match_count, avg_max_similarity)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if not cv_skills or not jd_skills:
        return 0, 0.0

    cv_embs = [skill_embeddings[s.strip()] for s in cv_skills
               if s.strip() in skill_embeddings]
    jd_embs = [skill_embeddings[s.strip()] for s in jd_skills
               if s.strip() in skill_embeddings]

    if not cv_embs or not jd_embs:
        return 0, 0.0

    cv_matrix = np.array(cv_embs)
    jd_matrix = np.array(jd_embs)

    sim_matrix = cosine_similarity(jd_matrix, cv_matrix)  # (n_jd, n_cv)
    max_sims = sim_matrix.max(axis=1)  # best CV match for each JD skill

    matches = int((max_sims >= threshold).sum())
    avg_max_sim = float(max_sims.mean())

    return matches, avg_max_sim


def precompute_semantic_overlaps(df, cv_cache, jd_cache, skill_embeddings):
    """Pre-compute semantic skill overlap for all pairs in the DataFrame."""
    print("  Computing semantic skill overlaps for all pairs...")

    hard_matches = []
    hard_avg_sims = []
    soft_matches = []
    soft_avg_sims = []

    for idx, row in df.iterrows():
        cv_id = row["cv_id"]
        jd_id = row["jd_id"]

        cv = cv_cache.get(cv_id, {})
        jd = jd_cache.get(jd_id, {})
        req = jd.get("requirements", jd)

        cv_hard = cv.get("hard_skills") or []
        jd_hard = req.get("hard_skills") or []
        cv_soft = cv.get("soft_skills") or []
        jd_soft = req.get("soft_skills") or []

        h_match, h_sim = compute_semantic_skill_overlap(cv_hard, jd_hard, skill_embeddings, threshold=0.55)
        s_match, s_sim = compute_semantic_skill_overlap(cv_soft, jd_soft, skill_embeddings, threshold=0.50)

        hard_matches.append(h_match)
        hard_avg_sims.append(h_sim)
        soft_matches.append(s_match)
        soft_avg_sims.append(s_sim)

        if (idx + 1) % 5000 == 0:
            print(f"    Processed {idx + 1}/{len(df)} pairs...")

    df["semantic_hard_matches"] = hard_matches
    df["semantic_hard_avg_sim"] = hard_avg_sims
    df["semantic_soft_matches"] = soft_matches
    df["semantic_soft_avg_sim"] = soft_avg_sims

    return df


def annotator_1_label(row) -> int:
    """
    Strict HR annotator using semantic skill matching.
    Criteria: semantic hard skill match >= 1, education level met, field relevant.
    """
    score = 0.0

    # Semantic hard skill overlap (0-4 points)
    sem_hard = row["semantic_hard_matches"]
    sem_hard_sim = row["semantic_hard_avg_sim"]
    if sem_hard >= 3:
        score += 4.0
    elif sem_hard >= 2:
        score += 3.0
    elif sem_hard >= 1:
        score += 2.0
    elif sem_hard_sim >= 0.45:
        # sedikit poin walau tidak ada skill yang cocok persis
        score += 0.5

    # Semantic soft skill overlap (0-1 point)
    sem_soft = row["semantic_soft_matches"]
    if sem_soft >= 2:
        score += 1.0
    elif sem_soft >= 1:
        score += 0.5

    # Education level (0-2 points)
    cv_edu = EDU_MAP.get(str(row.get("cv_education", "")), 0)
    jd_level = row.get("jd_level", "")
    if jd_level == "entry-level":
        min_edu = 4  # D3
    elif jd_level == "senior":
        min_edu = 5  # S1
    else:
        min_edu = 5  # S1 for mid-level

    if cv_edu >= min_edu:
        score += 2.0
    elif cv_edu == min_edu - 1:
        score += 0.5

    # Experience (0-2 points)
    cv_exp = row.get("cv_experience_years", 0) or 0
    if jd_level == "entry-level":
        req_exp = 0
    elif jd_level == "mid-level":
        req_exp = 2
    else:
        req_exp = 4

    if cv_exp >= req_exp:
        score += 2.0
    elif cv_exp >= req_exp * 0.5:
        score += 1.0

    # Major relevance (0-1.5 points)
    if is_major_relevant(str(row.get("cv_major", "")), row.get("jd_department", "")):
        score += 1.5

    # Threshold: strict annotator requires score >= 5.5
    threshold = 5.5
    noise = random.uniform(-0.3, 0.3)

    return 1 if (score + noise) >= threshold else 0


def annotator_2_label(row) -> int:
    """
    Lenient HR annotator using semantic skill matching.
    More weight on potential, transferable skills, and education background.
    """
    score = 0.0

    # Semantic hard skill overlap (0-3 points) - more lenient
    sem_hard = row["semantic_hard_matches"]
    sem_hard_sim = row["semantic_hard_avg_sim"]
    if sem_hard >= 2:
        score += 3.0
    elif sem_hard >= 1:
        score += 2.0
    elif sem_hard_sim >= 0.40:
        # semantic dekat tapi tidak ada yang match threshold
        score += 1.0
    else:
        # Check total skill count (potential)
        cv_skills = str(row.get("cv_hard_skills", ""))
        n_skills = len(cv_skills.split(",")) if cv_skills else 0
        if n_skills >= 5:
            score += 0.5

    # Semantic soft skill overlap (0-1 point)
    sem_soft = row["semantic_soft_matches"]
    if sem_soft >= 1:
        score += 1.0
    elif row["semantic_soft_avg_sim"] >= 0.40:
        score += 0.5

    # Education (0-2 points) - more lenient
    cv_edu = EDU_MAP.get(str(row.get("cv_education", "")), 0)
    jd_level = row.get("jd_level", "")
    if jd_level == "entry-level":
        min_edu = 3  # SMA ok for entry
    elif jd_level == "senior":
        min_edu = 5
    else:
        min_edu = 4  # D3 ok for mid

    if cv_edu >= min_edu:
        score += 2.0
    elif cv_edu == min_edu - 1:
        score += 1.0

    # Experience (0-2 points)
    cv_exp = row.get("cv_experience_years", 0) or 0
    if jd_level == "entry-level":
        if cv_exp >= 0:
            score += 2.0
    elif jd_level == "mid-level":
        if cv_exp >= 1.5:
            score += 2.0
        elif cv_exp >= 0.5:
            score += 1.0
    else:
        if cv_exp >= 3:
            score += 2.0
        elif cv_exp >= 2:
            score += 1.0

    # Major relevance (0-2 points) - more weight
    if is_major_relevant(str(row.get("cv_major", "")), row.get("jd_department", "")):
        score += 2.0

    # Threshold: lenient annotator at 5.0
    threshold = 5.0
    noise = random.uniform(-0.3, 0.3)

    return 1 if (score + noise) >= threshold else 0


def main():
    label_path = Path("data/features/labeling_sheet.csv")
    print(f"Loading {label_path}...")
    df = pd.read_csv(label_path)
    print(f"Total pairs: {len(df)}")

    # Load CV and JD data
    print("\nLoading CV/JD data...")
    cv_cache, jd_cache = load_cv_jd_data()
    print(f"Loaded {len(cv_cache)} CVs and {len(jd_cache)} JDs")

    # Build skill embeddings
    print("\nBuilding S-BERT skill embeddings...")
    skill_embeddings = build_skill_embeddings(cv_cache, jd_cache)

    # Pre-compute semantic skill overlaps
    print("\nComputing semantic overlaps...")
    df = precompute_semantic_overlaps(df, cv_cache, jd_cache, skill_embeddings)

    # Print semantic stats
    print(f"\n--- Semantic Skill Stats ---")
    print(f"Hard skill matches: mean={df['semantic_hard_matches'].mean():.2f}, "
          f"max={df['semantic_hard_matches'].max()}")
    print(f"Hard skill avg sim: mean={df['semantic_hard_avg_sim'].mean():.3f}")
    print(f"Soft skill matches: mean={df['semantic_soft_matches'].mean():.2f}, "
          f"max={df['semantic_soft_matches'].max()}")
    print(f"Pairs with >=1 hard match: {(df['semantic_hard_matches']>=1).sum()} "
          f"({(df['semantic_hard_matches']>=1).mean()*100:.1f}%)")

    # Annotate
    print("\nAnnotating as Annotator 1 (strict HR + semantic)...")
    df["annotator_1"] = df.apply(annotator_1_label, axis=1)

    print("Annotating as Annotator 2 (lenient HR + semantic)...")
    df["annotator_2"] = df.apply(annotator_2_label, axis=1)

    # Save
    df.to_csv(label_path, index=False)

    xlsx_path = Path("data/features/labeling_sheet.xlsx")
    try:
        df.to_excel(xlsx_path, index=False, sheet_name="Labeling")
    except PermissionError:
        print(f"WARNING: Could not write {xlsx_path} (file may be open). CSV saved OK.")

    # Stats
    agree = (df["annotator_1"] == df["annotator_2"]).sum()
    disagree = len(df) - agree

    print(f"\n{'='*50}")
    print(f"AUTO-ANNOTATION COMPLETE (Semantic)")
    print(f"{'='*50}")
    print(f"Annotator 1 (strict):  {df['annotator_1'].sum()} match / {(df['annotator_1']==0).sum()} no match ({df['annotator_1'].mean()*100:.1f}% match)")
    print(f"Annotator 2 (lenient): {df['annotator_2'].sum()} match / {(df['annotator_2']==0).sum()} no match ({df['annotator_2'].mean()*100:.1f}% match)")
    print(f"Agreement: {agree} ({agree/len(df)*100:.1f}%)")
    print(f"Disagreement: {disagree} ({disagree/len(df)*100:.1f}%)")

    # Show some examples
    print(f"\n--- Sample Annotations ---")
    sample_cols = ["jd_id", "jd_title", "cv_name", "cv_education",
                   "semantic_hard_matches", "semantic_hard_avg_sim",
                   "annotator_1", "annotator_2"]
    print("\nMatches (both agree = 1):")
    matches = df[(df["annotator_1"] == 1) & (df["annotator_2"] == 1)]
    if len(matches) > 0:
        print(matches[sample_cols].head(5).to_string(index=False))

    print("\nNo Matches (both agree = 0):")
    no_matches = df[(df["annotator_1"] == 0) & (df["annotator_2"] == 0)]
    if len(no_matches) > 0:
        print(no_matches[sample_cols].head(5).to_string(index=False))

    print("\nDisagreements:")
    disagrees = df[df["annotator_1"] != df["annotator_2"]]
    if len(disagrees) > 0:
        print(disagrees[sample_cols].head(5).to_string(index=False))

    print(f"\nNext: python scripts/resolve_labels.py")


if __name__ == "__main__":
    main()
