"""
Resolve annotations and build the final labeled dataset.

This script:
1. Reads the completed labeling sheet (with annotator_1 and annotator_2 filled)
2. Computes Cohen's Kappa for inter-annotator agreement
3. Resolves disagreements (conservative: default to 0)
4. Builds the 15-dim feature matrix using the labeled pairs
   - S-BERT embeddings are computed in batch (once per unique CV/JD) for efficiency
5. Saves the final dataset to data/features/dataset.csv

Usage:
    python scripts/resolve_labels.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_NER_CV, DATA_RAW_JD, FEATURE_COLUMNS, SBERT_MODEL_NAME
from src.utils.labeling import compute_cohen_kappa, resolve_disagreements
from src.features.structured import compute_structured_features
from src.features.semantic import json_to_narrative


def main():
    # Load labeling sheet
    label_path = Path("data/features/labeling_sheet.csv")
    if not label_path.exists():
        label_path = Path("data/features/labeling_sheet.xlsx")

    print(f"Loading labels from {label_path}...")
    if str(label_path).endswith(".xlsx"):
        df = pd.read_excel(label_path)
    else:
        df = pd.read_csv(label_path)

    # Check annotations are filled
    if df["annotator_1"].isna().all() or df["annotator_1"].eq("").all():
        print("ERROR: annotator_1 column is empty. Please fill in annotations first.")
        return

    # Convert to int
    df["annotator_1"] = pd.to_numeric(df["annotator_1"], errors="coerce").fillna(0).astype(int)
    df["annotator_2"] = pd.to_numeric(df["annotator_2"], errors="coerce").fillna(0).astype(int)

    # Step 1: Cohen's Kappa
    print("\n=== Inter-Annotator Agreement ===")
    a1 = df["annotator_1"].tolist()
    a2 = df["annotator_2"].tolist()
    kappa = compute_cohen_kappa(a1, a2)

    # Step 2: Resolve disagreements (preserve manual final_labels)
    print("\n=== Resolving Disagreements ===")
    resolved = resolve_disagreements(a1, a2)

    # Preserve manually filled final_labels, only auto-resolve empty ones
    manual_filled = pd.to_numeric(df["final_label"], errors="coerce")
    n_manual = manual_filled.notna().sum()
    if n_manual > 0:
        print(f"  Preserving {n_manual} manually annotated final_labels")
        for i in range(len(resolved)):
            if pd.notna(manual_filled.iloc[i]):
                resolved[i] = int(manual_filled.iloc[i])
    df["final_label"] = resolved

    # Save resolved labels
    df.to_csv("data/features/labeling_sheet_resolved.csv", index=False)
    print(f"Resolved labels saved to data/features/labeling_sheet_resolved.csv")

    # Step 3: Build feature matrix (optimized with batch S-BERT)
    print("\n=== Building Feature Matrix ===")

    # Load all NER and JD data
    ner_dir = Path(DATA_NER_CV)
    jd_dir = Path(DATA_RAW_JD)

    cv_cache = {}
    jd_cache = {}

    unique_cv_ids = df["cv_id"].unique()
    unique_jd_ids = df["jd_id"].unique()

    print(f"  Loading {len(unique_cv_ids)} CVs and {len(unique_jd_ids)} JDs...")
    for cv_id in unique_cv_ids:
        cv_path = ner_dir / f"{cv_id}.json"
        if cv_path.exists():
            with open(cv_path, "r", encoding="utf-8") as f:
                cv_cache[cv_id] = json.load(f)

    for jd_id in unique_jd_ids:
        jd_path = jd_dir / f"{jd_id}.json"
        if jd_path.exists():
            with open(jd_path, "r", encoding="utf-8") as f:
                jd_cache[jd_id] = json.load(f)

    # Batch S-BERT: encode all unique CVs and JDs once
    print(f"  Computing S-BERT embeddings for {len(cv_cache)} CVs + {len(jd_cache)} JDs...")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    cv_ids_ordered = sorted(cv_cache.keys())
    jd_ids_ordered = sorted(jd_cache.keys())

    cv_narratives = [json_to_narrative(cv_cache[cid], is_cv=True) for cid in cv_ids_ordered]
    jd_narratives = [json_to_narrative(jd_cache[jid], is_cv=False) for jid in jd_ids_ordered]

    print(f"  Encoding {len(cv_narratives)} CV texts...")
    cv_embeddings = sbert.encode(cv_narratives, normalize_embeddings=True,
                                  show_progress_bar=True, batch_size=64)
    print(f"  Encoding {len(jd_narratives)} JD texts...")
    jd_embeddings = sbert.encode(jd_narratives, normalize_embeddings=True,
                                  show_progress_bar=True, batch_size=64)

    # Compute full cosine similarity matrix (n_cv x n_jd)
    cosine_matrix = sklearn_cosine(cv_embeddings, jd_embeddings)
    cosine_matrix = np.clip(cosine_matrix, 0.0, 1.0)

    cv_id_to_idx = {cid: i for i, cid in enumerate(cv_ids_ordered)}
    jd_id_to_idx = {jid: i for i, jid in enumerate(jd_ids_ordered)}

    print(f"  Cosine matrix shape: {cosine_matrix.shape}")

    # Build skill embeddings for semantic skill matching
    print(f"  Building skill embeddings for semantic matching...")
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
    skill_vecs = sbert.encode(skill_list, normalize_embeddings=True,
                               show_progress_bar=True, batch_size=128)
    skill_embeddings = {skill: vec for skill, vec in zip(skill_list, skill_vecs)}

    # Helper: compute semantic skill overlap
    def semantic_skill_overlap(cv_skills, jd_skills, threshold=0.55):
        """Count JD skills semantically matched by CV skills."""
        if not cv_skills or not jd_skills:
            return 0, 0.0
        cv_embs = [skill_embeddings[s.strip()] for s in cv_skills
                   if s.strip() in skill_embeddings]
        jd_embs = [skill_embeddings[s.strip()] for s in jd_skills
                   if s.strip() in skill_embeddings]
        if not cv_embs or not jd_embs:
            return 0, 0.0
        sim = sklearn_cosine(np.array(jd_embs), np.array(cv_embs))
        max_sims = sim.max(axis=1)
        return int((max_sims >= threshold).sum()), float(max_sims.mean())

    # Build features per pair
    print(f"  Computing structured + semantic skill features for {len(df)} pairs...")
    rows = []
    errors = 0

    for idx, row in df.iterrows():
        cv_id = row["cv_id"]
        jd_id = row["jd_id"]
        label = row["final_label"]

        if cv_id not in cv_cache or jd_id not in jd_cache:
            errors += 1
            continue

        cv_entities = cv_cache[cv_id]
        jd_data = jd_cache[jd_id]

        try:
            # Semantic feature from pre-computed matrix
            ci = cv_id_to_idx[cv_id]
            ji = jd_id_to_idx[jd_id]
            cosine_sim = float(cosine_matrix[ci, ji])

            # Structured features (base)
            jd_req = jd_data.get("requirements", jd_data)
            structured = compute_structured_features(cv_entities, jd_req)

            # Override skill features with semantic matching
            cv_hard = cv_entities.get("hard_skills") or []
            jd_hard = jd_req.get("hard_skills") or []
            cv_soft = cv_entities.get("soft_skills") or []
            jd_soft = jd_req.get("soft_skills") or []

            h_count, _ = semantic_skill_overlap(cv_hard, jd_hard, threshold=0.55)
            s_count, _ = semantic_skill_overlap(cv_soft, jd_soft, threshold=0.50)

            structured["hard_skill_match_count"] = h_count
            structured["hard_skill_match_ratio"] = h_count / max(len(jd_hard), 1)
            structured["soft_skill_match_count"] = s_count
            structured["soft_skill_match_ratio"] = s_count / max(len(jd_soft), 1)

            features = {"cosine_similarity": cosine_sim}
            features.update(structured)
            features["cv_id"] = cv_id
            features["jd_id"] = jd_id
            features["label"] = label
            rows.append(features)
        except Exception as e:
            print(f"  Error for {cv_id} x {jd_id}: {e}")
            errors += 1

        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} pairs...")

    # Build final dataset
    dataset = pd.DataFrame(rows)
    feature_cols = FEATURE_COLUMNS + ["cv_id", "jd_id", "label"]
    dataset = dataset[feature_cols]

    # Save
    output_path = Path("data/features/dataset.csv")
    dataset.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"DATASET COMPLETE")
    print(f"{'='*50}")
    print(f"Total pairs: {len(dataset)}")
    print(f"Unique CVs: {dataset['cv_id'].nunique()}")
    print(f"Unique JDs: {dataset['jd_id'].nunique()}")
    print(f"Errors skipped: {errors}")
    print(f"Label distribution:")
    print(f"  Match (1):    {(dataset['label'] == 1).sum()} ({(dataset['label'] == 1).mean()*100:.1f}%)")
    print(f"  No Match (0): {(dataset['label'] == 0).sum()} ({(dataset['label'] == 0).mean()*100:.1f}%)")
    print(f"\nCohen's Kappa: {kappa:.4f}")
    print(f"\n--- NEXT STEP ---")
    print(f"Run model training: python scripts/train_model.py")


if __name__ == "__main__":
    main()
