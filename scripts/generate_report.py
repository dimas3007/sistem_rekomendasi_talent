"""
generate_report.py
==================
Buat laporan Markdown dari data sistem — statistik CV, JD, hasil matching.
Output ke reports/thesis_report_[tanggal].md

Usage: python scripts/generate_report.py [--top 15] [--skip-matching]
"""

import sys, os, json, glob, argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_RAW_CV, DATA_RAW_JD, DATA_NER_CV,
    DATA_FEATURES, MODEL_PATH, FEATURE_COLUMNS,
    SBERT_MODEL_NAME, GEMINI_MODEL, XGBOOST_PARAMS,
)
from src.features.structured import compute_structured_features, EDU_MAP

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
REPORT_DIR = ROOT / "reports"
EDU_ORDER = ["SD", "SMP", "SMA", "D3", "S1", "S2", "S3"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, key, default=""):
    v = d.get(key)
    return v if v is not None else default


def fmt_pct(val, decimals=2):
    return f"{val * 100:.{decimals}f}%"


def fmt_float(val, decimals=4):
    return f"{val:.{decimals}f}"


def md_table(headers, rows, align=None):
    """Build a Markdown table string."""
    if align is None:
        align = ["---"] * len(headers)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(align) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ===========================================================================
# Data Loading
# ===========================================================================
def load_cv_ner_files():
    """Load all NER JSON files from data/processed/cv_ner/."""
    ner_dir = Path(DATA_NER_CV)
    files = sorted(ner_dir.glob("*.json"))
    data = {}
    for f in files:
        try:
            data[f.stem] = load_json(f)
        except Exception:
            pass
    return data


def load_jd_files():
    """Load all JD JSON files from data/raw/jds/."""
    jd_dir = Path(DATA_RAW_JD)
    files = sorted(jd_dir.glob("*.json"))
    data = {}
    for f in files:
        try:
            data[f.stem] = load_json(f)
        except Exception:
            pass
    return data


def count_raw_cvs():
    """Count PDF files in data/raw/cvs/."""
    cv_dir = Path(DATA_RAW_CV)
    return len(list(cv_dir.glob("*.pdf")))


def load_model():
    """Load trained XGBoost model."""
    from xgboost import XGBClassifier
    model_file = Path(MODEL_PATH) / "xgboost_model.json"
    if not model_file.exists():
        return None
    model = XGBClassifier()
    model.load_model(str(model_file))
    return model


def load_dataset():
    """Load feature dataset from data/features/dataset.csv."""
    ds_path = Path(DATA_FEATURES) / "dataset.csv"
    if not ds_path.exists():
        return None
    return pd.read_csv(ds_path)


def load_model_results():
    """Load saved model results."""
    res_path = Path(DATA_FEATURES) / "model_results.csv"
    if not res_path.exists():
        return None
    return pd.read_csv(res_path).iloc[0].to_dict()


# ===========================================================================
# Section Builders
# ===========================================================================
def section_system_overview(n_raw_cv, n_ner, n_jd, model, model_results):
    lines = []
    lines.append("## 1. System Overview\n")
    model_status = "Trained" if model is not None else "Not trained"
    model_file = Path(MODEL_PATH) / "xgboost_model.json"
    model_size = f"{model_file.stat().st_size / 1024:.1f} KB" if model_file.exists() else "-"

    rows = [
        ["Total CV (PDF)", str(n_raw_cv)],
        ["CV Processed (NER)", str(n_ner)],
        ["NER Success Rate", fmt_pct(n_ner / max(n_raw_cv, 1))],
        ["Total Job Descriptions", str(n_jd)],
        ["Model Status", model_status],
        ["Model File Size", model_size],
        ["Feature Dimensions", "15 (1 semantic + 14 structured)"],
        ["Semantic Model", SBERT_MODEL_NAME],
        ["NER Model", GEMINI_MODEL],
        ["Classifier", f"XGBoost (n_estimators={XGBOOST_PARAMS['n_estimators']}, "
                        f"max_depth={XGBOOST_PARAMS['max_depth']}, "
                        f"lr={XGBOOST_PARAMS['learning_rate']})"],
    ]
    lines.append(md_table(["Parameter", "Value"], rows, [":---", ":---"]))

    if model_results:
        lines.append("\n### Model Performance Summary\n")
        perf_rows = [
            ["Accuracy", fmt_float(model_results.get("accuracy", 0))],
            ["Precision", fmt_float(model_results.get("precision", 0))],
            ["Recall", fmt_float(model_results.get("recall", 0))],
            ["F1-Score", fmt_float(model_results.get("f1_score", 0))],
            ["AUC-ROC", fmt_float(model_results.get("auc_roc", 0))],
            ["P@5", fmt_float(model_results.get("P@5", 0))],
            ["P@10", fmt_float(model_results.get("P@10", 0))],
            ["NDCG@5", fmt_float(model_results.get("NDCG@5", 0))],
            ["NDCG@10", fmt_float(model_results.get("NDCG@10", 0))],
        ]
        lines.append(md_table(["Metric", "Score"], perf_rows, [":---", ":---"]))

    return "\n".join(lines)


def section_ner_distribution(cv_ner_data):
    lines = []
    lines.append("## 2. NER Data Distribution\n")
    n_total = len(cv_ner_data)
    lines.append(f"Total CVs analyzed: **{n_total}**\n")

    # --- 2.1 Education Level ---
    lines.append("### 2.1 Education Level Distribution\n")
    edu_counter = Counter()
    edu_null = 0
    for cv in cv_ner_data.values():
        edu = cv.get("education_level")
        if edu and edu in EDU_MAP:
            edu_counter[edu] += 1
        else:
            edu_null += 1

    edu_rows = []
    for level in EDU_ORDER:
        count = edu_counter.get(level, 0)
        pct = count / max(n_total, 1)
        bar = "█" * int(pct * 40)
        edu_rows.append([level, str(count), fmt_pct(pct), bar])
    edu_rows.append(["*Unknown/Null*", str(edu_null),
                     fmt_pct(edu_null / max(n_total, 1)), ""])
    lines.append(md_table(
        ["Education Level", "Count", "Percentage", "Distribution"],
        edu_rows, [":---", "---:", "---:", ":---"]))

    # --- 2.2 Experience Distribution ---
    lines.append("\n### 2.2 Experience Distribution\n")
    experiences = []
    exp_null = 0
    for cv in cv_ner_data.values():
        exp = cv.get("years_experience")
        if exp is not None and exp > 0:
            experiences.append(float(exp))
        else:
            exp_null += 1

    if experiences:
        exp_arr = np.array(experiences)
        bins = [0, 1, 2, 3, 5, 10, float("inf")]
        labels = ["< 1 tahun", "1-2 tahun", "2-3 tahun", "3-5 tahun",
                  "5-10 tahun", "> 10 tahun"]
        counts, _ = np.histogram(exp_arr, bins=bins)
        exp_rows = []
        for label, count in zip(labels, counts):
            pct = count / max(n_total, 1)
            bar = "█" * int(pct * 40)
            exp_rows.append([label, str(count), fmt_pct(pct), bar])
        exp_rows.append(["*No experience / Null*", str(exp_null),
                         fmt_pct(exp_null / max(n_total, 1)), ""])
        lines.append(md_table(
            ["Range", "Count", "Percentage", "Distribution"],
            exp_rows, [":---", "---:", "---:", ":---"]))

        lines.append(f"\n**Statistics:** Mean = {exp_arr.mean():.2f} years, "
                     f"Median = {np.median(exp_arr):.2f} years, "
                     f"Min = {exp_arr.min():.2f}, Max = {exp_arr.max():.2f}")
    else:
        lines.append("*No experience data available.*")

    # --- 2.3 Skills per CV ---
    lines.append("\n### 2.3 Average Skills per CV\n")
    hard_counts = []
    soft_counts = []
    total_skills = []
    for cv in cv_ner_data.values():
        h = len(cv.get("hard_skills") or [])
        s = len(cv.get("soft_skills") or [])
        hard_counts.append(h)
        soft_counts.append(s)
        total_skills.append(h + s)

    hard_arr = np.array(hard_counts)
    soft_arr = np.array(soft_counts)
    total_arr = np.array(total_skills)

    skill_rows = [
        ["Hard Skills", fmt_float(hard_arr.mean(), 2),
         fmt_float(np.median(hard_arr), 1),
         str(int(hard_arr.min())), str(int(hard_arr.max())),
         str(int(hard_arr.sum()))],
        ["Soft Skills", fmt_float(soft_arr.mean(), 2),
         fmt_float(np.median(soft_arr), 1),
         str(int(soft_arr.min())), str(int(soft_arr.max())),
         str(int(soft_arr.sum()))],
        ["**Total Skills**", f"**{fmt_float(total_arr.mean(), 2)}**",
         f"**{fmt_float(np.median(total_arr), 1)}**",
         str(int(total_arr.min())), str(int(total_arr.max())),
         str(int(total_arr.sum()))],
    ]
    lines.append(md_table(
        ["Skill Type", "Mean", "Median", "Min", "Max", "Total"],
        skill_rows, [":---", "---:", "---:", "---:", "---:", "---:"]))

    # --- 2.4 Top Hard Skills ---
    lines.append("\n### 2.4 Top 20 Hard Skills (across all CVs)\n")
    all_hard = Counter()
    for cv in cv_ner_data.values():
        for s in (cv.get("hard_skills") or []):
            all_hard[s.strip()] += 1
    top_hard = all_hard.most_common(20)
    top_rows = []
    for rank, (skill, cnt) in enumerate(top_hard, 1):
        top_rows.append([str(rank), skill, str(cnt),
                         fmt_pct(cnt / max(n_total, 1))])
    lines.append(md_table(
        ["#", "Skill", "Count", "% of CVs"],
        top_rows, ["---:", ":---", "---:", "---:"]))

    # --- 2.5 Top Soft Skills ---
    lines.append("\n### 2.5 Top 15 Soft Skills (across all CVs)\n")
    all_soft = Counter()
    for cv in cv_ner_data.values():
        for s in (cv.get("soft_skills") or []):
            all_soft[s.strip()] += 1
    top_soft = all_soft.most_common(15)
    soft_rows = []
    for rank, (skill, cnt) in enumerate(top_soft, 1):
        soft_rows.append([str(rank), skill, str(cnt),
                          fmt_pct(cnt / max(n_total, 1))])
    lines.append(md_table(
        ["#", "Skill", "Count", "% of CVs"],
        soft_rows, ["---:", ":---", "---:", "---:"]))

    # --- 2.6 Location Distribution ---
    lines.append("\n### 2.6 Location Distribution (Top 15)\n")
    loc_counter = Counter()
    loc_null = 0
    for cv in cv_ner_data.values():
        loc = cv.get("location")
        if loc:
            loc_counter[loc.strip()] += 1
        else:
            loc_null += 1
    top_loc = loc_counter.most_common(15)
    loc_rows = []
    for rank, (loc, cnt) in enumerate(top_loc, 1):
        loc_rows.append([str(rank), loc, str(cnt), fmt_pct(cnt / max(n_total, 1))])
    loc_rows.append(["", "*Unknown/Null*", str(loc_null),
                     fmt_pct(loc_null / max(n_total, 1))])
    lines.append(md_table(
        ["#", "Location", "Count", "% of CVs"],
        loc_rows, ["---:", ":---", "---:", "---:"]))

    return "\n".join(lines)


def section_jd_overview(jd_data):
    lines = []
    lines.append("## 3. Job Description Overview\n")
    lines.append(f"Total JDs: **{len(jd_data)}**\n")

    # Summary table
    rows = []
    dept_counter = Counter()
    level_counter = Counter()
    for jd_id in sorted(jd_data.keys()):
        jd = jd_data[jd_id]
        req = jd.get("requirements", {})
        dept = jd.get("department", "-")
        dept_counter[dept] += 1
        level = jd.get("level", "-")
        level_counter[level] += 1
        n_hard = len(req.get("hard_skills") or [])
        n_soft = len(req.get("soft_skills") or [])
        rows.append([
            jd.get("jd_id", jd_id),
            jd.get("title", "-"),
            dept,
            level,
            req.get("education_level", "-"),
            str(req.get("min_experience_years", 0)),
            str(n_hard),
            str(n_soft),
            safe_get(req, "location", "-"),
        ])

    lines.append(md_table(
        ["JD ID", "Title", "Dept", "Level", "Min Edu",
         "Min Exp (yr)", "Hard Skills", "Soft Skills", "Location"],
        rows,
        [":---", ":---", ":---", ":---", ":---",
         "---:", "---:", "---:", ":---"]))

    # Department distribution
    lines.append("\n### 3.1 Department Distribution\n")
    dept_rows = []
    for dept, cnt in dept_counter.most_common():
        dept_rows.append([dept, str(cnt), fmt_pct(cnt / max(len(jd_data), 1))])
    lines.append(md_table(
        ["Department", "Count", "Percentage"],
        dept_rows, [":---", "---:", "---:"]))

    # Level distribution
    lines.append("\n### 3.2 Level Distribution\n")
    lvl_rows = []
    for lvl, cnt in level_counter.most_common():
        lvl_rows.append([lvl, str(cnt), fmt_pct(cnt / max(len(jd_data), 1))])
    lines.append(md_table(
        ["Level", "Count", "Percentage"],
        lvl_rows, [":---", "---:", "---:"]))

    return "\n".join(lines)


def section_matching_per_jd(cv_ner_data, jd_data, model, top_n=10):
    """Run matching for all CVs against each JD and show top-N."""
    lines = []
    lines.append("## 4. Matching Results per Job Description\n")

    if model is None:
        lines.append("*Model not available. Skipping matching.*")
        return "\n".join(lines)

    lines.append(f"For each JD, the top **{top_n}** candidates are shown "
                 f"ranked by XGBoost prediction score.\n")
    lines.append(f"Total CVs evaluated per JD: **{len(cv_ner_data)}**\n")

    # ----- Pre-compute S-BERT embeddings in batch for speed -----
    print("  Loading S-BERT model...")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from src.features.semantic import json_to_narrative

    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    cv_ids = list(cv_ner_data.keys())
    jd_ids = sorted(jd_data.keys())

    print(f"  Encoding {len(cv_ids)} CV narratives...")
    cv_narratives = [json_to_narrative(cv_ner_data[c], is_cv=True) for c in cv_ids]
    cv_embeddings = sbert.encode(cv_narratives, show_progress_bar=True,
                                 normalize_embeddings=True, batch_size=64)

    print(f"  Encoding {len(jd_ids)} JD narratives...")
    jd_narratives = [json_to_narrative(jd_data[j], is_cv=False) for j in jd_ids]
    jd_embeddings = sbert.encode(jd_narratives, show_progress_bar=False,
                                 normalize_embeddings=True)

    # Cosine similarity matrix: (n_cv, n_jd)
    print("  Computing cosine similarity matrix...")
    sim_matrix = cos_sim(cv_embeddings, jd_embeddings)

    # ----- Per-JD matching -----
    all_matching_rows = []  # for CSV export

    for jd_idx, jd_id in enumerate(jd_ids):
        jd = jd_data[jd_id]
        jd_req = jd.get("requirements", jd)
        title = jd.get("title", jd_id)
        dept = jd.get("department", "-")
        level = jd.get("level", "-")

        print(f"  Matching: {jd_id} - {title}...")

        # Build features for all CVs against this JD
        feature_rows = []
        for cv_idx, cv_id in enumerate(cv_ids):
            cv = cv_ner_data[cv_id]
            structured = compute_structured_features(cv, jd_req)
            features = {"cosine_similarity": float(sim_matrix[cv_idx, jd_idx])}
            features.update(structured)
            features["_cv_id"] = cv_id
            feature_rows.append(features)

        df = pd.DataFrame(feature_rows)
        X = df[FEATURE_COLUMNS].values
        scores = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        df["score"] = scores
        df["prediction"] = preds
        df["_jd_id"] = jd_id

        # Sort by score descending
        df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

        # Collect for CSV export
        for _, row in df_sorted.iterrows():
            all_matching_rows.append({
                "jd_id": jd_id,
                "jd_title": title,
                "cv_id": row["_cv_id"],
                "score": round(row["score"], 4),
                "prediction": "Match" if row["prediction"] == 1 else "No Match",
                "cosine_similarity": round(row["cosine_similarity"], 4),
                "hard_skill_match_ratio": round(row["hard_skill_match_ratio"], 4),
                "soft_skill_match_ratio": round(row["soft_skill_match_ratio"], 4),
                "education_match": int(row["education_match"]),
                "experience_meets_req": int(row["experience_meets_req"]),
            })

        # Summary stats for this JD
        n_match = int((df_sorted["prediction"] == 1).sum())
        n_no_match = int((df_sorted["prediction"] == 0).sum())
        avg_score = df_sorted["score"].mean()

        lines.append(f"### 4.{jd_idx + 1} {jd_id}: {title}\n")
        lines.append(f"- **Department:** {dept} | **Level:** {level}")
        lines.append(f"- **Predicted Match:** {n_match} | "
                     f"**No Match:** {n_no_match} | "
                     f"**Avg Score:** {avg_score:.4f}\n")

        # Top-N table
        top = df_sorted.head(top_n)
        t_rows = []
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            cv_name = safe_get(cv_ner_data[r["_cv_id"]], "name", r["_cv_id"])
            status = "Match" if r["prediction"] == 1 else "No Match"
            t_rows.append([
                str(rank),
                cv_name,
                r["_cv_id"][:40] + ("..." if len(r["_cv_id"]) > 40 else ""),
                fmt_float(r["score"]),
                f"**{status}**",
                fmt_float(r["cosine_similarity"]),
                fmt_pct(r["hard_skill_match_ratio"]),
                fmt_pct(r["soft_skill_match_ratio"]),
                "Yes" if r["education_match"] == 1 else "No",
                "Yes" if r["experience_meets_req"] == 1 else "No",
            ])
        lines.append(md_table(
            ["Rank", "Name", "CV ID", "Score", "Status",
             "Cosine Sim", "Hard Skill %", "Soft Skill %",
             "Edu Match", "Exp Match"],
            t_rows,
            ["---:", ":---", ":---", "---:", ":---",
             "---:", "---:", "---:", ":---", ":---"]))
        lines.append("")

    # Export all matching to CSV
    if all_matching_rows:
        export_path = REPORT_DIR / "all_matching_results.csv"
        pd.DataFrame(all_matching_rows).to_csv(export_path, index=False)
        lines.append(f"\n> Full matching results exported to: `reports/all_matching_results.csv` "
                     f"({len(all_matching_rows)} rows)\n")

    return "\n".join(lines)


def section_analytics(cv_ner_data, jd_data, model, dataset, model_results):
    lines = []
    lines.append("## 5. Analytics & Model Analysis\n")

    # ----- 5.1 Score Distribution (from dataset) -----
    lines.append("### 5.1 Score Distribution (Training Dataset)\n")
    if dataset is not None:
        X = dataset[FEATURE_COLUMNS]
        y = dataset["label"]

        if model is not None:
            scores = model.predict_proba(X)[:, 1]
            dataset_with_scores = dataset.copy()
            dataset_with_scores["score"] = scores

            match_scores = scores[y == 1]
            no_match_scores = scores[y == 0]

            score_rows = [
                ["All Candidates", str(len(scores)),
                 fmt_float(scores.mean()), fmt_float(np.std(scores)),
                 fmt_float(np.median(scores)),
                 fmt_float(scores.min()), fmt_float(scores.max())],
                ["Match (label=1)", str(len(match_scores)),
                 fmt_float(match_scores.mean()), fmt_float(np.std(match_scores)),
                 fmt_float(np.median(match_scores)),
                 fmt_float(match_scores.min()), fmt_float(match_scores.max())],
                ["No Match (label=0)", str(len(no_match_scores)),
                 fmt_float(no_match_scores.mean()), fmt_float(np.std(no_match_scores)),
                 fmt_float(np.median(no_match_scores)),
                 fmt_float(no_match_scores.min()), fmt_float(no_match_scores.max())],
            ]
            lines.append(md_table(
                ["Group", "Count", "Mean", "Std", "Median", "Min", "Max"],
                score_rows,
                [":---", "---:", "---:", "---:", "---:", "---:", "---:"]))

            # Score distribution buckets
            lines.append("\n**Score Distribution Buckets:**\n")
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
            labels = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
                      "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
            hist_counts, _ = np.histogram(scores, bins=bins)
            hist_rows = []
            for lbl, cnt in zip(labels, hist_counts):
                pct = cnt / max(len(scores), 1)
                bar = "█" * int(pct * 50)
                hist_rows.append([lbl, str(cnt), fmt_pct(pct), bar])
            lines.append(md_table(
                ["Score Range", "Count", "Percentage", "Distribution"],
                hist_rows, [":---", "---:", "---:", ":---"]))

    # ----- 5.2 Top Candidates (from dataset) -----
    lines.append("\n### 5.2 Top 20 Candidates (Highest Scores in Dataset)\n")
    if dataset is not None and model is not None:
        scores = model.predict_proba(dataset[FEATURE_COLUMNS])[:, 1]
        ds = dataset.copy()
        ds["score"] = scores
        ds_sorted = ds.sort_values("score", ascending=False).head(20)
        top_rows = []
        for rank, (_, r) in enumerate(ds_sorted.iterrows(), 1):
            cv_id = r.get("cv_id", "-")
            jd_id = r.get("jd_id", "-")
            top_rows.append([
                str(rank), cv_id[:35], jd_id,
                fmt_float(r["score"]),
                "Match" if r["label"] == 1 else "No Match",
                fmt_float(r["cosine_similarity"]),
                fmt_pct(r["hard_skill_match_ratio"]),
            ])
        lines.append(md_table(
            ["Rank", "CV ID", "JD ID", "Score", "Actual Label",
             "Cosine Sim", "Hard Skill %"],
            top_rows,
            ["---:", ":---", ":---", "---:", ":---", "---:", "---:"]))

    # ----- 5.3 Feature Statistics -----
    lines.append("\n### 5.3 Feature Statistics (Dataset)\n")
    if dataset is not None:
        feat_rows = []
        for col in FEATURE_COLUMNS:
            vals = dataset[col]
            feat_rows.append([
                col,
                fmt_float(vals.mean()),
                fmt_float(vals.std()),
                fmt_float(vals.median()),
                fmt_float(vals.min()),
                fmt_float(vals.max()),
            ])
        lines.append(md_table(
            ["Feature", "Mean", "Std", "Median", "Min", "Max"],
            feat_rows,
            [":---", "---:", "---:", "---:", "---:", "---:"]))

    # ----- 5.4 Feature Importance -----
    lines.append("\n### 5.4 Feature Importance (XGBoost)\n")
    if model is not None:
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        imp_rows = []
        for rank, idx in enumerate(sorted_idx, 1):
            name = FEATURE_COLUMNS[idx]
            imp = importance[idx]
            bar = "█" * int(imp * 80)
            ftype = "Semantic" if name == "cosine_similarity" else "Structured"
            imp_rows.append([
                str(rank), name, fmt_float(imp),
                fmt_pct(imp), ftype, bar
            ])
        lines.append(md_table(
            ["Rank", "Feature", "Importance", "Percentage", "Type", "Visual"],
            imp_rows,
            ["---:", ":---", "---:", "---:", ":---", ":---"]))

        semantic_imp = importance[0]
        structured_imp = sum(importance[1:])
        lines.append(f"\n**Semantic Contribution (cosine_similarity):** "
                     f"{fmt_pct(semantic_imp)}")
        lines.append(f"**Structured Contribution (14 features):** "
                     f"{fmt_pct(structured_imp)}")
        lines.append(f"**Ratio:** 1:{structured_imp / max(semantic_imp, 0.0001):.1f} "
                     f"(semantic : structured)")

    # ----- 5.5 Feature Breakdown per Label -----
    lines.append("\n### 5.5 Feature Breakdown: Match vs No Match\n")
    if dataset is not None:
        match_df = dataset[dataset["label"] == 1]
        no_match_df = dataset[dataset["label"] == 0]
        breakdown_rows = []
        for col in FEATURE_COLUMNS:
            m_mean = match_df[col].mean()
            nm_mean = no_match_df[col].mean()
            diff = m_mean - nm_mean
            breakdown_rows.append([
                col,
                fmt_float(m_mean),
                fmt_float(nm_mean),
                f"{'+' if diff >= 0 else ''}{fmt_float(diff)}",
            ])
        lines.append(md_table(
            ["Feature", "Match (Mean)", "No Match (Mean)", "Difference"],
            breakdown_rows,
            [":---", "---:", "---:", "---:"]))

    # ----- 5.6 Classification Confusion Matrix -----
    lines.append("\n### 5.6 Classification Results (Test Set)\n")
    if model_results:
        lines.append("```")
        lines.append("                precision    recall  f1-score")
        lines.append("")
        p = model_results.get("precision", 0)
        r = model_results.get("recall", 0)
        f1 = model_results.get("f1_score", 0)
        acc = model_results.get("accuracy", 0)
        lines.append(f"       Match     {p:.4f}      {r:.4f}    {f1:.4f}")
        lines.append(f"")
        lines.append(f"    Accuracy                          {acc:.4f}")
        lines.append(f"     AUC-ROC                          {model_results.get('auc_roc', 0):.4f}")
        lines.append("```")

    # ----- 5.7 Data Split Info -----
    lines.append("\n### 5.7 Data Split Information\n")
    if dataset is not None:
        total = len(dataset)
        n_train = int(total * 0.70)
        n_val = int(total * 0.15)
        n_test = total - n_train - n_val
        pos_ratio = dataset["label"].mean()
        split_rows = [
            ["Train", str(n_train), fmt_pct(n_train / total)],
            ["Validation", str(n_val), fmt_pct(n_val / total)],
            ["Test", str(n_test), fmt_pct(n_test / total)],
            ["**Total**", f"**{total}**", "**100%**"],
        ]
        lines.append(md_table(
            ["Split", "Count", "Percentage"],
            split_rows, [":---", "---:", "---:"]))
        lines.append(f"\n**Label Distribution:** Match = {int(dataset['label'].sum())} "
                     f"({fmt_pct(pos_ratio)}), "
                     f"No Match = {int((dataset['label'] == 0).sum())} "
                     f"({fmt_pct(1 - pos_ratio)})")
        lines.append(f"**Positive Class Ratio:** {pos_ratio:.4f}")

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive report")
    parser.add_argument("--top", type=int, default=10,
                        help="Top-N candidates per JD (default: 10)")
    parser.add_argument("--skip-matching", action="store_true",
                        help="Skip per-JD matching (faster, no S-BERT needed)")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_file = REPORT_DIR / "full_report.md"

    print("=" * 60)
    print("  TALENT RECOMMENDER SYSTEM - FULL REPORT GENERATOR")
    print("=" * 60)

    # Load all data
    print("\n[1/5] Loading data...")
    n_raw_cv = count_raw_cvs()
    cv_ner_data = load_cv_ner_files()
    jd_data = load_jd_files()
    model = load_model()
    dataset = load_dataset()
    model_results = load_model_results()
    print(f"  Raw CVs: {n_raw_cv} | NER: {len(cv_ner_data)} | "
          f"JDs: {len(jd_data)} | Model: {'OK' if model else 'N/A'} | "
          f"Dataset: {len(dataset) if dataset is not None else 'N/A'} rows")

    # Build report sections
    sections = []

    # Header
    sections.append(f"# Talent Recommender System - Comprehensive Report\n")
    sections.append(f"**Generated:** {timestamp}  ")
    sections.append(f"**System:** S-BERT + XGBoost CV-JD Matching Pipeline  ")
    sections.append(f"**Author:** Auto-generated by `scripts/generate_report.py`\n")
    sections.append("---\n")

    # Section 1
    print("\n[2/5] Generating System Overview...")
    sections.append(section_system_overview(
        n_raw_cv, len(cv_ner_data), len(jd_data), model, model_results))
    sections.append("\n---\n")

    # Section 2
    print("\n[3/5] Generating NER Distribution...")
    sections.append(section_ner_distribution(cv_ner_data))
    sections.append("\n---\n")

    # Section 3
    print("\n[4/5] Generating JD Overview...")
    sections.append(section_jd_overview(jd_data))
    sections.append("\n---\n")

    # Section 4
    if not args.skip_matching:
        print(f"\n[5/5] Running Matching per JD (top {args.top})...")
        print("  This may take a few minutes (S-BERT encoding + XGBoost scoring)...")
        sections.append(section_matching_per_jd(
            cv_ner_data, jd_data, model, top_n=args.top))
    else:
        print("\n[5/5] Skipping per-JD matching (--skip-matching)")
        sections.append("## 4. Matching Results per Job Description\n")
        sections.append("*Skipped (--skip-matching flag used).*\n")
    sections.append("\n---\n")

    # Section 5
    print("\n[6/6] Generating Analytics...")
    sections.append(section_analytics(
        cv_ner_data, jd_data, model, dataset, model_results))

    # Write report
    full_report = "\n".join(sections)
    report_file.write_text(full_report, encoding="utf-8")

    print("\n" + "=" * 60)
    print("  REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Report: {report_file}")
    print(f"  Size:   {report_file.stat().st_size / 1024:.1f} KB")
    if (REPORT_DIR / "all_matching_results.csv").exists():
        csv_path = REPORT_DIR / "all_matching_results.csv"
        print(f"  CSV:    {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Time:   {timestamp}")


if __name__ == "__main__":
    main()
