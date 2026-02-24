import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.header("📊 Dashboard")

# --- Data counts ---
cv_dir = Path("data/raw/cvs")
ner_dir = Path("data/processed/cv_ner")
jd_dir = Path("data/raw/jds")
model_path = Path("models/xgboost_model.json")

n_cvs = len(list(cv_dir.glob("*.pdf"))) if cv_dir.exists() else 0
n_ner = len(list(ner_dir.glob("*.json"))) if ner_dir.exists() else 0
n_jds = len(list(jd_dir.glob("*.json"))) if jd_dir.exists() else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total CV (PDF)", n_cvs)
col2.metric("CV Processed", n_ner)
col3.metric("Job Descriptions", n_jds)
col4.metric("Model", "Ready" if model_path.exists() else "Not Trained")

st.markdown("---")

# --- NER Statistics ---
if n_ner > 0:
    st.subheader("NER Extraction Statistics")

    all_ner = []
    for f in sorted(ner_dir.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                all_ner.append(json.load(fp))
        except Exception:
            pass

    if all_ner:
        # Education distribution
        edu_counts = {}
        for ner in all_ner:
            edu = ner.get("education_level", "Unknown") or "Unknown"
            edu_counts[edu] = edu_counts.get(edu, 0) + 1

        # Experience distribution
        experiences = [
            ner.get("years_experience", 0) or 0
            for ner in all_ner
        ]

        # Skills count
        avg_hard = sum(len(ner.get("hard_skills") or []) for ner in all_ner) / len(all_ner)
        avg_soft = sum(len(ner.get("soft_skills") or []) for ner in all_ner) / len(all_ner)

        stat_col1, stat_col2, stat_col3 = st.columns(3)

        with stat_col1:
            st.markdown("**Education Level Distribution**")
            edu_df = pd.DataFrame(
                list(edu_counts.items()),
                columns=["Education", "Count"]
            ).sort_values("Count", ascending=False)
            st.bar_chart(edu_df.set_index("Education"))

        with stat_col2:
            st.markdown("**Experience Distribution**")
            exp_df = pd.DataFrame({"Years": experiences})
            st.bar_chart(exp_df["Years"].value_counts().sort_index())

        with stat_col3:
            st.markdown("**Average Skills per CV**")
            st.metric("Hard Skills", f"{avg_hard:.1f}")
            st.metric("Soft Skills", f"{avg_soft:.1f}")
            st.metric("Total CVs Analyzed", len(all_ner))

# --- JD Overview ---
if n_jds > 0:
    st.markdown("---")
    st.subheader("Job Description Overview")

    jd_list = []
    for f in sorted(jd_dir.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                jd = json.load(fp)
                jd_list.append({
                    "ID": jd.get("jd_id", f.stem),
                    "Title": jd.get("title", "N/A"),
                    "Department": jd.get("department", "N/A"),
                    "Level": jd.get("level", "N/A"),
                })
        except Exception:
            pass

    if jd_list:
        st.dataframe(pd.DataFrame(jd_list), width="stretch")

# --- Matching Results Summary ---
if st.session_state.get("matching_results") is not None:
    st.markdown("---")
    st.subheader("Latest Matching Results")
    results_df = st.session_state["matching_results"]
    st.write(f"Total candidates ranked: {len(results_df)}")
    st.dataframe(results_df.head(10), width="stretch")
else:
    st.markdown("---")
    st.info("No matching results yet. Go to the Matching page to run CV-JD matching.")
