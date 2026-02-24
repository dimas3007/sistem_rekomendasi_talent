import streamlit as st
import json
import os
from pathlib import Path

st.set_page_config(
    page_title="Talent Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "cv_data" not in st.session_state:
    st.session_state["cv_data"] = {}
if "jd_data" not in st.session_state:
    # Load existing JDs from disk
    jd_dir = Path("data/raw/jds")
    jd_data = {}
    if jd_dir.exists():
        for f in sorted(jd_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    jd = json.load(fp)
                    jd_data[jd.get("jd_id", f.stem)] = jd
            except Exception:
                pass
    st.session_state["jd_data"] = jd_data
if "matching_results" not in st.session_state:
    st.session_state["matching_results"] = None

st.title("Talent Recommendation System")
st.caption("S-BERT x XGBoost | Powered by Gemini NER")

st.markdown("---")

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)

# Count CVs
cv_dir = Path("data/raw/cvs")
n_cvs = len(list(cv_dir.glob("*.pdf"))) if cv_dir.exists() else 0

# Count NER results
ner_dir = Path("data/processed/cv_ner")
n_ner = len(list(ner_dir.glob("*.json"))) if ner_dir.exists() else 0

# Count JDs
n_jds = len(st.session_state.get("jd_data", {}))

# Model info
model_path = Path("models/xgboost_model.json")
model_exists = model_path.exists()

col1.metric("Total CV (PDF)", n_cvs)
col2.metric("CV Processed (NER)", n_ner)
col3.metric("Total Job Descriptions", n_jds)
col4.metric("Model Status", "Ready" if model_exists else "Not Trained")

st.markdown("---")

# Quick guide
st.subheader("Quick Guide")
st.markdown("""
1. **Upload CV** - Upload PDF CVs, parse them, and extract entities with Gemini NER
2. **Manage JD** - Create, edit, and manage Job Descriptions
3. **Optimize JD** - Use Gemini AI to analyze and improve your Job Descriptions
4. **Matching** - Run CV-JD matching and get ranked candidate recommendations
5. **Analytics** - Visualize matching results and model performance
""")

# System status
st.subheader("System Status")
status_col1, status_col2 = st.columns(2)

with status_col1:
    st.markdown("**Data Pipeline**")
    st.write(f"- CVs in storage: {n_cvs}")
    st.write(f"- CVs processed (NER): {n_ner}")
    st.write(f"- JDs available: {n_jds}")

with status_col2:
    st.markdown("**Model**")
    if model_exists:
        st.write("- XGBoost model: Loaded")
        st.write("- Features: 15-dimensional (1 semantic + 14 structured)")
    else:
        st.write("- XGBoost model: Not yet trained")
        st.write("- Train the model using notebooks or the training script")
