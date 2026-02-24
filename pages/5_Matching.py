import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path

st.set_page_config(page_title="Matching", page_icon="🎯", layout="wide")
st.header("🎯 Matching & Ranking Kandidat")

st.markdown("""
Jalankan matching antara CV kandidat dan Job Description yang dipilih.
Sistem menggunakan model XGBoost dengan 15 fitur (1 semantik + 14 terstruktur).
""")

# --- Load data sources ---
# JDs from session state or disk
jd_data = st.session_state.get("jd_data", {})
if not jd_data:
    jd_dir = Path("data/raw/jds")
    if jd_dir.exists():
        for f in sorted(jd_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    jd = json.load(fp)
                    jd_data[jd.get("jd_id", f.stem)] = jd
            except Exception:
                pass
        st.session_state["jd_data"] = jd_data

# CVs from NER directory or session state
cv_data = st.session_state.get("cv_data", {})
ner_dir = Path("data/processed/cv_ner")
if ner_dir.exists():
    for f in sorted(ner_dir.glob("*.json")):
        if f.stem not in cv_data:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    cv_data[f.stem] = json.load(fp)
            except Exception:
                pass
    st.session_state["cv_data"] = cv_data

# Check model
model_path = Path("models/xgboost_model.json")

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    st.metric("CV Tersedia", len(cv_data))
with col2:
    st.metric("JD Tersedia", len(jd_data))

if not jd_data:
    st.warning("Belum ada Job Description. Tambahkan di halaman Manage JD.")
    st.stop()

if not cv_data:
    st.warning("Belum ada CV yang diproses. Upload CV di halaman Upload CV.")
    st.stop()

if not model_path.exists():
    st.warning("Model XGBoost belum tersedia. Latih model terlebih dahulu.")
    st.stop()

# Select JD
jd_options = {f"{jd.get('title', k)} ({k})": k for k, jd in jd_data.items()}
selected_jd_label = st.selectbox("Pilih Job Description:", list(jd_options.keys()))
selected_jd_id = jd_options[selected_jd_label]

# Show JD details
with st.expander("Detail JD", expanded=False):
    jd = jd_data[selected_jd_id]
    req = jd.get("requirements", {})
    st.write(f"**Posisi:** {jd.get('title')}")
    st.write(f"**Department:** {jd.get('department')}")
    st.write(f"**Level:** {jd.get('level')}")
    st.write(f"**Hard Skills:** {', '.join(req.get('hard_skills', []))}")
    st.write(f"**Pengalaman:** {req.get('min_experience_years', 0)} tahun")

# Run matching
if st.button("Jalankan Matching", type="primary"):
    with st.spinner("Memuat model dan menghitung fitur..."):
        try:
            from xgboost import XGBClassifier
            from src.features.assembler import build_feature_vector
            from config.settings import FEATURE_COLUMNS

            # Load model
            model = XGBClassifier()
            model.load_model(str(model_path))

            jd = jd_data[selected_jd_id]
            results = []

            progress = st.progress(0)
            total = len(cv_data)

            for i, (cv_name, cv_entities) in enumerate(cv_data.items()):
                features = build_feature_vector(cv_entities, jd)
                feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

                score = model.predict_proba(feature_df)[:, 1][0]
                prediction = model.predict(feature_df)[0]

                results.append({
                    "CV": cv_name,
                    "Skor XGBoost": round(score, 4),
                    "Prediksi": "Match" if prediction == 1 else "No Match",
                    "Cosine Similarity": round(features["cosine_similarity"], 4),
                    "Hard Skill Match": f"{features['hard_skill_match_ratio']*100:.0f}%",
                    "Soft Skill Match": f"{features['soft_skill_match_ratio']*100:.0f}%",
                    "Experience Gap": f"{features['experience_gap']:+.1f} thn",
                    "Education": "Match" if features["education_match"] else "No",
                })

                progress.progress((i + 1) / total)

            # Sort by score descending
            results_df = pd.DataFrame(results).sort_values("Skor XGBoost", ascending=False)
            results_df.index = range(1, len(results_df) + 1)
            results_df.index.name = "Rank"

            # Save to session state
            st.session_state["matching_results"] = results_df
            st.session_state["matching_jd"] = selected_jd_id

            st.success(f"Matching selesai! {len(results)} kandidat dianalisis.")

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# --- Display results ---
if st.session_state.get("matching_results") is not None:
    st.markdown("---")
    results_df = st.session_state["matching_results"]

    st.subheader(f"Hasil Ranking ({len(results_df)} kandidat)")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_match_only = st.checkbox("Tampilkan hanya Match")
    with col2:
        top_n = st.slider("Top N kandidat", min_value=5, max_value=len(results_df), value=min(20, len(results_df)))

    display_df = results_df.copy()
    if show_match_only:
        display_df = display_df[display_df["Prediksi"] == "Match"]

    st.dataframe(display_df.head(top_n), width="stretch")

    # Export
    csv = display_df.to_csv(index=True)
    st.download_button(
        label="Download Hasil (CSV)",
        data=csv,
        file_name=f"ranking_{st.session_state.get('matching_jd', 'unknown')}.csv",
        mime="text/csv"
    )
