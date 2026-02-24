import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")
st.header("📈 Analytics & Visualization")

# --- Matching Results Visualization ---
if st.session_state.get("matching_results") is not None:
    results_df = st.session_state["matching_results"]

    st.subheader("Distribusi Skor Matching")

    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Top Candidates", "Feature Breakdown"])

    with tab1:
        # Score histogram using Streamlit native
        import plotly.express as px

        fig = px.histogram(
            results_df, x="Skor XGBoost", nbins=30,
            color="Prediksi",
            color_discrete_map={"Match": "#2ecc71", "No Match": "#e74c3c"},
            title="Distribusi Skor XGBoost",
        )
        st.plotly_chart(fig, width="stretch")

        # Summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Rata-rata Skor", f"{results_df['Skor XGBoost'].mean():.4f}")
        col2.metric("Kandidat Match", len(results_df[results_df['Prediksi'] == 'Match']))
        col3.metric("Kandidat No Match", len(results_df[results_df['Prediksi'] == 'No Match']))

    with tab2:
        # Top candidates bar chart
        top_n = st.slider("Jumlah Top Kandidat", 5, 20, 10, key="top_n_analytics")
        top_df = results_df.head(top_n)

        fig = px.bar(
            top_df, x="Skor XGBoost", y="CV",
            orientation="h",
            title=f"Top {top_n} Kandidat",
            color="Skor XGBoost",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

    with tab3:
        # Feature breakdown for selected candidate
        st.subheader("Detail Fitur per Kandidat")

        selected_cv = st.selectbox("Pilih kandidat:", results_df["CV"].tolist())

        if selected_cv:
            row = results_df[results_df["CV"] == selected_cv].iloc[0]
            st.write(f"**Skor:** {row['Skor XGBoost']}")
            st.write(f"**Prediksi:** {row['Prediksi']}")
            st.write(f"**Cosine Similarity:** {row['Cosine Similarity']}")
            st.write(f"**Hard Skill Match:** {row['Hard Skill Match']}")
            st.write(f"**Soft Skill Match:** {row['Soft Skill Match']}")
            st.write(f"**Experience Gap:** {row['Experience Gap']}")
            st.write(f"**Education:** {row['Education']}")

            # Radar chart if CV data available
            cv_data = st.session_state.get("cv_data", {})
            if selected_cv in cv_data:
                import plotly.graph_objects as go

                # Normalize features for radar chart
                radar_features = ["Cosine Sim.", "Hard Skills", "Soft Skills", "Education", "Experience"]
                radar_values = [
                    row["Cosine Similarity"],
                    float(row["Hard Skill Match"].replace("%", "")) / 100,
                    float(row["Soft Skill Match"].replace("%", "")) / 100,
                    1.0 if row["Education"] == "Match" else 0.0,
                    min(1.0, max(0.0, 0.5 + float(row["Experience Gap"].replace(" thn", "").replace("+", "")) / 10)),
                ]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=radar_values,
                    theta=radar_features,
                    fill="toself",
                    name=selected_cv,
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f"Profil: {selected_cv}",
                )
                st.plotly_chart(fig, width="stretch")
else:
    st.info("Belum ada hasil matching. Jalankan matching di halaman Matching terlebih dahulu.")

# --- Model Performance (if available) ---
st.markdown("---")
st.subheader("Model Performance")

model_path = Path("models/xgboost_model.json")
features_path = Path("data/features/dataset.csv")

if model_path.exists() and features_path.exists():
    st.write("Model XGBoost tersedia. Gunakan notebook evaluasi untuk melihat metrik lengkap.")

    # Show feature importance if model is loaded
    if st.button("Tampilkan Feature Importance"):
        try:
            from xgboost import XGBClassifier
            from config.settings import FEATURE_COLUMNS
            import plotly.express as px

            model = XGBClassifier()
            model.load_model(str(model_path))

            importance = model.feature_importances_
            imp_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": importance,
            }).sort_values("Importance", ascending=True)

            fig = px.bar(
                imp_df, x="Importance", y="Feature",
                orientation="h",
                title="Feature Importance (XGBoost)",
                color="Importance",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, width="stretch")

            # Semantic vs Structured
            semantic = importance[0]
            structured = sum(importance[1:])
            col1, col2 = st.columns(2)
            col1.metric("Semantic Contribution", f"{semantic*100:.1f}%")
            col2.metric("Structured Contribution", f"{structured*100:.1f}%")

        except Exception as e:
            st.error(f"Error loading model: {e}")
else:
    st.info("Model belum dilatih atau dataset fitur belum tersedia.")
