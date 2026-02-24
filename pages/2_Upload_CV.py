import streamlit as st
import json
import os
from pathlib import Path

st.set_page_config(page_title="Upload CV", page_icon="📄", layout="wide")
st.header("📄 Upload CV")

st.markdown("""
Upload satu atau beberapa CV dalam format PDF. Sistem akan:
1. **Parse** teks dari PDF (pdfplumber + OCR fallback)
2. **Ekstrak entitas** menggunakan Gemini NER (nama, skills, pendidikan, dll.)
3. **Simpan** hasil untuk digunakan dalam matching
""")

uploaded_files = st.file_uploader(
    "Upload CV (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    progress = st.progress(0)
    success_count = 0
    fail_count = 0

    for i, file in enumerate(uploaded_files):
        with st.expander(f"📋 {file.name}", expanded=(i == 0)):
            # Step 1: Parse PDF
            with st.spinner("Parsing PDF..."):
                try:
                    from src.parsing.pdf_parser import parse_pdf
                    text, error = parse_pdf(file)
                except Exception as e:
                    text, error = None, f"Import error: {e}"

            if error:
                st.error(f"Error: {error}")
                fail_count += 1
                continue

            st.success(f"Berhasil diekstrak ({len(text)} karakter)")

            # Show preview
            with st.expander("Preview teks", expanded=False):
                st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

            # Step 2: NER with Gemini
            with st.spinner("Ekstraksi entitas dengan Gemini NER..."):
                try:
                    from src.ner.gemini_ner import extract_entities
                    entities = extract_entities(text)
                except Exception as e:
                    entities = None
                    st.warning(f"NER error: {e}")

            if entities:
                st.json(entities)

                # Save to session state
                if "cv_data" not in st.session_state:
                    st.session_state["cv_data"] = {}
                st.session_state["cv_data"][file.name] = entities

                # Save parsed text to disk
                parsed_dir = Path("data/processed/cv_parsed")
                parsed_dir.mkdir(parents=True, exist_ok=True)
                cv_id = file.name.replace(".pdf", "")
                with open(parsed_dir / f"{cv_id}.txt", "w", encoding="utf-8") as f:
                    f.write(text)

                # Save NER result to disk
                ner_dir = Path("data/processed/cv_ner")
                ner_dir.mkdir(parents=True, exist_ok=True)
                with open(ner_dir / f"{cv_id}.json", "w", encoding="utf-8") as f:
                    json.dump(entities, f, ensure_ascii=False, indent=2)

                success_count += 1
            else:
                st.warning("NER gagal untuk file ini")
                fail_count += 1

        progress.progress((i + 1) / len(uploaded_files))

    st.markdown("---")
    st.markdown(f"**Hasil:** {success_count} berhasil, {fail_count} gagal dari {len(uploaded_files)} file")

# --- Show existing CV data ---
st.markdown("---")
st.subheader("CV yang Sudah Diproses")

ner_dir = Path("data/processed/cv_ner")
if ner_dir.exists():
    ner_files = sorted(ner_dir.glob("*.json"))
    if ner_files:
        st.write(f"Total: {len(ner_files)} CV")

        selected = st.selectbox(
            "Pilih CV untuk melihat detail:",
            [f.stem for f in ner_files]
        )
        if selected:
            with open(ner_dir / f"{selected}.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            st.json(data)
    else:
        st.info("Belum ada CV yang diproses.")
else:
    st.info("Belum ada CV yang diproses. Upload CV di atas untuk memulai.")
