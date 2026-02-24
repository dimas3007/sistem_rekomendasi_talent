import streamlit as st
import json
import os
from pathlib import Path

st.set_page_config(page_title="Manage JD", page_icon="💼", layout="wide")
st.header("💼 Manage Job Descriptions")

jd_dir = Path("data/raw/jds")
jd_dir.mkdir(parents=True, exist_ok=True)

# Initialize session state for JDs
if "jd_data" not in st.session_state:
    st.session_state["jd_data"] = {}

# Load existing JDs from disk
def load_jds():
    jd_data = {}
    for f in sorted(jd_dir.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                jd = json.load(fp)
                jd_data[jd.get("jd_id", f.stem)] = jd
        except Exception:
            pass
    return jd_data

if not st.session_state["jd_data"]:
    st.session_state["jd_data"] = load_jds()

# --- Add New JD ---
st.subheader("Tambah Job Description Baru")

with st.form("add_jd_form"):
    col1, col2 = st.columns(2)

    with col1:
        jd_id = st.text_input("JD ID", placeholder="JD-001")
        title = st.text_input("Judul Posisi", placeholder="Data Analyst")
        company = st.text_input("Perusahaan", placeholder="PT Coptera Career Indonesia")
        department = st.selectbox("Department", ["IT", "Marketing", "Finance", "Operations", "HR", "Other"])
        level = st.selectbox("Level", ["entry-level", "mid-level", "senior"])

    with col2:
        education_level = st.selectbox("Pendidikan Minimal", ["SMA", "D3", "S1", "S2", "S3"])
        education_major = st.text_input("Jurusan (pisahkan dengan koma)", placeholder="Statistika, Informatika")
        min_experience = st.number_input("Pengalaman Minimal (tahun)", min_value=0, max_value=30, value=2)
        location = st.text_input("Lokasi", placeholder="Jakarta")

    description = st.text_area("Deskripsi Pekerjaan", height=150, placeholder="Uraikan tanggung jawab posisi...")
    hard_skills = st.text_input("Hard Skills (pisahkan dengan koma)", placeholder="Python, SQL, Tableau, Excel")
    soft_skills = st.text_input("Soft Skills (pisahkan dengan koma)", placeholder="Analytical Thinking, Communication")
    certifications = st.text_input("Sertifikasi (opsional, pisahkan dengan koma)", placeholder="")
    languages = st.text_input("Bahasa (pisahkan dengan koma)", placeholder="Bahasa Indonesia, English")

    submitted = st.form_submit_button("Simpan JD")

    if submitted and jd_id and title:
        jd_data = {
            "jd_id": jd_id,
            "title": title,
            "company": company,
            "department": department,
            "level": level,
            "description": description,
            "requirements": {
                "education_level": education_level,
                "education_major": [m.strip() for m in education_major.split(",") if m.strip()],
                "min_experience_years": min_experience,
                "hard_skills": [s.strip() for s in hard_skills.split(",") if s.strip()],
                "soft_skills": [s.strip() for s in soft_skills.split(",") if s.strip()],
                "certifications": [c.strip() for c in certifications.split(",") if c.strip()],
                "languages": [l.strip() for l in languages.split(",") if l.strip()],
                "location": location,
            }
        }

        # Save to disk
        with open(jd_dir / f"{jd_id}.json", "w", encoding="utf-8") as f:
            json.dump(jd_data, f, ensure_ascii=False, indent=2)

        # Save to session state
        st.session_state["jd_data"][jd_id] = jd_data
        st.success(f"JD '{title}' ({jd_id}) berhasil disimpan!")
        st.rerun()

# --- List Existing JDs ---
st.markdown("---")
st.subheader("Daftar Job Descriptions")

jd_data = st.session_state.get("jd_data", {})

if jd_data:
    for jd_id, jd in jd_data.items():
        with st.expander(f"**{jd.get('title', 'N/A')}** ({jd_id}) - {jd.get('department', 'N/A')} | {jd.get('level', 'N/A')}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Perusahaan:** {jd.get('company', 'N/A')}")
                st.write(f"**Deskripsi:** {jd.get('description', 'N/A')[:300]}...")

            with col2:
                req = jd.get("requirements", {})
                st.write(f"**Pendidikan:** {req.get('education_level', 'N/A')}")
                st.write(f"**Pengalaman:** {req.get('min_experience_years', 0)} tahun")
                st.write(f"**Hard Skills:** {', '.join(req.get('hard_skills', []))}")
                st.write(f"**Soft Skills:** {', '.join(req.get('soft_skills', []))}")
                st.write(f"**Lokasi:** {req.get('location', 'N/A')}")

            # Delete button
            if st.button(f"Hapus {jd_id}", key=f"del_{jd_id}"):
                jd_path = jd_dir / f"{jd_id}.json"
                if jd_path.exists():
                    os.remove(jd_path)
                del st.session_state["jd_data"][jd_id]
                st.success(f"JD {jd_id} berhasil dihapus!")
                st.rerun()
else:
    st.info("Belum ada Job Description. Tambahkan melalui form di atas.")
