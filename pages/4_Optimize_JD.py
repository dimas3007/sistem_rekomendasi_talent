import streamlit as st
from google import genai
from google.genai import types
from config.settings import GEMINI_API_KEY, GEMINI_MODEL

st.set_page_config(page_title="Optimize JD", page_icon="✨", layout="wide")
st.header("✨ Optimize Job Description")

st.markdown("""
Gunakan Gemini AI untuk menganalisis dan mengoptimalkan Job Description Anda.
Sistem akan mengevaluasi kelengkapan, kejelasan, dan memberikan saran perbaikan.
""")

# Option 1: Use existing JD from session
jd_data = st.session_state.get("jd_data", {})
use_existing = st.checkbox("Gunakan JD yang sudah ada")

jd_text = ""
if use_existing and jd_data:
    selected_jd = st.selectbox("Pilih JD:", list(jd_data.keys()))
    if selected_jd:
        jd = jd_data[selected_jd]
        req = jd.get("requirements", {})
        jd_text = f"""Judul: {jd.get('title', '')}
Perusahaan: {jd.get('company', '')}
Department: {jd.get('department', '')}
Level: {jd.get('level', '')}

Deskripsi:
{jd.get('description', '')}

Requirements:
- Pendidikan: {req.get('education_level', '')} {', '.join(req.get('education_major', []))}
- Pengalaman: {req.get('min_experience_years', 0)} tahun
- Hard Skills: {', '.join(req.get('hard_skills', []))}
- Soft Skills: {', '.join(req.get('soft_skills', []))}
- Sertifikasi: {', '.join(req.get('certifications', []))}
- Bahasa: {', '.join(req.get('languages', []))}
- Lokasi: {req.get('location', '')}"""

# Option 2: Manual input
jd_text = st.text_area("Job Description:", value=jd_text, height=300)

if st.button("Analisis & Optimasi") and jd_text.strip():
    with st.spinner("Menganalisis JD dengan Gemini..."):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)

            prompt = f"""Analisis Job Description berikut dan berikan saran perbaikan:

{jd_text}

Evaluasi berdasarkan kriteria berikut dan berikan skor 1-10 untuk setiap aspek:

1. **Kelengkapan Informasi** (requirements, responsibilities, qualifications)
   - Apakah semua informasi penting sudah tercakup?

2. **Kejelasan Deskripsi Posisi**
   - Apakah tanggung jawab dan ekspektasi jelas?

3. **Spesifisitas Hard Skills**
   - Apakah hard skills yang dibutuhkan spesifik dan terukur?

4. **Realisme Experience Requirement**
   - Apakah requirement pengalaman realistis untuk level posisi?

5. **Daya Tarik Keseluruhan**
   - Apakah JD ini menarik bagi kandidat potensial?

Untuk setiap aspek, berikan:
- Skor (1-10)
- Penjelasan singkat
- Saran perbaikan konkret

Di akhir, berikan versi JD yang sudah dioptimalkan.

Respond in Bahasa Indonesia."""

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4096,
                ),
            )

            st.markdown("---")
            st.subheader("Hasil Analisis")
            st.markdown(response.text)

        except Exception as e:
            st.error(f"Error saat menganalisis: {e}")
            st.info("Pastikan GEMINI_API_KEY sudah dikonfigurasi di file .env")
elif not jd_text.strip():
    st.info("Masukkan atau pilih Job Description untuk dianalisis.")
