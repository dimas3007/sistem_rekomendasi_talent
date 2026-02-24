NER_SYSTEM_PROMPT = """You are an HR data extraction assistant specializing in
parsing Indonesian CVs/resumes. Extract structured information and return ONLY
valid JSON (no markdown, no backticks, no explanation)."""

NER_SCHEMA = """{
    "name": "string (full name)",
    "email": "string or null",
    "phone": "string or null",
    "education_level": "SD|SMP|SMA|D3|S1|S2|S3 or null",
    "education_major": "string or null",
    "university": "string or null",
    "hard_skills": ["list of technical/measurable skills"],
    "soft_skills": ["list of interpersonal/character traits"],
    "years_experience": "float (total years from all positions)",
    "certifications": ["list of certifications"],
    "languages": ["list of languages"],
    "job_titles": ["list of previous job titles"],
    "industries": ["list of industries worked in"],
    "location": "string (current city) or null"
}"""

NER_RULES = """Extraction Rules:
1. hard_skills = programming languages, tools, frameworks, technical abilities
2. soft_skills = leadership, communication, teamwork, problem-solving, etc.
3. years_experience = calculate TOTAL duration from all work experiences
4. education_level = highest education level achieved
5. If information is not found, use null (for strings) or [] (for arrays)
6. Certifications include professional certs, courses, and training programs
7. Always extract in the original language used in the CV"""

FEW_SHOT_EXAMPLE = """Example Input: "Muhammad Rizki, S.Kom. Python Developer di PT ABC
selama 3 tahun. Menguasai Python, Django, PostgreSQL. Lulusan Universitas Padjadjaran."

Example Output:
{"name":"Muhammad Rizki","email":null,"phone":null,"education_level":"S1",
"education_major":"Teknik Informatika","university":"Universitas Padjadjaran",
"hard_skills":["Python","Django","PostgreSQL"],"soft_skills":[],
"years_experience":3.0,"certifications":[],"languages":["Bahasa Indonesia"],
"job_titles":["Python Developer"],"industries":["Technology"],"location":null}"""
