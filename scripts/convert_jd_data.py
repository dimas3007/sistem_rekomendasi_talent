"""
Convert job_dataset_complete.json to project JD format (individual JSON files).

Maps from:
  job_id, job_title, company, job_type, location, requirements.education,
  requirements.skills, requirements.soft_skills, requirements.experience,
  responsibilities, benefits

To project format:
  jd_id, title, company, department, level, description,
  requirements.education_level, requirements.education_major,
  requirements.min_experience_years, requirements.hard_skills,
  requirements.soft_skills, requirements.certifications,
  requirements.languages, requirements.location
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_education_level(edu_text: str) -> str:
    """Extract education level from free text."""
    text = edu_text.lower()
    if "master" in text or "s2" in text:
        return "S1"  # minimum S1, prefer S2
    if "bachelor" in text or "s1" in text or "sarjana" in text:
        return "S1"
    if "diploma" in text or "d3" in text:
        return "D3"
    if "mahasiswa" in text or "semester" in text:
        return "S1"  # university student = pursuing S1
    if "sma" in text:
        return "SMA"
    return "S1"  # default


def parse_education_major(edu_text: str) -> list:
    """Extract education majors from free text."""
    major_keywords = {
        "computer science": "Informatika",
        "informatika": "Informatika",
        "teknik informatika": "Teknik Informatika",
        "sistem informasi": "Sistem Informasi",
        "information technology": "Teknik Informatika",
        "statistics": "Statistika",
        "statistika": "Statistika",
        "mathematics": "Matematika",
        "matematika": "Matematika",
        "marketing": "Marketing",
        "communication": "Komunikasi",
        "komunikasi": "Komunikasi",
        "design": "Desain",
        "dkv": "DKV",
        "design communication visual": "DKV",
        "film": "Film",
        "media": "Media",
        "finance": "Keuangan",
        "economics": "Ekonomi",
        "accounting": "Akuntansi",
        "business": "Bisnis",
        "engineering": "Teknik",
        "information security": "Keamanan Informasi",
        "psikologi": "Psikologi",
        "manajemen": "Manajemen",
        "pendidikan": "Pendidikan",
    }

    text = edu_text.lower()
    majors = []
    for keyword, major in major_keywords.items():
        if keyword in text and major not in majors:
            majors.append(major)

    # "semua jurusan" = all majors → return empty list (no restriction)
    if "semua jurusan" in text or "any field" in text:
        return []

    return majors if majors else []


def parse_experience_years(job: dict) -> int:
    """Extract min experience years from free text."""
    exp_text = job.get("requirements", {}).get("experience", "")
    if not exp_text:
        # Interns and fresh graduates
        job_type = job.get("job_type", "").lower()
        if "intern" in job_type:
            return 0
        return 0

    # Extract first number: "2-5 years" → 2, "1-2 years" → 1
    match = re.search(r"(\d+)", exp_text)
    if match:
        years = int(match.group(1))
        # "Fresh graduate or 1-2 years" → 0
        if "fresh" in exp_text.lower() or "graduate" in exp_text.lower():
            return 0
        return years
    return 0


def determine_department(job_title: str) -> str:
    """Map job title to department."""
    title = job_title.lower()
    dept_map = [
        (["developer", "engineer", "devops", "backend", "frontend", "full stack",
          "web dev", "mobile", "qa", "cybersecurity", "ai/ml", "ml", "data analyst",
          "data science", "scrum master"], "IT"),
        (["marketing", "seo", "digital marketing", "social media", "content writer",
          "content creator"], "Marketing"),
        (["hr", "human resource", "recruitment", "learning & development",
          "training"], "HR"),
        (["designer", "ui/ux", "graphic", "product designer", "visual"], "Design"),
        (["finance", "financial", "accounting"], "Finance"),
        (["product manager"], "Product"),
        (["business analyst", "business development"], "Business"),
        (["event", "program officer"], "Operations"),
        (["partnership"], "Business Development"),
        (["customer success", "customer service", "support"], "Operations"),
        (["video"], "Creative"),
    ]
    for keywords, dept in dept_map:
        for kw in keywords:
            if kw in title:
                return dept
    return "General"


def determine_level(job: dict) -> str:
    """Determine job level from job_type and experience."""
    job_type = job.get("job_type", "").lower()
    exp_text = job.get("requirements", {}).get("experience", "")

    if "intern" in job_type:
        return "entry-level"

    if exp_text:
        match = re.search(r"(\d+)", exp_text)
        if match:
            years = int(match.group(1))
            if years >= 5:
                return "senior"
            elif years >= 2:
                return "mid-level"
            else:
                return "entry-level"

    if "fresh" in exp_text.lower():
        return "entry-level"

    return "mid-level"


def parse_certifications(job: dict) -> list:
    """Extract certifications from requirements."""
    req = job.get("requirements", {})
    certs_text = req.get("certifications", "")
    if not certs_text:
        return []

    if isinstance(certs_text, list):
        return certs_text

    # Extract individual cert names
    # "Scrum Master certification (CSM, PSM) preferred" → ["CSM", "PSM"]
    # "CISSP, CEH, or Security+ preferred" → ["CISSP", "CEH", "Security+"]
    cleaned = re.sub(r"\s*(preferred|or similar|or equivalent)\s*", "", certs_text, flags=re.IGNORECASE).strip()

    # Check for parenthetical abbreviations: "... (CSM, PSM)"
    paren_match = re.search(r"\(([^)]+)\)", cleaned)
    if paren_match:
        abbrevs = re.split(r"[,;]\s*|\s+or\s+", paren_match.group(1))
        return [a.strip() for a in abbrevs if a.strip()]

    # Otherwise split on comma/semicolon/or
    parts = re.split(r"[,;]\s*|\s+or\s+", cleaned)
    certs = [re.sub(r"^or\s+", "", p).strip() for p in parts if p.strip() and len(p.strip()) > 1]
    return certs


def build_description(job: dict) -> str:
    """Build a description from job data (Indonesian style)."""
    title = job["job_title"]
    company = job["company"]
    responsibilities = job.get("responsibilities", [])

    desc_parts = [f"Kami mencari {title} untuk bergabung dengan tim {company}."]

    if responsibilities:
        resp_text = "; ".join(responsibilities[:4])
        desc_parts.append(f"Tanggung jawab utama meliputi: {resp_text}.")

    job_type = job.get("job_type", "")
    if "intern" in job_type.lower():
        desc_parts.append("Posisi ini adalah program magang yang memberikan kesempatan belajar dan pengembangan karir.")

    return " ".join(desc_parts)


def convert_job(job: dict, index: int) -> dict:
    """Convert a single job from new format to project format."""
    jd_id = f"JD-{index:03d}"
    req = job.get("requirements", {})
    edu_text = req.get("education", "")

    # Parse skills
    hard_skills = req.get("skills", [])
    soft_skills = req.get("soft_skills", [])
    # Also include preferred_skills as hard skills
    preferred = req.get("preferred_skills", [])
    if preferred:
        hard_skills = hard_skills + preferred

    # Parse location
    location = job.get("location", "")
    # Simplify: "Jakarta, Indonesia" → "Jakarta", "Remote - Indonesia" → "Remote"
    if "," in location:
        location = location.split(",")[0].strip()
    if " - " in location:
        location = location.split(" - ")[0].strip()

    return {
        "jd_id": jd_id,
        "title": job["job_title"],
        "company": job["company"],
        "department": determine_department(job["job_title"]),
        "level": determine_level(job),
        "description": build_description(job),
        "requirements": {
            "education_level": parse_education_level(edu_text),
            "education_major": parse_education_major(edu_text),
            "min_experience_years": parse_experience_years(job),
            "hard_skills": hard_skills,
            "soft_skills": soft_skills,
            "certifications": parse_certifications(job),
            "languages": ["Bahasa Indonesia", "English"],
            "location": location,
        }
    }


def main():
    source = ROOT / "skripsi" / "job_dataset_complete.json"
    output_dir = ROOT / "data" / "raw" / "jds"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {source}")
    with open(source, "r", encoding="utf-8") as f:
        data = json.load(f)

    jobs = data["jobs"]
    print(f"Found {len(jobs)} jobs")

    # Delete existing JD files
    existing = list(output_dir.glob("JD-*.json"))
    for f in existing:
        f.unlink()
    print(f"Removed {len(existing)} existing JD files")

    # Convert and save each job
    for i, job in enumerate(jobs, start=1):
        converted = convert_job(job, i)
        out_path = output_dir / f"{converted['jd_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(jobs)} JD files to {output_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  JD CONVERSION SUMMARY")
    print(f"{'='*60}")
    for i, job in enumerate(jobs, start=1):
        converted = convert_job(job, i)
        req = converted["requirements"]
        print(f"  JD-{i:03d}: {converted['title']:<35} "
              f"Dept={converted['department']:<15} "
              f"Level={converted['level']:<12} "
              f"Edu={req['education_level']} "
              f"Exp={req['min_experience_years']}yr "
              f"Skills={len(req['hard_skills'])}h/{len(req['soft_skills'])}s")


if __name__ == "__main__":
    main()
