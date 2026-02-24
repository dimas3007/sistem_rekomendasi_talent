"""
Generate a labeling spreadsheet for CV-JD pair annotation.

Strategy: EXHAUSTIVE pairing — every CV is paired with every JD.
- 580 CVs x 28 JDs = 16,240 pairs
- Sorted by JD then by heuristic score (descending) for easy review

Usage:
    python scripts/generate_labeling_sheet.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_RAW_JD, DATA_NER_CV


def load_jds():
    """Load all JD files."""
    jd_dir = Path(DATA_RAW_JD)
    jds = {}
    for f in sorted(jd_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            jd = json.load(fp)
            jds[jd["jd_id"]] = jd
    return jds


def load_cv_ner():
    """Load all CV NER results."""
    ner_dir = Path(DATA_NER_CV)
    cvs = {}
    for f in sorted(ner_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            cvs[f.stem] = json.load(fp)
    return cvs


def compute_quick_score(cv: dict, jd: dict) -> float:
    """Quick heuristic score for pre-sorting (not the model score)."""
    req = jd.get("requirements", jd)
    score = 0.0

    cv_hard = set(s.lower() for s in (cv.get("hard_skills") or []))
    jd_hard = set(s.lower() for s in (req.get("hard_skills") or []))
    if jd_hard:
        score += len(cv_hard & jd_hard) / len(jd_hard) * 3

    edu_map = {"SD": 1, "SMP": 2, "SMA": 3, "D3": 4, "S1": 5, "S2": 6, "S3": 7}
    cv_edu = edu_map.get(cv.get("education_level"), 0)
    jd_edu = edu_map.get(req.get("education_level"), 0)
    if cv_edu >= jd_edu and jd_edu > 0:
        score += 1

    cv_exp = cv.get("years_experience") or 0
    jd_exp = req.get("min_experience_years") or 0
    if cv_exp >= jd_exp:
        score += 1

    return score


def generate_all_pairs(jds, cvs):
    """Generate exhaustive CV-JD pairs (every CV x every JD)."""
    all_pairs = []

    for jd_id in sorted(jds.keys()):
        jd = jds[jd_id]
        req = jd.get("requirements", jd)
        jd_hard = set(s.lower() for s in (req.get("hard_skills") or []))

        # Score and sort CVs for this JD (best matches first)
        scored = []
        for cv_id in sorted(cvs.keys()):
            cv = cvs[cv_id]
            h_score = compute_quick_score(cv, jd)
            scored.append((cv_id, h_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        for cv_id, heuristic_score in scored:
            cv = cvs[cv_id]
            cv_hard = set(s.lower() for s in (cv.get("hard_skills") or []))
            overlap = cv_hard & jd_hard

            all_pairs.append({
                "jd_id": jd_id,
                "jd_title": jd.get("title", ""),
                "jd_department": jd.get("department", ""),
                "jd_level": jd.get("level", ""),
                "cv_id": cv_id,
                "cv_name": cv.get("name", ""),
                "cv_education": cv.get("education_level", ""),
                "cv_major": cv.get("education_major", ""),
                "cv_experience_years": cv.get("years_experience", 0),
                "cv_hard_skills": ", ".join(cv.get("hard_skills") or []),
                "jd_hard_skills": ", ".join(req.get("hard_skills") or []),
                "skill_overlap": ", ".join(overlap) if overlap else "NONE",
                "skill_overlap_count": len(overlap),
                "heuristic_score": round(heuristic_score, 2),
                "annotator_1": "",
                "annotator_2": "",
                "final_label": "",
                "notes": "",
            })

    return all_pairs


def main():
    print("Loading data...")
    jds = load_jds()
    cvs = load_cv_ner()
    print(f"Loaded {len(jds)} JDs and {len(cvs)} CVs")

    total = len(jds) * len(cvs)
    print(f"Generating ALL {total} pairs ({len(cvs)} CVs x {len(jds)} JDs)...")
    pairs = generate_all_pairs(jds, cvs)
    print(f"Generated {len(pairs)} pairs")

    df = pd.DataFrame(pairs)

    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "labeling_sheet.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to: {csv_path}")

    print(f"\n{'='*50}")
    print(f"LABELING SHEET GENERATED")
    print(f"{'='*50}")
    print(f"Total pairs: {len(df)}")
    print(f"Unique JDs: {df['jd_id'].nunique()}")
    print(f"Unique CVs: {df['cv_id'].nunique()}")
    print(f"Pairs per JD: {len(df) // df['jd_id'].nunique()}")
    print(f"\nHeuristic score distribution:")
    print(f"  Mean: {df['heuristic_score'].mean():.2f}")
    print(f"  High (>=2): {(df['heuristic_score'] >= 2).sum()} ({(df['heuristic_score'] >= 2).mean()*100:.0f}%)")
    print(f"  Low  (<2):  {(df['heuristic_score'] < 2).sum()} ({(df['heuristic_score'] < 2).mean()*100:.0f}%)")
    print(f"\nNext: python scripts/auto_annotate.py")


if __name__ == "__main__":
    main()
