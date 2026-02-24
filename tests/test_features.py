"""Tests for feature engineering modules."""

import pytest
from src.features.structured import compute_structured_features


class TestStructuredFeatures:
    """Test the 14 structured numerical features."""

    @pytest.fixture
    def sample_cv(self):
        return {
            "name": "Test Person",
            "education_level": "S1",
            "education_major": "Informatika",
            "hard_skills": ["Python", "SQL", "Excel"],
            "soft_skills": ["Communication", "Teamwork"],
            "years_experience": 3.0,
            "certifications": ["AWS Certified"],
            "languages": ["Bahasa Indonesia", "English"],
            "industries": ["Technology"],
            "location": "Jakarta",
            "job_titles": ["Data Analyst"],
        }

    @pytest.fixture
    def sample_jd_req(self):
        return {
            "education_level": "S1",
            "hard_skills": ["Python", "SQL", "Tableau", "Excel"],
            "soft_skills": ["Communication", "Analytical Thinking"],
            "min_experience_years": 2,
            "certifications": [],
            "languages": ["Bahasa Indonesia", "English"],
            "industries": ["Technology"],
            "location": "Jakarta",
        }

    def test_output_has_14_features(self, sample_cv, sample_jd_req):
        features = compute_structured_features(sample_cv, sample_jd_req)
        assert len(features) == 14

    def test_hard_skill_match(self, sample_cv, sample_jd_req):
        features = compute_structured_features(sample_cv, sample_jd_req)
        # CV has Python, SQL, Excel matching JD's Python, SQL, Tableau, Excel
        # 3 matches out of 4 required
        assert features["hard_skill_match_ratio"] == 3 / 4
        assert features["hard_skill_match_count"] == 3

    def test_soft_skill_match(self, sample_cv, sample_jd_req):
        features = compute_structured_features(sample_cv, sample_jd_req)
        # CV has Communication matching, Teamwork not in JD
        assert features["soft_skill_match_ratio"] == 1 / 2
        assert features["soft_skill_match_count"] == 1

    def test_experience_gap(self, sample_cv, sample_jd_req):
        features = compute_structured_features(sample_cv, sample_jd_req)
        # CV: 3 years, JD requires: 2 years => gap = +1
        assert features["experience_gap"] == 1.0
        assert features["experience_meets_req"] == 1

    def test_experience_below_req(self, sample_cv, sample_jd_req):
        sample_cv["years_experience"] = 1.0
        features = compute_structured_features(sample_cv, sample_jd_req)
        assert features["experience_gap"] == -1.0
        assert features["experience_meets_req"] == 0

    def test_education_match(self, sample_cv, sample_jd_req):
        features = compute_structured_features(sample_cv, sample_jd_req)
        # Both S1 => match
        assert features["education_match"] == 1
        assert features["education_level_numeric"] == 5  # S1 = 5

    def test_education_below(self, sample_cv, sample_jd_req):
        sample_cv["education_level"] = "D3"
        features = compute_structured_features(sample_cv, sample_jd_req)
        assert features["education_match"] == 0
        assert features["education_level_numeric"] == 4  # D3 = 4

    def test_language_match(self, sample_cv, sample_jd_req):
        features = compute_structured_features(sample_cv, sample_jd_req)
        assert features["language_match_ratio"] == 1.0  # Both languages match

    def test_location_match(self, sample_cv, sample_jd_req):
        features = compute_structured_features(sample_cv, sample_jd_req)
        assert features["location_match"] == 1

    def test_location_mismatch(self, sample_cv, sample_jd_req):
        sample_cv["location"] = "Surabaya"
        features = compute_structured_features(sample_cv, sample_jd_req)
        assert features["location_match"] == 0

    def test_empty_cv_skills(self, sample_jd_req):
        empty_cv = {
            "hard_skills": [],
            "soft_skills": [],
            "years_experience": 0,
            "education_level": None,
            "certifications": [],
            "languages": [],
            "industries": [],
            "location": None,
        }
        features = compute_structured_features(empty_cv, sample_jd_req)
        assert features["hard_skill_match_ratio"] == 0
        assert features["soft_skill_match_ratio"] == 0
        assert features["total_skill_count"] == 0

    def test_total_skill_count(self, sample_cv, sample_jd_req):
        features = compute_structured_features(sample_cv, sample_jd_req)
        # 3 hard + 2 soft = 5
        assert features["total_skill_count"] == 5
