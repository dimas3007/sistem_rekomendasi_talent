"""Tests for NER prompt templates and schema validation."""

import json
import pytest
from src.ner.prompts import NER_SCHEMA, NER_SYSTEM_PROMPT, NER_RULES, FEW_SHOT_EXAMPLE


class TestNERPrompts:
    def test_schema_has_required_fields(self):
        required_fields = [
            "name", "email", "phone", "education_level",
            "education_major", "university", "hard_skills",
            "soft_skills", "years_experience", "certifications",
            "languages", "job_titles", "industries", "location"
        ]
        for field in required_fields:
            assert field in NER_SCHEMA, f"Missing field: {field}"

    def test_system_prompt_not_empty(self):
        assert len(NER_SYSTEM_PROMPT) > 0
        assert "JSON" in NER_SYSTEM_PROMPT

    def test_rules_not_empty(self):
        assert len(NER_RULES) > 0

    def test_few_shot_example_valid_json(self):
        # Extract the JSON part from the example
        json_start = FEW_SHOT_EXAMPLE.index("{")
        json_end = FEW_SHOT_EXAMPLE.rindex("}") + 1
        json_str = FEW_SHOT_EXAMPLE[json_start:json_end]
        parsed = json.loads(json_str)

        assert parsed["name"] == "Muhammad Rizki"
        assert parsed["education_level"] == "S1"
        assert "Python" in parsed["hard_skills"]


def validate_ner_output(ner_result: dict) -> list:
    """Validate NER output against expected schema. Returns list of issues."""
    issues = []

    if not isinstance(ner_result, dict):
        return ["NER result is not a dictionary"]

    # Check required fields
    string_fields = ["name", "email", "phone", "education_level",
                     "education_major", "university", "location"]
    list_fields = ["hard_skills", "soft_skills", "certifications",
                   "languages", "job_titles", "industries"]

    for field in string_fields:
        if field in ner_result:
            val = ner_result[field]
            if val is not None and not isinstance(val, str):
                issues.append(f"{field} should be string or null, got {type(val)}")

    for field in list_fields:
        if field in ner_result:
            val = ner_result[field]
            if not isinstance(val, list):
                issues.append(f"{field} should be a list, got {type(val)}")

    if "years_experience" in ner_result:
        val = ner_result["years_experience"]
        if val is not None and not isinstance(val, (int, float)):
            issues.append(f"years_experience should be numeric, got {type(val)}")

    return issues


class TestNERValidation:
    def test_valid_output(self):
        valid = {
            "name": "Test Person",
            "email": "test@test.com",
            "phone": "081234567890",
            "education_level": "S1",
            "education_major": "Informatika",
            "university": "UI",
            "hard_skills": ["Python"],
            "soft_skills": ["Leadership"],
            "years_experience": 3.0,
            "certifications": [],
            "languages": ["Bahasa Indonesia"],
            "job_titles": ["Developer"],
            "industries": ["IT"],
            "location": "Jakarta",
        }
        issues = validate_ner_output(valid)
        assert len(issues) == 0

    def test_null_fields(self):
        valid = {
            "name": "Test",
            "email": None,
            "phone": None,
            "education_level": None,
            "education_major": None,
            "university": None,
            "hard_skills": [],
            "soft_skills": [],
            "years_experience": None,
            "certifications": [],
            "languages": [],
            "job_titles": [],
            "industries": [],
            "location": None,
        }
        issues = validate_ner_output(valid)
        assert len(issues) == 0

    def test_invalid_type(self):
        invalid = {
            "hard_skills": "Python, SQL",  # Should be list
        }
        issues = validate_ner_output(invalid)
        assert len(issues) > 0
