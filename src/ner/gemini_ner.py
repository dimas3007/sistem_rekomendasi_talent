import json
import time

from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE
from src.ner.prompts import NER_SYSTEM_PROMPT, NER_SCHEMA, NER_RULES, FEW_SHOT_EXAMPLE

_client = genai.Client(api_key=GEMINI_API_KEY)


def extract_entities(cv_text: str, max_retries: int = 3) -> dict | None:
    """
    Ekstrak entitas terstruktur dari teks CV menggunakan Gemini.

    Args:
        cv_text: Raw CV text from PDF parser.
        max_retries: Number of retries on failure.

    Returns:
        Dictionary of extracted entities, or None on failure.
    """
    prompt = f"""Extract structured data from this CV text.

Output Schema:
{NER_SCHEMA}

{NER_RULES}

{FEW_SHOT_EXAMPLE}

--- CV TEXT START ---
{cv_text[:8000]}
--- CV TEXT END ---

Return ONLY the JSON object, no other text."""

    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=NER_SYSTEM_PROMPT,
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=2048,
                ),
            )

            # Parse JSON from response
            raw = response.text.strip()

            # Clean markdown code blocks if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)

            # Validate required fields exist
            required_fields = [
                "name", "hard_skills", "soft_skills",
                "years_experience", "education_level"
            ]
            for field in required_fields:
                if field not in result:
                    result[field] = None if field in ("name", "education_level") else []

            return result

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None

        except Exception as e:
            print(f"NER Error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
