import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_RAW_CV = str(BASE_DIR / "data" / "raw" / "cvs")
DATA_RAW_JD = str(BASE_DIR / "data" / "raw" / "jds")
DATA_PROCESSED_CV = str(BASE_DIR / "data" / "processed" / "cv_parsed")
DATA_NER_CV = str(BASE_DIR / "data" / "processed" / "cv_ner")
DATA_PROCESSED_JD = str(BASE_DIR / "data" / "processed" / "jd_parsed")
DATA_FEATURES = str(BASE_DIR / "data" / "features")
MODEL_PATH = str(BASE_DIR / "models")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.1

# S-BERT
SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SBERT_EMBEDDING_DIM = 384

# XGBoost Defaults
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.85,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}

# Data Split
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# Feature columns (1 semantic + 14 structured = 15 total)
FEATURE_COLUMNS = [
    "cosine_similarity",
    "hard_skill_match_ratio",
    "hard_skill_match_count",
    "soft_skill_match_ratio",
    "soft_skill_match_count",
    "experience_gap",
    "experience_meets_req",
    "education_match",
    "education_level_numeric",
    "certification_count",
    "certification_relevance",
    "language_match_ratio",
    "industry_overlap",
    "location_match",
    "total_skill_count",
]
