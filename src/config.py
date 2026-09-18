import os

from dotenv import load_dotenv

load_dotenv()


# Regression detection settings

SIGNIFICANCE_LEVEL = float(
    os.getenv("SIGNIFICANCE_LEVEL", "0.05")
)

MIN_ACCURACY_DROP = float(
    os.getenv("MIN_ACCURACY_DROP", "0.02")
)

MIN_F1_DROP = float(
    os.getenv("MIN_F1_DROP", "0.02")
)


# Project paths

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


# Model paths

BASELINE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "baseline_model.pkl"
)

CANDIDATE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "candidate_model.pkl"
)


# Golden test set

TEST_DATA_PATH = os.path.join(
    DATA_DIR,
    "golden_sets",
    "adult_income_test.csv"
)


# Evaluation history database

DB_PATH = os.path.join(
    DATA_DIR,
    "run_history.db"
)