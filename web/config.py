import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_BASE_URL = os.getenv(
    "SOLAR_API_BASE_URL",
    "http://127.0.0.1:8001/api",
).rstrip("/")
