import os
from dotenv import load_dotenv
from pathlib import Path

# Load backend/.env before any tests run
ENV_PATH = Path(__file__).parent / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    raise RuntimeError("backend/.env file not found for tests")

# Optional safety check for e2e tests
REQUIRED_VARS = ["OPENAI_API_KEY", "SUPABASE_DB_URL"]

missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing required env vars for tests: {missing}")
