"""Central paths and application settings."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "knowledge.db"
MEMORY_PATH = ROOT / "memory.md"
load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_CHARS = 500_000
SUPPORTED_TYPES = {".txt", ".md", ".pdf", ".docx"}


def ensure_dirs():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def has_api_key():
    value = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(value and value not in {"your_openai_api_key_here", "replace_with_your_openai_api_key"})
