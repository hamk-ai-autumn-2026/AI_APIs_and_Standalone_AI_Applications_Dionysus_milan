"""Local file library and SQLite persistence for documents and conversations.

This example is intended for one trusted user on a local computer; no login or
multi-tenant isolation is provided.
"""
import hashlib
import io
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from config import (
    DATA_DIR, DB_PATH, DOCUMENTS_DIR, EMBEDDING_MODEL, MAX_EXTRACTED_CHARS,
    MAX_UPLOAD_BYTES, SUPPORTED_TYPES, ensure_dirs,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path = DB_PATH):
    """Commit or roll back, AND close the SQLite connection on exit."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                chunk_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                UNIQUE(document_id, chunk_number)
            );
            CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks(document_id);
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                retrieved_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_conversation_id ON messages(conversation_id);
        """)


def safe_filename(name: str) -> str:
    basename = name.replace("\\", "/").split("/")[-1]
    # Use an application-chosen basename; no user input is treated as a path.
    basename = re.sub(r"[^a-zA-Z0-9._ -]", "_", basename).strip(" .")[:150]
    if not basename or basename in {".", ".."}:
        raise ValueError("Invalid filename.")
    if Path(basename).suffix.lower() not in SUPPORTED_TYPES:
        raise ValueError("Only TXT, MD, PDF and DOCX files are supported.")
    return basename


def extract_text(name: str, data: bytes) -> str:
    """Extract readable text; scanned/image-only PDFs need OCR (not included)."""
    suffix = Path(name).suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            content = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("Text files must use UTF-8 encoding.") from error
    elif suffix == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            if len(reader.pages) > 100:
                raise ValueError("PDF has more than 100 pages; split it into smaller files.")
            content = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("Could not read this PDF. Make sure it is not encrypted or damaged.") from error
    elif suffix == ".docx":
        from docx import Document
        try:
            doc = Document(io.BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs]
            tables = [" | ".join(cell.text for cell in row.cells)
                      for table in doc.tables for row in table.rows]
            content = "\n".join(paragraphs + tables)
        except Exception as error:
            raise ValueError("Could not read this Word document.") from error
    else:
        raise ValueError("Unsupported document type.")
    content = content.replace("\x00", "").strip()
    if not content:
        raise ValueError("No selectable text was found (scanned PDFs require OCR).")
    if len(content) > MAX_EXTRACTED_CHARS:
        raise ValueError("Extracted text is too large (limit: 500,000 characters).")
    return content


def upload_document(name: str, data: bytes, directory: Path = DOCUMENTS_DIR) -> tuple[str, str]:
    """Validate, then save locally. Return (saved filename, status)."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Maximum size is 10 MB per file.")
    if not data:
        raise ValueError("The file is empty.")
    basename = safe_filename(name)
    extract_text(basename, data)  # validate before writing
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / basename
    counter = 2
    while destination.exists():
        if destination.read_bytes() == data:
            return destination.name, "already exists"
        destination = directory / f"{Path(basename).stem}_{counter}{Path(basename).suffix}"
        counter += 1
    # Exclusive creation prevents overwriting an existing document.
    with destination.open("xb") as writer:
        writer.write(data)
    return destination.name, "uploaded"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def library_inventory(directory: Path = DOCUMENTS_DIR, db_path: Path = DB_PATH) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    with connect(db_path) as conn:
        indexed = {r["filename"]: dict(r) for r in conn.execute("SELECT * FROM documents")}
    results = []
    for path in sorted(directory.iterdir(), key=lambda x: x.name.lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_TYPES:
            continue
        item = indexed.get(path.name)
        digest = sha256_path(path)
        status = "Ready to index" if not item else (
            "Indexed" if item["sha256"] == digest and item["embedding_model"] == EMBEDDING_MODEL else "Needs re-indexing"
        )
        results.append({
            "filename": path.name, "size_kb": round(path.stat().st_size / 1024, 1),
            "sha256": digest, "status": status,
            "chunks": item["chunk_count"] if item else 0,
            "indexed_at": item["indexed_at"] if item else None,
            "embedding_model": item["embedding_model"] if item else None,
        })
    return results


def delete_document(filename: str, directory: Path = DOCUMENTS_DIR, db_path: Path = DB_PATH) -> None:
    path = directory / safe_filename(filename)
    if path.name != filename or not path.is_file():
        raise ValueError("Document not found.")
    path.unlink()
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM documents WHERE filename=?", (filename,))


def stats(db_path: Path = DB_PATH) -> dict:
    init_db(db_path)
    with connect(db_path) as conn:
        return {
            "indexed_documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "conversations": conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
        }


def create_conversation(db_path: Path = DB_PATH) -> str:
    init_db(db_path)
    conversation_id = uuid.uuid4().hex
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations(id, title, created_at) VALUES(?,?,?)",
            (conversation_id, "New conversation", utc_now()),
        )
    return conversation_id


def list_conversations(db_path: Path = DB_PATH) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM conversations ORDER BY created_at DESC, rowid DESC LIMIT 100"
        )]


def get_messages(conversation_id: str, db_path: Path = DB_PATH) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT role, content, retrieved_json, created_at FROM messages "
            "WHERE conversation_id=? ORDER BY id", (conversation_id,)
        )]


def append_message(conversation_id: str, role: str, content: str,
                   retrieved: list | None = None, db_path: Path = DB_PATH) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid chat role")
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("""INSERT INTO messages(conversation_id,role,content,retrieved_json,created_at)
                        VALUES(?,?,?,?,?)""",
                     (conversation_id, role, content, json.dumps(retrieved) if retrieved is not None else None, utc_now()))
        if role == "user":
            conn.execute("UPDATE conversations SET title=? WHERE id=? AND title='New conversation'",
                         (content.strip().replace("\n", " ")[:46] or "Conversation", conversation_id))


def delete_conversation(conversation_id: str, db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
