"""Shared document indexer for UI and CLI.

Clicking 'Index documents' in the app calls index_collection(), the same function
used by `python index_documents.py`.
"""
import json
from pathlib import Path

from config import DB_PATH, DOCUMENTS_DIR, EMBEDDING_MODEL, SUPPORTED_TYPES
from rag import chunk_text, vectorize
from storage import connect, extract_text, init_db, sha256_path, utc_now


def index_collection(directory: Path = DOCUMENTS_DIR, db_path: Path = DB_PATH,
                     force: bool = False, on_progress=None, embedder=None) -> dict:
    """Incrementally index changed files and remove records for deleted files.

    Each document is replaced only after its embeddings have been generated.
    The optional embedder is injectable so offline tests can avoid API calls.
    """
    directory.mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    files = sorted(f for f in directory.iterdir()
                   if f.is_file() and f.suffix.lower() in SUPPORTED_TYPES)
    result = {"total": len(files), "indexed": 0, "skipped": 0,
              "chunks_written": 0, "removed": 0, "errors": []}
    embed = embedder or vectorize
    with connect(db_path) as conn:
        previous = {row["filename"]: dict(row)
                    for row in conn.execute("SELECT * FROM documents")}
    for n, path in enumerate(files, 1):
        try:
            digest = sha256_path(path)
            past = previous.get(path.name)
            if past and past["sha256"] == digest and past["embedding_model"] == EMBEDDING_MODEL and not force:
                result["skipped"] += 1
                if on_progress:
                    on_progress(n, len(files), path.name, "unchanged")
                continue
            if path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("File exceeds the 10 MB size limit")
            content = extract_text(path.name, path.read_bytes())
            chunks = chunk_text(content)
            if not chunks:
                raise ValueError("No readable chunks")
            embeddings = embed(chunks)
            if len(embeddings) != len(chunks):
                raise RuntimeError("Embedding count mismatch")
            # Atomic document-level update: an API failure never erases its old index.
            with connect(db_path) as conn:
                conn.execute("DELETE FROM documents WHERE filename=?", (path.name,))
                cursor = conn.execute(
                    "INSERT INTO documents(filename,sha256,indexed_at,embedding_model,chunk_count) "
                    "VALUES(?,?,?,?,?)", (path.name, digest, utc_now(), EMBEDDING_MODEL, len(chunks))
                )
                conn.executemany(
                    "INSERT INTO chunks(document_id,chunk_number,content,embedding_json) VALUES(?,?,?,?)",
                    [(cursor.lastrowid, i, text, json.dumps(vector))
                     for i, (text, vector) in enumerate(zip(chunks, embeddings), 1)]
                )
            result["indexed"] += 1
            result["chunks_written"] += len(chunks)
            if on_progress:
                on_progress(n, len(files), path.name, "indexed")
        except Exception as error:
            result["errors"].append(f"{path.name}: {error}")
            if on_progress:
                on_progress(n, len(files), path.name, "error")
    # Only clean up no-longer-existing files if no indexing failures occurred.
    if not result["errors"]:
        active_names = {p.name for p in files}
        obsolete = set(previous) - active_names
        with connect(db_path) as conn:
            for filename in obsolete:
                conn.execute("DELETE FROM documents WHERE filename=?", (filename,))
        result["removed"] = len(obsolete)
    return result


def main():
    print("The Honest Assistant — SQLite document indexer")
    print(f"Document folder: {DOCUMENTS_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    def progress(n, total, name, status):
        print(f"[{n}/{total}] {name} — {status}")

    summary = index_collection(on_progress=progress)
    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
