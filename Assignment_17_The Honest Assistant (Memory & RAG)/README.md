# The Honest Assistant (Memory & RAG)

**Assignment #17**, A single-user Streamlit app. It remembers profile preferences, accepts user documents through the browser, embeds them into **SQLite**, retrieves relevant chunks, and **refuses unsupported factual questions**. No external vector database is required.

## Features

- **Personalized memory:** Name, language (English, Nepali, Finnish, Hindi, Spanish, French or auto-detect), explanation style, maximum answer lines (1–30), interests; stored in `memory.md` after restarts.
- **Documents:** Upload `.txt`, `.md`, `.pdf` (selectable text), `.docx` in the UI. Support for duplicates, file management and deletion. Maximum 10 MB/file, 100 PDF pages, 500,000 extracted text characters.
- **Index in UI:** Click **Index documents**. It invokes exactly the same Python function that `python index_documents.py` invokes. Also includes **Save & index now** for one-click upload + indexing. Automatically skip unchanged files and re-embed updated files.
- **SQLite RAG:** Chunk text into overlapping passages, embed with `text-embedding-3-small`, store chunks + vectors in `data/knowledge.db`, compare query embedding by cosine similarity.
- **Show sources:** Each assistant response can expand a list of retrieved chunks, including filenames, chunk numbers, full text and similarity; cited chunks are marked.
- **Honest answers:** OpenAI is instructed to answer only from excerpts, and to return a structured support decision and evidence quotes. The app checks quoted evidence against retrieved chunks and refuses when support is insufficient. (This is a guardrail, not a 100% guarantee.)
- **Local preference recall:** Questions such as `What is my name?` can be answered directly from saved memory, without an API call. Refusals are available in the chosen language.
- **Persistent chat:** Previous conversations stored in SQLite. New chats, delete, switch, and download a Markdown conversation export.

## 1 — Requirements

Python **3.10+**, an OpenAI API key with active API billing, an internet connection for the API calls, and a browser. No GitHub login or cloud deployment is needed.

## 2 — Install

Extract the ZIP, open the `the_honest_assistant` project folder in VS Code, and open its terminal.

**Windows PowerShell:**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3 — Configure the API key

Copy `.env.example` to `.env` in the project root:

```env
OPENAI_API_KEY=your_real_key_here
CHAT_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
```

Create a key at https://platform.openai.com/api-keys . The API may require billing; ChatGPT subscriptions do not cover API calls. **Never commit `.env` to GitHub or share your key.** If you edit `.env` while Streamlit runs, restart it.

## 4 — Launch

```bash
streamlit run app.py
```

Open the local address printed by Streamlit (usually `http://localhost:8501`).

### In the browser

1. **⚙️ Preferences** → set *Name, Answer language, Explanation style, Maximum answer lines* (and optional interests). Click **Save preferences**. They are written to `memory.md`.
2. **📚 Knowledge Base** → upload files. Choose **Save selected files** (staging) or **Save & index now** (both steps). Press **Index documents** to index staged files or the three already-included sample documents. The progress bar runs the embedding/indexing pipeline, creating the SQLite database.
3. **💬 Chat** → ask questions about documents. Expand **View retrieved chunks** to see evidence. Ask an unknown question to verify refusal. Create and switch between chats and export an existing chat.

### The CLI alternative

```bash
python index_documents.py
```

**The UI button and CLI run the same `index_collection()` function.** Both require a working API key to embed new or modified files. Re-indexing an unchanged collection makes no new embeddings API call.

## Demonstrate the assignment

Included example files: `documents/greenhouse_project.txt`, `documents/ros2_basics.txt`, `documents/rag_overview.txt`.

| Question | Expected behavior |
|---|---|
| `Which sensor measures temperature and humidity in the greenhouse?` | Supported answer about DHT22 with cited source |
| `What is the difference between a ROS 2 topic and service?` | Supported explanation of publish/subscribe and request/response |
| `What is the total project budget in euros?` | `I don't know based on the provided documents.` (or a translated refusal when your preferred language is not English; assuming the uploaded files do not mention a budget) |
| `What is my name?` | Answers from locally saved `memory.md`, without inventing a document citation |

**Screenshots to submit:** saved preferences in settings / indexed files / a cited answer with expanded chunks / the refused unsupported question.

## File structure

```text
the_honest_assistant/
├── app.py                # Streamlit UI: chat, library, preferences
├── memory.py             # Read/write and validate memory.md
├── memory.md             # Five persistent preferences
├── index_documents.py    # Reusable UI + CLI document indexer
├── rag.py                # Chunking, embeddings, cosine search, answer guardrails
├── storage.py            # File processing, SQLite tables, chat history
├── config.py             # Paths, API models, configuration
├── documents/            # Example documents and your future uploaded files
├── data/                 # Generated knowledge.db appears here
├── .streamlit/config.toml# App colors, upload limit
├── .env.example          # Copy to .env and add secret key
├── requirements.txt
└── tests/test_core.py    # Offline tests
```

## How the retrieval pipeline works

```text
User adds files (.txt, .md, .pdf, .docx)
               ↓
         Text extraction
               ↓
    Chunking with word overlap
               ↓
  Embedding API (batches of 32)
               ↓
SQLite: documents + chunks + embeddings
               ↓
User question → embedding → cosine similarity → top 4 chunks
               ↓
Model checks whether the chunks explicitly support the answer
               ↓
Exact quote / source ID verification
               ↓
Answer with inspectable evidence  OR  fixed refusal
```

## Tests (offline)

```bash
python -m unittest discover -s tests -v
```

Offline tests mock OpenAI. They check saved memory, document uploads, duplicates, incremental index, search, grounded answers and refusal, failed re-index safety, evidence validation, saved chat history, PDF/DOCX extraction, and localized refusals. Real network calls still need an API key for end-to-end testing.

## What makes this safe and what it does NOT do

- **Single trusted local user:** No login, user accounts, encryption at rest or tenant isolation. **Do not publish this app on a public server without authentication, request quotas, access controls, and security review.** Anyone who can reach a public deployment might access locally stored documents and chats.
- The app sends selected document chunks and questions to the OpenAI API; during indexing, it sends extracted text for embeddings. **Do not upload confidential data unless you are authorized to send it to an API provider.**
- Retrieval passages are treated as untrusted data and do not grant instruction authority. Quote checking reduces fabricated citations, **but the AI may still mistakenly judge sufficiency, so no absolute hallucination guarantee is possible**.
- Similarity scores indicate relevance, not correctness; only an evidence decision determines the answer.
- A maximum-line setting limits newline-separated *logical* lines; text may visibly wrap across more screen lines on a phone. Cited source details are always available separately.
- Scanned image-only PDF pages won't work without OCR; encrypted PDF may fail.
- SQLite stores embeddings but **doesn't provide a vector-search index**. Brute-force cosine similarity is appropriate for small to medium coursework collections, not millions of chunks.
- Deleting a document removes current indexed chunks, but prior saved chat answers remain historical snapshots. You can delete those conversations separately.
- This demo stores the personal preferences locally in `memory.md`. The memory is **not** a learned memory system and does not automatically infer new personal details from your chat.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Missing OpenAI API key warning | Confirm `.env` exists in the project root and has `OPENAI_API_KEY`; restart Streamlit. |
| Indexing fails with 401/429 | Verify the key, model access, project billing and limits. |
| PDF says no selectable text | Try a born-digital PDF or add OCR separately. |
| No relevant answers | Upload relevant files, then run **Index documents**; try a precise question. |
| Changing `EMBEDDING_MODEL` | Re-index all documents with the new embedding model. Use `python index_documents.py` after deleting `data/knowledge.db` or add a forced re-index in code. |
| Uploaded file seems stuck | Files must contain readable text; check the specific error beside its filename. |

## Extensions you could build later

For a deployable multi-user service: authentication, per-user encrypted storage, PDF OCR, reranking, a faster vector search extension, and citations linked directly to highlighted PDF pages.
