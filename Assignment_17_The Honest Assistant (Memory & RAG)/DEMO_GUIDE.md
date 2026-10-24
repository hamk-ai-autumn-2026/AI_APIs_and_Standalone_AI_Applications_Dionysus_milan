# Assignment #17 — Demo & Explanation Guide

## Project title

**The Honest Assistant (Memory & RAG)**

> An AI assistant with persistent user preferences and a SQLite-based Retrieval-Augmented Generation pipeline. It extracts text from user-uploaded files, uses embeddings to retrieve relevant chunks, makes its sources visible, and refuses to invent facts not supported by the documents.

## 1. Memory: what is remembered?

The Preferences page lets a user save five preferences in `memory.md`:

1. Name — e.g., Alex
2. Preferred response language — English, Nepali, Finnish, Hindi and others
3. Explanation style — step by step, concise, beginner friendly, detailed, technical
4. Maximum answer lines — 1–30 logical lines
5. Topics of interest — optional text

`memory.py` uses file reading/writing. Preferences are **not** just stored in Streamlit session state, so they survive application restarts. Preferences influence the AI's response style only, and never count as evidence about a document. A user can also ask "What is my name?" and receive a deterministic answer from the saved preferences.

## 2. RAG: where do the facts come from?

1. User uploads `.txt`, `.md`, `.pdf` or `.docx` files in the Knowledge Base page.
2. Text is extracted using the built-in text reader, pypdf or python-docx.
3. `chunk_text()` splits text into chunks of up to 180 words, with 35 words overlapping.
4. OpenAI `text-embedding-3-small` converts each chunk into a numerical vector.
5. `storage.py` persists source filenames, chunk numbers, text and vectors in SQLite.
6. A question receives its own embedding.
7. `search_chunks()` compares the question embedding with the stored vectors using cosine similarity and retrieves the 4 highest-scoring chunks.
8. The model answers using **only those retrieved excerpts**.

The SQLite database is at `data/knowledge.db`. This is SQLite **RAG**, even though cosine similarity is performed in Python rather than by a vector extension.

## 3. Why does indexing work from the UI?

The Knowledge Base page executes:

```python
from index_documents import index_collection
result = index_collection(on_progress=progress_callback)
```

The CLI version `python index_documents.py` calls `index_collection()` too. There is only **one indexing implementation**, so clicking **Index documents** and running the terminal command do the same job.

The indexer skips unchanged files, re-embeds changed ones and removes deleted files. An API failure leaves the previous version of a document's index untouched.

## 4. Why is it called "Honest"?

The assistant asks the model for a structured support decision, answer and evidence quote. If the model says the answer is unsupported, or if the evidence quote does not appear verbatim in the cited retrieved chunk, the code displays a refusal.

This is a pragmatic guardrail, **not a proof** that all generated answers are correct. The retrieved chunks are shown so the user can independently verify the answer.


