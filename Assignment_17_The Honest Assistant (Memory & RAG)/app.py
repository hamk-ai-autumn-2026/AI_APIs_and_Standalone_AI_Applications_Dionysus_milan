"""Professional Streamlit UI for The Honest Assistant (Memory & RAG)."""
import html
import json
from pathlib import Path

import streamlit as st

from config import CHAT_MODEL, DOCUMENTS_DIR, EMBEDDING_MODEL, has_api_key
from index_documents import index_collection
from memory import ALLOWED_LANGUAGES, ALLOWED_STYLES, load_memory, save_memory, personal_memory_reply
from rag import answer_question, search_chunks
from storage import (
    append_message, create_conversation, delete_conversation, delete_document,
    get_messages, library_inventory, list_conversations, stats, upload_document,
)

st.set_page_config(
    page_title="The Honest Assistant | Memory & RAG", page_icon="🧠",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }
[data-testid="stSidebar"] { background: #F0F3FA; border-right: 1px solid #E3E8F3; }
.hero { background: linear-gradient(120deg, #14294E 2%, #284C92 55%, #6359B4 100%);
  padding: 31px 33px; border-radius: 19px; color: white; margin-bottom: 22px;
  box-shadow: 0 13px 36px rgba(26,50,115,.14); }
.hero h1 { margin: 0 0 6px; color: #fff; font-size: 32px; letter-spacing: -1px; font-weight: 800; }
.hero p { margin: 0; font-size: 14px; color: #E2E9FF; }
.eyebrow { font-weight: 700; letter-spacing: 2px; font-size: 10px; color: #C6D7FF; margin-bottom: 8px; }
.info-card { background: white; border: 1px solid #E2E9F5; border-radius: 14px;
  padding: 18px 21px; min-height: 89px; margin-bottom: 12px; }
.info-card h4 { margin: 0 0 6px; color: #16294B; font-size: 15px; }
.info-card p { margin: 0; color: #64748b; font-size: 13px; line-height: 1.5; }
.soft-pill { display: inline-block; background: #E8F7F0; color: #216B50; border-radius: 20px;
  padding: 4px 11px; font-weight: 700; font-size: 11px; margin-right: 6px; }
.soft-pill.gray { background: #EFF2F8; color: #4A6282; }
.stButton > button[kind="primary"] { border-radius: 10px; font-weight: 700; }
[data-testid="stChatMessage"] { background: white; border: 1px solid #E5EAF5;
  border-radius: 14px; margin-bottom: 10px; padding: 12px; }
[data-testid="stMetric"] { background: white; border: 1px solid #E5EAF5;
  border-radius: 13px; padding: 15px; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

prefs = load_memory()
counts = stats()

with st.sidebar:
    st.markdown("### 🧠 The Honest Assistant")
    st.caption("MEMORY  /  RAG  /  EVIDENCE")
    st.divider()
    page = st.radio(
        "Workspace", ["💬 Chat", "📚 Knowledge Base", "⚙️ Preferences"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Workspace status**")
    key_status = "🟢 API ready" if has_api_key() else "🟠 API key needed"
    st.caption(key_status)
    st.caption(f"📄 {counts['indexed_documents']} indexed document(s)")
    st.caption(f"🧩 {counts['chunks']} searchable chunks")
    st.caption(f"🤖 {CHAT_MODEL}")
    st.divider()
    st.markdown("**Your profile**")
    st.write(f"**{prefs['Name']}**")
    st.caption(f"{prefs['Preferred language']} · {prefs['Explanation style']}")
    st.caption(f"Maximum {prefs['Maximum answer lines']} answer lines")
    st.write("")
    st.caption("Private, local project · Files stored on this computer")


def hero(section: str, subtitle: str):
    st.markdown(
        "<div class='hero'>"
        "<div class='eyebrow'>THE HONEST ASSISTANT · MEMORY & RAG</div>"
        f"<h1>{html.escape(section)}</h1><p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def source_expander(chunks: list[dict], used: list[int] | None = None):
    if not chunks:
        return
    cited = set(used or [])
    with st.expander(f"🔎 View retrieved chunks ({len(chunks)})", expanded=False):
        st.caption("Cosine similarity measures relevance, not factual certainty. The cited chunks are marked below.")
        for index, chunk in enumerate(chunks, 1):
            title = "✅ Cited" if index in cited else "Retrieved"
            st.write(f"**[{index}] {chunk['source']} · Chunk {chunk['chunk_number']} · {title}**")
            st.caption(f"Similarity: {chunk['similarity']:.3f}")
            st.code(chunk["content"], language=None, wrap_lines=True)
            if index < len(chunks):
                st.divider()


def render_message(message: dict):
    with st.chat_message(message["role"], avatar="🧠" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("retrieved_json"):
            try:
                data = json.loads(message["retrieved_json"])
                ids = data.get("sources", [])
                chunks = data.get("chunks", [])
                if data.get("origin") == "memory":
                    st.caption("Based on saved preferences · memory.md")
                if ids:
                    st.caption("Verified source citations: " + ", ".join(f"[{n}]" for n in ids))
                elif chunks:
                    st.caption("No source adequately supported the answer.")
                source_expander(chunks, ids)
            except (TypeError, ValueError):
                st.caption("Saved source details are unavailable.")


def do_indexing():
    progress = st.progress(0, text="Preparing the document index...")

    def progress_callback(n: int, total: int, filename: str, status: str):
        progress.progress(n / max(1, total), text=f"{filename}: {status} ({n}/{total})")

    with st.spinner("Building embeddings and updating SQLite..."):
        summary = index_collection(on_progress=progress_callback)
    progress.empty()
    if summary["errors"]:
        for error in summary["errors"]:
            st.error(error)
    else:
        st.success(
            f"Index updated. {summary['indexed']} file(s) indexed, "
            f"{summary['skipped']} unchanged, {summary['chunks_written']} new chunks."
        )
    return summary


if page == "💬 Chat":
    hero("Ask. Verify. Trust.", "Answers grounded in your documents — personalised to you, with visible evidence.")
    left, middle, right = st.columns(3)
    left.metric("Indexed documents", counts["indexed_documents"])
    middle.metric("Knowledge chunks", counts["chunks"])
    right.metric("Saved conversations", counts["conversations"])
    st.write("")

    if not has_api_key():
        st.warning("To chat or index documents, add OPENAI_API_KEY to .env and restart the app.")
    if counts["chunks"] == 0:
        st.info("Your knowledge base has no indexed chunks. Open **📚 Knowledge Base** and click **Index documents** to get started.")

    chats = list_conversations()
    if not chats:
        st.session_state["active_chat"] = create_conversation()
        chats = list_conversations()
    available_ids = [c["id"] for c in chats]
    if st.session_state.get("active_chat") not in available_ids:
        st.session_state["active_chat"] = available_ids[0]
    chat_info = {c["id"]: c for c in chats}

    a, b, c = st.columns([5, 1.4, 1.4], vertical_alignment="bottom")
    with a:
        selected = st.selectbox(
            "Conversation", available_ids,
            index=available_ids.index(st.session_state["active_chat"]),
            format_func=lambda chat_id: chat_info[chat_id]["title"],
        )
        st.session_state["active_chat"] = selected
    with b:
        if st.button("＋ New chat", use_container_width=True):
            st.session_state["active_chat"] = create_conversation()
            st.rerun()
    with c:
        with st.popover("🗑 Delete chat", use_container_width=True):
            st.warning("Delete this chat permanently? Uploaded files and your preferences will stay.")
            if st.button("Confirm delete", type="primary", key="confirm_delete_chat"):
                delete_conversation(st.session_state["active_chat"])
                st.session_state.pop("active_chat", None)
                st.rerun()

    messages = get_messages(st.session_state["active_chat"])
    if not messages:
        name = html.escape(prefs["Name"])
        st.markdown(
            f"<div class='info-card'><h4>Welcome, {name} 👋</h4>"
            "<p>Ask questions about the files in your knowledge base. "
            "Every answer is evidence-checked, and unsupported questions receive a clear refusal.</p></div>",
            unsafe_allow_html=True,
        )
        st.caption("Try a sample question:")
        suggestions = [
            "Which sensor measures temperature and humidity in the greenhouse?",
            "What is the difference between a ROS 2 topic and service?",
            "What is the project's total budget in euros?",
        ]
        cols = st.columns(3)
        suggestion = None
        for i, example in enumerate(suggestions):
            if cols[i].button(example, key=f"suggest_{i}", use_container_width=True):
                suggestion = example
    else:
        suggestion = None
        for message in messages:
            render_message(message)
        export = "# The Honest Assistant — Conversation\n\n" + "\n\n".join(
            f"## {m['role'].title()}\n{m['content']}" for m in messages
        )
        st.download_button("↓ Export chat (.md)", export, file_name="honest_assistant_chat.md",
                           mime="text/markdown")

    question = st.chat_input(
        "Ask something about your indexed documents...",
        disabled=False,
    )
    question = question or suggestion
    if question:
        history = get_messages(st.session_state["active_chat"])
        append_message(st.session_state["active_chat"], "user", question)
        try:
            with st.spinner("Retrieving chunks and checking the evidence..."):
                memory_response = personal_memory_reply(question, prefs)
                if memory_response is not None:
                    chunks = []
                    result = {"answer": memory_response, "sources": [], "supported": True}
                    origin = "memory"
                else:
                    # Include the last user message for short follow-up queries.
                    recent_queries = [m["content"] for m in history if m["role"] == "user"]
                    retrieval_query = question
                    if len(question.split()) <= 7 and recent_queries:
                        retrieval_query = recent_queries[-1][:260] + "\nFollow-up: " + question
                    chunks = search_chunks(retrieval_query, top_k=4)
                    result = answer_question(question, chunks, preferences=prefs, history=history)
                    origin = "documents"
            append_message(st.session_state["active_chat"], "assistant", result["answer"],
                           retrieved={"chunks": chunks, "sources": result["sources"],
                                      "supported": result["supported"], "origin": origin})
        except Exception:
            append_message(st.session_state["active_chat"], "assistant",
                           "⚠️ The request could not be completed. Check your API key, billing, "
                           "network connection and document index, then try again.")
        st.rerun()

elif page == "📚 Knowledge Base":
    hero("Your knowledge base", "Upload documents, build the SQLite index, and inspect your collection — no terminal commands needed.")
    a, b, c = st.columns(3)
    a.metric("Files indexed", counts["indexed_documents"])
    b.metric("Searchable chunks", counts["chunks"])
    inventory = library_inventory()
    c.metric("Files on disk", len(inventory))
    st.write("")

    st.subheader("01 · Add documents")
    st.caption("Supported formats: TXT, MD, PDF and DOCX · 10 MB per file · PDFs must contain selectable text")
    uploads = st.file_uploader(
        "Choose one or more documents", type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    def save_selected() -> tuple[int, int]:
        good, failed = 0, 0
        for file in uploads or []:
            try:
                saved_name, state = upload_document(file.name, file.getvalue())
                if state == "uploaded":
                    good += 1
                    st.toast(f"Saved {saved_name}", icon="📄")
                else:
                    st.info(f"{saved_name} is already in your library — skipped.")
            except Exception as exc:
                failed += 1
                st.error(f"Could not save {file.name}: {exc}")
        return good, failed

    save_col, both_col = st.columns(2)
    if save_col.button("↓ Save selected files", disabled=not uploads, use_container_width=True):
        added, failed = save_selected()
        if added:
            st.success(f"{added} file(s) added. Click **Index documents** below to make them searchable.")
    if both_col.button("⚡ Save & index now", type="primary", disabled=not uploads or not has_api_key(),
                       use_container_width=True):
        added, failed = save_selected()
        if not failed:
            do_indexing()

    st.divider()
    st.subheader("02 · Index documents")
    st.markdown(
        "Clicking the button runs **the same `index_collection()` function** as "
        "`python index_documents.py`. Only new or changed files are embedded again."
    )
    if st.button("⚡ Index documents", type="primary", disabled=not has_api_key(),
                 use_container_width=True):
        do_indexing()
    if not has_api_key():
        st.caption("An API key is required to generate embeddings. Your uploaded files can still be saved locally.")

    st.divider()
    st.subheader("03 · Manage library")
    inventory = library_inventory()
    if not inventory:
        st.info("There are no documents yet. Upload a file above to start your library.")
    for info in inventory:
        with st.container(border=True):
            col1, col2, col3 = st.columns([5, 2, 1], vertical_alignment="center")
            with col1:
                st.markdown("**📄 " + html.escape(info["filename"]) + "**")
                st.caption(f"{info['size_kb']} KB · {info['chunks']} indexed chunks")
            with col2:
                status = info["status"]
                st.caption(("✅ " if status == "Indexed" else "🟡 ") + status)
            with col3:
                with st.popover("⋯", use_container_width=True):
                    st.write("Remove this file and its indexed chunks?")
                    if st.button("Delete file", key=f"remove_{info['filename']}", type="primary"):
                        delete_document(info["filename"])
                        st.rerun()
    st.caption("Deleted documents are also removed from search. Existing saved chat answers are historical snapshots.")

else:
    hero("Make it yours", "Change how the assistant addresses you and how much detail it includes. Settings survive restarts.")
    st.markdown(
        "<div class='info-card'><h4>Long-term memory</h4><p>These five preferences "
        "are stored in <code>memory.md</code>, not just in your browser session. "
        "They control presentation, never provide evidence for document facts.</p></div>",
        unsafe_allow_html=True,
    )
    with st.form("preferences_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your name", value=prefs["Name"], max_chars=80)
            default_lang = prefs["Preferred language"]
            language = st.selectbox(
                "Answer language", ALLOWED_LANGUAGES,
                index=ALLOWED_LANGUAGES.index(default_lang) if default_lang in ALLOWED_LANGUAGES else 0,
            )
            interests = st.text_input("Interests (optional)", value=prefs["Topics of interest"], max_chars=120)
        with col2:
            default_style = prefs["Explanation style"]
            style = st.selectbox(
                "Explanation style", ALLOWED_STYLES,
                index=ALLOWED_STYLES.index(default_style) if default_style in ALLOWED_STYLES else 0,
            )
            max_lines = st.slider("Maximum answer lines", 1, 30,
                                  value=max(1, min(int(prefs["Maximum answer lines"]), 30)))
            st.caption("Counts logical text lines, not visual wrapping on narrow screens. Source details are separate.")
        saved = st.form_submit_button("💾 Save preferences", type="primary", use_container_width=True)
    if saved:
        saved_memory = save_memory({
            "Name": name, "Preferred language": language, "Explanation style": style,
            "Maximum answer lines": str(max_lines), "Topics of interest": interests,
        })
        st.success("Your preferences have been saved to memory.md. New answers use them immediately.")
        st.session_state["profile_saved"] = True
    st.markdown("#### Current memory")
    current = load_memory()
    for key, value in current.items():
        st.markdown(f"**{html.escape(key)}:** {html.escape(value)}")
    st.caption("Personalisation cannot override the evidence-only rule or make the assistant invent unknown facts.")
