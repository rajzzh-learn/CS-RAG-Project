"""
Streamlit chat interface for the CS Tutor RAG Agent.
Run with:  streamlit run cs_tutor/app.py
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path for Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from cs_tutor.config import get_config
from cs_tutor.ingest import build_vector_store
from cs_tutor.rag_chain import build_rag_chain, convert_history
from cs_tutor.file_utils import (
    extract_text,
    image_to_base64_uri,
    is_image,
    normalize_math,
    sanitize_html,
    split_html_blocks,
    SUPPORTED_EXTS,
)
from cs_tutor.components.paste_button import paste_image_button

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CS Tutor — Class 12 CBSE (Python 083)",
    page_icon="💻",
    layout="wide",
)

st.title("💻 Class 12 Computer Science Tutor")
st.caption("Powered by your NCERT textbooks, chapter notes, solutions, and PYQs — CBSE Board Exam Edition")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    current_provider = get_config("LLM_PROVIDER", "groq").lower()
    st.caption(f"🤖 **Active Provider**: `{current_provider.upper()}`")
    st.header("🗂️ Chapters & Syllabus")
    st.markdown("""
**Unit 1: Computational Thinking and Programming – 2**
1. Exception Handling in Python
2. File Handling (Text, Binary, CSV)
3. Using Python Libraries
4. Data Structures: Stack (Push/Pop)
5. Data Structures: Queue (Enqueue/Dequeue)
6. Searching (Linear, Binary) & Sorting (Bubble, Insertion)

---

**Unit 2: Computer Networks**
1. Evolution, Data Communication & Media
2. Network Topologies & Types (PAN, LAN, MAN, WAN)
3. Network Devices (Hub, Switch, Router, Gateway)
4. Protocols (TCP/IP, HTTP/S, FTP, SMTP, POP3, VoIP)
5. Cyber Safety, Security & Case Study Layouts

---

**Unit 3: Database Management & SQL**
1. Database Concepts & Relational Data Model
2. Structured Query Language (SQL - DDL & DML)
3. Aggregate Functions, Grouping & Joins
4. Python-SQL Database Connectivity
""")
    st.divider()

    # ── Study Material Links ──────────────────────────────────────────────
    REPO = "https://github.com/rajzzh-learn/CS-RAG-Project/tree/main"
    st.header("📚 Study Material")
    st.markdown(f"""
| Folder | Link |
|--------|------|
| 📖 NCERT Books | [book]({REPO}/book) |
| 📝 NCERT Solutions | [Ncert Solutions]({REPO}/Ncert%20Solutions) |
| 📌 Revision Notes | [Notes]({REPO}/Notes) |
| 📋 Previous Year Papers (PYQ) | [PYQ]({REPO}/PYQ) |
| 💻 Lab Programs | [Lab Programs]({REPO}/Lab%20Programs) |
| 🎯 SSM Question Bank | [SSM Question Bank]({REPO}/SSM%20Question%20Bank) |
""")
    st.divider()

    st.markdown("**💡 Try asking:**")
    st.markdown("""
- *Write a Python program to read and count words in a text file*
- *Explain binary file handling using pickle dump and load with EOFError*
- *Write Push and Pop functions for a list-based Stack in Python*
- *Solve a 5-mark Computer Networks Case Study for server & repeater placement*
- *Write SQL queries with GROUP BY, HAVING, and Equi-Join*
""")
    st.divider()
    if st.button("🔄 Rebuild Vector Store"):
        with st.spinner("Re-ingesting all PDFs … (this takes a couple of minutes)"):
            st.session_state.pop("rag_chain", None)
            st.session_state.pop("vector_store", None)
            build_vector_store(force_rebuild=True)
        st.success("Vector store rebuilt!")

# ── Init vector store & chain (cached in session state) ───────────────────
if "vector_store" not in st.session_state:
    with st.spinner("⚙️ Loading your study material … first load takes ~1 min"):
        st.session_state["vector_store"] = build_vector_store(force_rebuild=False)

# Re-instantiate if provider changed or not initialized
target_provider = get_config("LLM_PROVIDER", "groq").lower()
if "rag_chain" not in st.session_state or st.session_state.get("_active_provider") != target_provider:
    try:
        st.session_state["rag_chain"] = build_rag_chain(st.session_state["vector_store"])
        st.session_state["_active_provider"] = target_provider
    except Exception as e:
        st.error(
            f"⚠️ **LLM Initialization Error**: {e}\n\n"
            "👉 If using Groq, ensure `LLM_PROVIDER = \"groq\"` and `GROQ_API_KEY = \"gsk_...\"` in `.env` / Streamlit Secrets.\n"
            "👉 If using watsonx, configure `WATSONX_API_KEY` and `WATSONX_PROJECT_ID`.\n"
            "👉 If using OpenAI, configure `OPENAI_API_KEY`."
        )
        st.stop()

# ── Rendering helpers ──────────────────────────────────────────────────────
def render_answer(text: str) -> None:
    """Render markdown with KaTeX math; ```html blocks get a sanitized live preview."""
    for kind, chunk in split_html_blocks(text):
        if kind == "html":
            st.caption("🌐 Rendered HTML preview")
            st.html(sanitize_html(chunk))
            with st.expander("View HTML source", expanded=False):
                st.code(chunk, language="html")
        else:
            st.markdown(normalize_math(chunk))


def render_attachments(attachments: list[dict]) -> None:
    for att in attachments:
        st.caption(f"📎 **Attached:** `{att['name']}`")
        if att.get("preview") is not None:
            with st.expander(f"👁️ Preview — {att['name']}", expanded=False):
                if att["is_image"]:
                    st.image(att["preview"], use_container_width=True)
                else:
                    st.text(str(att["preview"])[:2000])


# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Class 12 Computer Science teacher, here to help you "
                "ace your **CBSE Board Exams (Python 083)**.\n\n"
                "Tell me which chapter, Python code, SQL query, or networking case study you'd like to work on! "
                "You can also **attach images, PDF, Word, PowerPoint, Excel, or text/HTML files** using the 📎 button below."
            ),
        }
    ]

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        render_attachments(msg.get("attachments", []))
        render_answer(msg["content"])


# ── Helper: call vision-capable LLM directly (image path) ──────────────────
def _ask_vision_llm(question: str, image_data_uris: list[str], context_text: str) -> str:
    """
    Send a multimodal (text + image) message to a vision-capable LLM.
    Vision is always handled by Google Gemini Flash (free tier) regardless of LLM_PROVIDER,
    since Groq has no vision models and OpenAI requires paid credits.
    Falls back gracefully when GOOGLE_API_KEY is not set.
    Returns the answer string.
    """
    import base64, re
    from google import genai
    from google.genai import types

    google_key = get_config("GOOGLE_API_KEY")
    if not google_key:
        return (
            "⚠️ **Image analysis requires a `GOOGLE_API_KEY`** (free).\n\n"
            "👉 Get one at https://aistudio.google.com/app/apikey — it's free, no billing needed.\n"
            "Then add `GOOGLE_API_KEY = \"AIza...\"` to your `.env` file or Streamlit Secrets and reload the app."
        )

    # Extract raw base64 bytes from each data URI (data:<mime>;base64,<data>)
    image_parts = []
    for uri in image_data_uris:
        match = re.match(r"data:(?P<mime>[^;]+);base64,(?P<data>.+)", uri)
        if not match:
            return "⚠️ Could not parse an attached image. Please try uploading it again."
        image_parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(match.group("data")),
                mime_type=match.group("mime"),
            )
        )

    prompt = (
        "You are an expert Class 12 CBSE Computer Science teacher.\n"
        "Write math in LaTeX using $...$ (inline) or $$...$$ (block).\n\n"
        f"Context from the student's study materials:\n{context_text}\n\n"
        f"The student has attached image(s) and asks:\n{question}"
    )

    client = genai.Client(api_key=google_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[prompt, *image_parts],
    )
    return response.text


# ── Input area: file uploader + paste button + chat input ──────────────────
ext_list = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTS))

col_upload, col_paste = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_files = st.file_uploader(
        f"📎 Attach files — {ext_list}",
        type=sorted(SUPPORTED_EXTS),
        accept_multiple_files=True,
        label_visibility="visible",
        help="Attach one or more files (images, PDF, Word, PowerPoint, Excel, text/HTML) to include as extra context. Max 10 MB each.",
    )
with col_paste:
    paste_result = paste_image_button(
        "📋 Paste image",
        background_color="#444654",
        hover_background_color="#565869",
        key="clipboard_paste",
    )

if user_input := st.chat_input("Ask your CS teacher …"):
    # ── Process attachments (any number, mixed formats) ───────────────────
    attachments: list[dict] = []             # {name, is_image, preview}
    attachment_texts: list[str] = []         # extracted text blocks
    attachment_image_uris: list[str] = []    # base64 data URIs (images)

    if paste_result.image_data is not None:
        import io as _io
        buf = _io.BytesIO()
        paste_result.image_data.save(buf, format="PNG")
        file_bytes = buf.getvalue()
        uri, _ = image_to_base64_uri(file_bytes, "pasted-image.png")
        attachment_image_uris.append(uri)
        attachments.append({"name": "pasted-image.png", "is_image": True, "preview": file_bytes})

    for uploaded in uploaded_files or []:
        file_bytes = uploaded.getvalue()
        try:
            if is_image(uploaded.name):
                uri, _ = image_to_base64_uri(file_bytes, uploaded.name)
                attachment_image_uris.append(uri)
                attachments.append({"name": uploaded.name, "is_image": True, "preview": file_bytes})
            else:
                text = extract_text(file_bytes, uploaded.name)
                attachment_texts.append(f"--- Attached file: {uploaded.name} ---\n{text}")
                attachments.append({"name": uploaded.name, "is_image": False, "preview": text})
        except Exception:
            st.warning(f"⚠️ Could not read `{uploaded.name}` — it may be corrupted, encrypted, or too large. Skipped.")

    user_msg: dict = {"role": "user", "content": user_input, "attachments": attachments}
    st.session_state["messages"].append(user_msg)

    # Show student message
    with st.chat_message("user"):
        render_attachments(attachments)
        st.markdown(user_input)

    # ── Get answer from RAG chain ─────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            # Limit chat history to the last 2 turns (4 messages) to minimize prompt token footprint
            recent_messages = st.session_state["messages"][:-1]
            if len(recent_messages) > 4:
                recent_messages = recent_messages[-4:]

            # Strip attachment metadata before passing to LangChain history
            clean_history = [
                {"role": m["role"], "content": m["content"]}
                for m in recent_messages
            ]
            chat_history = convert_history(clean_history)

            try:
                # ── Image path: use vision LLM directly ──────────────────
                augmented_question = user_input
                if attachment_texts:
                    augmented_question += "\n\n" + "\n\n".join(attachment_texts)

                if attachment_image_uris:
                    # Still run retriever to get relevant context chunks
                    retriever = st.session_state["vector_store"].as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 6, "fetch_k": 20},
                    )
                    source_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join(
                        f"Document {i+1} (Source: {d.metadata.get('source','?')}, "
                        f"Page: {d.metadata.get('page','?')}):\n{d.page_content}"
                        for i, d in enumerate(source_docs)
                    )
                    answer = _ask_vision_llm(augmented_question, attachment_image_uris, context_text)
                    sources = source_docs

                # ── Text/document attachment path: inject extracted text into query ─
                elif attachment_texts:
                    result = st.session_state["rag_chain"].invoke({
                        "question": augmented_question,
                        "chat_history": chat_history,
                    })
                    answer = result["answer"]
                    sources = result.get("source_documents", [])

                # ── Plain text question (no attachment) ───────────────────
                else:
                    result = st.session_state["rag_chain"].invoke({
                        "question": user_input,
                        "chat_history": chat_history,
                    })
                    answer = result["answer"]
                    sources = result.get("source_documents", [])

                render_answer(answer)

                # Show source references (collapsed)
                if sources:
                    with st.expander("📎 Sources from your study material", expanded=False):
                        seen = set()
                        for doc in sources:
                            src = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")
                            label = f"{src}  — page {page}"
                            if label not in seen:
                                st.markdown(f"- `{label}`")
                                seen.add(label)

                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                err_msg = str(e)
                provider = get_config("LLM_PROVIDER", "groq").lower()
                if "rate_limit" in err_msg.lower() or "quota" in err_msg.lower() or "429" in err_msg:
                    if provider == "groq":
                        st.warning(
                            "⚠️ **Groq Rate Limit (RPM/TPM)**: Groq's free tier has a per-minute token limit. Please wait ~10-15 seconds and try again."
                        )
                    else:
                        st.error(
                            "⚠️ **Quota / Rate Limit Exceeded**: Account usage limit reached. Check billing or switch provider."
                        )
                else:
                    st.error(f"⚠️ Error processing your request: {err_msg}")
