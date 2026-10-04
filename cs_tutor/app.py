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

# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Class 12 Computer Science teacher, here to help you "
                "ace your **CBSE Board Exams (Python 083)**.\n\n"
                "Tell me which chapter, Python code, SQL query, or networking case study you'd like to work on!"
            ),
        }
    ]

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask your CS teacher …"):
    # Show student message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            # Limit chat history to the last 2 turns (4 messages) to minimize prompt token footprint
            recent_messages = st.session_state["messages"][:-1]
            if len(recent_messages) > 4:
                recent_messages = recent_messages[-4:]
            chat_history = convert_history(recent_messages)

            try:
                result = st.session_state["rag_chain"].invoke({
                    "question": user_input,
                    "chat_history": chat_history,
                })
                answer = result["answer"]
                sources = result.get("source_documents", [])

                st.markdown(answer)

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
