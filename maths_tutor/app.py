"""
Streamlit chat interface for the Maths Tutor RAG Agent.
Run with:  streamlit run maths_tutor/app.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from maths_tutor.config import get_config
from maths_tutor.ingest import build_vector_store
from maths_tutor.rag_chain import build_rag_chain, convert_history
from shared.file_utils import (
    extract_text_from_pdf,
    extract_text_from_txt,
    image_to_base64_uri,
    is_image,
    SUPPORTED_EXTS,
)
from streamlit_paste_button import paste_image_button

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maths Tutor — Class 12 CBSE (041)",
    page_icon="📐",
    layout="wide",
)

st.title("📐 Class 12 Mathematics Tutor")
st.caption("Powered by NCERT Mathematics — CBSE Board Exam Edition (Code 041)")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    current_provider = get_config("LLM_PROVIDER", "groq").lower()
    st.caption(f"🤖 **Active Provider**: `{current_provider.upper()}`")
    st.header("🗂️ Chapters & Syllabus")
    st.markdown("""
**Unit 1: Relations & Functions**
1. Relations & Functions (Types, Composition, Invertibility)
2. Inverse Trigonometric Functions

**Unit 2: Algebra**
3. Matrices (Operations, Transpose, Symmetric)
4. Determinants (Cofactors, Adjoint, Inverse, Cramer's Rule)

**Unit 3: Calculus**
5. Continuity & Differentiability
6. Application of Derivatives (Tangents, Maxima/Minima, Rate of Change)
7. Integrals (Standard Forms, Substitution, By Parts, Partial Fractions)
8. Application of Integrals (Area Between Curves)
9. Differential Equations (Formation, Variable Separable, Homogeneous, Linear)

**Unit 4: Vectors & 3D Geometry**
10. Vector Algebra (Dot & Cross Products)
11. Three-Dimensional Geometry (Lines, Planes, Shortest Distance)

**Unit 5: Linear Programming**
12. Linear Programming (Graphical Method, Corner Points)

**Unit 6: Probability**
13. Probability (Conditional, Bayes' Theorem, Distributions, Mean & Variance)
""")
    st.divider()
    st.markdown("**💡 Try asking:**")
    st.markdown("""
- *Prove that the function f(x) = x³ is continuous and differentiable everywhere*
- *Integrate ∫ x·eˣ dx using integration by parts*
- *Find the equation of the plane passing through three given points*
- *Solve the differential equation dy/dx + y·tan(x) = sec(x)*
- *Find the maximum area of a rectangle inscribed in a circle of radius r*
""")
    st.divider()
    if st.button("🔄 Rebuild Vector Store"):
        with st.spinner("Re-ingesting all PDFs …"):
            st.session_state.pop("rag_chain", None)
            st.session_state.pop("vector_store", None)
            build_vector_store(force_rebuild=True)
        st.success("Vector store rebuilt!")

# ── Init vector store & chain ──────────────────────────────────────────────
if "vector_store" not in st.session_state:
    with st.spinner("⚙️ Loading Maths study material … first load takes ~1 min"):
        st.session_state["vector_store"] = build_vector_store(force_rebuild=False)

target_provider = get_config("LLM_PROVIDER", "groq").lower()
if "rag_chain" not in st.session_state or st.session_state.get("_active_provider") != target_provider:
    try:
        st.session_state["rag_chain"] = build_rag_chain(st.session_state["vector_store"])
        st.session_state["_active_provider"] = target_provider
    except Exception as e:
        st.error(f"⚠️ **LLM Initialization Error**: {e}")
        st.stop()

# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Class 12 Mathematics teacher, here to help you "
                "ace your **CBSE Board Exams (Code 041)**.\n\n"
                "Ask me about calculus, algebra, vectors, probability, or any Maths problem! "
                "You can also **attach an image, PDF, or text file** using the 📎 button below, "
                "or **paste a screenshot** with the 📋 button."
            ),
        }
    ]

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("attachment_name"):
            st.caption(f"📎 **Attached:** `{msg['attachment_name']}`")
            if msg.get("attachment_preview"):
                with st.expander("👁️ Attachment preview", expanded=False):
                    if msg.get("attachment_is_image"):
                        st.image(msg["attachment_preview"], use_container_width=True)
                    else:
                        st.text(msg["attachment_preview"][:2000])
        st.markdown(msg["content"])


# ── Helper: vision LLM call ────────────────────────────────────────────────
def _ask_vision_llm(question: str, image_data_uri: str, context_text: str) -> str:
    provider = get_config("LLM_PROVIDER", "groq").lower()
    vision_content = [
        {
            "type": "text",
            "text": (
                "You are an expert Class 12 CBSE Mathematics teacher.\n\n"
                f"Context from the student's study materials:\n{context_text}\n\n"
                f"The student has attached an image and asks:\n{question}"
            ),
        },
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ]
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=get_config("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=get_config("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": vision_content}],
            max_tokens=4096, temperature=0.3,
        )
        return resp.choices[0].message.content
    elif provider == "groq":
        from openai import OpenAI
        client = OpenAI(api_key=get_config("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=get_config("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
            messages=[{"role": "user", "content": vision_content}],
            max_tokens=4096, temperature=0.3,
        )
        return resp.choices[0].message.content
    else:
        return (
            "⚠️ **Image vision is not supported for the `watsonx` provider.** "
            "Please switch to `groq` or `openai`, or describe your question in text."
        )


# ── Input area ─────────────────────────────────────────────────────────────
ext_list = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTS))

col_upload, col_paste = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_file = st.file_uploader(
        f"📎 Attach a file — {ext_list}",
        type=list(SUPPORTED_EXTS),
        label_visibility="visible",
        help="Attach an image (maths problem screenshot / graph), PDF, or .txt file.",
    )
with col_paste:
    paste_result = paste_image_button(
        "📋 Paste image",
        background_color="#444654",
        hover_background_color="#565869",
        key="clipboard_paste",
    )

if user_input := st.chat_input("Ask your Maths teacher …"):
    attachment_name: str | None = None
    attachment_text: str | None = None
    attachment_image_uri: str | None = None
    attachment_preview = None
    attachment_is_image = False

    if paste_result.image_data is not None:
        import io as _io
        buf = _io.BytesIO()
        paste_result.image_data.save(buf, format="PNG")
        file_bytes = buf.getvalue()
        attachment_name = "pasted-image.png"
        attachment_image_uri, _ = image_to_base64_uri(file_bytes, attachment_name)
        attachment_preview = file_bytes
        attachment_is_image = True
    elif uploaded_file is not None:
        attachment_name = uploaded_file.name
        file_bytes = uploaded_file.read()
        if is_image(attachment_name):
            attachment_image_uri, _ = image_to_base64_uri(file_bytes, attachment_name)
            attachment_preview = file_bytes
            attachment_is_image = True
        elif attachment_name.lower().endswith(".pdf"):
            attachment_text = extract_text_from_pdf(file_bytes)
            attachment_preview = attachment_text
        else:
            attachment_text = extract_text_from_txt(file_bytes)
            attachment_preview = attachment_text

    user_msg: dict = {
        "role": "user",
        "content": user_input,
        "attachment_name": attachment_name,
        "attachment_preview": attachment_preview,
        "attachment_is_image": attachment_is_image,
    }
    st.session_state["messages"].append(user_msg)

    with st.chat_message("user"):
        if attachment_name:
            st.caption(f"📎 **Attached:** `{attachment_name}`")
            if attachment_preview is not None:
                with st.expander("👁️ Attachment preview", expanded=False):
                    if attachment_is_image:
                        st.image(attachment_preview, use_container_width=True)
                    else:
                        st.text(str(attachment_preview)[:2000])
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            recent_messages = st.session_state["messages"][:-1]
            if len(recent_messages) > 4:
                recent_messages = recent_messages[-4:]
            clean_history = [{"role": m["role"], "content": m["content"]} for m in recent_messages]
            chat_history = convert_history(clean_history)

            try:
                if attachment_image_uri is not None:
                    retriever = st.session_state["vector_store"].as_retriever(
                        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20},
                    )
                    source_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join(
                        f"Document {i+1} (Source: {d.metadata.get('source','?')}, "
                        f"Page: {d.metadata.get('page','?')}):\n{d.page_content}"
                        for i, d in enumerate(source_docs)
                    )
                    answer = _ask_vision_llm(user_input, attachment_image_uri, context_text)
                    sources = source_docs
                elif attachment_text is not None:
                    augmented = f"{user_input}\n\n--- Attached file: {attachment_name} ---\n{attachment_text}"
                    result = st.session_state["rag_chain"].invoke(
                        {"question": augmented, "chat_history": chat_history}
                    )
                    answer = result["answer"]
                    sources = result.get("source_documents", [])
                else:
                    result = st.session_state["rag_chain"].invoke(
                        {"question": user_input, "chat_history": chat_history}
                    )
                    answer = result["answer"]
                    sources = result.get("source_documents", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📎 Sources from your study material", expanded=False):
                        seen = set()
                        for doc in sources:
                            label = f"{doc.metadata.get('source','Unknown')}  — page {doc.metadata.get('page','?')}"
                            if label not in seen:
                                st.markdown(f"- `{label}`")
                                seen.add(label)

                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                err_msg = str(e)
                provider = get_config("LLM_PROVIDER", "groq").lower()
                if "rate_limit" in err_msg.lower() or "quota" in err_msg.lower() or "429" in err_msg:
                    if provider == "groq":
                        st.warning("⚠️ **Groq Rate Limit**: Please wait ~10-15 seconds and try again.")
                    else:
                        st.error("⚠️ **Quota / Rate Limit Exceeded**: Check billing or switch provider.")
                else:
                    st.error(f"⚠️ Error: {err_msg}")
