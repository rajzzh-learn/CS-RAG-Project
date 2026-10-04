# 💻 CBSE Class 12 Computer Science AI Tutor (CS RAG Project)

An intelligent Retrieval-Augmented Generation (RAG) conversational tutor designed specifically for **CBSE Class 12 Computer Science (Python Code 083)** students and educators.

Built with **LangChain 1.x LCEL**, **ChromaDB**, **Local HuggingFace Embeddings (`all-MiniLM-L6-v2`)**, **Groq / IBM watsonx.ai / OpenAI**, and **Streamlit**.

---

## 🌟 Architecture & Features

1. **Exact Parity with Physics RAG Project**:
   - Provider flexibility: Groq (`openai/gpt-oss-120b`), IBM watsonx.ai (`ibm/granite-3-8b-instruct`), and OpenAI (`gpt-4o`).
   - Local, high-speed HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) running locally on CPU.
   - Streamlit Cloud ready with dynamic secret resolver via `get_config()` (`st.secrets` & `.env` fallback).

2. **Complete CBSE Class 12 CS Knowledge Base (1,157 pages)**:
   - 📖 **Textbooks**: NCERT Class 12 Computer Science Python Volumes.
   - 📝 **NCERT Solutions**: Chapters 1 to 12.
   - 📌 **Revision Notes**: Chapter-wise quick revision guides.
   - 📋 **Previous Year Papers (PYQ)**: CBSE Board Papers from 2013 to 2019 with solutions.

3. **Authentic CBSE Board Standard Prompting**:
   - Sectional scoring calibration (Python, Stacks/Queues, SQL, Computer Networks).
   - Case-study layout logic (server placement, repeaters, hubs/switches, network topologies).
   - Marking-scheme aligned Python code and dry-run tables.

4. **Detailed Architecture & Flow Diagrams**:
   - Comprehensive technical documentation available in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 📁 Project Structure

```
CS RAG Project/
├── .env                             # Local environment variables & provider keys
├── .env.example                     # Environment template
├── requirements.txt                 # Pinned dependencies
├── .streamlit/
│   └── config.toml                  # Streamlit configuration
├── cs_tutor/
│   ├── __init__.py                  # Module descriptor
│   ├── config.py                    # Dynamic configuration & paths
│   ├── ingest.py                    # PDF Loader & Chroma vector store builder
│   ├── rag_chain.py                 # Multi-provider LCEL RAG pipeline
│   ├── app.py                       # Streamlit interactive chat UI
│   └── vectorstore/                 # Persisted ChromaDB vector database
├── book/                            # Class 12 CS Textbooks
├── Ncert Solutions/                 # NCERT chapter-wise solutions
├── Notes/                           # CBSE chapter revision notes
├── PYQ/                             # CBSE Previous Year Question Papers
├── Lab Programs/                    # Class 12 Computer Science Lab Practicals
└── SSM Question Bank/               # Question Bank & Solutions Key
```

---

## 🚀 Quickstart Guide

### 1. Set Up Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

The project comes pre-configured with `.env` matching your Physics RAG setup:
```env
LLM_PROVIDER = "groq"
GROQ_API_KEY = "gsk_..."
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = sentence-transformers/all-MiniLM-L6-v2
RETRIEVER_TOP_K = 6
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
```

---

### 3. Build / Re-index Vectorstore

The vector database has already been built and persisted. To rebuild anytime:

```bash
python -m cs_tutor.ingest
```

---

### 4. Launch the Web Application

```bash
streamlit run cs_tutor/app.py
```

The application will be accessible at: `http://localhost:8501`
