"""
PDF ingestion pipeline for the Chemistry Tutor.
Loads all PDFs from configured directories, splits them into chunks,
and persists a ChromaDB vector store for retrieval.
"""
import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
import chromadb.utils.embedding_functions as ef

from chemistry_tutor.config import (
    PDF_DIRS,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class SafeMiniLMEmbeddings(Embeddings):
    """ONNX-based all-MiniLM-L6-v2 embeddings — no PyTorch dependency."""
    def __init__(self):
        self._ef = ef.ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._ef([text])[0]


def load_pdfs(pdf_dirs: list[Path]) -> list:
    """Load every PDF found under the given directories."""
    docs = []
    for directory in pdf_dirs:
        if not directory.exists():
            logger.warning("Directory not found, skipping: %s", directory)
            continue
        for pdf_path in sorted(directory.glob("*.pdf")):
            logger.info("Loading: %s", pdf_path.name)
            loader = PyPDFLoader(str(pdf_path))
            docs.extend(loader.load())
    logger.info("Total pages loaded: %d", len(docs))
    return docs


def get_embedding_function() -> Embeddings:
    return SafeMiniLMEmbeddings()


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """Build (or load) the ChromaDB vector store."""
    embeddings = get_embedding_function()

    if VECTOR_STORE_DIR.exists() and not force_rebuild:
        logger.info("Loading existing vector store from %s", VECTOR_STORE_DIR)
        return Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings,
        )

    logger.info("Building vector store — this may take a few minutes …")
    raw_docs = load_pdfs(PDF_DIRS)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("Total chunks: %d", len(chunks))

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    logger.info("Vector store saved to %s", VECTOR_STORE_DIR)
    return vector_store


if __name__ == "__main__":
    build_vector_store(force_rebuild=True)
