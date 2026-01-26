#!/usr/bin/env python3
import sys
from pathlib import Path
import importlib

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CANDIDATE_IMPORTS = [
    "rag.pdf_loader",                 # src/rag/pdf_loader.py
    "rag.loader.pdf_loader",          # src/rag/loader/pdf_loader.py
    "rag.loaders.pdf_loader",         # src/rag/loaders/pdf_loader.py
    "loader.pdf_loader",              # src/loader/pdf_loader.py
    "loaders.pdf_loader",             # src/loaders/pdf_loader.py
]

def import_pdf_loader():
    last_err = None
    for mod in CANDIDATE_IMPORTS:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "PDFLoader"):
                print(f"✅ PDFLoader importado desde: {mod}.PDFLoader")
                return m.PDFLoader
        except Exception as e:
            last_err = e
    raise ModuleNotFoundError(
        "No se pudo localizar PDFLoader. Probé: "
        + ", ".join(CANDIDATE_IMPORTS)
        + f"\nÚltimo error: {last_err}"
    )

PDFLoader = import_pdf_loader()

from rag.chunker import DocumentChunker
from rag.embeddings import EmbeddingsGenerator
from rag.vector_store import VectorStore


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    persist_dir = PROJECT_ROOT / "data" / "processed" / "chromadb"

    print("📥 Cargando PDFs desde:", raw_dir)
    loader = PDFLoader(str(raw_dir))
    docs = loader.load_all_pdfs()
    loader.print_stats(docs)

    chunker = DocumentChunker(chunk_size=1500, chunk_overlap=150)
    chunks = chunker.chunk_all_documents(docs)
    chunker.print_stats(chunks)

    emb = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
    embedded_chunks = emb.embed_chunks(chunks)

    vs = VectorStore(persist_directory=str(persist_dir), collection_name="sarfire_docs")
    vs.clear()
    vs.add_chunks(embedded_chunks)
    vs.print_stats()

    print("✅ Indexación finalizada.")


if __name__ == "__main__":
    main()
