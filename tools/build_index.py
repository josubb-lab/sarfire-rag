#!/usr/bin/env python3
import sys
import logging
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from document_loaders import (  # noqa: E402
    MultiFormatLoader,
    langchain_documents_to_pdfloader_documents,
)
from rag.chunker import DocumentChunker  # noqa: E402
from rag.embeddings import EmbeddingsGenerator  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    persist_dir = PROJECT_ROOT / "data" / "processed" / "chromadb"

    print("📥 Cargando documentos multi-formato desde:", raw_dir)
    loader = MultiFormatLoader()
    lc_docs = loader.load_all(raw_dir)
    if not lc_docs:
        print("⚠️  No se cargó ningún documento desde data/raw")
        return

    pre_fmt = Counter((d.metadata.get("format") or "?") for d in lc_docs)
    print("📑 Documentos LangChain por formato (pre-chunking):", dict(sorted(pre_fmt.items())))

    docs = langchain_documents_to_pdfloader_documents(lc_docs)

    chunker = DocumentChunker(chunk_size=1500, chunk_overlap=150)
    chunks = chunker.chunk_all_documents(docs)
    chunker.print_stats(chunks)

    chunk_by_fmt = Counter()
    for c in chunks:
        ext = Path(c["metadata"]["filename"]).suffix.lower() or "(sin extensión)"
        chunk_by_fmt[ext] += 1
    print("📑 Chunks por formato (antes de indexar en ChromaDB):", dict(sorted(chunk_by_fmt.items())))

    emb = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
    embedded_chunks = emb.embed_chunks(chunks)

    vs = VectorStore(persist_directory=str(persist_dir), collection_name="sarfire_docs")
    vs.clear()
    vs.add_chunks(embedded_chunks)
    vs.print_stats()

    print("✅ Indexación finalizada.")


if __name__ == "__main__":
    main()
