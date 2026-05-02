#!/usr/bin/env python3
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from document_loaders import MultiFormatLoader, langchain_documents_to_pdfloader_documents
from rag import DocumentChunker, EmbeddingsGenerator, VectorStore

print("🔥 REGENERANDO BASE DE DATOS\n")

raw_dir = PROJECT_ROOT / "data" / "raw"
print("📥 Cargando documentos (multi-formato)...")
loader = MultiFormatLoader()
lc_docs = loader.load_all(raw_dir)
if not lc_docs:
    print("⚠️  No hay documentos en data/raw")
    sys.exit(1)

pre_fmt = Counter((d.metadata.get("format") or "?") for d in lc_docs)
print("📑 Documentos LangChain por formato (pre-chunking):", dict(sorted(pre_fmt.items())))

documents = langchain_documents_to_pdfloader_documents(lc_docs)

print("\n📝 Creando chunks (nuevos parámetros)...")
chunker = DocumentChunker(
    chunk_size=1500,
    chunk_overlap=150,
)
chunks = chunker.chunk_all_documents(documents)
chunker.print_stats(chunks)

chunk_by_fmt = Counter()
for c in chunks:
    ext = Path(c["metadata"]["filename"]).suffix.lower() or "(sin extensión)"
    chunk_by_fmt[ext] += 1
print("📑 Chunks por formato (antes de indexar en ChromaDB):", dict(sorted(chunk_by_fmt.items())))

print("\n🧠 Generando embeddings...")
embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
embedded_chunks = embeddings_gen.embed_chunks(chunks)

print("\n💾 Recreando ChromaDB...")
vector_store = VectorStore(
    persist_directory=str(PROJECT_ROOT / "data" / "processed" / "chromadb"),
    collection_name="sarfire_docs",
)

print("🗑️  Limpiando base de datos antigua...")
vector_store.clear()

vector_store.add_chunks(embedded_chunks)
vector_store.print_stats()

print("\n✅ Base de datos regenerada correctamente!")
print("\nPrueba ahora con: python test_rag_complete.py")
