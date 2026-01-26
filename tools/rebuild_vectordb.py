#!/usr/bin/env python3
import sys
sys.path.append('src')
from loaders import PDFLoader
from rag import DocumentChunker, EmbeddingsGenerator, VectorStore

print("🔥 REGENERANDO BASE DE DATOS\n")

# 1. Cargar PDFs
print("📄 Cargando PDFs...")
loader = PDFLoader("data/raw")
documents = loader.load_all_pdfs()

# 2. Chunking con nuevos parámetros
print("\n📝 Creando chunks (nuevos parámetros)...")
chunker = DocumentChunker(
    chunk_size=1500,    # Más grande
    chunk_overlap=150   # Menos overlap
)
chunks = chunker.chunk_all_documents(documents)
chunker.print_stats(chunks)

# 3. Embeddings
print("\n🧠 Generando embeddings...")
embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
embedded_chunks = embeddings_gen.embed_chunks(chunks)

# 4. LIMPIAR y recrear ChromaDB
print("\n💾 Recreando ChromaDB...")
vector_store = VectorStore(
    persist_directory="data/processed/chromadb",
    collection_name="sarfire_docs"
)

# LIMPIAR base de datos antigua
print("🗑️  Limpiando base de datos antigua...")
vector_store.clear()

# AÑADIR nuevos chunks
vector_store.add_chunks(embedded_chunks)
vector_store.print_stats()

print("\n✅ Base de datos regenerada correctamente!")
print("\nPrueba ahora con: python test_rag_complete.py")