#!/usr/bin/env python3
import sys
sys.path.append('src')
from rag import EmbeddingsGenerator, VectorStore, RAGPipeline

# Cargar componentes
embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
vector_store = VectorStore(
    persist_directory="data/processed/chromadb",
    collection_name="sarfire_docs"
)

rag = RAGPipeline(
    vector_store=vector_store,
    embeddings_generator=embeddings_gen,
    model_name="gemini-2.5-flash",
    temperature=0.3,
    top_k=15  # ← MÁS CHUNKS
)

# Probar con más contexto
query = "¿Qué es el Efecto Foëhn?"
result = rag.query(query, top_k=15)  # Forzar 15 chunks
rag.print_result(result)