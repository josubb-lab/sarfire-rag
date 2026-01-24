#!/usr/bin/env python3
import sys
sys.path.append('src')
from rag import EmbeddingsGenerator, VectorStore, HybridSearch

# Cargar componentes
embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
vector_store = VectorStore(
    persist_directory="data/processed/chromadb",
    collection_name="sarfire_docs"
)

# Crear búsqueda híbrida
hybrid_search = HybridSearch(vector_store, embeddings_gen)

# Probar con la pregunta problemática
query = "¿Qué es el Efecto Foëhn?"

print(f"🔍 Búsqueda Híbrida: {query}\n")
print("="*70)

results = hybrid_search.hybrid_search(
    query=query,
    top_k=5,
    semantic_weight=0.4,  # Menos peso a semántica
    keyword_weight=0.6    # Más peso a keywords
)

for i, result in enumerate(results, 1):
    print(f"\n--- RESULTADO {i} ---")
    print(f"Score híbrido: {result['hybrid_score']:.3f}")
    print(f"  - Semántico: {result['semantic_score']:.3f}")
    print(f"  - Keywords: {result['keyword_score']:.3f}")
    print(f"Documento: {result['metadata']['filename']}")
    print(f"Página: {result['metadata']['page_num']}")
    print(f"Texto: {result['text'][:300]}...")
    print("-" * 70)