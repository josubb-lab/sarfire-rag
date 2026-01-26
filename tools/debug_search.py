#!/usr/bin/env python3
import sys
sys.path.append('src')
from rag import VectorStore

vector_store = VectorStore(
    persist_directory="data/processed/chromadb",
    collection_name="sarfire_docs"
)

# Buscar "brisas" directamente
print("🔍 Buscando 'brisas' en los chunks...\n")
results = vector_store.collection.get()

found = 0
for i, doc in enumerate(results['documents']):
    if 'brisa' in doc.lower():
        found += 1
        if found <= 5:  # Mostrar solo primeros 5
            print(f"✅ CHUNK {i}")
            print(f"Doc: {results['metadatas'][i]['filename']}, pág {results['metadatas'][i]['page_num']}")
            print(f"Texto: {doc[:300]}")
            print("-" * 70)

print(f"\nTotal encontrados: {found}")

# Buscar "bibliografía"
print("\n\n🔍 Buscando 'bibliografía' en los chunks...\n")
found = 0
for i, doc in enumerate(results['documents']):
    if 'bibliograf' in doc.lower():
        found += 1
        print(f"✅ CHUNK {i}")
        print(f"Doc: {results['metadatas'][i]['filename']}, pág {results['metadatas'][i]['page_num']}")
        print(f"Texto: {doc[:300]}")
        print("-" * 70)

print(f"\nTotal encontrados: {found}")