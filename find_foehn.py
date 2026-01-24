#!/usr/bin/env python3
import sys
sys.path.append('src')
from rag import VectorStore

vector_store = VectorStore(
    persist_directory="data/processed/chromadb",
    collection_name="sarfire_docs"
)

# Buscar todos los chunks que contengan "Foëhn" o "foehn"
print("🔍 Buscando chunks que contengan 'Foëhn'...\n")

# Obtener todos los documentos
results = vector_store.collection.get()

found = False
for i, doc in enumerate(results['documents']):
    if 'foëhn' in doc.lower() or 'foehn' in doc.lower():
        found = True
        print(f"✅ ENCONTRADO en chunk {i}")
        print(f"Documento: {results['metadatas'][i]['filename']}")
        print(f"Página: {results['metadatas'][i]['page_num']}")
        print(f"Texto: {doc[:500]}")
        print("-" * 70)

if not found:
    print("❌ NO SE ENCONTRÓ ningún chunk con 'Foëhn'")
    print("\nPosibles causas:")
    print("1. El contenido está en una imagen/tabla (no extraído por pypdf)")
    print("2. El PDF tiene ese contenido en una página que no se procesó bien")
    print("3. El término está escrito de otra forma (Foehn, föhn, etc.)")