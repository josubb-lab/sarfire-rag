#!/usr/bin/env python3
"""
Test de Embeddings - Paso 2 Día 2
Ejecutar desde raíz: python test_embeddings.py
"""

import sys
sys.path.append('src')

from loaders import PDFLoader
from rag import DocumentChunker, EmbeddingsGenerator


def main():
    print("🔥 SARFIRE-RAG - Test de Embeddings (Paso 2/4)\n")
    
    # 1. Cargar documentos
    print("📄 Paso 1/3: Cargando PDFs...")
    loader = PDFLoader("data/raw")
    documents = loader.load_all_pdfs()
    
    if not documents:
        print("❌ No hay documentos")
        return
    
    # 2. Crear chunks
    print("\n📝 Paso 2/3: Creando chunks...")
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_all_documents(documents)
    print(f"✅ {len(chunks)} chunks creados")
    
    # 3. Generar embeddings
    print("\n🧠 Paso 3/3: Generando embeddings...")
    embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
    
    # NOTA: Esto puede tardar 1-2 minutos la primera vez
    # (descarga el modelo y procesa todos los chunks)
    embedded_chunks = embeddings_gen.embed_chunks(chunks)
    
    # 4. Mostrar estadísticas
    embeddings_gen.print_stats(embedded_chunks)
    
    # 5. Test de similitud
    embeddings_gen.test_similarity(embedded_chunks, n_examples=2)
    
    # 6. Validación
    print("\n🔍 VALIDACIÓN:")
    print("-" * 70)
    print(f"✓ Chunks procesados: {len(embedded_chunks)}")
    print(f"✓ Todos tienen embeddings: {all('embedding' in c for c in embedded_chunks)}")
    print(f"✓ Dimensión: {len(embedded_chunks[0]['embedding'])}")
    print("-" * 70)
    
    print("\n✅ Test de Embeddings completado!")
    print("\n📝 Siguiente paso: Almacenar en ChromaDB")
    print("   Ejecuta: python test_chromadb.py")


if __name__ == "__main__":
    main()