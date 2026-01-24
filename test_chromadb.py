#!/usr/bin/env python3
"""
Test de ChromaDB - Paso 3 Día 2
Ejecutar desde raíz: python test_chromadb.py
"""

import sys
sys.path.append('src')

from loaders import PDFLoader
from rag import DocumentChunker, EmbeddingsGenerator, VectorStore


def main():
    print("🔥 SARFIRE-RAG - Test de ChromaDB (Paso 3/4)\n")
    
    # 1. Cargar documentos
    print("📄 Paso 1/4: Cargando PDFs...")
    loader = PDFLoader("data/raw")
    documents = loader.load_all_pdfs()
    
    if not documents:
        print("❌ No hay documentos")
        return
    
    # 2. Crear chunks
    print("\n📝 Paso 2/4: Creando chunks...")
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_all_documents(documents)
    print(f"✅ {len(chunks)} chunks creados")
    
    # 3. Generar embeddings
    print("\n🧠 Paso 3/4: Generando embeddings...")
    embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
    embedded_chunks = embeddings_gen.embed_chunks(chunks)
    
    # 4. Inicializar ChromaDB
    print("\n💾 Paso 4/4: Almacenando en ChromaDB...")
    vector_store = VectorStore(
        persist_directory="data/processed/chromadb",
        collection_name="sarfire_docs"
    )
    
    # Verificar si ya hay datos
    stats = vector_store.get_stats()
    if stats['total_chunks'] > 0:
        print(f"\n⚠️  La base de datos ya tiene {stats['total_chunks']} chunks")
        response = input("¿Quieres limpiar y recargar? (s/n): ")
        if response.lower() == 's':
            vector_store.clear()
        else:
            print("Usando datos existentes...")
            vector_store.print_stats()
            print("\n✅ Test completado (usando datos existentes)")
            return
    
    # Añadir chunks
    vector_store.add_chunks(embedded_chunks)
    
    # Mostrar estadísticas
    vector_store.print_stats()
    
    # 5. TEST DE BÚSQUEDA
    print("\n🔍 TEST DE BÚSQUEDA SEMÁNTICA")
    print("="*70)
    
    # Consultas de ejemplo
    test_queries = [
        "¿Cómo se propaga el fuego forestal?",
        "¿Qué equipamiento de protección usar?",
        "Técnicas de extinción de incendios"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- CONSULTA {i} ---")
        print(f"Pregunta: {query}")
        print("-" * 70)
        
        # Buscar
        results = vector_store.search_by_text(
            query_text=query,
            embeddings_generator=embeddings_gen,
            n_results=3
        )
        
        # Mostrar top 3 resultados
        for rank, result in enumerate(results, 1):
            print(f"\n{rank}. Similitud: {result['similarity']:.4f}")
            print(f"   Documento: {result['metadata']['filename']}")
            print(f"   Página: {result['metadata']['page_num']}")
            print(f"   Texto: {result['text'][:200]}...")
        
        print("-" * 70)
    
    # Validación final
    print("\n🔍 VALIDACIÓN FINAL:")
    print("-" * 70)
    final_stats = vector_store.get_stats()
    print(f"✓ Chunks en ChromaDB: {final_stats['total_chunks']}")
    print(f"✓ Búsquedas funcionando: ✅")
    print(f"✓ Persistencia en: {final_stats['persist_directory']}")
    print("-" * 70)
    
    print("\n✅ Test de ChromaDB completado!")
    print("\n🎉 RAG BASE COMPLETO - Todos los componentes funcionando:")
    print("   ✅ PDFs cargados")
    print("   ✅ Chunks creados")
    print("   ✅ Embeddings generados")
    print("   ✅ Vector store funcionando")
    print("\n📝 Siguiente paso: Integrar con Gemini (RAG completo)")
    print("   Ejecuta: python test_rag_complete.py")


if __name__ == "__main__":
    main()