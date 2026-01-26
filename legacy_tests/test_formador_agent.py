#!/usr/bin/env python3
"""
Test del Agente FORMADOR
Ejecutar desde raíz: python test_formador_agent.py
"""

import sys
sys.path.append('src')

from rag import EmbeddingsGenerator, VectorStore, RAGPipeline
from agents import FormadorAgent


def main():
    print("🔥 SARFIRE-RAG - Test Agente FORMADOR\n")
    
    # 1. Cargar componentes RAG
    print("📦 Cargando sistema RAG...")
    embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
    
    vector_store = VectorStore(
        persist_directory="data/processed/chromadb",
        collection_name="sarfire_docs"
    )
    
    # Verificar datos
    stats = vector_store.get_stats()
    if stats['total_chunks'] == 0:
        print("❌ No hay datos en ChromaDB. Ejecuta: python rebuild_vectordb.py")
        return
    
    print(f"✅ Vector store: {stats['total_chunks']} chunks")
    
    # 2. Inicializar RAG Pipeline
    rag = RAGPipeline(
        vector_store=vector_store,
        embeddings_generator=embeddings_gen,
        model_name="gemini-2.0-flash",
        temperature=0.3,
        top_k=5,
        use_hybrid_search=True
    )
    
    # 3. Crear Agente Formador
    formador = FormadorAgent(rag_pipeline=rag)
    
    print("\n" + "="*70)
    print("🎓 AGENTE FORMADOR INICIALIZADO")
    print("="*70)
    
    # 4. TEST - Consultas de formación
    test_queries = [
        "¿Qué es el PMA y cuál es su función en un incendio forestal?",
        "Explícame las fases de una emergencia por incendio forestal",
        "¿Cuáles son los tipos de incendio forestal según su complejidad?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_queries)}")
        print('#'*70)
        
        result = formador.process_query(query)
        formador.print_response(result)
        
        if i < len(test_queries):
            input("\n[Presiona ENTER para continuar...]")
    
    # 5. Modo interactivo
    print("\n" + "="*70)
    print("💡 MODO INTERACTIVO - Haz tus consultas de formación")
    print("="*70)
    
    while True:
        user_query = input("\n❓ Tu consulta (o 'salir'): ").strip()
        
        if user_query.lower() in ['salir', 'exit', 'quit', '']:
            print("\n👋 ¡Hasta luego!")
            break
        
        result = formador.process_query(user_query)
        formador.print_response(result)
    
    print("\n✅ Test del Agente Formador completado")


if __name__ == "__main__":
    main()