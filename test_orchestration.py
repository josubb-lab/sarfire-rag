#!/usr/bin/env python3
"""
Test del Sistema de Orquestación Completo
Ejecutar desde raíz: python test_orchestration.py
"""

import sys
sys.path.append('src')

from rag import EmbeddingsGenerator, VectorStore, RAGPipeline
from agents import FormadorAgent, SimuladorAgent, DirectorAgent, OrchestrationSystem


def main():
    print("🔥 SARFIRE-RAG - Test Sistema de Orquestación Completo\n")
    
    # 1. Cargar componentes RAG
    print("📦 Cargando sistema RAG...")
    embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
    
    vector_store = VectorStore(
        persist_directory="data/processed/chromadb",
        collection_name="sarfire_docs"
    )
    
    stats = vector_store.get_stats()
    if stats['total_chunks'] == 0:
        print("❌ No hay datos en ChromaDB")
        return
    
    print(f"✅ Vector store: {stats['total_chunks']} chunks")
    
    # 2. Inicializar RAG
    rag = RAGPipeline(
        vector_store=vector_store,
        embeddings_generator=embeddings_gen,
        model_name="gemini-2.0-flash",
        temperature=0.3,
        top_k=5,
        use_hybrid_search=True
    )
    
    # 3. Crear agentes
    print("\n🎭 Inicializando agentes especializados...")
    formador = FormadorAgent(rag_pipeline=rag)
    simulador = SimuladorAgent(rag_pipeline=rag)
    director = DirectorAgent(model_name="gemini-2.0-flash")
    
    # 4. Crear sistema de orquestación
    orchestration = OrchestrationSystem(
        formador_agent=formador,
        simulador_agent=simulador,
        director_agent=director
    )
    
    print("\n" + "="*70)
    print("✅ SISTEMA COMPLETO INICIALIZADO")
    print("="*70)
    print("🎓 Agente Formador: LISTO")
    print("🎭 Agente Simulador: LISTO")
    print("🎯 Agente Director: LISTO")
    print("="*70)
    
    # 5. TEST - Consultas mixtas (automático)
    test_queries = [
        "¿Qué es el PMA y para qué sirve?",  # Debería ir a Formador
        "Genera un escenario de incendio con viento fuerte",  # Debería ir a Simulador
        "Explícame las fases de una emergencia forestal"  # Debería ir a Formador
    ]
    
    print("\n🧪 MODO TEST AUTOMÁTICO - 3 consultas")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_queries)}")
        print('#'*70)
        print(f"Consulta: {query}")
        
        result = orchestration.process_query(query)
        
        # Mostrar resultado según el agente
        if result.get('answer'):  # Formador
            formador.print_response(result)
        elif result.get('scenario'):  # Simulador
            simulador.print_scenario(result)
        
        if i < len(test_queries):
            input("\n[Presiona ENTER para continuar...]")
    
    # 6. Modo interactivo
    print("\n" + "="*70)
    print("💡 MODO INTERACTIVO")
    print("="*70)
    print("El Director enrutará automáticamente tus consultas")
    print("Escribe 'salir' para terminar")
    
    while True:
        user_query = input("\n❓ Tu consulta: ").strip()
        
        if user_query.lower() in ['salir', 'exit', 'quit', '']:
            print("\n👋 ¡Hasta luego!")
            break
        
        result = orchestration.process_query(user_query)
        
        # Mostrar resultado según el agente usado
        if result.get('answer'):  # Formador
            formador.print_response(result)
        elif result.get('scenario'):  # Simulador
            simulador.print_scenario(result)
            
            # Si es simulador, ofrecer evaluar decisión
            decision = input("\n💡 ¿Quieres tomar una decisión sobre este escenario? (s/n): ").strip()
            if decision.lower() == 's':
                user_decision = input("❓ Tu decisión: ").strip()
                if user_decision:
                    eval_result = simulador.evaluate_decision(user_decision)
                    simulador.print_evaluation(eval_result)
    
    print("\n✅ Test del Sistema de Orquestación completado")


if __name__ == "__main__":
    main()