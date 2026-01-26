#!/usr/bin/env python3
"""
Test del Agente SIMULADOR
Ejecutar desde raíz: python test_simulador_agent.py
"""

import sys
sys.path.append('src')

from rag import EmbeddingsGenerator, VectorStore, RAGPipeline
from agents import SimuladorAgent


def main():
    print("🔥 SARFIRE-RAG - Test Agente SIMULADOR\n")
    
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
    
    # 3. Crear Agente Simulador
    simulador = SimuladorAgent(rag_pipeline=rag)
    
    print("\n" + "="*70)
    print("🎭 AGENTE SIMULADOR INICIALIZADO")
    print("="*70)
    
    # 4. TEST - Escenario automático
    print("\n🎬 GENERANDO ESCENARIO DE PRUEBA...")
    
    scenario_result = simulador.create_scenario(
        topic="Incendio forestal con cambio de viento y riesgo para el personal"
    )
    
    simulador.print_scenario(scenario_result)
    
    # 5. Evaluación de ejemplo
    print("\n💡 Ahora tú tomas una decisión sobre este escenario...")
    user_decision = input("\n❓ ¿Qué harías? (describe tu decisión): ").strip()
    
    if user_decision:
        eval_result = simulador.evaluate_decision(user_decision)
        simulador.print_evaluation(eval_result)
    
    # 6. Modo interactivo
    print("\n" + "="*70)
    print("💡 MODO INTERACTIVO - Genera más escenarios")
    print("="*70)
    
    while True:
        print("\n📋 OPCIONES:")
        print("1. Generar nuevo escenario (tema automático)")
        print("2. Generar escenario sobre tema específico")
        print("3. Salir")
        
        choice = input("\nElige opción (1-3): ").strip()
        
        if choice == '1':
            result = simulador.create_scenario()
            simulador.print_scenario(result)
            
            decision = input("\n❓ Tu decisión: ").strip()
            if decision:
                eval_result = simulador.evaluate_decision(decision)
                simulador.print_evaluation(eval_result)
        
        elif choice == '2':
            topic = input("\n📝 Tema del escenario: ").strip()
            if topic:
                result = simulador.create_scenario(topic=topic)
                simulador.print_scenario(result)
                
                decision = input("\n❓ Tu decisión: ").strip()
                if decision:
                    eval_result = simulador.evaluate_decision(decision)
                    simulador.print_evaluation(eval_result)
        
        elif choice == '3':
            print("\n👋 ¡Hasta luego!")
            break
    
    print("\n✅ Test del Agente Simulador completado")


if __name__ == "__main__":
    main()