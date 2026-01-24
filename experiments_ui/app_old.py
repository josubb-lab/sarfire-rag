#!/usr/bin/env python3
"""
Interfaz Gradio para SARFIRE-RAG
Sistema Multi-Agente de Formación en Emergencias
"""

import gradio as gr
import sys
sys.path.append('src')

from rag import EmbeddingsGenerator, VectorStore, RAGPipeline
from agents import FormadorAgent, SimuladorAgent, DirectorAgent, OrchestrationSystem


# Variables globales para el sistema
orchestration = None
formador = None
simulador = None


def initialize_system():
    """Inicializa el sistema RAG + Agentes"""
    global orchestration, formador, simulador
    
    print("🔥 Inicializando SARFIRE-RAG...")
    
    # RAG
    embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(
        persist_directory="data/processed/chromadb",
        collection_name="sarfire_docs"
    )
    
    rag = RAGPipeline(
        vector_store=vector_store,
        embeddings_generator=embeddings_gen,
        model_name="gemini-2.0-flash",
        temperature=0.3,
        top_k=5,
        use_hybrid_search=True
    )
    
    # Agentes
    formador = FormadorAgent(rag_pipeline=rag)
    simulador = SimuladorAgent(rag_pipeline=rag)
    director = DirectorAgent(model_name="gemini-2.0-flash")
    
    # Orquestación
    orchestration = OrchestrationSystem(
        formador_agent=formador,
        simulador_agent=simulador,
        director_agent=director
    )
    
    print("✅ Sistema inicializado correctamente")
    return "✅ Sistema listo para usar"


def format_response(result, mode):
    """Formatea la respuesta según el tipo de agente"""
    
    # Añadir badge del modo usado
    if mode == "Automático":
        agent_used = result.get('classification', {}).get('agent', 'formador')
        agent_emoji = {'formador': '🎓', 'simulador': '🎭'}.get(agent_used, '❓')
        mode_badge = f"{agent_emoji} **Agente usado:** {agent_used.upper()}\n\n"
    else:
        mode_badge = ""
    
    # Formatear según tipo de respuesta
    if result.get('answer'):  # Formador
        response = f"{mode_badge}📖 **RESPUESTA:**\n\n{result['answer']}\n\n"
        
        # Fuentes
        if result.get('sources'):
            response += "📚 **FUENTES CONSULTADAS:**\n\n"
            for i, source in enumerate(result['sources'][:3], 1):
                response += f"{i}. {source['filename']} (Pág. {source['page']})\n"
        
        return response
    
    elif result.get('scenario'):  # Simulador
        return f"{mode_badge}🎭 **ESCENARIO OPERATIVO:**\n\n{result['scenario']}"
    
    elif result.get('evaluation'):  # Evaluación
        return f"⚖️ **EVALUACIÓN:**\n\n{result['evaluation']}"
    
    else:
        return "⚠️ No se pudo generar una respuesta"


def chat_response(message, history, mode):
    """Procesa mensaje del usuario"""
    
    if not message.strip():
        return history, ""
    
    # Determinar qué agente usar
    if mode == "Automático":
        result = orchestration.process_query(message)
    elif mode == "Formador":
        result = orchestration.process_query(message, force_agent='formador')
    elif mode == "Simulador":
        result = orchestration.process_query(message, force_agent='simulador')
    else:
        return history, ""
    
    # Formatear respuesta
    response = format_response(result, mode)
    
    # Actualizar historial
    history.append((message, response))
    
    return history, ""


def evaluate_decision(decision, history):
    """Evalúa una decisión sobre el escenario actual del simulador"""
    
    if not decision.strip():
        return history, ""
    
    if not simulador.current_scenario:
        warning = "⚠️ No hay un escenario activo. Primero solicita un escenario en modo Simulador."
        history.append((f"[Evaluación]: {decision}", warning))
        return history, ""
    
    # Evaluar decisión
    result = simulador.evaluate_decision(decision)
    response = format_response(result, "Simulador")
    
    # Actualizar historial
    history.append((f"[Mi decisión]: {decision}", response))
    
    return history, ""


# Crear interfaz
with gr.Blocks(title="SARFIRE-RAG", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🔥 SARFIRE-RAG
    ## Sistema Multi-Agente de Formación en Emergencias Forestales
    
    **Basado en:** DTF-13 - Organización y Gestión de Incendios Forestales (Consorcio Provincial de Bomberos de Valencia)
    """)
    
    # Estado de inicialización
    with gr.Row():
        init_status = gr.Textbox(
            label="Estado del Sistema",
            value="⏳ Inicializando...",
            interactive=False
        )
        init_btn = gr.Button("🔄 Reiniciar Sistema", size="sm")
    
    gr.Markdown("---")
    
    # Selector de modo
    mode_selector = gr.Radio(
        choices=["Automático", "Formador", "Simulador"],
        value="Automático",
        label="🎛️ Modo de Operación",
        info="Automático: El sistema decide | Formador: Solo explicaciones | Simulador: Solo escenarios"
    )
    
    # Chat principal
    chatbot = gr.Chatbot(
        label="Conversación",
        height=500,
        show_copy_button=True
    )
    
    with gr.Row():
        msg_input = gr.Textbox(
            label="Tu consulta",
            placeholder="Escribe tu pregunta o solicitud...",
            scale=4
        )
        submit_btn = gr.Button("Enviar", variant="primary", scale=1)
    
    with gr.Row():
        clear_btn = gr.Button("🗑️ Limpiar Chat")
    
    # Panel de evaluación (solo visible en modo Simulador)
    gr.Markdown("### 💡 Evaluación de Decisiones (Modo Simulador)")
    
    with gr.Row():
        decision_input = gr.Textbox(
            label="Tu decisión sobre el escenario",
            placeholder="Describe qué decisión tomarías...",
            scale=4
        )
        eval_btn = gr.Button("⚖️ Evaluar", variant="secondary", scale=1)
    
    gr.Markdown("---")
    
    # Footer
    gr.Markdown("""
    ### ℹ️ Información del Sistema
    
    **Agentes disponibles:**
    - 🎓 **Formador**: Explica conceptos, procedimientos y protocolos del DTF-13
    - 🎭 **Simulador**: Genera escenarios operativos y evalúa decisiones
    - 🎯 **Director**: Clasifica automáticamente y enruta al agente apropiado
    
    **Desarrollado por:** Josué (Bombero + Data Scientist)  
    **Proyecto:** TFM MDATA + Capstone IIA
    """)
    
    # Event handlers
    init_btn.click(
        fn=initialize_system,
        outputs=init_status
    )
    
    submit_btn.click(
        fn=chat_response,
        inputs=[msg_input, chatbot, mode_selector],
        outputs=[chatbot, msg_input]
    )
    
    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot, mode_selector],
        outputs=[chatbot, msg_input]
    )
    
    eval_btn.click(
        fn=evaluate_decision,
        inputs=[decision_input, chatbot],
        outputs=[chatbot, decision_input]
    )
    
    clear_btn.click(
        fn=lambda: [],
        outputs=chatbot
    )
    
    # Inicializar al cargar
    demo.load(
        fn=initialize_system,
        outputs=init_status
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )