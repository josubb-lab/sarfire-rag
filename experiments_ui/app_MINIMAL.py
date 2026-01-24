#!/usr/bin/env python3
"""
SARFIRE-RAG v3.0 - Diseño Retro Minimalista
Sistema Multi-Agente para Formación en Emergencias
"""

import gradio as gr
import sys
import re
sys.path.append('src')

from rag import EmbeddingsGenerator, VectorStore, RAGPipeline
from agents import FormadorAgent, SimuladorAgent, DirectorAgent, OrchestrationSystem


# ============================================================================
# INICIALIZACIÓN DEL SISTEMA
# ============================================================================

print("🔥 Inicializando SARFIRE-RAG v3.0...")

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
    use_hybrid_search=True,
    enable_external_fallback=True,
    relevance_threshold=0.5
)

formador = FormadorAgent(rag_pipeline=rag)
simulador = SimuladorAgent(rag_pipeline=rag)
director = DirectorAgent(model_name="gemini-2.0-flash")

orchestration = OrchestrationSystem(
    formador_agent=formador,
    simulador_agent=simulador,
    director_agent=director
)

print("✅ Sistema inicializado\n")


# ============================================================================
# UTILIDADES
# ============================================================================

def detect_user_intention(message: str, scenario_active: bool) -> str:
    """Detecta la intención con PRIORIDAD al escenario activo"""
    message_lower = message.lower()
    
    scenario_keywords = [
        'genera', 'crea', 'dame', 'escenario', 'simulacro', 
        'caso', 'situación', 'plantea', 'simula', 'nuevo'
    ]
    
    if scenario_active:
        asking_new_scenario = any(kw in message_lower for kw in scenario_keywords)
        if asking_new_scenario:
            return 'new_scenario'
        else:
            return 'scenario_response'
    
    if any(kw in message_lower for kw in scenario_keywords):
        return 'new_scenario'
    
    return 'general_query'


def build_conversation_context(history: list) -> str:
    """Construye el contexto conversacional"""
    if not history:
        return ""
    
    context = "HISTORIAL DE LA CONVERSACIÓN:\n\n"
    for i, msg in enumerate(history[-5:], 1):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue
        
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        content = msg["content"]
        
        if not isinstance(content, str):
            content = str(content)
        
        clean_content = re.sub(r'\*\*.*?\*\*', '', content)
        clean_content = re.sub(r'---.*?---', '', clean_content, flags=re.DOTALL)
        
        if len(clean_content) > 200:
            clean_content = clean_content[:200] + "..."
        
        context += f"{i}. {role}: {clean_content}\n\n"
    
    return context


# ============================================================================
# FORMATEO DE RESPUESTAS
# ============================================================================

def format_formador_response(result: dict) -> str:
    """Formatea la respuesta del Agente Formador"""
    response = "**🎓 AGENTE FORMADOR**\n\n"
    response += result['answer']
    
    if result.get('source') == 'external':
        response += f"\n\n{result.get('disclaimer', '')}"
        
        if result.get('external_sources'):
            response += "\n\n**🌐 Fuentes externas:**\n"
            for i, source in enumerate(result['external_sources'][:3], 1):
                response += f"\n{i}. [{source['title']}]({source['url']})"
    
    elif result.get('sources'):
        response += "\n\n---\n**📚 Fuentes consultadas:**\n"
        for i, source in enumerate(result['sources'][:3], 1):
            response += f"\n{i}. {source['filename']} (Página {source['page']})"
    
    if result.get('should_ask_user'):
        response += f"\n\n{result.get('question_for_user', '')}"
    
    return response


def format_simulador_response(result: dict, intention: str, scenario_state: dict) -> tuple:
    """Formatea la respuesta del Agente Simulador"""
    response = "**🎭 AGENTE SIMULADOR**\n\n"
    
    text_content = (
        result.get('scenario') or 
        result.get('evaluation') or 
        result.get('answer') or 
        result.get('response') or
        "No se pudo procesar la solicitud."
    )
    
    if intention == 'new_scenario':
        response += text_content
        
        new_state = {
            "text": text_content,
            "active": True
        }
        
        response += "\n\n---\n💡 **¿Qué decisión tomarías?**"
        
        if result.get('source') == 'external' and result.get('disclaimer'):
            response += f"\n\n{result['disclaimer']}"
        
        return response, new_state
    
    elif intention == 'scenario_response' and scenario_state.get("active"):
        response += "**⚖️ EVALUACIÓN**\n\n"
        response += text_content
        
        new_state = {"text": None, "active": False}
        
        response += "\n\n---\n✅ *Evaluación completada.*"
        
        return response, new_state
    
    else:
        response += text_content
        new_state = {"text": None, "active": False}
        return response, new_state


# ============================================================================
# LÓGICA DE PROCESAMIENTO
# ============================================================================

def process_message(
    message: str, 
    mode: str, 
    history: list, 
    scenario_state: dict,
    allow_external: bool
):
    """Procesa el mensaje del usuario"""
    
    internal_history = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg:
            internal_history.append(msg)
    
    try:
        intention = detect_user_intention(message, scenario_state.get("active", False))
        context = build_conversation_context(internal_history)
        
        if mode == "🎯 Automático":
            result = orchestration.process_query(message)
            agent_used = result.get('classification', {}).get('agent', 'formador')
            
        elif mode == "🎓 Formador":
            if hasattr(formador, 'rag_pipeline'):
                rag_result = formador.rag_pipeline.query(
                    question=message,
                    allow_external=allow_external if allow_external else None
                )
                result = {
                    'answer': rag_result['answer'],
                    'sources': rag_result.get('sources', []),
                    'source': rag_result.get('source', 'internal'),
                    'external_sources': rag_result.get('external_sources'),
                    'disclaimer': rag_result.get('disclaimer'),
                    'should_ask_user': rag_result.get('should_ask_user', False),
                    'question_for_user': rag_result.get('question_for_user')
                }
            else:
                result = orchestration.process_query(message, force_agent='formador')
            
            agent_used = 'formador'
            
        elif mode == "🎭 Simulador":
            if intention == 'scenario_response' and scenario_state.get("active"):
                result = simulador.evaluate_decision(user_decision=message)
                result['answer'] = result.get('evaluation', '')
                
            elif intention == 'new_scenario':
                result = simulador.create_scenario(
                    topic=message, 
                    allow_external=True if allow_external else False
                )
                result['answer'] = result.get('scenario', '')
            
            else:
                result = orchestration.process_query(message, force_agent='simulador')
            
            agent_used = 'simulador'
        
        if agent_used == 'formador':
            response = format_formador_response(result)
            scenario_state = {"text": None, "active": False}
        else:
            response, scenario_state = format_simulador_response(result, intention, scenario_state)
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
    except Exception as e:
        import traceback
        error_msg = f"⚠️ **Error del sistema**\n\nPor favor, inténtalo de nuevo.\n\n<details><summary>Detalles</summary>\n\n```\n{str(e)}\n```\n</details>"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        scenario_state = {"text": None, "active": False}
    
    return history, "", scenario_state, False


def clear_chat():
    """Limpia el chat"""
    return [], "", {"text": None, "active": False}, False


def cancel_scenario(history: list, scenario_state: dict):
    """Cancela el escenario activo"""
    if scenario_state.get("active"):
        history.append({
            "role": "assistant", 
            "content": "🔄 Escenario cancelado."
        })
    return history, {"text": None, "active": False}


# ============================================================================
# CSS - DISEÑO RETRO MINIMALISTA
# ============================================================================

custom_css = """
/* Retro Minimalista - Inspirado en terminales clásicas */
.gradio-container {
    font-family: 'Courier New', 'Monaco', monospace !important;
    background: #0d1117 !important;
    color: #c9d1d9 !important;
}

/* Encabezado simple */
.header {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 2rem;
    border-bottom: 2px solid #30363d;
}

.header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    margin: 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.header p {
    color: #8b949e;
    font-size: 0.85rem;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

/* Status badge */
.status {
    display: inline-block;
    background: #238636;
    color: #ffffff;
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}

/* Banner de escenario activo */
.scenario-active {
    background: #1f2937;
    border-left: 3px solid #f59e0b;
    padding: 0.75rem 1rem;
    margin: 1rem 0;
    color: #f59e0b;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
}

/* Botones minimalistas */
button {
    border-radius: 2px !important;
    font-weight: 600 !important;
    font-family: 'Courier New', monospace !important;
    letter-spacing: 0.03em !important;
    transition: all 0.15s ease !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
}

.primary {
    background: #58a6ff !important;
    color: #0d1117 !important;
    border: none !important;
}

.primary:hover {
    background: #79c0ff !important;
}

.secondary {
    background: transparent !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}

.secondary:hover {
    border-color: #58a6ff !important;
    color: #58a6ff !important;
}

/* Inputs */
textarea, input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 2px !important;
    font-family: 'Courier New', monospace !important;
}

textarea:focus, input:focus {
    border-color: #58a6ff !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Labels */
label {
    color: #8b949e !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Panel lateral */
.sidebar {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 1.5rem;
    font-size: 0.85rem;
    line-height: 1.6;
}

.sidebar h3 {
    color: #58a6ff;
    font-size: 0.9rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid #30363d;
    padding-bottom: 0.5rem;
}

.sidebar h3:first-child {
    margin-top: 0;
}

.sidebar ul {
    list-style: none;
    padding-left: 0;
}

.sidebar li {
    padding: 0.25rem 0;
}

.sidebar li:before {
    content: "> ";
    color: #58a6ff;
}

.sidebar p {
    margin: 0.5rem 0;
}

.sidebar strong {
    color: #c9d1d9;
}

/* Radio buttons */
.radio-group {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 0.75rem;
}

/* Chatbot */
.message {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #0d1117;
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #484f58;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1rem;
    margin-top: 2rem;
    border-top: 1px solid #30363d;
    color: #484f58;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}
"""


# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

with gr.Blocks(title="SARFIRE-RAG") as demo:
    
    scenario_state = gr.State({"text": None, "active": False})
    
    # Header minimalista
    gr.HTML("""
        <div class="header">
            <h1>SARFIRE-RAG</h1>
            <p>Sistema Multi-Agente · Emergencias Forestales</p>
            <span class="status">● ONLINE</span>
        </div>
    """)
    
    with gr.Row():
        # Columna principal
        with gr.Column(scale=3):
            
            # Selector de modo
            mode_selector = gr.Radio(
                choices=["🎯 Automático", "🎓 Formador", "🎭 Simulador"],
                value="🎯 Automático",
                label="Modo",
                container=True
            )
            
            # Checkbox fuentes externas
            external_checkbox = gr.Checkbox(
                label="🌐 Permitir fuentes externas",
                value=False,
                info="Buscar en web si DTF-13 no tiene información"
            )
            
            # Chat
            chatbot = gr.Chatbot(
                value=[],
                label="Conversación",
                height=450,
                show_label=False
            )
            
            # Input
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Escribe tu mensaje...",
                    label="Mensaje",
                    scale=4,
                    show_label=False
                )
                submit_btn = gr.Button("Enviar", variant="primary", scale=1)
            
            # Botones
            with gr.Row():
                cancel_btn = gr.Button("Cancelar escenario", variant="secondary", size="sm")
                clear_btn = gr.Button("Limpiar", variant="secondary", size="sm")
        
        # Sidebar
        with gr.Column(scale=1):
            gr.HTML("""
                <div class="sidebar">
                    <h3>Sistema</h3>
                    <p><strong>Modelo:</strong> Gemini 2.0 Flash</p>
                    <p><strong>DB:</strong> ChromaDB (66 chunks)</p>
                    <p><strong>Docs:</strong> DTF-13</p>
                    <p><strong>Búsqueda:</strong> Híbrida</p>
                    
                    <h3>Modos</h3>
                    <p><strong>Automático</strong><br>
                    El Director enruta automáticamente.</p>
                    
                    <p><strong>Formador</strong><br>
                    Consultas y explicaciones técnicas.</p>
                    
                    <p><strong>Simulador</strong><br>
                    Escenarios operativos y evaluación.</p>
                    
                    <h3>Ejemplos</h3>
                    <ul>
                        <li>¿Qué es el PMA?</li>
                        <li>Niveles de activación</li>
                        <li>Genera escenario nocturno</li>
                        <li>Escenario con drones</li>
                    </ul>
                    
                    <h3>Fuentes Externas</h3>
                    <p>Activa para consultar Tavily API cuando DTF-13 no tenga datos (relevancia < 0.5).</p>
                </div>
            """)
    
    # Footer
    gr.HTML("""
        <div class="footer">
            PROYECTO ACADÉMICO · CAPSTONE IIA · ENERO 2026
        </div>
    """)
    
    # Event handlers
    submit_btn.click(
        fn=process_message,
        inputs=[msg, mode_selector, chatbot, scenario_state, external_checkbox],
        outputs=[chatbot, msg, scenario_state, external_checkbox]
    )
    
    msg.submit(
        fn=process_message,
        inputs=[msg, mode_selector, chatbot, scenario_state, external_checkbox],
        outputs=[chatbot, msg, scenario_state, external_checkbox]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, msg, scenario_state, external_checkbox]
    )
    
    cancel_btn.click(
        fn=cancel_scenario,
        inputs=[chatbot, scenario_state],
        outputs=[chatbot, scenario_state]
    )


# ============================================================================
# LANZAR
# ============================================================================

if __name__ == "__main__":
    print("🚀 Lanzando SARFIRE-RAG v3.0...")
    print("📱 http://localhost:7860")
    print("🎨 Diseño: Retro Minimalista\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=custom_css
    )
