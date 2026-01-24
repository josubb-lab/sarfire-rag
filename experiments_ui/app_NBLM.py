#!/usr/bin/env python3
"""
SARFIRE-RAG v4.0 - Estilo NotebookLM
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

print("🔥 Inicializando SARFIRE-RAG v4.0...")

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
    response = result['answer']
    
    if result.get('source') == 'external':
        response += f"\n\n---\n\n{result.get('disclaimer', '')}"
        
        if result.get('external_sources'):
            response += "\n\n**Fuentes externas:**\n"
            for i, source in enumerate(result['external_sources'][:3], 1):
                response += f"- [{source['title']}]({source['url']})\n"
    
    elif result.get('sources'):
        response += "\n\n---\n\n**Fuentes:**\n"
        for i, source in enumerate(result['sources'][:3], 1):
            response += f"- {source['filename']} · Página {source['page']}\n"
    
    if result.get('should_ask_user'):
        response += f"\n\n💡 {result.get('question_for_user', '')}"
    
    return response


def format_simulador_response(result: dict, intention: str, scenario_state: dict) -> tuple:
    """Formatea la respuesta del Agente Simulador"""
    
    text_content = (
        result.get('scenario') or 
        result.get('evaluation') or 
        result.get('answer') or 
        result.get('response') or
        "No se pudo procesar la solicitud."
    )
    
    if intention == 'new_scenario':
        response = text_content
        
        new_state = {
            "text": text_content,
            "active": True
        }
        
        response += "\n\n---\n\n**💭 ¿Cómo actuarías en esta situación?**"
        
        if result.get('source') == 'external' and result.get('disclaimer'):
            response += f"\n\n{result['disclaimer']}"
        
        return response, new_state
    
    elif intention == 'scenario_response' and scenario_state.get("active"):
        response = f"**Evaluación de tu decisión**\n\n{text_content}"
        
        new_state = {"text": None, "active": False}
        
        response += "\n\n---\n\n✓ Evaluación completada"
        
        return response, new_state
    
    else:
        response = text_content
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
        
        if mode == "Automático":
            result = orchestration.process_query(message)
            agent_used = result.get('classification', {}).get('agent', 'formador')
            
        elif mode == "Formador":
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
            
        elif mode == "Simulador":
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
        error_msg = f"⚠️ Se produjo un error. Por favor, inténtalo de nuevo.\n\n<details><summary>Detalles técnicos</summary>\n\n```\n{str(e)}\n```\n</details>"
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
            "content": "Escenario cancelado. Puedes solicitar uno nuevo cuando quieras."
        })
    return history, {"text": None, "active": False}


# ============================================================================
# CSS - ESTILO NOTEBOOKLM
# ============================================================================

custom_css = """
/* NotebookLM Style - Limpio, moderno, profesional */

/* Variables */
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fa;
    --text-primary: #202124;
    --text-secondary: #5f6368;
    --border: #e8eaed;
    --accent: #1a73e8;
    --accent-hover: #1557b0;
    --success: #1e8e3e;
    --warning: #ea8600;
}

/* Contenedor principal */
.gradio-container {
    font-family: 'Google Sans', 'Segoe UI', Roboto, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    max-width: 1400px;
    margin: 0 auto;
}

/* Header estilo Google */
.header {
    padding: 32px 24px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-primary);
}

.header-content {
    max-width: 900px;
    margin: 0 auto;
}

.app-title {
    font-size: 28px;
    font-weight: 500;
    color: var(--text-primary);
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}

.app-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 400;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #e8f5e9;
    color: var(--success);
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    margin-top: 12px;
}

.status-dot {
    width: 6px;
    height: 6px;
    background: var(--success);
    border-radius: 50%;
}

/* Escenario activo */
.scenario-banner {
    background: #fef7e0;
    border-left: 3px solid var(--warning);
    padding: 12px 16px;
    margin: 16px 0;
    border-radius: 0 4px 4px 0;
    color: var(--text-primary);
    font-size: 14px;
}

/* Controles */
.control-section {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

/* Botones estilo Material */
button {
    border-radius: 20px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 8px 24px !important;
    transition: all 0.2s ease !important;
    border: none !important;
    letter-spacing: 0.25px !important;
}

.primary {
    background: var(--accent) !important;
    color: white !important;
}

.primary:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24) !important;
}

.secondary {
    background: white !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

.secondary:hover {
    background: var(--bg-secondary) !important;
}

/* Inputs estilo Material */
textarea, input {
    background: white !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    padding: 12px !important;
}

textarea:focus, input:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.1) !important;
}

/* Labels */
label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    margin-bottom: 8px !important;
}

/* Radio buttons */
.radio-group {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
}

/* Chatbot */
.message {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 8px 0 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* Sidebar */
.info-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

.info-card h3 {
    font-size: 16px;
    font-weight: 500;
    color: var(--text-primary);
    margin: 0 0 12px 0;
}

.info-card p {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin: 8px 0;
}

.info-card ul {
    list-style: none;
    padding: 0;
    margin: 8px 0;
}

.info-card li {
    padding: 6px 0;
    color: var(--text-secondary);
    font-size: 14px;
}

.info-card li:before {
    content: "• ";
    color: var(--accent);
    font-weight: bold;
}

.info-card strong {
    color: var(--text-primary);
    font-weight: 500;
}

/* Divider */
.divider {
    height: 1px;
    background: var(--border);
    margin: 16px 0;
}

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    color: var(--text-secondary);
    font-size: 12px;
    border-top: 1px solid var(--border);
    margin-top: 32px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
    background: #dadce0;
    border-radius: 6px;
    border: 3px solid var(--bg-secondary);
}

::-webkit-scrollbar-thumb:hover {
    background: #bdc1c6;
}
"""


# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

with gr.Blocks(title="SARFIRE-RAG") as demo:
    
    scenario_state = gr.State({"text": None, "active": False})
    
    # Header estilo Google
    gr.HTML("""
        <div class="header">
            <div class="header-content">
                <h1 class="app-title">SARFIRE-RAG</h1>
                <p class="app-subtitle">Asistente Inteligente para Formación en Emergencias Forestales</p>
                <div class="status-badge">
                    <span class="status-dot"></span>
                    Sistema operativo
                </div>
            </div>
        </div>
    """)
    
    with gr.Row():
        # Columna principal
        with gr.Column(scale=7):
            
            # Controles
            with gr.Row():
                mode_selector = gr.Radio(
                    choices=["Automático", "Formador", "Simulador"],
                    value="Automático",
                    label="Modo de operación",
                    container=True
                )
                
                external_checkbox = gr.Checkbox(
                    label="Permitir fuentes externas",
                    value=False,
                    info="Buscar en web si DTF-13 no tiene información",
                    container=True
                )
            
            # Chat
            chatbot = gr.Chatbot(
                value=[],
                label="",
                height=500,
                show_label=False
            )
            
            # Input
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Escribe tu mensaje...",
                    label="",
                    scale=5,
                    show_label=False
                )
                submit_btn = gr.Button("Enviar", variant="primary", scale=1)
            
            # Botones de acción
            with gr.Row():
                cancel_btn = gr.Button("Cancelar escenario", variant="secondary", size="sm")
                clear_btn = gr.Button("Nueva conversación", variant="secondary", size="sm")
        
        # Sidebar
        with gr.Column(scale=3):
            
            # Info del sistema
            gr.HTML("""
                <div class="info-card">
                    <h3>💡 Guía rápida</h3>
                    
                    <p><strong>Automático</strong><br>
                    El sistema decide qué agente usar según tu consulta.</p>
                    
                    <p><strong>Formador</strong><br>
                    Consultas técnicas y explicaciones basadas en DTF-13.</p>
                    
                    <p><strong>Simulador</strong><br>
                    Genera escenarios operativos y evalúa decisiones.</p>
                </div>
                
                <div class="info-card">
                    <h3>📚 Ejemplos</h3>
                    
                    <p><strong>Consultas:</strong></p>
                    <ul>
                        <li>¿Qué es el PMA?</li>
                        <li>Explica los niveles de activación</li>
                        <li>¿Cuáles son las funciones del DTF?</li>
                    </ul>
                    
                    <p><strong>Simulacros:</strong></p>
                    <ul>
                        <li>Genera escenario de incendio nocturno</li>
                        <li>Escenario con drones autónomos</li>
                        <li>Caso de evacuación urbana</li>
                    </ul>
                </div>
                
                <div class="info-card">
                    <h3>⚙️ Sistema</h3>
                    <p><strong>Modelo:</strong> Gemini 2.0 Flash</p>
                    <p><strong>Base de datos:</strong> ChromaDB</p>
                    <p><strong>Documentos:</strong> 66 chunks · DTF-13</p>
                    <p><strong>Búsqueda:</strong> Híbrida (semántica + keywords)</p>
                    <p><strong>Fallback:</strong> Tavily API</p>
                </div>
            """)
    
    # Footer
    gr.HTML("""
        <div class="footer">
            Proyecto académico · Capstone IIA · Enero 2026
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
    print("🚀 Lanzando SARFIRE-RAG v4.0...")
    print("📱 http://localhost:7860")
    print("🎨 Diseño: NotebookLM Style\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=custom_css
    )
