#!/usr/bin/env python3
"""
SARFIRE-RAG v3.0 - Panel de Control de Emergencias
Sistema Multi-Agente con Diseño Profesional
"""

import gradio as gr
import sys
import re
from datetime import datetime
sys.path.append('src')

from rag import EmbeddingsGenerator, VectorStore, RAGPipeline
from agents import FormadorAgent, SimuladorAgent, DirectorAgent, OrchestrationSystem


# ============================================================================
# INICIALIZACIÓN DEL SISTEMA
# ============================================================================

print("🔥 Inicializando SARFIRE-RAG v3.0...")

# Cargar componentes RAG
embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
vector_store = VectorStore(
    persist_directory="data/processed/chromadb",
    collection_name="sarfire_docs"
)

# RAG Pipeline con fallback externo
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

# Crear agentes
formador = FormadorAgent(rag_pipeline=rag)
simulador = SimuladorAgent(rag_pipeline=rag)
director = DirectorAgent(model_name="gemini-2.0-flash")

# Sistema de orquestación
orchestration = OrchestrationSystem(
    formador_agent=formador,
    simulador_agent=simulador,
    director_agent=director
)

print("✅ Sistema inicializado correctamente\n")


# ============================================================================
# UTILIDADES
# ============================================================================

def get_current_time():
    """Retorna hora actual formateada"""
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def get_current_date():
    """Retorna fecha actual formateada"""
    now = datetime.now()
    return now.strftime("%d/%m/%Y")

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
        response += "\n\n---\n**📚 Fuentes consultadas (DTF-13):**\n"
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
        
        response += "\n\n---\n💡 **¿Qué decisión tomarías en esta situación?**\n"
        response += "*Escribe tu respuesta en el siguiente mensaje para que pueda evaluarla.*"
        
        # Añadir disclaimer si es externo
        if result.get('source') == 'external' and result.get('disclaimer'):
            response += f"\n\n{result['disclaimer']}"
        
        return response, new_state
    
    elif intention == 'scenario_response' and scenario_state.get("active"):
        response += "**⚖️ EVALUACIÓN DE TU DECISIÓN**\n\n"
        response += text_content
        
        new_state = {"text": None, "active": False}
        
        response += "\n\n---\n✅ *Evaluación completada. Puedes solicitar un nuevo escenario cuando quieras.*"
        
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
        
        # Procesar según modo
        if mode == "🎯 Automático (Director decide)":
            result = orchestration.process_query(message)
            agent_used = result.get('classification', {}).get('agent', 'formador')
            
        elif mode == "🎓 Formador (Explicaciones)":
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
            
        elif mode == "🎭 Simulador (Escenarios)":
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
        
        # Formatear respuesta
        if agent_used == 'formador':
            response = format_formador_response(result)
            scenario_state = {"text": None, "active": False}
        else:
            response, scenario_state = format_simulador_response(result, intention, scenario_state)
        
        # Añadir al historial
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
    except Exception as e:
        import traceback
        error_msg = f"⚠️ **Error del sistema**\n\nSe ha producido un error. Por favor, inténtalo de nuevo o contacta con soporte.\n\n<details><summary>Detalles técnicos</summary>\n\n```\n{str(e)}\n```\n</details>"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        scenario_state = {"text": None, "active": False}
    
    return history, "", scenario_state, False


def clear_chat():
    """Limpia el historial del chat"""
    return [], "", {"text": None, "active": False}, False


def cancel_scenario(history: list, scenario_state: dict):
    """Cancela el escenario activo sin borrar el chat"""
    if scenario_state.get("active"):
        history.append({
            "role": "assistant", 
            "content": "🔄 **Escenario cancelado.** Puedes solicitar uno nuevo cuando quieras."
        })
    return history, {"text": None, "active": False}


# ============================================================================
# INTERFAZ GRADIO - DISEÑO PROFESIONAL
# ============================================================================

# CSS personalizado - Diseño tipo panel de control
custom_css = """
/* Variables de color - Tema oscuro profesional */
:root {
    --primary-bg: #0a0e27;
    --secondary-bg: #1a1f3a;
    --accent-color: #00d4ff;
    --accent-hover: #00b8e6;
    --text-primary: #e0e6ed;
    --text-secondary: #8b95a8;
    --border-color: #2d3548;
    --success: #00ff88;
    --warning: #ffa726;
    --danger: #ff5252;
}

/* Contenedor principal */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background: var(--primary-bg) !important;
    color: var(--text-primary) !important;
}

/* Header personalizado */
.main-header {
    background: linear-gradient(135deg, #1a1f3a 0%, #0a0e27 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
}

.header-title {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-color), #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 14px;
    margin-top: 8px;
}

/* Widgets del panel */
.control-panel {
    background: var(--secondary-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
}

.widget {
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
}

.widget-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
    margin-bottom: 6px;
    font-weight: 600;
}

.widget-value {
    font-size: 20px;
    font-weight: 700;
    color: var(--accent-color);
    font-family: 'Courier New', monospace;
}

/* Status indicators */
.status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

.status-online {
    background: var(--success);
    box-shadow: 0 0 10px var(--success);
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Escenario activo banner */
.scenario-active-banner {
    background: linear-gradient(90deg, rgba(255, 167, 38, 0.2), rgba(255, 167, 38, 0.05));
    border-left: 4px solid var(--warning);
    padding: 12px 16px;
    border-radius: 6px;
    margin: 12px 0;
    font-size: 13px;
    color: var(--warning);
    font-weight: 600;
}

/* Botones */
button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--border-color) !important;
}

button.primary {
    background: var(--accent-color) !important;
    color: var(--primary-bg) !important;
}

button.primary:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3) !important;
}

/* Chatbot */
.message {
    border-radius: 8px !important;
    padding: 12px 16px !important;
}

/* Inputs */
textarea, input {
    background: var(--secondary-bg) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
}

textarea:focus, input:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1) !important;
}

/* Labels */
label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* Info panels */
.info-panel {
    background: var(--secondary-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    font-size: 13px;
    line-height: 1.6;
}

.info-panel h3 {
    color: var(--accent-color);
    font-size: 14px;
    font-weight: 700;
    margin-top: 16px;
    margin-bottom: 8px;
}

.info-panel ul {
    list-style: none;
    padding-left: 0;
}

.info-panel li:before {
    content: "▹ ";
    color: var(--accent-color);
    font-weight: bold;
}
"""

# JavaScript para actualizar hora en tiempo real
js_code = """
function updateTime() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('es-ES');
    const dateStr = now.toLocaleDateString('es-ES');
    
    // Actualizar elementos si existen
    const timeEl = document.querySelector('#live-time');
    const dateEl = document.querySelector('#live-date');
    
    if (timeEl) timeEl.textContent = timeStr;
    if (dateEl) dateEl.textContent = dateStr;
}

// Actualizar cada segundo
setInterval(updateTime, 1000);
updateTime();
"""

# Crear interfaz
with gr.Blocks(title="SARFIRE-RAG Control Panel", css=custom_css, js=js_code) as demo:
    
    # Estados persistentes
    scenario_state = gr.State({"text": None, "active": False})
    
    # Header principal
    gr.HTML("""
        <div class="main-header">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 class="header-title">🔥 SARFIRE-RAG</h1>
                    <p class="header-subtitle">Sistema Multi-Agente · Panel de Control de Emergencias</p>
                </div>
                <div style="text-align: right;">
                    <span class="status-indicator status-online"></span>
                    <span style="color: #00ff88; font-weight: 600;">SISTEMA OPERATIVO</span>
                </div>
            </div>
        </div>
    """)
    
    with gr.Row():
        # Columna principal - Chat
        with gr.Column(scale=3):
            
            # Panel de control superior
            gr.HTML("""
                <div class="control-panel">
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                        <div class="widget">
                            <div class="widget-title">📅 Fecha</div>
                            <div class="widget-value" id="live-date">--/--/----</div>
                        </div>
                        <div class="widget">
                            <div class="widget-title">🕐 Hora</div>
                            <div class="widget-value" id="live-time">--:--:--</div>
                        </div>
                        <div class="widget">
                            <div class="widget-title">📍 Ubicación</div>
                            <div class="widget-value" style="font-size: 16px;">Madrid, ES</div>
                        </div>
                    </div>
                </div>
            """)
            
            # Selector de modo
            mode_selector = gr.Radio(
                choices=[
                    "🎯 Automático (Director decide)",
                    "🎓 Formador (Explicaciones)",
                    "🎭 Simulador (Escenarios)"
                ],
                value="🎯 Automático (Director decide)",
                label="⚙️ Modo de Operación",
                info="Selecciona el modo de funcionamiento del sistema"
            )
            
            # Checkbox fuentes externas
            external_checkbox = gr.Checkbox(
                label="🌐 Permitir búsqueda en fuentes externas",
                value=False,
                info="Activar para consultar web cuando DTF-13 no tenga información"
            )
            
            # Indicador escenario activo (inicialmente oculto)
            scenario_indicator = gr.HTML(
                visible=False,
                value="""
                <div class="scenario-active-banner">
                    ⚠️ ESCENARIO ACTIVO - Esperando tu decisión
                </div>
                """
            )
            
            # Chat
            chatbot = gr.Chatbot(
                value=[],
                label="💬 Conversación",
                height=400,
                show_label=False
            )
            
            # Input
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Escribe tu consulta o respuesta...",
                    label="Mensaje",
                    scale=4,
                    show_label=False
                )
                submit_btn = gr.Button("📤 Enviar", variant="primary", scale=1)
            
            # Botones de control
            with gr.Row():
                cancel_btn = gr.Button("🚫 Cancelar escenario", variant="secondary", size="sm")
                clear_btn = gr.Button("🗑️ Limpiar conversación", variant="secondary", size="sm")
        
        # Columna lateral - Info
        with gr.Column(scale=1):
            
            gr.HTML("""
                <div class="info-panel">
                    <h3>📊 Estado del Sistema</h3>
                    <p><strong>Modelo:</strong> Gemini 2.0 Flash</p>
                    <p><strong>Vector DB:</strong> ChromaDB</p>
                    <p><strong>Documentos:</strong> 66 chunks</p>
                    <p><strong>Búsqueda:</strong> Híbrida</p>
                    <p><strong>Fallback:</strong> Tavily API</p>
                    
                    <h3>🎯 Guía Rápida</h3>
                    
                    <p><strong>Modo Automático:</strong><br>
                    El Director analiza tu consulta y la enruta automáticamente.</p>
                    
                    <p><strong>Modo Formador:</strong><br>
                    Consultas técnicas y explicaciones basadas en DTF-13.</p>
                    
                    <p><strong>Modo Simulador:</strong><br>
                    Genera escenarios operativos y evalúa tus decisiones.</p>
                    
                    <h3>💡 Ejemplos</h3>
                    
                    <p><strong>Formador:</strong></p>
                    <ul>
                        <li>"¿Qué es el PMA?"</li>
                        <li>"Niveles de activación"</li>
                    </ul>
                    
                    <p><strong>Simulador:</strong></p>
                    <ul>
                        <li>"Genera escenario nocturno"</li>
                        <li>"Escenario con drones"</li>
                    </ul>
                    
                    <h3>⚙️ Configuración</h3>
                    <p>Activa <strong>fuentes externas</strong> para consultar información actualizada de la web cuando DTF-13 no tenga datos.</p>
                    
                    <hr style="border-color: var(--border-color); margin: 16px 0;">
                    
                    <p style="font-size: 11px; color: var(--text-secondary); text-align: center;">
                        <strong>Proyecto académico</strong><br>
                        Capstone IIA · Enero 2026
                    </p>
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
# LANZAR APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Lanzando SARFIRE-RAG Control Panel v3.0...")
    print("📱 URL: http://localhost:7860")
    print("🎨 Diseño: Panel de Control Profesional")
    print("\n⚠️  Presiona Ctrl+C para detener\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
