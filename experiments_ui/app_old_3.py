#!/usr/bin/env python3
"""
SARFIRE-RAG - Interfaz Gradio v2.0
Sistema Multi-Agente para Formación en Emergencias Forestales

NUEVAS CARACTERÍSTICAS:
- Fallback a fuentes externas (Tavily API)
- Scoring de relevancia automático
- Disclaimer para fuentes externas
- Control de usuario sobre búsquedas externas
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

print("🔥 Inicializando SARFIRE-RAG v2.0...")

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
    enable_external_fallback=True,  # ← NUEVO
    relevance_threshold=0.5  # ← NUEVO
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

def detect_user_intention(message: str, scenario_active: bool) -> str:
    """
    Detecta la intención con lógica diferenciada si hay escenario activo
    """
    message_lower = message.lower()
    
    # Palabras clave generales para pedir un escenario (cuando no hay nada activo)
    start_keywords = [
        'genera', 'crea', 'dame', 'escenario', 'simulacro', 
        'caso', 'situación', 'plantea', 'simula', 'nuevo'
    ]
    
    # Palabras clave ESTRICTAS para reiniciar/cambiar (cuando YA hay escenario activo)
    reset_keywords = [
        'nuevo', 'otro', 'otra', 'siguiente', 'cambia', 'reinicia', 'borra'
    ]
    
    # PRIORIDAD 1: Si hay escenario activo
    if scenario_active:
        # Solo generar nuevo si usa palabras explícitas de cambio/reinicio
        asking_new_scenario = any(kw in message_lower for kw in reset_keywords)
        
        # O si dice explícitamente "generar escenario" (combinación fuerte)
        explicit_command = "genera" in message_lower and "escenario" in message_lower
        
        if asking_new_scenario or explicit_command:
            print("🔍 DEBUG: Usuario pide CAMBIAR escenario activo")
            return 'new_scenario'
        else:
            print("🔍 DEBUG: Usuario responde a escenario ACTIVO")
            return 'scenario_response'
    
    # PRIORIDAD 2: Si pide escenario y no hay activo (usa la lista amplia)
    if any(kw in message_lower for kw in start_keywords):
        print("🔍 DEBUG: Usuario pide nuevo escenario (inicio)")
        return 'new_scenario'
    
    print("🔍 DEBUG: Consulta general")
    return 'general_query'


def build_conversation_context(history: list) -> str:
    """
    Construye el contexto conversacional completo para el agente
    """
    if not history:
        return ""
    
    context = "HISTORIAL DE LA CONVERSACIÓN:\n\n"
    for i, msg in enumerate(history[-5:], 1):
        # Manejar formato de diccionario
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue
        
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        content = msg["content"]
        
        # Asegurarse de que content es string
        if not isinstance(content, str):
            content = str(content)
        
        # Limpiar formatos markdown del contenido para el contexto
        clean_content = re.sub(r'\*\*.*?\*\*', '', content)
        clean_content = re.sub(r'---.*?---', '', clean_content, flags=re.DOTALL)
        
        # Limitar longitud
        if len(clean_content) > 200:
            clean_content = clean_content[:200] + "..."
        
        context += f"{i}. {role}: {clean_content}\n\n"
    
    return context


# ============================================================================
# FORMATEO DE RESPUESTAS
# ============================================================================

def format_formador_response(result: dict) -> str:
    """
    Formatea la respuesta del Agente Formador
    Ahora con soporte de fuentes externas
    """
    response = "**🎓 AGENTE FORMADOR**\n\n"
    response += result['answer']
    
    # NUEVO: Disclaimer si es fuente externa
    if result.get('source') == 'external':
        response += f"\n\n{result.get('disclaimer', '')}"
        
        # Fuentes externas
        if result.get('external_sources'):
            response += "\n\n**🌐 Fuentes externas:**\n"
            for i, source in enumerate(result['external_sources'][:3], 1):
                response += f"\n{i}. [{source['title']}]({source['url']})"
    
    # Fuentes internas (DTF-13)
    elif result.get('sources'):
        response += "\n\n---\n**📚 Fuentes consultadas (DTF-13):**\n"
        for i, source in enumerate(result['sources'][:3], 1):
            response += f"\n{i}. {source['filename']} (Página {source['page']})"
    
    # NUEVO: Mensaje si debe preguntar al usuario
    if result.get('should_ask_user'):
        response += f"\n\n{result.get('question_for_user', '')}"
    
    return response


def format_simulador_response(result: dict, intention: str, scenario_state: dict) -> tuple:
    """
    Formatea la respuesta del Agente Simulador
    
    Returns:
        (response_text, updated_scenario_state)
    """
    response = "**🎭 AGENTE SIMULADOR**\n\n"
    
    # Obtener el texto de respuesta (puede estar en varios campos)
    text_content = (
        result.get('scenario') or 
        result.get('evaluation') or 
        result.get('answer') or 
        result.get('response') or
        "No se pudo procesar la solicitud."
    )
    
    # CASO 1: Es un escenario nuevo
    if intention == 'new_scenario':
        response += text_content
        
        # Actualizar estado
        new_state = {
            "text": text_content,
            "active": True
        }
        
        response += "\n\n---\n💡 **¿Qué decisión tomarías en esta situación?**\n"
        response += "*Escribe tu respuesta en el siguiente mensaje para que pueda evaluarla.*"
        
        return response, new_state
    
    # CASO 2: Es una evaluación de decisión
    elif intention == 'scenario_response' and scenario_state.get("active"):
        response += "**⚖️ EVALUACIÓN DE TU DECISIÓN**\n\n"
        response += text_content
        
        # Limpiar estado (escenario terminado)
        new_state = {"text": None, "active": False}
        
        response += "\n\n---\n✅ *Evaluación completada. Puedes solicitar un nuevo escenario cuando quieras.*"
        
        return response, new_state
    
    # CASO 3: Respuesta genérica del simulador
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
    allow_external: bool  # ← NUEVO parámetro
):
    """
    Procesa el mensaje del usuario según el modo seleccionado
    
    Args:
        message: Mensaje del usuario
        mode: Modo seleccionado (Automático/Formador/Simulador)
        history: Historial de la conversación
        scenario_state: Estado del escenario activo
        allow_external: Si permite búsqueda externa
        
    Returns:
        Tuple (historial, input vacío, scenario_state, allow_external resetted)
    """
    
    # Convertir historial de Gradio a formato interno
    internal_history = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg:
            internal_history.append(msg)
    
    try:
        # Detectar intención del usuario
        intention = detect_user_intention(message, scenario_state.get("active", False))
        
        # Construir contexto conversacional
        context = build_conversation_context(internal_history)
        
        # Procesar según modo
        if mode == "🎯 Automático (Director decide)":
            result = orchestration.process_query(message)
            agent_used = result.get('classification', {}).get('agent', 'formador')
            
        elif mode == "🎓 Formador (Explicaciones)":
            # NUEVO: Pasar allow_external al agente formador
            if hasattr(formador, 'rag_pipeline'):
                # Llamar directamente al RAG con control de external
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
                # Usuario está respondiendo a un escenario - llamar evaluate_decision
                print("🔍 DEBUG: Evaluando decisión del usuario")
                
                # Llamar directamente al agente simulador
                result = simulador.evaluate_decision(user_decision=message)
                
                # Añadir campos necesarios para formateo
                result['answer'] = result.get('evaluation', '')
                
            elif intention == 'new_scenario':
                # Usuario pide nuevo escenario - llamar create_scenario
                print("🔍 DEBUG: Generando nuevo escenario")
                
                result = simulador.create_scenario(topic=message)
                
                # Añadir campos necesarios para formateo
                result['answer'] = result.get('scenario', '')
            
            else:
                # Fallback genérico
                result = orchestration.process_query(message, force_agent='simulador')
            
            agent_used = 'simulador'
        
        # Formatear respuesta según el agente
        if agent_used == 'formador':
            response = format_formador_response(result)
            scenario_state = {"text": None, "active": False}
        else:
            response, scenario_state = format_simulador_response(result, intention, scenario_state)
        
        # Añadir mensajes al historial en formato Gradio 6
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        scenario_state = {"text": None, "active": False}
    
    # Resetear allow_external después de cada mensaje
    return history, "", scenario_state, False


def clear_chat():
    """Limpia el historial del chat y el estado del escenario"""
    return [], "", {"text": None, "active": False}, False


# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

# CSS personalizado
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.external-warning {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
}
"""

# Crear interfaz
with gr.Blocks(title="SARFIRE-RAG v2.0") as demo:
    
    # Estados persistentes
    scenario_state = gr.State({"text": None, "active": False})
    allow_external_state = gr.State(False)  # ← NUEVO
    
    # Header
    gr.HTML("""
        <div class="header">
            <h1>🔥 SARFIRE-RAG v2.0</h1>
            <p>Sistema Multi-Agente con Fallback Externo</p>
            <p style="font-size: 14px; opacity: 0.9;">DTF-13 + Fuentes Web (Tavily API)</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            
            # Selector de modo
            mode_selector = gr.Radio(
                choices=[
                    "🎯 Automático (Director decide)",
                    "🎓 Formador (Explicaciones)",
                    "🎭 Simulador (Escenarios)"
                ],
                value="🎯 Automático (Director decide)",
                label="Modo de Operación",
                info="El Director enruta automáticamente, o puedes forzar un agente específico"
            )
            
            # NUEVO: Checkbox para fuentes externas
            external_checkbox = gr.Checkbox(
                label="🌐 Permitir búsqueda en fuentes externas",
                value=False,
                info="Si la documentación interna no tiene info suficiente, buscar en web"
            )
            
            # Chat
            chatbot = gr.Chatbot(
                value=[],
                label="Conversación",
                height=450
            )
            
            # Input
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Escribe tu consulta...",
                    label="Tu mensaje",
                    scale=4
                )
                submit_btn = gr.Button("Enviar", variant="primary", scale=1)
            
            # Botones de control
            with gr.Row():
                clear_btn = gr.Button("🗑️ Limpiar conversación", variant="secondary")
        
        with gr.Column(scale=1):
            
            # Info panel
            gr.Markdown("""
            ### 📋 Guía de Uso
            
            **🎯 Modo Automático:**
            - El Director decide el agente
            - Ideal para uso general
            
            **🎓 Modo Formador:**
            - Consultas sobre DTF-13
            - Explicaciones técnicas
            
            **🎭 Modo Simulador:**
            - Genera escenarios
            - Evalúa decisiones
            
            ---
            
            ### 🌐 Fuentes Externas
            
            **Cuándo se activa:**
            - Relevancia < 0.5 en DTF-13
            - Checkbox activado
            
            **Advertencia:**
            - Info externa NO oficial
            - Verificar con protocolos
            
            ---
            
            ### 💡 Ejemplos
            
            **Interno (DTF-13):**
            - "¿Qué es el PMA?"
            - "Niveles de activación"
            
            **Externo (Web):**
            - "Últimas tecnologías drones"
            - "Normativa europea"
            """)
            
            gr.Markdown("""
            ---
            ### ℹ️ Info Técnica
            
            **Docs:** DTF-13 (66 chunks)  
            **Modelo:** Gemini 2.0 Flash  
            **Búsqueda:** Híbrida  
            **Fallback:** Tavily API  
            **Umbral:** 0.5  
            
            ---
            
            **Proyecto académico**  
            Capstone IIA · Enero 2026
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


# ============================================================================
# LANZAR APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    print("🚀 Lanzando SARFIRE-RAG v2.0...")
    print("📱 URL: http://localhost:7860")
    print("🌐 Fallback externo: ACTIVADO")
    print("\n⚠️  Presiona Ctrl+C para detener\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=custom_css
    )
