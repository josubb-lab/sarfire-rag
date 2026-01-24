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
    """Detecta intención con prioridad al escenario activo."""
    message_lower = message.lower().strip()

    if scenario_active:
        explicit_new = re.search(
            r"\b(nuevo|otra|otro)\s+(escenario|simulacro|caso)\b|\b(genera|crea|plantea)\s+(otro|un\s+nuevo)\b",
            message_lower
        )
        if explicit_new:
            return "new_scenario"
        return "scenario_response"

    scenario_keywords = [
        "genera", "crea", "dame", "escenario", "simulacro",
        "plantea", "simula", "ponme a prueba", "quiero practicar"
    ]
    if any(k in message_lower for k in scenario_keywords):
        return "new_scenario"

    return "general_query"


def build_conversation_context(history: list) -> str:
    """Construye contexto conversacional (últimos 5 turnos)"""
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
        clean = re.sub(r"\*\*.*?\*\*", "", content)
        clean = re.sub(r"---.*?---", "", clean, flags=re.DOTALL)
        if len(clean) > 200:
            clean = clean[:200] + "..."
        context += f"{i}. {role}: {clean}\n\n"
    return context


# ============================================================================
# FORMATEO DE RESPUESTAS
# ============================================================================

def format_formador_response(result: dict) -> str:
    response = "**🎓 AGENTE FORMADOR**\n\n"
    response += result.get("answer", "")

    # Externo
    if result.get("source") == "external":
        if result.get("disclaimer"):
            response += f"\n\n{result['disclaimer']}"
        if result.get("external_sources"):
            response += "\n\n**🌐 Fuentes externas:**\n"
            for i, source in enumerate(result["external_sources"][:3], 1):
                response += f"\n{i}. [{source.get('title','Sin título')}]({source.get('url','')})"

    # Interno
    elif result.get("sources"):
        response += "\n\n---\n**📚 Fuentes consultadas (DTF-13):**\n"
        for i, source in enumerate(result["sources"][:3], 1):
            response += f"\n{i}. {source.get('filename','')} (Página {source.get('page','')})"

    # Pregunta de confirmación
    if result.get("should_ask_user"):
        response += f"\n\n{result.get('question_for_user','')}"

    return response


def format_simulador_response(result: dict, intention: str, scenario_state: dict) -> tuple:
    response = "**🎭 AGENTE SIMULADOR**\n\n"

    if intention == "new_scenario":
        text = result.get("scenario") or result.get("answer") or result.get("response") or "No se pudo procesar la solicitud."
        response += text
        new_state = {"text": text, "active": True}
        response += "\n\n---\n💡 **¿Qué decisión tomarías en esta situación?**\n"
        response += "*Escribe tu respuesta en el siguiente mensaje para que pueda evaluarla.*"
        return response, new_state

    if intention == "scenario_response" and scenario_state.get("active"):
        text = result.get("evaluation") or result.get("answer") or result.get("response") or "No se pudo procesar la solicitud."
        response += "**⚖️ EVALUACIÓN DE TU DECISIÓN**\n\n"
        response += text
        new_state = {"text": None, "active": False}
        response += "\n\n---\n✅ *Evaluación completada. Puedes solicitar un nuevo escenario cuando quieras.*"
        return response, new_state

    text = result.get("answer") or result.get("response") or result.get("scenario") or result.get("evaluation") or "No se pudo procesar la solicitud."
    response += text
    return response, {"text": None, "active": False}


# ============================================================================
# LÓGICA DE PROCESAMIENTO
# ============================================================================

def _is_yes(msg: str) -> bool:
    return bool(re.match(r"^(s[ií]|sí|si|vale|ok|okay|de acuerdo|adelante|hazlo|busca|buscar|procede)\b", msg.strip().lower()))

def _is_no(msg: str) -> bool:
    return bool(re.match(r"^(no|nah|mejor no|negativo|cancela|cancelar)\b", msg.strip().lower()))


def process_message(
    message: str,
    mode: str,
    history: list,
    scenario_state: dict,
    allow_external: bool,
    formador_pending_state: dict,
    simulador_pending_state: dict
):
    """Procesa el mensaje del usuario según el modo seleccionado."""

    # Convertir historial de Gradio a formato interno
    internal_history = [m for m in history if isinstance(m, dict) and "role" in m and "content" in m]

    try:
        # 0) Confirmación pendiente SIMULADOR
        if simulador_pending_state.get("pending"):
            if _is_yes(message) and simulador_pending_state.get("topic"):
                topic = simulador_pending_state["topic"]
                sim_result = simulador.create_scenario(topic=topic, allow_external=True)
                simulador_pending_state = {"pending": False, "topic": None}
                response, scenario_state = format_simulador_response(sim_result, "new_scenario", scenario_state)

            elif _is_no(message):
                simulador_pending_state = {"pending": False, "topic": None}
                response, scenario_state = format_simulador_response(
                    {"scenario": "De acuerdo. No usaré búsqueda web. Puedes reformular el tema o activar la opción para permitir enriquecimiento externo."},
                    "general_query",
                    scenario_state
                )
            else:
                response, scenario_state = format_simulador_response(
                    {
                        "scenario": "Necesito una confirmación explícita.",
                        "should_ask_user": True,
                        "question_for_user": "¿Deseas que enriquezca el contexto con una búsqueda en la web? Responde 'sí' o 'no'."
                    },
                    "general_query",
                    scenario_state
                )

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, "", scenario_state, allow_external, formador_pending_state, simulador_pending_state

        # 1) Confirmación pendiente FORMADOR
        if formador_pending_state.get("pending"):
            if _is_yes(message) and formador_pending_state.get("question"):
                q = formador_pending_state["question"]
                rag_result = formador.rag.query(question=q, allow_external=True)
                result = {
                    "answer": rag_result.get("answer",""),
                    "source": rag_result.get("source","internal"),
                    "sources": rag_result.get("sources", []),
                    "external_sources": rag_result.get("external_sources"),
                    "disclaimer": rag_result.get("disclaimer"),
                    "should_ask_user": False
                }
                formador_pending_state = {"pending": False, "question": None}
                response = format_formador_response(result)

            elif _is_no(message):
                formador_pending_state = {"pending": False, "question": None}
                response = format_formador_response({
                    "answer": "De acuerdo. No haré búsqueda externa. Si quieres, reformula la pregunta o activa la opción para permitirla.",
                    "source": "none"
                })
            else:
                response = format_formador_response({
                    "answer": "Necesito una confirmación explícita para continuar.",
                    "source": "none",
                    "should_ask_user": True,
                    "question_for_user": "¿Deseas que busque en fuentes externas? Responde 'sí' o 'no'."
                })

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, "", {"text": None, "active": False}, allow_external, formador_pending_state, simulador_pending_state

        # 2) Intención + contexto
        intention = detect_user_intention(message, scenario_state.get("active", False))
        context = build_conversation_context(internal_history)

        # 3) Enrutamiento
        if mode == "🎯 Automático (Director decide)":
            # Si hay un escenario activo, el siguiente mensaje del usuario debe evaluarse SIEMPRE
            # con el Simulador (evita que el Director re-enrute y rompa el flujo).
            if intention == "scenario_response" and scenario_state.get("active"):
                sim_result = simulador.evaluate_decision(user_decision=message)
                response, scenario_state = format_simulador_response(sim_result, intention, scenario_state)

            else:
                classification = director.route(message)
                agent_used = classification.get("agent", "formador")

                if agent_used == "simulador":
                    allow_external_mode = None if allow_external else False
                    sim_result = simulador.create_scenario(topic=message, allow_external=allow_external_mode)

                    if sim_result.get("should_ask_user"):
                        simulador_pending_state = {"pending": True, "topic": message}
                        # Mostrar pregunta de confirmación sin activar escenario
                        response, scenario_state = format_simulador_response(sim_result, "general_query", scenario_state)
                    else:
                        response, scenario_state = format_simulador_response(sim_result, "new_scenario", scenario_state)

                else:
                    allow_external_mode = None if allow_external else False
                    rag_result = formador.rag.query(question=message, allow_external=allow_external_mode)

                    result = {
                        "answer": rag_result.get("answer",""),
                        "source": rag_result.get("source","internal"),
                        "sources": rag_result.get("sources", []),
                        "external_sources": rag_result.get("external_sources"),
                        "disclaimer": rag_result.get("disclaimer"),
                        "should_ask_user": rag_result.get("should_ask_user", False),
                        "question_for_user": rag_result.get("question_for_user")
                    }

                    if result.get("should_ask_user"):
                        formador_pending_state = {"pending": True, "question": message}

                    response = format_formador_response(result)
                    scenario_state = {"text": None, "active": False}
        elif mode == "🎓 Formador (Explicaciones)":
            allow_external_mode = None if allow_external else False
            rag_result = formador.rag.query(question=message, allow_external=allow_external_mode)

            result = {
                "answer": rag_result.get("answer",""),
                "source": rag_result.get("source","internal"),
                "sources": rag_result.get("sources", []),
                "external_sources": rag_result.get("external_sources"),
                "disclaimer": rag_result.get("disclaimer"),
                "should_ask_user": rag_result.get("should_ask_user", False),
                "question_for_user": rag_result.get("question_for_user")
            }

            if result.get("should_ask_user"):
                formador_pending_state = {"pending": True, "question": message}

            response = format_formador_response(result)
            scenario_state = {"text": None, "active": False}

        else:  # 🎭 Simulador (Escenarios)
            if intention == "scenario_response" and scenario_state.get("active"):
                sim_result = simulador.evaluate_decision(user_decision=message)
                response, scenario_state = format_simulador_response(sim_result, intention, scenario_state)

            elif intention == "new_scenario":
                allow_external_mode = None if allow_external else False
                sim_result = simulador.create_scenario(topic=message, allow_external=allow_external_mode)

                if sim_result.get("should_ask_user"):
                    simulador_pending_state = {"pending": True, "topic": message}
                    response, scenario_state = format_simulador_response(sim_result, "general_query", scenario_state)
                else:
                    response, scenario_state = format_simulador_response(sim_result, "new_scenario", scenario_state)

            else:
                # Fallback: tratar como solicitud de escenario genérica
                allow_external_mode = None if allow_external else False
                sim_result = simulador.create_scenario(topic=message, allow_external=allow_external_mode)

                if sim_result.get("should_ask_user"):
                    simulador_pending_state = {"pending": True, "topic": message}
                    response, scenario_state = format_simulador_response(sim_result, "general_query", scenario_state)
                else:
                    response, scenario_state = format_simulador_response(sim_result, "new_scenario", scenario_state)

        # 4) Persistir historial
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

    except Exception as e:
        import traceback
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Error: {e}\n\n{traceback.format_exc()}"})
        scenario_state = {"text": None, "active": False}

    return history, "", scenario_state, allow_external, formador_pending_state, simulador_pending_state


def clear_chat():
    return [], "", {"text": None, "active": False}, False, {"pending": False, "question": None}, {"pending": False, "topic": None}

# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

custom_css = """
.gradio-container { font-family: Arial, sans-serif; }
"""

with gr.Blocks(title="SARFIRE-RAG v2.0") as demo:
    scenario_state = gr.State({"text": None, "active": False})
    formador_pending_state = gr.State({"pending": False, "question": None})
    simulador_pending_state = gr.State({"pending": False, "topic": None})

    gr.HTML("""
        <div class="header">
            <h1>🔥 SARFIRE-RAG v2.0</h1>
            <p>Sistema Multi-Agente con enriquecimiento web bajo confirmación</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            mode_selector = gr.Radio(
                choices=[
                    "🎯 Automático (Director decide)",
                    "🎓 Formador (Explicaciones)",
                    "🎭 Simulador (Escenarios)"
                ],
                value="🎯 Automático (Director decide)",
                label="Modo de Operación"
            )

            external_checkbox = gr.Checkbox(
                label="🌐 Permitir enriquecimiento con web (bajo confirmación)",
                value=False
            )

            chatbot = gr.Chatbot(value=[], label="Conversación", height=450)

            with gr.Row():
                msg = gr.Textbox(placeholder="Escribe tu consulta...", label="Tu mensaje", scale=4)
                submit_btn = gr.Button("Enviar", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Limpiar conversación", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("""
### 📌 Comportamiento esperado

- **Simulador**
  - Relevancia >= umbral: crea escenario con DTF-13.
  - Relevancia < umbral:
    - Checkbox OFF: avisa y crea escenario SOLO con DTF-13.
    - Checkbox ON: pregunta si quieres web; si respondes "sí", enriquece y crea escenario (base DTF-13).

- **Formador**
  - Mismo patrón: pregunta confirmación antes de ir a web.
""")

    submit_btn.click(
        fn=process_message,
        inputs=[msg, mode_selector, chatbot, scenario_state, external_checkbox, formador_pending_state, simulador_pending_state],
        outputs=[chatbot, msg, scenario_state, external_checkbox, formador_pending_state, simulador_pending_state]
    )

    msg.submit(
        fn=process_message,
        inputs=[msg, mode_selector, chatbot, scenario_state, external_checkbox, formador_pending_state, simulador_pending_state],
        outputs=[chatbot, msg, scenario_state, external_checkbox, formador_pending_state, simulador_pending_state]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, msg, scenario_state, external_checkbox, formador_pending_state, simulador_pending_state]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=custom_css)
