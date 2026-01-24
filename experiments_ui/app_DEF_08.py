#!/usr/bin/env python3
"""
SARFIRE-RAG - MVP Interface (Gradio)

Principios MVP:
- Estado explícito (simulacro activo + confirmación externa pendiente).
- Política externa consistente: never | ask | always.
- El Director solo enruta cuando NO hay un flujo activo (p.ej. evaluación de simulacro).
- Web:
  - Formador: fallback con disclaimer si no hay suficiente DTF-13.
  - Simulador: solo para enriquecer contexto; el escenario se basa en DTF-13.
"""

import gradio as gr
import sys
import re
from typing import Dict, Any, Optional

# Import path: el proyecto usa "rag" y "agents" bajo ./src
sys.path.append("src")

from rag import EmbeddingsGenerator, VectorStore, RAGPipeline
from agents import FormadorAgent, SimuladorAgent, DirectorAgent


# ============================================================================
# INIT
# ============================================================================
print("🔥 Inicializando SARFIRE-RAG (MVP)...")

embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
vector_store = VectorStore(
    persist_directory="data/processed/chromadb",
    collection_name="sarfire_docs",
)

# Defaults MVP (coherentes con híbrido + corpus técnico)
rag = RAGPipeline(
    vector_store=vector_store,
    embeddings_generator=embeddings_gen,
    model_name="gemini-2.0-flash",
    temperature=0.3,
    top_k=10,
    use_hybrid_search=True,
    enable_external_fallback=True,
    relevance_threshold=0.3,
)

# Los agentes pueden mantener su propio prompt/estilo, pero usamos el mismo RAG
formador = FormadorAgent(rag_pipeline=rag)
simulador = SimuladorAgent(rag_pipeline=rag)
director = DirectorAgent(model_name="gemini-2.0-flash")

print("✅ Sistema inicializado\n")


# ============================================================================
# HELPERS
# ============================================================================
def _policy_to_allow_external(policy: str) -> Optional[bool]:
    """
    Map UI policy to allow_external:
      - "never"  -> False
      - "ask"    -> None (ask user if low relevance)
      - "always" -> True
    """
    if policy == "never":
        return False
    if policy == "always":
        return True
    return None  # ask


def _is_yes(msg: str) -> bool:
    return bool(
        re.match(
            r"^(s[ií]|sí|si|vale|ok|okay|de acuerdo|adelante|hazlo|busca|buscar|procede)\b",
            msg.strip().lower(),
        )
    )


def _is_no(msg: str) -> bool:
    return bool(re.match(r"^(no|nah|mejor no|negativo|cancela|cancelar)\b", msg.strip().lower()))


def detect_user_intention(message: str, scenario_active: bool) -> str:
    """
    Prioridad: si hay simulacro activo, por defecto interpretamos la entrada como
    respuesta a evaluar, salvo petición explícita de nuevo escenario.
    """
    m = message.lower().strip()
    if scenario_active:
        explicit_new = re.search(
            r"\b(nuevo|otra|otro)\s+(escenario|simulacro|caso)\b|\b(genera|crea|plantea)\s+(otro|un\s+nuevo)\b",
            m,
        )
        return "new_scenario" if explicit_new else "scenario_response"

    scenario_keywords = [
        "genera",
        "crea",
        "dame",
        "escenario",
        "simulacro",
        "plantea",
        "simula",
        "ponme a prueba",
        "quiero practicar",
    ]
    return "new_scenario" if any(k in m for k in scenario_keywords) else "general_query"


# ============================================================================
# FORMATTERS
# ============================================================================
def format_formador_response(result: Dict[str, Any]) -> str:
    response = "**🎓 AGENTE FORMADOR**\n\n"
    response += result.get("answer", "")

    if result.get("source") == "external":
        disclaimer = result.get("disclaimer")
        if disclaimer:
            response += "\n\n" + disclaimer
        if result.get("external_sources"):
            response += "\n\n**🌐 Fuentes externas:**\n"
            for i, src in enumerate(result["external_sources"][:3], 1):
                title = src.get("title", "Sin título")
                url = src.get("url", "")
                response += f"\n{i}. [{title}]({url})"

    if result.get("source") in ("internal", "none") and result.get("sources"):
        response += "\n\n---\n**📚 Fuentes consultadas (DTF-13):**\n"
        for i, src in enumerate(result["sources"][:3], 1):
            response += f"\n{i}. {src.get('filename','')} (Página {src.get('page','')})"

    if result.get("should_ask_user"):
        question = result.get(
            "question_for_user",
            '¿Deseas que busque en fuentes externas? Responde "sí" o "no".',
        )
        response += "\n\n" + question

    return response


def format_simulador_scenario(result: Dict[str, Any]) -> str:
    response = "**🎭 AGENTE SIMULADOR**\n\n"
    response += result.get("scenario", "") or "No se pudo generar el escenario."

    disclaimer = result.get("disclaimer")
    if disclaimer:
        response += "\n\n" + disclaimer

    # Evidencias (DTF-13)
    evidence = result.get("evidence") or []
    if evidence:
        response += "\n\n---\n**📚 Evidencias (DTF-13):**\n"
        for i, ev in enumerate(evidence[:3], 1):
            fn = ev.get("filename") or ""
            pg = ev.get("page")
            sn = (ev.get("snippet") or "").strip()
            if pg is not None:
                response += f"\n{i}. {fn} (p. {pg}) — {sn}"
            else:
                response += f"\n{i}. {fn} — {sn}"

    response += "\n\n---\n💡 **¿Qué decisión tomarías en esta situación?**\n"
    response += "*Escribe tu respuesta en el siguiente mensaje para que pueda evaluarla.*"
    return response


def format_simulador_evaluation(result: Dict[str, Any]) -> str:
    response = "**🎭 AGENTE SIMULADOR**\n\n"
    response += "**⚖️ EVALUACIÓN DE TU DECISIÓN**\n\n"
    response += result.get("evaluation", "") or "No se pudo generar la evaluación."
    response += "\n\n---\n✅ *Evaluación completada. Puedes solicitar un nuevo escenario cuando quieras.*"
    return response


# ============================================================================
# CORE PROCESSOR (MVP STATE MACHINE)
# ============================================================================
def process_message(
    message: str,
    mode: str,
    history: list,
    state: dict,
    external_policy: str,
):
    """
    state structure:
      {
        "scenario": {"active": bool, "text": str|null},
        "pending": {"active": bool, "kind": "formador|simulador", "payload": str|null}
      }
    """
    message = (message or "").strip()
    if not message:
        return history, "", state, external_policy

    try:
        # Ensure state shape
        state.setdefault("scenario", {"active": False, "text": None})
        state.setdefault("pending", {"active": False, "kind": None, "payload": None})

        # 0) Pending external confirmation
        if state["pending"]["active"]:
            kind = state["pending"]["kind"]
            payload = state["pending"]["payload"]  # question/topic

            if _is_yes(message):
                if kind == "formador":
                    rag_result = rag.query(question=payload, allow_external=True)
                    response = format_formador_response(rag_result)
                    state["pending"] = {"active": False, "kind": None, "payload": None}
                    state["scenario"] = {"active": False, "text": None}
                else:  # simulador
                    sim_result = simulador.create_scenario(topic=payload, allow_external=True)
                    if sim_result.get("blocked"):
                        response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "No se pudo generar el escenario.")
                        evidence = sim_result.get("evidence") or []
                        if evidence:
                            response += "\n\n---\n**📚 Evidencias (DTF-13):**\n"
                            for i, ev in enumerate(evidence[:3], 1):
                                fn = ev.get("filename") or ""
                                pg = ev.get("page")
                                sn = (ev.get("snippet") or "").strip()
                                if pg is not None:
                                    response += f"\n{i}. {fn} (p. {pg}) — {sn}"
                                else:
                                    response += f"\n{i}. {fn} — {sn}"
                        state["scenario"] = {"active": False, "text": None}
                    else:
                        response = format_simulador_scenario(sim_result)
                        state["scenario"] = {"active": True, "text": sim_result.get("scenario")}
                    state["pending"] = {"active": False, "kind": None, "payload": None}

            elif _is_no(message):
                if kind == "formador":
                    rag_result = rag.query(question=payload, allow_external=False)
                    response = format_formador_response(rag_result)
                    state["pending"] = {"active": False, "kind": None, "payload": None}
                    state["scenario"] = {"active": False, "text": None}
                else:
                    sim_result = simulador.create_scenario(topic=payload, allow_external=False)
                    if sim_result.get("blocked"):
                        response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "No se pudo generar el escenario.")
                        evidence = sim_result.get("evidence") or []
                        if evidence:
                            response += "\n\n---\n**📚 Evidencias (DTF-13):**\n"
                            for i, ev in enumerate(evidence[:3], 1):
                                fn = ev.get("filename") or ""
                                pg = ev.get("page")
                                sn = (ev.get("snippet") or "").strip()
                                if pg is not None:
                                    response += f"\n{i}. {fn} (p. {pg}) — {sn}"
                                else:
                                    response += f"\n{i}. {fn} — {sn}"
                        state["scenario"] = {"active": False, "text": None}
                    else:
                        response = format_simulador_scenario(sim_result)
                        state["scenario"] = {"active": True, "text": sim_result.get("scenario")}
                    state["pending"] = {"active": False, "kind": None, "payload": None}
            else:
                response = (
                    "**ℹ️ Confirmación requerida**\n\n"
                    'Responde **"sí"** para continuar con búsqueda externa o **"no"** para seguir solo con DTF-13.'
                )

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, "", state, external_policy

        # 1) If scenario active, prioritize evaluation (regardless of Director)
        intention = detect_user_intention(message, state["scenario"]["active"])
        if state["scenario"]["active"] and intention == "scenario_response":
            eval_result = simulador.evaluate_decision(user_decision=message)
            response = format_simulador_evaluation(eval_result)
            state["scenario"] = {"active": False, "text": None}

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, "", state, external_policy

        # 2) Determine target agent (mode)
        if mode == "🎯 Automático (Director decide)":
            classification = director.route(message)
            target = classification.get("agent", "formador")
        elif mode == "🎓 Formador (Explicaciones)":
            target = "formador"
        else:
            target = "simulador"

        allow_external_mode = _policy_to_allow_external(external_policy)

        # 3) Execute
        if target == "simulador":
            sim_result = simulador.create_scenario(topic=message, allow_external=allow_external_mode)

            if sim_result.get("blocked"):
                response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "No se pudo generar el escenario.")
                evidence = sim_result.get("evidence") or []
                if evidence:
                    response += "\n\n---\n**📚 Evidencias (DTF-13):**\n"
                    for i, ev in enumerate(evidence[:3], 1):
                        fn = ev.get("filename") or ""
                        pg = ev.get("page")
                        sn = (ev.get("snippet") or "").strip()
                        if pg is not None:
                            response += f"\n{i}. {fn} (p. {pg}) — {sn}"
                        else:
                            response += f"\n{i}. {fn} — {sn}"
                state["scenario"] = {"active": False, "text": None}
            elif sim_result.get("should_ask_user"):
                # Ask and set pending; no activar scenario todavía
                ask = sim_result.get(
                    "question_for_user",
                    '¿Deseas enriquecer con web? Responde "sí" o "no".',
                )
                response = (
                    "**🎭 AGENTE SIMULADOR**\n\n"
                    "⚠️ No hay suficiente concordancia en DTF-13 para generar un escenario fiable.\n\n"
                    + ask
                )
                state["pending"] = {"active": True, "kind": "simulador", "payload": message}
                state["scenario"] = {"active": False, "text": None}
            else:
                response = format_simulador_scenario(sim_result)
                state["scenario"] = {"active": True, "text": sim_result.get("scenario")}

        else:
            rag_result = rag.query(question=message, allow_external=allow_external_mode)
            response = format_formador_response(rag_result)
            # If ask user, set pending
            if rag_result.get("should_ask_user"):
                state["pending"] = {"active": True, "kind": "formador", "payload": message}
            else:
                state["pending"] = {"active": False, "kind": None, "payload": None}
            state["scenario"] = {"active": False, "text": None}

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

    except Exception as e:
        import traceback

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Error: {e}\n\n{traceback.format_exc()}"})
        state = {
            "scenario": {"active": False, "text": None},
            "pending": {"active": False, "kind": None, "payload": None},
        }

    return history, "", state, external_policy


def clear_chat():
    return [], "", {"scenario": {"active": False, "text": None}, "pending": {"active": False, "kind": None, "payload": None}}, "ask"


# ============================================================================
# UI
# ============================================================================
custom_css = """
.gradio-container { font-family: Arial, sans-serif; }
"""

with gr.Blocks(title="SARFIRE-RAG MVP") as demo:
    state = gr.State({"scenario": {"active": False, "text": None}, "pending": {"active": False, "kind": None, "payload": None}})

    gr.HTML(
        """
        <div class="header">
            <h1>🔥 SARFIRE-RAG (MVP)</h1>
            <p>Director + Formador + Simulador con estado explícito y política web consistente</p>
        </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=3):
            mode_selector = gr.Radio(
                choices=[
                    "🎯 Automático (Director decide)",
                    "🎓 Formador (Explicaciones)",
                    "🎭 Simulador (Escenarios)",
                ],
                value="🎯 Automático (Director decide)",
                label="Modo de Operación",
            )

            external_policy = gr.Radio(
                choices=[
                    ("Nunca (solo DTF-13)", "never"),
                    ("Preguntar (si baja relevancia)", "ask"),
                    ("Siempre (enriquecer si baja relevancia)", "always"),
                ],
                value="ask",
                label="Política de búsqueda externa",
            )

            chatbot = gr.Chatbot(value=[], label="Conversación", height=480)

            with gr.Row():
                msg = gr.Textbox(placeholder="Escribe tu consulta...", label="Tu mensaje", scale=4)
                submit_btn = gr.Button("Enviar", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Limpiar conversación", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown(
                """
### MVP: reglas de flujo

- Si hay **simulacro activo**, el siguiente mensaje se interpreta como **decisión** y se evalúa (no se re-rutea por el Director).
- Si hay **confirmación pendiente** (web), el sistema espera **sí/no**.
- La web se usa:
  - **Formador**: para responder con disclaimer cuando no hay suficiente info interna.
  - **Simulador**: solo para **enriquecer** el contexto; la base es DTF-13.
"""
            )

    submit_btn.click(
        fn=process_message,
        inputs=[msg, mode_selector, chatbot, state, external_policy],
        outputs=[chatbot, msg, state, external_policy],
    )

    msg.submit(
        fn=process_message,
        inputs=[msg, mode_selector, chatbot, state, external_policy],
        outputs=[chatbot, msg, state, external_policy],
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, msg, state, external_policy],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=custom_css)
