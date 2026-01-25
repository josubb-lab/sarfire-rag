#!/usr/bin/env python3
"""
SARFIRE-RAG - Interfaz Gradio (MVP estable)

- Selector de agente: Director / Formador / Simulador
- Política de búsqueda externa: Nunca / Preguntar / Siempre
- Flujo Simulador:
    - Crea escenario (si relevancia baja y política=Preguntar, solicita confirmación)
    - Evalúa respuesta del usuario sobre escenario activo
    - Modo estricto: puede bloquear generación si no hay base suficiente (sin "inventar")
"""

from __future__ import annotations

import os
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_TELEMETRY"] = "FALSE"

import gradio as gr
import sys
from typing import Any, Dict, List, Optional
sys.path.append("src")

from rag import EmbeddingsGenerator, VectorStore, RAGPipeline
from agents import FormadorAgent, SimuladorAgent, DirectorAgent


def initialize_system() -> Dict[str, Any]:
    print("🔥 Inicializando SARFIRE-RAG (MVP)...")

    embeddings_gen = EmbeddingsGenerator(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(
        persist_directory="data/processed/chromadb",
        collection_name="sarfire_docs",
    )

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

    formador = FormadorAgent(rag_pipeline=rag)
    simulador = SimuladorAgent(rag_pipeline=rag)
    director = DirectorAgent()

    return {"rag": rag, "formador": formador, "simulador": simulador, "director": director}


SYS = initialize_system()


def _director_analyze(director: Any, message: str) -> Dict[str, Any]:
    """Intenta usar el método de análisis disponible en DirectorAgent.
    Compatibilidad: analyze_query / analyze_request / route / decide / recommend_agent.
    """
    for name in ("analyze_query", "analyze_request", "analyze", "route_query", "route", "decide", "recommend_agent"):
        fn = getattr(director, name, None)
        if callable(fn):
            try:
                out = fn(message)
                if isinstance(out, dict):
                    return out
            except TypeError:
                # algunos métodos pueden requerir firma distinta; ignoramos y probamos el siguiente
                pass
            except Exception:
                # si el método falla internamente, seguimos con fallback
                pass

    # Fallback heurístico (MVP)
    m = message.lower()
    if any(k in m for k in ("simulacro", "escenario", "juego de rol", "roleplay", "intervención")):
        return {"recommended_agent": "SIMULADOR", "confidence": "media", "reason": "Fallback heurístico: solicitud de simulacro/escenario."}
    return {"recommended_agent": "FORMADOR", "confidence": "media", "reason": "Fallback heurístico: consulta conceptual/informativa."}


def _formador_answer(formador: Any, rag: Any, question: str, allow_external: Optional[bool]) -> Dict[str, Any]:
    """Compatibilidad para FormadorAgent.
    Intenta métodos comunes; si no existen, cae a rag.query().
    Retorna dict con claves: answer, disclaimer, should_ask_user, question_for_user.
    """
    # Métodos más probables
    candidates = [
        "answer_question",
        "answer",
        "respond",
        "handle_question",
        "handle_query",
        "run",
        "query",
    ]
    for name in candidates:
        fn = getattr(formador, name, None)
        if callable(fn):
            try:
                out = fn(question=question, allow_external=allow_external)  # type: ignore
            except TypeError:
                try:
                    out = fn(question, allow_external)  # type: ignore
                except TypeError:
                    try:
                        out = fn(question)  # type: ignore
                    except Exception:
                        continue
            except Exception:
                continue

            if isinstance(out, dict):
                return out
            if isinstance(out, str):
                return {"answer": out}

    # Fallback directo al RAG
    try:
        r = rag.query(question=question, allow_external=allow_external)
        if isinstance(r, dict):
            # normalizar a clave 'answer'
            if "answer" in r:
                return r
            if "response" in r:
                return {"answer": r.get("response")}
            # si devuelve otro esquema, devolvemos algo razonable
            return {"answer": r.get("text") or str(r)}
        return {"answer": str(r)}
    except Exception as e:
        return {"answer": f"Error interno al responder: {e}"}




def _external_policy_to_allow_external(policy: str) -> Optional[bool]:
    # mapping:
    # - Nunca -> False
    # - Preguntar -> None
    # - Siempre -> True
    if policy == "Nunca":
        return False
    if policy == "Siempre":
        return True
    return None


def _append_turn(history: List[tuple[str, str]], user_text: str, assistant_text: str) -> List[tuple[str, str]]:
    history = history or []
    history.append((user_text, assistant_text))
    return history


def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return ""
    lines = ["", "---", "**📚 Evidencias (DTF-13):**"]
    for i, ev in enumerate(evidence[:3], 1):
        fn = ev.get("filename") or ""
        pg = ev.get("page")
        sn = (ev.get("snippet") or "").strip()
        if pg is not None:
            lines.append(f"{i}. {fn} (p. {pg}) — {sn}")
        else:
            lines.append(f"{i}. {fn} — {sn}")
    return "\n".join(lines)


def format_sources(result: Dict[str, Any], max_sources: int = 3) -> str:
    sources = result.get("sources") or []
    if not sources:
        return ""
    lines = ["", "---", f"Fuentes (top {min(max_sources, len(sources))}):"]
    for src in sources[:max_sources]:
        fn = src.get("filename") or "N/A"
        pg = src.get("page")
        score = src.get("hybrid_score")
        if score is None:
            score = src.get("similarity")
        if pg is not None:
            lines.append(f"- {fn} (p. {pg}) — score: {score}")
        else:
            lines.append(f"- {fn} — score: {score}")
    return "\n".join(lines)


def _simulador_create(simulador: Any, topic: str, allow_external: Optional[bool]) -> Dict[str, Any]:
    """Compatibilidad para SimuladorAgent: create_scenario / create / generate / run.
    Retorna dict esperado por la app.
    """
    candidates = ["create_scenario", "create", "generate_scenario", "generate", "run", "query"]
    last_err: Optional[Exception] = None
    for name in candidates:
        fn = getattr(simulador, name, None)
        if callable(fn):
            try:
                out = fn(topic=topic, allow_external=allow_external)  # type: ignore
            except TypeError:
                try:
                    out = fn(topic, allow_external)  # type: ignore
                except TypeError:
                    try:
                        out = fn(topic)  # type: ignore
                    except Exception as e:
                        last_err = e
                        continue
            except Exception as e:
                last_err = e
                continue

            if isinstance(out, dict):
                return out
            if isinstance(out, str):
                return {"scenario": out, "agent": "SIMULADOR", "topic": topic, "source": "unknown"}

    return {
        "scenario": f"Error interno al generar escenario: {last_err}",
        "agent": "SIMULADOR",
        "topic": topic,
        "source": "none",
        "blocked": True,
    }


def _simulador_evaluate(simulador: Any, user_decision: str) -> Dict[str, Any]:
    """Compatibilidad para evaluación: evaluate_decision / evaluate / grade / assess.
    Devuelve dict con clave 'evaluation'.
    """
    candidates = ["evaluate_decision", "evaluate", "grade", "assess", "run_evaluation"]
    last_err: Optional[Exception] = None
    for name in candidates:
        fn = getattr(simulador, name, None)
        if callable(fn):
            try:
                out = fn(user_decision=user_decision)  # type: ignore
            except TypeError:
                try:
                    out = fn(user_decision)  # type: ignore
                except Exception as e:
                    last_err = e
                    continue
            except Exception as e:
                last_err = e
                continue

            if isinstance(out, dict):
                return out
            if isinstance(out, str):
                return {"evaluation": out}

    return {"evaluation": f"Error interno al evaluar: {last_err}"}


def process_message(
    message: str,
    history: List[tuple[str, str]],
    agent_choice: str,
    external_policy: str,
    state: Dict[str, Any],
) -> tuple:
    message = (message or "").strip()
    if not message:
        return history, "", state

    rag = SYS["rag"]
    formador: Any = SYS["formador"]
    simulador: Any = SYS["simulador"]
    director: Any = SYS["director"]

    allow_external = _external_policy_to_allow_external(external_policy)

    # Estado por defecto
    state = state or {}
    state.setdefault("pending", {"active": False, "kind": None, "payload": None})
    state.setdefault("scenario", {"active": False})

    # 1) Confirmación pendiente (sí/no)
    if state["pending"]["active"]:
        answer = message.lower()
        yes = answer in {"sí", "si", "s", "yes", "y"}
        no = answer in {"no", "n"}

        if not (yes or no):
            history = _append_turn(history, message, "Responde 'sí' o 'no'.")
            return history, "", state

        kind = state["pending"]["kind"]
        payload = state["pending"]["payload"]
        state["pending"] = {"active": False, "kind": None, "payload": None}

        if kind == "simulador_web":
            sim_result = _simulador_create(simulador, topic=payload, allow_external=True if yes else False)
            if sim_result.get("blocked"):
                response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "No se pudo generar el escenario.")
                response += _format_evidence(sim_result.get("evidence") or [])
                state["scenario"] = {"active": False}
            else:
                response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "")
                response += _format_evidence(sim_result.get("evidence") or [])
                response += "\n\n---\n💡 **¿Qué decisión tomarías en esta situación?**\n*Escribe tu respuesta para que pueda evaluarla.*"
                state["scenario"] = {"active": True}

            history = _append_turn(history, message, response)
            return history, "", state

        if kind == "formador_web":
            # Formador: si el usuario acepta, forzamos allow_external=True, si no, False
            result = _formador_answer(formador, rag, question=payload, allow_external=True if yes else False)
            response = "**🎓 AGENTE FORMADOR**\n\n" + (result.get("answer") or "")
            if result.get("disclaimer"):
                response += "\n\n" + result["disclaimer"]
            response += format_sources(result)
            history = _append_turn(history, message, response)
            return history, "", state

    # 2) Si hay escenario activo: evaluar (independiente del agente seleccionado)
    if state.get("scenario", {}).get("active") and agent_choice in {"Director", "Simulador"}:
        eval_result = _simulador_evaluate(simulador, user_decision=message)
        response = "**⚖️ EVALUACIÓN (SIMULADOR)**\n\n" + (eval_result.get("evaluation") or "")
        history = _append_turn(history, message, response)
        # Mantener escenario activo para iterar (si quieres cerrarlo, pon active False)
        return history, "", state

    # 3) Ruteo por agente
    if agent_choice == "Director":
        analysis = _director_analyze(director, message)
        target = analysis.get("recommended_agent", "FORMADOR")
        if target == "SIMULADOR":
            sim_result = _simulador_create(simulador, topic=message, allow_external=allow_external)
            if sim_result.get("should_ask_user"):
                state["pending"] = {"active": True, "kind": "simulador_web", "payload": message}
                response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "")
                response += "\n\n" + sim_result.get("question_for_user", "¿Deseas que busque en fuentes externas? Responde 'sí' o 'no'.")
                response += _format_evidence(sim_result.get("evidence") or [])
                history = _append_turn(history, message, response)
                return history, "", state

            if sim_result.get("blocked"):
                response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "No se pudo generar el escenario.")
                response += _format_evidence(sim_result.get("evidence") or [])
                state["scenario"] = {"active": False}
                history = _append_turn(history, message, response)
                return history, "", state

            response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "")
            response += _format_evidence(sim_result.get("evidence") or [])
            response += "\n\n---\n💡 **¿Qué decisión tomarías en esta situación?**\n*Escribe tu respuesta para que pueda evaluarla.*"
            state["scenario"] = {"active": True}
            history = _append_turn(history, message, response)
            return history, "", state

        # FORMADOR por defecto
        result = _formador_answer(formador, rag, question=message, allow_external=allow_external)
        if result.get("should_ask_user"):
            state["pending"] = {"active": True, "kind": "formador_web", "payload": message}
            response = "**🎓 AGENTE FORMADOR**\n\n" + (result.get("answer") or "")
            response += "\n\n" + result.get("question_for_user", "¿Deseas que busque en fuentes externas? Responde 'sí' o 'no'.")
            response += format_sources(result)
            history = _append_turn(history, message, response)
            return history, "", state

        response = "**🎓 AGENTE FORMADOR**\n\n" + (result.get("answer") or "")
        if result.get("disclaimer"):
            response += "\n\n" + result["disclaimer"]
        response += format_sources(result)
        history = _append_turn(history, message, response)
        return history, "", state

    if agent_choice == "Simulador":
        sim_result = _simulador_create(simulador, topic=message, allow_external=allow_external)
        if sim_result.get("should_ask_user"):
            state["pending"] = {"active": True, "kind": "simulador_web", "payload": message}
            response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "")
            response += "\n\n" + sim_result.get("question_for_user", "¿Deseas que busque en fuentes externas? Responde 'sí' o 'no'.")
            response += _format_evidence(sim_result.get("evidence") or [])
            history = _append_turn(history, message, response)
            return history, "", state

        if sim_result.get("blocked"):
            response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "No se pudo generar el escenario.")
            response += _format_evidence(sim_result.get("evidence") or [])
            state["scenario"] = {"active": False}
            history = _append_turn(history, message, response)
            return history, "", state

        response = "**🎭 AGENTE SIMULADOR**\n\n" + (sim_result.get("scenario") or "")
        response += _format_evidence(sim_result.get("evidence") or [])
        response += "\n\n---\n💡 **¿Qué decisión tomarías en esta situación?**\n*Escribe tu respuesta para que pueda evaluarla.*"
        state["scenario"] = {"active": True}
        history = _append_turn(history, message, response)
        return history, "", state

    # Formador
    result = _formador_answer(formador, rag, question=message, allow_external=allow_external)
    if result.get("should_ask_user"):
        state["pending"] = {"active": True, "kind": "formador_web", "payload": message}
        response = "**🎓 AGENTE FORMADOR**\n\n" + (result.get("answer") or "")
        response += "\n\n" + result.get("question_for_user", "¿Deseas que busque en fuentes externas? Responde 'sí' o 'no'.")
        response += format_sources(result)
        history = _append_turn(history, message, response)
        return history, "", state

    response = "**🎓 AGENTE FORMADOR**\n\n" + (result.get("answer") or "")
    if result.get("disclaimer"):
        response += "\n\n" + result["disclaimer"]
    response += format_sources(result)
    history = _append_turn(history, message, response)
    return history, "", state


with gr.Blocks(title="SARFIRE-RAG (MVP)") as demo:
    gr.Markdown("# SARFIRE-RAG (MVP)\nInterfaz estable para Director / Formador / Simulador con RAG + búsqueda externa opcional.")

    with gr.Row():
        agent_choice = gr.Dropdown(choices=["Director", "Formador", "Simulador"], value="Director", label="Agente")
        external_policy = gr.Radio(choices=["Nunca", "Preguntar", "Siempre"], value="Preguntar", label="Búsqueda externa")

    chatbot = gr.Chatbot(label="Conversación")
    msg = gr.Textbox(label="Mensaje", placeholder="Escribe aquí…", lines=2)
    state = gr.State({})

    msg.submit(process_message, inputs=[msg, chatbot, agent_choice, external_policy, state], outputs=[chatbot, msg, state])

demo.launch()
