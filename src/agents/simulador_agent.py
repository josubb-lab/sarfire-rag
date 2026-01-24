#!/usr/bin/env python3
"""
Agente SIMULADOR para SARFIRE-RAG.

- Genera escenarios operativos basados en el DTF-13 (RAG interno).
- Evalúa decisiones del usuario contra el DTF-13.
- Fallback externo (opcional) SOLO para enriquecer el contexto cuando la relevancia interna es baja.

Contrato (producto):
- Si relevancia >= umbral: crea escenario con DTF-13.
- Si relevancia < umbral:
    - allow_external is None  -> NO busca aún: pregunta al usuario si desea usar web.
    - allow_external is False -> avisa de baja concordancia y crea escenario SOLO con DTF-13.
    - allow_external is True  -> usa web SOLO para enriquecer contexto y crea escenario basándose en DTF-13.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal, Any

import google.generativeai as genai


class SimuladorAgent:
    """Agente especializado en simulación de escenarios operativos."""

    def __init__(self, rag_pipeline: Any):
        """
        Args:
            rag_pipeline: Instancia de RAGPipeline configurada.
        """
        self.rag = rag_pipeline
        self.agent_name = "Agente Simulador"
        self.current_scenario: Optional[Dict[str, Any]] = None

        # Modo estricto: evita generar escenarios "plausibles" sin base documental suficiente
        self.strict_evidence_mode: bool = True
        self.min_evidence_items: int = 2

    # ---------------------------------------------------------------------
    # Prompts
    # ---------------------------------------------------------------------
    def generate_scenario_prompt(
        self,
        topic: str,
        internal_context: str,
        external_context: str = "",
        external_mode: Literal["none", "enrich"] = "none",
    ) -> str:
        """
        Genera prompt para crear escenario.

        - external_mode="none": escenario basado solo en DTF-13.
        - external_mode="enrich": usa DTF-13 como base y añade contexto externo SOLO para enriquecer,
          indicando explícitamente que lo externo no es oficial.
        """
        if external_mode == "enrich":
            return f"""Eres un INSTRUCTOR DE SIMULACROS del Consorcio Provincial de Bomberos de Valencia.

Tu misión es crear un ESCENARIO OPERATIVO REALISTA para entrenar al personal.

CONTEXTO BASE (DTF-13 OFICIAL) - USA ESTO COMO REFERENCIA PRINCIPAL:
{internal_context}

CONTEXTO EXTERNO (NO OFICIAL) - ÚSALO SOLO PARA ENRIQUECER DETALLES, NO PARA SUSTITUIR EL DTF-13:
{external_context}

REGLAS:
- La base normativa/procedimental debe apoyarse en el DTF-13.
- Si un detalle proviene del contexto externo, indícalo como "información externa no oficial".
- No inventes información.
- Si el CONTEXTO DEL DTF-13 no contiene información suficiente para sustentar el escenario,
  responde EXACTAMENTE con el token: INSUFICIENTE_EVIDENCIA (sin texto adicional).
- Longitud: 150-250 palabras.
- Estructura:
  1) **SITUACIÓN INICIAL**
  2) **EVOLUCIÓN**
  3) **TU DECISIÓN** (3-4 opciones)
  4) **PREGUNTA** ("¿Qué harías tú?")
- Añade al final un aviso: "Nota: Se ha usado información externa para enriquecer el contexto. Validar con DTF-13."

TEMA DEL ESCENARIO:
{topic}

GENERA EL ESCENARIO:
"""
        return f"""Eres un INSTRUCTOR DE SIMULACROS del Consorcio Provincial de Bomberos de Valencia.

Tu misión es crear ESCENARIOS OPERATIVOS REALISTAS para entrenar al personal.

ESTRUCTURA DEL ESCENARIO:
1. **SITUACIÓN INICIAL**: Describe el incidente (ubicación, extensión, condiciones, recursos)
2. **EVOLUCIÓN**: Qué está pasando en este momento crítico
3. **TU DECISIÓN**: Plantea 3-4 opciones de actuación
4. **PREGUNTA**: "¿Qué harías tú?"

REGLAS:
- Basado en protocolos del DTF-13
- Si el CONTEXTO DEL DTF-13 no contiene información suficiente para sustentar el escenario,
  responde EXACTAMENTE con el token: INSUFICIENTE_EVIDENCIA (sin texto adicional).
- Escenario realista y detallado
- Opciones plausibles
- Incluye dilemas operativos reales
- Longitud: 150-250 palabras

CONTEXTO DEL MANUAL DTF-13:
{internal_context}

TEMA DEL ESCENARIO:
{topic}

GENERA EL ESCENARIO:
"""

    def generate_evaluation_prompt(self, scenario: str, user_decision: str, context: str) -> str:
        """Genera prompt para evaluar la decisión del usuario."""
        return f"""Eres un EVALUADOR EXPERTO de simulacros del Consorcio Provincial de Bomberos de Valencia.

ESTRUCTURA DE LA EVALUACIÓN:
1. **ANÁLISIS**: Implicaciones de la decisión
2. **ACIERTOS**: Aspectos correctos según el DTF-13
3. **RIESGOS/ERRORES**: Problemas que puede causar
4. **DECISIÓN ÓPTIMA**: Mejor actuación según protocolo
5. **LECCIONES**: 2-3 puntos clave

TONO: Constructivo y formativo

CONTEXTO DTF-13:
{context}

ESCENARIO:
{scenario}

DECISIÓN DEL USUARIO:
{user_decision}

EVALUACIÓN:
"""

    # ---------------------------------------------------------------------
    # Core
    # ---------------------------------------------------------------------
    def create_scenario(
        self,
        topic: Optional[str] = None,
        top_k: int = 5,
        allow_external: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Crea un escenario operativo.

        allow_external:
          - False: no usar web.
          - None : web permitida con confirmación (preguntar primero si relevancia baja).
          - True : usar web para enriquecer (solo si relevancia baja y hay searcher).
        """
        print(f"🎭 {self.agent_name} activado")

        if not topic:
            topic = "Genera un escenario operativo de incendio forestal con dilemas"

        print(f"🎬 Creando escenario: {topic[:60]}...")

        # Paso 1: recuperar chunks internos (DTF-13)
        retrieved_chunks = self.rag.retrieve(topic, top_k=top_k)
        if not retrieved_chunks:
            return {
                "scenario": "No se pudo generar el escenario por falta de información interna (DTF-13).",
                "agent": self.agent_name,
                "topic": topic,
                "source": "none",
            }

        # Evidencias (DTF-13): snippet corto + metadatos
        evidence: List[Dict[str, Any]] = []
        for c in retrieved_chunks[:5]:
            meta = c.get("metadata", {}) or {}
            snippet = (c.get("text") or "").strip().replace("\n", " ")
            evidence.append({
                "filename": meta.get("filename"),
                "page": meta.get("page_num"),
                "similarity": c.get("similarity", 0.0),
                "hybrid_score": c.get("hybrid_score"),
                "snippet": snippet[:220],
            })

# Paso 2: evaluar relevancia
        relevance = float(self.rag.assess_relevance(retrieved_chunks))
        threshold = float(getattr(self.rag, "relevance_threshold", 0.5))
        print(f"📊 Relevancia: {relevance:.3f} (umbral: {threshold:.3f})")

        low_relevance = relevance < threshold
        external_searcher = getattr(self.rag, "external_searcher", None)
        external_available = external_searcher is not None

        # Paso 3A: relevancia baja + modo confirmación
        if low_relevance and external_available and allow_external is None:
            print("⚠️ Baja relevancia - consulta externa disponible (pendiente de confirmación)")
            return {
                "scenario": (
                    "No hay suficiente concordancia en el DTF-13 para crear un escenario sólido con garantías."
                ),
                "agent": self.agent_name,
                "topic": topic,
                "source": "none",
                "should_ask_user": True,
                "question_for_user": (
                    "¿Deseas que realice una búsqueda en la web para enriquecer el contexto? "
                    "(La información externa NO es oficial y debe validarse con DTF-13). "
                    "Responde 'sí' o 'no'."
                ),
                "relevance_score": relevance,
            "evidence": evidence,
                "evidence": evidence,
                "internal_sources": [
                    {
                        "filename": c.get("metadata", {}).get("filename"),
                        "page": c.get("metadata", {}).get("page_num"),
                        "similarity": c.get("similarity", 0.0),
                    }
                    for c in retrieved_chunks[:3]
                ],
            }

        # Paso 3B: construir contexto interno
                # Modo estricto: con relevancia baja, NO generar escenario sin enriquecimiento web o reformulación.
        if self.strict_evidence_mode and low_relevance:
            if allow_external is False or not external_available:
                msg = (
                    "No dispongo de base documental suficiente en el DTF-13 para generar un simulacro fiable con garantías.\n\n"
                    "Opciones:\n"
                    "1) Reformula el tema con más detalle (ubicación, fase del incendio, recursos, objetivo).\n"
                    "2) Permite enriquecimiento web (con confirmación) para ampliar el contexto.\n"
                )
                return {
                    "scenario": msg,
                    "agent": self.agent_name,
                    "topic": topic,
                    "source": "none",
                    "blocked": True,
                    "relevance_score": relevance,
            "evidence": evidence,
                    "evidence": evidence,
                }
            # allow_external=True con searcher disponible: intentaremos enriquecer a continuación.

        internal_context = self.rag.format_context(retrieved_chunks)

        # Paso 3C: enriquecimiento externo (si procede)
        external_context = ""
        external_sources: Optional[List[Dict[str, Any]]] = None
        external_error: Optional[str] = None
        used_external = False

        if low_relevance and external_available and allow_external is True:
            print("🌐 Buscando en fuentes externas para ENRIQUECER contexto...")
            external_results = external_searcher.search(topic, max_results=3)

            if external_results.get("success") and external_results.get("results"):
                used_external = True
                external_sources = external_results["results"]

                parts: List[str] = []
                for i, r in enumerate(external_sources, start=1):
                    title = r.get("title", "Sin título")
                    url = r.get("url", "")
                    content = str(r.get("content", ""))[:500]
                    parts.append(
                        f"[FUENTE EXTERNA {i}]\n"
                        f"Título: {title}\n"
                        f"URL: {url}\n"
                        f"Contenido: {content}...\n"
                    )
                external_context = "[FUENTES EXTERNAS]\n" + "\n---\n".join(parts)
            else:
                external_error = external_results.get("error") or "Búsqueda externa falló"
                print(f"❌ Búsqueda externa falló - continuando SOLO con DTF-13. Motivo: {external_error}")
                if self.strict_evidence_mode and low_relevance:
                    msg = (
                        "No dispongo de base documental suficiente en el DTF-13 para generar un simulacro fiable.\n\n"
                        "Se intentó enriquecer con web, pero falló.\n"
                        f"Motivo: {external_error}\n\n"
                        "Opciones: reformula la consulta o reintenta la búsqueda externa.\n"
                    )
                    return {
                        "scenario": msg,
                        "agent": self.agent_name,
                        "topic": topic,
                        "source": "none",
                        "blocked": True,
                        "relevance_score": relevance,
            "evidence": evidence,
                        "evidence": evidence,
                        "external_error": external_error,
                    }

                used_external = False

        # Paso 4: generar escenario con el prompt adecuado
        scenario_prompt = self.generate_scenario_prompt(
            topic=topic,
            internal_context=internal_context,
            external_context=external_context,
            external_mode="enrich" if used_external else "none",
        )

        # Advertencia si relevancia baja y sin web
        relevance_warning = ""
        if low_relevance and not used_external:
            relevance_warning = (
                "⚠️ Nota: La concordancia con DTF-13 ha sido baja para esta consulta. "
                "El escenario se ha construido con la información interna disponible; "
                "considera reformular el tema o permitir enriquecimiento web.\n\n"
            )
            # Si el usuario autorizó web (allow_external=True) pero la búsqueda falló, dejar constancia.
            if allow_external is True and external_available:
                if external_error:
                    relevance_warning += (
                        "⚠️ Enriquecimiento web: NO disponible (fallo en búsqueda externa). "
                        f"Motivo: {external_error}\n\n"
                    )
                else:
                    relevance_warning += (
                        "⚠️ Enriquecimiento web: NO disponible (sin resultados externos).\n\n"
                    )

        print("🤖 Generando escenario...")
        response = self.rag.model.generate_content(
            scenario_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
        )

        raw_text = (response.text or "").strip()
        if raw_text == "INSUFICIENTE_EVIDENCIA":
            msg = (
                "No dispongo de base documental suficiente en el DTF-13 para generar un simulacro fiel.\n\n"
                "Opciones: reformula el tema o permite enriquecimiento web."
            )
            return {
                "scenario": msg,
                "agent": self.agent_name,
                "topic": topic,
                "source": "none",
                "blocked": True,
                "relevance_score": relevance,
            "evidence": evidence,
                "evidence": evidence,
            }

        scenario_text = relevance_warning + raw_text


        # Guardar escenario actual
        self.current_scenario = {
            "text": scenario_text,
            "topic": topic,
            "source": "enriched" if used_external else "internal",
            "sources": retrieved_chunks[:3],
            "external_sources": external_sources,
            "evidence": evidence,
        }

        result: Dict[str, Any] = {
            "scenario": scenario_text,
            "agent": self.agent_name,
            "topic": topic,
            "source": "external_enriched" if used_external else "internal",
            "relevance_score": relevance,
            "evidence": evidence,
            "sources": [
                {
                    "filename": c.get("metadata", {}).get("filename"),
                    "page": c.get("metadata", {}).get("page_num"),
                }
                for c in retrieved_chunks[:3]
            ],
        }

        if used_external and external_sources:
            result["external_sources"] = external_sources
            result["disclaimer"] = (
                "⚠️ Se ha usado información externa para enriquecer el contexto. Validar con DTF-13."
            )

        return result

    def evaluate_decision(self, user_decision: str, top_k: int = 5) -> Dict[str, Any]:
        """Evalúa la decisión del usuario."""
        if not self.current_scenario:
            return {
                "evaluation": "No hay un escenario activo para evaluar.",
                "agent": self.agent_name,
            }

        print("⚖️ Evaluando tu decisión...")

        query = f"{self.current_scenario['topic']} {user_decision}"
        retrieved_chunks = self.rag.retrieve(query, top_k=top_k)
        context = self.rag.format_context(retrieved_chunks)

        eval_prompt = self.generate_evaluation_prompt(
            scenario=self.current_scenario["text"],
            user_decision=user_decision,
            context=context,
        )

        print("🤖 Analizando según DTF-13...")
        response = self.rag.model.generate_content(
            eval_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.4),
        )

        return {
            "evaluation": response.text or "",
            "scenario": self.current_scenario["text"],
            "user_decision": user_decision,
            "agent": self.agent_name,
        }

    # ---------------------------------------------------------------------
    # CLI helpers (opcionales)
    # ---------------------------------------------------------------------
    def print_scenario(self, result: Dict[str, Any]) -> None:
        """Imprime el escenario."""
        print("\n" + "=" * 70)
        print(f"🎭 {result.get('agent', '').upper()} - ESCENARIO OPERATIVO")
        print("=" * 70)
        print(result.get("scenario", ""))
        if result.get("disclaimer"):
            print("\n" + result["disclaimer"])
        print("\n" + "=" * 70)

    def print_evaluation(self, result: Dict[str, Any]) -> None:
        """Imprime la evaluación."""
        print("\n" + "=" * 70)
        print("⚖️ EVALUACIÓN DE TU DECISIÓN")
        print("=" * 70)
        print(f"\n🔍 TU DECISIÓN: {result.get('user_decision', '')}")
        print(f"\n📊 ANÁLISIS:\n{result.get('evaluation', '')}")
        print("\n" + "=" * 70)
