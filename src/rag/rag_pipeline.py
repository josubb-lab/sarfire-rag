"""
RAG Pipeline para SARFIRE-RAG
Integra recuperación (ChromaDB) + generación (Gemini) + Fallback externo (Tavily)
"""
from __future__ import annotations

from typing import List, Dict, Optional
import os

from dotenv import load_dotenv
import google.generativeai as genai

from .hybrid_search import HybridSearch
from .external_search import ExternalSearcher


class RAGPipeline:
    """Pipeline RAG completo: Retrieve + Generate + External Fallback"""

    def __init__(
        self,
        vector_store,
        embeddings_generator,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        top_k: int = 10,
        use_hybrid_search: bool = True,
        enable_external_fallback: bool = True,
        relevance_threshold: float = 0.3,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        self.vector_store = vector_store
        self.embeddings_generator = embeddings_generator
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.use_hybrid_search = use_hybrid_search
        self.enable_external_fallback = enable_external_fallback
        self.relevance_threshold = relevance_threshold
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        if use_hybrid_search:
            self.hybrid_search = HybridSearch(vector_store, embeddings_generator)
        else:
            self.hybrid_search = None

        if enable_external_fallback:
            try:
                self.external_searcher = ExternalSearcher()
                print("✅ Fallback externo habilitado (Tavily)")
            except Exception as e:
                print(f"⚠️  Fallback externo deshabilitado: {e}")
                self.external_searcher = None
        else:
            self.external_searcher = None

        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY no encontrada. Asegúrate de tenerla en el archivo .env")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        print("✅ RAG Pipeline inicializado")
        print(f"   Modelo: {model_name}")
        print(f"   Temperature: {temperature}")
        print(f"   Top-K retrieval: {top_k}")
        print(f"   Hybrid: {use_hybrid_search} (sem={self.semantic_weight}, key={self.keyword_weight})")
        print(f"   Relevance threshold: {self.relevance_threshold}")
        print(f"   Fallback externo: {enable_external_fallback}")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        if top_k is None:
            top_k = self.top_k

        if self.use_hybrid_search and self.hybrid_search is not None:
            return self.hybrid_search.hybrid_search(
                query=query,
                top_k=top_k,
                semantic_weight=self.semantic_weight,
                keyword_weight=self.keyword_weight,
            )

        return self.vector_store.search_by_text(
            query_text=query,
            embeddings_generator=self.embeddings_generator,
            n_results=top_k,
        )

    def assess_relevance(self, retrieved_chunks: List[Dict]) -> float:
        if not retrieved_chunks:
            return 0.0

        scores: List[float] = []
        for c in retrieved_chunks:
            if "hybrid_score" in c and c["hybrid_score"] is not None:
                scores.append(float(c["hybrid_score"]))
            else:
                scores.append(float(c.get("similarity", 0.0)))

        return sum(scores) / len(scores)

    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            parts.append(
                f"[FUENTE {i}]\n"
                f"Documento: {meta.get('filename', 'N/A')}\n"
                f"Página: {meta.get('page_num', 'N/A')}\n"
                f"Contenido: {chunk.get('text', '')}\n"
            )
        return "\n---\n".join(parts)

    def format_external_context(self, external_results: Dict) -> str:
        if not external_results.get("success") or not external_results.get("results"):
            return "No se encontró información externa relevante."

        parts = []
        for i, r in enumerate(external_results["results"], 1):
            parts.append(
                f"[FUENTE EXTERNA {i}]\n"
                f"Título: {r.get('title', '')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Contenido: {str(r.get('content', ''))[:500]}...\n"
            )
        return "\n---\n".join(parts)

    def generate_prompt(self, query: str, context: str, is_external: bool = False) -> str:
        if is_external:
            return f"""Eres un asistente experto en emergencias forestales y protocolos de bomberos.

Estás consultando FUENTES EXTERNAS porque no hay información suficiente en la documentación interna.

IMPORTANTE:
- Esta información proviene de fuentes externas, NO de los manuales oficiales
- Sintetiza la información de manera clara y objetiva
- Indica que debe ser verificada con protocolos oficiales (DTF-13)
- NO inventes información que no esté en las fuentes

INFORMACIÓN DE FUENTES EXTERNAS:
{context}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:"""

        return f"""Eres un asistente experto en emergencias forestales y protocolos de bomberos.

Tu tarea es responder basándote ÚNICAMENTE en la información proporcionada en el contexto.

REGLAS IMPORTANTES:
1. Responde SOLO con información del contexto proporcionado
2. Si la información no está en el contexto, di claramente "No encuentro esa información en los manuales"
3. Cita las fuentes (documento y página) cuando respondas
4. Sé preciso y técnico, este es material para profesionales
5. Si hay procedimientos de seguridad, menciónalos SIEMPRE

CONTEXTO DE LOS MANUALES:
{context}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:"""

    def generate(self, query: str, context: str, is_external: bool = False) -> Dict:
        prompt = self.generate_prompt(query, context, is_external)
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=self.temperature),
        )
        return {"answer": response.text, "prompt_tokens": len(prompt.split())}

    def _llm_error_response(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        relevance: float,
    ) -> Dict:
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            raw_text = (top_chunk.get("text") or "").strip()
            extract = raw_text[:400]
            if raw_text and len(raw_text) > 400:
                extract += "..."
            answer = (
                "⚠️ No pude generar la respuesta con el modelo en este momento. "
                "Aquí tienes un extracto de los manuales recuperados:"
            )
            if extract:
                answer += f"\n\n{extract}"
            else:
                answer += "\n\n(No hay extractos disponibles; revisa las fuentes.)"

            return {
                "question": question,
                "answer": answer,
                "source": "internal",
                "sources": [
                    {
                        "filename": c.get("metadata", {}).get("filename"),
                        "page": c.get("metadata", {}).get("page_num"),
                        "similarity": c.get("similarity"),
                        "hybrid_score": c.get("hybrid_score"),
                        "text_preview": c.get("text", "")[:200],
                    }
                    for c in retrieved_chunks[:5]
                ],
                "relevance_score": relevance,
                "metadata": {
                    "chunks_retrieved": len(retrieved_chunks),
                    "model": self.model_name,
                    "temperature": self.temperature,
                },
            }

        return {
            "question": question,
            "answer": (
                "No encuentro esa información en los manuales. "
                "Si dispones de búsqueda externa, puedes activarla para intentar obtener más contexto."
            ),
            "source": "none",
            "relevance_score": relevance,
        }

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        allow_external: Optional[bool] = None,
    ) -> Dict:
        print("\n🔍 Buscando información relevante...")
        retrieved_chunks = self.retrieve(question, top_k)
        print(f"✅ Encontrados {len(retrieved_chunks)} chunks relevantes")

        relevance = self.assess_relevance(retrieved_chunks)
        print(f"📊 Relevancia promedio: {relevance:.3f} (umbral: {self.relevance_threshold})")

        external_available = self.external_searcher is not None
        should_use_external = relevance < self.relevance_threshold and external_available

        # Forzar SOLO DTF-13
        if allow_external is False or not should_use_external:
            if should_use_external:
                print("✅ Usando documentación interna (forzado)")
            else:
                print("✅ Usando documentación interna")

            context = self.format_context(retrieved_chunks)
            print(f"\n🤖 Generando respuesta con {self.model_name}...")
            try:
                generation_result = self.generate(question, context, is_external=False)
            except Exception as e:
                print(f"⚠️ Error en Gemini: {type(e).__name__}: {e}")
                return self._llm_error_response(question, retrieved_chunks, relevance)

            return {
                "question": question,
                "answer": generation_result["answer"],
                "source": "internal",
                "sources": [
                    {
                        "filename": c.get("metadata", {}).get("filename"),
                        "page": c.get("metadata", {}).get("page_num"),
                        "similarity": c.get("similarity"),
                        "hybrid_score": c.get("hybrid_score"),
                        "text_preview": c.get("text", "")[:200],
                    }
                    for c in retrieved_chunks
                ],
                "relevance_score": relevance,
                "metadata": {
                    "chunks_retrieved": len(retrieved_chunks),
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "use_hybrid_search": self.use_hybrid_search,
                    "semantic_weight": self.semantic_weight,
                    "keyword_weight": self.keyword_weight,
                },
            }

        # Relevancia baja y policy ask
        if allow_external is None:
            print("⚠️  Baja relevancia - consulta externa disponible")
            return {
                "question": question,
                "answer": "No encuentro información suficiente en la documentación interna (DTF-13).",
                "source": "none",
                "should_ask_user": True,
                "question_for_user": (
                    "¿Deseas que busque en fuentes externas? "
                    "(La información externa debe ser verificada con protocolos oficiales)"
                ),
                "relevance_score": relevance,
                "internal_sources": [
                    {
                        "filename": c.get("metadata", {}).get("filename"),
                        "page": c.get("metadata", {}).get("page_num"),
                        "similarity": c.get("similarity"),
                        "hybrid_score": c.get("hybrid_score"),
                    }
                    for c in retrieved_chunks
                ],
            }

        # allow_external True => búsqueda externa
        print("🌐 Buscando en fuentes externas (Tavily)...")
        external_results = self.external_searcher.search(question)

        if not external_results.get("success") or not external_results.get("results"):
            print("❌ Búsqueda externa falló - usando interno")
            context = self.format_context(retrieved_chunks)
            try:
                generation_result = self.generate(question, context, is_external=False)
            except Exception as e:
                print(f"⚠️ Error en Gemini: {type(e).__name__}: {e}")
                return self._llm_error_response(question, retrieved_chunks, relevance)
            return {
                "question": question,
                "answer": generation_result["answer"],
                "source": "internal",
                "sources": [
                    {
                        "filename": c.get("metadata", {}).get("filename"),
                        "page": c.get("metadata", {}).get("page_num"),
                        "similarity": c.get("similarity"),
                        "hybrid_score": c.get("hybrid_score"),
                    }
                    for c in retrieved_chunks
                ],
                "relevance_score": relevance,
                "external_search_failed": True,
                "external_error": external_results.get("error"),
            }

        print(f"✅ Encontrados {len(external_results['results'])} resultados externos")
        context = self.format_external_context(external_results)
        try:
            generation_result = self.generate(question, context, is_external=True)
        except Exception as e:
            print(f"⚠️ Error en Gemini: {type(e).__name__}: {e}")
            return self._llm_error_response(question, retrieved_chunks, relevance)

        return {
            "question": question,
            "answer": generation_result["answer"],
            "source": "external",
            "external_sources": external_results["results"],
            "relevance_score": relevance,
            "disclaimer": "⚠️ INFORMACIÓN DE FUENTES EXTERNAS - Debe ser verificada con protocolos oficiales (DTF-13)",
            "metadata": {"model": self.model_name, "temperature": self.temperature},
        }
