"""
Agente FORMADOR para SARFIRE-RAG (MVP)
- Responde con base en DTF-13 (RAG interno).
- Si la relevancia es baja y la política lo permite, delega en RAGPipeline.query() para gestionar confirmación o búsqueda externa.
"""

from typing import Dict, Optional, Any


class FormadorAgent:
    """Agente especializado en formación y explicación de procedimientos."""

    def __init__(self, rag_pipeline: Any):
        self.rag = rag_pipeline
        self.agent_name = "Agente Formador"

    def process_query(self, query: str, top_k: int = 5, allow_external: Optional[bool] = None) -> Dict:
        """
        Procesa consulta en modo formación delegando la lógica de relevancia/fallback al RAGPipeline.

        allow_external:
          - False: solo DTF-13
          - None : preguntar al usuario si la relevancia es baja
          - True : usar fuentes externas si la relevancia es baja (con disclaimer)
        """
        result = self.rag.query(question=query, top_k=top_k, allow_external=allow_external)

        # Normalizar algunos campos
        result.setdefault("agent", self.agent_name)
        result.setdefault("query", query)
        return result
