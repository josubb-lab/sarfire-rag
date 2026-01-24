"""
Búsqueda Híbrida: Semántica + Keyword
Combina embeddings con coincidencias de palabras clave y reranking ponderado.
"""
from __future__ import annotations

from typing import List, Dict
import re
import unicodedata


def _normalize(text: str) -> str:
    text = text.lower()
    # quitar acentos para mejorar matching en español
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


class HybridSearch:
    """Combina búsqueda semántica y keyword search."""

    def __init__(self, vector_store, embeddings_generator):
        self.vector_store = vector_store
        self.embeddings_generator = embeddings_generator

    def keyword_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Búsqueda por palabras clave (rápida, en memoria)."""
        stopwords = {
            "para", "como", "donde", "cuando", "cual", "cuales", "porque",
            "sobre", "este", "esta", "esto", "estos", "estas", "desde", "hasta",
            "segun", "según", "entre", "ante", "bajo", "tras", "sino", "solo", "sólo",
        }
        qn = _normalize(query)
        words = re.findall(r"\b\w{3,}\b", qn)
        keywords = [w for w in words if w not in stopwords]

        if not keywords:
            return []

        all_docs = self.vector_store.collection.get()
        matches: List[Dict] = []

        for i, doc_text in enumerate(all_docs["documents"]):
            doc_lower = _normalize(doc_text)

            keyword_count = sum(1 for kw in keywords if kw in doc_lower)
            if keyword_count > 0:
                matches.append(
                    {
                        "index": i,
                        "text": doc_text,
                        "metadata": all_docs["metadatas"][i],
                        "keyword_score": keyword_count / max(1, len(keywords)),
                        "id": all_docs["ids"][i],
                    }
                )

        matches.sort(key=lambda x: x["keyword_score"], reverse=True)
        return matches[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[Dict]:
        """Búsqueda híbrida combinando semántica y keywords."""
        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError("Los pesos no pueden ser negativos.")
        if semantic_weight + keyword_weight == 0:
            raise ValueError("La suma de pesos no puede ser 0.")

        # Normalizar pesos si no suman 1
        total = semantic_weight + keyword_weight
        semantic_weight /= total
        keyword_weight /= total

        semantic_results = self.vector_store.search_by_text(
            query_text=query,
            embeddings_generator=self.embeddings_generator,
            n_results=top_k * 2,
        )

        keyword_results = self.keyword_search(query, top_k=top_k * 2)

        combined: Dict[str, Dict] = {}

        for r in semantic_results:
            doc_id = r["id"]
            combined[doc_id] = {
                "semantic_score": r.get("similarity", 0.0),
                "keyword_score": 0.0,
                "data": r,
            }

        for r in keyword_results:
            doc_id = r["id"]
            if doc_id in combined:
                combined[doc_id]["keyword_score"] = r["keyword_score"]
            else:
                combined[doc_id] = {
                    "semantic_score": 0.0,
                    "keyword_score": r["keyword_score"],
                    "data": {
                        "id": doc_id,
                        "text": r["text"],
                        "metadata": r["metadata"],
                        "distance": None,
                        "similarity": 0.0,
                    },
                }

        for doc_id, item in combined.items():
            semantic = item["semantic_score"]
            keyword = item["keyword_score"]
            item["hybrid_score"] = (semantic * semantic_weight) + (keyword * keyword_weight)

        ranked = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]

        final: List[Dict] = []
        for item in ranked:
            data = dict(item["data"])
            data["hybrid_score"] = item["hybrid_score"]
            data["semantic_score"] = item["semantic_score"]
            data["keyword_score"] = item["keyword_score"]
            final.append(data)

        return final
