"""
Vector Store para SARFIRE-RAG
Gestiona almacenamiento y búsqueda en ChromaDB
"""
from __future__ import annotations

from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
import numpy as np


class VectorStore:
    """Gestiona el vector store con ChromaDB (persistente)."""

    def __init__(
        self,
        persist_directory: str = "data/processed/chromadb",
        collection_name: str = "sarfire_docs",
        *,
        disable_telemetry: bool = True,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Inicializar cliente ChromaDB
        print(f"⚙️  Inicializando ChromaDB en {self.persist_directory}...")

        settings = Settings(anonymized_telemetry=not disable_telemetry)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=settings,
        )

        # Obtener o crear colección
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Similitud coseno
        )
        print(f"✅ ChromaDB inicializado (colección: {collection_name})")

    def add_chunks(self, embedded_chunks: List[Dict], batch_size: int = 100) -> None:
        """Añade chunks con embeddings al vector store."""
        print(f"\n💾 Añadiendo {len(embedded_chunks)} chunks a ChromaDB...")

        ids: List[str] = []
        embeddings: List[list] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for chunk in embedded_chunks:
            meta = chunk["metadata"]
            chunk_id = f"{meta['filename']}_p{meta['page_num']}_c{meta['chunk_idx']}"
            ids.append(chunk_id)

            emb = chunk["embedding"]
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            embeddings.append(emb)

            documents.append(chunk["text"])

            metadatas.append(
                {
                    "filename": meta["filename"],
                    "page_num": meta["page_num"],
                    "chunk_idx": meta["chunk_idx"],
                    "chunk_size": meta.get("chunk_size", len(chunk["text"])),
                }
            )

        total_batches = (len(ids) + batch_size - 1) // batch_size
        for i in range(0, len(ids), batch_size):
            batch_num = i // batch_size + 1
            print(f"   Batch {batch_num}/{total_batches}...", end="\r")
            end_idx = min(i + batch_size, len(ids))

            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        print(f"\n✅ {len(ids)} chunks añadidos correctamente")

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict:
        """Busca chunks similares."""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

    def search_by_text(
        self,
        query_text: str,
        embeddings_generator,
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Busca usando texto (genera embedding automáticamente)."""
        query_emb = embeddings_generator.generate_embedding(query_text)
        results = self.search(query_emb, n_results, where)

        formatted_results: List[Dict] = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Chroma devuelve distancia para cosine. Convertimos a similitud en [~0,1]
            similarity = 1 - distance

            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    "similarity": similarity,
                }
            )

        return formatted_results

    def get_stats(self) -> Dict:
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "persist_directory": str(self.persist_directory),
        }

    def print_stats(self) -> None:
        stats = self.get_stats()
        print("\n" + "=" * 70)
        print("📊 ESTADÍSTICAS DE CHROMADB")
        print("=" * 70)
        print(f"Colección: {stats['collection_name']}")
        print(f"Total de chunks: {stats['total_chunks']}")
        print(f"Directorio: {stats['persist_directory']}")
        print("=" * 70 + "\n")

    def clear(self) -> None:
        """Elimina todos los documentos de la colección."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print("🗑️  Colección limpiada")
