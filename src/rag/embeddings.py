"""
Embeddings Generator para SARFIRE-RAG
Convierte texto en vectores usando sentence-transformers
"""
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingsGenerator:
    """Genera embeddings de chunks de texto"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Modelo de sentence-transformers a usar
                       'all-MiniLM-L6-v2' = rápido, 384 dims, bueno para español
        """
        self.model_name = model_name
        print(f"⚙️  Cargando modelo de embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Modelo cargado (dimensión: {self.model.get_sentence_embedding_dimension()})")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding de un texto
        
        Args:
            text: Texto a convertir
            
        Returns:
            Vector numpy con el embedding
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Genera embeddings para múltiples textos
        
        Args:
            texts: Lista de textos
            batch_size: Tamaño del batch para procesar
            show_progress: Mostrar barra de progreso
            
        Returns:
            Lista de embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Añade embeddings a los chunks
        
        Args:
            chunks: Lista de chunks del DocumentChunker
            
        Returns:
            Chunks con embeddings añadidos
        """
        print(f"\n🔄 Generando embeddings para {len(chunks)} chunks...")
        
        # Extraer textos
        texts = [chunk['text'] for chunk in chunks]
        
        # Generar embeddings en batch (más eficiente)
        embeddings = self.generate_embeddings_batch(texts, show_progress=True)
        
        # Añadir embeddings a cada chunk
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embedding
            embedded_chunks.append(embedded_chunk)
        
        print(f"✅ Embeddings generados correctamente")
        return embedded_chunks
    
    def print_stats(self, embedded_chunks: List[Dict]):
        """Muestra estadísticas de los embeddings"""
        if not embedded_chunks:
            print("No hay chunks con embeddings")
            return
        
        # Verificar que todos tienen embeddings
        chunks_with_emb = sum(1 for c in embedded_chunks if 'embedding' in c)
        
        # Dimensión de embeddings
        first_emb = embedded_chunks[0]['embedding']
        emb_dim = len(first_emb)
        
        # Calcular algunas estadísticas
        embeddings = np.array([c['embedding'] for c in embedded_chunks])
        mean_norm = np.linalg.norm(embeddings, axis=1).mean()
        
        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS DE EMBEDDINGS")
        print("="*70)
        print(f"Total de chunks: {len(embedded_chunks)}")
        print(f"Chunks con embeddings: {chunks_with_emb}")
        print(f"Dimensión de embeddings: {emb_dim}")
        print(f"Modelo usado: {self.model_name}")
        print(f"Norma media de vectores: {mean_norm:.4f}")
        print("="*70 + "\n")
    
    def test_similarity(self, embedded_chunks: List[Dict], n_examples: int = 3):
        """
        Muestra ejemplos de similitud entre chunks
        
        Args:
            embedded_chunks: Chunks con embeddings
            n_examples: Número de ejemplos a mostrar
        """
        print("\n" + "="*70)
        print(f"🔍 TEST DE SIMILITUD (primeros {n_examples} chunks)")
        print("="*70)
        
        for i in range(min(n_examples, len(embedded_chunks))):
            query_chunk = embedded_chunks[i]
            query_emb = query_chunk['embedding']
            
            # Calcular similitud con todos los chunks
            similarities = []
            for j, other_chunk in enumerate(embedded_chunks):
                if i != j:  # No comparar consigo mismo
                    other_emb = other_chunk['embedding']
                    # Similitud coseno
                    similarity = np.dot(query_emb, other_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(other_emb)
                    )
                    similarities.append((j, similarity, other_chunk))
            
            # Ordenar por similitud
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Mostrar query y top 2 más similares
            print(f"\n--- QUERY CHUNK {i+1} ---")
            print(f"Doc: {query_chunk['metadata']['filename']}")
            print(f"Texto: {query_chunk['text'][:150]}...")
            print(f"\nTop 2 chunks más similares:")
            
            for rank, (idx, sim, chunk) in enumerate(similarities[:2], 1):
                print(f"\n  {rank}. Similitud: {sim:.4f}")
                print(f"     Doc: {chunk['metadata']['filename']}")
                print(f"     Texto: {chunk['text'][:100]}...")
            
            print("-" * 70)