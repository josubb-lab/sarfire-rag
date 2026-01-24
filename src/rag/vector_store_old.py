"""
Vector Store para SARFIRE-RAG
Gestiona almacenamiento y búsqueda en ChromaDB
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from pathlib import Path




class VectorStore:
        """Gestiona el vector store con ChromaDB"""
    
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

    
        """
        Args:
            persist_directory: Directorio donde persistir la BD
            collection_name: Nombre de la colección
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Inicializar cliente ChromaDB
        print(f"⚙️  Inicializando ChromaDB en {persist_directory}...")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Obtener o crear colección
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Similitud coseno
        )
        print(f"✅ ChromaDB inicializado (colección: {collection_name})")
    
    def add_chunks(self, embedded_chunks: List[Dict], batch_size: int = 100):
        """
        Añade chunks con embeddings al vector store
        
        Args:
            embedded_chunks: Chunks con embeddings del EmbeddingsGenerator
            batch_size: Tamaño del batch para inserción
        """
        print(f"\n💾 Añadiendo {len(embedded_chunks)} chunks a ChromaDB...")
        
        # Preparar datos para ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for idx, chunk in enumerate(embedded_chunks):
            # ID único para cada chunk
            chunk_id = f"{chunk['metadata']['filename']}_p{chunk['metadata']['page_num']}_c{chunk['metadata']['chunk_idx']}"
            ids.append(chunk_id)
            
            # Embedding (convertir a lista si es numpy array)
            emb = chunk['embedding']
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            embeddings.append(emb)
            
            # Texto del documento
            documents.append(chunk['text'])
            
            # Metadata (ChromaDB solo acepta tipos básicos)
            metadata = {
                'filename': chunk['metadata']['filename'],
                'page_num': chunk['metadata']['page_num'],
                'chunk_idx': chunk['metadata']['chunk_idx'],
                'chunk_size': chunk['metadata']['chunk_size']
            }
            metadatas.append(metadata)
        
        # Insertar en batches
        total_batches = (len(ids) + batch_size - 1) // batch_size
        
        for i in range(0, len(ids), batch_size):
            batch_num = i // batch_size + 1
            print(f"   Batch {batch_num}/{total_batches}...", end='\r')
            
            end_idx = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print(f"\n✅ {len(ids)} chunks añadidos correctamente")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Busca chunks similares
        
        Args:
            query_embedding: Embedding de la consulta
            n_results: Número de resultados a devolver
            where: Filtros de metadata (opcional)
            
        Returns:
            Diccionario con resultados de ChromaDB
        """
        # Convertir a lista si es numpy
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def search_by_text(
        self,
        query_text: str,
        embeddings_generator,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Busca usando texto (genera embedding automáticamente)
        
        Args:
            query_text: Texto de la consulta
            embeddings_generator: Instancia de EmbeddingsGenerator
            n_results: Número de resultados
            where: Filtros de metadata
            
        Returns:
            Lista de resultados formateados
        """
        # Generar embedding de la consulta
        query_emb = embeddings_generator.generate_embedding(query_text)
        
        # Buscar
        results = self.search(query_emb, n_results, where)
        
        # Formatear resultados
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convertir distancia a similitud
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del vector store"""
        count = self.collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'persist_directory': str(self.persist_directory)
        }
    
    def print_stats(self):
        """Imprime estadísticas del vector store"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS DE CHROMADB")
        print("="*70)
        print(f"Colección: {stats['collection_name']}")
        print(f"Total de chunks: {stats['total_chunks']}")
        print(f"Directorio: {stats['persist_directory']}")
        print("="*70 + "\n")
    
    def clear(self):
        """Elimina todos los documentos de la colección"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("🗑️  Colección limpiada")