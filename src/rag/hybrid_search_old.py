"""
Búsqueda Híbrida: Semántica + Keyword
Mejora la recuperación combinando embeddings con búsqueda de palabras clave
"""
from typing import List, Dict
import re


class HybridSearch:
    """Combina búsqueda semántica y keyword search"""
    
    def __init__(self, vector_store, embeddings_generator):
        self.vector_store = vector_store
        self.embeddings_generator = embeddings_generator
    
    def keyword_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Búsqueda por palabras clave
        
        Args:
            query: Texto de búsqueda
            top_k: Número máximo de resultados
            
        Returns:
            Lista de chunks que contienen las palabras clave
        """
        # Extraer palabras clave (palabras de 4+ caracteres, sin stopwords comunes)
        stopwords = {'para', 'como', 'donde', 'cuando', 'cual', 'cuales', 'porque', 'sobre', 'este', 'esta'}
        words = re.findall(r'\b\w{4,}\b', query.lower())
        keywords = [w for w in words if w not in stopwords]
        
        if not keywords:
            return []
        
        # Obtener todos los documentos
        all_docs = self.vector_store.collection.get()
        
        # Buscar coincidencias
        matches = []
        for i, doc_text in enumerate(all_docs['documents']):
            doc_lower = doc_text.lower()
            
            # Contar cuántas keywords aparecen
            keyword_count = sum(1 for kw in keywords if kw in doc_lower)
            
            if keyword_count > 0:
                matches.append({
                    'index': i,
                    'text': doc_text,
                    'metadata': all_docs['metadatas'][i],
                    'keyword_score': keyword_count / len(keywords),  # Ratio de keywords encontradas
                    'id': all_docs['ids'][i]
                })
        
        # Ordenar por keyword_score
        matches.sort(key=lambda x: x['keyword_score'], reverse=True)
        
        return matches[:top_k]
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5
    ) -> List[Dict]:
        """
        Búsqueda híbrida combinando semántica y keywords
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados finales
            semantic_weight: Peso de la búsqueda semántica (0-1)
            keyword_weight: Peso de la búsqueda por keywords (0-1)
            
        Returns:
            Lista de chunks rankeados por score combinado
        """
        # 1. Búsqueda semántica
        semantic_results = self.vector_store.search_by_text(
            query_text=query,
            embeddings_generator=self.embeddings_generator,
            n_results=top_k * 2  # Recuperar más para luego rerank
        )
        
        # 2. Búsqueda por keywords
        keyword_results = self.keyword_search(query, top_k=top_k * 2)
        
        # 3. Combinar scores
        combined_scores = {}
        
        # Añadir scores semánticos
        for result in semantic_results:
            doc_id = result['id']
            combined_scores[doc_id] = {
                'semantic_score': result['similarity'],
                'keyword_score': 0.0,
                'data': result
            }
        
        # Añadir scores de keywords
        for result in keyword_results:
            doc_id = result['id']
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword_score'] = result['keyword_score']
            else:
                # Documento solo encontrado por keywords
                combined_scores[doc_id] = {
                    'semantic_score': 0.0,
                    'keyword_score': result['keyword_score'],
                    'data': {
                        'id': doc_id,
                        'text': result['text'],
                        'metadata': result['metadata'],
                        'similarity': 0.0
                    }
                }
        
        # 4. Calcular score final combinado
        for doc_id in combined_scores:
            semantic = combined_scores[doc_id]['semantic_score']
            keyword = combined_scores[doc_id]['keyword_score']
            
            # Score final ponderado
            final_score = (semantic * semantic_weight) + (keyword * keyword_weight)
            combined_scores[doc_id]['final_score'] = final_score
        
        # 5. Ordenar por score final
        ranked_results = sorted(
            combined_scores.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )[:top_k]
        
        # 6. Formatear resultados
        final_results = []
        for item in ranked_results:
            result = item['data'].copy()
            result['hybrid_score'] = item['final_score']
            result['semantic_score'] = item['semantic_score']
            result['keyword_score'] = item['keyword_score']
            final_results.append(result)
        
        return final_results