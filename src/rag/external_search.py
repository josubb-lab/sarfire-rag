"""
External Search Module - Tavily Integration
Búsqueda en fuentes externas cuando RAG interno no tiene información suficiente
"""

from tavily import TavilyClient
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class ExternalSearcher:
    """
    Wrapper para Tavily API
    Gestiona búsquedas en fuentes externas con manejo de errores
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el cliente de Tavily
        
        Args:
            api_key: API key de Tavily (opcional, lee de .env si no se proporciona)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY no encontrada en variables de entorno")
        
        self.client = TavilyClient(api_key=self.api_key)
        self.max_results = 3  # Límite de resultados por búsqueda
    
    def search(
        self, 
        query: str, 
        max_results: Optional[int] = None,
        search_depth: str = "basic"
    ) -> Dict:
        """
        Busca información en fuentes externas
        
        Args:
            query: Pregunta o consulta del usuario
            max_results: Número máximo de resultados (default: 3)
            search_depth: "basic" o "advanced" (advanced consume más quota)
        
        Returns:
            {
                'success': bool,
                'results': [
                    {
                        'title': str,
                        'url': str,
                        'content': str,
                        'score': float
                    }
                ],
                'query': str,
                'source': 'external',
                'error': str (si success=False)
            }
        """
        max_results = max_results or self.max_results
        
        try:
            # Llamada a Tavily API
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=False  # No necesitamos HTML completo
            )
            
            # Procesar resultados
            processed_results = []
            for result in response.get('results', []):
                processed_results.append({
                    'title': result.get('title', 'Sin título'),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'score': result.get('score', 0.0)
                })
            
            return {
                'success': True,
                'results': processed_results,
                'query': query,
                'source': 'external',
                'total_results': len(processed_results)
            }
            
        except Exception as e:
            # Manejo de errores
            return {
                'success': False,
                'results': [],
                'query': query,
                'source': 'external',
                'error': f"Error en búsqueda externa: {str(e)}",
                'total_results': 0
            }
    
    def format_results_for_llm(self, search_results: Dict) -> str:
        """
        Formatea los resultados de búsqueda para contexto del LLM
        
        Args:
            search_results: Resultados del método search()
        
        Returns:
            String formateado para incluir en el prompt
        """
        if not search_results['success'] or not search_results['results']:
            return "No se encontró información externa relevante."
        
        context = "INFORMACIÓN DE FUENTES EXTERNAS:\n\n"
        
        for i, result in enumerate(search_results['results'], 1):
            context += f"[{i}] {result['title']}\n"
            context += f"Fuente: {result['url']}\n"
            context += f"{result['content'][:500]}...\n\n"  # Limitar a 500 chars
        
        return context


# Función auxiliar para testing rápido
def test_external_search():
    """Test básico del External Searcher"""
    searcher = ExternalSearcher()
    
    # Test query
    query = "protocolo actuación incendios forestales España"
    results = searcher.search(query)
    
    print(f"✅ Búsqueda: {query}")
    print(f"Success: {results['success']}")
    print(f"Resultados: {results['total_results']}")
    
    if results['success']:
        for i, r in enumerate(results['results'], 1):
            print(f"\n{i}. {r['title']}")
            print(f"   URL: {r['url']}")
            print(f"   Score: {r['score']}")
    else:
        print(f"Error: {results.get('error', 'Desconocido')}")


if __name__ == "__main__":
    test_external_search()
