"""
RAG Pipeline completo para SARFIRE-RAG
Integra recuperación (ChromaDB) + generación (Gemini) + Fallback externo (Tavily)
"""
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
        top_k: int = 5,
        use_hybrid_search: bool = True,
        enable_external_fallback: bool = True,  # ← NUEVO
        relevance_threshold: float = 0.5  # ← NUEVO
    ):
        """
        Args:
            vector_store: Instancia de VectorStore
            embeddings_generator: Instancia de EmbeddingsGenerator
            model_name: Modelo de Gemini a usar
            temperature: Temperatura del LLM (0-1)
            top_k: Número de chunks a recuperar
            use_hybrid_search: Si True, usa búsqueda híbrida
            enable_external_fallback: Si True, permite búsqueda externa
            relevance_threshold: Umbral de relevancia (0-1) para activar fallback
        """
        self.vector_store = vector_store
        self.embeddings_generator = embeddings_generator
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.use_hybrid_search = use_hybrid_search
        self.enable_external_fallback = enable_external_fallback
        self.relevance_threshold = relevance_threshold
    
        # Inicializar búsqueda híbrida si está habilitada
        if use_hybrid_search:
            self.hybrid_search = HybridSearch(vector_store, embeddings_generator)
        
        # Inicializar external searcher si está habilitado
        if enable_external_fallback:
            try:
                self.external_searcher = ExternalSearcher()
                print("✅ Fallback externo habilitado (Tavily)")
            except Exception as e:
                print(f"⚠️  Fallback externo deshabilitado: {e}")
                self.external_searcher = None
        else:
            self.external_searcher = None
        
        # Cargar API key
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY no encontrada. "
                "Asegúrate de tenerla en el archivo .env"
            )
        
        # Configurar Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        print(f"✅ RAG Pipeline inicializado")
        print(f"   Modelo: {model_name}")
        print(f"   Temperature: {temperature}")
        print(f"   Top-K retrieval: {top_k}")
        print(f"   Fallback externo: {enable_external_fallback}")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Recupera chunks relevantes

        Args:
            query: Consulta del usuario
            top_k: Número de resultados (usa self.top_k por defecto)
        
        Returns:
            Lista de chunks relevantes
        """
        if top_k is None:
            top_k = self.top_k
    
        # Usar búsqueda híbrida si está habilitada
        if self.use_hybrid_search:
            results = self.hybrid_search.hybrid_search(
                query=query,
                top_k=top_k,
                semantic_weight=0.4,
                keyword_weight=0.6
            )
        else:
            # Búsqueda semántica tradicional
            results = self.vector_store.search_by_text(
                query_text=query,
                embeddings_generator=self.embeddings_generator,
                n_results=top_k
            )
    
        return results
    
    def assess_relevance(self, retrieved_chunks: List[Dict]) -> float:
        """
        Evalúa la relevancia promedio de los chunks recuperados
        
        Args:
            retrieved_chunks: Lista de chunks con 'similarity'
            
        Returns:
            Score de relevancia 0-1 (1 = muy relevante)
        """
        if not retrieved_chunks:
            return 0.0
        
        # Calcular promedio de similarity
        similarities = [chunk.get('similarity', 0.0) for chunk in retrieved_chunks]
        avg_similarity = sum(similarities) / len(similarities)
        
        return avg_similarity
    
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Formatea los chunks recuperados como contexto para el LLM
        
        Args:
            retrieved_chunks: Chunks del retrieve
            
        Returns:
            Contexto formateado
        """
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk['metadata']
            context_parts.append(
                f"[FUENTE {i}]\n"
                f"Documento: {meta['filename']}\n"
                f"Página: {meta['page_num']}\n"
                f"Contenido: {chunk['text']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def format_external_context(self, external_results: Dict) -> str:
        """
        Formatea resultados externos para el LLM
        
        Args:
            external_results: Resultados de Tavily
            
        Returns:
            Contexto formateado de fuentes externas
        """
        if not external_results.get('success') or not external_results.get('results'):
            return "No se encontró información externa relevante."
        
        context_parts = []
        
        for i, result in enumerate(external_results['results'], 1):
            context_parts.append(
                f"[FUENTE EXTERNA {i}]\n"
                f"Título: {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Contenido: {result['content'][:500]}...\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str, is_external: bool = False) -> str:
        """
        Genera el prompt completo para el LLM
        
        Args:
            query: Pregunta del usuario
            context: Contexto recuperado
            is_external: Si True, el contexto es de fuentes externas
            
        Returns:
            Prompt formateado
        """
        if is_external:
            prompt = f"""Eres un asistente experto en emergencias forestales y protocolos de bomberos.

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
        else:
            prompt = f"""Eres un asistente experto en emergencias forestales y protocolos de bomberos.

Tu tarea es responder preguntas basándote ÚNICAMENTE en la información proporcionada en el contexto.

REGLAS IMPORTANTES:
1. Responde SOLO con información del contexto proporcionado
2. Si la información no está en el contexto, di claramente "No encuentro esa información en los manuales"
3. Cita las fuentes (documento y página) cuando respondas
4. Sé preciso y técnico, este es material para profesionales
5. Si hay procedimientos de seguridad, menciόnalos SIEMPRE

CONTEXTO DE LOS MANUALES:
{context}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:"""
        
        return prompt
    
    def generate(self, query: str, context: str, is_external: bool = False) -> Dict:
        """
        Genera respuesta usando Gemini
        
        Args:
            query: Pregunta del usuario
            context: Contexto recuperado
            is_external: Si el contexto es de fuentes externas
            
        Returns:
            Dict con respuesta y metadata
        """
        # Crear prompt
        prompt = self.generate_prompt(query, context, is_external)
        
        # Generar respuesta
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
            )
        )
        
        return {
            'answer': response.text,
            'prompt_tokens': len(prompt.split()),  # Aproximado
        }
    
    def query(
        self, 
        question: str, 
        top_k: Optional[int] = None,
        allow_external: Optional[bool] = None
    ) -> Dict:
        """
        Pipeline completo: Retrieve + Generate + External Fallback
        
        Args:
            question: Pregunta del usuario
            top_k: Número de chunks a recuperar
            allow_external: Si None, pregunta al usuario; si True/False, fuerza decisión
            
        Returns:
            Dict con respuesta, fuentes y metadata
        """
        print(f"\n🔍 Buscando información relevante...")
        
        # 1. Retrieve de fuentes internas
        retrieved_chunks = self.retrieve(question, top_k)
        
        print(f"✅ Encontrados {len(retrieved_chunks)} chunks relevantes")
        
        # 2. Evaluar relevancia
        relevance = self.assess_relevance(retrieved_chunks)
        print(f"📊 Relevancia promedio: {relevance:.3f} (umbral: {self.relevance_threshold})")
        
        # 3. Decidir si usar fuentes externas
        should_use_external = (
            relevance < self.relevance_threshold and 
            self.external_searcher is not None
        )
        
        # CASO 1: Relevancia suficiente - usar RAG interno
        # CASO 1: allow_external=False → SOLO DTF-13
        if allow_external == False:
            print("✅ Usando documentación interna")
            
            similarities = [f"{c['similarity']:.3f}" for c in retrieved_chunks[:3]]
            print(f"   Similitudes: {similarities}")
            
            # Format context
            context = self.format_context(retrieved_chunks)
            
            # Generate
            print(f"\n🤖 Generando respuesta con {self.model_name}...")
            generation_result = self.generate(question, context, is_external=False)
            
            # Preparar respuesta
            result = {
                'question': question,
                'answer': generation_result['answer'],
                'source': 'internal',
                'sources': [
                    {
                        'filename': chunk['metadata']['filename'],
                        'page': chunk['metadata']['page_num'],
                        'similarity': chunk['similarity'],
                        'text_preview': chunk['text'][:200]
                    }
                    for chunk in retrieved_chunks
                ],
                'relevance_score': relevance,
                'metadata': {
                    'chunks_retrieved': len(retrieved_chunks),
                    'model': self.model_name,
                    'temperature': self.temperature
                }
            }
            
            return result
        

        # CASO 2: Relevancia suficiente → usar RAG interno
        if not should_use_external:
            print("✅ Usando documentación interna (relevancia suficiente)")
            
            similarities = [f"{c['similarity']:.3f}" for c in retrieved_chunks[:3]]
            print(f"   Similitudes: {similarities}")
            
            context = self.format_context(retrieved_chunks)
            print(f"\n🤖 Generando respuesta con {self.model_name}...")
            generation_result = self.generate(question, context, is_external=False)
            
            result = {
                'question': question,
                'answer': generation_result['answer'],
                'source': 'internal',
                'sources': [
                    {
                        'filename': chunk['metadata']['filename'],
                        'page': chunk['metadata']['page_num'],
                        'similarity': chunk['similarity'],
                        'text_preview': chunk['text'][:200]
                    }
                    for chunk in retrieved_chunks
                ],
                'relevance_score': relevance,
                'metadata': {
                    'chunks_retrieved': len(retrieved_chunks),
                    'model': self.model_name,
                    'temperature': self.temperature
                }
            }
            
            return result
        # CASO 2: Baja relevancia - preguntar al usuario
        if allow_external is None:
            print("⚠️  Baja relevancia - consulta externa disponible")
            
            return {
                'question': question,
                'answer': "No encuentro información suficiente en la documentación interna (DTF-13).",
                'source': 'none',
                'should_ask_user': True,
                'question_for_user': "¿Deseas que busque en fuentes externas? (La información externa debe ser verificada con protocolos oficiales)",
                'relevance_score': relevance,
                'internal_sources': [
                    {
                        'filename': chunk['metadata']['filename'],
                        'page': chunk['metadata']['page_num'],
                        'similarity': chunk['similarity']
                    }
                    for chunk in retrieved_chunks
                ]
            }
        
        # CASO 3: Usuario autorizó búsqueda externa
        print("🌐 Buscando en fuentes externas (Tavily)...")
        external_results = self.external_searcher.search(question)
        
        if not external_results['success'] or not external_results['results']:
            print("❌ Búsqueda externa falló - usando interno")
            
            context = self.format_context(retrieved_chunks)
            generation_result = self.generate(question, context, is_external=False)
            
            return {
                'question': question,
                'answer': generation_result['answer'],
                'source': 'internal',
                'sources': [
                    {
                        'filename': chunk['metadata']['filename'],
                        'page': chunk['metadata']['page_num'],
                        'similarity': chunk['similarity']
                    }
                    for chunk in retrieved_chunks
                ],
                'relevance_score': relevance,
                'external_search_failed': True
            }
        
        # Búsqueda externa exitosa
        print(f"✅ Encontrados {len(external_results['results'])} resultados externos")
        
        context = self.format_external_context(external_results)
        generation_result = self.generate(question, context, is_external=True)
        
        return {
            'question': question,
            'answer': generation_result['answer'],
            'source': 'external',
            'external_sources': external_results['results'],
            'relevance_score': relevance,
            'disclaimer': '⚠️ INFORMACIÓN DE FUENTES EXTERNAS - Debe ser verificada con protocolos oficiales (DTF-13)',
            'metadata': {
                'model': self.model_name,
                'temperature': self.temperature
            }
        }
    
    def print_result(self, result: Dict):
        """Imprime el resultado de forma legible"""
        print("\n" + "="*70)
        print("❓ PREGUNTA")
        print("="*70)
        print(result['question'])
        
        print("\n" + "="*70)
        print("💡 RESPUESTA")
        print("="*70)
        print(result['answer'])
        
        # Disclaimer si es externo
        if result.get('disclaimer'):
            print("\n" + result['disclaimer'])
        
        # Fuentes internas
        if result.get('sources'):
            print("\n" + "="*70)
            print("📚 FUENTES CONSULTADAS (DTF-13)")
            print("="*70)
            for i, source in enumerate(result['sources'], 1):
                print(f"\n{i}. {source['filename']} (página {source['page']})")
                if 'similarity' in source:
                    print(f"   Similitud: {source['similarity']:.3f}")
                if 'text_preview' in source:
                    print(f"   Vista previa: {source['text_preview']}...")
        
        # Fuentes externas
        if result.get('external_sources'):
            print("\n" + "="*70)
            print("🌐 FUENTES EXTERNAS")
            print("="*70)
            for i, source in enumerate(result['external_sources'], 1):
                print(f"\n{i}. {source['title']}")
                print(f"   URL: {source['url']}")
                print(f"   Score: {source.get('score', 'N/A')}")
        
        print("\n" + "="*70)