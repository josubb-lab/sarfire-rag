"""
Agente FORMADOR para SARFIRE-RAG
Especializado en explicaciones detalladas y formación sobre protocolos
"""
from typing import Dict, List
import google.generativeai as genai


class FormadorAgent:
    """Agente especializado en formación y explicación de procedimientos"""
    
    def __init__(self, rag_pipeline):
        """
        Args:
            rag_pipeline: Instancia de RAGPipeline configurada
        """
        self.rag = rag_pipeline
        self.agent_name = "Agente Formador"
    
    def generate_formador_prompt(self, query: str, context: str) -> str:
        """
        Genera prompt especializado para formación
        
        Args:
            query: Pregunta del usuario
            context: Contexto recuperado del RAG
            
        Returns:
            Prompt formateado para el agente formador
        """
        prompt = f"""Eres un INSTRUCTOR EXPERTO en protocolos de emergencias forestales del Consorcio Provincial de Bomberos de Valencia.

Tu misión es FORMAR y EDUCAR al personal sobre procedimientos operativos.

ESTILO DE RESPUESTA:
- Explica de forma CLARA y PEDAGÓGICA
- Usa ESTRUCTURA: Definición → Procedimiento → Consideraciones importantes
- Cita SIEMPRE las secciones del documento (ej: "Según la DTF-13, sección 3.2...")
- Si hay pasos, numéralos claramente
- Resalta PRECAUCIONES DE SEGURIDAD con énfasis
- Usa lenguaje técnico pero EXPLICATIVO

CONTEXTO DEL MANUAL DTF-13:
{context}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA FORMATIVA:
"""
        return prompt
    
    def process_query(self, query: str, top_k: int = 5) -> Dict:
        """
        Procesa consulta en modo formación
        
        Args:
            query: Pregunta del usuario
            top_k: Chunks a recuperar
            
        Returns:
            Dict con respuesta formativa y metadata
        """
        print(f"\n🎓 {self.agent_name} activado")
        print(f"📚 Consultando manual DTF-13...")
        
        # 1. Retrieve del RAG
        retrieved_chunks = self.rag.retrieve(query, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                'answer': "No encontré información relevante en el manual DTF-13 para responder tu consulta de formación.",
                'sources': [],
                'agent': self.agent_name,
                'query': query
            }
        
        print(f"✅ Encontrados {len(retrieved_chunks)} fragmentos relevantes")
        
        # 2. Format context
        context = self.rag.format_context(retrieved_chunks)
        
        # 3. Generate con prompt especializado
        formador_prompt = self.generate_formador_prompt(query, context)
        
        print(f"🤖 Generando explicación formativa...")
        
        response = self.rag.model.generate_content(
            formador_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,  # Algo más creativo que el RAG base para explicaciones
            )
        )
        
        # 4. Preparar respuesta
        result = {
            'answer': response.text,
            'sources': [
                {
                    'filename': chunk['metadata']['filename'],
                    'page': chunk['metadata']['page_num'],
                    'relevance': chunk.get('hybrid_score', chunk.get('similarity', 0.0)),
                    'preview': chunk['text'][:200]
                }
                for chunk in retrieved_chunks[:3]  # Top 3 fuentes
            ],
            'agent': self.agent_name,
            'query': query,
            'chunks_used': len(retrieved_chunks)
        }
        
        return result
    
    def print_response(self, result: Dict):
        """Imprime la respuesta de forma estructurada"""
        print("\n" + "="*70)
        print(f"🎓 {result['agent'].upper()}")
        print("="*70)
        print(f"\n❓ CONSULTA DE FORMACIÓN:")
        print(f"{result['query']}")
        
        print(f"\n📖 EXPLICACIÓN:")
        print(result['answer'])
        
        print(f"\n📚 FUENTES CONSULTADAS:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['filename']} (Página {source['page']})")
            print(f"   Relevancia: {source['relevance']:.3f}")
            print(f"   Fragmento: {source['preview']}...")
        
        print("\n" + "="*70)