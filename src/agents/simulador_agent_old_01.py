"""
Agente SIMULADOR para SARFIRE-RAG v2.0 - VERSIÓN SIMPLIFICADA
Especializado en generar escenarios operativos y evaluar decisiones
Con fallback externo simplificado
"""
from typing import Dict, List, Optional
import google.generativeai as genai


class SimuladorAgent:
    """Agente especializado en simulación de escenarios operativos"""
    
    def __init__(self, rag_pipeline):
        """
        Args:
            rag_pipeline: Instancia de RAGPipeline configurada
        """
        self.rag = rag_pipeline
        self.agent_name = "Agente Simulador"
        self.current_scenario = None
    
    def generate_scenario_prompt(self, topic: str, context: str, is_external: bool = False) -> str:
        """
        Genera prompt para crear escenario
        
        Args:
            topic: Tema del escenario
            context: Contexto del manual o fuentes externas
            is_external: Si el contexto es de fuentes externas
            
        Returns:
            Prompt para generación de escenario
        """
        if is_external:
            prompt = f"""Eres un INSTRUCTOR DE SIMULACROS creando un escenario de entrenamiento.

⚠️ IMPORTANTE: Estás usando INFORMACIÓN DE FUENTES EXTERNAS (no del DTF-13 oficial).

Genera un escenario educativo basado en esta información:
- Mantén el enfoque en dilemas de toma de decisiones
- Usa la información de manera pedagógica
- Al final indica que debe validarse con protocolos oficiales

INFORMACIÓN DISPONIBLE:
{context}

TEMA: {topic}

GENERA ESCENARIO (150-250 palabras):
1. Situación inicial
2. Evolución
3. Dilema operativo
4. "¿Qué harías?"

⚠️ Añade al final: "Nota: Escenario basado en fuentes externas. Validar con DTF-13."
"""
        else:
            prompt = f"""Eres un INSTRUCTOR DE SIMULACROS del Consorcio Provincial de Bomberos de Valencia.

Tu misión es crear ESCENARIOS OPERATIVOS REALISTAS para entrenar al personal.

ESTRUCTURA DEL ESCENARIO:
1. **SITUACIÓN INICIAL**: Describe el incendio (ubicación, extensión, condiciones, recursos)
2. **EVOLUCIÓN**: Qué está pasando en este momento crítico
3. **TU DECISIÓN**: Plantea 3-4 opciones de actuación
4. **PREGUNTA**: "¿Qué harías tú?"

REGLAS:
- Basado en protocolos del DTF-13
- Escenario realista y detallado
- Opciones plausibles
- Incluye dilemas operativos reales
- Longitud: 150-250 palabras

CONTEXTO DEL MANUAL DTF-13:
{context}

TEMA DEL ESCENARIO:
{topic}

GENERA EL ESCENARIO:
"""
        return prompt
    
    def generate_evaluation_prompt(self, scenario: str, user_decision: str, context: str) -> str:
        """Genera prompt para evaluar decisión"""
        prompt = f"""Eres un EVALUADOR EXPERTO de simulacros del Consorcio Provincial de Bomberos de Valencia.

ESTRUCTURA DE LA EVALUACIÓN:
1. **ANÁLISIS**: ¿Qué implicaciones tiene esta decisión?
2. **ACIERTOS**: Aspectos correctos según el DTF-13
3. **RIESGOS/ERRORES**: Problemas que puede causar
4. **DECISIÓN ÓPTIMA**: Mejor actuación según protocolo
5. **LECCIONES**: 2-3 puntos clave

TONO: Constructivo y formativo

CONTEXTO DTF-13:
{context}

ESCENARIO:
{scenario}

DECISIÓN DEL USUARIO:
{user_decision}

EVALUACIÓN:
"""
        return prompt
    
    def create_scenario(
        self, 
        topic: str = None, 
        top_k: int = 5,
        allow_external: bool = False
    ) -> Dict:
        """
        Crea un escenario operativo
        
        FLUJO SIMPLIFICADO:
        1. Busca en DTF-13 (retrieve)
        2. Si relevancia baja Y allow_external=True → busca en Tavily
        3. Genera escenario con el contexto obtenido
        """
        print(f"\n🎭 {self.agent_name} activado")
        
        if not topic:
            topic = "Genera un escenario operativo de incendio forestal con dilemas"
        
        print(f"🎬 Creando escenario: {topic[:60]}...")
        
        # PASO 1: Buscar en DTF-13
        retrieved_chunks = self.rag.retrieve(topic, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                'scenario': "No se pudo generar el escenario por falta de información.",
                'agent': self.agent_name,
                'topic': topic
            }
        
        # PASO 2: Evaluar relevancia
        relevance = self.rag.assess_relevance(retrieved_chunks)
        print(f"📊 Relevancia: {relevance:.3f} (umbral: {self.rag.relevance_threshold})")
        
        # PASO 3: Decidir fuente
        use_external = (
            relevance < self.rag.relevance_threshold and 
            allow_external and 
            self.rag.external_searcher is not None
        )
        
        if use_external:
            # Buscar en fuentes externas
            print(f"🌐 Buscando en fuentes externas (Tavily)...")
            
            external_results = self.rag.external_searcher.search(topic, max_results=3)
            
            if external_results['success'] and external_results['results']:
                print(f"✅ Encontrados {len(external_results['results'])} resultados externos")
                
                # Formatear contexto externo
                context = "INFORMACIÓN DE FUENTES EXTERNAS:\n\n"
                for i, result in enumerate(external_results['results'], 1):
                    context += f"[{i}] {result['title']}\n"
                    context += f"URL: {result['url']}\n"
                    context += f"{result['content'][:400]}...\n\n"
                
                source_type = 'external'
                sources_list = external_results['results']
            else:
                # Fallback a interno si falla
                print("❌ Búsqueda externa falló - usando DTF-13")
                context = self.rag.format_context(retrieved_chunks)
                source_type = 'internal'
                sources_list = retrieved_chunks[:3]
        else:
            # Usar fuentes internas
            print(f"✅ Usando DTF-13 (relevancia suficiente o external desactivado)")
            context = self.rag.format_context(retrieved_chunks)
            source_type = 'internal'
            sources_list = retrieved_chunks[:3]
        
        # PASO 4: Generar escenario
        scenario_prompt = self.generate_scenario_prompt(topic, context, is_external=(source_type=='external'))
        
        print(f"🤖 Generando escenario...")
        
        response = self.rag.model.generate_content(
            scenario_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.7)
        )
        
        # Guardar escenario
        self.current_scenario = {
            'text': response.text,
            'topic': topic,
            'source': source_type,
            'sources': sources_list
        }
        
        # Preparar resultado
        result = {
            'scenario': response.text,
            'agent': self.agent_name,
            'topic': topic,
            'source': source_type
        }
        
        if source_type == 'internal':
            result['sources'] = [
                {
                    'filename': chunk['metadata']['filename'],
                    'page': chunk['metadata']['page_num']
                }
                for chunk in sources_list
            ]
        else:
            result['external_sources'] = sources_list
            result['disclaimer'] = '⚠️ Escenario basado en fuentes externas - Validar con DTF-13'
        
        return result
    
    def evaluate_decision(self, user_decision: str, top_k: int = 5) -> Dict:
        """Evalúa la decisión del usuario"""
        if not self.current_scenario:
            return {
                'evaluation': "No hay un escenario activo para evaluar.",
                'agent': self.agent_name
            }
        
        print(f"\n⚖️ Evaluando tu decisión...")
        
        # Recuperar contexto
        query = f"{self.current_scenario['topic']} {user_decision}"
        retrieved_chunks = self.rag.retrieve(query, top_k=top_k)
        context = self.rag.format_context(retrieved_chunks)
        
        # Generar evaluación
        eval_prompt = self.generate_evaluation_prompt(
            self.current_scenario['text'],
            user_decision,
            context
        )
        
        print(f"🤖 Analizando según DTF-13...")
        
        response = self.rag.model.generate_content(
            eval_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.4)
        )
        
        return {
            'evaluation': response.text,
            'scenario': self.current_scenario['text'],
            'user_decision': user_decision,
            'agent': self.agent_name
        }
    
    def print_scenario(self, result: Dict):
        """Imprime el escenario"""
        print("\n" + "="*70)
        print(f"🎭 {result['agent'].upper()} - ESCENARIO OPERATIVO")
        print("="*70)
        print(result['scenario'])
        if result.get('source') == 'external':
            print("\n" + result.get('disclaimer', ''))
        print("\n" + "="*70)
    
    def print_evaluation(self, result: Dict):
        """Imprime la evaluación"""
        print("\n" + "="*70)
        print(f"⚖️ EVALUACIÓN DE TU DECISIÓN")
        print("="*70)
        print(f"\n🔍 TU DECISIÓN: {result['user_decision']}")
        print(f"\n📊 ANÁLISIS:\n{result['evaluation']}")
        print("\n" + "="*70)