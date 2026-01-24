"""
Agente SIMULADOR para SARFIRE-RAG
Especializado en generar escenarios operativos y evaluar decisiones
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
    
    def generate_scenario_prompt(self, topic: str, context: str) -> str:
        """
        Genera prompt para crear escenario
        
        Args:
            topic: Tema del escenario
            context: Contexto del manual
            
        Returns:
            Prompt para generación de escenario
        """
        prompt = f"""Eres un INSTRUCTOR DE SIMULACROS del Consorcio Provincial de Bomberos de Valencia.

Tu misión es crear ESCENARIOS OPERATIVOS REALISTAS para entrenar al personal.

ESTRUCTURA DEL ESCENARIO:
1. **SITUACIÓN INICIAL**: Describe el incendio (ubicación, extensión, condiciones meteorológicas, recursos disponibles)
2. **EVOLUCIÓN**: Qué está pasando en este momento crítico
3. **TU DECISIÓN**: Plantea 3-4 opciones de actuación (algunas correctas, otras con riesgos)
4. **PREGUNTA**: "¿Qué harías tú?"

REGLAS:
- Basado ESTRICTAMENTE en protocolos del DTF-13
- Escenario realista y detallado (usa datos concretos: hectáreas, velocidad viento, etc.)
- Opciones deben ser plausibles (no obvias)
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
        """
        Genera prompt para evaluar decisión
        
        Args:
            scenario: Escenario planteado
            user_decision: Decisión del usuario
            context: Contexto del manual
            
        Returns:
            Prompt para evaluación
        """
        prompt = f"""Eres un EVALUADOR EXPERTO de simulacros del Consorcio Provincial de Bomberos de Valencia.

Tu misión es evaluar la decisión del usuario según los protocolos del DTF-13.

ESTRUCTURA DE LA EVALUACIÓN:
1. **ANÁLISIS DE LA DECISIÓN**: ¿Qué implicaciones tiene?
2. **ACIERTOS**: Qué aspectos son correctos según el DTF-13 (cita secciones)
3. **RIESGOS/ERRORES**: Qué problemas puede causar (si los hay)
4. **DECISIÓN ÓPTIMA**: Cuál sería la mejor actuación según el protocolo
5. **LECCIONES APRENDIDAS**: 2-3 puntos clave para recordar

TONO:
- Constructivo y formativo
- Reconoce aciertos antes de señalar errores
- Explica el "por qué" de cada punto
- Cita secciones del DTF-13 cuando sea relevante

CONTEXTO DEL MANUAL DTF-13:
{context}

ESCENARIO PLANTEADO:
{scenario}

DECISIÓN DEL USUARIO:
{user_decision}

EVALUACIÓN:
"""
        return prompt
    
    def create_scenario(self, topic: str = None, top_k: int = 5, allow_external: bool = False) -> Dict:
        """
        Crea un escenario operativo
        
        Args:
            topic: Tema específico (opcional, si no se da genera uno general)
            top_k: Chunks a recuperar
            
        Returns:
            Dict con el escenario generado
        """
        print(f"\n🎭 {self.agent_name} activado")
        
        # Si no hay tema, usar query general
        if not topic:
            topic = "Genera un escenario operativo de incendio forestal con dilemas de toma de decisiones"
        
        print(f"🎬 Creando escenario sobre: {topic[:60]}...")
        
        # 1. Retrieve contexto relevante
        retrieved_chunks = self.rag.retrieve(topic, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                'scenario': "No se pudo generar el escenario por falta de información en el manual.",
                'agent': self.agent_name,
                'topic': topic
            }
        
        print(f"✅ Contexto recuperado ({len(retrieved_chunks)} fragmentos)")
        
        # 2. Format context
        context = self.rag.format_context(retrieved_chunks)
        
        # 3. Generate escenario
        scenario_prompt = self.generate_scenario_prompt(topic, context)
        
        print(f"🤖 Generando escenario operativo...")
        
        response = self.rag.model.generate_content(
            scenario_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # Más creatividad para escenarios variados
            )
        )
        
        # Guardar escenario actual
        self.current_scenario = {
            'text': response.text,
            'topic': topic,
            'sources': retrieved_chunks[:3]
        }
        
        result = {
            'scenario': response.text,
            'agent': self.agent_name,
            'topic': topic,
            'sources': [
                {
                    'filename': chunk['metadata']['filename'],
                    'page': chunk['metadata']['page_num']
                }
                for chunk in retrieved_chunks[:3]
            ]
        }
        
        return result
    
    def evaluate_decision(self, user_decision: str, top_k: int = 5) -> Dict:
        """
        Evalúa la decisión del usuario
        
        Args:
            user_decision: Decisión tomada por el usuario
            top_k: Chunks a recuperar
            
        Returns:
            Dict con la evaluación
        """
        if not self.current_scenario:
            return {
                'evaluation': "No hay un escenario activo para evaluar.",
                'agent': self.agent_name
            }
        
        print(f"\n⚖️ Evaluando tu decisión...")
        
        # Recuperar contexto (puede ser diferente según la decisión)
        query = f"{self.current_scenario['topic']} {user_decision}"
        retrieved_chunks = self.rag.retrieve(query, top_k=top_k)
        context = self.rag.format_context(retrieved_chunks)
        
        # Generate evaluación
        eval_prompt = self.generate_evaluation_prompt(
            self.current_scenario['text'],
            user_decision,
            context
        )
        
        print(f"🤖 Analizando según protocolos DTF-13...")
        
        response = self.rag.model.generate_content(
            eval_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,  # Balance entre creatividad y precisión
            )
        )
        
        result = {
            'evaluation': response.text,
            'scenario': self.current_scenario['text'],
            'user_decision': user_decision,
            'agent': self.agent_name
        }
        
        return result
    
    def print_scenario(self, result: Dict):
        """Imprime el escenario de forma estructurada"""
        print("\n" + "="*70)
        print(f"🎭 {result['agent'].upper()} - ESCENARIO OPERATIVO")
        print("="*70)
        print(result['scenario'])
        print("\n" + "="*70)
    
    def print_evaluation(self, result: Dict):
        """Imprime la evaluación de forma estructurada"""
        print("\n" + "="*70)
        print(f"⚖️ EVALUACIÓN DE TU DECISIÓN")
        print("="*70)
        print(f"\n📝 TU DECISIÓN FUE:")
        print(f"{result['user_decision']}")
        print(f"\n📊 ANÁLISIS:")
        print(result['evaluation'])
        print("\n" + "="*70)