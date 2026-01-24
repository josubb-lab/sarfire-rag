"""
Agente SIMULADOR para SARFIRE-RAG v2.0
Especializado en generar escenarios operativos y evaluar decisiones
Ahora con soporte de fallback a fuentes externas
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
    
    def create_scenario(
        self, 
        topic: str = None, 
        top_k: int = 5,
        allow_external: bool = False  # ← NUEVO parámetro
    ) -> Dict:
        """
        Crea un escenario operativo
        
        Args:
            topic: Tema específico (opcional, si no se da genera uno general)
            top_k: Chunks a recuperar
            allow_external: Si permite usar fuentes externas para el escenario
            
        Returns:
            Dict con el escenario generado
        """
        print(f"\n🎭 {self.agent_name} activado")
        
        # Si no hay tema, usar query general
        if not topic:
            topic = "Genera un escenario operativo de incendio forestal con dilemas de toma de decisiones"
        
        print(f"🎬 Creando escenario sobre: {topic[:60]}...")
        
        # 1. Obtener contexto usando RAG query (con fallback si allow_external=True)
        rag_result = self.rag.query(
            question=topic,
            top_k=top_k,
            allow_external=True if allow_external else False  # Forzar True o False, no None
        )
        
        # 2. Extraer contexto según la fuente
        if rag_result.get('source') == 'external':
            print(f"🌐 Usando fuentes EXTERNAS para el escenario")
            
            # Formatear contexto externo
            if hasattr(self.rag, 'format_external_context'):
                context = self.rag.format_external_context({
                    'success': True,
                    'results': rag_result.get('external_sources', [])
                })
            else:
                # Fallback manual si no existe el método
                context = "INFORMACIÓN DE FUENTES EXTERNAS:\n\n"
                for i, source in enumerate(rag_result.get('external_sources', []), 1):
                    context += f"[{i}] {source['title']}\n{source['content'][:300]}...\n\n"
            
            source_type = 'external'
            
        elif rag_result.get('sources'):
            print(f"✅ Usando fuentes INTERNAS (DTF-13)")
            
            # Usar formato interno
            retrieved_chunks = rag_result['sources']
            context = self.rag.format_context(retrieved_chunks)
            source_type = 'internal'
            
        else:
            return {
                'scenario': "No se pudo generar el escenario por falta de información.",
                'agent': self.agent_name,
                'topic': topic
            }
        
        # 3. Generar escenario con prompt adecuado
        if source_type == 'external':
            # Prompt especial para fuentes externas
            scenario_prompt = f"""Eres un INSTRUCTOR DE SIMULACROS creando un escenario de entrenamiento.

⚠️ IMPORTANTE: Estás usando FUENTES EXTERNAS (no del DTF-13 oficial).
Genera un escenario basado en esta información, pero:
- Indica claramente que debe ser validado con protocolos oficiales
- Usa la información externa de manera educativa
- Mantén el enfoque en dilemas de toma de decisiones

INFORMACIÓN DISPONIBLE:
{context}

TEMA DEL ESCENARIO:
{topic}

GENERA EL ESCENARIO (150-250 palabras):
Estructura:
1. Situación inicial
2. Evolución del incidente
3. Dilema operativo
4. "¿Qué harías tú?"

⚠️ AL FINAL añade: "Nota: Este escenario usa información de fuentes externas y debe ser validado con protocolos oficiales."
"""
        else:
            # Usar prompt normal con DTF-13
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
            'source': source_type,
            'sources': rag_result.get('sources', [])[:3] if source_type == 'internal' else rag_result.get('external_sources', [])[:3]
        }
        
        # Preparar resultado
        result = {
            'scenario': response.text,
            'agent': self.agent_name,
            'topic': topic,
            'source': source_type  # ← NUEVO: Indicar fuente
        }
        
        # Añadir fuentes según el tipo
        if source_type == 'internal':
            result['sources'] = [
                {
                    'filename': chunk.get('filename', 'DTF-13'),
                    'page': chunk.get('page', chunk.get('page_num', '?'))
                }
                for chunk in rag_result.get('sources', [])[:3]
            ]
        else:
            result['external_sources'] = rag_result.get('external_sources', [])[:3]
            result['disclaimer'] = '⚠️ Escenario basado en fuentes externas - Validar con DTF-13'
        
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
        
        # Mostrar fuente si es externa
        if result.get('source') == 'external':
            print("\n" + result.get('disclaimer', ''))
        
        print("\n" + "="*70)
    
    def print_evaluation(self, result: Dict):
        """Imprime la evaluación de forma estructurada"""
        print("\n" + "="*70)
        print(f"⚖️ EVALUACIÓN DE TU DECISIÓN")
        print("="*70)
        print(f"\n🔍 TU DECISIÓN FUE:")
        print(f"{result['user_decision']}")
        print(f"\n📊 ANÁLISIS:")
        print(result['evaluation'])
        print("\n" + "="*70)
