"""
Agente DIRECTOR para SARFIRE-RAG
Clasifica intención y enruta al agente apropiado
"""
from typing import Dict, Literal
import google.generativeai as genai
import os
from dotenv import load_dotenv


AgentType = Literal["formador", "simulador", "ambiguo"]


class DirectorAgent:
    """Agente que clasifica intención y enruta a agentes especializados"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Args:
            model_name: Modelo de Gemini para clasificación
        """
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY no encontrada en .env")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.agent_name = "Agente Director"
    
    def classify_intent(self, user_query: str) -> Dict:
        """
        Clasifica la intención del usuario
        
        Args:
            user_query: Consulta del usuario
            
        Returns:
            Dict con el agente recomendado y confianza
        """
        classification_prompt = f"""Eres un CLASIFICADOR DE INTENCIONES para un sistema de formación de bomberos.

Tu misión es determinar qué tipo de asistencia necesita el usuario:

**AGENTE FORMADOR** - Usa cuando el usuario:
- Hace preguntas sobre conceptos, definiciones, procedimientos
- Quiere aprender o entender algo
- Pregunta "qué es", "cómo funciona", "explícame", "cuáles son"
- Busca información o formación teórica

**AGENTE SIMULADOR** - Usa cuando el usuario:
- Quiere practicar con casos reales
- Pide escenarios, simulacros o ejercicios
- Dice "ponme a prueba", "genera un caso", "quiero practicar"
- Busca evaluación de decisiones

**AMBIGUO** - Usa solo si realmente no está claro

CONSULTA DEL USUARIO:
"{user_query}"

RESPONDE EXACTAMENTE en este formato (solo una palabra):
AGENTE: [formador|simulador|ambiguo]
CONFIANZA: [alta|media|baja]
RAZÓN: [breve explicación en una línea]
"""
        
        try:
            response = self.model.generate_content(
                classification_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                )
            )
            
            # Parsear respuesta
            text = response.text.lower()
            
            # Extraer agente
            if "formador" in text:
                agent = "formador"
            elif "simulador" in text:
                agent = "simulador"
            else:
                agent = "ambiguo"
            
            # Extraer confianza
            if "alta" in text:
                confidence = "alta"
            elif "media" in text:
                confidence = "media"
            else:
                confidence = "baja"
            
            # Extraer razón
            lines = response.text.split('\n')
            reason = ""
            for line in lines:
                if 'razón' in line.lower() or 'razon' in line.lower():
                    reason = line.split(':', 1)[-1].strip()
                    break
            
            if not reason:
                reason = "Clasificación basada en el contenido de la consulta"
            
            return {
                'agent': agent,
                'confidence': confidence,
                'reason': reason,
                'query': user_query
            }
            
        except Exception as e:
            print(f"⚠️ Error en clasificación: {e}")
            # Fallback: palabras clave simples
            query_lower = user_query.lower()
            
            simulator_keywords = ['escenario', 'simulacro', 'caso', 'práctica', 'prueba', 'ejercicio']
            formador_keywords = ['qué es', 'cómo', 'explica', 'define', 'cuál', 'cuáles']
            
            if any(kw in query_lower for kw in simulator_keywords):
                return {
                    'agent': 'simulador',
                    'confidence': 'media',
                    'reason': 'Detección por palabras clave (fallback)',
                    'query': user_query
                }
            else:
                return {
                    'agent': 'formador',
                    'confidence': 'media',
                    'reason': 'Agente por defecto (fallback)',
                    'query': user_query
                }
    
    def route(self, user_query: str) -> Dict:
        """
        Clasifica y retorna información de enrutamiento
        
        Args:
            user_query: Consulta del usuario
            
        Returns:
            Dict con clasificación y recomendación
        """
        print(f"\n🎯 {self.agent_name} analizando consulta...")
        
        classification = self.classify_intent(user_query)
        
        agent_emoji = {
            'formador': '🎓',
            'simulador': '🎭',
            'ambiguo': '❓'
        }
        
        print(f"{agent_emoji[classification['agent']]} Agente recomendado: {classification['agent'].upper()}")
        print(f"   Confianza: {classification['confidence']}")
        print(f"   Razón: {classification['reason']}")
        
        return classification


class OrchestrationSystem:
    """Sistema completo de orquestación con Director + Agentes"""
    
    def __init__(self, formador_agent, simulador_agent, director_agent=None):
        """
        Args:
            formador_agent: Instancia de FormadorAgent
            simulador_agent: Instancia de SimuladorAgent
            director_agent: Instancia de DirectorAgent
        """
        self.formador = formador_agent
        self.simulador = simulador_agent
        self.director = director_agent or DirectorAgent()
    
    def process_query(self, user_query: str, force_agent: str = None) -> Dict:
        """
        Procesa consulta con enrutamiento automático
        
        Args:
            user_query: Consulta del usuario
            force_agent: Forzar agente específico
            
        Returns:
            Dict con respuesta del agente apropiado
        """
        if force_agent:
            print(f"\n⚡ Modo manual: usando {force_agent.upper()}")
            if force_agent == 'formador':
                return self.formador.process_query(user_query)
            elif force_agent == 'simulador':
                return self.simulador.create_scenario(user_query)
            else:
                return {'error': f'Agente desconocido: {force_agent}'}
        
        # Clasificación automática
        classification = self.director.route(user_query)
        
        # Enrutar
        if classification['agent'] == 'formador':
            result = self.formador.process_query(user_query)
            result['classification'] = classification
            return result
        
        elif classification['agent'] == 'simulador':
            result = self.simulador.create_scenario(user_query)
            result['classification'] = classification
            return result
        
        else:  # ambiguo
            print("\n⚠️ Consulta ambigua - usando Agente FORMADOR por defecto")
            result = self.formador.process_query(user_query)
            result['classification'] = classification
            return result