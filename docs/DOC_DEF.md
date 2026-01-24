\# 🔥 SARFIRE-RAG - DOCUMENTO MAESTRO

\## Asistente Inteligente para Emergencias basado en RAG



\*\*Proyecto:\*\* Capstone IIA (Enero 2026) + TFM MDATA (Abril 2026)  

\*\*Autor:\*\* Josué - Bombero Profesional + Data Scientist  

\*\*Fecha Inicio:\*\* 14 Enero 2026  

\*\*Fecha Actual:\*\* 20 Enero 2026 (Día 7)  

\*\*Estado:\*\* MVP Funcional - Fase de Consolidación



---



\## 📋 ÍNDICE



1\. \[Resumen Ejecutivo](#resumen-ejecutivo)

2\. \[Objetivos del Proyecto](#objetivos-del-proyecto)

3\. \[Arquitectura del Sistema](#arquitectura-del-sistema)

4\. \[Stack Tecnológico](#stack-tecnológico)

5\. \[Decisiones Técnicas](#decisiones-técnicas)

6\. \[Progreso Actual](#progreso-actual)

7\. \[Plan de Entrega](#plan-de-entrega)

8\. \[Métricas de Calidad](#métricas-de-calidad)

9\. \[Limitaciones y Mejoras Futuras](#limitaciones-y-mejoras-futuras)

10\. \[Guía de Uso](#guía-de-uso)



---



\## 1. RESUMEN EJECUTIVO



\### 1.1 Concepto



SARFIRE-RAG es un asistente inteligente especializado en emergencias que utiliza Retrieval-Augmented Generation (RAG) para transformar manuales técnicos y protocolos de bomberos en:



\- ✅ \*\*Respuestas operativas\*\* basadas en documentación oficial

\- ✅ \*\*Formación automatizada\*\* personalizada y contextualizada

\- ✅ \*\*Simulacros interactivos\*\* para entrenamiento práctico

\- ✅ \*\*Evaluación de decisiones\*\* en tiempo real



\### 1.2 Problema que Resuelve



Los servicios de emergencias tienen gran cantidad de documentación técnica (protocolos, manuales, procedimientos) que:

\- Está dispersa y es difícil de consultar en tiempo real

\- Requiere experiencia para interpretar correctamente

\- No existe herramienta que genere formación o escenarios automáticos



\*\*SARFIRE-RAG\*\* convierte esta documentación en conocimiento accionable, accesible y contextualizado.



\### 1.3 Propuesta de Valor



\- \*\*Para bomberos:\*\* Consulta rápida de protocolos + entrenamiento con casos prácticos

\- \*\*Para formadores:\*\* Generación automática de material didáctico y simulacros

\- \*\*Para el sector:\*\* Demostración de cómo IA puede mejorar preparación operativa



---



\## 2. OBJETIVOS DEL PROYECTO



\### 2.1 Objetivo Principal



> Desarrollar un asistente RAG funcional, multiformato y especializado en emergencias, que convierta documentación técnica en conocimiento aplicable, preciso y estructurado.



\### 2.2 Objetivos Específicos



\#### ✅ Procesamiento de Documentación Técnica

\- Integrar documentos en PDF (extensible a DOCX, TXT, CSV)

\- Construir pipeline RAG con recuperación contextual

\- Chunking inteligente para mantener coherencia



\#### ✅ Generación de Contenido Operativo

\- Responder preguntas técnicas basadas en protocolos reales

\- Crear automáticamente casos prácticos y simulacros

\- Elaborar recomendaciones accionables



\#### ✅ Agentes Especializados

\- \*\*Agente Formador:\*\* Explicaciones técnicas pedagógicas

\- \*\*Agente Simulador:\*\* Generación y evaluación de escenarios

\- \*\*Agente Director:\*\* Clasificación automática de intenciones



\#### ⚠️ Complementario (NO central)

\- Integración opcional de predicción de riesgo

\- Fallback a fuentes externas (Tavily API)



---



\## 3. ARQUITECTURA DEL SISTEMA



\### 3.1 Diagrama de Arquitectura



```mermaid

flowchart TB

&nbsp;   subgraph FRONTEND\["🖥️ FRONTEND (Gradio)"]

&nbsp;       UI\[Chat Interface]

&nbsp;       MODE{Selector de Modo}

&nbsp;       SIM\[🎮 Simulador]

&nbsp;       FORM\[📚 Formador]

&nbsp;       AUTO\[🎯 Automático]

&nbsp;   end



&nbsp;   subgraph BACKEND\["⚙️ BACKEND"]

&nbsp;       subgraph ORCHESTRATION\["Orquestación"]

&nbsp;           DIRECTOR\[🎯 Agente Director<br/>Clasifica intención]

&nbsp;       end

&nbsp;       

&nbsp;       subgraph AGENTS\["Agentes Especializados"]

&nbsp;           AG\_SIM\[🎭 Agente Simulador<br/>Escenarios + Evaluación]

&nbsp;           AG\_FORM\[🎓 Agente Formador<br/>Explicaciones + Consultas]

&nbsp;       end

&nbsp;   end



&nbsp;   subgraph RAG\_SYSTEM\["🔍 RAG"]

&nbsp;       RETRIEVER\[Retriever<br/>Búsqueda Híbrida]

&nbsp;       

&nbsp;       subgraph MEMORY\["💾 Memoria"]

&nbsp;           SHORT\[Memoria de Sesión<br/>gr.State()]

&nbsp;       end

&nbsp;   end



&nbsp;   subgraph VECTORSTORE\["📦 Vector Store"]

&nbsp;       COLL1\[(ChromaDB<br/>66 chunks<br/>DTF-13)]

&nbsp;   end



&nbsp;   UI --> MODE

&nbsp;   MODE --> SIM

&nbsp;   MODE --> FORM

&nbsp;   MODE --> AUTO

&nbsp;   

&nbsp;   SIM --> AG\_SIM

&nbsp;   FORM --> AG\_FORM

&nbsp;   AUTO --> DIRECTOR

&nbsp;   

&nbsp;   DIRECTOR -->|Enruta| AG\_SIM

&nbsp;   DIRECTOR -->|Enruta| AG\_FORM

&nbsp;   

&nbsp;   AG\_SIM -->|Consulta| RETRIEVER

&nbsp;   AG\_FORM -->|Consulta| RETRIEVER

&nbsp;   

&nbsp;   RETRIEVER <--> VECTORSTORE

&nbsp;   RAG\_SYSTEM <--> MEMORY

&nbsp;   

&nbsp;   classDef frontend fill:#4CAF50,color:white

&nbsp;   classDef backend fill:#2196F3,color:white

&nbsp;   classDef rag fill:#FF9800,color:white

&nbsp;   classDef storage fill:#9C27B0,color:white

&nbsp;   

&nbsp;   class UI,MODE,SIM,FORM,AUTO frontend

&nbsp;   class DIRECTOR,AG\_SIM,AG\_FORM backend

&nbsp;   class RETRIEVER,SHORT rag

&nbsp;   class COLL1 storage

```



\### 3.2 Flujo de Operación



\#### Modo Formador

```

Usuario: "¿Qué es el PMA?"

&nbsp;   ↓

Retriever busca en ChromaDB (híbrida: semántica + keywords)

&nbsp;   ↓

Agente Formador genera explicación pedagógica

&nbsp;   ↓

Respuesta con fuentes citadas

```



\#### Modo Simulador

```

Usuario: "Genera escenario de incendio nocturno"

&nbsp;   ↓

Agente Simulador consulta RAG para contexto realista

&nbsp;   ↓

Genera escenario basado en protocolos

&nbsp;   ↓

Estado: {active: True, text: escenario}

&nbsp;   ↓

Usuario: "Activaría nivel 2 y solicitaría medios aéreos"

&nbsp;   ↓

Agente Simulador recibe: ESCENARIO + DECISIÓN

&nbsp;   ↓

Evalúa decisión según DTF-13

&nbsp;   ↓

Respuesta con análisis + aspectos positivos/negativos

```



\#### Modo Automático

```

Usuario: "..."

&nbsp;   ↓

Director clasifica intención

&nbsp;   ↓

Enruta al agente apropiado

&nbsp;   ↓

Agente procesa y responde

```



---



\## 4. STACK TECNOLÓGICO



\### 4.1 Stack Completo



| Componente | Tecnología | Versión | Justificación |

|------------|------------|---------|---------------|

| \*\*LLM Principal\*\* | Gemini 2.0 Flash | Latest | Balance velocidad/calidad, API gratuita |

| \*\*Vector Store\*\* | ChromaDB | 0.4.x | Persistente, simple, suficiente para MVP |

| \*\*Embeddings\*\* | sentence-transformers | all-MiniLM-L6-v2 | Ligero (384 dim), rápido, open-source |

| \*\*Framework RAG\*\* | LangChain | 0.1.x | Ecosistema maduro, buena documentación |

| \*\*Orquestación\*\* | LangGraph | 0.0.x | Gestión de estados para multi-agente |

| \*\*Frontend\*\* | Gradio | 6.3.0 | Rápido prototipado, built-in chat |

| \*\*Procesamiento\*\* | pypdf | 3.x | Simple, efectivo para PDFs |

| \*\*Python\*\* | 3.11+ | - | Última versión estable |



\### 4.2 Dependencias Clave



```python

\# requirements.txt (esenciales)

langchain>=0.1.0

langchain-google-genai>=1.0.0

chromadb>=0.4.22

sentence-transformers>=2.2.0

gradio>=6.0.0

pypdf>=3.17.0

python-dotenv>=1.0.0

```



\### 4.3 Infraestructura



\- \*\*Desarrollo:\*\* Local (Windows 11 + VSCode)

\- \*\*Versionado:\*\* Git + GitHub

\- \*\*Deploy:\*\* Hugging Face Spaces (pendiente)

\- \*\*Datos:\*\* ChromaDB persistente en `/data/processed/`



---



\## 5. DECISIONES TÉCNICAS



\### 5.1 Documento Único vs Múltiples PDFs



\*\*DECISIÓN:\*\* Usar 1 PDF limpio (DTF-13, 31 páginas)



\*\*Contexto:\*\*

\- Inicialmente se probó con 7 PDFs (154 páginas) → 491 chunks

\- Problemas: duplicados, contexto fragmentado, respuestas inconsistentes



\*\*Justificación:\*\*

\- ✅ Chunking más coherente (66 chunks vs 491)

\- ✅ Respuestas más precisas y consistentes

\- ✅ Más fácil de evaluar para Capstone

\- ✅ Suficiente para demostrar capacidades del sistema



\*\*Trade-off Aceptado:\*\*

\- ❌ Menos cobertura temática

\- ✅ Mayor calidad en las respuestas



\### 5.2 Búsqueda Híbrida (Semántica + Keywords)



\*\*DECISIÓN:\*\* 40% semántica + 60% keywords



\*\*Justificación:\*\*

\- Embeddings generales (all-MiniLM-L6-v2) no especializados en terminología bomberos

\- Keywords capturan términos técnicos específicos (ej: "Foëhn", "PMA", "UBF")

\- Búsqueda híbrida compensa debilidad de embeddings generales



\*\*Alternativa Considerada (descartada por MVP):\*\*

\- Embeddings especializados españoles: paraphrase-multilingual-mpnet-base-v2

\- Razón: Más pesado, no crítico para demostración



\### 5.3 Chunking Strategy



\*\*Configuración:\*\*

\- Tamaño: 1500 caracteres

\- Overlap: 150 caracteres (10%)

\- Separadores: Párrafos → Puntos → Espacios



\*\*Justificación:\*\*

\- Balance entre contexto (suficiente para LLM) y granularidad (recuperación precisa)

\- Overlap evita pérdida de información en fronteras

\- Resultado: 66 chunks con tamaño medio 1019 caracteres



\### 5.4 Arquitectura de Agentes



\*\*DECISIÓN:\*\* 2 agentes especializados + 1 director



\*\*Agentes:\*\*

1\. \*\*Formador:\*\* Consultas técnicas, explicaciones pedagógicas

2\. \*\*Simulador:\*\* Generación de escenarios + evaluación de decisiones

3\. \*\*Director:\*\* Clasificación simple de intención (formación vs simulacro)



\*\*Alternativas Descartadas:\*\*

\- ❌ Agente Evaluador (tests) → Complejidad innecesaria para MVP

\- ❌ 5+ agentes especializados → Over-engineering



\*\*Justificación:\*\*

\- Suficiente para demostrar capacidades multi-agente

\- Cada agente tiene propósito claro y diferenciado

\- Escalable a más agentes en futuras versiones



\### 5.5 Gestión de Estado en Gradio



\*\*PROBLEMA ORIGINAL:\*\*

\- Variable global `current\_scenario = {}` no persistía entre mensajes

\- Evaluaciones sin contexto del escenario activo



\*\*SOLUCIÓN:\*\*

```python

scenario\_state = gr.State({"text": None, "active": False})

```



\*\*Justificación:\*\*

\- `gr.State()` persiste entre interacciones

\- Se pasa como input/output en event handlers

\- Permite flujo: Generar escenario → Guardar estado → Evaluar decisión



\### 5.6 API Rate Limits (Gemini)



\*\*LIMITACIÓN:\*\* 20 requests/día en tier gratuito



\*\*MITIGACIÓN:\*\*

\- Desarrollo incremental (reseteo diario)

\- Testing selectivo de funcionalidades

\- Plan para tier pago si necesario pre-deploy



\*\*Alternativa Futura:\*\*

\- Claude API (mayor límite, pero de pago desde inicio)



---



\## 6. PROGRESO ACTUAL



\### 6.1 Timeline Real



| Día | Fecha | Objetivo | Estado | Notas |

|-----|-------|----------|--------|-------|

| \*\*1\*\* | 14 Ene | Setup + Ingesta | ✅ | 7 PDFs → 154 páginas procesadas |

| \*\*2\*\* | 15 Ene | RAG Básico | ✅ | ChromaDB + búsqueda híbrida |

| \*\*3\*\* | 16 Ene | Multi-Agente | ✅ | 3 agentes implementados |

| \*\*4\*\* | 17 Ene | Testing | ⚠️ | Límite API alcanzado |

| \*\*5\*\* | 18 Ene | - | - | Día no laborable |

| \*\*6\*\* | 19 Ene | - | - | Día no laborable |

| \*\*7\*\* | 20 Ene | UI Gradio | ✅ | Interfaz funcional |



\### 6.2 Componentes Completados



\#### ✅ FASE 1: Fundamentos (Días 1-2)

\- \[x] Estructura del proyecto

\- \[x] Loader de PDFs (pypdf)

\- \[x] Chunking (RecursiveCharacterTextSplitter)

\- \[x] Embeddings (all-MiniLM-L6-v2)

\- \[x] ChromaDB configurado (66 chunks finales)

\- \[x] Pipeline RAG básico con Gemini

\- \[x] Búsqueda híbrida (40% semántica + 60% keywords)



\#### ✅ FASE 2: Sistema Multi-Agente (Días 3-4)

\- \[x] Agente Formador (consultas + explicaciones)

\- \[x] Agente Simulador (escenarios + evaluación)

\- \[x] Agente Director (clasificación de intención)

\- \[x] Sistema de orquestación (LangGraph básico)



\#### ✅ FASE 4: Frontend (Día 7)

\- \[x] Interfaz Gradio con chat

\- \[x] Selector de 3 modos (Formador/Simulador/Automático)

\- \[x] Gestión de estado persistente (gr.State)

\- \[x] Formateo de respuestas por agente

\- \[x] Citación de fuentes



\### 6.3 Componentes Pendientes



\#### ⏳ FASE 3: Mejoras RAG (Opcional para Capstone)

\- \[ ] Scoring de relevancia explícito (umbral 0.5)

\- \[ ] Respuesta "no tengo info" controlada

\- \[ ] Fallback a Tavily API (fuentes externas)

\- \[ ] Memoria conversacional persistente (LangGraph)



\#### ⏳ FASE 5: Deploy y Documentación

\- \[ ] Polish UX (loading indicators, estilos)

\- \[ ] Deploy a Hugging Face Spaces

\- \[ ] README narrativo completo

\- \[ ] Video demo (2-3 minutos)

\- \[ ] Verificación requisitos Capstone/TFM



---



\## 7. PLAN DE ENTREGA



\### 7.1 Entregas Duales



\#### 🎯 CAPSTONE IIA (Enero 2026) - PRIORIDAD 1

\*\*Fecha límite:\*\* ~30 Enero (10 días restantes)



\*\*Enfoque:\*\* Demostración funcional



\*\*Entregables Mínimos:\*\*

\- ✅ Sistema RAG multiformato funcionando

\- ✅ 3 agentes operativos con UI

\- ⏳ Deploy público (Hugging Face Spaces)

\- ⏳ Video demo (2-3 min)

\- ⏳ README básico explicativo



\*\*Criterios de Éxito:\*\*

\- Sistema funciona end-to-end sin errores

\- Los 3 modos operan correctamente

\- Video muestra flujos completos

\- Código accesible públicamente



---



\#### 📚 TFM MDATA (Abril 2026) - PRIORIDAD 2

\*\*Fecha límite:\*\* Abril 2026 (3 meses adicionales)



\*\*Enfoque:\*\* Documentación académica + análisis



\*\*Estructura TFM:\*\*

\- \*\*Proyecto 1 (60%):\*\* Modelización Predictiva \[OTRO PROYECTO]

\- \*\*Proyecto 2 (40%):\*\* SARFIRE-RAG \[ESTE]



\*\*Entregables Proyecto 2:\*\*

\- README narrativo (no Q\&A)

\- Decisiones técnicas justificadas

\- Análisis de resultados y métricas

\- Limitaciones documentadas

\- Mejoras futuras propuestas

\- Esquema visual del sistema



\*\*Mejoras Opcionales para TFM:\*\*

\- Fallback a fuentes externas

\- Análisis de relevancia avanzado

\- Comparativa de embeddings

\- Testing exhaustivo con casos reales



---



\### 7.2 Roadmap Próximos 10 Días



```

DÍAS 8-9: POLISH + DEPLOY

├─ Día 8: UX Polish

│  ├─ Loading indicators (30 min)

│  ├─ Estilos visuales mejorados (1h)

│  ├─ Botón limpiar visible (15 min)

│  └─ Testing 3 modos (30 min)

│

└─ Día 9: Deploy HF Spaces

&nbsp;  ├─ requirements.txt limpio (30 min)

&nbsp;  ├─ Configurar secrets (1h)

&nbsp;  ├─ Deploy inicial (2h)

&nbsp;  └─ Debug errores (2h)



DÍAS 10-12: DOCUMENTACIÓN

├─ Día 10: README Narrativo

│  ├─ Historia del proyecto (1h)

│  ├─ Arquitectura explicada (1h)

│  ├─ Screenshots (30 min)

│  └─ Instrucciones uso (30 min)

│

├─ Día 11-12: Video Demo

│  ├─ Script del video (1h)

│  ├─ Grabación 3 modos (1h)

│  ├─ Edición básica (2h)

│  └─ Subida y preparación (30 min)



DÍAS 13-17: BUFFER

└─ Testing exhaustivo

&nbsp;  ├─ Corrección bugs finales

&nbsp;  ├─ Preparar presentación

&nbsp;  └─ Reserva para imprevistos

```



---



\## 8. MÉTRICAS DE CALIDAD



\### 8.1 Métricas Técnicas



| Métrica | Valor | Objetivo | Estado |

|---------|-------|----------|--------|

| \*\*Chunks generados\*\* | 66 | - | ✅ |

| \*\*Tamaño medio chunk\*\* | 1019 chars | ~1000 | ✅ |

| \*\*Documentos procesados\*\* | 1 PDF (31 pág) | 1+ | ✅ |

| \*\*Tiempo respuesta\*\* | <10 seg | <10 seg | ✅ |

| \*\*Citación fuentes\*\* | Sí | Sí | ✅ |

| \*\*Precisión respuestas RAG\*\* | 7/10 | >6/10 | ✅ |



\### 8.2 Métricas de Funcionalidad



| Funcionalidad | Estado | Validación |

|---------------|--------|------------|

| \*\*Modo Formador\*\* | ✅ | Responde consultas técnicas correctamente |

| \*\*Modo Simulador\*\* | ✅ | Genera escenarios + evalúa decisiones |

| \*\*Modo Automático\*\* | ✅ | Director clasifica y enruta correctamente |

| \*\*Persistencia Estado\*\* | ✅ | Escenarios mantienen contexto entre mensajes |

| \*\*Gestión de Errores\*\* | ✅ | Try-catch en procesos críticos |



\### 8.3 Ejemplos de Salida



\#### Ejemplo 1: Modo Formador

```

Usuario: "¿Qué es el PMA?"



Agente Formador:

El PMA (Puesto de Mando Avanzado) es el punto físico donde se 

establece la dirección técnica de la extinción de un incendio 

forestal. Desde el PMA se coordinan todas las operaciones, se 

gestionan los recursos y se toman las decisiones estratégicas...



Fuentes consultadas:

1\. DTF-13-ORGANIZACION-GESTION-INCENDIOS.pdf (Página 15)

2\. DTF-13-ORGANIZACION-GESTION-INCENDIOS.pdf (Página 18)

```



\#### Ejemplo 2: Modo Simulador

```

Usuario: "Genera escenario de incendio nocturno con viento"



Agente Simulador:

\[Escenario detallado de 200-300 palabras con:

\- Situación inicial (hora, ubicación, condiciones)

\- Recursos disponibles

\- Evolución del incendio

\- Complicaciones]



💡 ¿Qué decisión tomarías en esta situación?



---



Usuario: "Activaría nivel 2 y solicitaría medios aéreos"



Agente Simulador:

⚖️ EVALUACIÓN DE TU DECISIÓN



Tu decisión de activar el nivel 2 es correcta dado que...

\[Análisis de aspectos positivos]

\[Aspectos a mejorar]

\[Recomendaciones según DTF-13]



✅ Evaluación completada.

```



---



\## 9. LIMITACIONES Y MEJORAS FUTURAS



\### 9.1 Limitaciones Conocidas



\#### Técnicas

1\. \*\*Embeddings Generales\*\*

&nbsp;  - Modelo: all-MiniLM-L6-v2 (no especializado en español técnico)

&nbsp;  - Impacto: Requiere búsqueda híbrida para términos específicos

&nbsp;  - Mitigación: Keywords compensan (60% del scoring)



2\. \*\*Corpus Limitado\*\*

&nbsp;  - Solo 1 documento (DTF-13, 31 páginas)

&nbsp;  - Impacto: Cobertura temática reducida

&nbsp;  - Justificación: Suficiente para MVP, mejor calidad



3\. \*\*Límites API\*\*

&nbsp;  - Gemini: 20 requests/día (tier gratuito)

&nbsp;  - Impacto: Desarrollo iterativo, testing selectivo

&nbsp;  - Plan: Upgrade a tier pago pre-deploy



4\. \*\*Memoria No Persistente\*\*

&nbsp;  - Estado solo durante sesión (gr.State)

&nbsp;  - Impacto: No hay historial entre sesiones

&nbsp;  - Plan: LangGraph memory para TFM



\#### Funcionales

5\. \*\*Sin Validación de Fiabilidad\*\*

&nbsp;  - No hay scoring explícito de relevancia (umbral)

&nbsp;  - Responde siempre, incluso si contexto es débil



6\. \*\*Sin Fallback Externo\*\*

&nbsp;  - No integra Tavily API

&nbsp;  - Limitado a conocimiento del DTF-13



\### 9.2 Mejoras Futuras



\#### Corto Plazo (TFM Abril 2026)

\- \[ ] Embeddings multilingües especializados

\- \[ ] Fallback a Tavily API con disclaimer

\- \[ ] Scoring de relevancia (umbral 0.5)

\- \[ ] Memoria conversacional persistente

\- \[ ] Múltiples documentos (5+ PDFs)



\#### Medio Plazo (Post-Académico)

\- \[ ] Fine-tuning de embeddings en terminología bomberos

\- \[ ] Re-ranking con modelo más potente

\- \[ ] Integración con bases de datos operativas

\- \[ ] Módulo de predicción de riesgo

\- \[ ] API REST para integración externa



\#### Largo Plazo (Startup)

\- \[ ] App móvil nativa

\- \[ ] Modo offline con modelos locales

\- \[ ] Multi-tenant para diferentes cuerpos

\- \[ ] Integración con sistemas de despacho

\- \[ ] Generación automática de informes



---



\## 10. GUÍA DE USO



\### 10.1 Instalación Local



```bash

\# 1. Clonar repositorio

git clone https://github.com/tu-usuario/sarfire-rag

cd sarfire-rag



\# 2. Crear entorno virtual

python -m venv venv

source venv/bin/activate  # Linux/Mac

\# venv\\Scripts\\activate   # Windows



\# 3. Instalar dependencias

pip install -r requirements.txt



\# 4. Configurar API key

cp .env.example .env

\# Editar .env y añadir tu GOOGLE\_API\_KEY



\# 5. Ejecutar aplicación

python app.py

```



\### 10.2 Estructura de Archivos



```

sarfire-rag/

├── data/

│   ├── raw/                        # PDFs originales

│   │   └── DTF-13-ORGANIZACION-GESTION-INCENDIOS.pdf

│   └── processed/

│       └── chromadb/               # Base de datos vectorial

│           └── sarfire\_docs/       # Colección con 66 chunks

├── src/

│   ├── loaders/

│   │   └── pdf\_loader.py           # Ingesta de PDFs

│   ├── rag/

│   │   ├── embeddings.py           # Generación embeddings

│   │   ├── vector\_store.py         # ChromaDB wrapper

│   │   └── rag\_pipeline.py         # Pipeline RAG completo

│   └── agents/

│       ├── formador\_agent.py       # Agente Formador

│       ├── simulador\_agent.py      # Agente Simulador

│       ├── director\_agent.py       # Agente Director

│       └── orchestration.py        # Sistema orquestación

├── docs/

│   ├── PROGRESO\_DIARIO.md          # Log de desarrollo

│   ├── DECISIONES\_TECNICAS.md      # Arquitectura y decisiones

│   └── sarfire\_architecture\_v2.mermaid  # Diagrama sistema

├── app.py                          # Interfaz Gradio

├── requirements.txt                # Dependencias Python

├── .env.example                    # Template configuración

└── README.md                       # Este documento

```



\### 10.3 Uso de la Aplicación



\#### Modo 🎓 Formador

\*\*Propósito:\*\* Consultas técnicas y explicaciones



\*\*Ejemplos:\*\*

```

"¿Qué es el PMA?"

"Explica las situaciones operativas en incendios forestales"

"¿Cuáles son los niveles de activación?"

"¿Qué funciones tiene el DTF?"

```



\*\*Comportamiento:\*\*

\- Consulta RAG (ChromaDB)

\- Genera explicación pedagógica

\- Cita fuentes (PDF + página)



---



\#### Modo 🎭 Simulador

\*\*Propósito:\*\* Entrenamiento con escenarios



\*\*Paso 1 - Generar Escenario:\*\*

```

"Genera un escenario de incendio nocturno con viento del oeste"

"Crea un caso de evacuación en zona urbana"

"Dame un simulacro de incendio en terreno montañoso"

```



\*\*Paso 2 - Responder al Escenario:\*\*

```

"Activaría el nivel 2 y solicitaría medios aéreos"

"Establecería perímetro de seguridad y evacuaría la zona"

```



\*\*Comportamiento:\*\*

\- Genera escenario realista basado en DTF-13

\- Guarda estado del escenario

\- Evalúa decisión del usuario con contexto completo

\- Proporciona feedback constructivo



---



\#### Modo 🎯 Automático

\*\*Propósito:\*\* Director decide agente apropiado



\*\*Ejemplos:\*\*

```

"¿Qué es el Foëhn?" → Enruta a Formador

"Genera un caso práctico" → Enruta a Simulador

```



\### 10.4 Flujo de Trabajo Típico



```

1\. Abrir aplicación (http://localhost:7860)

2\. Seleccionar modo según necesidad

3\. Interactuar naturalmente en el chat

4\. Para simulacros:

&nbsp;  a. Solicitar escenario

&nbsp;  b. Leer situación

&nbsp;  c. Responder con decisión

&nbsp;  d. Recibir evaluación

5\. Usar "Limpiar conversación" para reset

```



---



\## 11. EVALUACIÓN ACADÉMICA



\### 11.1 Criterios Capstone IIA



\*\*Categorías de Evaluación:\*\*



1\. \*\*Investigación y Justificación Técnica (3 pts)\*\*

&nbsp;  - ✅ Elección argumentada de herramientas

&nbsp;  - ✅ Comparativa de alternativas (embeddings, chunking)

&nbsp;  - ✅ Decisiones arquitectónicas documentadas



2\. \*\*Resultados y Visualizaciones (2.5 pts)\*\*

&nbsp;  - ✅ Sistema funcional end-to-end

&nbsp;  - ✅ Interfaz demo clara

&nbsp;  - ⏳ Video demo mostrando capacidades



3\. \*\*Calidad del Código (2 pts)\*\*

&nbsp;  - ✅ Código modular y comentado

&nbsp;  - ✅ Estructura clara de proyecto

&nbsp;  - ✅ Requirements.txt completo



4\. \*\*Innovación y Creatividad (1.5 pts)\*\*

&nbsp;  - ✅ Sistema multi-agente para dominio específico

&nbsp;  - ✅ Flujo de simulacros con estado persistente

&nbsp;  - ✅ Búsqueda híbrida custom



5\. \*\*Documentación (1 pt)\*\*

&nbsp;  - ✅ README narrativo (este documento)

&nbsp;  - ✅ Decisiones técnicas explicadas

&nbsp;  - ⏳ Video demo



\*\*Nota Esperada:\*\* 8.5-9.5/10



---



\### 11.2 Criterios TFM MDATA



\*\*Proyecto 2 - Asistente RAG Multiformato (40% nota final)\*\*



\*\*Preguntas a Responder:\*\*



1\. \*\*¿Qué documentos y formatos utilizaste?\*\*

&nbsp;  - 1 PDF (DTF-13, 31 páginas, 66 chunks)

&nbsp;  - Extensible a DOCX, TXT, CSV (arquitectura preparada)



2\. \*\*¿Cómo dividiste y preparaste los datos?\*\*

&nbsp;  - RecursiveCharacterTextSplitter

&nbsp;  - Chunks: 1500 chars, overlap: 150 chars

&nbsp;  - Estrategia híbrida (semántica + keywords)



3\. \*\*¿Qué modelo y vector DB usaste y por qué?\*\*

&nbsp;  - LLM: Gemini 2.0 Flash (balance velocidad/calidad)

&nbsp;  - Embeddings: all-MiniLM-L6-v2 (ligero, suficiente)

&nbsp;  - Vector DB: ChromaDB (persistente, simple)



4\. \*\*¿Cómo evalúas la calidad de las respuestas?\*\*

&nbsp;  - Testing manual con preguntas de control

&nbsp;  - Verificación de citación de fuentes

&nbsp;  - Coherencia con protocolos oficiales



5\. \*\*¿Cuál fue tu propuesta de mejora?\*\*

&nbsp;  - Sistema multi-agente especializado

&nbsp;  - Búsqueda híbrida (40/60 semántica/keywords)

&nbsp;  - Estado persistente para simulacros



6\. \*\*¿Qué aprendiste sobre RAG y limitaciones?\*\*

&nbsp;  - Importancia de chunking coherente

&nbsp;  - Embeddings generales vs especializados

&nbsp;  - Balance corpus (calidad vs cantidad)

&nbsp;  - Gestión de estado en apps conversacionales



---



\## 12. CONCLUSIONES



\### 12.1 Logros Principales



1\. \*\*Sistema RAG Funcional\*\*

&nbsp;  - Pipeline completo de ingesta → recuperación → generación

&nbsp;  - 66 chunks de alta calidad del DTF-13

&nbsp;  - Búsqueda híbrida optimizada



2\. \*\*Multi-Agente Operativo\*\*

&nbsp;  - 3 agentes con roles diferenciados

&nbsp;  - Orquestación automática funcional

&nbsp;  - Estado persistente en simulacros



3\. \*\*Interfaz de Usuario\*\*

&nbsp;  - 3 modos de operación claros

&nbsp;  - Flujo conversacional natural

&nbsp;  - Gestión de estados compleja



4\. \*\*Documentación Técnica\*\*

&nbsp;  - Decisiones justificadas

&nbsp;  - Limitaciones reconocidas

&nbsp;  - Roadmap claro de mejoras



\### 12.2 Aprendizajes Clave



\*\*Técnicos:\*\*

\- Chunking es más arte que ciencia (1 doc limpio > múltiples fragmentados)

\- Embeddings generales suficientes con búsqueda híbrida

\- Estado persistente crítico para flujos complejos



\*\*Metodológicos:\*\*

\- MVP iterativo > sistema completo desde inicio

\- Documentación continua facilita entrega final

\- Testing temprano evita sorpresas finales



\*\*Dominio:\*\*

\- Terminología específica (bomberos) requiere estrategia híbrida

\- Protocolos técnicos necesitan citación precisa

\- Simulacros requieren contexto completo para evaluación



\### 12.3 Impacto Potencial



\*\*Sector Emergencias:\*\*

\- Democratiza acceso a conocimiento especializado

\- Acelera formación de nuevos profesionales

\- Mejora preparación operativa continua



\*\*Académico:\*\*

\- Demuestra aplicabilidad RAG en dominio crítico

\- Validación de multi-agente en caso real

\- Base para investigación futura



\*\*Personal:\*\*

\- Integración de experiencia bombero + data science

\- Prototipo viable para startup

\- Portfolio técnico diferenciador



---



\## 13. CONTACTO Y RECURSOS



\### Autor

\*\*Josué\*\*

\- 🚒 Bombero Profesional (experiencia operativa real)

\- 🎓 Máster en Data Science con IA (BIG School)

\- 💻 Desarrollador 10x (IIA)



\### Enlaces del Proyecto

\- \*\*Repositorio:\*\* \[GitHub - sarfire-rag] (pendiente publicar)

\- \*\*Demo:\*\* \[Hugging Face Spaces] (pendiente deploy)

\- \*\*Video:\*\* \[YouTube/Loom] (pendiente grabar)



\### Referencias Técnicas

\- \[LangChain Documentation](https://python.langchain.com/)

\- \[ChromaDB Guide](https://docs.trychroma.com/)

\- \[Gradio Quickstart](https://gradio.app/docs/)

\- \[Sentence Transformers](https://www.sbert.net/)



---



\## ANEXOS



\### A. Prompt Templates



\#### Agente Formador

```python

"""

Eres un instructor experto en emergencias y protocolos de bomberos.



Contexto disponible:

{context}



Pregunta del usuario:

{question}



Proporciona una explicación clara y pedagógica basada ÚNICAMENTE 

en el contexto proporcionado. Cita las fuentes específicas.

"""

```



\#### Agente Simulador - Generación

```python

"""

Eres un instructor de bomberos creando escenarios de entrenamiento.



Contexto de protocolos:

{context}



Solicitud:

{query}



Genera un escenario REALISTA y DETALLADO basado en los protocolos.

Incluye: situación inicial, recursos, evolución, y dilema táctico.

"""

```



\#### Agente Simulador - Evaluación

```python

"""

ESCENARIO ACTIVO:

{scenario}



DECISIÓN DEL USUARIO:

{decision}



PROTOCOLO DE REFERENCIA:

{context}



Evalúa la decisión del usuario según el protocolo DTF-13.

Proporciona feedback constructivo con aspectos positivos y mejorables.

"""

```



---



\### B. Configuración ChromaDB



```python

\# src/rag/vector\_store.py

settings = chromadb.Settings(

&nbsp;   chroma\_db\_impl="duckdb+parquet",

&nbsp;   persist\_directory="data/processed/chromadb",

&nbsp;   anonymized\_telemetry=False

)



collection = client.create\_collection(

&nbsp;   name="sarfire\_docs",

&nbsp;   metadata={"hnsw:space": "cosine"},

&nbsp;   embedding\_function=embedding\_function

)

```



---



\### C. Métricas de Chunks



```

Total chunks: 66

Tamaño medio: 1019 caracteres

Tamaño mínimo: 234 caracteres

Tamaño máximo: 1498 caracteres

Desviación estándar: 387 caracteres

```



---



\*\*Versión del Documento:\*\* 1.0  

\*\*Fecha de Creación:\*\* 20 Enero 2026  

\*\*Última Actualización:\*\* 20 Enero 2026  

\*\*Estado:\*\* Documento Maestro Consolidado  



---



\*Este documento es el único punto de verdad sobre el proyecto SARFIRE-RAG. Todas las decisiones técnicas, arquitectónicas y de planificación están consolidadas aquí.\*

