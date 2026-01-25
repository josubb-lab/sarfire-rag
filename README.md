# 🔥 ¿Qué es SARFIRE-RAG (MVP)?

SARFIRE-RAG es un asistente basado en RAG (recuperación y generación aumentada) para emergencias forestales. Usa documentación interna para responder preguntas técnicas y apoyar simulaciones operativas desde una interfaz web.

# ✅ Qué hace

- Responde consultas técnicas basadas en manuales internos.
- Genera escenarios de simulación y roleplay.
- Enruta automáticamente entre modos Director, Formador y Simulador.
- Muestra evidencias y señales de soporte en la UI.

# 🧱 Arquitectura (breve)

```
UI (Gradio) → Director/Formador/Simulador → RAG Pipeline
RAG Pipeline → Recuperación (ChromaDB) + Generación (Gemini)
```

# ▶️ Cómo ejecutar en WSL

Entrypoint: `app.py`

```bash
cd ~/projects/sarfire-rag
source venv/bin/activate
pip install -r requirements.txt  # solo si hace falta
python app.py
```

# 🧠 Diseño de la solución (memoria técnica)

SARFIRE-RAG se ha diseñado como un MVP funcional orientado a apoyar formación y simulación en el ámbito de emergencias forestales, partiendo de documentación técnica real.

### Problema abordado

La información crítica para bomberos suele encontrarse dispersa en manuales extensos, poco accesibles en situaciones formativas o de consulta rápida. SARFIRE-RAG busca transformar esa documentación estática en un sistema interactivo capaz de responder, formar y simular.

### Enfoque adoptado

Se ha optado por una arquitectura basada en RAG (Retrieval-Augmented Generation), que permite:
- Recuperar fragmentos relevantes desde manuales internos.
- Generar respuestas apoyadas en esa evidencia.
- Mantener trazabilidad entre respuesta y fuentes.

### Componentes principales

- **RAG Pipeline**: indexa los documentos mediante embeddings y ChromaDB, y recupera contexto relevante para cada consulta.
- **Agentes especializados**:
  - *Formador*: responde consultas técnicas y formativas.
  - *Simulador*: genera escenarios operativos y roleplay.
  - *Director*: enruta automáticamente cada consulta al agente más adecuado.
- **Interfaz Gradio**: permite interactuar con el sistema de forma clara y controlada.

### 🔍 Configuración técnica del sistema RAG

El comportamiento del sistema RAG en SARFIRE está definido por las siguientes decisiones técnicas:

- **Fragmentación (chunking)**:
  Los documentos se dividen en fragmentos solapados, equilibrando:
  - suficiencia semántica por fragmento,
  - precisión en la recuperación.
  
  Esto evita respuestas basadas en contexto incompleto o excesivo.

- **Modelo de embeddings**:
  Se emplea `all-MiniLM-L6-v2` (dimensión 384), basado en *sentence-transformers*, seleccionado por ofrecer un equilibrio adecuado entre:
  - calidad semántica adecuada en texto técnico multilingüe (incluido español),
  - bajo coste computacional,
  - viabilidad en ejecución local para un MVP.

- **Almacenamiento vectorial**:
  Los embeddings se persisten en **ChromaDB** de forma local (`data/processed/chromadb`), permitiendo:
  - búsquedas semánticas eficientes,
  - persistencia local sin dependencias externas,
  - reutilización del índice,
  - tiempos de arranque reducidos.

- **Modelo generativo (LLM)**:
  Se utiliza `gemini-2.0-flash`, configurado con:
  - temperatura = 0.3 (prioriza precisión sobre creatividad),
  - adecuada para contextos técnicos y formativos donde prima la fidelidad al contexto sobre la creatividad narrativa,
  - generación controlada y reproducible.

- **Recuperación (Top-K híbrida)**:
  Se recuperan los 10 fragmentos más relevantes por consulta mediante una estrategia híbrida que combina:
  - similitud semántica,
  - coincidencia por palabras clave.
  
  Esto permite mantener suficiente contexto sin introducir ruido excesivo, aumentando la robustez frente a preguntas mal formuladas o muy específicas.

- **Control de calidad y fallback**:
  Se aplica un umbral mínimo de relevancia (0.3) para decidir cuándo:
  - una respuesta puede basarse en documentación interna,
  - debe recurrirse a búsqueda externa (Tavily) o declararse sin soporte.
  
  La búsqueda externa actúa como fallback opcional, manteniendo siempre control explícito en la UI.

### Justificación del diseño

Las principales decisiones de diseño se han tomado buscando un equilibrio entre viabilidad técnica, claridad conceptual y alineación con un MVP funcional:

- **Uso de RAG frente a un LLM puro**: permite mantener trazabilidad entre respuestas y manuales reales, requisito clave en entornos críticos como emergencias. Incluye un bloque de "Evidencia" en la UI para transparencia y control de calidad.

- **Separación por agentes especializados**: evita respuestas genéricas y permite diferenciar claramente formación, simulación y enrutamiento, mejorando control semántico y extensibilidad futura.

- **Stack local-first (ChromaDB, MiniLM, Gemini Flash)**: se prioriza ejecución local sin dependencias cloud obligatorias para facilitar reproducibilidad, control total del entorno y evaluación académica. Se busca eficiencia y estabilidad antes que máximo rendimiento, coherente con un MVP académico.

### Alcance del MVP

Este MVP no pretende sustituir protocolos reales ni simuladores físicos, sino demostrar la viabilidad de integrar RAG y agentes especializados como apoyo a formación y análisis operativo.

# 🔐 Variables de entorno

Configura `.env` con:

```
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
```

- `GOOGLE_API_KEY`: requerida para Gemini.
- `TAVILY_API_KEY`: solo si activas búsqueda externa.

Pueden definirse también como variables del sistema si no se usa `.env`.

# 🧪 Demo en 3 preguntas

1) Formador: "¿Cuál es el procedimiento de seguridad al trabajar con motosierras en incendios forestales?"
2) Simulador: "Roleplay: soy jefe de brigada, plantea una situación crítica y evalúa mi respuesta."
3) Ambigua/fuera de dominio: "¿Qué me recomiendas para preparar una barbacoa en el bosque?"

# 🧭 Guion rápido de demo

1) Selecciona **Director**.
2) Lanza las 3 preguntas anteriores.
3) Revisa el enrutamiento y el bloque de evidencia.
4) Cambia a **Simulador** y responde la decisión solicitada.

# Validación y pruebas realizadas

La validación del MVP se ha basado en pruebas funcionales manuales sobre los tres modos del sistema:

- Consultas técnicas verificadas contra manuales reales (Formador).
- Generación de escenarios coherentes y decisiones operativas simuladas (Simulador).
- Comprobación del enrutamiento correcto por parte del Director.
- Verificación del bloque de evidencia y semáforo de soporte para distintos tipos de consulta.

Estas pruebas permiten confirmar que el sistema cumple su objetivo como demostrador funcional de RAG aplicado a emergencias forestales.

# ⚠️ Limitaciones

- La calidad depende del contenido de los manuales cargados.
- La búsqueda externa es opcional y requiere Tavily.
- No sustituye protocolos oficiales ni decisiones operativas reales.
