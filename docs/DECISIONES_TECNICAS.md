\# Decisiones Técnicas - SARFIRE-RAG



Documento de decisiones arquitectónicas y técnicas del proyecto.



---



\## Stack Tecnológico



\### LLM

\- \*\*Modelo:\*\* Gemini 2.0 Flash

\- \*\*Por qué:\*\* Balance entre velocidad y calidad, límites razonables

\- \*\*Alternativas consideradas:\*\* Gemini 2.5 Flash (límites muy restrictivos)



\### Embeddings

\- \*\*Modelo:\*\* all-MiniLM-L6-v2 (sentence-transformers)

\- \*\*Dimensión:\*\* 384

\- \*\*Por qué:\*\* Ligero, rápido, suficiente para corpus pequeño

\- \*\*Limitación:\*\* No especializado en español técnico

\- \*\*Mejora futura:\*\* paraphrase-multilingual-mpnet-base-v2



\### Vector Store

\- \*\*Base de datos:\*\* ChromaDB

\- \*\*Por qué:\*\* Simple, persistente, suficiente para MVP

\- \*\*Configuración:\*\* Similitud coseno



\### Búsqueda

\- \*\*Estrategia:\*\* Híbrida (semántica + keywords)

\- \*\*Por qué:\*\* Compensa debilidad de embeddings generales

\- \*\*Pesos:\*\* 40% semántica, 60% keywords

\- \*\*Mejora futura:\*\* Re-ranking con modelo más potente



---



\## Decisiones de Arquitectura



\### Documento Único vs Múltiples

\- \*\*Decisión:\*\* Usar 1 PDF limpio (DTF-13, 31 páginas)

\- \*\*Por qué:\*\*

&nbsp; - Menos problemas de chunking entre documentos

&nbsp; - Contexto más coherente

&nbsp; - Respuestas más precisas

&nbsp; - Más fácil de evaluar para Capstone

\- \*\*Trade-off:\*\* Menos cobertura temática, pero mejor calidad



\### Chunking

\- \*\*Tamaño:\*\* 1500 caracteres

\- \*\*Overlap:\*\* 150 caracteres (10%)

\- \*\*Por qué:\*\* Balance entre contexto y granularidad

\- \*\*Resultado:\*\* 66 chunks (vs 491 con múltiples PDFs)



\### Agentes

\- \*\*Cantidad:\*\* 2 agentes especializados + 1 director

\- \*\*Por qué:\*\*

&nbsp; - Formador: Explicaciones técnicas pedagógicas

&nbsp; - Simulador: Entrenamiento práctico con escenarios

&nbsp; - Director: Clasificación simple de intención

\- \*\*Alternativa descartada:\*\* 3+ agentes (complejidad innecesaria para MVP)



---



\## Limitaciones Conocidas



1\. \*\*Embeddings generales:\*\* No especializados en terminología de bomberos

2\. \*\*Dependencia de keywords:\*\* Para términos muy específicos (ej: Foëhn)

3\. \*\*Límites API Gemini:\*\* 20 requests/día en tier gratuito

4\. \*\*Contexto único:\*\* Solo 1 documento (escalable a más)



---



\## Métricas de Calidad



\- \*\*Chunks generados:\*\* 66

\- \*\*Tamaño medio chunk:\*\* 1019 caracteres

\- \*\*Respuestas RAG:\*\* 6-7/10 (suficiente para demo)

\- \*\*Tiempo respuesta:\*\* <10 segundos

\- \*\*Citación fuentes:\*\* ✅ Funcional



---



\## Próximos Pasos



1\. Interfaz Gradio con selector de modo

2\. Testing exhaustivo de ambos agentes

3\. Documentación README narrativo

4\. Video demo del sistema

5\. Subir a GitHub

