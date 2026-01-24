# 🌐 FALLBACK A FUENTES EXTERNAS - IMPLEMENTADO

**Fecha:** 20 Enero 2026 (Día 1 - Parte 2)  
**Tiempo invertido:** 4 horas  
**Estado:** ✅ Funcional y probado

---

## Componentes Implementados

### 1. External Search Module (`src/rag/external_search.py`)

**Funcionalidad:**
- Wrapper de Tavily API
- Búsqueda en fuentes web externas
- Manejo robusto de errores
- Formateo de resultados para LLM

**Características:**
```python
class ExternalSearcher:
    - search(query, max_results=3) → Dict
    - format_results_for_llm(results) → str
    - Manejo de excepciones completo
```

**Configuración:**
- API: Tavily (https://tavily.com)
- Límite gratuito: 1000 requests/mes
- Resultados por query: 3 (configurable)

---

### 2. RAG Pipeline Actualizado (`src/rag/rag_pipeline.py`)

**Nuevas capacidades añadidas:**

#### Parámetros nuevos:
- `enable_external_fallback: bool = True` - Activa/desactiva fallback
- `relevance_threshold: float = 0.5` - Umbral para activar búsqueda externa

#### Nuevos métodos:
1. **`assess_relevance(chunks)`** - Evalúa calidad de resultados internos
   - Calcula similarity promedio
   - Retorna score 0-1
   
2. **`format_external_context(results)`** - Formatea resultados Tavily para LLM

3. **`query()` mejorado** - Lógica de 3 casos:
   - **CASO 1:** Relevancia alta (≥0.5) → RAG interno
   - **CASO 2:** Relevancia baja + `allow_external=None` → Preguntar usuario
   - **CASO 3:** `allow_external=True` → Buscar en Tavily

---

## Flujo de Funcionamiento

### Escenario 1: Query Interna (Alta Relevancia)

```
Usuario: "¿Qué es el PMA?"
    ↓
RAG busca en ChromaDB
    ↓
Relevancia: 0.555 (> 0.5)
    ↓
✅ Responde con DTF-13
    ↓
Fuentes: DTF-13, páginas 7 y 17
```

**Resultado:** Sistema usa documentación oficial, no consulta externo.

---

### Escenario 2: Query Externa (Baja Relevancia)

```
Usuario: "¿Cuáles son las últimas tecnologías en drones para bomberos?"
    ↓
RAG busca en ChromaDB
    ↓
Relevancia: 0.12 (< 0.5)
    ↓
⚠️ Pregunta al usuario: "¿Buscar en fuentes externas?"
    ↓
Usuario acepta (allow_external=True)
    ↓
🌐 Tavily busca en web
    ↓
✅ Responde con fuentes externas
    ↓
⚠️ Disclaimer: "Información externa - verificar con DTF-13"
```

**Resultado:** Sistema informa de fuente externa y recomienda verificación.

---

## Decisiones Técnicas

### ¿Por qué Tavily y no Google/Bing?

| Criterio | Tavily | Google API | Bing API |
|----------|--------|------------|----------|
| **Enfoque** | Search for LLMs | General | General |
| **Formato** | Optimizado para AI | Requiere parsing | Requiere parsing |
| **Límite gratuito** | 1000/mes | No existe | Limitado |
| **Setup** | Simple | Complejo | Medio |
| **Precio** | $30/mes (si necesario) | $5/1000 | Variable |

**Decisión:** Tavily por simplicidad y enfoque en IA.

---

### ¿Por qué umbral 0.5?

**Testing realizado:**
- Queries sobre PMA, DTF, niveles → Relevancia 0.5-0.8
- Queries fuera corpus → Relevancia 0.1-0.3

**Umbral 0.5:**
- ✅ No genera falsos positivos (búsquedas externas innecesarias)
- ✅ Detecta correctamente queries fuera del corpus
- ✅ Balance entre precisión y cobertura

**Configurable:** Se puede ajustar según feedback de usuarios.

---

## Métricas de Validación

### Tests Realizados (20 Enero 2026)

| Query | Relevancia | Fuente | Resultado |
|-------|------------|--------|-----------|
| "¿Qué es el PMA?" | 0.555 | Interna | ✅ Correcto |
| "¿Últimas tecnologías drones bomberos?" | 0.12 | Externa | ✅ Correcto |
| "Explica situaciones operativas" | 0.623 | Interna | ✅ Correcto |
| "Normativa europea drones emergencias" | 0.08 | Externa | ✅ Correcto |

**Precisión:** 100% (4/4 queries correctamente enrutadas)

---

## Estructura de Respuestas

### Respuesta Interna:
```python
{
    'question': str,
    'answer': str,
    'source': 'internal',
    'sources': [
        {
            'filename': 'DTF-13...',
            'page': 7,
            'similarity': 0.555
        }
    ],
    'relevance_score': 0.555,
    'metadata': {...}
}
```

### Respuesta Externa:
```python
{
    'question': str,
    'answer': str,
    'source': 'external',
    'external_sources': [
        {
            'title': str,
            'url': str,
            'content': str,
            'score': float
        }
    ],
    'relevance_score': 0.12,
    'disclaimer': '⚠️ INFORMACIÓN DE FUENTES EXTERNAS...',
    'metadata': {...}
}
```

### Pregunta al Usuario:
```python
{
    'question': str,
    'answer': 'No encuentro información suficiente...',
    'source': 'none',
    'should_ask_user': True,
    'question_for_user': '¿Deseas que busque en fuentes externas?',
    'relevance_score': 0.12,
    'internal_sources': [...]  # Mejores resultados internos
}
```

---

## Limitaciones Conocidas

1. **Tavily API Limits:**
   - Plan gratuito: 1000 requests/mes
   - Suficiente para desarrollo y demos
   - Producción requeriría plan de pago ($30/mes)

2. **Calidad de Fuentes Externas:**
   - No controlamos calidad de fuentes web
   - Disclaimer obligatorio en todas las respuestas externas
   - Usuario debe verificar con protocolos oficiales

3. **Idioma:**
   - Tavily optimizado para inglés
   - Queries en español funcionan pero con menor precisión
   - Mejora futura: traducir query a inglés antes de buscar

---

## Próximos Pasos (Día 2)

- [ ] Integrar en `app.py` (Gradio)
- [ ] Añadir botón "Buscar en fuentes externas"
- [ ] Mostrar disclaimer visual cuando source='external'
- [ ] Testing en interfaz completa

---

## Archivos Modificados/Creados

### Nuevos:
- `src/rag/external_search.py` (187 líneas)

### Modificados:
- `src/rag/rag_pipeline.py` (añadidas ~150 líneas)

### Configuración:
- `.env` → Añadido `TAVILY_API_KEY`
- `requirements.txt` → Añadido `tavily-python>=0.3.0`

---

**Implementación completada:** 20 Enero 2026, 22:00h  
**Tiempo total Día 1:** ~8 horas (Setup + RAG base + Fallback externo)  
**Estado proyecto:** 60% → 70% completado
