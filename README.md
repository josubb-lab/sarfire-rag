# 🔥 SARFIRE-RAG

**Asistente Inteligente para Emergencias basado en RAG**

Sistema RAG especializado para el sector de emergencias que transforma manuales y protocolos en conocimiento operativo, formación automatizada y casos prácticos contextualizados.

---

## 🎯 Objetivo

Desarrollar un asistente RAG funcional que convierta documentación técnica de emergencias en:
- ✅ Respuestas operativas precisas
- ✅ Contenido formativo personalizado
- ✅ Simulacros y casos prácticos automáticos

---

## 🏗️ Arquitectura MVP

```
SARFIRE-RAG/
├── RAG Core (ChromaDB + Gemini)
├── 2 Agentes Especializados
│   ├── FORMADOR: Consultas + Formación
│   └── SIMULADOR: Casos prácticos + Escenarios
├── UI (Gradio)
└── Documentos (7 PDFs manuales bomberos)
```

---

## 🚀 Quick Start

### MVP Entry Point (WSL)

- **MVP entrypoint:** `app.py`
- **Experimental UIs:** `experiments_ui/` (no afectan al MVP)

**WSL (Ubuntu) – pasos rápidos:**

```bash
cd /home/sarfirerag/projects/sarfire-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` y añade `GOOGLE_API_KEY`, luego ejecuta:

```bash
python app.py
```

### 1. Instalación

```bash
# Clonar repo
git clone [tu-repo]
cd sarfire-rag

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración

```bash
# Copiar archivo de configuración
cp .env.example .env

# Editar .env y añadir tu GOOGLE_API_KEY
nano .env
```

### 3. Añadir Documentos

```bash
# Copiar tus PDFs a la carpeta data/raw/
cp /ruta/a/tus/pdfs/*.pdf data/raw/
```

### 4. Probar el Loader

```bash
# Test de carga de PDFs
python src/loaders/pdf_loader.py
```

---

## 📦 Stack Tecnológico

| Componente | Herramienta |
|------------|-------------|
| **LLM** | Gemini 1.5 Flash |
| **Vector Store** | ChromaDB |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Framework** | LangChain |
| **UI** | Gradio |
| **Docs** | pypdf |

---

## 📁 Estructura del Proyecto

```
sarfire-rag/
├── data/
│   ├── raw/              # PDFs originales
│   └── processed/        # Embeddings/chunks
├── src/
│   ├── loaders/          # Ingesta de documentos
│   ├── rag/              # RAG core
│   └── agents/           # Agentes especializados
├── tests/                # Tests unitarios
├── notebooks/            # Pruebas y análisis
├── requirements.txt      # Dependencias
└── README.md
```

---

## 🔄 Roadmap

- [x] Estructura del proyecto
- [x] Loader de PDFs
- [ ] Pipeline RAG básico
- [ ] Agente Formador
- [ ] Agente Simulador
- [ ] Interfaz Gradio
- [ ] Deploy

---

## 👨‍🚒 Autor

**Josué** - Bombero Profesional + Data Scientist

- 🎓 Máster en Data Science con IA (BIG School)
- 🚒 Experiencia operativa en emergencias
- 💻 Especialización en ML + RAG

---

## 📄 Licencia

MIT License - Proyecto académico (TFM + Capstone)

---

## 🔗 Enlaces

- [Plan de Desarrollo](docs/plan.md)
- [Arquitectura](docs/arquitectura.md)
- [Documentación API](docs/api.md)
