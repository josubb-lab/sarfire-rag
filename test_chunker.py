#!/usr/bin/env python3
"""
Test del Chunker - Paso 1 Día 2
Ejecutar desde raíz: python test_chunker.py
"""

import sys
sys.path.append('src')

from loaders import PDFLoader
from rag import DocumentChunker


def main():
    print("🔥 SARFIRE-RAG - Test del Chunker (Paso 1/4)\n")
    
    # 1. Cargar documentos
    print("📄 Cargando PDFs...")
    loader = PDFLoader("data/raw")
    documents = loader.load_all_pdfs()
    
    if not documents:
        print("❌ No hay documentos. Asegúrate de tener PDFs en data/raw/")
        return
    
    # 2. Crear chunker con configuración optimizada
    print("\n⚙️  Configurando chunker...")
    chunker = DocumentChunker(
        chunk_size=1000,      # ~250 palabras por chunk
        chunk_overlap=200     # 20% overlap para mantener contexto
    )
    
    print(f"   - Tamaño de chunk: {chunker.chunk_size} caracteres")
    print(f"   - Overlap: {chunker.chunk_overlap} caracteres")
    
    # 3. Procesar documentos
    print("\n📝 Dividiendo documentos en chunks...")
    chunks = chunker.chunk_all_documents(documents)
    
    # 4. Mostrar estadísticas
    chunker.print_stats(chunks)
    
    # 5. Mostrar ejemplos
    chunker.show_sample_chunks(chunks, n=3)
    
    # 6. Validación de calidad
    print("\n🔍 VALIDACIÓN DE CALIDAD:")
    print("-" * 70)
    
    # Verificar que no hay chunks vacíos
    empty_chunks = [c for c in chunks if not c['text'].strip()]
    print(f"✓ Chunks vacíos: {len(empty_chunks)} (debería ser 0)")
    
    # Verificar distribución de tamaños
    chunk_sizes = [c['metadata']['chunk_size'] for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    within_range = sum(1 for s in chunk_sizes if 500 <= s <= 1500)
    pct_within_range = (within_range / len(chunk_sizes)) * 100
    
    print(f"✓ Chunks dentro de rango óptimo (500-1500): {within_range}/{len(chunks)} ({pct_within_range:.1f}%)")
    print(f"✓ Total de chunks generados: {len(chunks)}")
    
    print("-" * 70)
    
    print("\n✅ Test del Chunker completado!")
    print("\n📝 Siguiente paso: Generar embeddings de estos chunks")


if __name__ == "__main__":
    main()