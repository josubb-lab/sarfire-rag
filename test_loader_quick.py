#!/usr/bin/env python3
"""
Test rápido del PDF Loader
Ejecutar desde la raíz del proyecto: python test_loader_quick.py
"""

import sys
sys.path.append('src')

from loaders import PDFLoader

def main():
    print("🔥 SARFIRE-RAG - Test del PDF Loader\n")
    
    # Inicializar loader
    try:
        loader = PDFLoader("data/raw")
        print("✅ Loader inicializado correctamente\n")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Asegúrate de:")
        print("   1. Estar en la raíz del proyecto")
        print("   2. Tener la carpeta data/raw/ creada")
        print("   3. Haber copiado tus PDFs a data/raw/")
        return
    
    # Cargar PDFs
    documents = loader.load_all_pdfs()
    
    if not documents:
        print("\n⚠️  No se encontraron documentos")
        print("💡 Copia tus PDFs a la carpeta data/raw/")
        return
    
    # Mostrar estadísticas
    loader.print_stats(documents)
    
    # Mostrar muestra del primer documento
    print("\n📄 MUESTRA DEL PRIMER DOCUMENTO:")
    print("-" * 70)
    first_text = loader.get_full_text(documents[0])
    print(first_text[:800])
    print("\n" + "." * 70)
    print(f"(Mostrando 800 de {len(first_text)} caracteres totales)")
    print("-" * 70)
    
    print("\n✅ Test completado exitosamente!")
    print("📝 Siguiente paso: Implementar chunking y embeddings")

if __name__ == "__main__":
    main()
