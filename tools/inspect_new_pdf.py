#!/usr/bin/env python3
import sys
sys.path.append('src')
from loaders import PDFLoader
import pypdf

# Ver extracción completa
reader = pypdf.PdfReader("data/raw/DTF-13-ORGANIZACION-GESTION-INCENDIOS.pdf")

print(f"📄 PDF: {len(reader.pages)} páginas\n")
print("="*70)

# Ver primeras 3 páginas
for i in range(min(3, len(reader.pages))):
    text = reader.pages[i].extract_text()
    print(f"\n--- PÁGINA {i+1} ---")
    print(f"Caracteres extraídos: {len(text)}")
    print(f"Contenido:\n{text[:800]}")
    print("="*70)

# Ver últimas 2 páginas (suelen tener bibliografía)
for i in range(max(0, len(reader.pages)-2), len(reader.pages)):
    text = reader.pages[i].extract_text()
    print(f"\n--- PÁGINA {i+1} ---")
    print(f"Caracteres extraídos: {len(text)}")
    print(f"Contenido:\n{text[:800]}")
    print("="*70)