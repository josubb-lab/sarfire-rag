#!/usr/bin/env python3
import sys
sys.path.append('src')
from loaders import PDFLoader

loader = PDFLoader("data/raw")

# Ver el PDF de bibliografía específicamente
import pypdf
reader = pypdf.PdfReader("data/raw/03-06-IVM1-IIFF-Bibliografía.pdf")

print(f"📄 Bibliografía PDF - {len(reader.pages)} páginas\n")

for i, page in enumerate(reader.pages, 1):
    text = page.extract_text()
    print(f"--- PÁGINA {i} ---")
    print(text[:500])
    print("\n" + "="*70 + "\n")