#!/usr/bin/env python3
import os
import shutil

print("🔥 PREPARANDO PROYECTO PARA UN ÚNICO PDF\n")

# 1. Hacer backup de PDFs antiguos
backup_dir = "data/raw/backup_old_pdfs"
os.makedirs(backup_dir, exist_ok=True)

print("📦 Haciendo backup de PDFs antiguos...")
for file in os.listdir("data/raw"):
    if file.endswith(".pdf") and not file.startswith("DTF"):
        old_path = os.path.join("data/raw", file)
        new_path = os.path.join(backup_dir, file)
        shutil.move(old_path, new_path)
        print(f"   Moved: {file}")

print(f"\n✅ PDFs antiguos movidos a: {backup_dir}")
print(f"✅ PDF activo: DTF-13-ORGANIZACION-GESTION-INCENDIOS.pdf")