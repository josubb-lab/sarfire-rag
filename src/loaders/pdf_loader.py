"""
PDF Loader para SARFIRE-RAG
Extrae texto de PDFs de manuales de bomberos
"""
import os
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from tqdm import tqdm


class PDFLoader:
    """Carga y procesa documentos PDF"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Args:
            data_dir: Directorio donde están los PDFs
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directorio {data_dir} no existe")
    
    def load_single_pdf(self, pdf_path: Path) -> Dict:
        """
        Carga un único PDF
        
        Args:
            pdf_path: Ruta al PDF
            
        Returns:
            Dict con metadata y contenido extraído
        """
        try:
            reader = PdfReader(pdf_path)
            
            # Extraer texto de todas las páginas
            text_pages = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():  # Solo guardar páginas con contenido
                    text_pages.append({
                        'page_num': page_num,
                        'text': text.strip()
                    })
            
            # Metadata del documento
            metadata = {
                'filename': pdf_path.name,
                'num_pages': len(reader.pages),
                'num_pages_with_text': len(text_pages),
                'path': str(pdf_path)
            }
            
            return {
                'metadata': metadata,
                'pages': text_pages
            }
            
        except Exception as e:
            print(f"❌ Error procesando {pdf_path.name}: {e}")
            return None
    
    def load_all_pdfs(self) -> List[Dict]:
        """
        Carga todos los PDFs del directorio
        
        Returns:
            Lista de documentos procesados
        """
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No se encontraron PDFs en {self.data_dir}")
            return []
        
        print(f"📄 Encontrados {len(pdf_files)} PDFs")
        
        documents = []
        for pdf_path in tqdm(pdf_files, desc="Cargando PDFs"):
            doc = self.load_single_pdf(pdf_path)
            if doc:
                documents.append(doc)
        
        print(f"✅ Procesados {len(documents)} documentos correctamente")
        return documents
    
    def get_full_text(self, document: Dict) -> str:
        """
        Obtiene el texto completo de un documento
        
        Args:
            document: Documento procesado
            
        Returns:
            Texto completo del documento
        """
        return "\n\n".join([
            f"[Página {page['page_num']}]\n{page['text']}"
            for page in document['pages']
        ])
    
    def print_stats(self, documents: List[Dict]):
        """Imprime estadísticas de los documentos cargados"""
        if not documents:
            print("No hay documentos para analizar")
            return
        
        total_pages = sum(doc['metadata']['num_pages'] for doc in documents)
        total_pages_text = sum(doc['metadata']['num_pages_with_text'] for doc in documents)
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS DE CARGA")
        print("="*60)
        print(f"Documentos cargados: {len(documents)}")
        print(f"Total de páginas: {total_pages}")
        print(f"Páginas con texto: {total_pages_text}")
        print(f"Tasa de extracción: {total_pages_text/total_pages*100:.1f}%")
        print("\nDetalles por documento:")
        for doc in documents:
            meta = doc['metadata']
            print(f"  - {meta['filename']}: {meta['num_pages_with_text']}/{meta['num_pages']} páginas")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test del loader
    loader = PDFLoader("data/raw")
    docs = loader.load_all_pdfs()
    
    if docs:
        loader.print_stats(docs)
        
        # Mostrar muestra del primer documento
        print("📄 MUESTRA DEL PRIMER DOCUMENTO:")
        print("-" * 60)
        first_text = loader.get_full_text(docs[0])
        print(first_text[:500] + "...\n")
