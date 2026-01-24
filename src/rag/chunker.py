"""
Chunker para SARFIRE-RAG
Divide documentos en fragmentos (chunks) para procesamiento RAG
"""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Divide documentos en chunks optimizados para RAG"""
    
    def __init__(
        self,
        chunk_size: int = 1500,    # Chunks más grandes
        chunk_overlap: int = 150,   # Menos overlap (evita duplicados)
        separators: List[str] = None
    ):
        """
        Args:
            chunk_size: Tamaño máximo de cada chunk en caracteres
            chunk_overlap: Solapamiento entre chunks para mantener contexto
            separators: Lista de separadores para dividir (por defecto: párrafos, líneas, palabras)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Separadores optimizados para manuales técnicos
        if separators is None:
            separators = [
                "\n\n\n",    # Secciones (triple salto)
                "\n\n",      # Párrafos
                "\n",        # Líneas
                ". ",        # Frases
                " ",         # Palabras
                ""           # Caracteres
            ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Divide un documento en chunks manteniendo metadata
        
        Args:
            document: Documento procesado por PDFLoader
            
        Returns:
            Lista de chunks con metadata
        """
        chunks = []
        
        # Procesar cada página
        for page_data in document['pages']:
            page_num = page_data['page_num']
            text = page_data['text']
            
            # Dividir el texto de la página en chunks
            page_chunks = self.text_splitter.split_text(text)
            
            # Añadir metadata a cada chunk
            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunk = {
                    'text': chunk_text,
                    'metadata': {
                        'filename': document['metadata']['filename'],
                        'page_num': page_num,
                        'chunk_idx': chunk_idx,
                        'total_chunks_in_page': len(page_chunks),
                        'chunk_size': len(chunk_text)
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def chunk_all_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Procesa múltiples documentos
        
        Args:
            documents: Lista de documentos del PDFLoader
            
        Returns:
            Lista de todos los chunks con metadata
        """
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
        
        return all_chunks
    
    def print_stats(self, chunks: List[Dict]):
        """Muestra estadísticas de chunking"""
        if not chunks:
            print("No hay chunks para analizar")
            return
        
        # Estadísticas por documento
        docs_stats = {}
        for chunk in chunks:
            filename = chunk['metadata']['filename']
            if filename not in docs_stats:
                docs_stats[filename] = {
                    'num_chunks': 0,
                    'total_chars': 0,
                    'pages': set()
                }
            
            docs_stats[filename]['num_chunks'] += 1
            docs_stats[filename]['total_chars'] += chunk['metadata']['chunk_size']
            docs_stats[filename]['pages'].add(chunk['metadata']['page_num'])
        
        # Estadísticas globales
        total_chunks = len(chunks)
        chunk_sizes = [c['metadata']['chunk_size'] for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        
        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS DE CHUNKING")
        print("="*70)
        print(f"Total de chunks: {total_chunks}")
        print(f"Tamaño medio: {avg_size:.0f} caracteres")
        print(f"Tamaño mínimo: {min_size} caracteres")
        print(f"Tamaño máximo: {max_size} caracteres")
        print(f"Overlap configurado: {self.chunk_overlap} caracteres")
        print(f"\nDetalles por documento:")
        
        for filename, stats in docs_stats.items():
            avg_doc = stats['total_chars'] / stats['num_chunks']
            print(f"  - {filename}:")
            print(f"      Chunks: {stats['num_chunks']}")
            print(f"      Páginas: {len(stats['pages'])}")
            print(f"      Tamaño medio: {avg_doc:.0f} chars")
        
        print("="*70 + "\n")
    
    def show_sample_chunks(self, chunks: List[Dict], n: int = 3):
        """Muestra ejemplos de chunks para inspección"""
        print("\n" + "="*70)
        print(f"📄 MUESTRA DE {n} CHUNKS")
        print("="*70)
        
        for i, chunk in enumerate(chunks[:n]):
            meta = chunk['metadata']
            print(f"\n--- CHUNK {i+1} ---")
            print(f"Documento: {meta['filename']}")
            print(f"Página: {meta['page_num']} | Chunk: {meta['chunk_idx']+1}/{meta['total_chunks_in_page']}")
            print(f"Tamaño: {meta['chunk_size']} caracteres")
            print("-" * 70)
            print(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
            print("-" * 70)