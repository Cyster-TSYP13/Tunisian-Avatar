"""
Multilingual Document Processor
Handles multilingual document processing with language-aware chunking
"""

from typing import List, Dict
import re
from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MultilingualDocumentProcessor:
    def __init__(self, chunk_size=1024, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_document(self, document: Dict) -> List[Dict]:
        """
        Process document with language-aware chunking
        
        Args:
            document: Dict with 'content', 'metadata', 'page_number'
            
        Returns:
            List of processed chunks with metadata
        """
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        page_number = document.get('page_number', 0)
        
        if not content.strip():
            return []
        
        # Detect language
        try:
            language = detect(content)
        except:
            language = 'unknown'
        
        # Handle RTL text properly (Arabic, Hebrew, etc.)
        is_rtl = language in ['ar', 'he', 'fa', 'ur']
        if is_rtl:
            content = self._normalize_rtl_text(content)
        
        # Split into chunks maintaining semantic boundaries
        chunks = self.text_splitter.split_text(content)
        
        # Create processed chunks with preserved metadata
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                'page_number': page_number,
                'chunk_index': idx,
                'language': language,
                'is_rtl': is_rtl,
                'total_chunks': len(chunks)
            }
            
            processed_chunks.append({
                'content': chunk,
                'metadata': chunk_metadata
            })
        
        return processed_chunks
    
    def _normalize_rtl_text(self, text: str) -> str:
        """Normalize RTL text for better processing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Ensure proper text direction markers
        text = '\u202B' + text + '\u202C'  # RLE and PDF markers
        return text
