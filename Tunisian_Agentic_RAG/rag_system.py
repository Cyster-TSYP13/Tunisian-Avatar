"""
Main Multilingual RAG System
Integrates all components for intelligent document Q&A
"""

from typing import List, Dict, Optional
import os
import requests
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os

from multilingual_processor import MultilingualDocumentProcessor
from web_scraper import IntelligentWebScraper
from vector_store import OptimizedVectorStore
from memory_engine import ConversationMemoryEngine
from filter_tn import load_bad_words, sanitize_text
import json


class MultilingualRAGSystem:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        persist_directory: str = "chromadb_data",
        collection_name: str = "rag_collection"
    ):
        """
        Initialize the Multilingual RAG System
        
        Args:
            embedding_model_name: Name of the embedding model to use
            persist_directory: Directory to persist vector store
            collection_name: Name of the collection
        """
        print("🚀 Initializing Multilingual RAG System...")
        
        # Initialize embedding model
        print(f"📥 Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize components
        print("🔧 Initializing components...")
        self.document_processor = MultilingualDocumentProcessor()
        self.web_scraper = IntelligentWebScraper()
        self.vector_store = OptimizedVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        self.memory_engine = ConversationMemoryEngine()
        
        print("✅ RAG System initialized successfully!\n")

        # Load Tunisian bad/abusive words from XLSX for filtering
        bad_words_path = os.getenv("BAD_WORDS_XLSX", "T-HSAB.xlsx")
        try:
            self.bad_words = load_bad_words(bad_words_path)
            print(f"🔐 Loaded {len(self.bad_words)} filtered words from {bad_words_path}")
        except Exception:
            self.bad_words = set()
    
    def load_pdf(self, pdf_path: str) -> int:
        """
        Load and process a PDF document
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of chunks added to vector store
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"📄 Loading PDF: {pdf_path}")
        
        # Read PDF
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"📖 Total pages: {total_pages}")
        
        all_chunks = []
        
        # Process each page
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            
            if not text.strip():
                continue
            
            document = {
                'content': text,
                'metadata': {
                    'source': pdf_path,
                    'source_type': 'pdf'
                },
                'page_number': page_num
            }
            
            # Process document into chunks
            chunks = self.document_processor.process_document(document)
            all_chunks.extend(chunks)
            
            if page_num % 10 == 0:
                print(f"  Processed {page_num}/{total_pages} pages...")
        
        print(f"✂️  Created {len(all_chunks)} chunks from {total_pages} pages")
        
        # Generate embeddings and add to vector store
        if all_chunks:
            print("🧮 Generating embeddings...")
            documents = [chunk['content'] for chunk in all_chunks]
            metadata = [chunk['metadata'] for chunk in all_chunks]
            
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True
            ).tolist()
            
            print("💾 Adding to vector store...")
            self.vector_store.add_documents_batch(documents, embeddings, metadata)
        
        return len(all_chunks)
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        use_memory: bool = True,
        scrape_urls: bool = False
    ) -> Dict:
        """
        Query the RAG system
        
        Args:
            query_text: The user's query
            n_results: Number of results to retrieve
            use_memory: Whether to use conversation memory
            scrape_urls: Whether to scrape URLs found in documents
            
        Returns:
            Dict with answer, sources, and metadata
        """
        print(f"\n🔍 Querying: {query_text}")
        
        # Get relevant context from memory if enabled
        relevant_history = []
        if use_memory:
            relevant_history = self.memory_engine.get_relevant_context(query_text)
            if relevant_history:
                print(f"💭 Found {len(relevant_history)} relevant past interactions")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Search vector store
        print(f"🔎 Searching vector store for top {n_results} results...")
        search_results = self.vector_store.search(query_embedding, n_results=n_results)
        
        if not search_results or not search_results.get('documents'):
            print("⚠️  No results found — using LLM fallback if configured")
            # Use LLM fallback (Gemini) when configured
            llm_answer = self._gemini_generate(query_text)
            llm_answer = sanitize_text(llm_answer, self.bad_words)
            return {
                'answer': llm_answer or "I couldn't find relevant information to answer your question.",
                'sources': [],
                'relevant_history': relevant_history
            }
        
        # Extract results
        documents = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        distances = search_results.get('distances', [[]])[0]
        
        # Prepare sources
        sources = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            # sanitize source snippets to avoid exposing abusive terms
            safe_content = sanitize_text(doc, self.bad_words)
            sources.append({
                'content': safe_content,
                'metadata': meta,
                'similarity_score': 1 - dist if dist else 0,
                'rank': i + 1
            })
        
        print(f"✓ Found {len(sources)} relevant sources")
        
        # Optional: Scrape URLs from documents
        scraped_content = []
        if scrape_urls:
            print("🌐 Scraping URLs from documents...")
            all_text = "\n".join(documents)
            scraped_content = self.web_scraper.extract_and_scrape(query_text, all_text)
            if scraped_content:
                print(f"✓ Scraped {len(scraped_content)} URLs")
        
        # Generate answer (simple concatenation for now)
        # In production, you'd use an LLM here
        answer = self._generate_answer(query_text, sources, relevant_history)
        
        # Store interaction in memory
        if use_memory:
            self.memory_engine.add_interaction(query_text, answer, sources)
        
        return {
            'answer': answer,
            'sources': sources,
            'scraped_content': scraped_content,
            'relevant_history': relevant_history
        }
    
    def _generate_answer(
        self,
        query: str,
        sources: List[Dict],
        history: List[Dict]
    ) -> str:
        """
        Generate exactly 2 short lines suitable for TTS - no punctuation
        """
        if not sources:
            return "لم أتمكن من العثور على معلومات"
        
        # Create exactly 2 short lines from top 2 sources
        lines = []
        for source in sources[:2]:
            content = source['content'].replace('\n', ' ').strip()
            # Remove leading punctuation and extract first ~60 chars
            content = content.lstrip('.،؛ ')
            summary = content[:60]
            if len(summary) == 60:
                summary = summary.rsplit(' ', 1)[0]
            # Remove all punctuation for clean TTS output
            summary = ''.join(c for c in summary if c not in '.،؛:؟!()[]{}""\'')
            lines.append(summary)
        
        # sanitize final answer lines again for safety
        final = "\n".join(lines)
        final = sanitize_text(final, self.bad_words)
        return final

    def _gemini_generate(self, prompt: str) -> str:
        """Generate a short safe answer using Gemini (requires `GEMINI_API_KEY` env var).

        This is a minimal, best-effort implementation: set `GEMINI_API_KEY` in env.
        If no key is provided the function returns an empty string.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return ""

        # Default Gemini/Generative Language endpoint (may require different auth in your setup)
        url = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt,
            "max_output_tokens": 256
        }

        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                # Try common response fields, fall back to raw text
                # The exact structure depends on the API; this is a best-effort parse
                if isinstance(data, dict):
                    # Try known keys
                    if 'candidates' in data and data['candidates']:
                        return data['candidates'][0].get('content', '')
                    if 'output' in data and isinstance(data['output'], list) and data['output']:
                        # e.g., {'output': [{'content': '...'}]}
                        o = data['output'][0]
                        if isinstance(o, dict) and 'content' in o:
                            return o['content']
                    if 'text' in data:
                        return data['text']
                return resp.text
            else:
                return ""
        except Exception:
            return ""
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        vector_stats = self.vector_store.get_collection_stats()
        memory_summary = self.memory_engine.get_summary()
        
        return {
            'vector_store': vector_stats,
            'memory': memory_summary
        }
    
    def clear_all(self):
        """Clear all data from the system"""
        print("🗑️  Clearing all data...")
        self.vector_store.clear_collection()
        self.memory_engine.clear_history()
        print("✅ All data cleared")
