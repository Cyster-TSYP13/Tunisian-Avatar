"""
Optimized Vector Store using ChromaDB
Handles document storage and semantic search
"""

import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Optional
import hashlib


class OptimizedVectorStore:
    def __init__(self, persist_directory="chromadb_data", collection_name="rag_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.document_hashes = set()
        self._load_existing_hashes()

    def _load_existing_hashes(self):
        """Load existing document hashes from collection"""
        try:
            results = self.collection.get()
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    if metadata and 'doc_hash' in metadata:
                        self.document_hashes.add(metadata['doc_hash'])
        except:
            pass

    def add_documents_batch(self, documents: List[str], embeddings: List[List[float]], metadata: List[Dict]):
        """
        Add documents in batch with deduplication and metadata indexing
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: List of metadata dicts
        """
        if not documents or not embeddings or not metadata:
            print("Empty input, skipping batch")
            return
        
        if len(documents) != len(embeddings) or len(documents) != len(metadata):
            print("Mismatch in lengths of documents, embeddings, and metadata")
            return
        
        # Automatic deduplication
        unique_docs = []
        unique_embeddings = []
        unique_metadata = []
        unique_ids = []
        
        for i, doc in enumerate(documents):
            # Create hash for deduplication
            doc_hash = hashlib.md5(doc.encode()).hexdigest()
            
            if doc_hash not in self.document_hashes:
                self.document_hashes.add(doc_hash)
                unique_docs.append(doc)
                unique_embeddings.append(embeddings[i])
                
                # Enhanced metadata with indexable fields
                enhanced_metadata = {
                    **metadata[i],
                    'doc_hash': doc_hash,
                    'indexed_at': datetime.now().isoformat(),
                    'char_count': len(doc)
                }
                unique_metadata.append(enhanced_metadata)
                
                # Generate unique ID
                unique_ids.append(f"doc_{doc_hash}_{int(datetime.now().timestamp() * 1000)}")
        
        if not unique_docs:
            print("All documents were duplicates, skipping")
            return
        
        # Batch processing for efficiency
        try:
            self.collection.add(
                ids=unique_ids,
                embeddings=unique_embeddings,
                documents=unique_docs,
                metadatas=unique_metadata
            )
            print(f"✓ Successfully added {len(unique_docs)} unique documents to ChromaDB")
        except Exception as e:
            print(f"✗ Error adding documents to ChromaDB: {e}")
    
    def search(self, query_embedding: List[float], n_results: int = 5, filter_metadata: Optional[Dict] = None) -> Dict:
        """
        Search for similar documents with optional metadata filtering
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Search results with documents and metadata
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            return results
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return None
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.document_hashes.clear()
            print("Collection cleared successfully")
        except Exception as e:
            print(f"Error clearing collection: {e}")
