"""
Intelligent Web Scraper
Extracts and scrapes relevant URLs from documents
"""

import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer, util


class IntelligentWebScraper:
    def __init__(self, max_depth=2, timeout=10):
        self.max_depth = max_depth
        self.timeout = timeout
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.visited_urls = set()

    def extract_and_scrape(self, query_context: str, document_content: str) -> List[Dict]:
        """
        Extract URLs from document and scrape relevant ones
        
        Args:
            query_context: User's query for relevance ranking
            document_content: Document content to extract URLs from
            
        Returns:
            List of scraped content with metadata
        """
        # 1. Extract URLs using regex
        urls = self.extract_urls(document_content)
        
        if not urls:
            return []
        
        # 2. Rank URLs by relevance to query
        relevant_urls = self.rank_by_relevance(urls, query_context)
        
        # 3. Intelligently scrape top URLs
        scraped_content = []
        for url in relevant_urls[:self.max_depth]:
            if url not in self.visited_urls:
                content = self.scrape_with_context(url)
                if content:
                    scraped_content.append(content)
                    self.visited_urls.add(url)
        
        return scraped_content

    def extract_urls(self, document_content: str) -> List[str]:
        """Extract URLs from document content"""
        # Regex pattern for URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls = re.findall(url_pattern, document_content)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls

    def rank_by_relevance(self, urls: List[str], query_context: str) -> List[str]:
        """Rank URLs by semantic relevance to query"""
        if not urls or not query_context:
            return urls
        
        # Create embeddings for query
        query_embedding = self.embedding_model.encode(query_context, convert_to_tensor=True)
        
        # Create embeddings for URLs (using domain and path as context)
        url_texts = [self._url_to_text(url) for url in urls]
        url_embeddings = self.embedding_model.encode(url_texts, convert_to_tensor=True)
        
        # Calculate similarity scores
        similarities = util.cos_sim(query_embedding, url_embeddings)[0]
        
        # Sort URLs by similarity (descending)
        ranked_indices = similarities.argsort(descending=True)
        ranked_urls = [urls[idx] for idx in ranked_indices]
        
        return ranked_urls
    
    def _url_to_text(self, url: str) -> str:
        """Convert URL to readable text for embedding"""
        parsed = urlparse(url)
        # Combine domain and path, replace separators with spaces
        text = f"{parsed.netloc} {parsed.path}".replace('/', ' ').replace('-', ' ').replace('_', ' ')
        return text

    def scrape_with_context(self, url: str) -> Dict:
        """Scrape URL and return structured content"""
        try:
            response = requests.get(url, timeout=self.timeout, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; RAG-Bot/1.0)'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract meaningful content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else url
            
            return {
                'url': url,
                'title': title_text,
                'content': text,
                'source_type': 'web_scrape'
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
