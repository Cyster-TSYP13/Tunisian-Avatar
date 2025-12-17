"""
Conversation Memory Engine
Manages conversation history with semantic relevance
"""

from datetime import datetime, timedelta
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import tiktoken


class ConversationMemoryEngine:
    def __init__(self, max_memory_tokens=4096):
        self.conversation_history = []
        self.max_tokens = max_memory_tokens
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None

    def add_interaction(self, query: str, response: str, sources: List[Dict]):
        """Add a new interaction to conversation history"""
        interaction = {
            'query': query,
            'response': response,
            'sources': sources,
            'timestamp': datetime.now(),
            'query_embedding': self.embedding_model.encode(query, convert_to_tensor=True)
        }
        self.conversation_history.append(interaction)
        self._manage_memory()

    def get_relevant_context(self, current_query: str, max_interactions: int = 5) -> List[Dict]:
        """
        Get relevant conversation context based on current query
        
        Uses:
        - Semantic similarity with past interactions
        - Temporal relevance weighting
        - Source continuity tracking
        
        Args:
            current_query: The current user query
            max_interactions: Maximum number of past interactions to return
            
        Returns:
            List of relevant past interactions
        """
        if not self.conversation_history:
            return []
        
        # Encode current query
        current_embedding = self.embedding_model.encode(current_query, convert_to_tensor=True)
        
        # Calculate relevance scores for each interaction
        scored_interactions = []
        current_time = datetime.now()
        
        for interaction in self.conversation_history:
            # Semantic similarity score
            semantic_score = util.cos_sim(
                current_embedding, 
                interaction['query_embedding']
            ).item()
            
            # Temporal relevance (exponential decay)
            time_diff = (current_time - interaction['timestamp']).total_seconds()
            hours_passed = time_diff / 3600
            temporal_score = 0.5 ** (hours_passed / 24)  # Half-life of 24 hours
            
            # Combined score (weighted average)
            combined_score = 0.7 * semantic_score + 0.3 * temporal_score
            
            scored_interactions.append({
                'interaction': interaction,
                'score': combined_score
            })
        
        # Sort by relevance score
        scored_interactions.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top interactions that fit within token limit
        relevant_history = []
        total_tokens = 0
        
        for item in scored_interactions[:max_interactions]:
            interaction = item['interaction']
            
            # Estimate tokens for this interaction
            interaction_text = f"{interaction['query']}\n{interaction['response']}"
            
            if self.tokenizer:
                interaction_tokens = len(self.tokenizer.encode(interaction_text))
            else:
                # Rough estimate: ~4 chars per token
                interaction_tokens = len(interaction_text) // 4
            
            if total_tokens + interaction_tokens <= self.max_tokens:
                relevant_history.append(interaction)
                total_tokens += interaction_tokens
            else:
                break
        
        # Sort by timestamp for coherent conversation flow
        relevant_history.sort(key=lambda x: x['timestamp'])
        
        return relevant_history

    def _manage_memory(self):
        """
        Manage memory by removing old or irrelevant interactions
        Uses a combination of recency and importance
        """
        if len(self.conversation_history) <= 50:
            return
        
        current_time = datetime.now()
        
        # Score interactions for retention
        scored_for_retention = []
        for interaction in self.conversation_history:
            # Calculate age in hours
            age_hours = (current_time - interaction['timestamp']).total_seconds() / 3600
            
            # Recency score (exponential decay)
            recency_score = 0.5 ** (age_hours / 48)  # Half-life of 48 hours
            
            # Importance score (based on response length as proxy)
            importance_score = min(len(interaction['response']) / 1000, 1.0)
            
            # Combined retention score
            retention_score = 0.6 * recency_score + 0.4 * importance_score
            
            scored_for_retention.append({
                'interaction': interaction,
                'score': retention_score
            })
        
        # Sort by retention score and keep top 40
        scored_for_retention.sort(key=lambda x: x['score'], reverse=True)
        self.conversation_history = [
            item['interaction'] for item in scored_for_retention[:40]
        ]
        
        # Re-sort by timestamp
        self.conversation_history.sort(key=lambda x: x['timestamp'])

    def clear_history(self):
        """Clear all conversation history"""
        self.conversation_history = []
    
    def export_history(self) -> List[Dict]:
        """Export conversation history for persistence"""
        return [{
            'query': item['query'],
            'response': item['response'],
            'sources': item['sources'],
            'timestamp': item['timestamp'].isoformat()
        } for item in self.conversation_history]
    
    def get_summary(self) -> str:
        """Get a summary of conversation history"""
        if not self.conversation_history:
            return "No conversation history"
        
        total = len(self.conversation_history)
        oldest = self.conversation_history[0]['timestamp']
        newest = self.conversation_history[-1]['timestamp']
        
        return f"Total interactions: {total}\nOldest: {oldest}\nNewest: {newest}"
