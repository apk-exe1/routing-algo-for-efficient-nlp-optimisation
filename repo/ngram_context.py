"""
N-gram Context Module for Context-Aware Ranking
Provides contextual probability scores for word suggestions
"""

import sqlite3
from typing import List, Dict, Optional, Tuple
import os

class NgramContext:
    """Provides n-gram based contextual scoring for word validation"""
    
    def __init__(self, db_path: str = "data/ngram_context.db"):
        """Initialize n-gram context manager"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        if os.path.exists(db_path):
            self._connect()
            print(f"✅ N-gram database loaded: {db_path}")
        else:
            print(f"⚠️  N-gram database not found: {db_path}")
    
    def _connect(self):
        """Connect to database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"❌ Error connecting to n-gram database: {e}")
    
    def get_bigram_probability(self, prev_word: str, next_word: str) -> float:
        """
        Get probability of next_word following prev_word
        
        P(next_word | prev_word) = count(prev_word, next_word) / sum(count(prev_word, *))
        """
        if not self.cursor:
            return 0.0
        
        try:
            # Get count of this specific bigram
            self.cursor.execute(
                "SELECT count FROM bigrams WHERE prev_word = ? AND next_word = ?",
                (prev_word, next_word)
            )
            result = self.cursor.fetchone()
            bigram_count = result[0] if result else 0
            
            if bigram_count == 0:
                return 0.0
            
            # Get total count of all bigrams starting with prev_word
            self.cursor.execute(
                "SELECT SUM(count) FROM bigrams WHERE prev_word = ?",
                (prev_word,)
            )
            total_count = self.cursor.fetchone()[0] or 1
            
            # Calculate probability
            probability = bigram_count / total_count
            return probability
            
        except Exception as e:
            print(f"Error calculating bigram probability: {e}")
            return 0.0
    
    def get_contextual_score(self, prev_word: Optional[str], candidate: str) -> float:
        """
        Calculate contextual score for a candidate word given previous word
        
        Returns score between 0.0 and 1.0
        """
        if not prev_word or not self.cursor:
            return 0.0
        
        probability = self.get_bigram_probability(prev_word, candidate)
        
        # Normalize to 0-1 range (log scale for better distribution)
        if probability > 0:
            # Use log scale: log(p) / log(0.0001) where 0.0001 is minimum threshold
            import math
            score = max(0.0, min(1.0, math.log(probability) / math.log(0.0001)))
            return score
        
        return 0.0
    
    def rank_candidates_with_context(
        self, 
        prev_word: Optional[str], 
        candidates: List[Tuple[str, float]]
    ) -> List[Tuple[str, float, float]]:
        """
        Re-rank candidates using contextual information
        
        Args:
            prev_word: Previous word for context
            candidates: List of (word, levenshtein_score) tuples
        
        Returns:
            List of (word, levenshtein_score, context_score) tuples
        """
        if not prev_word:
            # No context available
            return [(word, lev_score, 0.0) for word, lev_score in candidates]
        
        ranked = []
        for word, lev_score in candidates:
            context_score = self.get_contextual_score(prev_word, word)
            ranked.append((word, lev_score, context_score))
        
        return ranked
    
    def get_top_following_words(self, prev_word: str, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get most common words that follow prev_word
        
        Returns:
            List of (word, count) tuples
        """
        if not self.cursor:
            return []
        
        try:
            self.cursor.execute(
                """
                SELECT next_word, count 
                FROM bigrams 
                WHERE prev_word = ? 
                ORDER BY count DESC 
                LIMIT ?
                """,
                (prev_word, limit)
            )
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error getting following words: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Example usage
if __name__ == "__main__":
    # Test the module
    ngram = NgramContext()
    
    # Test bigram probability
    prob = ngram.get_bigram_probability("ಸೂರ್ಯನು", "ಬೆಳ್ಳಗೆ")
    print(f"P(ಬೆಳ್ಳಗೆ | ಸೂರ್ಯನು) = {prob:.6f}")
    
    # Test contextual scoring
    score = ngram.get_contextual_score("ಸೂರ್ಯನು", "ಬೆಳ್ಳಗೆ")
    print(f"Contextual score: {score:.4f}")
    
    # Test ranking
    candidates = [
        ("ಬೆಳ್ಳಗೆ", 0.85),
        ("ಬೆಲ್ಲವನ್ನು", 0.82),
        ("ಬೆಲ್ಲದ", 0.80)
    ]
    ranked = ngram.rank_candidates_with_context("ಸೂರ್ಯನು", candidates)
    
    print("\nRanked candidates:")
    for word, lev_score, ctx_score in ranked:
        print(f"  {word}: Lev={lev_score:.2f}, Context={ctx_score:.4f}")
    
    # Show top following words
    following = ngram.get_top_following_words("ಸೂರ್ಯನು", 5)
    print(f"\nTop words following 'ಸೂರ್ಯನು':")
    for word, count in following:
        print(f"  {word} (count: {count})")
    
    ngram.close()