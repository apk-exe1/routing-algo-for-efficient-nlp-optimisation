import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from Levenshtein import distance as lev_dist

class BKNode:
    def __init__(self, word):
        self.word = word
        self.children = {} # distance -> BKNode

class BKTree:
    def __init__(self):
        self.root = None

    def add(self, word):
        if self.root is None:
            self.root = BKNode(word)
            return

        curr = self.root
        while True:
            dist = lev_dist(word, curr.word)
            if dist == 0:
                return # word already in tree
            if dist in curr.children:
                curr = curr.children[dist]
            else:
                curr.children[dist] = BKNode(word)
                break

    def search(self, word, max_dist):
        if self.root is None:
            return []

        results = []
        candidates = [self.root]

        while candidates:
            curr = candidates.pop() # Use DFS (O(1) pop) instead of BFS pop(0) which is O(N)
            dist = lev_dist(word, curr.word)

            if dist <= max_dist:
                results.append((dist, curr.word))
                if dist == 0:
                    break # Early exit on perfect match

            # Add children within the distance range
            for d, child in curr.children.items():
                if dist - max_dist <= d <= dist + max_dist:
                    candidates.append(child)

        return results

class OptimizedValidator:
    def __init__(self, dictionary_path="data/sample_dictionary_100k.txt"):
        self.bktree = BKTree()
        self.frequencies = {}
        
        with open(dictionary_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                word = line.strip()
                self.bktree.add(word)
                # Assign a dummy frequency for adaptive ranking
                self.frequencies[word] = len(lines) - idx 
                
        print(f"[Optimized] Loaded {len(lines)} words into BK-Tree.")
        
        self.cache_hits = 0

    @lru_cache(maxsize=10000)
    def find_best_match_cached(self, word):
        # Using BKTree with max distance 1 to guarantee high speed 
        matches = self.bktree.search(word, max_dist=1)
            
        if not matches:
            return word # Return original if no close matches found
            
        # Adaptive Ranking:
        # Rank primarily by Levenshtein distance (ascending)
        # Then by frequency (descending) as a tie-breaker
        matches.sort(key=lambda x: (x[0], -self.frequencies.get(x[1], 0)))
        
        return matches[0][1]

    def process_word(self, word_token):
        word_start = time.perf_counter()
        clean_word = word_token.strip('.,!?;:()[]{}"\'')
        if clean_word:
            # Check cache statistics manually for tracking
            cache_info_before = self.find_best_match_cached.cache_info()
            best_match = self.find_best_match_cached(clean_word)
            cache_info_after = self.find_best_match_cached.cache_info()
            
            is_hit = cache_info_after.hits > cache_info_before.hits
                
            corrected_token = word_token.replace(clean_word, best_match)
            word_end = time.perf_counter()
            
            analysis = {
                "original": clean_word,
                "corrected": best_match,
                "latency_ms": round((word_end - word_start) * 1000, 4),
                "cache_hit": is_hit
            }
            return corrected_token, analysis, is_hit
        
        word_end = time.perf_counter()
        return word_token, {
                "original": word_token,
                "corrected": word_token,
                "latency_ms": round((word_end - word_start) * 1000, 4),
                "cache_hit": False
            }, False

    def process_text(self, text):
        words = text.split()
        if not words:
            return text, 0, 0, 0, 0, []
            
        self.cache_hits = 0
        self.find_best_match_cached.cache_clear()
        
        start_time = time.perf_counter()
        
        # Parallel processing
        num_threads = 4
        word_analysis = []
        corrected_words = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.process_word, words))
            
            for corrected_token, analysis, is_hit in results:
                corrected_words.append(corrected_token)
                word_analysis.append(analysis)
                if is_hit:
                    self.cache_hits += 1
            
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        return " ".join(corrected_words), latency_ms, len(words), self.cache_hits, num_threads, word_analysis
