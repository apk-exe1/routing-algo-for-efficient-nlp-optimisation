import time
from Levenshtein import distance as lev_dist

class StandardValidator:
    def __init__(self, dictionary_path="data/sample_dictionary_100k.txt"):
        self.dictionary = []
        with open(dictionary_path, "r", encoding="utf-8") as f:
            for line in f:
                self.dictionary.append(line.strip())
        print(f"[Standard] Loaded {len(self.dictionary)} words into a flat list.")

    def find_best_match(self, word):
        # Naive O(N) search through the dictionary
        best_match = word
        min_dist = float('inf')
        
        # In a real unoptimized scenario, we check every word in dictionary
        for dict_word in self.dictionary:
            dist = lev_dist(word, dict_word)
            if dist < min_dist:
                min_dist = dist
                best_match = dict_word
                if min_dist == 0:
                    break # perfect match
                    
        return best_match

    def process_text(self, text):
        words = text.split()
        if not words:
            return text, 0, 0, []
        
        corrected_words = []
        word_analysis = []
        
        start_time = time.perf_counter()
        for word in words:
            word_start = time.perf_counter()
            clean_word = word.strip('.,!?;:()[]{}"\'')
            if clean_word:
                best_match = self.find_best_match(clean_word)
                corrected_word = word.replace(clean_word, best_match)
                corrected_words.append(corrected_word)
                word_end = time.perf_counter()
                
                word_analysis.append({
                    "original": clean_word,
                    "corrected": best_match,
                    "latency_ms": round((word_end - word_start) * 1000, 4)
                })
            else:
                corrected_words.append(word)
                
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        return " ".join(corrected_words), latency_ms, len(words), word_analysis
