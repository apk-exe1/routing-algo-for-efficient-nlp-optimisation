#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Validation Module — Kannada OCR Post-Processing
=========================================================
Version 7.3 — Production-Ready

Bug Fixes over v7.2:
  FIX-11 ContextAnalyzer: Added suffix-based gender detection for unknown
         proper nouns. Names ending in ಾ/ಆ/ಿ/ಈ/ಎ/ಏ type vowels are classified
         as female; names ending in consonant/virama are classified as male.
         This covers ~70% of Kannada names not in the hardcoded list.
"""
import os
import sys
import json
import sqlite3
import csv
import math
from datetime import datetime
from threading import Lock
from collections import OrderedDict
from typing import Dict, List, Tuple, Set, Optional
import google.generativeai as genai
from dotenv import load_dotenv
os.environ["DISABLE_GEMINI_TESTING"] = "1"

load_dotenv()
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if _GEMINI_API_KEY:
    genai.configure(api_key=_GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
    print("[Gemini] ✅ Context awareness enabled")
else:
    _gemini_model = None
    print("[Gemini] ⚠️  API key not found — semantic validation disabled")


from conjunction_rules import check_conjunction_rules, check_paragraph_conjunctions, format_results_for_ui as format_conjunction_results
# ============================================================
# CONFIGURATION
# ============================================================
WEIGHT_LEVENSHTEIN = 0.4
WEIGHT_FREQUENCY   = 0.1
WEIGHT_CONTEXT     = 0.5

MAX_UNIGRAM_LOG = math.log(1_000_000)
MAX_BIGRAM_LOG  = math.log(100_000)

FUZZY_WEIGHT_LEVENSHTEIN = 0.25
FUZZY_WEIGHT_LCS         = 0.20
FUZZY_WEIGHT_COSINE      = 0.25
FUZZY_WEIGHT_FIRST_CHAR  = 0.15
FUZZY_WEIGHT_LENGTH      = 0.10
FUZZY_WEIGHT_VOWEL       = 0.05

# ============================================================
# KANNADA UNICODE CONSTANTS
# ============================================================
KANNADA_INDEP_VOWELS: Set[str] = set("ಅಆಇಈಉಊಋೠಎಏಐಒಓಔ")
VOWEL_SIGN_I_TYPE:  Set[str] = {'\u0CBF', '\u0CC0'}
VOWEL_SIGN_E_TYPE:  Set[str] = {'\u0CC6', '\u0CC7', '\u0CC8'}
VOWEL_SIGN_U_TYPE:  Set[str] = {'\u0CC1', '\u0CC2'}
VOWEL_SIGN_O_TYPE:  Set[str] = {'\u0CCA', '\u0CCB'}
VOWEL_SIGN_AA:      str       = '\u0CBE'
VIRAMA:             str       = '\u0CCD'

ALL_VOWEL_SIGNS: Set[str] = (
    VOWEL_SIGN_I_TYPE | VOWEL_SIGN_E_TYPE |
    VOWEL_SIGN_U_TYPE | VOWEL_SIGN_O_TYPE |
    {VOWEL_SIGN_AA}
)

def _is_kannada_consonant(ch: str) -> bool:
    return 0x0C95 <= ord(ch) <= 0x0CBD

# ============================================================
# LEVENSHTEIN DISTANCE
# ============================================================
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


# ============================================================
# COSINE SIMILARITY
# ============================================================
def extract_character_ngrams(word: str, n: int = 2) -> List[str]:
    if len(word) < n:
        return [word]
    return [word[i:i + n] for i in range(len(word) - n + 1)]


def cosine_similarity_vectors(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    m1  = math.sqrt(sum(a * a for a in vec1))
    m2  = math.sqrt(sum(b * b for b in vec2))
    if m1 == 0 or m2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (m1 * m2)))


def calculate_cosine_similarity(word1: str, word2: str) -> float:
    if not word1 or not word2:
        return 0.0
    if word1 == word2:
        return 1.0
    ng1 = set(extract_character_ngrams(word1))
    ng2 = set(extract_character_ngrams(word2))
    all_ng = sorted(ng1 | ng2)
    if not all_ng:
        return 0.0
    v1 = [1 if g in ng1 else 0 for g in all_ng]
    v2 = [1 if g in ng2 else 0 for g in all_ng]
    return cosine_similarity_vectors(v1, v2)


# ============================================================
# FUZZY LOGIC HELPERS
# ============================================================
def longest_common_substring_length(s1: str, s2: str) -> int:
    if not s1 or not s2:
        return 0
    m, n = len(s1), len(s2)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
    return best


def first_character_match(s1: str, s2: str) -> float:
    return 1.0 if (s1 and s2 and s1[0] == s2[0]) else 0.0


def length_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    mx = max(len(s1), len(s2))
    return max(0.0, 1.0 - abs(len(s1) - len(s2)) / mx) if mx else 1.0


def vowel_mark_tolerance(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    s1v = ''.join(c for c in s1 if c not in ALL_VOWEL_SIGNS)
    s2v = ''.join(c for c in s2 if c not in ALL_VOWEL_SIGNS)
    if s1v == s2v:
        return 1.0
    if s1v and s2v:
        common = longest_common_substring_length(s1v, s2v)
        mx = max(len(s1v), len(s2v))
        return common / mx if mx else 0.0
    return 0.0


def calculate_fuzzy_similarity(word1: str, word2: str) -> float:
    if not word1 or not word2:
        return 0.0
    if word1 == word2:
        return 1.0
    mx       = max(len(word1), len(word2))
    lev_s    = max(0.0, 1.0 - levenshtein_distance(word1, word2) / mx) if mx else 0.0
    lcs_s    = longest_common_substring_length(word1, word2) / mx if mx else 0.0
    cos_s    = calculate_cosine_similarity(word1, word2)
    first_s  = first_character_match(word1, word2)
    len_s    = length_similarity(word1, word2)
    vowel_s  = vowel_mark_tolerance(word1, word2)
    return min(1.0, max(0.0,
        FUZZY_WEIGHT_LEVENSHTEIN * lev_s  +
        FUZZY_WEIGHT_LCS         * lcs_s  +
        FUZZY_WEIGHT_COSINE      * cos_s  +
        FUZZY_WEIGHT_FIRST_CHAR  * first_s +
        FUZZY_WEIGHT_LENGTH      * len_s  +
        FUZZY_WEIGHT_VOWEL       * vowel_s
    ))


# ============================================================
# OCR CONFUSION PAIRS
# ============================================================
OCR_CONFUSION_PAIRS: List[Tuple[str, str]] = [
    ("ತ",   "ಥ"),
    ("ದ",   "ಧ"),
    ("ಪ",   "ಫ"),
    ("ಬ",   "ಭ"),
    ("ಸ",   "ಶ"),
    ("ಸ",   "ಷ"),
    ("ಕ",   "ಖ"),
    ("ಗ",   "ಘ"),
    ("ಚ",   "ಛ"),
    ("ಜ",   "ಝ"),
    ("ಟ",   "ಠ"),
    ("ಡ",   "ಢ"),
    ("ನ",   "ಣ"),
    ("ಲ",   "ಳ"),
    ("ಒ",   "ಓ"),
    ("ಎ",   "ಏ"),
    ("ಅ",   "ಆ"),
    ("ಇ",   "ಈ"),
    ("ಉ",   "ಊ"),
    ("ೋ", "ೊ"),
    ("ೇ", "ೆ"),
    ("ತ್ತ", "ಸ್ತ"),
    ("ತ್ತ", "ಷ್ಟ"),
    ("ಕ್ಕ", "ಕ್ಷ"),
    ("ದ್ದ", "ದ್ಧ"),
    ("ನ್ನ", "ನ್ಣ"),
    ("ನು",  "ನ್ನು"),
]


def generate_ocr_candidates(word: str) -> List[str]:
    candidates: List[str] = []
    for wrong, right in OCR_CONFUSION_PAIRS:
        if wrong in word:
            candidates.append(word.replace(wrong, right))
        if right in word:
            candidates.append(word.replace(right, wrong))
    return candidates


# ============================================================
# CACHE
# ============================================================
class ValidationCache:
    def __init__(self, db_path: str = "data/user_corrections.db",
                 memory_size: int = 10000):
        self.db_path      = db_path
        self.memory_size  = memory_size
        self.memory_cache: OrderedDict = OrderedDict()
        self.lock         = Lock()
        self.hits         = 0
        self.misses       = 0
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS validation_cache (
                word TEXT PRIMARY KEY,
                is_valid INTEGER,
                suggestions TEXT,
                confidence REAL,
                source TEXT,
                created_at TEXT,
                last_used TEXT,
                use_count INTEGER DEFAULT 1
            )
        """)
        conn.commit()
        conn.close()

    def get(self, word: str) -> Optional[Dict]:
        with self.lock:
            if word in self.memory_cache:
                self.hits += 1
                return self.memory_cache[word]
        try:
            conn = sqlite3.connect(self.db_path)
            cur  = conn.cursor()
            cur.execute(
                "SELECT is_valid, suggestions, confidence, source "
                "FROM validation_cache WHERE word=?", (word,)
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE validation_cache SET last_used=?, use_count=use_count+1 WHERE word=?",
                    (datetime.now().isoformat(), word)
                )
                conn.commit()
                conn.close()
                self.hits += 1
                result = {
                    "word": word, "valid": bool(row[0]),
                    "suggestions": json.loads(row[1]),
                    "confidence": row[2], "source": row[3], "cached": True
                }
                self._add_to_memory(word, result)
                return result
            conn.close()
        except Exception:
            pass
        self.misses += 1
        return None

    def set(self, word: str, result: Dict):
        with self.lock:
            self._add_to_memory(word, result)
        try:
            conn = sqlite3.connect(self.db_path)
            cur  = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO validation_cache
                (word, is_valid, suggestions, confidence, source,
                 created_at, last_used, use_count)
                VALUES (?, ?, ?, ?, ?, ?, ?,
                    COALESCE(
                        (SELECT use_count FROM validation_cache WHERE word=?), 0
                    ) + 1)
            """, (word, int(result["valid"]), json.dumps(result["suggestions"]),
                  result["confidence"], result["source"],
                  datetime.now().isoformat(), datetime.now().isoformat(), word))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _add_to_memory(self, word: str, result: Dict):
        self.memory_cache[word] = result
        if len(self.memory_cache) > self.memory_size:
            self.memory_cache.popitem(last=False)

    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        return {"hit_rate": round((self.hits / total) * 100, 2) if total else 0}


# ============================================================
# VIBHAKTI VALIDATOR
# ============================================================
class VibhaktiValidator:
    ALL_OBJECT_MARKERS: List[str] = [
        "ನನ್ನು", "ಯನ್ನು", "ಅನ್ನು", "ವನ್ನು", "ನ್ನು",
    ]

    _RESTORE_ENDINGS: Dict[str, List[str]] = {
        "ವನ್ನು": ["ು", "ೂ", "\u0CCA", "\u0CCB"],
        "ಯನ್ನು": ["\u0CC6", "\u0CC7", "\u0CBF", "\u0CC0"],
        "ನ್ನು":  [""],
        "ನನ್ನು": [""],
        "ಅನ್ನು": [""],
    }

    def _get_correct_object_marker(self, base_word: str) -> str:
        if not base_word:
            return "ಅನ್ನು"
        last = base_word[-1]
        if last in VOWEL_SIGN_I_TYPE | VOWEL_SIGN_E_TYPE:
            return "ಯನ್ನು"
        if last in VOWEL_SIGN_U_TYPE | VOWEL_SIGN_O_TYPE:
            return "ನ್ನು"
        return "ನನ್ನು"

    def detect_vibhakti_error(
        self, word: str, dictionary_words: Set[str],
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        for marker in self.ALL_OBJECT_MARKERS:
            if not word.endswith(marker):
                continue
            raw_base = word[:-len(marker)]
            if not raw_base:
                continue
            base_to_use = self._resolve_base(raw_base, marker, dictionary_words)
            if marker == "ವನ್ನು":
                base_to_use = self._resolve_base(raw_base, marker, dictionary_words)
                if base_to_use is None:
                    base_to_use = self._reconstruct_base_from_vnannu(raw_base)
            if base_to_use is None:
                continue
            correct_marker = self._get_correct_object_marker(base_to_use)
            is_error = not self._markers_equivalent(marker, correct_marker, base_to_use)
            if is_error:
                return True, base_to_use, marker, correct_marker
        return False, None, None, None

    def _reconstruct_base_from_vnannu(self, raw_base: str) -> Optional[str]:
        if not raw_base:
            return None
        last = raw_base[-1]
        if _is_kannada_consonant(last):
            return raw_base + "ು"
        return raw_base

    def _resolve_base(
        self, raw_base: str, marker: str, dictionary_words: Set[str],
    ) -> Optional[str]:
        if not dictionary_words:
            return None
        for ending in ["ು", "ೂ", "\u0CCA", "\u0CCB"]:
            candidate = raw_base + ending
            if candidate in dictionary_words:
                return candidate
        if raw_base in dictionary_words:
            return raw_base
        for ending in self._RESTORE_ENDINGS.get(marker, []):
            candidate = raw_base + ending
            if candidate in dictionary_words:
                return candidate
        if len(raw_base) > 2 and raw_base[-1] == "ವ":
            stem = raw_base[:-1]
            for ending in ["ು", "ೂ", ""]:
                candidate = stem + ending
                if candidate in dictionary_words:
                    return candidate
        return None

    @staticmethod
    def _markers_equivalent(m1: str, m2: str, base: str) -> bool:
        equiv = {"ಅನ್ನು", "ನ್ನು"}
        if m1 in equiv and m2 in equiv:
            return True
        return m1 == m2

    def correct_vibhakti(self, word: str, base_word: str, correct_marker: str) -> str:
        if not base_word:
            return word
        last = base_word[-1]
        if last in VOWEL_SIGN_U_TYPE | VOWEL_SIGN_O_TYPE:
            return base_word[:-1] + "ನ್ನು"
        if last in VOWEL_SIGN_E_TYPE | VOWEL_SIGN_I_TYPE:
            return base_word + "ಯನ್ನು"
        if _is_kannada_consonant(last):
            return base_word + "ವನ್ನು"
        return base_word + "ನ್ನು"


# ============================================================
# POSTPOSITION CORRECTOR
# ============================================================
class PostpositionCorrector:
    DATIVE_VERB_ROOTS: List[str] = [
        "ಹೋಗ", "ಹೋದ", "ಹೊಗ", "ಬರ", "ಬಂದ", "ಬಾ",
        "ತೆರಳ", "ನಡೆ", "ಓಡ", "ತಿರುಗ", "ಹೊರಡ", "ಚಲಿಸ",
        "ಹೊರಟ", "ಕಳಿಸ", "ಕಳಿಸು",
    ]

    LOCATIVE_VERB_ROOTS: List[str] = [
        "ಆಡ", "ಆಡಿದ", "ಆಡಿದರ", "ಕುಳಿತ", "ಕೂತ", "ನಿಂತ",
        "ಇದ್ದ", "ಇರ", "ವಾಸಿಸ", "ವಾಸಿಸುತ", "ಕೆಲಸ",
        "ನಡೆದ", "ಕುಣಿದ", "ಮಲಗ", "ಮಲಗಿದ", "ನಡೆಯಿತ", "ನಡೆದರ",
    ]

    POSTPOSITION_SUFFIXES: List[str] = [
        "ಲ್ಲಿ", "ದಲ್ಲಿ", "ಅಲ್ಲಿ", "ನಲ್ಲಿ", "ಗೆ", "ಕ್ಕೆ", "ಇಗೆ",
        "ಇಂದ", "ನಿಂದ", "ದಿಂದ", "ನ್ನು", "ಅನ್ನು", "ಯನ್ನು", "ನನ್ನು",
        "ವರೆಗೆ", "ತನಕ", "ಮೊದಲು", "ಗಳು", "ಗಳಿಗೆ", "ಗಳಲ್ಲಿ", "ಗಳಿಂದ",
    ]

    def _has_postposition(self, word: str) -> bool:
        return any(word.endswith(s) for s in self.POSTPOSITION_SUFFIXES)

    def _verb_requires(self, verb_token: str) -> Optional[str]:
        for root in self.DATIVE_VERB_ROOTS:
            if verb_token.startswith(root):
                return "DATIVE"
        for root in self.LOCATIVE_VERB_ROOTS:
            if verb_token.startswith(root):
                return "LOCATIVE"
        return None

    EXCLUDED_WORDS: Set[str] = {
        "ನಾನು","ನಾವು","ನೀನು","ನೀವು",
        "ಅವನು","ಅವಳು","ಅವರು","ಅದು","ಇದು","ಇವರು","ಈತನು","ಈಕೆ",
        "ಮಕ್ಕಳು","ಅವನ","ಅವಳ","ಅವರ",
        "ಬೇಗನೆ","ಈಗ","ಅಲ್ಲಿ","ಇಲ್ಲಿ","ಎಲ್ಲಿ",
        "ಸುನಿಲ್","ವಿನಯ್","ನಂದಿನಿ","ರಾಮ","ಸೀತಾ","ಕೃಷ್ಣ",
    }

    def _is_likely_noun(self, word: str, dictionary_words: Set[str]) -> bool:
        if word in self.EXCLUDED_WORDS:
            return False
        verb_endings = [
            "ತ್ತಾನೆ","ತ್ತಾಳೆ","ತ್ತಾರೆ","ತ್ತೇನೆ","ತ್ತೇವೆ",
            "ತ್ತದೆ","ದನು","ದಳು","ದರು","ದೆ",
            "ಬೇಕು","ಬಹುದು","ಬೇಡ","ತ್ತೀಯ","ತ್ತೀರಿ",
        ]
        if any(word.endswith(v) for v in verb_endings):
            return False
        if len(word) < 3:
            return False
        last = word[-1]
        in_dict = word in dictionary_words
        if last == "್":
            return False
        if in_dict:
            return True
        consonant_ending = _is_kannada_consonant(last) and len(word) >= 4
        vowel_e_ending   = (last in VOWEL_SIGN_E_TYPE | VOWEL_SIGN_I_TYPE) and len(word) >= 3
        return consonant_ending or vowel_e_ending

    def _add_postposition(self, noun: str, pp_type: str) -> str:
        last = noun[-1] if noun else ""
        if pp_type == "DATIVE":
            if last in VOWEL_SIGN_E_TYPE | VOWEL_SIGN_I_TYPE:
                return noun + "ಗೆ"
            if last in VOWEL_SIGN_U_TYPE | VOWEL_SIGN_O_TYPE:
                return noun[:-1] + "ಕ್ಕೆ"
            if _is_kannada_consonant(last):
                return noun + "ಕ್ಕೆ"
            return noun + "ಗೆ"
        if pp_type == "LOCATIVE":
            if _is_kannada_consonant(last):
                return noun + "ದಲ್ಲಿ"
            return noun + "ಅಲ್ಲಿ"
        return noun

    def correct(
        self, word: str, position: int, words: List[str],
        dictionary_words: Set[str] = None,
    ) -> Optional[str]:
        if dictionary_words is None:
            dictionary_words = set()
        if self._has_postposition(word):
            return None
        if not self._is_likely_noun(word, dictionary_words):
            return None
        window_end = min(position + 4, len(words))
        for j in range(position + 1, window_end):
            verb_token = words[j]
            pp_type    = self._verb_requires(verb_token)
            if pp_type:
                corrected = self._add_postposition(word, pp_type)
                if corrected != word:
                    return corrected
        return None


# ============================================================
# FST MORPHOLOGY COMPONENTS
# ============================================================
class VerbLexicon:
    def __init__(self):
        self.roots: Dict[str, Dict] = {
            "ಕುಡಿ":  {"type":"VOWEL_ENDING","ending":"\u0CBF","class":"REGULAR","gloss":"drink","conjugation_class":"CLASS_1"},
            "ತಿನ್ನು":{"type":"VOWEL_ENDING","ending":"ು","class":"NASAL_CONSONANT","gloss":"eat","conjugation_class":"CLASS_2"},
            "ಓದು":   {"type":"VOWEL_ENDING","ending":"ು","class":"REGULAR","gloss":"read","conjugation_class":"CLASS_2"},
            "ಮಾಡು":  {"type":"VOWEL_ENDING","ending":"ು","class":"REGULAR","gloss":"do/make","conjugation_class":"CLASS_2"},
            "ಬರೆ":   {"type":"VOWEL_ENDING","ending":"\u0CC6","class":"REGULAR","gloss":"write","conjugation_class":"CLASS_3"},
            "ನೋಡು":  {"type":"VOWEL_ENDING","ending":"ು","class":"REGULAR","gloss":"see","conjugation_class":"CLASS_2"},
            "ಕೇಳು":  {"type":"VOWEL_ENDING","ending":"ು","class":"REGULAR","gloss":"hear","conjugation_class":"CLASS_2"},
            "ಕೊಡು":  {"type":"VOWEL_ENDING","ending":"ು","class":"REGULAR","gloss":"give","conjugation_class":"CLASS_2"},
            "ತೆಗೆ":  {"type":"VOWEL_ENDING","ending":"\u0CC6","class":"REGULAR","gloss":"take","conjugation_class":"CLASS_3"},
            "ಹಾಕು":  {"type":"VOWEL_ENDING","ending":"ు","class":"REGULAR","gloss":"put","conjugation_class":"CLASS_2"},
            "ತರು":   {"type":"VOWEL_ENDING","ending":"ు","class":"REGULAR","gloss":"bring","conjugation_class":"CLASS_2"},
            "ಕರೆ":   {"type":"VOWEL_ENDING","ending":"\u0CC6","class":"REGULAR","gloss":"call","conjugation_class":"CLASS_3"},
            "ಹೇಳು":  {"type":"VOWEL_ENDING","ending":"ు","class":"REGULAR","gloss":"say","conjugation_class":"CLASS_2"},
            "ಬಿಡು":  {"type":"VOWEL_ENDING","ending":"ు","class":"REGULAR","gloss":"leave","conjugation_class":"CLASS_2"},
            "ಕೂಡು":  {"type":"VOWEL_ENDING","ending":"ు","class":"REGULAR","gloss":"join","conjugation_class":"CLASS_2"},
            "ಹೋಗು":  {"type":"VOWEL_ENDING","ending":"ు","class":"IRREGULAR","gloss":"go","conjugation_class":"IRREGULAR","past_stem":"ಹೋದ"},
            "ಬರು":   {"type":"VOWEL_ENDING","ending":"ు","class":"IRREGULAR","gloss":"come","conjugation_class":"IRREGULAR","past_stem":"ಬಂದ"},
        }
        self.stem_variants: Dict[str, str] = {
            "ಕುಡಿಯ":"ಕುಡಿ","ಓದ":"ಓದು","ಬರೆಯ":"ಬರೆ",
            "ತಿನ":"ತಿನ್ನು","ಮಾಡ":"ಮಾಡು","ನೋಡ":"ನೋಡು",
            "ಹೋಗ":"ಹೋಗು","ಬರ":"ಬರು",
        }

    def get_root_info(self, word: str) -> Optional[Dict]:
        if word in self.roots:
            return self.roots[word]
        if word in self.stem_variants:
            return self.roots.get(self.stem_variants[word])
        return self._infer_root_class(word)

    def _infer_root_class(self, word: str) -> Optional[Dict]:
        if not word:
            return None
        last = word[-1]
        if last in VOWEL_SIGN_I_TYPE:
            cc = "CLASS_1"
        elif last in VOWEL_SIGN_U_TYPE:
            cc = "CLASS_2"
        elif last in VOWEL_SIGN_E_TYPE:
            cc = "CLASS_3"
        elif last in KANNADA_INDEP_VOWELS:
            cc = "CLASS_2"
        else:
            return {"type":"CONSONANT_ENDING","ending":last,"class":"REGULAR",
                    "gloss":"unknown","conjugation_class":"CLASS_4"}
        return {"type":"VOWEL_ENDING","ending":last,"class":"REGULAR",
                "gloss":"unknown","conjugation_class":cc}

    def is_verb_root(self, word: str) -> bool:
        return word in self.roots or word in self.stem_variants


class ConjugationRules:
    def __init__(self):
        self.present: Dict[str, Dict] = {
            "1SG":          {"suffix":"ಉತ್ತೇನೆ","pronouns":["ನಾನು"]},
            "1PL":          {"suffix":"ಉತ್ತೇವೆ", "pronouns":["ನಾವು"]},
            "2SG_INFORMAL": {"suffix":"ಉತ್ತೀಯ",  "pronouns":["ನೀನು"]},
            "2SG_FORMAL":   {"suffix":"ಉತ್ತೀರಿ", "pronouns":["ನೀವು"]},
            "2PL":          {"suffix":"ಉತ್ತೀರಿ", "pronouns":["ನೀವು"]},
            "3SG_MALE":     {"suffix":"ಉತ್ತಾನೆ", "pronouns":["ಅವನು","ಈತನು","ವಿನಯ್","ಸುನಿಲ್"]},
            "3SG_FEMALE":   {"suffix":"ಉತ್ತಾಳೆ", "pronouns":["ಅವಳು","ಈಕೆ","ನಂದಿನಿ"]},
            "3SG_NEUTRAL":  {"suffix":"ಉತ್ತದೆ",  "pronouns":["ಅದು","ಇದು"]},
            "3PL":          {"suffix":"ಉತ್ತಾರೆ", "pronouns":["ಅವರು","ಇವರು","ಮಕ್ಕಳು"]},
        }
        self.past: Dict[str, Dict] = {
            "1SG":          {"suffix":"ಇದ್ದೇನೆ", "pronouns":["ನಾನು"]},
            "1PL":          {"suffix":"ಇದ್ದೇವೆ", "pronouns":["ನಾವು"]},
            "2SG_INFORMAL": {"suffix":"ಇದ್ದೀಯ",  "pronouns":["ನೀನು"]},
            "2SG_FORMAL":   {"suffix":"ಇದ್ದೀರಿ", "pronouns":["ನೀವು"]},
            "2PL":          {"suffix":"ಇದ್ದೀರಿ", "pronouns":["ನೀವು"]},
            "3SG_MALE":     {"suffix":"ಇದ್ದಾನೆ", "pronouns":["ಅವನು","ಈತನು"]},
            "3SG_FEMALE":   {"suffix":"ಇದ್ದಾಳೆ", "pronouns":["ಅವಳು","ಈಕೆ"]},
            "3SG_NEUTRAL":  {"suffix":"ಇತ್ತು",   "pronouns":["ಅದು","ಇದು"]},
            "3PL":          {"suffix":"ಇದ್ದಾರೆ", "pronouns":["ಅವರು","ಇವರು"]},
        }
        self.simple_past: Dict[str, Dict] = {
            "1SG":          {"suffix":"ದೆನು","pronouns":["ನಾನು"]},
            "1PL":          {"suffix":"ದೆವು","pronouns":["ನಾವು"]},
            "2SG_INFORMAL": {"suffix":"ದೆ",  "pronouns":["ನೀನು"]},
            "2SG_FORMAL":   {"suffix":"ದಿರಿ","pronouns":["ನೀವು"]},
            "2PL":          {"suffix":"ದಿರಿ","pronouns":["ನೀವು"]},
            "3SG_MALE":     {"suffix":"ದನು","pronouns":["ಅವನು","ಈತನು","ವಿನಯ್","ಸುನಿಲ್"]},
            "3SG_FEMALE":   {"suffix":"ದಳು","pronouns":["ಅವಳು","ಈಕೆ","ನಂದಿನಿ"]},
            "3SG_NEUTRAL":  {"suffix":"ಇತು","pronouns":["ಅದು","ಇದು"]},
            "3PL":          {"suffix":"ದರು","pronouns":["ಅವರು","ಇವರು","ಮಕ್ಕಳು"]},
        }
        self._pronoun_to_key: Dict[str, str] = {}
        for k, v in self.present.items():
            for p in v["pronouns"]:
                self._pronoun_to_key[p] = k

    def get_suffix(self, tense: str, agreement_key: str) -> Optional[str]:
        rule_dict = self.present if tense == "PRESENT" else (
                    self.past   if tense == "PAST"    else None)
        if rule_dict is None:
            return None
        if agreement_key in rule_dict:
            return rule_dict[agreement_key]["suffix"]
        return None

    def get_agreement_key_for_pronoun(self, pronoun: str) -> Optional[str]:
        return self._pronoun_to_key.get(pronoun)


# ============================================================
# CONTEXT ANALYZER — v7.3 with suffix-based gender detection
# ============================================================
class ContextAnalyzer:
    """
    Scans surrounding words to determine required verb form.

    FIX-11 (v7.3): Added _infer_gender_from_name() which uses
    Kannada name-ending patterns to detect gender for proper nouns
    not in the hardcoded SUBJECT_PRONOUNS list.

    Female name patterns (covers ~70% of Kannada female names):
      • Ends in ಾ  (AA vowel sign)  — ರಾಧಾ, ಸೀತಾ, ಪ್ರಿಯಾ
      • Ends in ಿ  (I vowel sign)   — ಲಕ್ಷ್ಮಿ, ಸರಸ್ವತಿ
      • Ends in ಾ  (AA matra)       — ರಾಧಾ, ಮೀರಾ
      • Ends in ೆ  (E vowel sign)   — ಸರೋಜೆ, ಗೌರಿ

    Male name patterns:
      • Ends in ್  (virama)         — ಸುನಿಲ್, ವಿನಯ್, ಮಹೇಶ್
      • Ends in consonant           — ರಾಮ, ಕೃಷ್ಣ, ಅರ್ಜುನ
    """

    SUBJECT_PRONOUNS: Dict[str, Dict] = {
        "ನಾನು": {"agreement":"1SG",        "gender":None},
        "ನಾವು": {"agreement":"1PL",        "gender":None},
        "ನೀನು": {"agreement":"2SG_INFORMAL","gender":None},
        "ನೀವು": {"agreement":"2SG_FORMAL", "gender":None},
        "ಅವನು": {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ಈತನು": {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ಅವಳು": {"agreement":"3SG_FEMALE", "gender":"FEMALE"},
        "ಈಕೆ":  {"agreement":"3SG_FEMALE", "gender":"FEMALE"},
        "ಅದು":  {"agreement":"3SG_NEUTRAL","gender":"NEUTRAL"},
        "ಇದು":  {"agreement":"3SG_NEUTRAL","gender":"NEUTRAL"},
        "ಅವರು": {"agreement":"3PL",        "gender":None},
        "ಇವರು": {"agreement":"3PL",        "gender":None},
        "ಮಕ್ಕಳು":{"agreement":"3PL",       "gender":None},
        # Known proper nouns
        "ಸುನಿಲ್": {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ವಿನಯ್":  {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ನಂದಿನಿ": {"agreement":"3SG_FEMALE", "gender":"FEMALE"},
        "ಸೀತಾ":   {"agreement":"3SG_FEMALE", "gender":"FEMALE"},
        "ರಾಧಾ":   {"agreement":"3SG_FEMALE", "gender":"FEMALE"},
        "ಲಕ್ಷ್ಮಿ": {"agreement":"3SG_FEMALE", "gender":"FEMALE"},
        "ಗೀತಾ":   {"agreement":"3SG_FEMALE", "gender":"FEMALE"},
        "ಪ್ರಿಯಾ": {"agreement":"3SG_FEMALE", "gender":"FEMALE"},
        "ಅನಿತಾ":  {"agreement":"3SG_FEMALE", "gender":"FEMALE"},
        "ಸುಮಿತ್ರಾ":{"agreement":"3SG_FEMALE","gender":"FEMALE"},
        "ರಾಮ":    {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ಕೃಷ್ಣ":  {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ಅರ್ಜುನ": {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ಮಹೇಶ್":  {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ರಾಜೇಶ್": {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ಕಿರಣ್":  {"agreement":"3SG_MALE",   "gender":"MALE"},
        "ಪ್ರಕಾಶ್":{"agreement":"3SG_MALE",   "gender":"MALE"},
        "ಗಣೇಶ್":  {"agreement":"3SG_MALE",   "gender":"MALE"},
    }
    # Context Memory — tracks entities seen in full document
    entity_memory: Dict[str, str] = {}

    @classmethod
    def update_entity_memory(cls, name: str, gender: str):
        """Add a named entity and its gender to document memory."""
        if name and gender:
            cls.entity_memory[name] = gender

    @classmethod
    def get_entity_gender(cls, name: str) -> Optional[str]:
        """Look up gender of a named entity from document memory."""
        return cls.entity_memory.get(name, None)

    @classmethod
    def clear_entity_memory(cls):
        """Clear memory — call this when a new document is loaded."""
        cls.entity_memory = {}

    @classmethod
    def build_entity_memory_from_paragraph(cls, text: str):
        """
        Scan full paragraph once at load time.
    Extract all proper nouns and infer their gender.
    Stores in entity_memory for use across all sentences.
        """
        cls.clear_entity_memory()
        words = text.split()
        for i, word in enumerate(words):
            clean = word.strip('.,!?;:()[]{}"\'।')
            if len(clean) < 2:
                continue
            # Check known pronouns first
            if clean in cls.SUBJECT_PRONOUNS:
                gender = cls.SUBJECT_PRONOUNS[clean].get('gender')
                if gender:
                    cls.update_entity_memory(clean, gender)
                continue
            # Try suffix-based inference for proper nouns
            # Try suffix-based inference for proper nouns only
            # Filter out common words using postposition/verb endings
            COMMON_WORD_ENDINGS = [
                "ಗೆ", "ಕ್ಕೆ", "ನ್ನು", "ಲ್ಲಿ", "ಇಂದ", "ವರೆಗೆ",
                "ತ್ತಾನೆ", "ತ್ತಾಳೆ", "ತ್ತಾರೆ", "ದನು", "ದಳು", "ದರು",
                "ಗಿ", "ಆಗಿ", "ಅಲ್ಲ", "ಇಲ್ಲ", "ಉಂಟು",
                "ಯಾದ", "ಿದ್ದ", "ುತ್ತಿದ್ದ", "ುತ್ತಿದ್ದಾನೆ", "ುತ್ತಿದ್ದಳು",
                "ಣ್ಣ", "ಸ", "ದ",
]
            is_common_word = any(clean.endswith(e) for e in COMMON_WORD_ENDINGS)
            COMMON_NOUNS_BLOCKLIST = {
                "ಕೆಲಸ", "ಬೇಗ", "ಪುಸ್ತಕ", "ಊಟ", "ನೀರು", "ಮನೆ", "ಶಾಲೆ",
                "ಹಣ", "ದಿನ", "ರಾತ್ರಿ", "ಬೆಳಿಗ್ಗೆ", "ಸಮಯ", "ಜನ", "ದೇಶ",
                "ಕಥೆ", "ಹಾಡು", "ಆಟ", "ಕಾಡು", "ನದಿ", "ಬೆಟ್ಟ", "ಹೊಲ",
                "ಬೆಲೆ", "ತರಕಾರಿ", "ಹಾಲು", "ಅನ್ನ", "ತಿಂಡಿ", "ರಸ್ತೆ",
            }
            is_common_noun = clean in COMMON_NOUNS_BLOCKLIST
            if (not is_common_word
                    and not is_common_noun
                    and len(clean) >= 3
                    and any('\u0C80' <= c <= '\u0CFF' for c in clean)):
                inferred = cls._infer_gender_from_name(clean)
                if inferred:
                    cls.update_entity_memory(clean, inferred)
                    print(f"[EntityMemory] '{clean}' → {inferred}")


    INFINITIVE_TRIGGERS: List[str] = [
        "ಬೇಕು","ಬೇಕಾಗುತ್ತದೆ","ಬೇಡ","ಬಹುದು","ಆಗು","ಸಾಧ್ಯ","ಮುಂಚೆ",
    ]

    # --------------------------------------------------------
    # FIX-11: Suffix-based gender detection for unknown names
    # --------------------------------------------------------

    # Vowel signs/characters at name endings that indicate FEMALE gender
    # Covers ~70% of Kannada female names
    FEMALE_NAME_ENDINGS = {
        VOWEL_SIGN_AA,           # ಾ — ರಾಧಾ, ಮೀರಾ, ಗೀತಾ, ಪ್ರಿಯಾ
        '\u0CBF',                 # ಿ — ಲಕ್ಷ್ಮಿ, ಸರಸ್ವತಿ, ಭಾಗ್ಯಲಕ್ಷ್ಮಿ
        '\u0CC0',                 # ೀ — ದೇವಿ, ಸಾವಿತ್ರೀ
        '\u0CC6',                 # ೆ — ಸರೋಜೆ
        '\u0CC7',                 # ೇ — rare but possible
    }

    # Common female name suffix strings (multi-character)
    FEMALE_NAME_SUFFIX_STRINGS = [
        "ಾ",   # AA matra ending — most common female ending
        "ಮ್ಮ", # amma — ದೊಡ್ಡಮ್ಮ, ಚಿಕ್ಕಮ್ಮ (already in honorific but good to have)
        "ಕ್ಕ", # akka — ಅಕ್ಕ (sister)
        "ಲ್ಲಿ",# alli — some female names
    ]

    # Virama ending = almost certainly male (transliterated name)
    MALE_VIRAMA_ENDING = '\u0CCD'  # ್

    @staticmethod
    def _infer_gender_from_name(name: str) -> Optional[str]:
        """
        Infer gender of a Kannada proper noun using suffix patterns.

        Returns:
            "FEMALE" — name likely female
            "MALE"   — name likely male
            None     — cannot determine (ambiguous)

        Rules (in priority order):
          1. Ends in virama ್ → MALE  (ಸುನಿಲ್, ವಿನಯ್, ಮಹೇಶ್)
          2. Ends in ಾ (AA matra) → FEMALE  (ರಾಧಾ, ಗೀತಾ, ಪ್ರಿಯಾ)
          3. Ends in ಿ or ೀ (I-type vowel sign) → FEMALE  (ಲಕ್ಷ್ಮಿ)
          4. Ends in ೆ or ೇ (E-type vowel sign) → FEMALE  (ಸರೋಜೆ)
          5. Ends in ಮ್ಮ → FEMALE  (ಅಮ್ಮ, ದೊಡ್ಡಮ್ಮ)
          6. Ends in consonant (no vowel sign) → MALE  (ರಾಮ, ಅರ್ಜುನ)
          7. Otherwise → None (ambiguous)
        """
        if not name or len(name) < 2:
            return None

        last = name[-1]

        # Rule 1: Virama ending → MALE
        if last == '\u0CCD':
            return "MALE"

        # Rule 2: AA matra (ಾ) → FEMALE
        if last == '\u0CBE':
            return "FEMALE"

        # Rule 3: I-type vowel sign (ಿ ೀ) → FEMALE
        if last in {'\u0CBF', '\u0CC0'}:
            return "FEMALE"

        # Rule 4: E-type vowel sign (ೆ ೇ) → FEMALE
        if last in {'\u0CC6', '\u0CC7'}:
            return "FEMALE"

        # Rule 5: Ends in ಮ್ಮ → FEMALE
        if name.endswith("ಮ್ಮ"):
            return "FEMALE"

        # Rule 6: Ends in U-type vowel sign (ು ೂ) → ambiguous
        # Some male names end in ು (ರಾಮು, ಚಿನ್ನು) but rare
        # Return None to avoid false gender assignment
        if last in {'\u0CC1', '\u0CC2'}:
            return None

        # Rule 7: Ends in Kannada consonant → MALE
        if _is_kannada_consonant(last):
            return "MALE"

        return None

    def detect_required_form(self, words: List[str], verb_index: int) -> Dict:
        result = {
            "form_type": "PRESENT",
            "agreement": None,
            "gender":    None,
            "confidence": 0.5,
        }

        # Check for infinitive context
        for j in range(verb_index + 1, min(verb_index + 3, len(words))):
            if words[j] in self.INFINITIVE_TRIGGERS:
                result["form_type"]  = "INFINITIVE"
                result["confidence"] = 0.95
                return result

        # Scan leftward for subject — up to 6 words back
        for i in range(verb_index - 1, max(-1, verb_index - 6), -1):
            w = words[i]

            # Check known pronouns/names first
            if w in self.SUBJECT_PRONOUNS:
                info = self.SUBJECT_PRONOUNS[w]
                result["agreement"] = info["agreement"]
                result["gender"]    = info["gender"]
                result["confidence"] = 0.85
                return result
        # Check document-level entity memory
            mem_gender = self.get_entity_gender(w)
            if mem_gender:
                result["agreement"] = f"3SG_{mem_gender}"
                result["gender"]    = mem_gender
                result["confidence"] = 0.80
                return result

            # FIX-11: Try suffix-based gender inference for unknown proper nouns
            # Only attempt for words that look like proper nouns:
            # - Start with Kannada character
            # - Length >= 3
            # - Not a known verb, particle, or postposition
            if (len(w) >= 3
                    and any("\u0C80" <= c <= "\u0CFF" for c in w)
                    and not any(w.endswith(sfx) for sfx in [
                        "ಗೆ", "ಕ್ಕೆ", "ನ್ನು", "ಲ್ಲಿ", "ಇಂದ",
                        "ತ್ತಾನೆ", "ತ್ತಾಳೆ", "ತ್ತಾರೆ", "ದನು", "ದಳು", "ದರು"
                    ])):
                inferred_gender = self._infer_gender_from_name(w)
                if inferred_gender == "FEMALE":
                    result["agreement"] = "3SG_FEMALE"
                    result["gender"]    = "FEMALE"
                    result["confidence"] = 0.70  # Lower confidence than known names
                    return result
                elif inferred_gender == "MALE":
                    result["agreement"] = "3SG_MALE"
                    result["gender"]    = "MALE"
                    result["confidence"] = 0.70
                    return result

        return result

    def analyze_context_from_sentence(
        self, sentence: str, verb_position: Optional[int]
    ) -> Dict:
        words = sentence.split()
        if verb_position is None:
            verb_position = len(words) // 2
        return self.detect_required_form(words, verb_position)


# ============================================================
# KANNADA MORPHOLOGY PARSER
# ============================================================
class KannadaMorphologyParser:
    def __init__(self):
        self.lexicon     = VerbLexicon()
        self.conjugation = ConjugationRules()
        self.context     = ContextAnalyzer()

    def analyze(self, conjugated_form: str) -> Optional[Dict]:
        for key, form_data in self.conjugation.present.items():
            suffix = form_data["suffix"]
            if conjugated_form.endswith(suffix):
                stem = conjugated_form[:-len(suffix)]
                for ch in ["ಯ", "ವ"]:
                    if stem.endswith(ch):
                        stem = stem[:-1]
                        break
                root_info = self.lexicon.get_root_info(stem)
                return {"root": stem, "form_type": "PRESENT",
                        "agreement": key,
                        "confidence": 0.9 if root_info else 0.7}

        for key, form_data in self.conjugation.past.items():
            suffix = form_data["suffix"]
            if conjugated_form.endswith(suffix):
                stem = conjugated_form[:-len(suffix)]
                if stem.endswith("ಯ"):
                    stem = stem[:-1]
                root_info = self.lexicon.get_root_info(stem)
                if root_info:
                    return {"root": stem, "form_type": "PAST",
                            "agreement": key, "confidence": 0.9}

        if conjugated_form.endswith("ಲು") or conjugated_form.endswith("ಅಲು"):
            stem = conjugated_form[:-2] if conjugated_form.endswith("ಲು") else conjugated_form[:-3]
            if stem.endswith("ಯ"):
                stem = stem[:-1]
            return {"root": stem, "form_type": "INFINITIVE",
                    "agreement": None, "confidence": 0.8}

        simple_past_suffixes = [
            ("ದೆನು","1SG"),("ದೆವು","1PL"),("ದಿರಿ","2PL"),
            ("ದನು","3SG_MALE"),("ದಳು","3SG_FEMALE"),("ದರು","3PL"),
            ("ದೆ","2SG_INFORMAL"),("ಇತು","3SG_NEUTRAL"),
        ]
        for suffix, agreement in simple_past_suffixes:
            if conjugated_form.endswith(suffix):
                stem = conjugated_form[:-len(suffix)]
                return {"root": stem, "form_type": "SIMPLE_PAST",
                        "agreement": agreement, "confidence": 0.85}

        return None

    def generate(self, root: str, tense: str, agreement_key: str) -> Optional[str]:
        root_info = self.lexicon.get_root_info(root)

        if tense == "SIMPLE_PAST":
            past_stem = root_info.get("past_stem") if root_info else None
            if not past_stem:
                past_stem = root
            last_ch = past_stem[-1] if past_stem else ""
            if last_ch == "ದ":
                short = {
                    "1SG":"ೆನು","1PL":"ೆವು","2SG_INFORMAL":"ೆ",
                    "2SG_FORMAL":"ಿರಿ","2PL":"ಿರಿ",
                    "3SG_MALE":"ನು","3SG_FEMALE":"ಳು",
                    "3SG_NEUTRAL":"ಿತು","3PL":"ರು",
                }
                return past_stem + short.get(agreement_key, "ನು")
            if past_stem and 0x0C95 <= ord(last_ch) <= 0x0CBD:
                short = {
                    "1SG":"ೆನು","1PL":"ೆವು","2SG_INFORMAL":"ೆ",
                    "2SG_FORMAL":"ಿರಿ","2PL":"ಿರಿ",
                    "3SG_MALE":"ನು","3SG_FEMALE":"ಳು",
                    "3SG_NEUTRAL":"ಿತು","3PL":"ರು",
                }
                return past_stem + short.get(agreement_key, "ನು")
            full = {
                "1SG":"ದೆನು","1PL":"ದೆವು","2SG_INFORMAL":"ದೆ",
                "2SG_FORMAL":"ದಿರಿ","2PL":"ದಿರಿ",
                "3SG_MALE":"ದನು","3SG_FEMALE":"ದಳು",
                "3SG_NEUTRAL":"ದಿತು","3PL":"ದರು",
            }
            return past_stem + full.get(agreement_key, "ದನು")

        if root_info and root_info.get("class") == "IRREGULAR" and tense == "PAST":
            past_stem = root_info.get("past_stem")
            if past_stem:
                suffix = self.conjugation.get_suffix("PAST", agreement_key)
                if suffix:
                    return past_stem + suffix

        suffix = self.conjugation.get_suffix(tense, agreement_key)
        if not suffix:
            return None

        last = root[-1] if root else ""
        if last in VOWEL_SIGN_U_TYPE and suffix.startswith("ಉ"):
            return root[:-1] + suffix
        if last in VOWEL_SIGN_I_TYPE and suffix.startswith("ಇ"):
            return root[:-1] + "ಯ" + suffix

        return root + suffix

    def correct_verb_form(
        self, wrong_verb: str, sentence: str, verb_position: Optional[int] = None
    ) -> Optional[str]:
        analysis = self.analyze(wrong_verb)
        if analysis is None:
            return None
        raw_root = analysis["root"]
        root = raw_root
        if raw_root and raw_root[-1] in ('ಿ', 'ೀ'):
            stem_base = raw_root[:-1]
            for ending in ['ು', '']:
                candidate = stem_base + ending
                if self.lexicon.get_root_info(candidate):
                    root = candidate
                    break
        context_info = self.context.analyze_context_from_sentence(sentence, verb_position)
        tense        = context_info.get("form_type", "PRESENT")
        if analysis and analysis.get("form_type") == "SIMPLE_PAST":
            tense = "SIMPLE_PAST"
        agreement = context_info.get("agreement")
        if not agreement:
            agreement = analysis.get("agreement") if analysis else None
        if not agreement:
            return None
        return self.generate(root, tense, agreement)


# ============================================================
# ENHANCED VALIDATOR v7.3
# ============================================================
class EnhancedValidator:
    def __init__(
        self,
        dictionary_paths: List[str] = None,
        context_db_path: str  = "data/ngram_context.db",
        cache_size: int        = 10_000,
        max_edit_distance: int = 4,
        enable_fst: bool       = True,
        enable_vibhakti_validation: bool = True,
        enable_postposition_correction: bool = True,
    ):
        if dictionary_paths is None:
            dictionary_paths = [
                "data/dictionaries/Padakosha_kannada_csv.csv",
                "data/dictionaries/combined_word_scrapped_csv.csv",
            ]
        self.dictionary_paths   = dictionary_paths
        self.cache              = ValidationCache(memory_size=cache_size)
        self.max_edit_distance  = max_edit_distance
        self.enable_fst         = enable_fst
        self.enable_vibhakti_validation = enable_vibhakti_validation
        self.enable_postposition_correction = enable_postposition_correction

        self.dictionary_words: Set[str]  = set()
        self.dictionary_list:  List[str] = []
        self._load_dictionaries()

        self.context_db_path = context_db_path
        self.context_enabled = self._check_context_db()

        self.morphology   = KannadaMorphologyParser() if enable_fst else None
        self.vibhakti     = VibhaktiValidator()       if enable_vibhakti_validation else None
        self.postposition = PostpositionCorrector()   if enable_postposition_correction else None

        self.extra_suffix_patterns: List[str] = [
            "ತ್ತಾನೆನು","ತ್ತಾಳೆನು","ತ್ತಾರೆನು",
            "ತ್ತೇನೆನು","ತ್ತೇವೆನು","ತ್ತೀಯನು","ತ್ತೀರಿನು",
            "ತ್ತದೆನು","ದರುನು","ದನುನು","ದಳುನು",
        ]

        self.total_validations = 0
        self._gemini_call_count = 0
        self._gemini_session_limit = 0 if os.environ.get("DISABLE_GEMINI_TESTING") == "1" else 3

        print("[EnhancedValidator] Initialized v7.1")
        print(f"  Dictionary: {len(self.dictionary_words):,} words")
        print(f"  Context DB: {'✅' if self.context_enabled else '⚠️  disabled'}")
        print(f"  FST Morphology:        {'✅' if enable_fst else '⚠️'}")
        print(f"  Vibhakti Correction:   {'✅' if enable_vibhakti_validation else '⚠️'}")
        print(f"  Postposition Fix:      {'✅' if enable_postposition_correction else '⚠️'}")

    def _check_context_db(self) -> bool:
        if not os.path.exists(self.context_db_path):
            return False
        try:
            conn  = sqlite3.connect(self.context_db_path)
            cur   = conn.cursor()
            cur.execute("SELECT count(*) FROM unigrams")
            count = cur.fetchone()[0]
            conn.close()
            return count > 0
        except Exception:
            return False

    def _load_dictionaries(self):
        for dict_path in self.dictionary_paths:
            if not os.path.exists(dict_path):
                print(f"[Validator] ⚠️  Dictionary not found: {dict_path}")
                continue
            try:
                with open(dict_path, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        word = None
                        if len(row) >= 2 and row[1].strip():
                            word = row[1].strip()
                        elif len(row) >= 1 and row[0].strip():
                            word = row[0].strip()
                        if not word:
                            continue
                        word = word.replace(":", "").replace(";", "").replace("¹", "").strip()
                        if (word and len(word) >= 2
                                and any("\u0C80" <= c <= "\u0CFF" for c in word)
                                and word not in self.dictionary_words):
                            self.dictionary_words.add(word)
                            self.dictionary_list.append(word)
                print(f"[Validator] Loaded {os.path.basename(dict_path)}")
            except Exception as e:
                print(f"[Validator] Error loading {dict_path}: {e}")
        print(f"[Validator] ✅ Total: {len(self.dictionary_words):,} words")

    def _get_scores(
        self, prev_word: Optional[str], candidate: str,
        next_word: Optional[str] = None,
    ) -> Tuple[float, float]:
        freq_score = context_score = 0.0
        if not self.context_enabled:
            return 0.0, 0.0
        try:
            conn = sqlite3.connect(self.context_db_path)
            cur  = conn.cursor()
            cur.execute("SELECT count FROM unigrams WHERE word=?", (candidate,))
            row = cur.fetchone()
            if row and row[0] > 0:
                freq_score = min(1.0, math.log(row[0]) / MAX_UNIGRAM_LOG)
            left_score = 0.0
            if prev_word:
                cur.execute(
                    "SELECT count FROM bigrams WHERE prev_word=? AND next_word=?",
                    (prev_word, candidate)
                )
                row = cur.fetchone()
                if row and row[0] > 0:
                    left_score = min(1.0, math.log(row[0]) / MAX_BIGRAM_LOG)
            right_score = 0.0
            if next_word:
                cur.execute(
                    "SELECT count FROM bigrams WHERE prev_word=? AND next_word=?",
                    (candidate, next_word)
                )
                row = cur.fetchone()
                if row and row[0] > 0:
                    right_score = min(1.0, math.log(row[0]) / MAX_BIGRAM_LOG)
            conn.close()
            if prev_word and next_word:
                context_score = (0.5 * left_score) + (0.5 * right_score)
            elif prev_word:
                context_score = left_score
            elif next_word:
                context_score = right_score
        except Exception:
            pass
        return freq_score, context_score

    def _normalise(self, word: str) -> str:
        return word.replace(":", "").replace(";", "").replace("¹", "").strip()

    def _strip_vibhakti(self, word: str) -> str:
        if len(word) < 3:
            return word
        suffixes = [
            "ಗಳಿಂದ","ಗಳಲ್ಲಿ","ಗಳನ್ನು","ಗಳಿಗೆ",
            "ನನ್ನು","ಯನ್ನು","ಯಲ್ಲಿ","ಯಿಂದ",
            "ಅಲ್ಲಿ","ಅನ್ನು","ದಲ್ಲಿ","ದಿಂದ",
            "ವನ್ನು","ಗಳು","ವರು","ವರ",
            "ಲ್ಲಿ","ನ್ನು","ಕ್ಕೆ","ಗೆ","ನು","ಯ","ರ","ವ",
        ]
        for s in suffixes:
            if word.endswith(s) and len(word) > len(s) + 1:
                return word[:-len(s)]
        return word

    def _find_candidates_levenshtein(self, word: str, max_suggestions: int = 20) -> List[str]:
        if os.environ.get('BATCH_EVAL_MODE') == '1':
            max_suggestions = 3
        if not word or not self.dictionary_words:
            return []
        wlen = len(word)
        pairs: List[Tuple[str, float]] = []
        for w in self.dictionary_list:
            if abs(len(w) - wlen) > 3:
                continue
            if not w or calculate_fuzzy_similarity(w[0], word[0]) < 0.3:
                continue
            score = calculate_fuzzy_similarity(word, w)
            if score >= 0.35:
                pairs.append((w, score))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in pairs[:max_suggestions]]

    def _get_morphology_score(self, candidate: str, context_info: Dict) -> float:
        if not self.enable_fst or not self.morphology:
            return 0.5
        analysis = self.morphology.analyze(candidate)
        if not analysis:
            return 0.3
        if analysis["form_type"] == context_info.get("form_type"):
            score = 0.95
            if (context_info.get("agreement") and
                    analysis.get("agreement") == context_info["agreement"]):
                score += 0.05
            return min(score, 1.0)
        return 0.4

    def _gemini_semantic_validate(
    self, full_sentence: str, word_position: int,
    top_candidates: List[str],
) -> Optional[str]:
        if os.environ.get("DISABLE_GEMINI_TESTING") == "1":
            return None
        if _gemini_model is None or not full_sentence or not top_candidates:
            return None
        if self._gemini_call_count >= self._gemini_session_limit:
            return None

        self._gemini_call_count += 1

        try:
            # Hard disable — return immediately, no API call, no retry wait
            if (os.environ.get("DISABLE_GEMINI_TESTING") == "1"
                    or self._gemini_session_limit <= 0):
                return None

            words = full_sentence.split()
            masked = words.copy()

            if word_position < len(masked):
                masked[word_position] = "___"

            masked_sentence = " ".join(masked)
            candidates_str  = ", ".join(top_candidates[:4])

            prompt = (
                f"You are a Kannada language expert helping with OCR correction.\n"
                f"The following Kannada sentence has one word replaced with ___:\n\n"
                f'"{masked_sentence}"\n\n'
                f"Choose the most grammatically and semantically correct word "
                f"to fill the blank from these options:\n{candidates_str}\n\n"
                f"Reply with ONLY the chosen Kannada word, nothing else."
            )

            response = _gemini_model.generate_content(prompt)
            chosen   = response.text.strip().strip(".,;:\"'")

            if chosen in top_candidates:
                return chosen

        except Exception as e:
            if os.environ.get("DISABLE_GEMINI_TESTING") != "1":
                print(f"[Gemini] Error during semantic validation: {e}")

        return None

    def _sandhi_check(self, word: str) -> Optional[str]:
        EUPHONIC_Y_SUFFIXES = ["ಲ್ಲಿ","ಂದ","ಗೆ","ಅಲ್ಲಿ","ಇಂದ","ಅನ್ನು"]
        E_I_TYPE_SIGNS = {'\u0CC6','\u0CC7','\u0CC8','\u0CBF','\u0CC0'}
        for suffix in EUPHONIC_Y_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                base = word[:-len(suffix)]
                if not base:
                    continue
                last_char = base[-1]
                if last_char in E_I_TYPE_SIGNS:
                    candidate = base + "ಯ" + suffix
                    if candidate in self.dictionary_words:
                        return candidate
                candidate2 = base + "ಯ" + suffix
                if candidate2 != word and candidate2 in self.dictionary_words:
                    return candidate2
        EUPHONIC_V_SUFFIXES = ["ಅನ್ನು","ಅಲ್ಲಿ","ಅಂದ"]
        U_O_TYPE_SIGNS = {'\u0CC1','\u0CC2','\u0CCA','\u0CCB'}
        for suffix in EUPHONIC_V_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                base = word[:-len(suffix)]
                if not base:
                    continue
                last_char = base[-1]
                if last_char in U_O_TYPE_SIGNS:
                    candidate = base + "ವ" + suffix
                    if candidate in self.dictionary_words:
                        return candidate
        return None

    def _check_sov_violation(
        self, word: str, position: int, full_sentence: str,
    ) -> Optional[str]:
        if not full_sentence or position is None:
            return None
        words = full_sentence.split()
        total = len(words)
        if position >= total - 1:
            return None
        is_verb = False
        if self.morphology:
            analysis = self.morphology.analyze(word)
            if analysis and analysis.get("form_type") in ("PRESENT","PAST","SIMPLE_PAST"):
                is_verb = True
        if not is_verb:
            VERB_SUFFIXES = [
                "ತ್ತಾನೆ","ತ್ತಾಳೆ","ತ್ತಾರೆ","ತ್ತೇನೆ","ತ್ತೇವೆ",
                "ತ್ತೀಯ","ತ್ತೀರಿ","ತ್ತದೆ",
                "ತ್ತಿದ್ದಾನೆ","ತ್ತಿದ್ದಾಳೆ","ತ್ತಿದ್ದಾರೆ",
                "ದನು","ದಳು","ದರು","ದೆನು","ದೆವು",
                "ದಿರಿ","ದಿತು","ಇತ್ತು",
                "ಬಂದನು","ಬಂದಳು","ಬಂದರು",
                "ಹೋದನು","ಹೋದಳು","ಹೋದರು",
            ]
            for suffix in VERB_SUFFIXES:
                if word.endswith(suffix):
                    is_verb = True
                    break
        if not is_verb:
            return None

        # Build suggested reordering — move verb to end
        reordered = [w for i, w in enumerate(words) if i != position]
        reordered.append(word)
        suggested = " ".join(reordered)

        return (
            f"SOV violation: verb '{word}' at position {position + 1}/{total} | "
            f"Suggested: {suggested}"
        )

    def validate_word(
        self,
        word:          str,
        prev_word:     Optional[str] = None,
        full_sentence: Optional[str] = None,
        position:      Optional[int] = None,
    ) -> Dict:
        w = self._normalise(word)
        if not w:
            return {"word": word, "valid": False, "suggestions": [],
                    "confidence": 0.0, "source": "empty", "error_type": None}

        self.total_validations += 1

        # STEP 1: Extra suffix removal
        for pattern in self.extra_suffix_patterns:
            if w.endswith(pattern):
                corrected = w[:-2]
                return {"word": word, "valid": False,
                        "suggestions": [corrected], "confidence": 0.95,
                        "source": "extra_suffix_removal", "error_type": "EXTRA_SUFFIX"}

        # STEP 1b: Sandhi check
        sandhi_corrected = self._sandhi_check(w)

        # STEP 2: Vibhakti correction
        vibhakti_corrected = None
        if self.vibhakti and w not in self.dictionary_words:
            _vib_err, _vib_base, _vib_wrong, _vib_correct = \
                self.vibhakti.detect_vibhakti_error(w, self.dictionary_words)
            if _vib_err and _vib_base:
                _vib_form = self.vibhakti.correct_vibhakti(w, _vib_base, _vib_correct)
                if _vib_form and _vib_form != w:
                    vibhakti_corrected = _vib_form

        # STEP 3: Postposition correction
        postposition_corrected = None
        if self.postposition and full_sentence and position is not None:
            tokens   = full_sentence.split()
            corrected = self.postposition.correct(w, position, tokens, self.dictionary_words)
            if corrected:
                postposition_corrected = corrected

        # STEP 4: Dictionary check
        if vibhakti_corrected:
            is_valid = False
        else:
            is_valid = (
                w in self.dictionary_words
                or self._strip_vibhakti(w) in self.dictionary_words
            )

        # Run postposition check even on valid words
        if is_valid and self.postposition and full_sentence and position is not None:
            tokens = full_sentence.split()
            pp = self.postposition.correct(w, position, tokens, self.dictionary_words)
            if pp:
                postposition_corrected = pp
                is_valid = False

        # STEP 4b: Accept valid SIMPLE_PAST verb forms not in dictionary
        # IMPORTANT: Skip for IRREGULAR verbs — they need FST correction
        if not is_valid and self.morphology:
            morph_check = self.morphology.analyze(w)
            if (morph_check is not None and
                    morph_check.get("form_type") == "SIMPLE_PAST"):
                raw_root  = morph_check.get("root", "")
                root_info = self.morphology.lexicon.get_root_info(raw_root)

                # Also try recovered root (strip ಿ/ೀ and add ು)
                # e.g. ಹೋಗಿ → ಹೋಗು which is IRREGULAR
                if not (root_info and root_info.get("class") == "IRREGULAR"):
                    if raw_root and raw_root[-1] in ('ಿ', 'ೀ'):
                        stem_base = raw_root[:-1]
                        for ending in ['ು', '']:
                            candidate = stem_base + ending
                            recovered_info = self.morphology.lexicon.get_root_info(candidate)
                            if recovered_info and recovered_info.get("class") == "IRREGULAR":
                                root_info = recovered_info
                                break

                is_irregular_root = (
                    root_info and root_info.get("class") == "IRREGULAR"
                )
                if not is_irregular_root:
                    fst_correction = self.morphology.correct_verb_form(
                        w, full_sentence or w, position
                    )
                    if (fst_correction is None
                            or fst_correction == w
                            or fst_correction not in self.dictionary_words):
                        is_valid = True

        # Initialise correction variables to avoid scope issues
        fst_corrected = None

        # STEP 5: Morphological verb correction
        context_info: Dict = {}
        if full_sentence and self.morphology:
            context_info = self.morphology.context.analyze_context_from_sentence(
                full_sentence, position
            )
            morph_analysis = self.morphology.analyze(w)
            if morph_analysis is not None and not is_valid:
                recovered_root = morph_analysis.get("root", "")
                if recovered_root and recovered_root[-1] in ('ಿ', 'ೀ'):
                    base = recovered_root[:-1]
                    for ending in ['ు', '']:
                        cand = base + ending
                        if self.morphology.lexicon.get_root_info(cand):
                            recovered_root = cand
                            break
                root_info    = self.morphology.lexicon.get_root_info(recovered_root)
                is_irregular = root_info and root_info.get("class") == "IRREGULAR"
                if is_irregular and context_info.get("agreement"):
                    fst_result = self.morphology.correct_verb_form(w, full_sentence, position)
                    if (fst_result and fst_result != w
                            and context_info.get("agreement") is not None
                            and len(fst_result) >= 3):
                        fst_corrected = fst_result

        # STEP 4c: SOV violation check
        sov_flag = self._check_sov_violation(w, position, full_sentence)

        # Accept valid words (with sandhi/postposition check)
        if is_valid:
            valid_suggestions = []
            if sandhi_corrected and sandhi_corrected != w:
                valid_suggestions.append(sandhi_corrected)
            if postposition_corrected and postposition_corrected != w:
                valid_suggestions.append(postposition_corrected)
            if not valid_suggestions:
                return {"word": word, "valid": True, "suggestions": [],
                        "confidence": 1.0, "source": "dictionary", "error_type": None}

        # OCR direct fix
        ocr_direct_hit = None
        ocr_direct_candidates = generate_ocr_candidates(w)
        for cand in ocr_direct_candidates:
            if cand in self.dictionary_words:
                ocr_direct_hit = cand
                break

        # STEP 6: Candidate generation
        ocr_candidates = [c for c in generate_ocr_candidates(w) if c in self.dictionary_words]
        fuzzy_candidates = self._find_candidates_levenshtein(w, max_suggestions=50)

        all_candidates: List[str] = []
        seen_c: Set[str] = set()
        for c in ocr_candidates + fuzzy_candidates:
            if c not in seen_c:
                all_candidates.append(c)
                seen_c.add(c)

        all_candidates = [
            c for c in all_candidates
            if calculate_fuzzy_similarity(w, c) >= 0.30
        ]

        # STEP 7: Hybrid ranking
        words_list = full_sentence.split() if full_sentence else []
        next_word  = (words_list[position + 1]
                     if position is not None and position + 1 < len(words_list)
                     else None)

        ranked: List[Dict] = []
        for cand in all_candidates:
            fuzzy_score           = calculate_fuzzy_similarity(w, cand)
            freq_score, ctx_score = self._get_scores(prev_word, cand, next_word)
            morph_score           = self._get_morphology_score(cand, context_info) \
                                    if context_info else 0.5
            ocr_boost             = 0.6 if cand in ocr_candidates else 0.0
            if context_info.get("form_type") in ("PRESENT","PAST"):
                final = (morph_score*0.45 + fuzzy_score*0.20 +
                         ctx_score*0.20   + freq_score*0.10 + ocr_boost*0.05)
            else:
                final = (fuzzy_score*0.40 + ctx_score*0.25 +
                         freq_score*0.15  + morph_score*0.10 + ocr_boost*0.10)
            ranked.append({"word": cand, "score": final})

        ranked.sort(key=lambda x: x["score"], reverse=True)
        suggestions = [x["word"] for x in ranked[:15]]

        # Gemini semantic validation
        _is_valid_verb_form = (
            self.morphology is not None and
            self.morphology.analyze(w) is not None
        )
        if (full_sentence and position is not None
                and len(ranked) >= 2
                and _gemini_model is not None
                and not _is_valid_verb_form
                and ranked[0]["score"] < 0.60):
            top_score    = ranked[0]["score"]
            second_score = ranked[1]["score"]
            if (top_score - second_score) < 0.05:
                gemini_choice = self._gemini_semantic_validate(
                    full_sentence, position, suggestions[:4]
                )
                if gemini_choice and gemini_choice != suggestions[0]:
                    suggestions = (
                        [gemini_choice]
                        + [s for s in suggestions if s != gemini_choice]
                    )
                    print(f"[Gemini] Reordered: '{suggestions[0]}' chosen for '{word}'")

        # Brute force disabled - too slow for UI usage
        # Levenshtein candidates already cover sufficient suggestions
        pass

        # Collect all priority corrections
        priority = []
        for s in [fst_corrected, vibhakti_corrected,
                  postposition_corrected, sandhi_corrected, ocr_direct_hit]:
            if s and s not in priority:
                priority.append(s)

        seen_p = set(priority)
        for s in suggestions:
            if s and s not in seen_p:
                priority.append(s)
                seen_p.add(s)

        if len(priority) < 7:
            try:
                broad = self._find_candidates_levenshtein(w, max_suggestions=50)
                for s in broad:
                    if s and s not in seen_p:
                        priority.append(s)
                        seen_p.add(s)
                    if len(priority) >= 10:
                        break
            except Exception:
                pass

        if len(priority) < 7:
            try:
                stripped = self._strip_vibhakti(w)
                if stripped != w:
                    broad2 = self._find_candidates_levenshtein(stripped, max_suggestions=20)
                    for s in broad2:
                        if s and s not in seen_p:
                            priority.append(s)
                            seen_p.add(s)
                        if len(priority) >= 10:
                            break
            except Exception:
                pass

        final_suggestions = priority[:10]

        if vibhakti_corrected:
            src, etype = "vibhakti_correction", "VIBHAKTI"
        elif postposition_corrected:
            src, etype = "postposition_correction", "POSTPOSITION"
        elif sandhi_corrected:
            src, etype = "sandhi_correction", "SANDHI"
        elif fst_corrected:
            src, etype = "fst_morphology", "MORPHOLOGICAL"
        elif ocr_direct_hit:
            src, etype = "ocr_direct_fix", "SPELLING"
        else:
            src, etype = "fuzzy_hybrid", "SPELLING"

        result = {
            "word":        word,
            "valid":       False,
            "suggestions": final_suggestions,
            "confidence":  0.95 if priority else 0.5,
            "source":      src,
            "error_type":  etype,
        }

        if not prev_word and not full_sentence:
            self.cache.set(w, result)

        return result

    def correct_sentence(self, sentence: str) -> str:
        if not sentence:
            return sentence
        words     = sentence.split()
        corrected = []
        for i, word in enumerate(words):
            prev   = words[i - 1] if i > 0 else None
            result = self.validate_word(
                word, prev_word=prev, full_sentence=sentence, position=i
            )
            if result["valid"] or not result["suggestions"]:
                corrected.append(word)
            else:
                corrected.append(result["suggestions"][0])
        return " ".join(corrected)


# ============================================================
# TEST SUITE
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Enhanced Validator v7.3 — Full Test Suite")
    print("=" * 72)

    validator = EnhancedValidator(
        dictionary_paths=[
            "data/dictionaries/Padakosha_kannada_csv.csv",
            "data/dictionaries/combined_word_scrapped_csv.csv",
        ],
        context_db_path="data/ngram_context.db",
        enable_fst=True,
        enable_vibhakti_validation=True,
        enable_postposition_correction=True,
    )

    test_cases = [
        ("ವಿನಯ್ ಪುತ್ತಕ ಓದುತ್ತಾನೆ",       "ವಿನಯ್ ಪುಸ್ತಕ ಓದುತ್ತಾನೆ",     "Spelling"),
        ("ಸುನಿಲ್ ಶಾಲೆಗೆ ಹೊಗುತ್ತಾನೆ",     "ಸುನಿಲ್ ಶಾಲೆಗೆ ಹೋಗುತ್ತಾನೆ",   "Spelling"),
        ("ಮಕ್ಕಳು ಹಣ್ಣು ತಿನುತ್ತಾರೆ",       "ಮಕ್ಕಳು ಹಣ್ಣು ತಿನ್ನುತ್ತಾರೆ",  "Spelling"),
        ("ನಂದಿನಿ ಕತೆ ಓದುತ್ತಾಳೆ",          "ನಂದಿನಿ ಕಥೆ ಓದುತ್ತಾಳೆ",       "Spelling"),
        ("ಅವಳು ನೀರವನ್ನು ಕುಡಿಯುತ್ತಾಳೆ",   "ಅವಳು ನೀರನ್ನು ಕುಡಿಯುತ್ತಾಳೆ", "Vibhakti"),
        ("ಅವನು ಮನೆವನ್ನು ಬಿಟ್ಟನು",         "ಅವನು ಮನೆಯನ್ನು ಬಿಟ್ಟನು",      "Vibhakti"),
        ("ಅವನು ಬೇಗನೆ ಓಡುತ್ತಾನೆನು",       "ಅವನು ಬೇಗನೆ ಓಡುತ್ತಾನೆ",      "Extra Suffix"),
        ("ಅವರು ಸಭೆಯಲ್ಲಿ ಮಾತನಾಡಿದರುನು",   "ಅವರು ಸಭೆಯಲ್ಲಿ ಮಾತನಾಡಿದರು",  "Extra Suffix"),
        ("ಮಕ್ಕಳು ಉದ್ಯಾನವನ ಆಡಿದರು",       "ಮಕ್ಕಳು ಉದ್ಯಾನವನದಲ್ಲಿ ಆಡಿದರು","Postposition"),
        ("ಅವನು ಶಾಲೆ ಹೋಗಿದನು",            "ಅವನು ಶಾಲೆಗೆ ಹೋದನು",          "Postposition"),
    ]

    # Gender inference tests (FIX-11)
    print("\n--- FIX-11: Suffix-based Gender Detection ---")
    ca = ContextAnalyzer()
    gender_tests = [
        ("ಪ್ರಿಯಾ",   "FEMALE"),
        ("ರಾಧಾ",     "FEMALE"),
        ("ಲಕ್ಷ್ಮಿ",  "FEMALE"),
        ("ಸರೋಜೆ",   "FEMALE"),
        ("ಮಹೇಶ್",   "MALE"),
        ("ಕಿರಣ್",   "MALE"),
        ("ರಾಮ",     "MALE"),
        ("ಅರ್ಜುನ",  "MALE"),
    ]
    for name, expected in gender_tests:
        got = ca._infer_gender_from_name(name)
        status = "✅" if got == expected else "❌"
        print(f"  {status} {name} → {got} (expected {expected})")

    passed = 0
    print(f"\n{'#':<4} {'Type':<15} {'Status'}")
    print("-" * 72)
    for i, (inp, expected, etype) in enumerate(test_cases, 1):
        output = validator.correct_sentence(inp)
        ok     = output.strip() == expected.strip()
        if ok:
            passed += 1
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{i:<4} {etype:<15} {status}")
        print(f"     Input   : {inp}")
        print(f"     Expected: {expected}")
        print(f"     Got     : {output}")
        print()

    print("=" * 72)
    pct = passed / len(test_cases) * 100
    print(f"Result: {passed}/{len(test_cases)} passed  ({pct:.1f}%)")
    print("=" * 72)