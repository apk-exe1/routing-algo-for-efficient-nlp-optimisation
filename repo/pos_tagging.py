# =============================================================
# pos_tagging.py  (v3 — fixed, no Stanza, works fully offline)
# Kannada POS Tagger using IndicNLP + morphological heuristics
# =============================================================

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# --- Try importing IndicNLP tokenizer ---
try:
    from indicnlp.tokenize import indic_tokenize
    INDICNLP_AVAILABLE = True
except ImportError:
    INDICNLP_AVAILABLE = False
    logger.warning(
        "indicnlp not installed. Using basic whitespace tokenizer.\n"
        "For better results: pip install indic-nlp-library"
    )

# --- Import ALL lexicon data in one block ---
from honorific_lexicon import (
    TIER_1_ALWAYS_HONORIFIC,
    TIER_2_ROLE_HONORIFIC,
    DIVINE_SIGNALS,
    HUMAN_SIGNALS,
    VERB_SUFFIX_MAP,
    STANDALONE_POSTPOSITIONS,
)


# =============================================================
# KANNADA VERB SUFFIXES
# Sorted longest first to match greedily
# =============================================================

VERB_SUFFIXES = sorted([
    # Present tense
    "ತ್ತಾರೆ", "ತ್ತಾನೆ", "ತ್ತಾಳೆ", "ತ್ತೇವೆ", "ತ್ತೀರಿ", "ತ್ತೇನೆ",
    "ತ್ತಿದ್ದಾರೆ", "ತ್ತಿದ್ದಾನೆ", "ತ್ತಿದ್ದಾಳೆ", "ತ್ತಿದ್ದೆ",
    "ತ್ತಿದ್ದೇನೆ",
    # Past tense
    "ದರು", "ದನು", "ದಳು", "ದೆ", "ದೆವು", "ದಿರಿ",
    "ನರು", "ನನು", "ನಳು",
    "ಇದ್ದರು", "ಇದ್ದನು", "ಇದ್ದಳು",
    "ಇತು", "ಇತ್ತು",
    # Future tense
    "ಆರು", "ಆನು", "ಆಳು", "ಆರೆ", "ಆನೆ", "ಆಳೆ",
    # Imperative / other
    "ಉ", "ಇ", "ಆ",
], key=len, reverse=True)


# =============================================================
# KANNADA POSTPOSITION SUFFIXES (suffix-based)
# =============================================================

ADP_SUFFIXES = [
    "ಗೆ", "ಕ್ಕೆ", "ನ್ನು", "ಅನ್ನು", "ಇಂದ", "ದಿಂದ",
    "ಲ್ಲಿ", "ದಲ್ಲಿ", "ಯಲ್ಲಿ", "ವರೆಗೆ", "ಕಾಗಿ",
]


# =============================================================
# TOKENIZER
# =============================================================

def tokenize(sentence: str) -> list:
    """Tokenize a Kannada sentence into words."""
    sentence = sentence.strip()
    if INDICNLP_AVAILABLE:
        try:
            tokens = indic_tokenize.trivial_tokenize(sentence, lang="kn")
            return [t.strip(".,\u0964?!\"'") for t in tokens if t.strip(".,\u0964?!\"'")]
        except Exception:
            pass
    return [w.strip(".,\u0964?!\"'") for w in sentence.split() if w.strip(".,\u0964?!\"'")]


# =============================================================
# WORD-LEVEL POS CLASSIFIER
# =============================================================

def classify_word(word, position, total_words, previous_word=None):
    """
    Assign a POS tag to a single Kannada word.

    Tags returned:
        PROPN  = proper noun (unknown word, likely a name)
        NOUN   = common noun (found in lexicon)
        VERB   = verb
        ADP    = postposition / case marker
        X      = unknown / short particle
    """
    if not word:
        return "X"

    # 1. Known lexicon entries → NOUN
    if word in TIER_1_ALWAYS_HONORIFIC:
        return "NOUN"
    if word in TIER_2_ROLE_HONORIFIC:
        return "NOUN"
    if word in DIVINE_SIGNALS or word in HUMAN_SIGNALS:
        return "NOUN"

    # 2. Standalone postpositions (whole-word match)
    if word in STANDALONE_POSTPOSITIONS:
        return "ADP"

    # 3. Verb detection by suffix (longest match first)
    for suffix in VERB_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return "VERB"

    # 4. Postposition detection by suffix
    for suffix in ADP_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return "ADP"

    # 5. SOV heuristic: last word in sentence is likely verb
    if position == total_words - 1 and total_words > 1:
        return "VERB"

    # 6. Unknown word of reasonable length = likely proper noun
    # Minimum 3 chars AND not a pure punctuation/number token
    if len(word) >= 3:
        return "PROPN"

    # 7. Two-char words — check if they look like Kannada words
    # (contain Kannada unicode range characters)
    if len(word) == 2 and any("ಀ" <= c <= "೿" for c in word):
        return "NOUN"

    return "X"


# =============================================================
# SUBJECT DETECTOR
# =============================================================

def detect_subject(tokens, pos_tags):
    """
    Find the grammatical subject.
    Returns (word, pos_tag, index) or None.

    Rule: first PROPN or NOUN not immediately followed by ADP.
    Words followed by postpositions are objects, not subjects.
    """
    for i, (word, tag) in enumerate(zip(tokens, pos_tags)):
        if tag in ("PROPN", "NOUN"):
            next_tag = pos_tags[i + 1] if i + 1 < len(pos_tags) else None
            if next_tag != "ADP":
                return (word, tag, i)

    # Fallback: return first PROPN or NOUN regardless
    for i, (word, tag) in enumerate(zip(tokens, pos_tags)):
        if tag in ("PROPN", "NOUN"):
            return (word, tag, i)

    return None


# =============================================================
# VERB DETECTOR
# =============================================================

def detect_verb(tokens, pos_tags):
    """
    Find the main verb — last VERB token (SOV assumption).
    Returns (word, index) or None.
    """
    verbs = [
        (word, i)
        for i, (word, tag) in enumerate(zip(tokens, pos_tags))
        if tag == "VERB"
    ]
    return verbs[-1] if verbs else None


# =============================================================
# CORE FUNCTION — analyze_sentence()
# Main entry point used by honorific_agreement.py
# =============================================================

def analyze_sentence(sentence: str) -> dict:
    """
    Full morphological analysis of a Kannada sentence.

    Returns:
    {
        "tokens":       [("ಕೃಷ್ಣ", "PROPN"), ("ಗೋವುಗಳ", "NOUN"), ...],
        "subject":      ("ಕೃಷ್ಣ", "PROPN"),
        "verb":         ("ಆಡುತ್ತಾನೆ", 4),
        "proper_nouns": ["ಕೃಷ್ಣ"],
        "success":      True
    }
    """
    try:
        words = tokenize(sentence)
        if not words:
            return _empty_result()

        n = len(words)
        pos_tags = []
        for i, word in enumerate(words):
            prev = words[i - 1] if i > 0 else None
            tag = classify_word(word, i, n, prev)
            pos_tags.append(tag)

        tokens       = list(zip(words, pos_tags))
        subject_info = detect_subject(words, pos_tags)
        verb_info    = detect_verb(words, pos_tags)
        proper_nouns = [w for w, t in tokens if t == "PROPN"]

        subject = (subject_info[0], subject_info[1]) if subject_info else None
        verb    = (verb_info[0],    verb_info[1])     if verb_info    else None

        return {
            "tokens":       tokens,
            "subject":      subject,
            "verb":         verb,
            "proper_nouns": proper_nouns,
            "success":      True
        }

    except Exception as e:
        logger.error(f"analyze_sentence error: {e}")
        return _empty_result()


def _empty_result():
    return {
        "tokens": [], "subject": None,
        "verb": None, "proper_nouns": [], "success": False
    }


# =============================================================
# CONVENIENCE WRAPPERS
# =============================================================

def get_pos_tags(sentence):
    return analyze_sentence(sentence)["tokens"]

def get_subject(sentence):
    return analyze_sentence(sentence)["subject"]

def get_main_verb(sentence):
    return analyze_sentence(sentence)["verb"]


# =============================================================
# QUICK TEST
# =============================================================

if __name__ == "__main__":
    print("pos_tagging.py — morphological tagger for Kannada")
    print(f"IndicNLP available : {INDICNLP_AVAILABLE}")
    print()

    test_sentences = [
        ("ತಾತ ಪುಸ್ತಕ ಓದುತ್ತಿದ್ದಾನೆ",
         "subject=ತಾತ(NOUN),  verb=ಓದುತ್ತಿದ್ದಾನೆ"),
        ("ಕೃಷ್ಣ ಗೋವುಗಳ ಜೊತೆ ಆಟ ಆಡುತ್ತಾನೆ",
         "subject=ಕೃಷ್ಣ(PROPN), verb=ಆಡುತ್ತಾನೆ  [ಜೊತೆ should now be ADP]"),
        ("ಮಗು ಶಾಲೆಗೆ ಹೋಯಿತು",
         "subject=ಮಗು, verb=ಹೋಯಿತು"),
        ("ಅಜ್ಜಿ ಸಂಗೀತ ಹಾಡಿದರು",
         "subject=ಅಜ್ಜಿ(NOUN), verb=ಹಾಡಿದರು"),
    ]

    for sentence, expected in test_sentences:
        r = analyze_sentence(sentence)
        print(f"Sentence : {sentence}")
        print(f"Expected : {expected}")
        print(f"Tokens   : {r['tokens']}")
        print(f"Subject  : {r['subject']}")
        print(f"Verb     : {r['verb']}")
        print(f"PROPNs   : {r['proper_nouns']}")
        print("-" * 55)