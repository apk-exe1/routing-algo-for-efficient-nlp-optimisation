# =============================================================
# honorific_agreement.py
# Core honorific agreement correction logic
# Implements all 4 layers of the disambiguation pipeline
#
# Layer 1 — Document genre classifier (runs once per document)
# Layer 2 — Sliding window signal scorer (runs per sentence)
# Layer 3 — MuRIL contextual classifier (runs on hard cases)
# Layer 4 — User escalation flag (for truly ambiguous cases)
#
# Imports: honorific_lexicon.py, pos_tagging.py
# =============================================================

import re
import logging
from typing import Optional
from honorific_lexicon import (
    TIER_1_ALWAYS_HONORIFIC,
    TIER_2_ROLE_HONORIFIC,
    DIVINE_SIGNALS,
    HUMAN_SIGNALS,
    HUMAN_EXCLUSIVE_PREDICATES,
    RELIGIOUS_GENRE_MARKERS,
    SECULAR_GENRE_MARKERS,
    VERB_SUFFIX_MAP,
    CONFIDENCE,
)
from pos_tagging import analyze_sentence

logger = logging.getLogger(__name__)


# =============================================================
# RESULT DATACLASS
# Standardised return object from process_sentence()
# =============================================================

class HonorificResult:
    """
    Holds the result of processing one sentence.

    Fields:
        original_sentence  : the input sentence unchanged
        corrected_sentence : sentence with verb form corrected
                             (same as original if no correction needed)
        subject_word       : the subject word that was analysed
        subject_tier       : 1 / 2 / 3 / 4
        classification     : "divine" / "human" / "ambiguous"
        correction_made    : True if verb was actually changed
        needs_user_input   : True if system could not decide (Layer 4)
        confidence         : float 0.0 – 1.0
        layer_used         : 1 / 2 / 3 / 4 (which layer decided)
        reason             : human-readable explanation
    """
    def __init__(self):
        self.original_sentence  = ""
        self.corrected_sentence = ""
        self.subject_word       = None
        self.subject_tier       = None
        self.classification     = "ambiguous"
        self.correction_made    = False
        self.needs_user_input   = False
        self.confidence         = 0.0
        self.layer_used         = None
        self.reason             = ""

    def __repr__(self):
        return (
            f"HonorificResult("
            f"subject='{self.subject_word}', "
            f"tier={self.subject_tier}, "
            f"class='{self.classification}', "
            f"corrected={self.correction_made}, "
            f"layer={self.layer_used}, "
            f"confidence={self.confidence:.2f})"
        )


# =============================================================
# LAYER 1 — Document Genre Classifier
# Call once per document before processing any sentences
# =============================================================

def classify_document_genre(document_text: str) -> dict:
    """
    Reads the full document text and returns a genre classification.

    Returns:
    {
        "genre":          "religious" / "secular" / "mixed" / "unknown",
        "religious_score": int,   # count of religious marker hits
        "secular_score":   int,   # count of secular marker hits
        "confidence":      float, # how strongly one side dominates
        "divine_prior":    float  # prior probability for Tier 3 names
                                  # used by Layer 2 scoring
    }

    How divine_prior works:
        0.8  = strong religious document → assume divine unless strong
               human evidence says otherwise
        0.5  = neutral / mixed → no prior, signals decide
        0.2  = strong secular document → assume human unless strong
               divine evidence says otherwise
    """
    words = document_text.split()
    word_set = set(words)

    religious_score = len(word_set & RELIGIOUS_GENRE_MARKERS)
    secular_score   = len(word_set & SECULAR_GENRE_MARKERS)
    total           = religious_score + secular_score

    if total == 0:
        return {
            "genre": "unknown",
            "religious_score": 0,
            "secular_score": 0,
            "confidence": 0.0,
            "divine_prior": 0.5
        }

    religious_ratio = religious_score / total

    if religious_ratio >= 0.70:
        genre = "religious"
        divine_prior = 0.8
        confidence = religious_ratio
    elif religious_ratio <= 0.30:
        genre = "secular"
        divine_prior = 0.2
        confidence = 1.0 - religious_ratio
    else:
        genre = "mixed"
        divine_prior = 0.5
        confidence = abs(religious_ratio - 0.5) * 2   # 0 at 50/50, 1 at extreme

    return {
        "genre": genre,
        "religious_score": religious_score,
        "secular_score": secular_score,
        "confidence": round(confidence, 3),
        "divine_prior": divine_prior
    }


# =============================================================
# LAYER 2 — Sliding Window Signal Scorer
# Called per sentence for Tier 3 (ambiguous proper noun) subjects
# =============================================================

def score_context_signals(
    sentence: str,
    adjacent_sentences: list[str] = None,
    divine_prior: float = 0.5
) -> dict:
    """
    Counts divine vs human context signals in a window around
    the sentence. Uses Layer 1's divine_prior to bias the score.

    Args:
        sentence           : the current sentence being analysed
        adjacent_sentences : list of 1-2 neighbouring sentences
                             (pass [] if processing in isolation)
        divine_prior       : from classify_document_genre()
                             0.8=religious doc, 0.5=neutral, 0.2=secular

    Returns:
    {
        "divine_score":    float,   # weighted signal count
        "human_score":     float,
        "classification":  "divine" / "human" / "ambiguous",
        "confidence":      float,
        "override":        bool,    # True if human-exclusive predicate found
        "matched_divine":  list,    # which divine words were found
        "matched_human":   list
    }
    """
    # --- Build the full window text ---
    window_parts = [sentence]
    if adjacent_sentences:
        window_parts.extend(adjacent_sentences[:2])   # max 2 neighbours
    window_text = " ".join(window_parts)
    window_words = set(window_text.split())

    # --- Check human-exclusive predicates first (hard override) ---
    for predicate in HUMAN_EXCLUSIVE_PREDICATES:
        if predicate in window_text:
            return {
                "divine_score":   0.0,
                "human_score":    1.0,
                "classification": "human",
                "confidence":     1.0,
                "override":       True,
                "matched_divine": [],
                "matched_human":  [predicate],
                "reason": f"Human-exclusive predicate found: '{predicate}'"
            }

    # --- Count signal hits ---
    matched_divine = list(window_words & DIVINE_SIGNALS)
    matched_human  = list(window_words & HUMAN_SIGNALS)

    raw_divine = len(matched_divine)
    raw_human  = len(matched_human)

    # --- Apply prior as a bias weight ---
    # Prior shifts the effective score before comparing
    # e.g. divine_prior=0.8 adds 0.8 to divine side automatically
    biased_divine = raw_divine + divine_prior
    biased_human  = raw_human  + (1.0 - divine_prior)

    total = biased_divine + biased_human
    if total == 0:
        return {
            "divine_score":   0.0,
            "human_score":    0.0,
            "classification": "ambiguous",
            "confidence":     0.0,
            "override":       False,
            "matched_divine": [],
            "matched_human":  [],
            "reason": "No context signals found in window"
        }

    divine_ratio = biased_divine / total

    # --- Classify based on ratio ---
    if divine_ratio >= CONFIDENCE["ACT"]:
        classification = "divine"
        confidence = divine_ratio
    elif divine_ratio <= (1.0 - CONFIDENCE["ACT"]):
        classification = "human"
        confidence = 1.0 - divine_ratio
    else:
        classification = "ambiguous"
        confidence = abs(divine_ratio - 0.5) * 2

    reason_parts = []
    if matched_divine:
        reason_parts.append(f"Divine signals: {matched_divine}")
    if matched_human:
        reason_parts.append(f"Human signals: {matched_human}")
    reason_parts.append(f"Prior: {divine_prior} (doc genre bias)")

    return {
        "divine_score":   round(biased_divine, 3),
        "human_score":    round(biased_human,  3),
        "classification": classification,
        "confidence":     round(confidence, 3),
        "override":       False,
        "matched_divine": matched_divine,
        "matched_human":  matched_human,
        "reason":         " | ".join(reason_parts)
    }


# =============================================================
# SUBJECT TIER CLASSIFIER
# Given a subject word and its POS tag, determine which tier
# =============================================================

def classify_subject_tier(subject_word: str, pos_tag: str) -> int:
    """
    Classify the subject into one of four tiers.

    Tier 1 → kinship/always-honorific: correct to honorific automatically
    Tier 2 → role/title: correct to honorific (default, override if needed)
    Tier 3 → ambiguous proper noun: needs context analysis (Layers 2–4)
    Tier 4 → non-honorific: no correction needed

    Returns: int (1, 2, 3, or 4)
    """
    clean = subject_word.strip(".,।")

    if clean in TIER_1_ALWAYS_HONORIFIC:
        return 1

    if clean in TIER_2_ROLE_HONORIFIC:
        return 2

    # Tier 3: proper noun not in known lists → ambiguous
    if pos_tag == "PROPN":
        return 3

    # Everything else: common noun, pronoun, etc. → non-honorific
    return 4


# =============================================================
# VERB CORRECTOR
# Applies VERB_SUFFIX_MAP to change a verb to its honorific form
# =============================================================

def apply_honorific_verb_form(verb_word: str) -> tuple[str, bool]:
    """
    Given a verb word, return its honorific form.
    Returns (corrected_verb, was_changed).

    Checks each suffix in VERB_SUFFIX_MAP — longest match first
    to avoid partial replacements.

    Example:
        "ಓದುತ್ತಿದ್ದಾನೆ" → ("ಓದುತ್ತಿದ್ದಾರೆ", True)
        "ಓದುತ್ತಿದ್ದಾರೆ" → ("ಓದುತ್ತಿದ್ದಾರೆ", False)  ← already honorific
    """
    # Sort by length descending — match longest suffix first
    sorted_suffixes = sorted(VERB_SUFFIX_MAP.keys(), key=len, reverse=True)

    for suffix in sorted_suffixes:
        if verb_word.endswith(suffix):
            honorific_suffix = VERB_SUFFIX_MAP[suffix]
            # Check if already in honorific form
            if verb_word.endswith(honorific_suffix):
                return (verb_word, False)
            corrected = verb_word[: -len(suffix)] + honorific_suffix
            return (corrected, True)

    return (verb_word, False)


def replace_verb_in_sentence(
    sentence: str,
    original_verb: str,
    corrected_verb: str
) -> str:
    """Replace the original verb with corrected verb in the sentence."""
    # Use word boundary replacement to avoid partial matches
    return sentence.replace(original_verb, corrected_verb, 1)


# =============================================================
# MAIN ENTRY POINT — process_sentence()
# This is what correction_pipeline.py calls
# =============================================================

def process_sentence(
    sentence: str,
    document_genre: dict = None,
    adjacent_sentences: list[str] = None,
    muril_classifier=None,
    user_declaration: str = "unknown"
) -> HonorificResult:
    """
    Full 4-layer honorific agreement processing for one sentence.

    Args:
        sentence           : Kannada sentence to process
        document_genre     : output of classify_document_genre()
                             Pass None if not available
        adjacent_sentences : neighbouring sentences for context window
        muril_classifier   : loaded MuRILClassifier instance (Layer 3)
                             Pass None to skip Layer 3
        user_declaration   : "religious" / "secular" / "unknown"
                             From UI pre-declaration (Level 1 user input)

    Returns: HonorificResult object
    """
    result = HonorificResult()
    result.original_sentence  = sentence
    result.corrected_sentence = sentence

    # --- Step 1: POS analysis ---
    analysis = analyze_sentence(sentence)
    if not analysis["success"] or analysis["subject"] is None:
        result.reason = "POS analysis failed or no subject found"
        return result

    subject_word, subject_pos = analysis["subject"]
    verb_info = analysis["verb"]

    result.subject_word = subject_word

    # --- Step 2: Classify subject tier ---
    tier = classify_subject_tier(subject_word, subject_pos)
    result.subject_tier = tier

    # ---- TIER 1: Always honorific ----
    if tier == 1:
        result.classification = "divine"   # treat as honorific entity
        result.layer_used     = 1
        result.confidence     = 1.0
        result.reason         = f"'{subject_word}' is a Tier 1 kinship/respected term"

        if verb_info:
            verb_word, verb_idx = verb_info
            corrected_verb, changed = apply_honorific_verb_form(verb_word)
            if changed:
                result.corrected_sentence = replace_verb_in_sentence(
                    sentence, verb_word, corrected_verb
                )
                result.correction_made = True
        return result

    # ---- TIER 2: Role/title — honorific by default ----
    if tier == 2:
        result.classification = "divine"
        result.layer_used     = 1
        result.confidence     = 0.9
        result.reason         = f"'{subject_word}' is a Tier 2 role/title term"

        if verb_info:
            verb_word, verb_idx = verb_info
            corrected_verb, changed = apply_honorific_verb_form(verb_word)
            if changed:
                result.corrected_sentence = replace_verb_in_sentence(
                    sentence, verb_word, corrected_verb
                )
                result.correction_made = True
        return result

    # ---- TIER 4: Non-honorific — no correction ----
    if tier == 4:
        result.classification = "human"
        result.layer_used     = 1
        result.confidence     = 0.9
        result.reason         = f"'{subject_word}' is a Tier 4 non-honorific term"
        return result

    # ---- TIER 3: Ambiguous proper noun — run Layers 2, 3, 4 ----

    # Determine divine_prior from user declaration or document genre
    divine_prior = 0.5   # default neutral
    if user_declaration == "religious":
        divine_prior = 0.85
    elif user_declaration == "secular":
        divine_prior = 0.15
    elif document_genre is not None:
        divine_prior = document_genre.get("divine_prior", 0.5)

    # --- LAYER 2: Signal scoring ---
    layer2 = score_context_signals(
        sentence,
        adjacent_sentences or [],
        divine_prior
    )

    if layer2["override"]:
        # Human-exclusive predicate found — act with full confidence
        result.classification = "human"
        result.layer_used     = 2
        result.confidence     = 1.0
        result.reason         = layer2["reason"]
        return result

    if layer2["confidence"] >= CONFIDENCE["ACT"]:
        # Layer 2 is confident enough to act
        result.classification = layer2["classification"]
        result.layer_used     = 2
        result.confidence     = layer2["confidence"]
        result.reason         = layer2["reason"]

        if result.classification == "divine" and verb_info:
            verb_word, _ = verb_info
            corrected_verb, changed = apply_honorific_verb_form(verb_word)
            if changed:
                result.corrected_sentence = replace_verb_in_sentence(
                    sentence, verb_word, corrected_verb
                )
                result.correction_made = True
        return result

    # --- LAYER 3: MuRIL classifier (if available) ---
    if muril_classifier is not None:
        try:
            layer3_result = muril_classifier.predict(sentence, subject_word)
            if layer3_result["confidence"] >= CONFIDENCE["ACT"]:
                result.classification = layer3_result["classification"]
                result.layer_used     = 3
                result.confidence     = layer3_result["confidence"]
                result.reason         = f"MuRIL classifier: {layer3_result['reason']}"

                if result.classification == "divine" and verb_info:
                    verb_word, _ = verb_info
                    corrected_verb, changed = apply_honorific_verb_form(verb_word)
                    if changed:
                        result.corrected_sentence = replace_verb_in_sentence(
                            sentence, verb_word, corrected_verb
                        )
                        result.correction_made = True
                return result
        except Exception as e:
            logger.warning(f"MuRIL classifier failed: {e}")

    # --- LAYER 4: Escalate to user ---
    result.classification  = "ambiguous"
    result.layer_used      = 4
    result.confidence      = layer2["confidence"]
    result.needs_user_input = True
    result.reason = (
        f"Could not determine if '{subject_word}' is divine or human. "
        f"Layer 2 score: divine={layer2['divine_score']}, "
        f"human={layer2['human_score']}. User input required."
    )
    return result


# =============================================================
# DOCUMENT-LEVEL PROCESSOR
# Processes all sentences in a document in sequence
# =============================================================

def process_document(
    sentences: list[str],
    muril_classifier=None,
    user_declaration: str = "unknown"
) -> dict:
    """
    Process all sentences in a document.
    Runs Layer 1 once, then processes each sentence.

    Args:
        sentences        : list of Kannada sentences
        muril_classifier : loaded MuRILClassifier or None
        user_declaration : from UI ("religious"/"secular"/"unknown")

    Returns:
    {
        "document_genre" : dict from classify_document_genre(),
        "results"        : list of HonorificResult objects,
        "flagged"        : list of (index, HonorificResult) needing user input,
        "stats"          : processing statistics dict
    }
    """
    # Layer 1 — classify full document once
    full_text = " ".join(sentences)
    document_genre = classify_document_genre(full_text)

    # Override genre with explicit user declaration if provided
    if user_declaration == "religious":
        document_genre["divine_prior"] = 0.85
    elif user_declaration == "secular":
        document_genre["divine_prior"] = 0.15

    results  = []
    flagged  = []
    stats    = {
        "total": len(sentences),
        "corrected": 0,
        "flagged_for_user": 0,
        "layer_1_handled": 0,
        "layer_2_handled": 0,
        "layer_3_handled": 0,
        "layer_4_escalated": 0,
        "no_subject_found": 0,
    }

    for i, sentence in enumerate(sentences):
        # Pass adjacent sentences as context window
        adjacent = []
        if i > 0:
            adjacent.append(sentences[i - 1])
        if i < len(sentences) - 1:
            adjacent.append(sentences[i + 1])

        result = process_sentence(
            sentence,
            document_genre=document_genre,
            adjacent_sentences=adjacent,
            muril_classifier=muril_classifier,
            user_declaration=user_declaration
        )

        results.append(result)

        # Update stats
        if result.correction_made:
            stats["corrected"] += 1
        if result.needs_user_input:
            stats["flagged_for_user"] += 1
            flagged.append((i, result))
        if result.layer_used == 1:
            stats["layer_1_handled"] += 1
        elif result.layer_used == 2:
            stats["layer_2_handled"] += 1
        elif result.layer_used == 3:
            stats["layer_3_handled"] += 1
        elif result.layer_used == 4:
            stats["layer_4_escalated"] += 1
        elif result.layer_used is None:
            stats["no_subject_found"] += 1

    return {
        "document_genre": document_genre,
        "results": results,
        "flagged": flagged,
        "stats": stats
    }


# =============================================================
# APPLY USER DECISION
# Called when user resolves a Layer 4 flagged sentence
# =============================================================

def apply_user_decision(
    result: HonorificResult,
    user_choice: str   # "divine" or "human"
) -> HonorificResult:
    """
    Takes a flagged HonorificResult and applies the user's decision.
    Called by user_declaration_handler.py after user clicks
    "Deity" or "Person" in the UI.

    Returns updated HonorificResult with correction applied.
    """
    result.classification  = user_choice
    result.needs_user_input = False
    result.layer_used      = 4
    result.confidence      = 1.0
    result.reason          = f"User manually classified as '{user_choice}'"

    if user_choice == "divine":
        # Re-parse to get verb
        analysis = analyze_sentence(result.original_sentence)
        if analysis["verb"]:
            verb_word, _ = analysis["verb"]
            corrected_verb, changed = apply_honorific_verb_form(verb_word)
            if changed:
                result.corrected_sentence = replace_verb_in_sentence(
                    result.original_sentence, verb_word, corrected_verb
                )
                result.correction_made = True
    else:
        result.corrected_sentence = result.original_sentence
        result.correction_made    = False

    return result


# =============================================================
# QUICK TEST
# =============================================================

if __name__ == "__main__":
    test_cases = [
        ("ತಾತ ಪುಸ್ತಕ ಓದುತ್ತಿದ್ದಾನೆ",           "neutral",  "Tier 1 — should correct"),
        ("ಕೃಷ್ಣ ಗೋವುಗಳ ಜೊತೆ ಆಟ ಆಡುತ್ತಾನೆ",   "religious","Tier 3 divine — should correct"),
        ("ಕೃಷ್ಣ ಶಾಲೆಗೆ ಹೋಗುತ್ತಾನೆ",            "secular",  "Tier 3 human — should NOT correct"),
        ("ಡಾಕ್ಟರ್ ಔಷಧ ಕೊಟ್ಟರು",                "neutral",  "Tier 2 — already honorific"),
    ]

    print("Testing honorific_agreement.py\n")
    for sentence, declaration, description in test_cases:
        print(f"Test     : {description}")
        print(f"Input    : {sentence}")
        r = process_sentence(sentence, user_declaration=declaration)
        print(f"Output   : {r.corrected_sentence}")
        print(f"Result   : {r}")
        print("-" * 60)