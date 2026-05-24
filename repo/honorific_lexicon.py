# =============================================================
# honorific_lexicon.py
# Pure data file — no logic, no imports
# All vocabulary sets and mappings for honorific agreement
# =============================================================


# -------------------------------------------------------------
# TIER 1 — Kinship and inherently respected terms
# These ALWAYS require honorific verb forms, no context needed
# -------------------------------------------------------------
TIER_1_ALWAYS_HONORIFIC = {
    # Grandparents
    "ತಾತ", "ಅಜ್ಜ", "ಅಜ್ಜಿ", "ಅಜ್ಜಮ್ಮ",
    # Parents
    "ಅಪ್ಪ", "ಅಮ್ಮ", "ತಂದೆ", "ತಾಯಿ", "ಅಪ್ಪಾಜಿ", "ಅಮ್ಮಾಜಿ",
    "ತಂದೆಯವರು", "ತಾಯಿಯವರು",
    # In-laws and elders
    "ಮಾವ", "ಅತ್ತೆ", "ಚಿಕ್ಕಪ್ಪ", "ದೊಡ್ಡಪ್ಪ", "ಚಿಕ್ಕಮ್ಮ", "ದೊಡ್ಡಮ್ಮ",
    "ಮಾವನವರು", "ಅತ್ತೆಯವರು",
    # Spiritual / guru
    "ಗುರು", "ಗುರುಗಳು", "ಗುರುಜಿ", "ಸ್ವಾಮೀಜಿ", "ಸ್ವಾಮಿ",
    "ಶ್ರೀಗಳು", "ಮಠಾಧೀಶ", "ಆಚಾರ್ಯ", "ಆಚಾರ್ಯರು",
    # General elder address
    "ಹಿರಿಯರು", "ಹಿರಿಯ", "ಮಾನ್ಯರು",
}


# -------------------------------------------------------------
# TIER 2 — Role and title terms
# These are honorific by default unless context says otherwise
# -------------------------------------------------------------
TIER_2_ROLE_HONORIFIC = {
    # Education
    "ಮೇಷ್ಟ್ರು", "ಮೇಷ್ಟ್ರ", "ಅಧ್ಯಾಪಕರು", "ಶಿಕ್ಷಕರು",
    "ಪ್ರಾಧ್ಯಾಪಕರು", "ಉಪಾಧ್ಯಾಯರು", "ಮುಖ್ಯೋಪಾಧ್ಯಾಯರು",
    # Medical
    "ಡಾಕ್ಟರ್", "ವೈದ್ಯರು", "ವೈದ್ಯ",
    # Legal / Government
    "ನ್ಯಾಯಾಧೀಶರು", "ನ್ಯಾಯಾಧೀಶ", "ಮಂತ್ರಿ", "ಮಂತ್ರಿಗಳು",
    "ಮುಖ್ಯಮಂತ್ರಿ", "ಮುಖ್ಯಮಂತ್ರಿಗಳು", "ರಾಜ್ಯಪಾಲ", "ರಾಜ್ಯಪಾಲರು",
    "ಸಂಸದ", "ಶಾಸಕ",
    # Religious roles
    "ಪೂಜಾರಿ", "ಪಂಡಿತ", "ಪಂಡಿತರು", "ಶಾಸ್ತ್ರಿ", "ಶಾಸ್ತ್ರಿಗಳು",
    # Honorific address suffixes (when these appear as standalone subjects)
    "ಅವರು", "ಇವರು", "ರವರು",
}


# -------------------------------------------------------------
# TIER 3 — Ambiguous proper noun markers
# These are NOT names — they are contextual cues that a word
# behaves as a divine proper noun in this sentence.
# The system detects proper nouns via POS tagging, then checks
# surrounding signals (below) to classify divine vs human.
# This set is intentionally left small — classification happens
# via context signals, not name matching.
# -------------------------------------------------------------
# NOTE: We do NOT list deity names here on purpose.
# Any proper noun (PROPN from POS tagger) goes through
# context scoring using DIVINE_SIGNALS and HUMAN_SIGNALS below.


# -------------------------------------------------------------
# DIVINE CONTEXT SIGNALS
# If these words appear near a proper noun subject,
# the subject is likely being referred to as a deity
# -------------------------------------------------------------
DIVINE_SIGNALS = {
    # Worship and ritual
    "ಪೂಜಿಸು", "ಪೂಜೆ", "ಪೂಜಿಸಿ", "ಪೂಜಿಸುತ್ತಾರೆ", "ಪೂಜಿಸಿದರು",
    "ಆರಾಧಿಸು", "ಆರಾಧನೆ", "ಭಜಿಸು", "ಭಜನೆ",
    "ಅರ್ಚಿಸು", "ಅರ್ಚನೆ", "ಅಭಿಷೇಕ",
    # Devotion and blessings
    "ಭಕ್ತ", "ಭಕ್ತರು", "ಭಕ್ತಿ", "ಆಶೀರ್ವದಿಸು", "ಆಶೀರ್ವಾದ",
    "ಕೃಪೆ", "ಅನುಗ್ರಹ", "ಪ್ರಸಾದ", "ದರ್ಶನ",
    # Divine attributes
    "ಅವತಾರ", "ಲೀಲೆ", "ಮಹಿಮೆ", "ದಿವ್ಯ", "ಚರಣ",
    "ದೇವ", "ದೇವರು", "ಭಗವಂತ", "ಈಶ್ವರ", "ಪರಮಾತ್ಮ",
    # Sacred places
    "ದೇವಾಲಯ", "ಮಂದಿರ", "ಗುಡಿ", "ಕ್ಷೇತ್ರ", "ತೀರ್ಥ",
    # Rituals
    "ಯಜ್ಞ", "ಹವನ", "ಮಂತ್ರ", "ಸ್ತೋತ್ರ", "ಸ್ಲೋಕ",
    # Objects associated with deities
    "ಗೋವು", "ಗೋವುಗಳ", "ವೇಣು", "ಚಕ್ರ", "ತ್ರಿಶೂಲ", "ಪದ್ಮ",
    # Mythology terms
    "ಸ್ವರ್ಗ", "ವೈಕುಂಠ", "ಕೈಲಾಸ", "ಮೋಕ್ಷ",
}


# -------------------------------------------------------------
# HUMAN CONTEXT SIGNALS
# If these words appear near a proper noun subject,
# the subject is likely a human person
# -------------------------------------------------------------
HUMAN_SIGNALS = {
    # Education
    "ಶಾಲೆ", "ಕಾಲೇಜು", "ವಿದ್ಯಾಲಯ", "ತರಗತಿ",
    "ಪರೀಕ್ಷೆ", "ಪಾಠ", "ಹೋಮ್‌ವರ್ಕ್", "ಅಧ್ಯಯನ",
    # Work and livelihood
    "ಕೆಲಸ", "ಕಚೇರಿ", "ಆಫೀಸ್", "ಸಂಬಳ", "ವ್ಯಾಪಾರ",
    # Daily life
    "ಊಟ", "ನಿದ್ದೆ", "ಮನೆ", "ಮದ್ದು", "ಆಸ್ಪತ್ರೆ",
    "ಬಸ್ಸು", "ರೈಲು", "ಕಾರು", "ಹಣ",
    # Social
    "ಸ್ನೇಹಿತ", "ಸ್ನೇಹಿತರು", "ಗೆಳೆಯ", "ಗೆಳತಿ",
    "ಮದುವೆ", "ಮಕ್ಕಳು", "ಕುಟುಂಬ",
    # Civic / legal
    "ಚುನಾವಣೆ", "ಮತ", "ನ್ಯಾಯಾಲಯ", "ಪೊಲೀಸ್",
}


# -------------------------------------------------------------
# HUMAN-EXCLUSIVE PREDICATES
# These verb phrases are categorically impossible for deities.
# If any of these appear in the sentence window, the subject
# is classified as human with 100% confidence — overrides all.
# -------------------------------------------------------------
HUMAN_EXCLUSIVE_PREDICATES = {
    "ಶಾಲೆಗೆ ಹೋಗು", "ಶಾಲೆಗೆ ಹೋದ", "ಶಾಲೆಗೆ ಹೋದನು",
    "ಶಾಲೆಗೆ ಹೋದಳು", "ಶಾಲೆಗೆ ಹೋದರು",
    "ಪರೀಕ್ಷೆ ಬರೆ", "ಪರೀಕ್ಷೆ ಬರೆದ", "ಪರೀಕ್ಷೆ ಬರೆದನು",
    "ಕೆಲಸಕ್ಕೆ ಹೋಗು", "ಕೆಲಸಕ್ಕೆ ಹೋದ",
    "ಸಂಬಳ ತೆಗೆದ", "ಸಂಬಳ ಪಡೆದ",
    "ಕಾಲೇಜಿಗೆ ಹೋಗು", "ಕಾಲೇಜಿಗೆ ಹೋದ",
}


# -------------------------------------------------------------
# DOCUMENT GENRE — Religious vocabulary
# Used by Layer 1 to classify the full document
# -------------------------------------------------------------
RELIGIOUS_GENRE_MARKERS = {
    "ಭಕ್ತಿ", "ಪೂಜೆ", "ದೇವಾಲಯ", "ಮಂದಿರ", "ಗುಡಿ",
    "ಅವತಾರ", "ಭಗವಂತ", "ಮೋಕ್ಷ", "ಧರ್ಮ", "ಪಾಪ", "ಪುಣ್ಯ",
    "ಸ್ವರ್ಗ", "ಆಶೀರ್ವಾದ", "ಯಜ್ಞ", "ಹವನ", "ಮಂತ್ರ",
    "ಸ್ತೋತ್ರ", "ಪ್ರಸಾದ", "ದರ್ಶನ", "ತೀರ್ಥ", "ಕ್ಷೇತ್ರ",
    "ಪುರಾಣ", "ಮಹಾಭಾರತ", "ರಾಮಾಯಣ", "ಗೀತೆ", "ವೇದ",
    "ಭಗವದ್ಗೀತೆ", "ಉಪನಿಷದ್", "ಸಂಸ್ಕೃತ",
}

SECULAR_GENRE_MARKERS = {
    "ಚುನಾವಣೆ", "ಸರ್ಕಾರ", "ಕಚೇರಿ", "ನ್ಯಾಯಾಲಯ",
    "ಶಾಲೆ", "ಕಾಲೇಜು", "ವಿಶ್ವವಿದ್ಯಾಲಯ",
    "ಆಸ್ಪತ್ರೆ", "ಔಷಧ", "ಚಿಕಿತ್ಸೆ",
    "ಬಸ್ಸು", "ರೈಲು", "ವಿಮಾನ",
    "ಸುದ್ದಿ", "ಪತ್ರಿಕೆ", "ವರದಿ",
    "ಕ್ರೀಡೆ", "ಕ್ರಿಕೆಟ್", "ಫುಟ್‌ಬಾಲ್",
}


# -------------------------------------------------------------
# VERB SUFFIX MAPPING
# Maps non-honorific verb endings to their honorific equivalents
# Format: non-honorific_suffix → honorific_suffix
# These are suffix-level rules — applied to verb endings only
# -------------------------------------------------------------
VERB_SUFFIX_MAP = {
    # Present tense — simple
    "ತ್ತಾನೆ":      "ತ್ತಾರೆ",
    "ತ್ತಾಳೆ":      "ತ್ತಾರೆ",
    # Present continuous
    "ತ್ತಿದ್ದಾನೆ":   "ತ್ತಿದ್ದಾರೆ",
    "ತ್ತಿದ್ದಾಳೆ":   "ತ್ತಿದ್ದಾರೆ",
    # Present perfect continuous
    "ದ್ದಾನೆ":      "ದ್ದಾರೆ",
    "ದ್ದಾಳೆ":      "ದ್ದಾರೆ",
    # Past tense — long forms (longest first — critical)
    "ಸಿದ್ದನು":     "ಸಿದ್ದರು",
    "ಸಿದ್ದಳು":     "ಸಿದ್ದರು",
    "ಟ್ಟನು":       "ಟ್ಟರು",
    "ಟ್ಟಳು":       "ಟ್ಟರು",
    "ದ್ದನು":       "ದ್ದರು",
    "ದ್ದಳು":       "ದ್ದರು",
    # Past tense — short forms
    "ದನು":         "ದರು",
    "ದಳು":         "ದರು",
    "ನನು":         "ನರು",
    "ನಳು":         "ನರು",
    "ತನು":         "ತರು",
    "ತಳು":         "ತರು",
    # Perfect forms
    "ಇದ್ದನು":      "ಇದ್ದರು",
    "ಇದ್ದಳು":      "ಇದ್ದರು",
    # Future tense
    "ಆನು":         "ಆರು",
    "ಆಳು":         "ಆರು",
    # Imperative / other
    "ಆನೆ":         "ಆರೆ",
    "ಆಳೆ":         "ಆರೆ",
}


# -------------------------------------------------------------
# STANDALONE POSTPOSITIONS
# Whole-word postpositions (not suffix-based)
# These are common Kannada words that always act as postpositions
# Added to pos_tagging.py classify_word() as whole-word ADP check
# -------------------------------------------------------------
STANDALONE_POSTPOSITIONS = {
    "ಜೊತೆ", "ಜೊತೆಗೆ", "ಮೇಲೆ", "ಕೆಳಗೆ", "ಹತ್ತಿರ",
    "ಬಳಿ", "ಮುಂದೆ", "ಹಿಂದೆ", "ಒಳಗೆ", "ಹೊರಗೆ",
    "ನಡುವೆ", "ಕುರಿತು", "ಬಗ್ಗೆ", "ಸಹ", "ಕೂಡ",
    "ಅನಂತರ", "ಮೊದಲು", "ನಂತರ", "ಪ್ರಕಾರ", "ವಿರುದ್ಧ",
}


# -------------------------------------------------------------
# CONFIDENCE THRESHOLDS
# Controls when the system acts vs when it escalates to user
# -------------------------------------------------------------
CONFIDENCE = {
    "ACT":     0.65,   # Score above this → system acts automatically
    "ESCALATE": 0.40,  # Score below this → always ask user
    # Between ACT and ESCALATE → system acts but logs uncertainty
}