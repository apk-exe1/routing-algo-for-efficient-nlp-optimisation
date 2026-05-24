#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fst_code.py
-----------
Finite State Transducer (FST) for Kannada Verb Morphology
Used in OCR Post-Correction Pipeline (Three-Tier Hybrid Framework)

Supports:
  - Infinitive form generation
  - Present tense conjugation (all persons/numbers/genders)
  - Past tense conjugation
  - Sandhi (euphonic junction) rules
  - Verb form analysis (reverse parse)
  - Context-aware verb correction
  - Generic rule-based generation for all verb classes

Author: MTech Thesis Project

Verb classes:
  vowel_i  — roots ending in ಿ  matra  (e.g. ಕುಡಿ, ಕಲಿ, ತಿಳಿ)
  vowel_e  — roots ending in ೆ  matra  (e.g. ಬರೆ, ತೆಗೆ, ನಡೆ)
  vowel_u  — roots ending in ು  matra  (e.g. ಮಾಡು, ನೋಡು, ಹೋಗು)
  nasal    — roots ending with nasal geminate + ು  (e.g. ತಿನ್ನು, ನಿಲ್ಲು)
  irregular — suppletive forms stored in lookup table  (e.g. ಬರು)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1.  VERB LEXICON
# ─────────────────────────────────────────────────────────────────────────────

VERB_LEXICON = {
    # vowel_i class (root ends in ಿ matra)
    "ಕುಡಿ":   {"class": "vowel_i",   "meaning": "drink"},
    "ಕಲಿ":    {"class": "vowel_i",   "meaning": "learn"},
    "ತಿಳಿ":   {"class": "vowel_i",   "meaning": "know/understand"},
    # vowel_e class (root ends in ೆ matra)
    "ಬರೆ":    {"class": "vowel_e",   "meaning": "write"},
    "ತೆಗೆ":   {"class": "vowel_e",   "meaning": "take"},
    "ನಡೆ":    {"class": "vowel_e",   "meaning": "walk"},
    # vowel_u class (root ends in ು matra)
    "ಮಾಡು":  {"class": "vowel_u",   "meaning": "do/make"},
    "ಹೋಗು":  {"class": "vowel_u",   "meaning": "go"},
    "ಕೊಡು":  {"class": "vowel_u",   "meaning": "give"},
    "ಇಡು":   {"class": "vowel_u",   "meaning": "place/put"},
    "ಕೇಳು":  {"class": "vowel_u",   "meaning": "listen/ask"},
    "ನೋಡು":  {"class": "vowel_u",   "meaning": "see/look"},
    "ಓಡು":   {"class": "vowel_u",   "meaning": "run"},
    "ಆಡು":   {"class": "vowel_u",   "meaning": "play"},
    "ಹಾಡು":  {"class": "vowel_u",   "meaning": "sing"},
    "ಓದು":   {"class": "vowel_u",   "meaning": "read"},
    "ಹೇಳು":  {"class": "vowel_u",   "meaning": "say/tell"},
    "ತಿರುಗು": {"class": "vowel_u",  "meaning": "turn/wander"},
    "ಕೂರು":  {"class": "vowel_u",   "meaning": "sit"},
    "ಬಿಡು":  {"class": "vowel_u",   "meaning": "leave/release"},
    "ಸೇರು":  {"class": "vowel_u",   "meaning": "join/reach"},
    # nasal class (root ends in nasal geminate + ು)
    "ತಿನ್ನು": {"class": "nasal",    "meaning": "eat"},
    "ನಿಲ್ಲು": {"class": "nasal",    "meaning": "stand/stop"},
    # irregular class (suppletive past, uses lookup table)
    "ಬರು":   {"class": "irregular", "meaning": "come"},
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SUFFIX TABLES
# ─────────────────────────────────────────────────────────────────────────────

# Infinitive base suffix (before sandhi is applied)
INFINITIVE_SUFFIX = "ಅಲು"

# Present tense suffixes keyed by (person, number, gender)
#   gender: 'M'=masculine, 'F'=feminine, 'N'=neuter, '-'=gender-neutral
PRESENT_SUFFIXES = {
    (1, "SG", "-"): "ಉತ್ತೇನೆ",
    (1, "PL", "-"): "ಉತ್ತೇವೆ",
    (2, "SG", "-"): "ಉತ್ತೀಯ",
    (2, "PL", "-"): "ಉತ್ತೀರಿ",
    (3, "SG", "M"): "ಉತ್ತಾನೆ",
    (3, "SG", "F"): "ಉತ್ತಾಳೆ",
    (3, "SG", "N"): "ಉತ್ತದೆ",
    (3, "PL", "-"): "ಉತ್ತಾರೆ",
}

# Past tense suffixes (attached to the past stem, not the root directly)
PAST_SUFFIXES = {
    (1, "SG", "-"): "ದೆನು",
    (1, "PL", "-"): "ದೆವು",
    (2, "SG", "-"): "ದೆ",
    (2, "PL", "-"): "ದಿರಿ",
    (3, "SG", "M"): "ದನು",
    (3, "SG", "F"): "ದಳು",
    (3, "SG", "N"): "ತು",
    (3, "PL", "-"): "ದರು",
}

# ── Past stem derivation ─────────────────────────────────────────────────────
#
# Generic rules (no per-verb listing needed):
#   vowel_u / nasal  →  strip final ು matra, append ಿ matra
#                         e.g.  ನೋಡು → ನೋಡಿ,  ತಿನ್ನು → ತಿನ್ನಿ
#   vowel_i          →  root unchanged  (ಕುಡಿ → ಕುಡಿ)
#   vowel_e          →  root unchanged  (ಬರೆ  → ಬರೆ)
#   irregular        →  handled by IRREGULAR_FORMS lookup table
#
# Add entries here only when a specific root deviates from the generic rule:
PAST_STEM_OVERRIDE = {
    # "ಕೊಡು": "ಕೊಟ್ಟ",  # example of an override entry (not needed here)
}


def _get_past_stem(root, verb_class):
    """
    Return the past stem of *root* for attaching a PAST_SUFFIXES entry.

    Generic rules:
      vowel_u / nasal  →  strip final ು, append ಿ
      vowel_i / vowel_e →  root unchanged
    """
    if root in PAST_STEM_OVERRIDE:
        return PAST_STEM_OVERRIDE[root]
    if verb_class in ("vowel_u", "nasal"):
        if root.endswith("ು"):     # U+0CC1
            return root[:-1] + "ಿ"  # U+0CBF
        return root
    return root  # vowel_i and vowel_e: attach directly


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONTEXT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

# Words in a sentence that signal an infinitive form is required
INFINITIVE_TRIGGERS = {
    "ಬೇಕು", "ಬೇಕಾಗಿದೆ", "ಬೇಕಿದೆ",
    "ಆಗುತ್ತದೆ", "ಆಗುವುದಿಲ್ಲ",
    "ಸಾಧ್ಯ", "ಆಗದು",
    "ಮಾಡಿ", "ಹೋಗು", "ಬಾ",
    "ಮರೆತ",
}

# Canonical subject pronouns only (genitive / oblique forms excluded)
SUBJECT_FEATURES = {
    "ನಾನು": (1, "SG", "-"),
    "ನಾವು": (1, "PL", "-"),
    "ನೀನು": (2, "SG", "-"),
    "ನೀವು": (2, "PL", "-"),
    "ಅವನು": (3, "SG", "M"),
    "ಅವಳು": (3, "SG", "F"),
    "ಅದು":  (3, "SG", "N"),
    "ಅವರು": (3, "PL", "-"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SANDHI (EUPHONIC JUNCTION) ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SandhiProcessor:
    """
    Applies vowel sandhi when a suffix is attached to a verb root.

    Rules
    -----
    R1  vowel_i  (root ends ಿ)
        + vowel-initial suffix  →  insert euphonic ಯ
          suffix starts ಅ  →  ಯ + rest       (ಅ is silent after ಯ)
          suffix starts V  →  ಯ + sign(V) + rest

        ಕುಡಿ + ಅಲು     → ಕುಡಿ + ಯ + ಲು         = ಕುಡಿಯಲು
        ಕುಡಿ + ಉತ್ತೇನೆ  → ಕುಡಿ + ಯ + ು + ತ್ತೇನೆ  = ಕುಡಿಯುತ್ತೇನೆ

    R2  vowel_e  (root ends ೆ)
        Same as R1.
        ಬರೆ + ಅಲು     → ಬರೆಯಲು
        ಬರೆ + ಉತ್ತೀಯ  → ಬರೆಯುತ್ತೀಯ

    R3  vowel_u  (root ends ು)
        suffix starts ಉ  →  drop leading ಉ of suffix
                            ಮಾಡು + ಉತ್ತೇನೆ → ಮಾಡು + ತ್ತೇನೆ = ಮಾಡುತ್ತೇನೆ
        suffix starts ಅ  →  drop root-final ು AND drop suffix-initial ಅ
                            ಮಾಡು + ಅಲು → ಮಾಡ + ಲು = ಮಾಡಲು
                            ಹೋಗು + ಅಲು → ಹೋಗಲು
                            ನೋಡು + ಅಲು → ನೋಡಲು

    R4  nasal  (root ends in nasal geminate + ು)
        suffix starts ಉ  →  drop leading ಉ of suffix
                            ತಿನ್ನು + ಉತ್ತೇನೆ → ತಿನ್ನು + ತ್ತೇನೆ = ತಿನ್ನುತ್ತೇನೆ
        suffix starts ಅ  →  drop root-final ು AND drop suffix-initial ಅ
                            ತಿನ್ನು + ಅಲು → ತಿನ್ನ + ಲು = ತಿನ್ನಲು

    R5  irregular  →  all forms come from IRREGULAR_FORMS; apply() not called.
    """

    # Independent vowel → dependent (matra) sign
    VOWEL_TO_SIGN = {
        "ಅ": "",   # inherent, silent after euphonic glide
        "ಆ": "ಾ",
        "ಇ": "ಿ",
        "ಈ": "ೀ",
        "ಉ": "ು",
        "ಊ": "ೂ",
        "ಎ": "ೆ",
        "ಏ": "ೇ",
        "ಒ": "ೊ",
        "ಓ": "ೋ",
    }

    # Fully irregular verb forms (all tenses/persons/numbers/genders)
    IRREGULAR_FORMS = {
        ("ಬರು", "infinitive"):             "ಬರಲು",
        ("ಬರು", "present", 1, "SG", "-"):  "ಬರುತ್ತೇನೆ",
        ("ಬರು", "present", 1, "PL", "-"):  "ಬರುತ್ತೇವೆ",
        ("ಬರು", "present", 2, "SG", "-"):  "ಬರುತ್ತೀಯ",
        ("ಬರು", "present", 2, "PL", "-"):  "ಬರುತ್ತೀರಿ",
        ("ಬರು", "present", 3, "SG", "M"):  "ಬರುತ್ತಾನೆ",
        ("ಬರು", "present", 3, "SG", "F"):  "ಬರುತ್ತಾಳೆ",
        ("ಬರು", "present", 3, "SG", "N"):  "ಬರುತ್ತದೆ",
        ("ಬರು", "present", 3, "PL", "-"):  "ಬರುತ್ತಾರೆ",
        ("ಬರು", "past",    1, "SG", "-"):  "ಬಂದೆನು",
        ("ಬರು", "past",    1, "PL", "-"):  "ಬಂದೆವು",
        ("ಬರು", "past",    2, "SG", "-"):  "ಬಂದೆ",
        ("ಬರು", "past",    2, "PL", "-"):  "ಬಂದಿರಿ",
        ("ಬರು", "past",    3, "SG", "M"):  "ಬಂದನು",
        ("ಬರು", "past",    3, "SG", "F"):  "ಬಂದಳು",
        ("ಬರು", "past",    3, "SG", "N"):  "ಬಂತು",
        ("ಬರು", "past",    3, "PL", "-"):  "ಬಂದರು",
    }

    _MATRA_U = "ು"   # U+0CC1

    def apply(self, root, suffix, verb_class):
        """
        Join *root* + *suffix* with correct Kannada sandhi.
        Returns the surface form string.
        """
        if not root or not suffix:
            return root + suffix

        # Irregular forms are resolved before apply() is called; should not arrive here.
        if verb_class == "irregular":
            verb_class = "vowel_u"   # safe fallback

        # ── R1 / R2: vowel_i and vowel_e ─────────────────────────────────────
        if verb_class in ("vowel_i", "vowel_e"):
            first = suffix[0]
            if first in self.VOWEL_TO_SIGN:
                sign = self.VOWEL_TO_SIGN[first]
                if first == "ಅ":
                    return root + "ಯ" + suffix[1:]
                return root + "ಯ" + sign + suffix[1:]
            return root + suffix

        # ── R3: vowel_u ───────────────────────────────────────────────────────
        if verb_class == "vowel_u":
            if suffix.startswith("ಉ"):
                # Drop leading ಉ; root already carries the /u/ vowel
                return root + suffix[1:]
            if suffix.startswith("ಅ"):
                # Drop root-final ು AND suffix-initial ಅ
                stem = root[:-1] if root.endswith(self._MATRA_U) else root
                return stem + suffix[1:]
            return root + suffix

        # ── R4: nasal ─────────────────────────────────────────────────────────
        if verb_class == "nasal":
            if suffix.startswith("ಉ"):
                return root + suffix[1:]
            if suffix.startswith("ಅ"):
                stem = root[:-1] if root.endswith(self._MATRA_U) else root
                return stem + suffix[1:]
            return root + suffix

        # Fallback
        return root + suffix


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FST STATE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class FSTState:
    """Named constants for FST transition states."""
    START                   = "START"
    VERB_ROOT               = "VERB_ROOT"
    TENSE_SELECTION         = "TENSE_SELECTION"
    PERSON_NUMBER_SELECTION = "PERSON_NUMBER_SELECTION"
    SUFFIX_ATTACHMENT       = "SUFFIX_ATTACHMENT"
    SANDHI_ADJUSTMENT       = "SANDHI_ADJUSTMENT"
    FINAL_SURFACE_FORM      = "FINAL_SURFACE_FORM"
    ERROR                   = "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN FST CLASS
# ─────────────────────────────────────────────────────────────────────────────

class FSTMorphology:
    """
    Finite State Transducer for Kannada Verb Morphology.

    FST state diagram
    -----------------
    START
      ↓
    VERB_ROOT               — look up root in lexicon, get verb class
      ↓
    TENSE_SELECTION         — choose tense: present / past / infinitive
      ↓
    PERSON_NUMBER_SELECTION — select (person, number, gender) features
      ↓
    SUFFIX_ATTACHMENT       — fetch correct suffix from table
      ↓
    SANDHI_ADJUSTMENT       — apply sandhi rules
      ↓
    FINAL_SURFACE_FORM      — output surface string

    Public API
    ----------
    generate(root, tense, person, number, gender) → str
    generate_from_context(root, sentence) → str
    analyze(word) → dict
    repair_ocr_verb_form(word) → str
    correct_verb_form(wrong_verb, sentence) → str
    filter_predictions_by_context(candidates, sentence, top_k) → list
    get_trace() → list
    print_trace()
    """

    def __init__(self):
        self.lexicon = VERB_LEXICON
        self.sandhi  = SandhiProcessor()
        self._state  = FSTState.START
        self._trace  = []

    # ── 6.1  Generate ─────────────────────────────────────────────────────────

    def generate(self, root, tense="present", person=3, number="SG", gender="-"):
        """
        Generate the surface form of *root* for the given morphological features.

        Parameters
        ----------
        root   : Kannada verb root (must be in VERB_LEXICON)
        tense  : 'present' | 'past' | 'infinitive'
        person : 1 | 2 | 3
        number : 'SG' | 'PL'
        gender : 'M' | 'F' | 'N' | '-'

        Returns
        -------
        Surface verb string, or '' on failure.
        """
        self._trace = []
        self._transition(FSTState.START)

        # ── VERB_ROOT ─────────────────────────────────────────────────────────
        if root not in self.lexicon:
            self._transition(FSTState.ERROR, f"Unknown root: {root}")
            return ""
        verb_class = self.lexicon[root]["class"]
        self._transition(FSTState.VERB_ROOT,
                         f"root={root}, class={verb_class}")

        # ── TENSE_SELECTION ───────────────────────────────────────────────────
        self._transition(FSTState.TENSE_SELECTION, f"tense={tense}")

        # ── Handle irregular verbs via lookup table ───────────────────────────
        if verb_class == "irregular":
            key = (root, "infinitive") if tense == "infinitive" \
                  else (root, tense, person, number, gender)
            form = self.sandhi.IRREGULAR_FORMS.get(key, "")
            if form:
                self._transition(FSTState.FINAL_SURFACE_FORM, form)
                return form
            # Not in table → fall through as vowel_u
            verb_class = "vowel_u"

        # ── PERSON_NUMBER_SELECTION ───────────────────────────────────────────
        self._transition(FSTState.PERSON_NUMBER_SELECTION,
                         f"person={person}, number={number}, gender={gender}")

        # ── SUFFIX_ATTACHMENT ─────────────────────────────────────────────────
        suffix = self._pick_suffix(tense, person, number, gender)
        if suffix is None:
            self._transition(FSTState.ERROR,
                             f"No suffix for tense={tense}, "
                             f"({person},{number},{gender})")
            return ""
        self._transition(FSTState.SUFFIX_ATTACHMENT, f"suffix={suffix!r}")

        # ── SANDHI_ADJUSTMENT ─────────────────────────────────────────────────
        self._transition(FSTState.SANDHI_ADJUSTMENT)

        if tense == "past":
            # Derive the past stem then attach suffix directly (no sandhi)
            stem    = _get_past_stem(root, verb_class)
            surface = stem + suffix
        else:
            # Present and infinitive: full sandhi
            surface = self.sandhi.apply(root, suffix, verb_class)

        # ── FINAL_SURFACE_FORM ────────────────────────────────────────────────
        self._transition(FSTState.FINAL_SURFACE_FORM, surface)
        return surface

    # ── 6.2  Generate from context ────────────────────────────────────────────

    def generate_from_context(self, root, sentence):
        """
        Auto-detect the required verb form from *sentence* and generate it.

        Priority:
          1. Infinitive trigger word present  → infinitive
          2. Subject pronoun found            → present tense matching subject
          3. Default                          → 3rd SG neuter present
        """
        if root not in self.lexicon:
            return ""

        tokens = sentence.split()

        for token in tokens:
            if token in INFINITIVE_TRIGGERS:
                return self.generate(root, tense="infinitive")

        for token in tokens:
            if token in SUBJECT_FEATURES:
                p, n, g = SUBJECT_FEATURES[token]
                return self.generate(root, tense="present",
                                     person=p, number=n, gender=g)

        return self.generate(root, tense="present",
                             person=3, number="SG", gender="N")

    # ── 6.3  Analyze (reverse parse) ──────────────────────────────────────────

    def analyze(self, word):
        """
        Reverse-parse *word* to extract root and morphological features.

        The word is first passed through OCR normalization so that common
        OCR errors do not prevent analysis.

        Returns a dict with keys:
          root, tense, person, number, gender, surface, confidence
        Returns {} if no match is found.
        """
        word = normalize_ocr_errors(word)
        for root in self.lexicon:
            # Check infinitive
            candidate = self.generate(root, tense="infinitive")
            if candidate and candidate == word:
                return {
                    "root": root, "tense": "infinitive",
                    "person": None, "number": None, "gender": None,
                    "surface": word, "confidence": 1.0,
                }
            # Check present and past
            for tense in ("present", "past"):
                table = PRESENT_SUFFIXES if tense == "present" else PAST_SUFFIXES
                for (p, n, g) in table:
                    candidate = self.generate(root, tense=tense,
                                              person=p, number=n, gender=g)
                    if candidate and candidate == word:
                        return {
                            "root": root, "tense": tense,
                            "person": p, "number": n, "gender": g,
                            "surface": word, "confidence": 1.0,
                        }
        return {}

    # ── 6.4  Correct verb form ────────────────────────────────────────────────

    def repair_ocr_verb_form(self, word):
        """
        Morphology-aware OCR repair.
        Attempts to repair a word using morphological knowledge.
        """
        import re

        # Step 1: Try to detect whether the word already matches a valid generated verb form
        if self.analyze(normalize_ocr_errors(word)):
            return word

        # Step 2: Attempt repairs using morphological patterns
        repaired = word

        # Rule A – Remove stray "ಅ" after a vowel matra
        MATRAS = "ಾಿೀುೂೆೇೈೊೋೌ"
        repaired = re.sub(f'([{re.escape(MATRAS)}])ಅ', r'\1', repaired)

        # Rule B – Restore missing consonant doubling in present tense suffixes
        VERB_SUFFIX_REPAIRS = [
            ("ತೇನೆ",  "ತ್ತೇನೆ"),
            ("ತೇವೆ",  "ತ್ತೇವೆ"),
            ("ತೀರಿ",  "ತ್ತೀರಿ"),
            ("ತೀಯ",   "ತ್ತೀಯ"),
            ("ತಾನೆ",  "ತ್ತಾನೆ"),
            ("ತಾಳೆ",  "ತ್ತಾಳೆ"),
            ("ತಾರೆ",  "ತ್ತಾರೆ"),
            ("ತದೆ",   "ತ್ತದೆ"),
        ]
        for bad, good in VERB_SUFFIX_REPAIRS:
            if repaired.endswith(bad):
                repaired = repaired[:-len(bad)] + good
                break

        # Rule C – Restore missing glide "ಯ" before "ಲು"
        repaired = re.sub(r'ಿಲು$', 'ಿಯಲು', repaired)
        repaired = re.sub(r'ೆಲು$', 'ೆಯಲು', repaired)

        # Verify that the repaired form is morphologically valid
        if self.analyze(repaired):
            return repaired

        # Return original word if repairs don't lead to a valid form
        return word

    def correct_verb_form(self, wrong_verb, sentence):
        """
        Given a (possibly incorrect) verb token and its sentence, return the
        most plausible correct surface form.

        Correction pipeline:
          1. normalize OCR (regex baseline)
          2. attempt morphological repair
          3. analyze -> generate correct form
        """
        # 1. Regex baseline
        normalized = normalize_ocr_errors(wrong_verb)
        
        # 2. Morphology-aware repair
        repaired_verb = self.repair_ocr_verb_form(normalized)

        # 3. Analyze and generate
        info = self.analyze(repaired_verb)
        if info:
            return self.generate_from_context(info["root"], sentence)

        root = self._guess_root(repaired_verb)
        if root:
            return self.generate_from_context(root, sentence)

        return repaired_verb

    # ── 6.5  Filter predictions by morphological context ─────────────────────

    def filter_predictions_by_context(self, candidates, sentence, top_k=5):
        """
        Re-rank *candidates* by morphological plausibility given *sentence*.

        Scoring:
          3  — candidate exactly matches the contextually expected form
          2  — candidate is a valid generated form for some known root
          1  — candidate's stem appears to be a known root
          0  — unknown
        """
        scored = []
        for cand in candidates:
            score = 0
            info  = self.analyze(cand)
            if info:
                expected = self.generate_from_context(info["root"], sentence)
                score = 3 if expected == cand else 2
            else:
                root = self._guess_root(cand)
                if root:
                    score = 1
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    # ── 6.6  Trace / debug ────────────────────────────────────────────────────

    def get_trace(self):
        """Return FST state-transition trace from the last generate() call."""
        return list(self._trace)

    def print_trace(self):
        """Pretty-print the FST transition path."""
        print("\n── FST Transition Trace ──────────────────────────────")
        for step, (state, note) in enumerate(self._trace):
            note_str = f"  [{note}]" if note else ""
            print(f"  Step {step:2d}: {state}{note_str}")
        print("─────────────────────────────────────────────────────\n")

    # ── 6.7  Internal helpers ─────────────────────────────────────────────────

    def _transition(self, state, note=""):
        self._state = state
        self._trace.append((state, note))

    def _pick_suffix(self, tense, person, number, gender):
        """Return the raw suffix string for the given features, or None."""
        if tense == "infinitive":
            return INFINITIVE_SUFFIX
        if tense == "present":
            return (PRESENT_SUFFIXES.get((person, number, gender))
                    or PRESENT_SUFFIXES.get((person, number, "-")))
        if tense == "past":
            return (PAST_SUFFIXES.get((person, number, gender))
                    or PAST_SUFFIXES.get((person, number, "-")))
        return None

    def _guess_root(self, word):
        """
        Try to recover a known root from a surface verb form.

        Strategy
        --------
        1. Build the set of all surface suffixes that can appear after sandhi.
        2. For each matching suffix, extract the bare stem.
        3. Apply candidate-root normalization rules to the stem:

           a) Direct match         : stem is already a root  (e.g. ಕುಡಿ, ಬರೆ)
           b) ಯ-reversal (vowel_i/e): stem ends "ಯ"  → try base + "ಿ" or "ೆ"
              e.g. ಕುಡಿಯ  →  ಕುಡಿ
                   ಬರೆಯ   →  ಬರೆ
           c) vowel_u infinitive   : stem is bare consonant (lost ು)
              e.g. ಮಾಡ  →  ಮಾಡು
           d) past stem reversal   : stem ends "ಿ" (stripped ು, added ಿ)
              e.g. ನೋಡಿ  →  ನೋಡು,  ತಿನ್ನಿ → ತಿನ್ನು

        All rules are generic (class-independent) and work for any verb in
        the lexicon.
        """
        # ── Build surface suffix set ──────────────────────────────────────────
        # Include raw suffix table values AND the actual surface forms produced
        # after sandhi (the vowel_i/e present suffixes begin with ಉ which becomes
        # ಯು on the surface, but we strip the raw suffix from the word directly).
        surface_suffixes = set()
        surface_suffixes.add("ಯಲು")    # vowel_i / vowel_e infinitive surface
        surface_suffixes.add("ಲು")     # vowel_u / nasal infinitive surface
        surface_suffixes.update(PRESENT_SUFFIXES.values())   # e.g. "ಉತ್ತೇನೆ"
        surface_suffixes.update(PAST_SUFFIXES.values())      # e.g. "ದೆನು"

        # Also add surface forms after vowel_i/e sandhi: ಉXXX → ಯುXXX
        # In the actual surface word the ಉ becomes ು matra after euphonic ಯ.
        # e.g. suffix "ಉತ್ತೇನೆ" → surface "ಯುತ್ತೇನೆ" (ಯ + ು + ತ್ತೇನೆ)
        # We build this by: "ಯ" + "ು" + suffix[1:]  (drop independent ಉ, add ು matra)
        for sfx in list(PRESENT_SUFFIXES.values()):
            if sfx.startswith("ಉ"):
                surface_suffixes.add("ಯ" + "ು" + sfx[1:])   # e.g. ಯುತ್ತೇನೆ

        for sfx in sorted(surface_suffixes, key=len, reverse=True):
            if not word.endswith(sfx):
                continue
            stem = word[: len(word) - len(sfx)]

            # ── a) Direct match ───────────────────────────────────────────────
            if stem in self.lexicon:
                return stem

            # ── b) ಯ-ending reversal (vowel_i / vowel_e sandhi) ───────────────
            # Sandhi inserts euphonic ಯ: root + ಯ + (vowel-sign) + suffix-rest
            # Stripping the raw suffix leaves: root + ಯ  (or root + ಯು if
            # the suffix started with ಉ and sign was ು, but stripping "ಉತ್ತ..."
            # leaves the ಯ without the ು because ು was part of the suffix).
            # Handle both "ಯ" and "ಯು" endings:
            if stem.endswith("ಯು"):
                base = stem[:-2]            # strip "ಯು"
                for cand in (base + "ಿ", base + "ೆ"):
                    if cand in self.lexicon:
                        return cand
            if stem.endswith("ಯ"):
                base = stem[:-1]            # strip "ಯ"
                for cand in (base + "ಿ", base + "ೆ"):
                    if cand in self.lexicon:
                        return cand

            # ── c) vowel_u infinitive: stem is bare consonant cluster ─────────
            # ಮಾಡಲು → strip ಲು → ಮಾಡ  → ಮಾಡ + ು = ಮಾಡು
            cand_u = stem + "ು"
            if cand_u in self.lexicon:
                return cand_u

            # ── d) Past stem reversal: stem ends "ಿ" → restore "ು" ───────────
            # ನೋಡಿದೆನು → strip ದೆನು → ನೋಡಿ  → ನೋಡಿ[:-1] + ು = ನೋಡು
            # ತಿನ್ನಿದೆನು → strip → ತಿನ್ನಿ → ತಿನ್ನು
            if stem.endswith("ಿ"):
                cand_past = stem[:-1] + "ು"
                if cand_past in self.lexicon:
                    return cand_past

        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 7.  OCR ERROR NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_ocr_errors(word):
    """
    Apply rule-based corrections for common Kannada OCR mistakes.

    This is a pre-processing step that should be called before morphological
    analysis or generation when the input may contain OCR noise.

    Rules applied (in order)
    ------------------------
    Case 1 — Spurious ಅ inserted between root and suffix
        Pattern : vowel-ending root + ಅ + suffix-initial consonant
        Symptom : ಕುಡಿಅಲು  (extra independent ಅ after root-final ಿ/ೆ/ು)
        Fix     : remove the stray ಅ
        e.g.    ಕುಡಿಅಲು → ಕುಡಿಯಲು  (via matra + glide insertion downstream)
        Simpler : any matra (ಿ ೆ ು ಾ ೀ ೇ ...) immediately followed by ಅ
                  → drop the ಅ

    Case 2 — Missing consonant doubling of ತ್ತ
        Pattern : single ತ followed by vowel sign, where ತ್ತ is expected
        Symptom : ಮಾಡುತೇನೆ  (ತ instead of ತ್ತ)
                  ಬರೆಯುತಾನೆ (ತ instead of ತ್ತ)
        Fix     : ತ + vowel-sign (ೇ/ಾ/ೀ/ೆ/ದ) → ತ್ತ + vowel-sign
        Rule    : if 'ತ' is followed by a vowel sign that is NOT preceded
                  by the virama (್), insert ್ತ before the vowel sign.

    Case 3 — Missing euphonic ಯ glide before ಲು
        Pattern : root-final ಿ or ೆ directly followed by ಲು
        Symptom : ಕುಡಿಲು  (missing ಯ), ಬರೆಲು
        Fix     : ಿ + ಲು → ಿಯಲು,  ೆ + ಲು → ೆಯಲು

    Parameters
    ----------
    word : str — possibly OCR-corrupted Kannada verb form

    Returns
    -------
    str — normalized word with common OCR errors corrected
    """
    import re

    result = word

    # ── Case 1: stray ಅ (U+0C85) after a dependent vowel sign ────────────────
    # Dependent vowel signs (matras): ಾ ಿ ೀ ು ೂ ೆ ೇ ೈ ೊ ೋ ೌ
    # If any of these is immediately followed by independent ಅ (U+0C85), drop it.
    MATRAS = "ಾಿೀುೂೆೇೈೊೋೌ"
    # Build a character class pattern
    matra_pattern = re.compile(f'([{re.escape(MATRAS)}])ಅ')
    result = matra_pattern.sub(r'\1', result)

    # ── Case 2: missing ್ತ in present-tense verb suffixes only ───────────────
    #
    # PROBLEM WITH THE PREVIOUS RULE:
    #   The original rule matched ANY ತ + vowel-sign regardless of context,
    #   which incorrectly modified non-verb words such as:
    #     ಪತಿ → ಪತ್ತಿ,  ಮತಿ → ಮತ್ತಿ,  ಮತೇ → ಮತ್ತೇ
    #
    # SAFE REPLACEMENT:
    #   Match ONLY the specific verb-suffix endings where ತ್ತ is expected.
    #   These are the truncated (OCR-dropped) forms of the standard present-tense
    #   suffixes:
    #
    #     ತ್ತೇನೆ  →  OCR may produce  ತೇನೆ
    #     ತ್ತೇವೆ  →  OCR may produce  ತೇವೆ
    #     ತ್ತೀಯ   →  OCR may produce  ತೀಯ
    #     ತ್ತೀರಿ  →  OCR may produce  ತೀರಿ
    #     ತ್ತಾನೆ  →  OCR may produce  ತಾನೆ
    #     ತ್ತಾಳೆ  →  OCR may produce  ತಾಳೆ
    #     ತ್ತದೆ   →  OCR may produce  ತದೆ
    #     ತ್ತಾರೆ  →  OCR may produce  ತಾರೆ
    #
    #   The pattern is: a verb root ending (ು or ಯು) + ತ + suffix-vowel-sign + rest
    #   We match the truncated suffix literally so that ಪತಿ, ಮತಿ, ಮತೇ are
    #   never touched.
    #
    #   Each entry in VERB_SUFFIX_REPAIRS maps the OCR-corrupted ending →
    #   the correct ending.  Only these exact strings are replaced.

    VERB_SUFFIX_REPAIRS = [
        # (corrupted_suffix,  correct_suffix)
        # Ordered longest-first so that more specific patterns match first.
        ("ತೇನೆ",  "ತ್ತೇನೆ"),   # 1st SG present
        ("ತೇವೆ",  "ತ್ತೇವೆ"),   # 1st PL present
        ("ತೀರಿ",  "ತ್ತೀರಿ"),   # 2nd PL present
        ("ತೀಯ",   "ತ್ತೀಯ"),    # 2nd SG present
        ("ತಾನೆ",  "ತ್ತಾನೆ"),   # 3rd SG M present
        ("ತಾಳೆ",  "ತ್ತಾಳೆ"),   # 3rd SG F present
        ("ತಾರೆ",  "ತ್ತಾರೆ"),   # 3rd PL present
        ("ತದೆ",   "ತ್ತದೆ"),    # 3rd SG N present
    ]

    for bad, good in VERB_SUFFIX_REPAIRS:
        if result.endswith(good):
            break 

        if result.endswith(bad):
            result = result[:-len(bad)] + good
            break  # at most one suffix repair per word

    # ── Case 3: missing ಯ glide — ಿಲು or ೆಲು → ಿಯಲು or ೆಯಲು ─────────────────
    result = re.sub(r'ಿಲು$', 'ಿಯಲು', result)
    result = re.sub(r'ೆಲು$', 'ೆಯಲು', result)

    return result




def get_fst():
    """Return a module-level singleton FSTMorphology instance."""
    if not hasattr(get_fst, "_instance"):
        get_fst._instance = FSTMorphology()
    return get_fst._instance


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SELF-TEST  (run: python fst_code.py)
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests():
    fst  = FSTMorphology()
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"

    # ── 8.1  Unit tests ───────────────────────────────────────────────────────
    # (description, root, tense, person, number, gender, expected_output)
    tests = [
        # vowel_i class
        ("ಕುಡಿ  infinitive",       "ಕುಡಿ",   "infinitive", 1, "SG", "-",  "ಕುಡಿಯಲು"),
        ("ಕುಡಿ  present 1SG",      "ಕುಡಿ",   "present",    1, "SG", "-",  "ಕುಡಿಯುತ್ತೇನೆ"),
        ("ಕುಡಿ  present 3SG M",    "ಕುಡಿ",   "present",    3, "SG", "M",  "ಕುಡಿಯುತ್ತಾನೆ"),
        ("ಕುಡಿ  past 1SG",         "ಕುಡಿ",   "past",       1, "SG", "-",  "ಕುಡಿದೆನು"),
        ("ಕಲಿ   present 1SG",      "ಕಲಿ",    "present",    1, "SG", "-",  "ಕಲಿಯುತ್ತೇನೆ"),
        ("ತಿಳಿ  present 1SG",      "ತಿಳಿ",   "present",    1, "SG", "-",  "ತಿಳಿಯುತ್ತೇನೆ"),
        # vowel_e class
        ("ಬರೆ   infinitive",       "ಬರೆ",    "infinitive", 1, "SG", "-",  "ಬರೆಯಲು"),
        ("ಬರೆ   present 2SG",      "ಬರೆ",    "present",    2, "SG", "-",  "ಬರೆಯುತ್ತೀಯ"),
        ("ಬರೆ   present 3SG M",    "ಬರೆ",    "present",    3, "SG", "M",  "ಬರೆಯುತ್ತಾನೆ"),
        ("ಬರೆ   past 3SG M",       "ಬರೆ",    "past",       3, "SG", "M",  "ಬರೆದನು"),
        # vowel_u class
        ("ಮಾಡು  infinitive",       "ಮಾಡು",  "infinitive", 1, "SG", "-",  "ಮಾಡಲು"),
        ("ಮಾಡು  present 1SG",      "ಮಾಡು",  "present",    1, "SG", "-",  "ಮಾಡುತ್ತೇನೆ"),
        ("ಮಾಡು  past 1SG",         "ಮಾಡು",  "past",       1, "SG", "-",  "ಮಾಡಿದೆನು"),
        ("ಹೋಗು  infinitive",       "ಹೋಗು",  "infinitive", 1, "SG", "-",  "ಹೋಗಲು"),
        ("ನೋಡು  infinitive",       "ನೋಡು",  "infinitive", 1, "SG", "-",  "ನೋಡಲು"),
        ("ನೋಡು  past 1SG",         "ನೋಡು",  "past",       1, "SG", "-",  "ನೋಡಿದೆನು"),
        ("ಕೂರು  infinitive",       "ಕೂರು",  "infinitive", 1, "SG", "-",  "ಕೂರಲು"),
        ("ಸೇರು  infinitive",       "ಸೇರು",  "infinitive", 1, "SG", "-",  "ಸೇರಲು"),
        # nasal class
        ("ತಿನ್ನು present 1SG",    "ತಿನ್ನು", "present",    1, "SG", "-",  "ತಿನ್ನುತ್ತೇನೆ"),
        ("ತಿನ್ನು infinitive",     "ತಿನ್ನು", "infinitive", 1, "SG", "-",  "ತಿನ್ನಲು"),
        ("ನಿಲ್ಲು infinitive",     "ನಿಲ್ಲು", "infinitive", 1, "SG", "-",  "ನಿಲ್ಲಲು"),
        # irregular class
        ("ಬರು   present 1SG",      "ಬರು",   "present",    1, "SG", "-",  "ಬರುತ್ತೇನೆ"),
        ("ಬರು   present 3SG M",    "ಬರು",   "present",    3, "SG", "M",  "ಬರುತ್ತಾನೆ"),
        ("ಬರು   past 3SG N",       "ಬರು",   "past",       3, "SG", "N",  "ಬಂತು"),
        ("ಬರು   past 3SG M",       "ಬರು",   "past",       3, "SG", "M",  "ಬಂದನು"),
        ("ಬರು   infinitive",       "ಬರು",   "infinitive", 1, "SG", "-",  "ಬರಲು"),
    ]

    print("\n" + "=" * 68)
    print("  FST MORPHOLOGY SELF-TEST")
    print("=" * 68)
    passed = 0
    for desc, root, tense, p, n, g, expected in tests:
        result = fst.generate(root, tense=tense, person=p, number=n, gender=g)
        ok     = result == expected
        passed += ok
        status  = PASS if ok else FAIL
        print(f"{status}  {desc}")
        if not ok:
            print(f"         expected : {expected}")
            print(f"         got      : {result}")

    print("-" * 68)
    print(f"  {passed}/{len(tests)} tests passed")
    print("=" * 68)

    # ── 8.2  Generic verb coverage (not explicitly in test set above) ─────────
    print("\n── Generic Verb Coverage ────────────────────────────────────────")
    generic = [
        ("ಕಲಿ",  "infinitive", 1, "SG", "-", "ಕಲಿಯಲು"),
        ("ತಿಳಿ", "infinitive", 1, "SG", "-", "ತಿಳಿಯಲು"),
        ("ಕೂರು", "present",    1, "SG", "-", "ಕೂರುತ್ತೇನೆ"),
        ("ಸೇರು", "present",    3, "SG", "M", "ಸೇರುತ್ತಾನೆ"),
        ("ಓದು",  "past",       1, "SG", "-", "ಓದಿದೆನು"),
    ]
    gen_passed = 0
    for root, tense, p, n, g, expected in generic:
        result = fst.generate(root, tense=tense, person=p, number=n, gender=g)
        ok     = result == expected
        gen_passed += ok
        status  = PASS if ok else FAIL
        print(f"  {status}  {root:8s} {tense:10s} → {result:22s}  expected: {expected}")
    print(f"  {gen_passed}/{len(generic)} generic tests passed")

    # ── 8.3  Context-based generation ─────────────────────────────────────────
    print("\n── Context-Based Generation ─────────────────────────────────────")
    ctx = [
        ("ಕುಡಿ",   "ನಾನು ನೀರನ್ನು ಕುಡಿಯಬೇಕು"),
        ("ಮಾಡು",  "ನಾನು ಕೆಲಸ ಮಾಡುತ್ತೇನೆ"),
        ("ಬರೆ",   "ಅವನು ಪತ್ರ ಬರೆಯುತ್ತಾನೆ"),
        ("ತಿನ್ನು", "ನಾವು ಊಟ ತಿನ್ನುತ್ತೇವೆ"),
        ("ನೋಡು",  "ಅವಳು ಚಿತ್ರ ನೋಡುತ್ತಾಳೆ"),
    ]
    for root, sentence in ctx:
        form = fst.generate_from_context(root, sentence)
        print(f"  Root: {root:8s}  →  {form}")
        print(f"         Sentence: {sentence}")

    # ── 8.4  Reverse analysis ─────────────────────────────────────────────────
    print("\n── Reverse Analysis ─────────────────────────────────────────────")
    for word in ["ಕುಡಿಯುತ್ತೇನೆ", "ಬರೆಯಲು", "ಮಾಡಲು",
                 "ನೋಡಿದೆನು",   "ಬಂತು",    "ತಿನ್ನಲು"]:
        info = fst.analyze(word)
        if info:
            print(f"  {word:20s}  root={info['root']:8s}  "
                  f"tense={info['tense']:10s}  "
                  f"({info['person']},{info['number']},{info['gender']})")
        else:
            print(f"  {word:20s}  → not analysed")

    print("\nDone.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  IMPROVEMENT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_improvement_tests():
    """Verify the three targeted improvements."""
    fst  = FSTMorphology()
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    passed = 0
    total  = 0

    def check(label, got, expected):
        nonlocal passed, total
        total += 1
        ok = got == expected
        passed += ok
        print(f"  {PASS if ok else FAIL}  {label}")
        if not ok:
            print(f"            expected : {expected}")
            print(f"            got      : {got}")

    # ── Improvement 1: Unicode consistency in IRREGULAR_FORMS ─────────────────
    print("\n" + "=" * 68)
    print("  IMPROVEMENT 1 — Unicode consistency in IRREGULAR_FORMS")
    print("=" * 68)
    wrong_root = "\u0CAC\u0CB0\u0C41"   # ಬರu with independent ಉ (U+0C41)
    right_root = "\u0CAC\u0CB0\u0CC1"   # ಬರು with matra ು   (U+0CC1)
    all_keys = [k[0] for k in SandhiProcessor.IRREGULAR_FORMS if isinstance(k, tuple)]
    has_wrong = any(k == wrong_root for k in all_keys)
    all_right = all(k == right_root for k in all_keys)
    total += 2; passed += (not has_wrong); passed += all_right
    print(f"  {PASS if not has_wrong else FAIL}  No wrong-Unicode key present")
    print(f"  {PASS if all_right else FAIL}  All keys use correct matra ు (U+0CC1)")
    # Functional check: past 2SG must resolve
    check("ಬರು past 2SG returns ಬಂದೆ",
          fst.generate("ಬರು", tense="past", person=2, number="SG", gender="-"),
          "ಬಂದೆ")

    # ── Improvement 2: _guess_root with sandhi reversal ───────────────────────
    print("\n" + "=" * 68)
    print("  IMPROVEMENT 2 — _guess_root sandhi reversal")
    print("=" * 68)
    guess_tests = [
        ("ಕುಡಿಯುತ್ತೇನೆ", "ಕುಡಿ"),
        ("ಬರೆಯುತ್ತಾನೆ",  "ಬರೆ"),
        ("ಕಲಿಯುತ್ತೇನೆ",  "ಕಲಿ"),
        ("ತಿಳಿಯುತ್ತೇನೆ",  "ತಿಳಿ"),
        ("ಮಾಡಲು",         "ಮಾಡು"),   # direct strip, no sandhi reversal needed
        ("ನೋಡಿದೆನು",      "ನೋಡು"),   # past stem strip
    ]
    for word, expected_root in guess_tests:
        got = fst._guess_root(word)
        check(f"_guess_root({word!r}) → {expected_root!r}", got, expected_root)

    # ── Improvement 3: OCR error normalization ────────────────────────────────
    print("\n" + "=" * 68)
    print("  IMPROVEMENT 3 — normalize_ocr_errors()")
    print("=" * 68)
    ocr_tests = [
        # Case 1: spurious ಅ after matra
        ("ಕುಡಿಅಲು",   "ಕುಡಿಯಲು"),  # ಿ+ಅ dropped → ಕುಡಿಲು, then ಿಲು→ಿಯಲು
        # Case 2: missing ತ್ತ doubling
        ("ಮಾಡುತೇನೆ",  "ಮಾಡುತ್ತೇನೆ"),
        ("ಬರೆಯುತಾನೆ", "ಬರೆಯುತ್ತಾನೆ"),
        # Case 3: missing ಯ glide
        ("ಕುಡಿಲು",    "ಕುಡಿಯಲು"),
        ("ಬರೆಲು",     "ಬರೆಯಲು"),
        # No change expected for correct forms
        ("ಕುಡಿಯಲು",   "ಕುಡಿಯಲು"),
        ("ಮಾಡುತ್ತೇನೆ", "ಮಾಡುತ್ತೇನೆ"),
    ]
    for word, expected in ocr_tests:
        check(f"normalize({word!r})", normalize_ocr_errors(word), expected)

    # Integration: analyze() transparently handles OCR errors
    print("\n── Integration: analyze() with OCR-noisy input ──────────────────")
    noisy_analysis_tests = [
        ("ಮಾಡುತೇನೆ",  "ಮಾಡು",  "present", 1, "SG", "-"),
        ("ಕುಡಿಲು",    "ಕುಡಿ",  "infinitive", None, None, None),
    ]
    for noisy, exp_root, exp_tense, ep, en, eg in noisy_analysis_tests:
        info = fst.analyze(noisy)
        ok = bool(info) and info["root"] == exp_root and info["tense"] == exp_tense
        total += 1; passed += ok
        print(f"  {PASS if ok else FAIL}  analyze({noisy!r}) → root={exp_root}, tense={exp_tense}")
        if not ok:
            print(f"            got: {info}")

    print("\n" + "-" * 68)
    print(f"  {passed}/{total} improvement tests passed")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    _run_tests()
    _run_improvement_tests()