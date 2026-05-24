"""
Conjunction Rules Module for Kannada OCR Post-Processing System
===============================================================
Implements formal grammatical rules for Kannada coordinating conjunctions.

Conjunctions handled:
    Additive     : ಮತ್ತು (mattu)   — and
    Emphatic     : ಹಾಗೂ  (haagu)   — and also
    Contrastive  : ಆದರೆ  (aadare)  — but

Rules implemented:
    Rule 1 : Additive Conjunction Redundancy
             Multiple additive conjunctions not permitted in same clause
    Rule 2 : Mutual Exclusivity of Conjunction Types
             Additive and contrastive conjunctions cannot co-occur
    Rule 3 : Clause-Level Coordination Validation
             Valid coordination must follow C1 CONJ C2 structure
"""

import re
from typing import List, Dict, Optional, Tuple


# ============================================================
# CONJUNCTION SETS
# ============================================================

ADDITIVE_CONJUNCTIONS    = {"ಮತ್ತು", "ಹಾಗೂ"}
CONTRASTIVE_CONJUNCTIONS = {"ಆದರೆ"}
ALL_CONJUNCTIONS         = ADDITIVE_CONJUNCTIONS | CONTRASTIVE_CONJUNCTIONS

# OCR variants of conjunctions — common misrecognitions
OCR_CONJUNCTION_VARIANTS = {
    # Additive variants
    "ಮತ್ತು":  ["ಮತ್ಮು", "ಮತ್ತಾ", "ಮತ್ಮಾ", "ಮತ್ಟು"],
    "ಹಾಗೂ": ["ಹಾಗು", "ಹಾಗ", "ಹಾಗಾ"],
    # Contrastive variants
    "ಆದರೆ":   ["ಆದರ", "ಆದŕ", "ಆದć", "ಅದರೆ", "ಆದರ್ಎ"],
}

# Reverse lookup — variant → standard form
VARIANT_TO_STANDARD = {}
for standard, variants in OCR_CONJUNCTION_VARIANTS.items():
    for variant in variants:
        VARIANT_TO_STANDARD[variant] = standard


# ============================================================
# RESULT CLASS
# ============================================================

class ConjunctionCheckResult:
    """
    Stores the result of conjunction rule checking for one sentence.
    """
    def __init__(self):
        self.original_sentence  = ""
        self.violations         = []       # list of dicts
        self.corrected_sentence = ""
        self.correction_made    = False
        self.conjunctions_found = []       # list of (word, type, position)

    def has_violations(self):
        return len(self.violations) > 0

    def __repr__(self):
        return (f"ConjunctionCheckResult("
                f"violations={len(self.violations)}, "
                f"correction_made={self.correction_made})")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_conjunction(word: str) -> Optional[str]:
    """
    Normalize a word to its standard conjunction form.
    Handles OCR variants and direct matches.

    Returns:
        Standard conjunction string if word is a conjunction or variant.
        None if word is not a conjunction.
    """
    clean = word.strip('.,!?;:()[]{}"\'।—-')

    # Direct match
    if clean in ALL_CONJUNCTIONS:
        return clean

    # OCR variant match
    if clean in VARIANT_TO_STANDARD:
        return VARIANT_TO_STANDARD[clean]

    return None


def get_conjunction_type(conjunction: str) -> str:
    """
    Returns conjunction type: ADDITIVE or CONTRASTIVE.
    """
    if conjunction in ADDITIVE_CONJUNCTIONS:
        return "ADDITIVE"
    if conjunction in CONTRASTIVE_CONJUNCTIONS:
        return "CONTRASTIVE"
    return "UNKNOWN"


def extract_conjunctions(sentence: str) -> List[Tuple[str, str, int]]:
    """
    Extract all conjunctions from sentence.

    Returns:
        List of (original_word, standard_form, position_index)
    """
    words   = sentence.split()
    results = []

    for idx, word in enumerate(words):
        standard = normalize_conjunction(word)
        if standard:
            results.append((word, standard, idx))

    return results


def split_into_sentences(text: str) -> List[str]:
    """
    Split paragraph into individual sentences.
    Splits on ., ।, and newlines.
    """
    sentences = re.split(r'[.।\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# RULE CHECKERS
# ============================================================

def check_rule_1(
    conjunctions: List[Tuple[str, str, int]],
    words: List[str]
) -> Optional[Dict]:
    """
    Rule 1 — Additive Conjunction Redundancy.

    Multiple additive conjunctions (ಮತ್ತು, ಹಾಗೂ) are not permitted
    within the same clause.

    If found: keep the last additive conjunction, convert preceding
    coordinated items to comma-separated list.

    Returns:
        Violation dict or None
    """
    additive = [(orig, std, pos) for orig, std, pos in conjunctions
                if std in ADDITIVE_CONJUNCTIONS]

    if len(additive) <= 1:
        return None

    # Violation found
    found_words = [orig for orig, std, pos in additive]
    positions   = [pos  for orig, std, pos in additive]

    # Build correction — keep last conjunction, replace earlier ones with comma
    corrected_words = words.copy()
    for i, (orig, std, pos) in enumerate(additive[:-1]):
        corrected_words[pos] = ","

    # Clean up double spaces and comma issues
    corrected = ' '.join(corrected_words)
    corrected = re.sub(r'\s,', ',', corrected)
    corrected = re.sub(r',\s+,', ',', corrected)
    corrected = re.sub(r'\s+', ' ', corrected).strip()

    return {
        'rule':        'Rule 1',
        'description': f'Multiple additive conjunctions found: {found_words}',
        'detail':      (f'ಮತ್ತು and ಹಾಗೂ cannot both appear in same clause. '
                        f'Found at positions: {positions}'),
        'suggestion':  corrected,
        'severity':    'MEDIUM',
    }


def check_rule_2(
    conjunctions: List[Tuple[str, str, int]],
    words: List[str]
) -> Optional[Dict]:
    """
    Rule 2 — Mutual Exclusivity of Conjunction Types.

    Additive (ಮತ್ತು/ಹಾಗೂ) and contrastive (ಆದರೆ) conjunctions
    cannot co-occur within the same clause boundary.

    If found: remove additive conjunction, keep contrastive.

    Returns:
        Violation dict or None
    """
    additive    = [(orig, std, pos) for orig, std, pos in conjunctions
                   if std in ADDITIVE_CONJUNCTIONS]
    contrastive = [(orig, std, pos) for orig, std, pos in conjunctions
                   if std in CONTRASTIVE_CONJUNCTIONS]

    if not additive or not contrastive:
        return None

    # Violation found
    add_words  = [orig for orig, std, pos in additive]
    cont_words = [orig for orig, std, pos in contrastive]

    # Build correction — remove additive conjunctions
    corrected_words = words.copy()
    for orig, std, pos in additive:
        corrected_words[pos] = ""

    corrected = ' '.join(w for w in corrected_words if w)
    corrected = re.sub(r'\s+', ' ', corrected).strip()

    return {
        'rule':        'Rule 2',
        'description': (f'Mixed conjunction types: '
                        f'additive {add_words} + contrastive {cont_words}'),
        'detail':      (f'Additive conjunctions (ಮತ್ತು/ಹಾಗೂ) and contrastive '
                        f'conjunction (ಆದರೆ) cannot appear in same clause. '
                        f'Keeping ಆದರೆ, removing {add_words}.'),
        'suggestion':  corrected,
        'severity':    'HIGH',
    }


def check_rule_3(
    conjunctions: List[Tuple[str, str, int]],
    words: List[str]
) -> Optional[Dict]:
    """
    Rule 3 — Clause-Level Coordination Validation.

    A valid coordination structure must be:
        Clause1 CONJ Clause2

    Violations:
        - Conjunction at sentence start (nothing before it)
        - Conjunction at sentence end (nothing after it)
        - Two conjunctions adjacent to each other

    Returns:
        Violation dict or None if structure is valid
    """
    if not conjunctions:
        return None

    violations_found = []

    for orig, std, pos in conjunctions:

        # Conjunction at start of sentence
        if pos == 0:
            violations_found.append(
                f"'{orig}' at sentence start — missing Clause1 before conjunction"
            )

        # Conjunction at end of sentence
        elif pos == len(words) - 1:
            violations_found.append(
                f"'{orig}' at sentence end — missing Clause2 after conjunction"
            )

        # Two conjunctions adjacent
        else:
            next_word = words[pos + 1] if pos + 1 < len(words) else ""
            if normalize_conjunction(next_word):
                violations_found.append(
                    f"Two conjunctions adjacent: '{orig}' followed by '{next_word}'"
                )

    if not violations_found:
        return None

    return {
        'rule':        'Rule 3',
        'description': 'Invalid coordination structure detected',
        'detail':      ' | '.join(violations_found),
        'suggestion':  'Ensure sentence follows: Clause1 + Conjunction + Clause2',
        'severity':    'HIGH',
    }


# ============================================================
# OCR VARIANT CORRECTION
# ============================================================

def correct_ocr_conjunction_variants(sentence: str) -> Tuple[str, List[Dict]]:
    """
    Correct OCR misrecognized conjunction variants to standard forms.

    Example:
        ಆದć → ಆದರೆ
        ¤ಾಗೂ → ಹಾಗೂ

    Returns:
        (corrected_sentence, list of corrections made)
    """
    words       = sentence.split()
    corrections = []
    corrected   = words.copy()

    for idx, word in enumerate(words):
        clean    = word.strip('.,!?;:()[]{}"\'।—-')
        standard = VARIANT_TO_STANDARD.get(clean)
        if standard:
            # Preserve surrounding punctuation
            prefix = word[:len(word) - len(word.lstrip('.,!?;:()[]{}"\'।—-'))]
            suffix = word[len(word.rstrip('.,!?;:()[]{}"\'।—-')):]
            corrected[idx] = prefix + standard + suffix
            corrections.append({
                'position': idx,
                'original': word,
                'corrected': corrected[idx],
                'type': 'OCR_CONJUNCTION_FIX'
            })

    corrected_sentence = ' '.join(corrected)
    return corrected_sentence, corrections


# ============================================================
# MAIN CHECKER FUNCTION
# ============================================================

def check_conjunction_rules(sentence: str) -> ConjunctionCheckResult:
    """
    Run all three conjunction rules on a single sentence.

    Args:
        sentence: Kannada sentence string

    Returns:
        ConjunctionCheckResult with violations and corrected sentence
    """
    result = ConjunctionCheckResult()
    result.original_sentence = sentence

    if not sentence or not sentence.strip():
        result.corrected_sentence = sentence
        return result

    # Step 1 — Fix OCR variants first
    working_sentence, ocr_fixes = correct_ocr_conjunction_variants(sentence)
    if ocr_fixes:
        for fix in ocr_fixes:
            result.violations.append({
                'rule':        'OCR Fix',
                'description': f"OCR variant corrected: '{fix['original']}' → '{fix['corrected']}'",
                'detail':      'Conjunction word was misrecognized by OCR',
                'suggestion':  working_sentence,
                'severity':    'LOW',
            })
        result.correction_made    = True
        result.corrected_sentence = working_sentence
    else:
        result.corrected_sentence = sentence

    # Step 2 — Extract conjunctions from (now cleaned) sentence
    words        = working_sentence.split()
    conjunctions = extract_conjunctions(working_sentence)
    result.conjunctions_found = conjunctions

    if not conjunctions:
        return result

    # Step 3 — Run Rule 1
    # Step 3 — Run Rule 3 FIRST (before correction)
    v3 = check_rule_3(conjunctions, words)
    if v3:
        result.violations.append(v3)

# Step 4 — Run Rule 1
    v1 = check_rule_1(conjunctions, words)
    if v1:
        result.violations.append(v1)
        result.corrected_sentence = v1['suggestion']
        result.correction_made = True

        words = result.corrected_sentence.split()
        conjunctions = extract_conjunctions(result.corrected_sentence)

# Step 5 — Run Rule 2
    v2 = check_rule_2(conjunctions, words)
    if v2:
        result.violations.append(v2)
        result.corrected_sentence = v2['suggestion']
        result.correction_made = True

    return result


def check_paragraph_conjunctions(text: str) -> List[ConjunctionCheckResult]:
    """
    Run conjunction checks on every sentence in a paragraph.

    Args:
        text: Full paragraph string

    Returns:
        List of ConjunctionCheckResult — one per sentence
    """
    sentences = split_into_sentences(text)
    results   = []
    for sentence in sentences:
        result = check_conjunction_rules(sentence)
        results.append(result)
    return results


def format_results_for_ui(results: List[ConjunctionCheckResult]) -> str:
    """
    Format conjunction check results for display in Gradio UI.

    Returns:
        Markdown formatted string
    """
    if not results:
        return "No text to check."

    all_violations = []
    for r in results:
        if r.has_violations():
            all_violations.append(r)

    if not all_violations:
        return "✅ Conjunction check complete — No violations found."

    total = sum(len(r.violations) for r in all_violations)
    lines = [f"**Conjunction Check — {total} violation(s) found:**\n"]

    for r in all_violations:
        lines.append(f"\n**Sentence:** {r.original_sentence}")
        for v in r.violations:
            lines.append(f"\n⚠️ **{v['rule']}** [{v['severity']}]")
            lines.append(f"  {v['description']}")
            lines.append(f"  💡 {v['detail']}")
            if v['suggestion'] and v['rule'] != 'Rule 3':
                lines.append(f"  ✏️ **Suggested:** {v['suggestion']}")
        if r.correction_made:
            lines.append(f"\n  ✅ **Corrected sentence:** {r.corrected_sentence}")

    return '\n'.join(lines)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CONJUNCTION RULES — TEST CASES")
    print("=" * 60)

    test_sentences = [
        # Rule 1 — Multiple additive conjunctions
        (
            "ರಾಮ ಮತ್ತು ಸೀತಾ ಹಾಗೂ ಲಕ್ಷ್ಮಣ ಹೋದರು",
            "Rule 1 — Multiple additive conjunctions"
        ),
        # Rule 2 — Mixed additive and contrastive
        (
            "ಅವನು ಬಂದನು ಮತ್ತು ಆದರೆ ಹೋದನು",
            "Rule 2 — Additive and contrastive mixed"
        ),
        # Rule 3 — Conjunction at start
        (
            "ಮತ್ತು ಅವಳು ಶಾಲೆಗೆ ಹೋದಳು",
            "Rule 3 — Conjunction at sentence start"
        ),
        # Rule 3 — Conjunction at end
        (
            "ಅವನು ಊಟ ಮಾಡಿದನು ಆದರೆ",
            "Rule 3 — Conjunction at sentence end"
        ),
        # OCR variant fix
        (
            "ಅವನು ಬಂದನು ಆದć ಹೋದನು",
            "OCR Fix — ಆದć variant of ಆದರೆ"
        ),
        # Rule 1 + Rule 2 combined
        (
            "ರಾಮ ಮತ್ತು ಸೀತಾ ಹಾಗೂ ಆದರೆ ಲಕ್ಷ್ಮಣ ಹೋದರು",
            "Rule 1 + Rule 2 — Multiple additive AND mixed types"
        ),
        # Valid sentence — no violations
        (
            "ರಾಮ ಮತ್ತು ಸೀತಾ ಶಾಲೆಗೆ ಹೋದರು",
            "Valid — single additive conjunction"
        ),
        # Valid contrastive
        (
            "ಅವನು ಬಂದನು ಆದರೆ ಅವಳು ಹೋದಳು",
            "Valid — contrastive conjunction"
        ),
    ]

    passed = 0
    for sentence, description in test_sentences:
        print(f"\n{'─'*60}")
        print(f"Test : {description}")
        print(f"Input: {sentence}")
        result = check_conjunction_rules(sentence)
        if result.has_violations():
            for v in result.violations:
                print(f"  ⚠️  {v['rule']}: {v['description']}")
                if v.get('suggestion') and v['rule'] != 'Rule 3':
                    print(f"  ✏️  Suggested: {v['suggestion']}")
            if result.correction_made:
                print(f"  ✅ Corrected: {result.corrected_sentence}")
        else:
            print(f"  ✅ No violations found")
        passed += 1

    print(f"\n{'='*60}")
    print(f"All {passed} test cases executed successfully")
    print(f"{'='*60}")