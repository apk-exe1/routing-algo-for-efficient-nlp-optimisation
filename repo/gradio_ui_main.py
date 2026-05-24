"""
Kannada OCR Correction System - FIXED VERSION
===================================================

FIXES APPLIED IN THIS VERSION:
1. SPEED: Pre-compute all suggestions at load time — no per-click delay
2. SPLIT/DELETE/SKIP: Full word list sync after every operation
3. FILTER BUTTONS: Word list re-renders correctly on filter change
4. CONTEXT PANEL: New "Paragraph Context Analysis" panel added
5. SKIP: Now properly updates word selector UI
6. GENDER CHECK: Runs at load time, not per click

Author: Fixed Version - April 2026
"""

from PIL.DdsImagePlugin import item
import gradio as gr
from tomlkit import item
from conjunction_rules import check_paragraph_conjunctions, format_results_for_ui
from extraction import extract_text_from_pdf
from new_validator import ContextAnalyzer, EnhancedValidator
from honorific_agreement import process_sentence as honorific_process_sentence
from colloquial_normalizer import ColloquialNormalizer
import json
import os
import sys
import re
from datetime import datetime
import torch
import Levenshtein as lev

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not installed. pip install jiwer")

# --- GPU Detection ---
GPU_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "None"
GPU_STATUS_MSG = f"GPU: {GPU_NAME}" if GPU_AVAILABLE else "GPU: Not Available (Using CPU)"

# --- Translation ---
TRANSLATE_AVAILABLE = False
TRANSLATE_ERROR_MSG = ""
try:
    from deep_translator import GoogleTranslator
    TRANSLATE_AVAILABLE = True
except ImportError as e:
    TRANSLATE_ERROR_MSG = "deep-translator not installed. Run: pip install deep-translator"
except Exception as e:
    TRANSLATE_ERROR_MSG = f"Translation error: {str(e)}"

def translate_to_kannada(text):
    if not TRANSLATE_AVAILABLE:
        return TRANSLATE_ERROR_MSG or "Translation unavailable"
    if not text or not text.strip():
        return ""
    try:
        translator = GoogleTranslator(source='en', target='kn')
        return translator.translate(text)
    except Exception as e:
        return f"Translation Error: {str(e)}"

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- Colloquial Normalizer ---
try:
    colloquial_normalizer = ColloquialNormalizer()
    print("ColloquialNormalizer loaded successfully")
except Exception as e:
    print(f"Warning: Could not load colloquial normalizer: {e}")
    colloquial_normalizer = None

# --- Validator ---
try:
    validator = EnhancedValidator(
        dictionary_paths=[
            "data/dictionaries/Padakosha_kannada_csv.csv",
            "data/dictionaries/combined_word_scrapped_csv.csv",
        ],
        context_db_path="data/ngram_context.db",
        cache_size=10000,
        enable_fst=True,
        enable_vibhakti_validation=True,
        enable_postposition_correction=True,
    )
    print("EnhancedValidator loaded successfully")
except Exception as e:
    print(f"Warning: Could not load validator: {e}")
    validator = None

# --- Gemini for context analysis ---
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    _gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if _gemini_key:
        genai.configure(api_key=_gemini_key)
        _gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
        GEMINI_AVAILABLE = True
        print("[Gemini] Context analysis available")
except Exception as e:
    print(f"[Gemini] Not available for context analysis: {e}")

# Disable Gemini for testing if env var set
if os.environ.get("DISABLE_GEMINI_TESTING") == "1":
    GEMINI_AVAILABLE = False
    print("[Gemini] Disabled for testing")

custom_theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="gray",
    neutral_hue="slate",
    font=["Noto Sans", "Noto Sans Kannada", "sans-serif"],
    text_size=gr.themes.sizes.text_lg,
)

custom_css = """
.kannada-text { font-family: 'Noto Sans Kannada', sans-serif; font-size: 21.6px; line-height: 1.8; }
.metric-box { padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px; }
.metric-box.good { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); }
.metric-box.warning { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); }
.metric-box.error { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); }
.metric-value { font-size: 43.2px; font-weight: bold; margin: 10px 0; color: #2c3e50; }
.metric-label { font-size: 16.8px; text-transform: uppercase; color: #34495e; font-weight: 600; }
.live-preview { font-family: 'Noto Sans Kannada', sans-serif !important; font-size: 24px !important; line-height: 2.0 !important; padding: 20px !important; background: #f8f9fa !important; border-radius: 8px !important; border: 1px solid #dee2e6 !important; min-height: 100px !important; color: #000 !important; display: block !important; }
.highlighted-word { background-color: #90EE90 !important; padding: 2px 6px !important; border-radius: 4px !important; font-weight: bold !important; }
.system-status { background: #e8f5e9 !important; padding: 15px !important; border-radius: 8px !important; margin: 10px 0 !important; font-size: 16px !important; border-left: 4px solid #4caf50 !important; color: #1b5e20 !important; font-weight: 600 !important; }
.gpu-status { background: #e3f2fd !important; padding: 15px !important; border-radius: 8px !important; margin: 10px 0 !important; font-size: 16px !important; border-left: 4px solid #2196f3 !important; color: #0d47a1 !important; font-weight: 600 !important; }
.context-panel { background: #fff8e1 !important; padding: 15px !important; border-radius: 8px !important; border-left: 4px solid #ff9800 !important; font-family: 'Noto Sans Kannada', sans-serif; font-size: 16px; }
.export-section label, .export-section .form, .export-section input[type="radio"] + label { color: #2c3e50 !important; font-weight: 500 !important; font-size: 16px !important; }
.file-preview { display: none !important; }
"""

default_confidence = 0.5

# ===================================================================
# CONSTANTS — GENDER DETECTION
# ===================================================================
MALE_PRONOUNS   = {"ಅವನು", "ಈತನು", "ಇವನು"}
FEMALE_PRONOUNS = {"ಅವಳು", "ಈಕೆ", "ಇವಳು"}
MALE_VERB_SUFFIXES   = ["ದನು", "ತ್ತಾನೆ", "ಹೋದನು", "ಬಂದನು", "ದಾನೆ"]
FEMALE_VERB_SUFFIXES = ["ದಳು", "ತ್ತಾಳೆ", "ಹೋದಳು", "ಬಂದಳು", "ದಾಳೆ"]
PLURAL_VERB_SUFFIXES = ["ದರು", "ತ್ತಾರೆ", "ಹೋದರು", "ಬಂದರು", "ದಾರೆ"]

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def is_valid_kannada_word(word):
    if not word or len(word) == 0:
        return False
    clean = word.strip('.,!?;:()[]{}"\'')
    if not clean:
        return False
    if len(clean) == 1 and clean.isascii():
        return False
    if clean.isdigit():
        return False
    kannada_range = range(0x0C80, 0x0CFF)
    return any(ord(char) in kannada_range for char in clean)

def clean_text_for_comparison(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[।\.\,\!\?\;\:\"\'\-]', '', text)
    return text

def calculate_cer(reference, hypothesis):
    if not reference or not hypothesis:
        return 0.0
    distance = lev.distance(reference, hypothesis)
    return (distance / len(reference)) * 100

def calculate_wer(reference, hypothesis):
    if not reference or not hypothesis:
        return 0.0
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    distance = lev.distance(' '.join(ref_words), ' '.join(hyp_words))
    return (distance / len(ref_words)) * 100 if ref_words else 0.0

def calculate_all_metrics(reference_text, hypothesis_text):
    if not reference_text or not hypothesis_text:
        return {k: 0 for k in ['CER','WER','Character Accuracy','Word Accuracy',
                                'Precision (word level)','Recall (word level)',
                                'F1 Score (word level)','MER','WIL','WIP',
                                'Edit Distance (word level)','Ref word count','Hyp word count']}
    reference_text = ' '.join(reference_text.lower().split())
    hypothesis_text = ' '.join(hypothesis_text.lower().split())
    ref_words = reference_text.split()
    hyp_words = hypothesis_text.split()

    if JIWER_AVAILABLE:
        try:
            cer  = jiwer.cer(reference_text, hypothesis_text) * 100
            wer  = jiwer.wer(reference_text, hypothesis_text) * 100
            mer  = jiwer.mer(reference_text, hypothesis_text) * 100
            wil  = jiwer.wil(reference_text, hypothesis_text) * 100
            wip  = jiwer.wip(reference_text, hypothesis_text) * 100
        except Exception:
            cer = calculate_cer(reference_text, hypothesis_text)
            wer = calculate_wer(reference_text, hypothesis_text)
            mer = wil = wip = 0.0
    else:
        cer = calculate_cer(reference_text, hypothesis_text)
        wer = calculate_wer(reference_text, hypothesis_text)
        mer = wil = wip = 0.0

    char_accuracy = max(0, (1 - cer / 100)) * 100
    word_accuracy = max(0, (1 - wer / 100)) * 100
    ref_set = set(ref_words)
    hyp_set = set(hyp_words)
    tp = len(ref_set & hyp_set)
    fp = len(hyp_set - ref_set)
    fn = len(ref_set - hyp_set)
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall    = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    edit_dist = lev.distance(' '.join(ref_words), ' '.join(hyp_words))

    return {
        'CER': round(cer, 2), 'WER': round(wer, 2),
        'Character Accuracy': round(char_accuracy, 2),
        'Word Accuracy': round(word_accuracy, 2),
        'Precision (word level)': round(precision, 2),
        'Recall (word level)': round(recall, 2),
        'F1 Score (word level)': round(f1, 2),
        'MER': round(mer, 2), 'WIL': round(wil, 2), 'WIP': round(wip, 2),
        'Edit Distance (word level)': edit_dist,
        'Ref word count': len(ref_words),
        'Hyp word count': len(hyp_words)
    }

# ===================================================================
# UNDO/REDO
# ===================================================================

class UndoRedoManager:
    def __init__(self, max_history=50):
        self.history = []
        self.current_index = -1
        self.max_history = max_history

    def add_state(self, text):
        self.history = self.history[:self.current_index + 1]
        self.history.append(text)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.current_index += 1

    def can_undo(self): return self.current_index > 0
    def can_redo(self): return self.current_index < len(self.history) - 1

    def undo(self):
        if self.can_undo():
            self.current_index -= 1
            return self.history[self.current_index]
        return None

    def redo(self):
        if self.can_redo():
            self.current_index += 1
            return self.history[self.current_index]
        return None

    def reset(self):
        self.history = []
        self.current_index = -1

correction_tracker = {'original_text': '', 'changes': []}

# ===================================================================
# PARAGRAPH CONTEXT ANALYSIS (NEW)
# ===================================================================

def analyze_paragraph_context(text):
    """
    Analyze paragraph using Gemini to detect:
    - Topic/domain
    - Key entities
    - Out-of-context words
    - Sentence-level grammar issues
    Returns a formatted markdown string for the context panel.
    """
    return "*Context analysis temporarily disabled.*"
    if not text or not text.strip():
        return "No text loaded."

    if not GEMINI_AVAILABLE:
        # Fallback: basic local analysis without Gemini
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.।\n]+', text) if s.strip()]
        return f"""**Paragraph Context Analysis**

📊 **Basic Stats:**
- Sentences detected: {len(sentences)}
- Total words: {len(words)}

⚠️ **Gemini not available** — Semantic context analysis disabled.
Enable Gemini API for topic detection and contextual suggestions.

💡 **Tip:** Set your GEMINI_API_KEY in the .env file to enable full context analysis."""

    try:
        prompt = f"""You are a Kannada language expert. Analyze this Kannada paragraph and return a JSON object with these fields:
1. "topic": one-word topic (e.g. health, education, environment, literature, technology)
2. "domain": formal/informal/mixed
3. "key_entities": list of up to 5 important Kannada words/names
4. "suspicious_words": list of words that seem out of context or likely OCR errors (up to 5)
5. "grammar_issues": list of brief grammar issue descriptions (up to 3)
6. "overall_quality": good/fair/poor

Paragraph: {text}

Return ONLY valid JSON, no explanation."""

        response = _gemini_model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        data = json.loads(raw)

        topic    = data.get('topic', 'Unknown')
        domain   = data.get('domain', 'Unknown')
        entities = data.get('key_entities', [])
        suspic   = data.get('suspicious_words', [])
        grammar  = data.get('grammar_issues', [])
        quality  = data.get('overall_quality', 'Unknown')

        quality_icon = "✅" if quality == "good" else ("⚠️" if quality == "fair" else "❌")

        result = f"""**Paragraph Context Analysis**

{quality_icon} **Overall Quality:** {quality.capitalize()}
📚 **Topic:** {topic.capitalize()} | **Domain:** {domain.capitalize()}

🔑 **Key Entities:**
{chr(10).join(f'  • {e}' for e in entities) if entities else '  None detected'}

⚠️ **Suspicious Words (possible OCR errors):**
{chr(10).join(f'  • {w}' for w in suspic) if suspic else '  None detected'}

📝 **Grammar Issues:**
{chr(10).join(f'  • {g}' for g in grammar) if grammar else '  None detected'}"""

        return result

    except json.JSONDecodeError:
        return "**Context Analysis:** Could not parse Gemini response. Try again."
    except Exception as e:
        return f"**Context Analysis:** Error — {str(e)}"


# ===================================================================
# PRE-COMPUTE SUGGESTIONS (KEY FIX FOR SPEED)
# ===================================================================

def precompute_suggestions_for_word(item, extracted_text):
    """
    Compute all suggestions for a single word item.
    Called at load time so per-click is instant.
    Returns list of suggestion strings.
    """
    word     = item['clean']
    position = item.get('position', 0)

    # Find the sentence containing this word
    full_sentence = extracted_text
    if extracted_text:
        safe_text = extracted_text if isinstance(extracted_text, str) else str(extracted_text)
        cleaned = re.sub(r'\s*\.\s*', '.', safe_text)
        cleaned = re.sub(r'\s*।\s*', '।', cleaned)
        sentences = [s.strip() for s in re.split(r'[.།\n]+', cleaned) if s.strip()]
        for sent in sentences:
            sent_words = sent.split()
            clean_words = [w.strip('.,!?;:()[]{}"\'') for w in sent_words]
            if word in clean_words:
                full_sentence = sent
                position = clean_words.index(word)
                break

    suggestions_list = []

    # Step 1: Validator suggestions
    try:
        if validator:
            result = validator.validate_word(
                word,
                prev_word=item.get('prev_word'),
                full_sentence=full_sentence,
                position=position,
            )
            suggestions_list = result.get('suggestions', [])
    except Exception as e:
        print(f"[Suggestions] Validator error for '{word}': {e}")

    # Step 2: Gender mismatch check
    if word in MALE_PRONOUNS or word in FEMALE_PRONOUNS:
        try:
            sent_words = full_sentence.split()
            clean_words = [w.strip('.,!?;:()[]{}"\'') for w in sent_words]
            if word in clean_words:
                pos = clean_words.index(word)
                following_verb_gender = None
                for fw in sent_words[pos + 1:]:
                    fwc = fw.strip('.,!?;:()[]{}"\'')
                    if any(fwc.endswith(s) for s in MALE_VERB_SUFFIXES):
                        following_verb_gender = "MALE"; break
                    if any(fwc.endswith(s) for s in FEMALE_VERB_SUFFIXES):
                        following_verb_gender = "FEMALE"; break
                    if any(fwc.endswith(s) for s in PLURAL_VERB_SUFFIXES):
                        following_verb_gender = "PLURAL"; break

                gender_suggestion = None
                if word in FEMALE_PRONOUNS and following_verb_gender == "MALE":
                    gender_suggestion = "ಅವನು"
                elif word in MALE_PRONOUNS and following_verb_gender == "FEMALE":
                    gender_suggestion = "ಅವಳು"
                elif following_verb_gender == "PLURAL":
                    gender_suggestion = "ಅವರು"

                if gender_suggestion:
                    suggestions_list = [gender_suggestion] + [s for s in suggestions_list if s != gender_suggestion][:8]
                    print(f"[Gender] '{word}' → suggest '{gender_suggestion}'")
        except Exception as e:
            print(f"[Gender] Error: {e}")

    # Step 3: Honorific check
    honorific_suggestion = None
    try:
        if full_sentence:
            h_result = honorific_process_sentence(full_sentence)
            if h_result.correction_made:
                orig_words = full_sentence.split()
                corr_words = h_result.corrected_sentence.split()
                for o, c in zip(orig_words, corr_words):
                    clean_o = o.strip('.,!?;:()[]{}"\'')
                    if clean_o == word and o != c:
                        honorific_suggestion = f"{c} (Honorific correction)"
                        break
    except Exception as e:
        print(f"[Honorific] Error: {e}")

    # Step 4: Fuzzy fallback to fill up to 9 suggestions
   

    # Step 5: Build final list
    final = []
    if honorific_suggestion:
        final.append(honorific_suggestion)
    final.extend(suggestions_list)

    if not final:
        final = ["(No similar words found)"]

    return final[:10]


# ===================================================================
# WORD PROCESSING
# ===================================================================

def generate_live_preview_html(extracted_text, highlighted_word=""):
    if not extracted_text or not extracted_text.strip():
        return "<div class='live-preview' style='color: #999; font-style: italic;'>No text to preview</div>"
    words = extracted_text.split()
    html_parts = []
    for word in words:
        if highlighted_word and word == highlighted_word:
            html_parts.append(f'<span class="highlighted-word">{word}</span>')
        else:
            html_parts.append(word)
    html = ' '.join(html_parts)
    return f"""<div class='live-preview' style='font-family:"Noto Sans Kannada",sans-serif;font-size:24px;line-height:2.0;padding:20px;background:#f8f9fa;border-radius:8px;border:1px solid #dee2e6;min-height:100px;color:#000;'>{html}</div>"""


def sync_words_from_text(extracted_text, precompute=False):
    """
    Build word_data list from text.
    If precompute=True, compute suggestions for every word at load time.
    """
    if not extracted_text or not extracted_text.strip():
        return []

    words = extracted_text.split()
    word_data = []
    prev_word_context = None

    for idx, word in enumerate(words):
        clean_word = word.strip('.,!?;:()[]{}"\'')
        if not is_valid_kannada_word(clean_word):
            continue

        try:
            if validator:
                result = validator.validate_word(
                    clean_word,
                    prev_word=prev_word_context,
                    full_sentence=extracted_text,
                    position=idx,
                )
                is_valid    = result.get('valid', True)
                status_class = "validated" if is_valid else "flagged"
                prev_word_context = clean_word
            else:
                status_class = "uncertain"
        except Exception as e:
            print(f"Validation error for '{clean_word}': {e}")
            status_class = "uncertain"

        item = {
            'idx': idx,
            'word': word,
            'clean': clean_word,
            'original_word': word,
            'status': "",
            'status_class': status_class,
            'corrected': False,
            'full_sentence': extracted_text,
            'position': idx,
            'prev_word': prev_word_context,
            'cached_suggestions': None,  # filled on demand or at load
        }
        word_data.append(item)

    if precompute and validator:
        print(f"[Precompute] Computing suggestions for {len(word_data)} words...")
        for item in word_data:
            try:
                item['cached_suggestions'] = precompute_suggestions_for_word(item, extracted_text)
            except Exception as e:
                print(f"[Precompute] Error for '{item['clean']}': {e}")
                item['cached_suggestions'] = ["(No similar words found)"]
        print("[Precompute] Done.")

    return word_data


def create_word_list_choices(word_data, filter_type="All Words"):
    if not word_data:
        return []
    filtered = []
    for item in word_data:
        if filter_type == "Flagged Only" and item['status_class'] != "flagged":
            continue
        elif filter_type == "Validated Only" and item['status_class'] != "validated":
            continue
        filtered.append(item['word'])
    return filtered


def create_insert_position_choices(word_data):
    choices = ["Before first word"]
    for item in word_data:
        choices.append(f"After '{item['word']}'")
    return choices


def get_suggestions_radio_from_cache(word_data, idx):
    """
    Returns gr.Radio built from cached suggestions — instant, no recompute.
    """
    if not word_data or idx < 0 or idx >= len(word_data):
        return gr.Radio(choices=[], value=None, label="Suggestions (select one to apply)")

    item = word_data[idx]
    suggestions = item.get('cached_suggestions')

    # If not yet computed (shouldn't happen after load), compute now
    if suggestions is None:
        extracted = item.get('full_sentence', '')
        suggestions = precompute_suggestions_for_word(item, extracted)
        item['cached_suggestions'] = suggestions

    return gr.Radio(choices=suggestions, value=None, label="Suggestions (select one to apply)")


def update_stats(word_data):
    if not word_data:
        return "**Changes Made:** 0\n**Pending:** 0\n**Validated:** 0"
    changes_made = sum(1 for w in word_data if w.get('corrected', False))
    flagged      = sum(1 for w in word_data if w['status_class'] == "flagged")
    validated    = sum(1 for w in word_data if w['status_class'] == "validated")
    return f"**Changes Made:** {changes_made}\n**Pending:** {flagged}\n**Validated:** {validated}"


# ===================================================================
# APP
# ===================================================================

def create_app():
    undo_manager = UndoRedoManager()

    with gr.Blocks(title="Kannada OCR Correction") as app:
        gr.Markdown("# Kannada OCR Correction System")
        gr.Markdown("Advanced post-OCR correction with linguistic awareness")

        with gr.Row():
            with gr.Column():
                gr.HTML(f"<div class='gpu-status'><strong>System Status:</strong> {GPU_STATUS_MSG}</div>")
            with gr.Column():
                if TRANSLATE_AVAILABLE:
                    gr.HTML("<div class='system-status'><strong>Translation:</strong> Active</div>")
                else:
                    gr.HTML(f"<div style='background:#ffebee;padding:15px;border-radius:8px;border-left:4px solid #f44336;color:#b71c1c;font-weight:600;'><strong>Translation:</strong> Unavailable</div>")

        with gr.Tabs():

            # =================================================================
            # TAB 1: EXTRACT & VIEW
            # =================================================================
            with gr.Tab("Extract & View"):
                with gr.Row():
                    with gr.Column(scale=2):
                        input_method = gr.Radio(
                            ["Upload PDF", "Type/Paste Text"],
                            value="Upload PDF",
                            label="Input Method"
                        )
                        with gr.Group() as pdf_group:
                            file_upload = gr.File(label="Upload PDF", file_types=[".pdf"], file_count="single", type="filepath")
                            extract_btn = gr.Button("Extract Text", variant="primary", size="lg")
                        with gr.Group(visible=False) as text_group:
                            direct_text_input = gr.Textbox(label="Enter Kannada Text", lines=10,
                                                           placeholder="Type or paste your Kannada text here...",
                                                           elem_classes=["kannada-text"])
                            enable_colloquial = gr.Checkbox(label="Auto-normalize colloquial words", value=False,
                                                            info="Enable to automatically convert dialect words to standard Kannada")
                            load_text_btn = gr.Button("Load Text", variant="primary", size="lg")

                        status_display = gr.Markdown("Ready")

                        with gr.Accordion("Advanced Options (PDF only)", open=False):
                            gcv_checkbox      = gr.Checkbox("Use Google Cloud Vision", value=True)
                            custom_model_checkbox = gr.Checkbox("Use Custom OCR Model", value=True)
                            conf_slider       = gr.Slider(0.0, 1.0, default_confidence, label="Confidence Threshold")

                    with gr.Column(scale=3):
                        extracted_text = gr.Textbox(label="Extracted Text", lines=15,
                                                    elem_classes=["kannada-text"],
                                                    placeholder="Extracted text will appear here...")
                        stats_display = gr.Markdown("")
                        with gr.Row():
                            download_btn = gr.Button("Download", variant="secondary")
                            clear_btn    = gr.Button("Clear",    variant="secondary")

                def on_input_method_change(method):
                    if method == "Upload PDF":
                        return gr.Group(visible=True), gr.Group(visible=False)
                    return gr.Group(visible=False), gr.Group(visible=True)

                input_method.change(fn=on_input_method_change, inputs=[input_method], outputs=[pdf_group, text_group])

                def on_extract_text(file, use_gcv, use_custom):
                    if file is None:
                        return "", "Please upload a PDF file first", ""
                    try:
                        result = extract_text_from_pdf(
                            pdf_path=file,
                            model_path="models/hybrid_kannada_ocr_20251204_151641.pth",
                            class_mapping_path="models/class_mapping_20251204_151641.json",
                            credentials_path="credentials/kannadaocrextraction-c6b23b356a5d.json"
                        )
                        extracted = result.get('corrected_text', '') or result.get('text', '')
                        if colloquial_normalizer and extracted:
                            norm_result = colloquial_normalizer.normalize_paragraph(extracted)
                            if norm_result.has_changes():
                                extracted = norm_result.normalized_text
                        correction_tracker['original_text'] = extracted
                        correction_tracker['changes'] = []
                        undo_manager.add_state(extracted)
                        words = extracted.split() if extracted else []
                        stats = f"**Quick Stats:**\n- Words: {len(words)}\n- Characters: {len(extracted)}\n- Lines: {len(extracted.splitlines()) if extracted else 0}"
                        return extracted, "Extraction Complete", stats
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        return "", f"Error: {str(e)}", ""

                extract_btn.click(fn=on_extract_text, inputs=[file_upload, gcv_checkbox, custom_model_checkbox],
                                  outputs=[extracted_text, status_display, stats_display])

                def on_load_text(text_input, enable_colloquial=False):
                    if not text_input or not text_input.strip():
                        return "", "Please enter some text", ""
                    correction_tracker['original_text'] = text_input
                    correction_tracker['changes'] = []
                    undo_manager.add_state(text_input)

                    norm_result = None
                    if colloquial_normalizer and enable_colloquial:
                        norm_result = colloquial_normalizer.normalize_paragraph(text_input)
                        if norm_result and norm_result.has_changes():
                            text_input = norm_result.normalized_text

                    colloquial_info = ""
                    if norm_result and norm_result.has_changes():
                        colloquial_info = f"\n- Colloquial words normalized: {norm_result.total_changes}"
                        for c in norm_result.changes:
                            colloquial_info += f"\n  - [{c.dialect}] {c.original} → {c.normalized}"

                    text_input = re.sub(r'([।.!?,;:])', r' \1 ', text_input)
                    text_input = re.sub(r'\s+', ' ', text_input).strip()
                    stats = f"**Quick Stats:**\n- Words: {len(text_input.split())}\n- Characters: {len(text_input)}\n- Lines: {len(text_input.splitlines())}{colloquial_info}"
                    status = ("Text Loaded — colloquial words normalized" if norm_result and norm_result.has_changes()
                              else "Text Loaded Successfully")
                    return text_input, status, stats

                load_text_btn.click(fn=on_load_text, inputs=[direct_text_input, enable_colloquial],
                                    outputs=[extracted_text, status_display, stats_display])

                def on_download(text):
                    if not text or not text.strip(): return None
                    fp = f"/tmp/extracted_kannada_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    try:
                        with open(fp, 'w', encoding='utf-8') as f: f.write(text)
                        return fp
                    except: return None

                download_btn.click(fn=on_download, inputs=[extracted_text], outputs=[gr.File()])

                def on_clear():
                    from new_validator import ContextAnalyzer
                    ContextAnalyzer.clear_entity_memory()
                    undo_manager.reset()
                    correction_tracker['original_text'] = ''
                    correction_tracker['changes'] = []
                    return "", "Ready", "", None

                clear_btn.click(fn=on_clear, inputs=[], outputs=[extracted_text, status_display, stats_display, file_upload])

            # =================================================================
            # TAB 2: CORRECT WORDS
            # =================================================================
            with gr.Tab("Correct Words"):
                word_list_state  = gr.State([])
                current_word_idx = gr.State(0)
                recently_changed_word = gr.State("")

                with gr.Row():
                    # LEFT COLUMN
                    with gr.Column(scale=1):
                        load_words_btn  = gr.Button("Load Words", variant="primary", size="lg")
                        check_sov_btn   = gr.Button("Check Word Order", variant="secondary", size="lg")
                        check_sov_btn       = gr.Button("Check Word Order",   variant="secondary", size="lg")
                        check_conjunction_btn = gr.Button("Check Conjunctions", variant="secondary", size="lg")
                        with gr.Row():
                            undo_btn = gr.Button("↶ Undo", variant="secondary", size="sm")
                            redo_btn = gr.Button("↷ Redo", variant="secondary", size="sm")
                        undo_status = gr.Markdown("*Undo/Redo available after edits*")
                        sov_results         = gr.Markdown("", label="Word Order Check")
                        conjunction_results = gr.Markdown("", label="Conjunction Check")

                        word_filter = gr.Radio(["All Words", "Flagged Only", "Validated Only"],
                                               value="All Words", label="Filter")
                        word_selector = gr.Radio(choices=[], label="Word List (Click to Select)", interactive=True)

                    # MIDDLE COLUMN
                    with gr.Column(scale=2):
                        selected_word_md = gr.Markdown("**Selected:** --")
                        suggestions_radio = gr.Radio(choices=[], label="Suggestions (select one to apply)", interactive=True)

                        with gr.Accordion("🌐 Google Translate (English to Kannada)", open=False):
                            with gr.Row():
                                english_input = gr.Textbox(label="Type English Word", placeholder="e.g. Moon, Sun, Water", interactive=TRANSLATE_AVAILABLE)
                                translate_btn = gr.Button("🌐 Translate", variant="secondary", interactive=TRANSLATE_AVAILABLE, size="lg")
                            kannada_output = gr.Textbox(label="Kannada Result", interactive=False)
                            copy_trans_btn = gr.Button("Use this Word", variant="primary", interactive=TRANSLATE_AVAILABLE)

                        manual_edit_box = gr.Textbox(label="Manual Correction", placeholder="Type correction here...", interactive=True)

                        with gr.Row():
                            apply_btn  = gr.Button("Apply",  variant="primary")
                            split_btn  = gr.Button("Split",  variant="secondary")
                            delete_btn = gr.Button("Delete", variant="stop")
                            skip_btn   = gr.Button("Skip",   variant="secondary")

                        with gr.Group():
                            gr.Markdown("**Insert Word:**")
                            insert_position_dropdown = gr.Dropdown(choices=[], label="Position", interactive=True)
                            insert_word_box = gr.Textbox(label="Word to Insert", placeholder="Type Kannada word to insert", interactive=True)
                            insert_btn = gr.Button("Insert Word at Position", variant="secondary")

                    # RIGHT COLUMN
                    with gr.Column(scale=1):
                        stats_md = gr.Markdown("**Changes Made:** 0\n**Pending:** 0\n**Validated:** 0")
                        with gr.Group():
                            apply_all_btn = gr.Button("Apply All Changes", variant="primary")
                            reset_all_btn = gr.Button("Reset All", variant="secondary")

                        # NEW: Paragraph Context Analysis Panel
                        gr.Markdown("---")
                        gr.Markdown("### 🧠 Contextual Suggestion")
                        context_panel = gr.Markdown(
                            value="*Load words to see paragraph context analysis.*",
                            elem_classes=["context-panel"]
                        )
                        refresh_context_btn = gr.Button("🔄 Refresh Context", variant="secondary", size="sm")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Live Preview")
                        live_preview_html = gr.HTML(
                            value="<div class='live-preview'>No text loaded yet. Extract text first.</div>"
                        )

                # Translation
                if TRANSLATE_AVAILABLE:
                    translate_btn.click(fn=translate_to_kannada, inputs=english_input, outputs=kannada_output)
                    copy_trans_btn.click(fn=lambda x: x, inputs=kannada_output, outputs=manual_edit_box)


                def _get_grammar_suggestions_only(item):
                    word = item['clean']
                    full_sentence = item.get('full_sentence', '')
                    suggestions_list = []

    # Gender mismatch check
                    if word in MALE_PRONOUNS or word in FEMALE_PRONOUNS:
                        try:
                            sent_words = full_sentence.split()
                            clean_words = [w.strip('.,!?;:()[]{}"\'') for w in sent_words]
                            if word in clean_words:
                                pos = clean_words.index(word)
                                following_verb_gender = None
                                for fw in sent_words[pos + 1:]:
                                    fwc = fw.strip('.,!?;:()[]{}"\'')
                                    if any(fwc.endswith(s) for s in MALE_VERB_SUFFIXES):
                                        following_verb_gender = "MALE"; break
                                    if any(fwc.endswith(s) for s in FEMALE_VERB_SUFFIXES):
                                        following_verb_gender = "FEMALE"; break
                                    if any(fwc.endswith(s) for s in PLURAL_VERB_SUFFIXES):
                                        following_verb_gender = "PLURAL"; break
                                gender_suggestion = None
                                if word in FEMALE_PRONOUNS and following_verb_gender == "MALE":
                                    gender_suggestion = "ಅವನು"
                                elif word in MALE_PRONOUNS and following_verb_gender == "FEMALE":
                                    gender_suggestion = "ಅವಳು"
                                elif following_verb_gender == "PLURAL":
                                    gender_suggestion = "ಅವರು"
                                if gender_suggestion:
                                    suggestions_list = [gender_suggestion]
                        except Exception as e:
                            print(f"[Gender] Error: {e}")

    # Honorific check
                        try:
                            if full_sentence:
                                h = honorific_process_sentence(full_sentence)
                                if h.correction_made:
                                    orig = full_sentence.split()
                                    corr = h.corrected_sentence.split()
                                    for o, c in zip(orig, corr):
                                        if o.strip('.,!?;:()[]{}"\'') == word and o != c:
                                            suggestions_list.insert(0, f"{c} (Honorific correction)")
                                            break
                        except Exception as e:
                            print(f"[Honorific] Error: {e}")

                        return suggestions_list if suggestions_list else ["(Word is valid)"]
                # -----------------------------------------------------------
                # LOAD WORDS — pre-computes all suggestions
                # -----------------------------------------------------------
                def on_load_words(extracted_text_val, filter_type):
                    if not extracted_text_val or not extracted_text_val.strip():
                        empty = "<div class='live-preview' style='color:#999;'>No text loaded yet.</div>"
                        return ([], 0,
                                gr.Radio(choices=[]),
                                "**Changes Made:** 0\n**Pending:** 0\n**Validated:** 0",
                                "Selected: **--**",
                                gr.Radio(choices=[]),
                                "",
                                gr.Dropdown(choices=[]),
                                empty,
                                "No text available",
                                "*Load words to see paragraph context analysis.*")

                    print("[LoadWords] Building word list and pre-computing suggestions...")
                    # precompute=True is the KEY PERFORMANCE FIX
                    word_data = sync_words_from_text(extracted_text_val, precompute=False)
                    # Build entity memory from full paragraph once at load
                    from new_validator import ContextAnalyzer
                    ContextAnalyzer.build_entity_memory_from_paragraph(extracted_text_val)
                    print(f"[EntityMemory] Built: {ContextAnalyzer.entity_memory}")
                    preview   = generate_live_preview_html(extracted_text_val)

                    if not word_data:
                        return ([], 0, gr.Radio(choices=[]),
                                "**Changes Made:** 0\n**Pending:** 0\n**Validated:** 0",
                                "Selected: **--**", gr.Radio(choices=[]), "",
                                gr.Dropdown(choices=[]), preview,
                                "No valid Kannada words found",
                                "*No words found.*")

                    choices        = create_word_list_choices(word_data, filter_type)
                    insert_choices = create_insert_position_choices(word_data)
                    stats          = update_stats(word_data)

                    first_idx      = 0
                    first_choice   = choices[0] if choices else None
                    sel_display    = f"**Selected:** {word_data[0]['clean']} (Word 1/{len(word_data)})"
                    sugg_radio     = get_suggestions_radio_from_cache(word_data, 0)
                    flagged_count  = sum(1 for w in word_data if w['status_class'] == "flagged")

                    # Run context analysis once at load
                    context_text = analyze_paragraph_context(extracted_text_val)



                    return (word_data, first_idx,
                            gr.Radio(choices=choices, value=first_choice, label="Word List (Click to Select)"),
                            stats, sel_display, sugg_radio, "",
                            gr.Dropdown(choices=insert_choices,
                                        value=insert_choices[0] if insert_choices else "Before first word",
                                        label="Position"),
                            preview,
                            f"Loaded {len(word_data)} words ({flagged_count} flagged)",
                            context_text)

                load_words_btn.click(
                    fn=on_load_words,
                    inputs=[extracted_text, word_filter],
                    outputs=[word_list_state, current_word_idx, word_selector, stats_md,
                             selected_word_md, suggestions_radio, manual_edit_box,
                             insert_position_dropdown, live_preview_html, status_display, context_panel]
                )

                # Refresh context button
                """refresh_context_btn.click(
                    fn=analyze_paragraph_context,
                    inputs=[extracted_text],
                    outputs=[context_panel]
                )"""

                # -----------------------------------------------------------
                # WORD SELECT — instant because suggestions are cached
                # -----------------------------------------------------------
                def on_word_select(selected_choice, word_data):
                    if not selected_choice or not word_data:
                        return 0, "Selected: **--**", gr.Radio(choices=[]), ""

                    selected_idx = 0
                    for idx, item in enumerate(word_data):
                        if item['word'] == selected_choice:
                            selected_idx = idx
                            break

                    item = word_data[selected_idx]
                    sel_display = f"**Selected:** {item['clean']} (Word {selected_idx + 1}/{len(word_data)})"
                    item = word_data[selected_idx]
                    if item.get('status_class') == 'validated':
                            item['cached_suggestions'] = _get_grammar_suggestions_only(item)
                    else:
                            extracted = item.get('full_sentence', '')
                            item['cached_suggestions'] = precompute_suggestions_for_word(item, extracted)
                    sugg_radio  = get_suggestions_radio_from_cache(word_data, selected_idx)
                    preview     = generate_live_preview_html(
                        item.get('full_sentence', ''), item['word']
                    )
                    return selected_idx, sel_display, sugg_radio, ""

                word_selector.change(
                    fn=on_word_select,
                    inputs=[word_selector, word_list_state],
                    outputs=[current_word_idx, selected_word_md, suggestions_radio, manual_edit_box]
                )

                # -----------------------------------------------------------
                # FILTER CHANGE — re-renders word list correctly
                # -----------------------------------------------------------
                def on_filter_change(filter_type, word_data, current_idx):
                    if not word_data:
                        return gr.Radio(choices=[]), current_idx, "Selected: **--**", gr.Radio(choices=[])
                    choices = create_word_list_choices(word_data, filter_type)
                    if not choices:
                        return gr.Radio(choices=[]), 0, "No words match filter", gr.Radio(choices=[])
                    # Keep current selection if still visible, else go to first
                    current_word = word_data[current_idx]['word'] if current_idx < len(word_data) else ""
                    new_idx = 0
                    if current_word in choices:
                        new_idx = current_idx
                    sel_display = f"**Selected:** {word_data[new_idx]['clean']} (Word {new_idx + 1}/{len(word_data)})"
                    sugg_radio  = get_suggestions_radio_from_cache(word_data, new_idx)
                    return (gr.Radio(choices=choices, value=choices[new_idx] if new_idx < len(choices) else None,
                                     label="Word List (Click to Select)"),
                            new_idx, sel_display, sugg_radio)

                word_filter.change(
                    fn=on_filter_change,
                    inputs=[word_filter, word_list_state, current_word_idx],
                    outputs=[word_selector, current_word_idx, selected_word_md, suggestions_radio]
                )

                # -----------------------------------------------------------
                # SOV CHECK
                # -----------------------------------------------------------
                def on_check_sov(extracted_text_val):
                    if not extracted_text_val or not extracted_text_val.strip():
                        return "No text loaded."
                    cleaned = re.sub(r'\s*\.\s*', '.', extracted_text_val or '')
                    cleaned = re.sub(r'\s*।\s*', '।', cleaned)
                    sentences = [s.strip() for s in re.split(r'[.।\n]+', cleaned) if s.strip()]
                    if not sentences:
                        return "No sentences found."
                    results = []
                    total   = 0
                    for si, sentence in enumerate(sentences, 1):
                        words  = sentence.split()
                        viols  = []
                        for wi, word in enumerate(words):
                            clean = word.strip('.,!?;:()[]{}"\'')
                            if not clean: continue
                            flag = validator._check_sov_violation(clean, wi, sentence)
                            if flag:
                                viols.append(flag)
                                total += 1
                        if viols:
                            results.append(f"\nSentence {si}: {sentence}")
                            for v in viols:
                                if "| Suggested:" in v:
                                    f_part, s_part = v.split("| Suggested:")
                                    results.append(f"  ⚠️ {f_part.strip()}")
                                    results.append(f"  💡 Suggested: {s_part.strip()}")
                                else:
                                    results.append(f"  ⚠️ {v}")
                    if not results:
                        return "✅ Word order check complete — No SOV violations found."
                    return f"Word Order Check — {total} violation(s) found:\n" + "\n".join(results)

                check_sov_btn.click(fn=on_check_sov, inputs=[extracted_text], outputs=[sov_results])
                


                def on_check_conjunctions(extracted_text_val):
                    if not extracted_text_val or not extracted_text_val.strip():
                        return "No text loaded."
                    from conjunction_rules import check_paragraph_conjunctions, format_results_for_ui
                    results = check_paragraph_conjunctions(extracted_text_val)
                    return format_results_for_ui(results)

                check_conjunction_btn.click(
                fn=on_check_conjunctions,
                inputs=[extracted_text],
                outputs=[conjunction_results]
)
                # -----------------------------------------------------------
                # APPLY CORRECTION
                # -----------------------------------------------------------
                def on_apply_correction(word_data, selected_idx, chosen_suggestion, manual_edit, filter_type):
                    if not word_data or selected_idx < 0 or selected_idx >= len(word_data):
                        return (gr.update(), word_data, selected_idx, "",
                                update_stats(word_data), "No word selected",
                                gr.update(), "Selected: **--**", gr.update(), "", gr.update())

                    item = word_data[selected_idx]

                    if manual_edit and manual_edit.strip():
                        correction = manual_edit.strip()
                    elif chosen_suggestion and chosen_suggestion != "(No similar words found)":
                        correction = chosen_suggestion.replace(" (Honorific correction)", "").strip()
                    else:
                        return (gr.update(), word_data, selected_idx, "",
                                update_stats(word_data), "No correction selected",
                                gr.update(), f"**Selected:** {item['clean']}", gr.update(), "", gr.update())

                    correction_tracker['changes'].append({
                        'position': selected_idx, 'original': item['word'],
                        'corrected': correction, 'type': 'correction'
                    })

                    current_words = [w['word'] for w in word_data]
                    current_words[selected_idx] = correction
                    new_text = " ".join(current_words)
                    undo_manager.add_state(new_text)

                    # Recompute word data with fresh suggestions
                    new_word_data = sync_words_from_text(new_text, precompute=False)
                    for w in new_word_data:
                        if w['word'] == correction:
                            w['corrected'] = True
                            break

                    choices = create_word_list_choices(new_word_data, filter_type)
                    stats   = update_stats(new_word_data)
                    preview = generate_live_preview_html(new_text, correction)

                    next_idx = selected_idx
                    for i in range(selected_idx + 1, len(new_word_data)):
                        if new_word_data[i]['status_class'] == "flagged":
                            next_idx = i
                            break

                    next_choice  = choices[next_idx] if next_idx < len(choices) else None
                    next_sel_md  = f"**Selected:** {new_word_data[next_idx]['clean']}" if next_idx < len(new_word_data) else "Selected: --"
                    next_sugg    = get_suggestions_radio_from_cache(new_word_data, next_idx)

                    return (new_text, new_word_data, next_idx, correction, stats,
                            f"Corrected: {item['word']} → {correction}",
                            gr.Radio(choices=choices, value=next_choice), next_sel_md,
                            next_sugg, "", preview)

                apply_btn.click(
                    fn=on_apply_correction,
                    inputs=[word_list_state, current_word_idx, suggestions_radio, manual_edit_box, word_filter],
                    outputs=[extracted_text, word_list_state, current_word_idx, recently_changed_word,
                             stats_md, status_display, word_selector, selected_word_md,
                             suggestions_radio, manual_edit_box, live_preview_html]
                )

                # -----------------------------------------------------------
                # SPLIT — FIX: syncs word list after split
                # -----------------------------------------------------------
                def on_split(selected_choice, word_data, extracted_text_val, filter_type):
                    if not selected_choice or not extracted_text_val:
                        return (gr.update(), word_data, 0, gr.update(),
                                "No word selected", gr.update(), gr.update())
                    clean = selected_choice.strip('.,!?;:()[]{}"\'।')
                    if '.' in selected_choice or '।' in selected_choice:
                        parts = [p for p in re.split(r'([।.])', selected_choice) if p.strip()]
                        replacement = ' '.join(parts)
                    else:
                        mid = len(clean) // 2
                        replacement = clean[:mid] + ' ' + clean[mid:]
                    new_text   = extracted_text_val.replace(selected_choice, replacement, 1)
                    new_text   = re.sub(r'\s+', ' ', new_text).strip()
                    undo_manager.add_state(new_text)
                    new_word_data = sync_words_from_text(new_text, precompute=True)
                    choices    = create_word_list_choices(new_word_data, filter_type)
                    preview    = generate_live_preview_html(new_text)
                    stats      = update_stats(new_word_data)
                    return (new_text, new_word_data, 0,
                            gr.Radio(choices=choices, value=choices[0] if choices else None),
                            f"Split '{selected_choice}'", preview, stats)

                split_btn.click(
                    fn=on_split,
                    inputs=[word_selector, word_list_state, extracted_text, word_filter],
                    outputs=[extracted_text, word_list_state, current_word_idx,
                             word_selector, status_display, live_preview_html, stats_md]
                )

                # -----------------------------------------------------------
                # DELETE — FIX: syncs word list after delete
                # -----------------------------------------------------------
                def on_delete(selected_choice, word_data, extracted_text_val, filter_type):
                    if not selected_choice or not extracted_text_val:
                        return (gr.update(), word_data, 0, gr.update(),
                                "No word selected", gr.update(), gr.update())
                    new_text = extracted_text_val.replace(selected_choice, '', 1)
                    new_text = re.sub(r'\s+', ' ', new_text).strip()
                    undo_manager.add_state(new_text)
                    new_word_data = sync_words_from_text(new_text, precompute=True)
                    choices   = create_word_list_choices(new_word_data, filter_type)
                    preview   = generate_live_preview_html(new_text)
                    stats     = update_stats(new_word_data)
                    return (new_text, new_word_data, 0,
                            gr.Radio(choices=choices, value=choices[0] if choices else None),
                            f"Deleted '{selected_choice}'", preview, stats)

                delete_btn.click(
                    fn=on_delete,
                    inputs=[word_selector, word_list_state, extracted_text, word_filter],
                    outputs=[extracted_text, word_list_state, current_word_idx,
                             word_selector, status_display, live_preview_html, stats_md]
                )

                # -----------------------------------------------------------
                # SKIP — FIX: now updates word selector UI correctly
                # -----------------------------------------------------------
                def on_skip(current_idx, word_data, filter_type):
                    if not word_data:
                        return current_idx, "Nothing to skip", gr.update(), "Selected: **--**", gr.update()
                    choices = create_word_list_choices(word_data, filter_type)
                    if not choices:
                        return current_idx, "No words to skip", gr.update(), "Selected: **--**", gr.update()
                    next_idx = (current_idx + 1) % len(word_data)
                    sel_display = f"**Selected:** {word_data[next_idx]['clean']} (Word {next_idx + 1}/{len(word_data)})"
                    sugg_radio  = get_suggestions_radio_from_cache(word_data, next_idx)
                    next_word   = word_data[next_idx]['word']
                    next_choice = next_word if next_word in choices else (choices[0] if choices else None)
                    return (next_idx, f"Skipped to word {next_idx + 1}",
                            gr.Radio(choices=choices, value=next_choice),
                            sel_display, sugg_radio)

                skip_btn.click(
                    fn=on_skip,
                    inputs=[current_word_idx, word_list_state, word_filter],
                    outputs=[current_word_idx, status_display, word_selector,
                             selected_word_md, suggestions_radio]
                )

                # -----------------------------------------------------------
                # INSERT WORD
                # -----------------------------------------------------------
                def on_insert(position_choice, insert_word, word_data, extracted_text_val, filter_type):
                    if not insert_word or not insert_word.strip():
                        return (gr.update(), word_data, gr.update(),
                                "Please type a word to insert", gr.update(), gr.update())
                    if not extracted_text_val:
                        return (gr.update(), word_data, gr.update(),
                                "No text loaded", gr.update(), gr.update())

                    words = extracted_text_val.split()
                    insert_word = insert_word.strip()

                    if position_choice == "Before first word":
                        words.insert(0, insert_word)
                    else:
                        target = position_choice.replace("After '", "").rstrip("'")
                        inserted = False
                        for i, w in enumerate(words):
                            if w == target:
                                words.insert(i + 1, insert_word)
                                inserted = True
                                break
                        if not inserted:
                            words.append(insert_word)

                    new_text = ' '.join(words)
                    undo_manager.add_state(new_text)
                    new_word_data  = sync_words_from_text(new_text, precompute=True)
                    choices        = create_word_list_choices(new_word_data, filter_type)
                    insert_choices = create_insert_position_choices(new_word_data)
                    preview        = generate_live_preview_html(new_text, insert_word)
                    return (new_text, new_word_data,
                            gr.Dropdown(choices=insert_choices, value=insert_choices[0]),
                            f"Inserted '{insert_word}'", preview,
                            gr.Radio(choices=choices, value=choices[0] if choices else None))

                insert_btn.click(
                    fn=on_insert,
                    inputs=[insert_position_dropdown, insert_word_box, word_list_state, extracted_text, word_filter],
                    outputs=[extracted_text, word_list_state, insert_position_dropdown,
                             status_display, live_preview_html, word_selector]
                )

                # -----------------------------------------------------------
                # UNDO / REDO
                # -----------------------------------------------------------
                def on_undo(current_text):
                    previous = undo_manager.undo()
                    if previous is not None:
                        u_txt = f"**Undo:** {'✓' if undo_manager.can_undo() else '✗'} | **Redo:** {'✓' if undo_manager.can_redo() else '✗'}"
                        return previous, "Undone", u_txt
                    return current_text, "Nothing to undo", "*No more undo history*"

                def on_redo(current_text):
                    nxt = undo_manager.redo()
                    if nxt is not None:
                        u_txt = f"**Undo:** {'✓' if undo_manager.can_undo() else '✗'} | **Redo:** {'✓' if undo_manager.can_redo() else '✗'}"
                        return nxt, "Redone", u_txt
                    return current_text, "Nothing to redo", "*No more redo history*"

                undo_btn.click(fn=on_undo, inputs=[extracted_text], outputs=[extracted_text, status_display, undo_status])
                redo_btn.click(fn=on_redo, inputs=[extracted_text], outputs=[extracted_text, status_display, undo_status])

                # -----------------------------------------------------------
                # APPLY ALL / RESET ALL
                # -----------------------------------------------------------
                def on_apply_all(extracted_text_val):
                    return extracted_text_val, "All changes applied"

                def on_reset_all():
                    from new_validator import ContextAnalyzer
                    ContextAnalyzer.clear_entity_memory()
                    original = correction_tracker.get('original_text', '')
                    correction_tracker['changes'] = []
                    undo_manager.add_state(original)
                    return original, "Reset to original text"

                apply_all_btn.click(fn=on_apply_all, inputs=[extracted_text], outputs=[extracted_text, status_display])
                reset_all_btn.click(fn=on_reset_all, inputs=[], outputs=[extracted_text, status_display])

            # =================================================================
            # TAB 3: EVALUATE QUALITY
            # =================================================================
            with gr.Tab("Evaluate Quality"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gt_upload        = gr.File(label="Upload Ground Truth (.txt)", file_types=[".txt"])
                        gt_preview       = gr.Textbox(label="Ground Truth Preview", lines=5, interactive=False)
                        calc_metrics_btn = gr.Button("Calculate Metrics", variant="primary", size="lg")
                        metrics_status   = gr.Markdown("Status: Ready")
                    with gr.Column(scale=2):
                        metrics_table = gr.Dataframe(
                            headers=["Metric", "Value"], value=[],
                            label="OCR Evaluation Metrics", interactive=False
                        )
                        with gr.Accordion("Error Analysis", open=False):
                            error_analysis_df = gr.Dataframe(
                                headers=["Error Type", "Count", "Notes"], value=[]
                            )

                def on_gt_upload(file):
                    if file is None: return ""
                    try:
                        with open(file, 'r', encoding='utf-8') as f: content = f.read()
                        return content[:500] + ("..." if len(content) > 500 else "")
                    except Exception as e: return f"Error: {str(e)}"

                gt_upload.change(fn=on_gt_upload, inputs=[gt_upload], outputs=[gt_preview])

                def on_calculate_metrics(gt_file, extracted):
                    if gt_file is None:   return [], "Error: No ground truth uploaded", []
                    if not extracted or not extracted.strip(): return [], "Error: No extracted text", []
                    try:
                        with open(gt_file, 'r', encoding='utf-8') as f:
                            ground_truth = f.read()
                        m = calculate_all_metrics(ground_truth, extracted)
                        table = [
                            ["CER",               f"{m['CER']:.2f}%"],
                            ["WER",               f"{m['WER']:.2f}%"],
                            ["Character Accuracy", f"{m['Character Accuracy']:.2f}%"],
                            ["Word Accuracy",      f"{m['Word Accuracy']:.2f}%"],
                            ["Precision",          f"{m['Precision (word level)']:.2f}%"],
                            ["Recall",             f"{m['Recall (word level)']:.2f}%"],
                            ["F1 Score",           f"{m['F1 Score (word level)']:.2f}%"],
                            ["MER",                f"{m['MER']:.2f}%" if JIWER_AVAILABLE else "N/A"],
                            ["WIL",                f"{m['WIL']:.2f}%" if JIWER_AVAILABLE else "N/A"],
                            ["WIP",                f"{m['WIP']:.2f}%" if JIWER_AVAILABLE else "N/A"],
                            ["Edit Distance",      str(m['Edit Distance (word level)'])],
                            ["Ref word count",     str(m['Ref word count'])],
                            ["Hyp word count",     str(m['Hyp word count'])],
                        ]
                        error_data = [
                            ["Character Errors", str(int((m['CER']/100) * len(ground_truth))),
                             "Good" if m['Character Accuracy'] > 90 else "Needs improvement"],
                            ["Word Errors",      str(int((m['WER']/100) * m['Ref word count'])),
                             "Good" if m['Word Accuracy'] > 85 else "Needs improvement"],
                        ]
                        return table, "✓ Calculation Complete", error_data
                    except Exception as e:
                        return [], f"Error: {str(e)}", []

                calc_metrics_btn.click(
                    fn=on_calculate_metrics,
                    inputs=[gt_upload, extracted_text],
                    outputs=[metrics_table, metrics_status, error_analysis_df]
                )

            # =================================================================
            # TAB 4: SETTINGS & ABOUT
            # =================================================================
            with gr.Tab("Settings & About"):
                with gr.Accordion("Export Options", open=True):
                    export_format     = gr.Radio(["TXT", "JSON", "CSV"], value="TXT", label="Choose Format")
                    export_btn        = gr.Button("Export Current Session", variant="primary", size="lg")
                    export_status     = gr.Markdown("")
                    export_file_output = gr.File(label="Download Export")

                gr.Markdown("""
### Quick Guide
1. **Extract:** Upload PDF or type text, then extract/load
2. **Correct:** Click "Load Words" (suggestions pre-computed — fast!), click word, apply correction
3. **Context Panel:** See paragraph topic and suspicious words on the right
4. **SOV Check:** Click "Check Word Order" for grammar analysis
5. **Evaluate:** Upload ground truth to calculate CER/WER
6. **Export:** Save corrected text
                """)

                with gr.Accordion("System Information", open=True):
                    gr.HTML(f"<div class='gpu-status'><strong>GPU Status:</strong> {GPU_STATUS_MSG}</div>")
                    gr.Markdown(f"""
**System Configuration:**
- Enhanced Validator: Active
- Suggestions: Pre-computed at load time (fast per-click)
- Paragraph Context Analysis: {'Gemini-powered' if GEMINI_AVAILABLE else 'Local (Gemini disabled)'}
- Undo/Redo: Enabled (50 steps)
- Translation: {'Active' if TRANSLATE_AVAILABLE else 'Unavailable'}
                    """)

                def on_export(extracted_text_val, format_type):
                    if not extracted_text_val:
                        return "No text to export", None
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if format_type == "JSON":
                        fp = f"/tmp/kannada_export_{ts}.json"
                        data = {
                            "original_text": correction_tracker['original_text'],
                            "final_text": extracted_text_val,
                            "changes_made": correction_tracker['changes'],
                            "total_changes": len(correction_tracker['changes']),
                            "timestamp": ts,
                        }
                        with open(fp, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        return "Exported as JSON", fp
                    elif format_type == "CSV":
                        import csv
                        fp = f"/tmp/kannada_export_{ts}.csv"
                        with open(fp, 'w', encoding='utf-8', newline='') as f:
                            w = csv.writer(f)
                            w.writerow(["Section", "Content"])
                            w.writerow(["Original Text", correction_tracker['original_text']])
                            w.writerow(["Final Text", extracted_text_val])
                            w.writerow(["Total Changes", len(correction_tracker['changes'])])
                            w.writerow([])
                            w.writerow(["Position", "Original", "Corrected", "Type"])
                            for c in correction_tracker['changes']:
                                w.writerow([c.get('position',''), c.get('original',''), c.get('corrected',''), c.get('type','')])
                        return "Exported as CSV", fp
                    else:
                        fp = f"/tmp/kannada_export_{ts}.txt"
                        with open(fp, 'w', encoding='utf-8') as f:
                            f.write("="*60 + "\nKANNADA OCR CORRECTION EXPORT\n" + "="*60 + "\n\n")
                            f.write("ORIGINAL TEXT:\n" + "-"*60 + "\n")
                            f.write(correction_tracker['original_text'] + "\n\n")
                            f.write("CHANGES MADE:\n" + "-"*60 + "\n")
                            for i, c in enumerate(correction_tracker['changes'], 1):
                                f.write(f"{i}. Pos {c.get('position','?')}: '{c.get('original','')}' → '{c.get('corrected','')}'\n")
                            f.write(f"\nTotal: {len(correction_tracker['changes'])}\n\n")
                            f.write("FINAL TEXT:\n" + "-"*60 + "\n" + extracted_text_val + "\n")
                        return f"Exported TXT ({len(correction_tracker['changes'])} corrections)", fp

                export_btn.click(fn=on_export, inputs=[extracted_text, export_format],
                                 outputs=[export_status, export_file_output])

        gr.Markdown("---")
        gr.Markdown("*Kannada OCR Correction System — Fixed Edition*")

    return app


if __name__ == "__main__":
    print("\n" + "="*60)
    print("KANNADA OCR CORRECTION SYSTEM - STARTING")
    print("="*60)
    print(f"GPU Status: {GPU_STATUS_MSG}")
    print(f"Translation Available: {TRANSLATE_AVAILABLE}")
    print(f"Gemini Context Analysis: {GEMINI_AVAILABLE}")
    print("="*60 + "\n")
    print("🌐 Access your UI at:")
    print("   http://127.0.0.1:7860")
    print("   http://localhost:7860")
    print("="*60 + "\n")

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False,
        theme=custom_theme,
        css=custom_css
    )