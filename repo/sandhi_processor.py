"""
Kannada Sandhi Rules Processor
Handles phonological changes when morphemes combine
"""

class SandhiProcessor:
    """
    Applies sandhi (phonological) rules when combining root + suffix
    
    Top 5 Sandhi Rules covering 90% of cases:
    1. Vowel + Vowel → Insert euphonic ಯ/ವ and merge vowels
    2. Consonant + Vowel → Direct concatenation
    3. Nasal assimilation
    4. Gemination (consonant doubling)
    5. Vowel lengthening
    """
    
    def __init__(self):
        # Kannada vowels
        self.vowels = ["ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ", "ೠ", "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ"]
        
        # Vowel signs (matras)
        self.vowel_signs = ["ಾ", "ಿ", "ೀ", "ು", "ೂ", "ೃ", "ೄ", "ೆ", "ೇ", "ೈ", "ೊ", "ೋ", "ೌ"]
        
        # Vowel to vowel sign mapping (CRITICAL FOR CONJUGATION!)
        self.vowel_to_sign = {
            "ಅ": "",    # Inherent vowel, no sign needed
            "ಆ": "ಾ",
            "ಇ": "ಿ",
            "ಈ": "ೀ",
            "ಉ": "ು",
            "ಊ": "ೂ",
            "ಋ": "ೃ",
            "ೠ": "ೄ",
            "ಎ": "ೆ",
            "ಏ": "ೇ",
            "ಐ": "ೈ",
            "ಒ": "ೊ",
            "ಓ": "ೋ",
            "ಔ": "ೌ"
        }
        
        # Consonants
        self.consonants = [
            "ಕ", "ಖ", "ಗ", "ಘ", "ಙ",  # Velars
            "ಚ", "ಛ", "ಜ", "ಝ", "ಞ",  # Palatals
            "ಟ", "ಠ", "ಡ", "ಢ", "ಣ",  # Retroflexes
            "ತ", "ಥ", "ದ", "ಧ", "ನ",  # Dentals
            "ಪ", "ಫ", "ಬ", "ಭ", "ಮ",  # Labials
            "ಯ", "ರ", "ಲ", "ವ",      # Semivowels
            "ಶ", "ಷ", "ಸ", "ಹ",      # Sibilants
            "ಳ", "ೞ"                 # Special
        ]
        
        # Nasal consonants (for nasal assimilation rule)
        self.nasals = ["ಙ", "ಞ", "ಣ", "ನ", "ಮ"]
        
        # Consonants that can geminate (double)
        self.geminable = ["ಕ", "ಟ", "ತ", "ಪ", "ಚ", "ದ", "ಬ", "ಗ"]
    
    def apply_sandhi(self, root, suffix, root_info):
        """
        Main sandhi application function
        
        Args:
            root: verb root (e.g., "ಕುಡಿ")
            suffix: grammatical suffix (e.g., "ಅಲು", "ಉತ್ತೇನೆ", "ಇದ್ದೇನೆ")
            root_info: dict with root classification info
        
        Returns:
            Combined form with sandhi applied
        """
        if not root or not suffix:
            return root + suffix
        
        # Get last character of root
        last_char = root[-1]
        
        # Get first character of suffix
        first_char = suffix[0] if suffix else ""
        
        # RULE 1: Vowel + Vowel → Insert ಯ/ವ and merge (most common!)
        if self._is_vowel_or_has_vowel_sign(last_char) and first_char in self.vowels:
            return self._apply_euphonic_insertion(root, suffix, root_info)
        
        # RULE 2: Consonant + Vowel → Direct concatenation
        if self._is_pure_consonant(last_char) and first_char in self.vowels:
            return self._apply_direct_concatenation(root, suffix)
        
        # RULE 3: Nasal assimilation (for certain roots)
        if root_info and root_info.get("class") == "NASAL_CONSONANT":
            return self._apply_nasal_assimilation(root, suffix)
        
        # RULE 4: Gemination (consonant doubling)
        if last_char in self.geminable and suffix.startswith("ಅ"):
            return self._apply_gemination(root, suffix)
        
        # Default: simple concatenation
        return root + suffix
    
    def _is_vowel_or_has_vowel_sign(self, char):
        """Check if character is vowel or ends with vowel sign"""
        if char in self.vowels:
            return True
        if char in self.vowel_signs:
            return True
        # Check if it's a character with inherent vowel (ಅ)
        if char in self.consonants:
            return True  # Kannada consonants have inherent 'a'
        return False
    
    def _is_pure_consonant(self, char):
        """Check if character is consonant with halant (್)"""
        return char == "್"
    
    def _apply_euphonic_insertion(self, root, suffix, root_info):
        """
        RULE 1: Insert euphonic semivowel between two vowels and merge
        
        Pattern: Vowel + Vowel → Vowel + ಯ/ವ + (vowel converted to sign)
        
        Examples:
            ಕುಡಿ + ಅಲು → ಕುಡಿ + ಯ + ಲು = ಕುಡಿಯಲು
            ಕುಡಿ + ಉತ್ತೇನೆ → ಕುಡಿ + ಯು + ತ್ತೇನೆ = ಕುಡಿಯುತ್ತೇನೆ
            ಕುಡಿ + ಇದ್ದೇನೆ → ಕುಡಿ + ಯಿ + ದ್ದೇನೆ = ಕುಡಿಯಿದ್ದೇನೆ
        
        KEY FIX: Vowels at suffix start must be converted to vowel signs (matras)
        """
        last_char = root[-1]
        
        # Determine which semivowel to insert based on last vowel
        # Most common: ಇ, ಈ, ಎ, ಏ → insert ಯ
        # Less common: ಉ, ಊ, ಒ, ಓ → insert ವ
        
        if last_char in ["ಇ", "ಈ", "ಎ", "ಏ", "ಐ"]:
            euphonic = "ಯ"
        elif last_char in ["ಉ", "ಊ", "ಒ", "ಓ", "ಔ"]:
            euphonic = "ವ"  # Less common, but grammatically correct
        else:
            # For consonants with inherent vowel, check the vowel sign
            euphonic = "ಯ"  # Default
        
        # Special handling for ಉ-ending verbs (like ಮಾಡು, ಓದು)
        if root_info and root_info.get("ending") == "ಉ":
            # ಉ + ಅ → drop ಉ, use consonant + suffix
            # Example: ಮಾಡು + ಅಲು → ಮಾಡ + ಲು = ಮಾಡಲು (no euphonic needed)
            if suffix.startswith("ಅ"):
                # Remove final ಉ, drop initial ಅ from suffix
                return root[:-1] + suffix[1:]
        
        # Standard euphonic insertion with vowel sign conversion
        # After inserting euphonic ಯ/ವ, the vowel at suffix start becomes a vowel sign
        
        first_vowel = suffix[0] if suffix else ""
        
        if first_vowel in self.vowel_to_sign:
            vowel_sign = self.vowel_to_sign[first_vowel]
            rest_of_suffix = suffix[1:]  # Everything after first vowel
            
            if vowel_sign == "":
                # Inherent ಅ - no sign needed, just drop the ಅ
                return root + euphonic + rest_of_suffix
            else:
                # Add vowel sign to euphonic consonant
                return root + euphonic + vowel_sign + rest_of_suffix
        else:
            # Fallback: just insert euphonic (shouldn't reach here normally)
            return root + euphonic + suffix
    
    def _apply_direct_concatenation(self, root, suffix):
        """
        RULE 2: Direct concatenation for consonant + vowel
        
        Pattern: Consonant್ + Vowel → merge
        Example: ಓದ್ + ಅಲು → ಓದಲು
        """
        # Remove halant and concatenate
        if root.endswith("್"):
            # If suffix starts with ಅ, drop it (inherent vowel)
            if suffix.startswith("ಅ"):
                return root[:-1] + suffix[1:]
            return root[:-1] + suffix
        return root + suffix
    
    def _apply_nasal_assimilation(self, root, suffix):
        """
        RULE 3: Nasal consonants may assimilate or geminate
        
        Example: ತಿನ್ನು + ಅಲು → ತಿನ್ನಲು (keep double nasal)
        """
        # For ತಿನ್ನು type verbs, remove final ಉ and drop ಅ from suffix
        if root.endswith("ನ್ನು") and suffix.startswith("ಅ"):
            return root[:-1] + suffix[1:]  # Keep the ನ್ನ, remove ಉ, drop ಅ
        
        if root.endswith("ಉ") and suffix.startswith("ಅ"):
            return root[:-1] + suffix[1:]
        
        return root + suffix
    
    def _apply_gemination(self, root, suffix):
        """
        RULE 4: Some consonants double before certain suffixes
        
        Pattern: C + ಅ → CC
        Example: ಹತ್ತು + ಅ → ಹತ್ತ (double ತ್)
        """
        # This is less common in standard verb conjugation
        # Usually applies in specific lexical items
        return root + suffix
    
    def remove_euphonic_y(self, word):
        """
        Reverse operation: Remove inserted ಯ to get root
        Example: ಕುಡಿಯಲು → ಕುಡಿ + ಅಲು
        """
        if "ಯ" in word:
            # Try removing ಯ and see if it makes sense
            parts = word.split("ಯ")
            if len(parts) == 2:
                return parts[0], "ಯ" + parts[1]
        return word, ""
    
    def detect_sandhi_type(self, root, conjugated_form):
        """
        Analyze what sandhi was applied to create conjugated form
        Useful for morphological analysis (reverse direction)
        
        Returns: "EUPHONIC_Y", "DIRECT", "NASAL", "GEMINATE", or "UNKNOWN"
        """
        if not conjugated_form.startswith(root):
            return "UNKNOWN"
        
        remainder = conjugated_form[len(root):]
        
        if remainder.startswith("ಯ"):
            return "EUPHONIC_Y"
        elif remainder.startswith("ವ"):
            return "EUPHONIC_V"
        else:
            return "DIRECT"