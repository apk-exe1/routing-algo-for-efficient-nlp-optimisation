"""
Kannada Verb Conjugation Rules
Defines suffixes for different tenses, persons, numbers, and genders
"""

class ConjugationRules:
    """
    Rule database for Kannada verb conjugation
    Organizes suffixes by tense, person, number, and gender
    """
    
    def __init__(self):
        # INFINITIVE forms (used with ಬೇಕು, ಬಹುದು, etc.)
        self.infinitive = {
            "suffix": "ಅಲು",  # Base suffix
            "contexts": ["ಬೇಕು", "ಬಹುದು", "ಆಗು", "ಸಾಧ್ಯ"],  # Words that take infinitive
        }
        
        # PRESENT TENSE conjugations (-ುತ್ತ- base)
        self.present = {
            # First person
            "1SG": {  # I (ನಾನು)
                "suffix": "ಉತ್ತೇನೆ",
                "pronouns": ["ನಾನು"]
            },
            "1PL": {  # We (ನಾವು)
                "suffix": "ಉತ್ತೇವೆ", 
                "pronouns": ["ನಾವು"]
            },
            
            # Second person
            "2SG_INFORMAL": {  # You-informal (ನೀನು)
                "suffix": "ಉತ್ತೀಯ",
                "pronouns": ["ನೀನು"]
            },
            "2SG_FORMAL": {  # You-formal (ನೀವು)
                "suffix": "ಉತ್ತೀರಿ",
                "pronouns": ["ನೀವು"]
            },
            "2PL": {  # You-plural
                "suffix": "ಉತ್ತೀರಿ",
                "pronouns": ["ನೀವು"]
            },
            
            # Third person singular
            "3SG_MALE": {  # He (ಅವನು)
                "suffix": "ಉತ್ತಾನೆ",
                "pronouns": ["ಅವನು", "ಈತನು"]
            },
            "3SG_FEMALE": {  # She (ಅವಳು)
                "suffix": "ಉತ್ತಾಳೆ",
                "pronouns": ["ಅವಳು", "ಈಕೆ"]
            },
            "3SG_NEUTRAL": {  # It (ಅದು)
                "suffix": "ಉತ್ತದೆ",
                "pronouns": ["ಅದು", "ಇದು"]
            },
            
            # Third person plural / formal
            "3PL": {  # They/You-honorific (ಅವರು)
                "suffix": "ಉತ್ತಾರೆ",
                "pronouns": ["ಅವರು", "ಇವರು"]
            },
        }
        
        # PAST TENSE conjugations (-ಇದ್- base)
        self.past = {
            # First person
            "1SG": {
                "suffix": "ಇದ್ದೇನೆ",  # I did
                "pronouns": ["ನಾನು"]
            },
            "1PL": {
                "suffix": "ಇದ್ದೇವೆ",  # We did
                "pronouns": ["ನಾವು"]
            },
            
            # Second person
            "2SG_INFORMAL": {
                "suffix": "ಇದ್ದೀಯ",  # You did (informal)
                "pronouns": ["ನೀನು"]
            },
            "2SG_FORMAL": {
                "suffix": "ಇದ್ದೀರಿ",  # You did (formal)
                "pronouns": ["ನೀವು"]
            },
            "2PL": {
                "suffix": "ಇದ್ದೀರಿ",
                "pronouns": ["ನೀವು"]
            },
            
            # Third person
            "3SG_MALE": {
                "suffix": "ಇದ್ದಾನೆ",  # He did
                "pronouns": ["ಅವನು", "ಈತನು"]
            },
            "3SG_FEMALE": {
                "suffix": "ಇದ್ದಾಳೆ",  # She did
                "pronouns": ["ಅವಳು", "ಈಕೆ"]
            },
            "3SG_NEUTRAL": {
                "suffix": "ಇತ್ತು",  # It did/was
                "pronouns": ["ಅದು", "ಇದು"]
            },
            "3PL": {
                "suffix": "ಇದ್ದಾರೆ",  # They did
                "pronouns": ["ಅವರು", "ಇವರು"]
            },
        }
        
        # FUTURE TENSE conjugations (lower priority for thesis)
        self.future = {
            "1SG": {
                "suffix": "ಉವೆನು",
                "pronouns": ["ನಾನು"]
            },
            "3SG_MALE": {
                "suffix": "ಉವನು",
                "pronouns": ["ಅವನು"]
            },
            "3SG_FEMALE": {
                "suffix": "ಉವಳು",
                "pronouns": ["ಅವಳು"]
            },
            "3PL": {
                "suffix": "ಉವರು",
                "pronouns": ["ಅವರು"]
            }
        }
        
        # NEGATIVE forms (with ಅಲ್ಲ)
        self.negative_present = {
            "1SG": "ಉವುದಿಲ್ಲ",
            "3SG_MALE": "ಉವುದಿಲ್ಲ",
        }
        
        # CONTINUOUS forms (-ುತ್ತಿ- base)
        self.continuous = {
            "3SG_MALE": {
                "suffix": "ುತ್ತಿದ್ದಾನೆ",  # is doing
                "pronouns": ["ಅವನು"]
            }
        }
    
    def get_suffix(self, tense, person=None, number=None, gender=None):
        """
        Get appropriate suffix for given grammatical features
        
        Args:
            tense: "INFINITIVE", "PRESENT", "PAST", "FUTURE"
            person: "1", "2", "3" (optional for infinitive)
            number: "SG", "PL" (optional)
            gender: "MALE", "FEMALE", "NEUTRAL" (optional)
        
        Returns:
            suffix string or None
        """
        if tense == "INFINITIVE":
            return self.infinitive["suffix"]
        
        # Build person key
        if tense == "PRESENT":
            rule_dict = self.present
        elif tense == "PAST":
            rule_dict = self.past
        elif tense == "FUTURE":
            rule_dict = self.future
        else:
            return None
        
        # Construct key like "3SG_MALE"
        if person and number:
            key = f"{person}{number}"
            if gender and person == "3" and number == "SG":
                key = f"{key}_{gender}"
            elif person == "2" and number == "SG":
                # Default to formal for 2SG
                key = f"{key}_FORMAL"
            
            if key in rule_dict:
                return rule_dict[key]["suffix"]
        
        return None
    
    def get_infinitive_contexts(self):
        """Return list of words that require infinitive form"""
        return self.infinitive["contexts"]
    
    def detect_person_from_pronoun(self, pronoun):
        """
        Detect person/number/gender from Kannada pronoun
        
        Returns: tuple (person, number, gender) or None
        """
        pronoun_map = {
            "ನಾನು": ("1", "SG", None),
            "ನಾವು": ("1", "PL", None),
            "ನೀನು": ("2", "SG", "INFORMAL"),
            "ನೀವು": ("2", "PL", None),  # Can also be formal 2SG
            "ಅವನು": ("3", "SG", "MALE"),
            "ಈತನು": ("3", "SG", "MALE"),
            "ಅವಳು": ("3", "SG", "FEMALE"),
            "ಈಕೆ": ("3", "SG", "FEMALE"),
            "ಅದು": ("3", "SG", "NEUTRAL"),
            "ಇದು": ("3", "SG", "NEUTRAL"),
            "ಅವರು": ("3", "PL", None),
            "ಇವರು": ("3", "PL", None),
        }
        return pronoun_map.get(pronoun)
    
    def get_all_present_forms(self, suffix_only=False):
        """Get all present tense suffixes (for matching/detection)"""
        if suffix_only:
            return [v["suffix"] for v in self.present.values()]
        return self.present
    
    def get_all_past_forms(self, suffix_only=False):
        """Get all past tense suffixes (for matching/detection)"""
        if suffix_only:
            return [v["suffix"] for v in self.past.values()]
        return self.past
    
    def detect_tense_from_suffix(self, suffix):
        """
        Detect what tense a suffix belongs to
        Returns: "PRESENT", "PAST", "INFINITIVE", or None
        """
        # Check present
        for form_data in self.present.values():
            if suffix == form_data["suffix"]:
                return "PRESENT"
        
        # Check past
        for form_data in self.past.values():
            if suffix == form_data["suffix"]:
                return "PAST"
        
        # Check infinitive
        if suffix == self.infinitive["suffix"] or suffix.endswith("ಅಲು"):
            return "INFINITIVE"
        
        return None