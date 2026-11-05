"""Post-Processing and Spelling Correction Module for Azerbaijani OCR (CRNN+CTC).
Uses SymSpell algorithm with a 116,000+ word frequency dictionary.
"""
import sys
import re
from pathlib import Path
from symspellpy import SymSpell, Verbosity

AZ_SUFFIXES = [
    "larımızdan", "lərimizdən", "larımızda", "lərimizdə", "larımızla", "lərimizlə",
    "larından", "lərindən", "larında", "lərində", "larının", "lərinin", "larına", "lərinə",
    "mışdır", "mişdir", "muşdur", "müşdür", "acağam", "əcəyəm", "acaqsan", "əcəksən",
    "malıdır", "məlidir", "maqdadır", "məkdədir",
    "ımızdan", "imizdən", "umuzdan", "ümüzdən",
    "larımız", "lərimiz", "larınız", "ləriniz",
    "sınız", "siniz", "sunuz", "sünüz",
    "mağa", "məyə", "maqla", "məklə",
    "acaq", "əcək", "saq", "sək", "san", "sən", "saydı", "səydi",
    "ından", "indən", "undan", "ündən",
    "yından", "yindən", "yundan", "yündən",
    "sından", "sindən", "sundan", "sündən",
    "mız", "miz", "muz", "müz", "nız", "niz", "nuz", "nüz",
    "ımız", "imiz", "umuz", "ümüz", "ınız", "iniz", "unuz", "ünüz",
    "dan", "dən", "tan", "tən",
    "nın", "nin", "nun", "nün",
    "yın", "yin", "yun", "yün",
    "sın", "sin", "sun", "sün",
    "lar", "lər",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "dım", "dim", "dum", "düm", "dıq", "dik", "duq", "dük",
    "dın", "din", "dun", "dün",
    "mış", "miş", "muş", "müş",
    "ya", "yə", "na", "nə", "sa", "sə",
    "da", "də", "ta", "tə",
    "ın", "in", "un", "ün",
    "nı", "ni", "nu", "nü",
    "yı", "yi", "yu", "yü",
    "sı", "si", "su", "sü",
    "m", "n", "q", "k", "a", "ə", "ı", "i", "u", "ü"
]


class AzerbaijaniPostProcessor:
    def __init__(self, dict_path=None, max_edit_distance=2):
        if dict_path is None:
            dict_path = Path(__file__).parent / "data" / "az_lexicon.txt"
        
        self.dict_path = Path(dict_path)
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        self.base_vocab = set()
        
        if self.dict_path.exists():
            # Pass encoding="utf-8" for proper Azerbaijani diacritics support
            self.sym_spell.load_dictionary(str(self.dict_path), term_index=0, count_index=1, separator=" ", encoding="utf-8")
            self.base_vocab = set(self.sym_spell.words.keys())
            self.loaded = True
        else:
            self.loaded = False

    def is_valid_morphologically(self, word_lower: str) -> bool:
        """Returns True if word is in vocabulary directly or by removing a valid Azerbaijani suffix."""
        if word_lower in self.base_vocab:
            return True
        for suf in AZ_SUFFIXES:
            if word_lower.endswith(suf) and len(word_lower) - len(suf) >= 2:
                stem = word_lower[:-len(suf)]
                if stem in self.base_vocab:
                    return True
        return False

    def clean_punctuation(self, text: str) -> str:
        """Fix repeated punctuation and OCR trailing artifacts."""
        t = text.strip()
        # Collapse multiple dots (e.g. '..' -> '.')
        t = re.sub(r"\.{2,}", ".", t)
        # Collapse multiple hyphens
        t = re.sub(r"-{2,}", "-", t)
        # Fix trailing noise after sentence terminal dot (e.g., '. -.' or '.-aa')
        t = re.sub(r"\.\s*[-.,a-zA-ZçəğıöşüÇƏĞIİÖŞÜ]+$", ".", t)
        # Normalize multiple spaces
        t = re.sub(r"\s+", " ", t)
        return t

    def correct_word(self, token: str, max_edit_distance=1) -> str:
        """Correct a single token preserving casing and punctuation."""
        if not token or not self.loaded:
            return token

        # Extract leading and trailing punctuation (e.g. '(abstract)' or 'göstərilməsi.')
        m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", token)
        if not m:
            return token
        prefix, core, suffix = m.groups()
        if not core:
            return token

        # Skip abbreviations, acronyms, or numbers (e.g. IT, CSV, ZIP, 2026)
        if re.match(r"^\d+$", core) or (core.isupper() and len(core) <= 4):
            return token

        # Handle hyphenated compounds (e.g. 'Veb-saytın')
        if "-" in core:
            parts = core.split("-")
            corr_parts = [self.correct_word(p, max_edit_distance) for p in parts]
            # Domain-specific fix for 'veb'
            if parts[0].lower() in ["y6b", "yeb", "vöb"]:
                corr_parts[0] = "Veb" if (parts[0][0].isupper() or parts[0] in ['Y', 'y']) else "veb"
            return prefix + "-".join(corr_parts) + suffix

        # If length is very short (1-2 chars), avoid aggressive changes
        if len(core) <= 2:
            return token

        core_lower = core.lower()
        # Deterministic Azerbaijani phonotactic OCR fix: -ləvirdi / -lavırdı -> -ləyirdi / -layırdı
        if re.search(r"(lə|la)v(ir|ır)(di|dı|dilər|dılar)$", core_lower):
            fixed_core = re.sub(r"(lə|la)v(ir|ır)(di|dı|dilər|dılar)$", r"\1y\2\3", core_lower)
            if core.isupper():
                fixed_core = fixed_core.upper()
            elif core[0].isupper():
                fixed_core = fixed_core.capitalize()
            return prefix + fixed_core + suffix

        if self.is_valid_morphologically(core_lower):
            return token

        # Look up with SymSpell
        suggestions = self.sym_spell.lookup(
            core_lower,
            Verbosity.CLOSEST,
            max_edit_distance=max_edit_distance,
            include_unknown=True
        )

        if suggestions:
            best = suggestions[0]
            # If a correction was found within distance
            if best.distance > 0 and best.term != core_lower:
                corrected_term = best.term
                # Match original capitalization
                if core.isupper():
                    core_out = corrected_term.upper()
                elif core[0].isupper():
                    core_out = corrected_term.capitalize()
                else:
                    core_out = corrected_term
                return prefix + core_out + suffix

        return token

    def process_line(self, line: str, max_edit_distance=1) -> str:
        """Complete post-processing pipeline for an OCR line."""
        if not line:
            return ""

        # Step 1: Pre-clean punctuation
        cleaned = self.clean_punctuation(line)

        # Step 2: Token-level spell correction
        tokens = cleaned.split(" ")
        corrected_tokens = [self.correct_word(tok, max_edit_distance) for tok in tokens]
        out = " ".join(corrected_tokens)

        # Step 3: Final punctuation check
        out = self.clean_punctuation(out)
        return out


# Singleton instance for quick access
_default_processor = None

def correct_ocr_text(text: str, max_edit_distance=1) -> str:
    """Convenience function to post-process and correct OCR output string."""
    global _default_processor
    if _default_processor is None:
        _default_processor = AzerbaijaniPostProcessor()
    return _default_processor.process_line(text, max_edit_distance)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    processor = AzerbaijaniPostProcessor()
    test_samples = [
        "y6b-saytın dizaynının və İstifadəçi interfeysinin yenilənməsi ",
        "hazırlanması..",
        "Nəşr və xldmət haqları barədə məlumatların ayrıca təqdim edilməsi.",
        "Onlayn məqalə təqdimetmə imkanlarının təkmlləşdirilməsi.",
        "Xafici müəliflər üçün inglis dlində möumatlamn eçalanığınn artınlması.-aa",
        "tərəfindən verilməsi imkanının yaradılması. -.",
        "Naxçıvan Dövlət Universiteti",
        "Daxilolma № 3-29-75/1-2-698/26"
    ]
    print("Testing AzerbaijaniPostProcessor:")
    for s in test_samples:
        corr = processor.process_line(s)
        print(f"RAW : {s}")
        print(f"CORR: {corr}\n")
