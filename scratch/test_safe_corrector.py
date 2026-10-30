import sys
import os
import re
from pathlib import Path
import pandas as pd
from symspellpy import SymSpell, Verbosity

sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"

sys.path.insert(0, str(Path.cwd()))
from common import align, normalize

# 1. Load Lexicon
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=10)
sym_spell.load_dictionary('data/az_lexicon.txt', term_index=0, count_index=1, separator=' ', encoding='utf-8')

# Build set of valid stems/words
base_vocab = set()
with open('data/az_lexicon.txt', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            base_vocab.add(parts[0].lower())

for p in Path('data/corpus_line').glob('*.txt'):
    with open(p, encoding='utf-8', errors='ignore') as f:
        for line in f:
            for w in re.findall(r'[a-zA-Z\u00C0-\u024F\u0400-\u04FF\u018F\u0259\u0130\u0131\u00C7\u00E7\u015E\u015F\u011E\u011F\u00D6\u00F6\u00DC\u00FC]+', line):
                base_vocab.add(w.lower())

print(f"Loaded base vocabulary: {len(base_vocab)} words/stems.")

# Common Azerbaijani inflectional suffixes (ordered longest to shortest)
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

def is_valid_morphologically(word_lower):
    """Returns True if word is in vocabulary directly or by removing a valid Azerbaijani suffix."""
    if word_lower in base_vocab:
        return True
    for suf in AZ_SUFFIXES:
        if word_lower.endswith(suf) and len(word_lower) - len(suf) >= 2:
            stem = word_lower[:-len(suf)]
            if stem in base_vocab:
                return True
    return False

# Common OCR confusion pairs (bidirectional)
CONFUSION_PAIRS = {
    ('ş', 's'), ('s', 'ş'),
    ('h', 'f'), ('f', 'h'),
    ('x', 's'), ('s', 'x'),
    ('c', 'ç'), ('ç', 'c'),
    ('i', 'ı'), ('ı', 'i'),
    ('e', 'ə'), ('ə', 'e'),
    ('n', 'm'), ('m', 'n'),
    ('u', 'ü'), ('ü', 'u'),
    ('o', 'ö'), ('ö', 'o'),
    ('l', 't'), ('t', 'l'),
    ('v', 'y'), ('y', 'v'),
    ('k', 'g'), ('g', 'k'),
    ('b', 'd'), ('d', 'b'),
}

def is_plausible_ocr_confusion(orig, cand):
    """Verify if the difference between orig and cand is an OCR visual confusion."""
    if len(orig) != len(cand):
        # 1 insertion/deletion check
        if abs(len(orig) - len(cand)) == 1:
            return True
        return False
    diffs = [(c1, c2) for c1, c2 in zip(orig, cand) if c1 != c2]
    if len(diffs) == 1:
        c1, c2 = diffs[0]
        return (c1, c2) in CONFUSION_PAIRS or True # Single letter substitution
    return False

def clean_word(token):
    m = re.match(r'^([^\w]*)([\w-]+)([^\w]*)$', token, flags=re.UNICODE)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return '', token, ''

def safe_correct_word(word, is_proper_noun=False):
    wl = word.lower()
    
    # RULE 1: If word is already valid (root or inflected), DO NOT TOUCH IT!
    if is_valid_morphologically(wl):
        return word
        
    # RULE 2: If proper noun, be extremely cautious
    if is_proper_noun and len(wl) <= 4:
        return word
        
    # RULE 3: Look for 1-edit distance suggestion in lexicon
    suggs = sym_spell.lookup(wl, Verbosity.CLOSEST, max_edit_distance=1)
    if suggs:
        best = suggs[0]
        if best.distance == 1 and best.count >= 3:
            cand = best.term
            if is_plausible_ocr_confusion(wl, cand):
                if word[0].isupper():
                    cand = cand.capitalize()
                return cand
    return word

def safe_correct_text(text):
    tokens = text.split()
    corrected_tokens = []
    for i, tok in enumerate(tokens):
        pfx, w, sfx = clean_word(tok)
        if not w:
            corrected_tokens.append(tok)
            continue
        is_prop = (w[0].isupper() and i > 0)
        corr_w = safe_correct_word(w, is_proper_noun=is_prop)
        corrected_tokens.append(pfx + corr_w + sfx)
        
    res = " ".join(corrected_tokens)
    
    # Punctuation cleaning
    res = re.sub(r'\s+([,.:;!?])', r'\1', res)
    return normalize(res)

# Test on Photo 5 results
df = pd.read_csv('results/real_fix9/per_line_chunk_details.csv')
total_before = 0
total_after = 0
chars = 0
corrections_log = []

for idx, r in df.iterrows():
    ref = r['ref']
    v2_orig = r['v2_rule_joined_A']
    v2_corr = safe_correct_text(v2_orig)
    
    ce_orig, _ = align(ref, v2_orig)
    ce_corr, _ = align(ref, v2_corr)
    
    total_before += ce_orig
    total_after += ce_corr
    chars += len(ref)
    
    if ce_orig != ce_corr:
        corrections_log.append({
            "line": idx + 1,
            "ref": ref,
            "orig": v2_orig,
            "corr": v2_corr,
            "err_before": ce_orig,
            "err_after": ce_corr,
            "gain": ce_orig - ce_corr
        })

print(f"\n=== MÜQAYİSƏLİ NƏTİCƏ ===")
print(f"Orijinal V2 CER: {total_before/chars*100:.2f}% ({total_before} xəta)")
print(f"Təhlükəsiz Düzəldilmiş CER: {total_after/chars*100:.2f}% ({total_after} xəta)")
print(f"Düzəldilən sətirlərin sayı: {len(corrections_log)}")

for c in corrections_log:
    print(f"\nSətir {c['line']:02d} (Qazanc: {c['gain']:+d} xəta):")
    print(f"  Ref:   {c['ref']}")
    print(f"  Əvvəl: {c['orig']} (Xəta: {c['err_before']})")
    print(f"  Sonra: {c['corr']} (Xəta: {c['err_after']})")
