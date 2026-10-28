import os
import sys
from pathlib import Path
from fontTools.ttLib import TTFont

sys.stdout.reconfigure(encoding='utf-8')

SPECIAL_ALL = 'ə ı ö ğ ü ş ç Ə İ Ö Ğ Ü Ş Ç'.split()

def get_font_family(path):
    try:
        f = TTFont(path, fontNumber=0 if path.lower().endswith(".ttc") else -1, lazy=True)
        for record in f['name'].names:
            if record.nameID == 1:
                return record.toUnicode()
    except Exception:
        pass
    return None

def check_font(path):
    p = Path(path)
    if not p.exists():
        return False, "file does not exist", []
    try:
        f = TTFont(str(p), fontNumber=0 if str(p).lower().endswith(".ttc") else -1, lazy=True)
        cmap = f.getBestCmap()
        missing = [c for c in SPECIAL_ALL if ord(c) not in cmap]
        if missing:
            return False, f"missing {len(missing)} special chars: {' '.join(missing)}", missing
        return True, "OK", []
    except Exception as e:
        return False, f"error: {e}", []

candidates_named = [
    'georgia.ttf', 'trebuc.ttf', 'verdana.ttf', 'verdanab.ttf', 'verdanai.ttf', 'verdanaz.ttf',
    'segoeui.ttf', 'segoeuib.ttf', 'segoeuii.ttf', 'segoeuiz.ttf', 'seguisb.ttf', 'seguili.ttf', 'seguibl.ttf', 'SegUIVar.ttf',
    'dubai.ttf', 'bahnschrift.ttf', 'constan.ttf', 'framd.ttf', 'lucon.ttf'
]

win_dir = Path("C:/Windows/Fonts")

print("=== CHECKING EXPLICIT CANDIDATES ===")
for name in candidates_named:
    ok, reason, missing = check_font(win_dir / name)
    fam = get_font_family(str(win_dir / name))
    print(f"{name:15s} | Family: {str(fam):20s} | OK: {str(ok):5s} | {reason}")

print("\n=== ALL FONTS IN WINDOWS FONTS PASSING FONT_OK ===")
all_files = sorted(list(win_dir.glob("*.ttf")) + list(win_dir.glob("*.ttc")) + list(win_dir.glob("*.otf")))
passed = []
for p in all_files:
    ok, _, _ = check_font(p)
    if ok:
        fam = get_font_family(str(p))
        passed.append((p.name, fam, str(p)))

for name, fam, p in passed:
    print(f"{name:25s} | {fam}")
