"""Ortaq hissə: simvollar, ön emal, metriklər (CER/WER, hizalama)."""
import unicodedata
from pathlib import Path

import cv2
import numpy as np

# Azərbaycan əlifbası (32 hərf). Böyük hərflərdə İ/I fərqinə diqqət: i->İ, ı->I
AZ_LOWER = "abcçdeəfgğhxıijkqlmnoöprsştuüvyz"
AZ_UPPER = "ABCÇDEƏFGĞHXIİJKQLMNOÖPRSŞTUÜVYZ"
DIGITS = "0123456789"
PUNCT = ".,-:;!?()\"'"
CHARS = AZ_LOWER + AZ_UPPER + DIGITS + PUNCT + " "
SPECIAL_LOWER = "əıöğüşç"
SPECIAL_ALL = "əıöğüşçƏİÖĞÜŞÇ"
CONFUSION_PAIRS = [("ə", "e"), ("ı", "i"), ("ö", "o"), ("ğ", "g"),
                   ("ü", "u"), ("ş", "s"), ("ç", "c")]
PRESETS = {"word": (32, 128), "line": (32, 512)}  # (hündürlük, en)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def az_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


class Vocab:
    """0 indeksi CTC blank üçün ayrılıb."""

    def __init__(self, chars: str = CHARS):
        self.chars = chars
        self.c2i = {c: i + 1 for i, c in enumerate(chars)}

    @property
    def size(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str):
        return [self.c2i[c] for c in text if c in self.c2i]

    def decode_greedy(self, ids) -> str:
        out, prev = [], 0
        for i in ids:
            if i != prev and i != 0:
                out.append(self.chars[i - 1])
            prev = i
        return "".join(out)


# ---------------------------------------------------------------- ön emal
DEFAULT_CFG = dict(mode="word", gray=True, filter="none", thresh="none", interp="bilinear")
_INTERP = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


def make_cfg(**kw):
    cfg = dict(DEFAULT_CFG)
    cfg.update({k: v for k, v in kw.items() if v is not None})
    cfg["height"], cfg["width"] = PRESETS[cfg["mode"]]
    return cfg


def preprocess(img, cfg):
    """img: BGR və ya gri uint8. Çıxış: float32 (C,H,W), [0,1]. Nisbət qorunur, sağdan ağ padding."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if cfg["gray"]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cfg["filter"] == "gaussian":
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif cfg["filter"] == "median":
        img = cv2.medianBlur(img, 3)
    if cfg["gray"]:  # eşikləmə yalnız gri şəkildə
        if cfg["thresh"] == "otsu":
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif cfg["thresh"] == "adaptive":
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    H, W = cfg["height"], cfg["width"]
    h, w = img.shape[:2]
    new_w = max(1, min(W, round(w * H / h)))  # çox enlidirsə W-yə sıxılır
    img = cv2.resize(img, (new_w, H), interpolation=_INTERP[cfg["interp"]])
    canvas = np.full((H, W) + img.shape[2:], 255, np.uint8)
    canvas[:, :new_w] = img
    x = canvas.astype(np.float32) / 255.0
    return x[None] if x.ndim == 2 else x.transpose(2, 0, 1)


# ---------------------------------------------------------------- metriklər
def align(ref, hyp):
    """Levenshtein məsafəsi + əməliyyatlar [(tip, ref_simvol, hyp_simvol)], tip: ok/sub/del/ins."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return int(d[n, m]), ops[::-1]


def cer_wer(refs, hyps):
    """Korpus səviyyəsində CER və WER; həm də simvol əməliyyatları."""
    ce = cn = we = wn = 0
    all_ops = []
    for r, h in zip(refs, hyps):
        e, ops = align(r, h)
        ce += e
        cn += len(r)
        all_ops.extend(ops)
        e2, _ = align(r.split(), h.split())
        we += e2
        wn += len(r.split())
    return ce / max(cn, 1), we / max(wn, 1), all_ops


def load_labels(data_dir):
    """data_dir/labels.tsv: 'images/000001.png<TAB>mətn'."""
    data_dir = Path(data_dir)
    items = []
    with open(data_dir / "labels.tsv", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fn, label = line.split("\t", 1)
            items.append((data_dir / fn, label))
    return items
