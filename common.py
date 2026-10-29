"""Ortaq modullar: Azərbaycan əlifbası, ön emal, lüğət, metriklər (CER/WER) və model inferensi."""
from typing import List, Tuple, Dict, Any, Optional
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# Azərbaycan əlifbası (32 hərf). Böyük hərflərdə İ/I fərqinə diqqət: i->İ, ı->I
AZ_LOWER: str = "abcçdeəfgğhxıijkqlmnoöprsştuüvyz"
AZ_UPPER: str = "ABCÇDEƏFGĞHXIİJKQLMNOÖPRSŞTUÜVYZ"
DIGITS: str = "0123456789"
PUNCT: str = ".,-:;!?()\"'"
CHARS: str = AZ_LOWER + AZ_UPPER + DIGITS + PUNCT + " "
SPECIAL_LOWER: str = "əıöğüşç"
SPECIAL_ALL: str = "əıöğüşçƏİÖĞÜŞÇ"
CONFUSION_PAIRS: List[Tuple[str, str]] = [
    ("ə", "e"), ("ı", "i"), ("ö", "o"), ("ğ", "g"),
    ("ü", "u"), ("ş", "s"), ("ç", "c")
]
PRESETS: Dict[str, Tuple[int, int]] = {"word": (32, 128), "line": (32, 512)}  # (hündürlük, en)


def normalize(text: str) -> str:
    """Mətni Unicode NFC standartına normallaşdırır."""
    return unicodedata.normalize("NFC", text)


def az_lower(s: str) -> str:
    """Azərbaycan dilinin xüsusi I/İ hərflərini nəzərə alaraq kiçik registrə çevirir."""
    return s.replace("İ", "i").replace("I", "ı").lower()


class Vocab:
    """Simvollar lüğəti. 0 indeksi CTC blank üçün ayrılıb."""

    def __init__(self, chars: str = CHARS):
        self.chars: str = chars
        self.c2i: Dict[str, int] = {c: i + 1 for i, c in enumerate(chars)}
        self.i2c: Dict[int, str] = {i + 1: c for i, c in enumerate(chars)}

    @property
    def size(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> List[int]:
        return [self.c2i[c] for c in text if c in self.c2i]

    def decode_greedy(self, ids: Any) -> str:
        """CTC Greedy (Best-Path) deşifrələməsi: ardıcıl eyni indeksləri və blank (0) indeksini kənarlaşdırır."""
        out: List[str] = []
        prev: int = 0
        for i in ids:
            if i != prev and i != 0:
                out.append(self.chars[i - 1])
            prev = i
        return "".join(out)


# ---------------------------------------------------------------- ön emal
DEFAULT_CFG: Dict[str, Any] = {
    "mode": "word",
    "gray": True,
    "filter": "none",
    "thresh": "none",
    "interp": "bilinear"
}
_INTERP = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


def make_cfg(**kwargs) -> Dict[str, Any]:
    """Konfiqurasiya lüğətini yaradır və rejimə görə ölçüləri təyin edir."""
    cfg = dict(DEFAULT_CFG)
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    cfg["height"], cfg["width"] = PRESETS[cfg["mode"]]
    return cfg


def preprocess(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """img: BGR və ya qri uint8. Çıxış: float32 (C,H,W), [0,1]. Nisbət qorunur, sağdan ağ padding."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if cfg.get("gray", True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Filtrlər
    filt = cfg.get("filter", "none")
    if filt == "gaussian":
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif filt == "median":
        img = cv2.medianBlur(img, 3)

    # Eşikləmə (yalnız qri rejimdə)
    if cfg.get("gray", True):
        thresh = cfg.get("thresh", "none")
        if thresh == "otsu":
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif thresh == "adaptive":
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    H, W = cfg["height"], cfg["width"]
    h, w = img.shape[:2]
    new_w = max(1, min(W, round(w * H / h)))  # Çox enlidirsə W-yə sıxılır
    img = cv2.resize(img, (new_w, H), interpolation=_INTERP[cfg.get("interp", "bilinear")])
    
    canvas = np.full((H, W) + img.shape[2:], 255, dtype=np.uint8)
    canvas[:, :new_w] = img
    x = canvas.astype(np.float32) / 255.0
    return x[None] if x.ndim == 2 else x.transpose(2, 0, 1)


# ---------------------------------------------------------------- metriklər
def align(ref: str | List[str], hyp: str | List[str]) -> Tuple[int, List[Tuple[str, Any, Any]]]:
    """Levenshtein məsafəsi və əməliyyatlar: [(tip, ref_simvol, hyp_simvol)], tip: ok/sub/del/ins."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)

    i, j, ops = n, m, []
    while i > 0 or j > 0:
        cost = 0 if (i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]) else 1
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + cost:
            ops.append(("ok" if cost == 0 else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return int(d[n, m]), ops[::-1]


def cer_wer(refs: List[str], hyps: List[str]) -> Tuple[float, float, List[Tuple[str, str, str]]]:
    """Korpus səviyyəsində CER, WER və bütün simvol əməliyyatlarını qaytarır."""
    ce, cn, we, wn = 0, 0, 0, 0
    all_ops: List[Tuple[str, str, str]] = []
    for r, h in zip(refs, hyps):
        e, ops = align(r, h)
        ce += e
        cn += len(r)
        all_ops.extend(ops)

        e2, _ = align(r.split(), h.split())
        we += e2
        wn += len(r.split())
    return ce / max(cn, 1), we / max(wn, 1), all_ops


def load_labels(data_dir: str | Path) -> List[Tuple[Path, str]]:
    """data_dir/labels.tsv faylını oxuyur: 'images/000001.png<TAB>mətn'."""
    data_dir = Path(data_dir)
    labels_file = data_dir / "labels.tsv"
    if not labels_file.exists():
        raise FileNotFoundError(f"labels.tsv tapılmadı: {labels_file}")

    items: List[Tuple[Path, str]] = []
    with open(labels_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            fn = parts[0]
            label = parts[1] if len(parts) > 1 else ""
            items.append((data_dir / fn, label))
    return items


# ---------------------------------------------------------------- Model Yükləmə və İnferens
def load_model_checkpoint(ckpt_path: str | Path, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[str, Any], Vocab, torch.device]:
    """Checkpoint faylından CRNN modelini, konfiqurasiyasını və lüğətini yükləyir."""
    from model import CRNN
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint faylı tapılmadı: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("cfg", DEFAULT_CFG)
    chars = ck.get("chars", CHARS)
    vocab = Vocab(chars)

    in_ch = 1 if cfg.get("gray", True) else 3
    model = CRNN(in_ch=in_ch, num_classes=vocab.size).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, vocab, device


def infer_single_crop(model: nn.Module, cfg: Dict[str, Any], vocab: Vocab, img: np.ndarray, device: torch.device) -> str:
    """Tək kəsilmiş təsvir üzərində model inferensini icra edir."""
    x = preprocess(img, cfg)
    x_tensor = torch.from_numpy(x)[None].to(device)
    with torch.no_grad():
        out = model(x_tensor)
        best_ids = out.argmax(2).T.cpu().numpy()[0]
        return vocab.decode_greedy(best_ids)
