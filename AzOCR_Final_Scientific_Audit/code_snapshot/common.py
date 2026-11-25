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


def split_line_into_chunks(crop_bgr: np.ndarray, n_chunks: int) -> List[np.ndarray]:
    """Uzun sətirləri sözarası ağ boşluqlardan (kölgəyə davamlı 0-mürəkkəb aralıqlarından) optimal hissələrə bölür."""
    if n_chunks <= 1:
        return [crop_bgr]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    h, w = gray.shape[:2]
    if w < 60:
        return [crop_bgr]
    sigma = min(25, max(5, h // 2))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    ink = ((bg.astype(np.int16) - gray.astype(np.int16)) > 18).astype(np.float32)
    proj = ink.sum(axis=0)

    # Sözarası tam boş (proj == 0) zolaqları tapırıq (hərfarası 4-5px kerning deyil, həqiqi sözarası boşluq)
    min_gap_w = max(6, int(h * 0.21))
    zero_runs = []
    in_z = False
    z_start = 0
    for x_idx, val in enumerate(proj):
        if val == 0 and not in_z:
            in_z = True
            z_start = x_idx
        elif val > 0 and in_z:
            in_z = False
            if (x_idx - z_start) >= min_gap_w and z_start > 10 and x_idx < (w - 10):
                zero_runs.append((z_start, x_idx, (z_start + x_idx) // 2))

    ksize = max(5, min(15, h // 3))
    smooth_proj = np.convolve(proj, np.ones(ksize) / ksize, mode="same")
    nominal_w = w / n_chunks
    search_radius = int(nominal_w * 0.38)
    splits = []
    for k in range(1, n_chunks):
        ideal_x = int(k * nominal_w)
        x0 = max(15, ideal_x - search_radius)
        x1 = min(w - 15, ideal_x + search_radius)
        if x1 <= x0:
            splits.append(ideal_x)
            continue
        # Əvvəlcə axtarış pəncərəsi daxilindəki həqiqi sözarası boşluqları yoxla
        cand_gaps = [mid for (gs, ge, mid) in zero_runs if x0 <= mid <= x1]
        if cand_gaps:
            best_x = min(cand_gaps, key=lambda m: abs(m - ideal_x))
        else:
            window = smooth_proj[x0:x1]
            min_val = window.min()
            candidates = np.where(window == min_val)[0] + x0
            best_x = int(candidates[np.argmin(np.abs(candidates - ideal_x))])
        splits.append(int(best_x))
    chunks = []
    prev_x = 0
    for sx in splits:
        chunks.append(crop_bgr[:, prev_x:sx])
        prev_x = sx
    chunks.append(crop_bgr[:, prev_x:])
    return chunks


def robust_trim_chunk(chunk_bgr: np.ndarray) -> np.ndarray:
    """Kənar kölgə zolaqlarını və xırda ləkələri təmizləyib mürəkkəb sərhədinə sıxır."""
    gray = cv2.cvtColor(chunk_bgr, cv2.COLOR_BGR2GRAY) if chunk_bgr.ndim == 3 else chunk_bgr
    h_chunk, w_chunk = gray.shape
    if h_chunk < 5 or w_chunk < 5:
        return chunk_bgr
    # İşıqlanmaya davamlı lokal fon fərqi
    sigma = min(25, max(5, h_chunk // 2))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    ink = ((bg.astype(np.int16) - gray.astype(np.int16)) > 18).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_x = w_chunk
    max_x = 0
    found = False
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        touches_border = (x == 0) or (x + w >= w_chunk)
        if area < 8 or (touches_border and h >= 0.70 * h_chunk):
            continue
        found = True
        if x < min_x:
            min_x = x
        if x + w > max_x:
            max_x = x + w
    if not found:
        return chunk_bgr
    x0 = max(0, min_x - 4)
    x1 = min(w_chunk, max_x + 4)
    return chunk_bgr[:, x0:x1]


def infer_line_smart(model: nn.Module, cfg: Dict[str, Any], vocab: Vocab, line_bgr: np.ndarray, device: torch.device) -> str:
    """Uzun sətirləri avtomatik hissələrə bölərək (squeeze olmadan) yüksək dəqiqliklə oxuyur."""
    h_c, w_c = line_bgr.shape[:2]
    if h_c == 0 or w_c == 0:
        return ""
    target_h = cfg.get("height", 32)
    target_w = cfg.get("width", 512)
    if target_w <= 128:
        # Söz modeli üçün birbaşa oxu
        return infer_single_crop(model, cfg, vocab, robust_trim_chunk(line_bgr), device)
    import math
    squeeze_ratio = (w_c * target_h) / (h_c * target_w)
    n_chunks = max(1, math.ceil(squeeze_ratio)) if squeeze_ratio > 1.05 else 1
    raw_chunks = split_line_into_chunks(line_bgr, n_chunks)
    preds = []
    for ch in raw_chunks:
        ch_trimmed = robust_trim_chunk(ch)
        p = infer_single_crop(model, cfg, vocab, ch_trimmed, device).strip()
        if p:
            preds.append(p)
    return " ".join(preds)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Kölgəyə davamlı lokal-adaptiv mürəkkəb maskası (0/1)."""
    h, w = gray.shape
    bs = int(max(15, (min(h, w) // 40) | 1))
    return cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, bs, 12)


def estimate_skew_angle(gray: np.ndarray, max_angle: float = 8.0, step: float = 0.25) -> float:
    """Sətir proyeksiyasının dispersiyasını maksimallaşdıran maillik bucağı (dərəcə)."""
    h, w = gray.shape
    scale = 700.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ink = _ink_mask(small).astype(np.float32)
    ink = cv2.dilate(ink, np.ones((1, 5), np.uint8))
    hs, ws = ink.shape
    best_a, best_v = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        M = cv2.getRotationMatrix2D((ws / 2, hs / 2), float(a), 1.0)
        r = cv2.warpAffine(ink, M, (ws, hs), flags=cv2.INTER_NEAREST)
        v = float(np.var(r.sum(axis=1)))
        if v > best_v:
            best_v, best_a = v, float(a)
    return best_a


def segment_lines_opencv(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Tam səhifə şəkli üçün: mailliyi düzəldir (deskew), sətirləri ayırır, birləşmiş sətirləri bölür."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    H, W = gray.shape
    # Əgər təsvir artıq tək sətir nisbətindədirsə, bölməyə ehtiyac yoxdur
    if H < 90 or (W / max(1, H)) > 4.2:
        return [img_bgr]

    ang = estimate_skew_angle(gray)
    if abs(ang) > 0.15:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        fill = int(np.median(gray))
        img_bgr = cv2.warpAffine(img_bgr, M, (W, H), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(fill, fill, fill))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr

    ink = _ink_mask(gray).astype(np.float32)
    margin_x = max(2, int(W * 0.03))
    row_proj = ink[:, margin_x:W - margin_x].sum(axis=1)
    row_smooth = np.convolve(row_proj, np.ones(5) / 5.0, mode="same")
    ref = float(np.percentile(row_smooth, 95))
    thresh = max(3.0, ref * 0.14)
    active = row_smooth > thresh

    bands = []
    in_band, start_y = False, 0
    for y, val in enumerate(active):
        if val and not in_band:
            in_band, start_y = True, y
        elif not val and in_band:
            in_band = False
            if (y - start_y) >= 10:
                bands.append((start_y, y))
    if in_band and (H - start_y) >= 10:
        bands.append((start_y, H))
    if len(bands) <= 1:
        return [img_bgr]

    # Birləşmiş (çox hündür) zolaqları proyeksiya dərələrindən bölmək
    heights = np.array([b[1] - b[0] for b in bands])
    med_h = float(np.median(heights))
    final_bands = []
    for y0, y1 in bands:
        h_b = y1 - y0
        if h_b > 1.7 * med_h:
            n = int(round(h_b / med_h))
            seg = row_smooth[y0:y1]
            cuts = [0]
            for k in range(1, n):
                ideal = int(k * h_b / n)
                lo = max(cuts[-1] + 8, ideal - int(med_h * 0.45))
                hi = min(h_b - 8, ideal + int(med_h * 0.45))
                if hi <= lo:
                    continue
                cuts.append(lo + int(np.argmin(seg[lo:hi])))
            cuts.append(h_b)
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a >= 10:
                    final_bands.append((y0 + a, y0 + b))
        else:
            final_bands.append((y0, y1))

    crops = []
    for y0, y1 in final_bands:
        pad = max(3, int((y1 - y0) * 0.14))
        cy0, cy1 = max(0, y0 - pad), min(H, y1 + pad)
        strip = img_bgr[cy0:cy1, :]
        # Kağız sərhədindən kənardakı fonu (masa, kölgə) at: ən uzun parlaq zolaq = kağız
        sg = gray[cy0:cy1, :]
        col = cv2.GaussianBlur(sg.mean(axis=0).reshape(1, -1).astype(np.float32), (0, 0), sigmaX=6).ravel()
        paper = col > 0.78 * float(np.percentile(col, 90))
        best_len, best_s, cur_s = 0, 0, None
        for x, v in enumerate(list(paper) + [False]):
            if v and cur_s is None:
                cur_s = x
            elif not v and cur_s is not None:
                if x - cur_s > best_len:
                    best_len, best_s = x - cur_s, cur_s
                cur_s = None
        if best_len > 0.3 * W:
            strip = strip[:, best_s + 4:best_s + best_len - 4]
        elif best_len < 0.1 * W:
            continue  # kağız deyil (fon zolağı)
        crops.append(strip)
    return crops
