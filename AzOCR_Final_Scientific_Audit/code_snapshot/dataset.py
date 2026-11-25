"""OCR Dataset & Data Augmentation Pipeline for GPU Training.

Supports:
- Augmentation modes: 'none', 'basic' (V1), 'photo' (V2 Realistic Phone)
- Random Border Padding augmentation (for padding robustness)
- Variable-width batching with dynamic canvas collation
"""

from typing import List, Tuple, Optional, Callable, Dict, Any
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import load_labels, preprocess, Vocab


AUG_CONFIG = {
    "rotation_prob": 0.5,
    "rotation_range": (-5.0, 5.0),
    "perspective_prob": 0.3,
    "background_tint_prob": 0.5,
    "gradient_range": (0.7, 1.0),
    "brightness_contrast_prob": 0.6,
    "alpha_range": (0.6, 1.1),
    "beta_range": (-40.0, 20.0),
    "gaussian_blur_prob": 0.6,
    "gaussian_blur_sigma": (0.5, 2.0),
    "motion_blur_prob": 0.15,
    "motion_blur_kernel": (3, 7),
    "sensor_noise_prob": 0.6,
    "sensor_noise_sigma": (3.0, 12.0),
    "salt_pepper_prob": 0.2,
    "downscale_prob": 0.4,
    "downscale_factor": (0.5, 0.9),
    "jpeg_prob": 0.4,
    "jpeg_quality": (40, 90),
    "contrast_stretch_prob": 0.5,
}


def augment_basic(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """V1 Basic Augmentations: ±5° rotation, Gaussian blur, salt-pepper, perspective."""
    h, w = img.shape[:2]
    # Rotation +-5 deg
    if rng.random() < 0.5:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-5.0, 5.0)), 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # Gaussian blur
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    # Salt and pepper
    if rng.random() < 0.3:
        img = img.copy()
        m = rng.random(img.shape[:2])
        img[m < 0.005] = 0
        img[m > 0.995] = 255
    # Perspective jitter
    if rng.random() < 0.3:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jit = rng.uniform(-1, 1, (4, 2)).astype(np.float32) * np.array([w * 0.02, h * 0.08], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, (src + jit).astype(np.float32))
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def augment_photo(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """V2 Realistic Phone-Camera Augmentations."""
    h, w = img.shape[:2]

    # 1. Rotation +-5 deg (p: 0.5)
    if rng.random() < AUG_CONFIG["rotation_prob"]:
        deg = float(rng.uniform(*AUG_CONFIG["rotation_range"]))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Perspective jitter (p: 0.3)
    if rng.random() < AUG_CONFIG["perspective_prob"]:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jit = rng.uniform(-1, 1, (4, 2)).astype(np.float32) * np.array([w * 0.02, h * 0.08], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, (src + jit).astype(np.float32))
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 3. Background illumination gradient & soft shadow band (p: 0.5)
    if rng.random() < AUG_CONFIG["background_tint_prob"]:
        img_f = img.astype(np.float32)
        gx = np.linspace(-1, 1, w, dtype=np.float32)
        gy = np.linspace(-1, 1, h, dtype=np.float32)
        X, Y = np.meshgrid(gx, gy)
        low_val = rng.uniform(*AUG_CONFIG["gradient_range"])
        angle = rng.uniform(0, 2 * np.pi)
        proj = np.cos(angle) * X + np.sin(angle) * Y
        p_min, p_max = proj.min(), proj.max()
        proj_norm = (proj - p_min) / max(p_max - p_min, 1e-5)
        grad = 1.0 - (1.0 - low_val) * proj_norm
        if rng.random() < 0.5:
            d0 = rng.uniform(-0.5, 0.5)
            dist = proj - d0
            bw = rng.uniform(0.2, 0.5)
            band = 1.0 - rng.uniform(0.12, 0.28) * np.exp(-(dist / bw) ** 2)
            grad *= band
        img = np.clip(img_f * grad[:, :, None], 0, 255).astype(np.uint8)

    # 4. Brightness / Contrast (p: 0.6)
    if rng.random() < AUG_CONFIG["brightness_contrast_prob"]:
        alpha = rng.uniform(*AUG_CONFIG["alpha_range"])
        beta = rng.uniform(*AUG_CONFIG["beta_range"])
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 5. Gaussian blur (p: 0.6)
    if rng.random() < AUG_CONFIG["gaussian_blur_prob"]:
        sigma = float(rng.uniform(*AUG_CONFIG["gaussian_blur_sigma"]))
        img = cv2.GaussianBlur(img, (0, 0), sigma)

    # 6. Motion blur (p: 0.15)
    if rng.random() < AUG_CONFIG["motion_blur_prob"]:
        ksize = int(rng.integers(AUG_CONFIG["motion_blur_kernel"][0], AUG_CONFIG["motion_blur_kernel"][1] + 1))
        if ksize % 2 == 0:
            ksize += 1
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        m_angle = rng.uniform(0, np.pi)
        cx, cy = ksize // 2, ksize // 2
        for step in range(-ksize // 2, ksize // 2 + 1):
            kx = int(round(cx + step * np.cos(m_angle)))
            ky = int(round(cy + step * np.sin(m_angle)))
            if 0 <= kx < ksize and 0 <= ky < ksize:
                kernel[ky, kx] = 1.0
        if kernel.sum() > 0:
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

    # 7. Gaussian sensor noise (p: 0.6)
    if rng.random() < AUG_CONFIG["sensor_noise_prob"]:
        n_sigma = float(rng.uniform(*AUG_CONFIG["sensor_noise_sigma"]))
        noise = rng.normal(0, n_sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 8. Salt-and-pepper noise (p: 0.2)
    if rng.random() < AUG_CONFIG["salt_pepper_prob"]:
        img = img.copy()
        m = rng.random(img.shape[:2])
        img[m < 0.005] = 0
        img[m > 0.995] = 255

    # 9. Downscale & upscale (p: 0.4)
    if rng.random() < AUG_CONFIG["downscale_prob"]:
        scale = float(rng.uniform(*AUG_CONFIG["downscale_factor"]))
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 10. JPEG compression artifacts (p: 0.4)
    if rng.random() < AUG_CONFIG["jpeg_prob"]:
        q = int(rng.uniform(*AUG_CONFIG["jpeg_quality"]))
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 11. Contrast stretch (p: 0.5)
    if rng.random() < AUG_CONFIG["contrast_stretch_prob"]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR)

    return img


class OCRDataset(Dataset):
    """PyTorch Dataset for line-level OCR training and validation."""

    def __init__(
        self,
        data_dir: str | Path,
        cfg: Dict[str, Any],
        vocab: Vocab,
        train: bool = False,
        aug_mode: str = "none",
        random_padding_max: int = 0,
        pad_alignment: str = "left",
        dynamic_width: bool = False,
        seed: int = 42
    ):
        self.items: List[Tuple[Path, str]] = load_labels(data_dir)
        self.cfg: Dict[str, Any] = cfg
        self.vocab: Vocab = vocab
        self.train: bool = train
        self.aug_mode: str = aug_mode.lower()
        self.random_padding_max: int = random_padding_max
        self.pad_alignment: str = pad_alignment
        self.dynamic_width: bool = dynamic_width
        self.seed: int = seed
        self.rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed + idx)

        path, label = self.items[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")

        # 1. Random Border Padding Augmentation (if enabled)
        if self.train and self.random_padding_max > 0:
            p_top = int(self.rng.integers(0, self.random_padding_max + 1))
            p_btm = int(self.rng.integers(0, self.random_padding_max + 1))
            p_left = int(self.rng.integers(0, self.random_padding_max + 1))
            p_right = int(self.rng.integers(0, self.random_padding_max + 1))
            img = cv2.copyMakeBorder(
                img, p_top, p_btm, p_left, p_right,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

        # 2. Main Data Augmentation
        if self.train:
            if self.aug_mode == "basic":
                img = augment_basic(img, self.rng)
            elif self.aug_mode in ("photo", "v2"):
                img = augment_photo(img, self.rng)

        # 3. Target width and alignment
        if self.dynamic_width:
            # Aspect ratio resize to H=32, calculate unpadded width
            h, w = img.shape[:2]
            H = self.cfg.get("height", 32)
            calc_w = max(16, round(w * H / max(h, 1)))
            # Keep multiple of 4 for CNN downsampling
            calc_w = ((calc_w + 3) // 4) * 4
            x = preprocess(img, self.cfg, target_width=calc_w, pad_alignment="left")
            actual_w = calc_w
        else:
            x = preprocess(img, self.cfg, pad_alignment=self.pad_alignment)
            actual_w = self.cfg.get("width", 512)

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.tensor(self.vocab.encode(label), dtype=torch.long)
        return x_tensor, y_tensor, label, actual_w


def collate_fn_fixed(batch):
    """Standard collate function for fixed-width tensors."""
    xs, ys, labels, _ = zip(*batch)
    x_tensor = torch.stack(xs)
    y_tensor = torch.cat(ys)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return x_tensor, y_tensor, target_lengths, list(labels)


def collate_fn_dynamic(batch):
    """Dynamic-width collate function padding to the max width within the current batch."""
    xs, ys, labels, widths = zip(*batch)
    max_w = max(w for w in widths)
    # Ensure multiple of 4
    max_w = max(32, ((max_w + 3) // 4) * 4)

    padded_xs = []
    for x in xs:
        c, h, w = x.shape
        if w < max_w:
            pad_canvas = torch.ones((c, h, max_w), dtype=x.dtype)
            pad_canvas[:, :, :w] = x
            padded_xs.append(pad_canvas)
        else:
            padded_xs.append(x[:, :, :max_w])

    x_tensor = torch.stack(padded_xs)
    y_tensor = torch.cat(ys)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return x_tensor, y_tensor, target_lengths, list(labels)
