"""Ortaq təlim utilitləri: Dataset, DataLoader collate və Epoch icrası."""
from typing import List, Tuple, Optional, Any, Callable
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from common import load_labels, preprocess, Vocab, cer_wer


class OCRDataset(Dataset):
    """OCR üçün PyTorch Dataset sinfi (söz və ya sətir təsvirləri üçün)."""

    def __init__(
        self,
        data_dir: str | Path,
        cfg: dict,
        vocab: Vocab,
        train: bool = False,
        augment_fn: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    ):
        self.items: List[Tuple[Path, str]] = load_labels(data_dir)
        self.cfg: dict = cfg
        self.vocab: Vocab = vocab
        self.train: bool = train
        self.augment_fn: Optional[Callable] = augment_fn
        self.rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.rng is None:
            self.rng = np.random.default_rng()
        path, label = self.items[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Təsvir faylı tapılmadı: {path}")

        if self.train and self.augment_fn is not None:
            img = self.augment_fn(img, self.rng)

        x = torch.from_numpy(preprocess(img, self.cfg))
        y = torch.tensor(self.vocab.encode(label), dtype=torch.long)
        return x, y, label


def collate(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Dəyişən uzunluqlu etiketləri CTC üçün paketləyir."""
    xs, ys, labels = zip(*batch)
    x_tensor = torch.stack(xs)
    y_tensor = torch.cat(ys)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return x_tensor, y_tensor, target_lengths, list(labels)


def run_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    vocab: Vocab,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_grad_norm: float = 5.0,
) -> Tuple[float, float]:
    """Bir təlim və ya validasiya epoxunu icra edir.
    
    Qaytarır: (orta_loss, cer)
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_samples = 0.0, 0
    refs, hyps = [], []

    with torch.set_grad_enabled(is_train):
        for x, y, y_lens, labels in loader:
            x = x.to(device)
            out = model(x)
            time_steps, batch_size, _ = out.shape
            input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)

            loss = criterion(out.float(), y, input_lengths, y_lens)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if not is_train:
                pred_ids = out.argmax(2).T.cpu().numpy()
                for p, ref in zip(pred_ids, labels):
                    hyps.append(vocab.decode_greedy(p))
                    refs.append(ref)

    cer = cer_wer(refs, hyps)[0] if refs else float("nan")
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, cer
