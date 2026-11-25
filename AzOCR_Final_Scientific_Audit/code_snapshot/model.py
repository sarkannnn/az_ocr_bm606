"""CRNN memarlığı: CNN (xüsusiyyət çıxarışı) -> Bi-LSTM (ardıcıllıq modelləşdirməsi) -> CTC log-ehtimallar."""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(in_c: int, out_c: int, pool: Optional[Tuple[int, int]] = None) -> List[nn.Module]:
    """Konvolyusiya + BatchNorm + ReLU bloku, istəyə bağlı MaxPool2d ilə."""
    layers: List[nn.Module] = [
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool is not None:
        layers.append(nn.MaxPool2d(kernel_size=pool))
    return layers


class CRNN(nn.Module):
    """Giriş (B, C, 32, W) -> Çıxış (T=W/4, B, num_classes) log_softmax paylanması."""

    def __init__(self, in_ch: int, num_classes: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        # CNN xüsusiyyət çıxarışı: Hündürlüyü 32 -> 1, Eni W -> W/4 azaldır
        self.cnn = nn.Sequential(
            *_block(in_ch, 64, (2, 2)),     # 16 x W/2
            *_block(64, 128, (2, 2)),       # 8  x W/4
            *_block(128, 256),
            *_block(256, 256, (2, 1)),      # 4  x W/4
            *_block(256, 512),
            *_block(512, 512, (2, 1)),      # 2  x W/4
            nn.Conv2d(512, 512, kernel_size=(2, 1)),    # 1  x W/4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Giriş x: (B, C, 32, W) -> Çıxış: (T, B, num_classes) log-probabilities."""
        f = self.cnn(x).squeeze(2)          # (B, 512, T)
        f = f.permute(2, 0, 1)              # (T, B, 512)
        out, _ = self.rnn(f)                # (T, B, 2*hidden)
        return F.log_softmax(self.fc(self.drop(out)), dim=2)

