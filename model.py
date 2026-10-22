"""CRNN: CNN (görüntü xüsusiyyətləri) -> Bi-LSTM (hərf ardıcıllığı) -> CTC üçün log-ehtimallar."""
import torch.nn as nn
import torch.nn.functional as F


def _block(i, o, pool=None):
    layers = [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return layers


class CRNN(nn.Module):
    """Giriş (B,C,32,W) -> çıxış (T=W/4, B, num_classes) log_softmax."""

    def __init__(self, in_ch, num_classes, hidden=256, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            *_block(in_ch, 64, (2, 2)),     # 16 x W/2
            *_block(64, 128, (2, 2)),       # 8  x W/4
            *_block(128, 256),
            *_block(256, 256, (2, 1)),      # 4  x W/4
            *_block(256, 512),
            *_block(512, 512, (2, 1)),      # 2  x W/4
            nn.Conv2d(512, 512, (2, 1)),    # 1  x W/4
            nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2)          # B,512,T
        f = f.permute(2, 0, 1)              # T,B,512
        out, _ = self.rnn(f)
        return F.log_softmax(self.fc(self.drop(out)), dim=2)
