# Table 3: Model Architecture and Training Hyperparameters (V1 vs. V2)

| Parameter / Configuration | Baseline CRNN (V1) | Photometric-Enhanced CRNN (V2) | Rationale / Reference |
| :--- | :--- | :--- | :--- |
| **Input Shape (Word Model)** | $1 	imes 32 	imes 128$ (Grayscale) | $1 	imes 32 	imes 128$ (Grayscale) | `common.py:PRESETS` |
| **Input Shape (Line Model)** | $1 	imes 32 	imes 512$ (Grayscale) | $1 	imes 32 	imes 512$ (Grayscale) | `common.py:PRESETS` |
| **CNN Backbone Architecture** | 7 Conv layers ($3\times 3$), BatchNorm, ReLU | 7 Conv layers ($3\times 3$), BatchNorm, ReLU | `model.py:CRNN` |
| **CNN Downsampling Factor** | Vertical: $32 \rightarrow 1$ (32x), Horiz: $W \rightarrow W/4$ | Vertical: $32 \rightarrow 1$ (32x), Horiz: $W \rightarrow W/4$ | `model.py:26-36` |
| **Recurrent Layers** | 2-layer BiLSTM ($h=256$, dropout=0.2) | 2-layer BiLSTM ($h=256$, dropout=0.2) | `model.py:37-43` |
| **Total Parameters** | 8,225,815 parameters (~8.23M) | 8,225,815 parameters (~8.23M) | `logs/line_v2_train.log:96` |
| **Vocabulary Size** | 87 (86 chars + 1 CTC blank at index 0) | 87 (86 chars + 1 CTC blank at index 0) | `common.py:Vocab` |
| **Optimizer** | Adam (lr = $1\times 10^{-3}$, weight_decay = $1\times 10^{-5}$) | Adam (lr = $1\times 10^{-3}$, weight_decay = $1\times 10^{-5}$) | `train.py:80`, `train_v2.py:246` |
| **Batch Size & Device** | 64, CPU-only execution | 64, CPU-only execution | Intel i7 / Windows CPU execution |
| **Patience (Early Stopping)** | 5 epochs without val loss improvement | 5 epochs without val loss improvement | `train_v2.py:215, 281` |
| **Training Rotation Range** | $\pm 5.0^\circ$ ($p=0.5$) | $\pm 5.0^\circ$ ($p=0.5$) | `train_v2.py:AUG_CONFIG` |
| **Perspective Jitter** | $p=0.3$ ($w\times 0.02, h\times 0.08$) | $p=0.3$ ($w\times 0.02, h\times 0.08$) | `train_v2.py:AUG_CONFIG` |
| **Illumination Gradient & Shadow** | None (Uniform pure white) | Gradient 0.7–1.0 ($p=0.5$) + soft shadow band | `train_v2.py:AUG_CONFIG` |
| **Brightness / Contrast Jitter**| None | $\alpha \in [0.6, 1.1]$, $\beta \in [-40, +20]$ ($p=0.6$) | `train_v2.py:AUG_CONFIG` |
| **Gaussian Blur** | Blur $k \in \{3, 5\}$ ($p=0.3$) | $\sigma \in [0.5, 2.0]$ ($p=0.6$) | `train_v2.py:AUG_CONFIG` |
| **Motion Blur** | None | Kernel 3–7 px, random angle ($p=0.15$) | `train_v2.py:AUG_CONFIG` |
| **Sensor Gaussian Noise** | None | Noise $\sigma \in [3.0, 12.0]$ ($p=0.6$) | `train_v2.py:AUG_CONFIG` |
| **Salt & Pepper Noise** | $p=0.3$ (0.5% pixels) | $p=0.2$ (0.5% pixels) | `train_v2.py:AUG_CONFIG` |
| **Downscaling Resolution Loss** | None | Factor $0.5\times - 0.9\times$ ($p=0.4$) | `train_v2.py:AUG_CONFIG` |
| **JPEG Compression Artifacts** | None | Quality factor $Q \in [40, 90]$ ($p=0.4$) | `train_v2.py:AUG_CONFIG` |
| **Contrast Min-Max Stretching** | None | Per-crop normalization ($p=0.5$) | `train_v2.py:AUG_CONFIG` |

Source: model.py:20-54, train.py:21-82, train_v2.py:26-48, common.py:12-25
