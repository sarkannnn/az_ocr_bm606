# AzOCR-V2 — RandomPadding Technical Specification (Task T05)

**Source Files:** `AzOCR_Final_Scientific_Audit/code_snapshot/dataset.py` (lines 179, 207–216, 236–237), `Exp_RandomPadding_GroupD_seed{42,43,44}/config.yaml`

| Parameter | Verified Value | Code Evidence |
|---|---|---|
| **Minimum Padding** | `0` px per side | `self.rng.integers(0, self.random_padding_max + 1)` (`dataset.py:209-212`) |
| **Maximum Padding** | `12` px per side | `random_padding: 12` in `Exp_RandomPadding_GroupD_seed*/config.yaml:17` |
| **Horizontal Padding** | Independent `p_left`, `p_right` $\in \{0, 1, \dots, 12\}$ | `dataset.py:211-212` |
| **Vertical Padding** | Independent `p_top`, `p_btm` $\in \{0, 1, \dots, 12\}$ | `dataset.py:209-210` |
| **Random Sampling Distribution** | Discrete Uniform over `{0, 1, ..., 12}` independently per side | `np.random.Generator.integers(0, 13)` |
| **Application Probability** | `1.0` (applied to 100% of training samples when `train=True` and `random_padding_max > 0`; each side independently draws 0 px with $p = 1/13 \approx 7.69\%$) | `if self.train and self.random_padding_max > 0:` (`dataset.py:208`) |
| **Padding Color** | Pure white `[255, 255, 255]` (BGR) | `cv2.copyMakeBorder(..., cv2.BORDER_CONSTANT, value=[255, 255, 255])` (`dataset.py:213-216`) |
| **Resize Order** | **1st:** Random border padding -> **2nd:** `augment_photo` -> **3rd:** `preprocess` resize to `H=32` | `dataset.py:208-237` |
| **Trim Behavior** | None during training (`robust_trim_chunk` is applied only at inference inside `infer_line_smart`) | `common.py:268-322` |
| **Fixed-Width Interaction** | `dynamic_width: false`; padded+augmented image is proportionally resized to `H=32` (clamped at `new_w <= 512`) and right-padded with white to `W=512` (`pad_alignment='left'`) | `dataset.py:235-237`, `common.py:103-111` |
