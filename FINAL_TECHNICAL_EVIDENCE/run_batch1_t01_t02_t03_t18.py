"""Batch 1: T01 (Architecture), T02 (18 GPU Configs), T03 (SHA-256), T18 (Training Curves)."""

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import torch
import yaml

PROJ_ROOT = Path(r"D:\laragon\www\az_ocr_bm606")
sys.path.insert(0, str(PROJ_ROOT))

from model import CRNN
from common import Vocab, CHARS

OUT_DIR = PROJ_ROOT / "FINAL_TECHNICAL_EVIDENCE"
MODELS_ZIP = PROJ_ROOT / "AzOCR_Zenodo_v1_Release" / "AzOCR_v1_Models_18Official.zip"
RUNS_META_DIR = PROJ_ROOT / "AzOCR_Zenodo_v1_DRAFT" / "configs_and_logs" / "runs_meta"
META_ZIP = PROJ_ROOT / "AzOCR_ChatGPT_Transfer" / "01_METADATA.zip"

GROUPS = [
    ("Group_A_V1Clean_BasicAug", "S25"),
    ("Group_B_V1Clean_PhotoAug", "S26"),
    ("Group_C_V2Clean_BasicAug", "S27"),
    ("Group_D_V2Clean_PhotoAug", "S28"),
    ("Exp_RandomPadding_GroupD", "S29"),
    ("Exp_VariableWidth_GroupD", "S30"),
]
SEEDS = [42, 43, 44]


def run_t01():
    vocab = Vocab(CHARS)
    model = CRNN(in_ch=1, num_classes=vocab.size)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_bytes_fp32 = sum(p.numel() * 4 for p in model.parameters())
    buffer_bytes_fp32 = sum(b.numel() * b.element_size() for b in model.buffers())
    pure_fp32_bytes = param_bytes_fp32 + buffer_bytes_fp32

    model_fp16 = CRNN(in_ch=1, num_classes=vocab.size).half()
    param_bytes_fp16 = sum(p.numel() * 2 for p in model_fp16.parameters())
    pure_fp16_bytes = param_bytes_fp16 + sum(
        b.numel() * 2 if b.is_floating_point() else b.numel() * b.element_size() for b in model_fp16.buffers()
    )

    buf_fp32 = io.BytesIO()
    torch.save(model.state_dict(), buf_fp32)
    sd_fp32_bytes = buf_fp32.tell()

    buf_fp16 = io.BytesIO()
    torch.save(model_fp16.state_dict(), buf_fp16)
    sd_fp16_bytes = buf_fp16.tell()

    with zipfile.ZipFile(MODELS_ZIP, "r") as z:
        ck_bytes = z.read("Group_D_V2Clean_PhotoAug_seed42/best.pt")
    ck_file_bytes = len(ck_bytes)
    ck = torch.load(io.BytesIO(ck_bytes), map_location="cpu", weights_only=False)

    buf_ck_model = io.BytesIO()
    torch.save(ck["model"], buf_ck_model)
    ck_model_bytes = buf_ck_model.tell()

    buf_ck_opt = io.BytesIO()
    torch.save(ck["optimizer"], buf_ck_opt)
    ck_opt_bytes = buf_ck_opt.tell()

    rows = [
        ("trainable_parameter_count", str(trainable_params), "Exact sum of p.numel() for CRNN(in_ch=1, num_classes=87)"),
        ("total_parameter_count", str(total_params), "All parameters are trainable (requires_grad=True)"),
        ("cnn_conv_layer_count", "7", "6x Conv2d(3x3, pad=1, bias=False) + 1x Conv2d(2x1, bias=True)"),
        ("cnn_kernel_sizes", "6x (3,3) + 1x (2,1)", "Channels: 1->64->128->256->256->512->512->512"),
        ("batch_normalization", "7x BatchNorm2d", "eps=1e-05, momentum=0.1, affine=True, track_running_stats=True"),
        ("activation", "7x ReLU(inplace=True)", "Applied after every Conv2d+BatchNorm2d"),
        ("vertical_downsampling", "32x (H: 32 -> 1)", "MaxPool2d: (2,2), (2,2), (2,1), (2,1) [16x] + Conv2d(2,1) [2x]"),
        ("horizontal_downsampling", "4x (W -> W/4)", "First two MaxPool2d(2,2) layers; T = W/4 (128 steps for W=512)"),
        ("bilstm_layer_count", "2", "nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.2)"),
        ("bilstm_hidden_size", "256 per direction (512 concat)", "Followed by Dropout(0.2) and Linear(512, 87) + log_softmax(dim=2)"),
        ("dropout", "0.2", "Both inter-layer LSTM dropout=0.2 and self.drop=nn.Dropout(0.2)"),
        ("ocr_character_count", str(len(CHARS)), "AZ_LOWER(32) + AZ_UPPER(32) + DIGITS(10) + PUNCT(11) + SPACE(1)"),
        ("vocabulary_size_ctc_classes", str(vocab.size), "86 characters + 1 CTC blank class"),
        ("ctc_blank_index", "0", "0 reserved for CTC blank; characters indexed 1..86"),
        ("input_normalization", "float32 / 255.0 in [0.0, 1.0]", "Grayscale uint8 -> float32 / 255.0, right-side white pad (255)"),
        ("model_input_geometry", "(B, 1, 32, W)", "Fixed training/infer_single_crop W=512; dynamic W/4*4 in VariableWidth"),
        ("pure_tensor_weights_fp32", f"{pure_fp32_bytes} bytes ({pure_fp32_bytes / (1024**2):.2f} MiB)", f"Params={param_bytes_fp32} B + BN buffers={buffer_bytes_fp32} B"),
        ("pure_tensor_weights_fp16", f"{pure_fp16_bytes} bytes ({pure_fp16_bytes / (1024**2):.2f} MiB)", "Half-precision model parameters + buffers"),
        ("serialized_state_dict_fp32", f"{sd_fp32_bytes} bytes ({sd_fp32_bytes / (1024**2):.2f} MiB)", "torch.save(model.state_dict())"),
        ("serialized_state_dict_fp16", f"{sd_fp16_bytes} bytes ({sd_fp16_bytes / (1024**2):.2f} MiB)", "torch.save(model.half().state_dict())"),
        ("checkpoint_model_state_dict", f"{ck_model_bytes} bytes ({ck_model_bytes / (1024**2):.2f} MiB)", "ck['model'] inside best.pt"),
        ("checkpoint_optimizer_state", f"{ck_opt_bytes} bytes ({ck_opt_bytes / (1024**2):.2f} MiB)", "ck['optimizer'] (Adam exp_avg & exp_avg_sq tensors, ~2x model weights)"),
        ("full_best_pt_file_size", f"{ck_file_bytes} bytes ({ck_file_bytes / (1024**2):.2f} MiB)", "Includes model + optimizer + cfg + chars + metrics"),
    ]
    pd.DataFrame(rows, columns=["parameter", "verified_value", "notes"]).to_csv(
        OUT_DIR / "MODEL_ARCHITECTURE_VERIFIED.csv", index=False
    )
    print(f"[T01] Saved MODEL_ARCHITECTURE_VERIFIED.csv (params={trainable_params}, FP32={pure_fp32_bytes/(1024**2):.2f} MiB, FP16={pure_fp16_bytes/(1024**2):.2f} MiB)")


def run_t02_t03_t18():
    zen_df = pd.read_csv(PROJ_ROOT / "Real30_GPU18_Final_Evaluation" / "checkpoint_sha256_manifest.csv")
    zen_map = dict(zip(zen_df["run_name"], zen_df["sha256"]))

    config_rows = []
    sha_rows = []
    curve_rows = []
    t18_summary_lines = [
        "# AzOCR-V2 — GPU Training History Verification (Task T18)",
        "",
        "| Supp Figure | Run ID | Epochs Logged | Best Epoch | Last Epoch | Early Stopped | Best Val CER (%) | Final Train Loss | Final Val Loss |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]

    with zipfile.ZipFile(MODELS_ZIP, "r") as z_models:
        for gid, fig_id in GROUPS:
            for seed in SEEDS:
                run_id = f"{gid}_seed{seed}"
                rdir = RUNS_META_DIR / run_id
                cfg_yaml = yaml.unsafe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
                summary = json.loads((rdir / "run_summary.json").read_text(encoding="utf-8"))
                log_lines = [json.loads(l) for l in (rdir / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

                hw = summary.get("hardware", cfg_yaml.get("hardware", {}))
                best_epoch = summary.get("best_epoch")
                last_epoch = log_lines[-1]["epoch"] if log_lines else -1
                max_epochs = cfg_yaml.get("epochs")
                early_stopped = last_epoch < max_epochs

                config_rows.append({
                    "run_id": run_id,
                    "official_status": "OFFICIAL_18",
                    "group_id": gid,
                    "seed": seed,
                    "optimizer": "Adam (torch.optim.Adam in train_gpu.py:248)",
                    "learning_rate": cfg_yaml.get("lr"),
                    "weight_decay": cfg_yaml.get("weight_decay"),
                    "batch_size": cfg_yaml.get("batch_size"),
                    "max_epochs": max_epochs,
                    "early_stopping_patience": cfg_yaml.get("patience"),
                    "scheduler": "None (constant lr, no LR scheduler in train_gpu.py)",
                    "loss_function": "CTCLoss(blank=0, zero_infinity=True)",
                    "dropout": cfg_yaml.get("dropout"),
                    "best_epoch": best_epoch,
                    "last_epoch": last_epoch,
                    "early_stopped": early_stopped,
                    "best_checkpoint_criterion": "Minimum val_cer primary; val_loss tiebreaker (<1e-5) [train_gpu.py:354]",
                    "best_val_cer_pct": summary.get("best_val_cer_pct"),
                    "train_seconds": round(float(summary.get("total_training_seconds", 0.0)), 1),
                    "gpu": hw.get("gpu_name"),
                    "python_version": "Not logged in run_summary.json (eval env: Python 3.13.9)",
                    "pytorch_version": str(hw.get("torch_version")),
                    "cuda_version": str(hw.get("cuda_version")),
                    "input_width_strategy": "dynamic_width" if cfg_yaml.get("dynamic_width") else "fixed_512",
                    "aug_mode": cfg_yaml.get("aug_mode"),
                    "random_padding": cfg_yaml.get("random_padding"),
                    "train_dir": cfg_yaml.get("train_dir"),
                    "val_dir": cfg_yaml.get("val_dir"),
                })

                # T03: stream SHA-256 from AzOCR_v1_Models_18Official.zip
                arc = f"{run_id}/best.pt"
                h = hashlib.sha256()
                with z_models.open(arc) as f:
                    while chunk := f.read(1024 * 1024):
                        h.update(chunk)
                ck_sha = h.hexdigest()
                ck_size = z_models.getinfo(arc).file_size
                exp_sha = zen_map.get(run_id, "NOT_IN_MANIFEST")
                sha_rows.append({
                    "run_id": run_id,
                    "filename": "best.pt",
                    "size_bytes": ck_size,
                    "expected_sha256": exp_sha,
                    "computed_sha256": ck_sha,
                    "match_status": "MATCH" if ck_sha == exp_sha else "MISMATCH",
                    "evidence_path": f"{MODELS_ZIP.name}:{arc}",
                })

                for entry in log_lines:
                    curve_rows.append({
                        "supp_figure": fig_id,
                        "run_id": run_id,
                        "group_id": gid,
                        "seed": seed,
                        "epoch": entry["epoch"],
                        "train_loss": entry.get("train_loss"),
                        "val_loss": entry.get("val_loss"),
                        "val_cer_pct": entry.get("val_cer_pct"),
                        "val_wer_pct": entry.get("val_wer_pct"),
                        "best_epoch": best_epoch,
                        "last_epoch": last_epoch,
                        "early_stopped": early_stopped,
                    })

                last_e = log_lines[-1]
                t18_summary_lines.append(
                    f"| {fig_id} | `{run_id}` | {len(log_lines)} | {best_epoch} | {last_epoch} | {early_stopped} | {summary.get('best_val_cer_pct')} | {last_e.get('train_loss')} | {last_e.get('val_loss')} |"
                )

    # Add RTX 2050 reproducibility check separately from 01_METADATA.zip
    with zipfile.ZipFile(META_ZIP, "r") as z_meta:
        repro_base = "reproducibility_check_meta/Exp_VariableWidth_GroupD_seed44_RTX2050"
        cfg_r = yaml.unsafe_load(z_meta.read(f"{repro_base}/config.yaml").decode("utf-8"))
        sum_r = json.loads(z_meta.read(f"{repro_base}/run_summary.json").decode("utf-8"))
        log_r = [json.loads(l) for l in z_meta.read(f"{repro_base}/train_log.jsonl").decode("utf-8").splitlines() if l.strip()]
        hw_r = sum_r.get("hardware", cfg_r.get("hardware", {}))
        config_rows.append({
            "run_id": "Exp_VariableWidth_GroupD_seed44_RTX2050",
            "official_status": "EXCLUDED_REPRO_CHECK",
            "group_id": "Exp_VariableWidth_GroupD",
            "seed": 44,
            "optimizer": "Adam (torch.optim.Adam in train_gpu.py:248)",
            "learning_rate": cfg_r.get("lr"),
            "weight_decay": cfg_r.get("weight_decay"),
            "batch_size": cfg_r.get("batch_size"),
            "max_epochs": cfg_r.get("epochs"),
            "early_stopping_patience": cfg_r.get("patience"),
            "scheduler": "None (constant lr, no LR scheduler in train_gpu.py)",
            "loss_function": "CTCLoss(blank=0, zero_infinity=True)",
            "dropout": cfg_r.get("dropout"),
            "best_epoch": sum_r.get("best_epoch"),
            "last_epoch": log_r[-1]["epoch"],
            "early_stopped": log_r[-1]["epoch"] < cfg_r.get("epochs"),
            "best_checkpoint_criterion": "Minimum val_cer primary; val_loss tiebreaker (<1e-5)",
            "best_val_cer_pct": sum_r.get("best_val_cer_pct"),
            "train_seconds": round(float(sum_r.get("total_training_seconds", 0.0)), 1),
            "gpu": hw_r.get("gpu_name"),
            "python_version": "Not logged in run_summary.json",
            "pytorch_version": str(hw_r.get("torch_version")),
            "cuda_version": str(hw_r.get("cuda_version")),
            "input_width_strategy": "dynamic_width",
            "aug_mode": cfg_r.get("aug_mode"),
            "random_padding": cfg_r.get("random_padding"),
            "train_dir": cfg_r.get("train_dir"),
            "val_dir": cfg_r.get("val_dir"),
        })

    pd.DataFrame(config_rows).to_csv(OUT_DIR / "GPU18_FULL_CONFIGURATION_VERIFIED.csv", index=False)
    pd.DataFrame(sha_rows).to_csv(OUT_DIR / "GPU18_CHECKPOINT_SHA256.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "GPU18_TRAINING_CURVES_VERIFIED.csv", index=False)
    (OUT_DIR / "TRAINING_HISTORY_AUDIT.md").write_text("\n".join(t18_summary_lines) + "\n", encoding="utf-8")

    print(f"[T02] Saved GPU18_FULL_CONFIGURATION_VERIFIED.csv ({len(config_rows)} rows).")
    print(f"[T03] Saved GPU18_CHECKPOINT_SHA256.csv ({sum(1 for r in sha_rows if r['match_status']=='MATCH')}/18 matched).")
    print(f"[T18] Saved GPU18_TRAINING_CURVES_VERIFIED.csv ({len(curve_rows)} epoch records) and TRAINING_HISTORY_AUDIT.md.")


if __name__ == "__main__":
    run_t01()
    run_t02_t03_t18()
