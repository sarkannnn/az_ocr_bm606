"""Standalone GPU Training Engine for AzOCR-V2.

Features:
- Full CUDA acceleration with Automatic Mixed Precision (AMP FP16)
- Gradient clipping & L2 weight decay
- Early stopping based on validation loss / CER
- Per-epoch JSONL logging (train_loss, val_loss, val_cer, val_wer, time)
- Best and Last checkpoint saving
- Run metadata & hardware tracking (GPU model, VRAM, CUDA version)
- Single-epoch smoke test mode (--smoke)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from common import Vocab, make_cfg, cer_wer
from model import CRNN
from dataset import OCRDataset, collate_fn_fixed, collate_fn_dynamic


def get_hardware_info(device: torch.device) -> Dict[str, Any]:
    """Retrieves hardware and environment specifications."""
    info = {
        "device_type": device.type,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_capability": f"{props.major}.{props.minor}",
            "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
            "cuda_version": torch.version.cuda
        })
    else:
        info["cpu_info"] = os.environ.get("PROCESSOR_IDENTIFIER", "CPU")
    return info


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    max_grad_norm: float = 5.0
) -> float:
    """Executes a single training epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = (scaler is not None) and (device.type == "cuda")

    for x, y, y_lens, _ in loader:
        x = x.to(device, non_blocking=True)
        batch_size = x.size(0)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(x)
                time_steps = out.size(0)
                input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)
                loss = criterion(out, y.to(device), input_lengths, y_lens.to(device))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            time_steps = out.size(0)
            input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)
            loss = criterion(out.float(), y.to(device), input_lengths, y_lens.to(device))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vocab: Vocab
) -> Tuple[float, float, float, int, int]:
    """Evaluates validation loss, CER, WER, and exact match count."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    refs = []
    hyps = []

    for x, y, y_lens, labels in loader:
        x = x.to(device, non_blocking=True)
        batch_size = x.size(0)

        out = model(x)
        time_steps = out.size(0)
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)
        loss = criterion(out.float(), y.to(device), input_lengths, y_lens.to(device))

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        pred_ids = out.argmax(dim=2).T.cpu().numpy()
        for p, ref in zip(pred_ids, labels):
            hyps.append(vocab.decode_greedy(p))
            refs.append(ref)

    val_loss = total_loss / max(total_samples, 1)
    cer, wer, _, _ = cer_wer(refs, hyps)
    exact_matches = sum(1 for r, h in zip(refs, hyps) if r == h)
    return val_loss, cer, wer, exact_matches, len(refs)


def main():
    parser = argparse.ArgumentParser(description="AzOCR-V2 GPU Training Script")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--aug-mode", choices=["none", "basic", "photo"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=None, help="Enable AMP mixed precision")
    parser.add_argument("--random-padding", type=int, default=None, help="Max pixels of random border padding")
    parser.add_argument("--dynamic-width", action="store_true", default=None, help="Enable dynamic batch width")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test on small subset")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from <out>/last.pt if it exists (otherwise start fresh)")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow training on CPU when CUDA is unavailable (default: refuse)")
    args = parser.parse_args()

    # Load configuration from YAML if provided
    cfg_data = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f)

    # CLI args override YAML config
    def get_val(key, default):
        cli_v = getattr(args, key.replace("-", "_"), None)
        if cli_v is not None:
            return cli_v
        return cfg_data.get(key, default)

    train_dir = get_val("train-dir", "data/synth_line_v2_clean/train")
    val_dir = get_val("val-dir", "data/synth_line_v2_clean/val")
    out_dir = Path(get_val("out", "runs/experiment_default"))
    aug_mode = get_val("aug-mode", "photo")
    epochs = 1 if args.smoke else int(get_val("epochs", 30))
    batch_size = int(get_val("batch-size", 64))
    lr = float(get_val("lr", 1e-3))
    weight_decay = float(get_val("weight-decay", 1e-5))
    dropout = float(get_val("dropout", 0.2))
    patience = int(get_val("patience", 7))
    workers = int(get_val("workers", 4))
    seed = int(get_val("seed", 42))
    use_amp = bool(get_val("amp", True))
    random_padding = int(get_val("random-padding", 0))
    dynamic_width = bool(get_val("dynamic-width", False))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available() and not args.allow_cpu:
        print("[!] XƏTA: CUDA GPU tapılmadı. Təlim CPU-da başladılmır.")
        print("    CUDA dəstəkli PyTorch quraşdırın və ya qəsdən CPU üçün --allow-cpu verin.")
        sys.exit(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw_info = get_hardware_info(device)

    print("=" * 80)
    print(f"AzOCR-V2 GPU Training | Run: {out_dir.name} | Seed: {seed}")
    print(f"Device: {device} | Info: {hw_info.get('gpu_name', 'CPU')} | AMP: {use_amp and device.type == 'cuda'}")
    print(f"Train Dir: {train_dir} | Aug Mode: {aug_mode} | Random Pad: {random_padding} px")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Dynamic Width: {dynamic_width}")
    print("=" * 80)

    ocr_cfg = make_cfg(mode="line", gray=True)
    vocab = Vocab()

    # Datasets
    train_dataset = OCRDataset(
        train_dir, ocr_cfg, vocab, train=True, aug_mode=aug_mode,
        random_padding_max=random_padding, dynamic_width=dynamic_width, seed=seed
    )
    val_dataset = OCRDataset(
        val_dir, ocr_cfg, vocab, train=False, aug_mode="none",
        random_padding_max=0, dynamic_width=dynamic_width, seed=seed
    )

    if args.smoke:
        print("[!] SMOKE TEST MODE ACTIVATED: Subsetting to 128 train and 64 val samples.")
        train_dataset = Subset(train_dataset, list(range(min(128, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(64, len(val_dataset)))))

    collate_fn = collate_fn_dynamic if dynamic_width else collate_fn_fixed
    loader_kwargs = {
        "num_workers": workers if os.name != "nt" or workers <= 2 else 2,  # safer multiprocessing on Windows
        "pin_memory": (device.type == "cuda"),
        "collate_fn": collate_fn
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # Model & Optimization
    model = CRNN(in_ch=1, num_classes=vocab.size, hidden=256, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # ---------------- Resume state ----------------
    best_val_cer = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1
    elapsed_before = 0.0
    last_ckpt_path = out_dir / "last.pt"
    log_file = out_dir / "train_log.jsonl"

    if args.resume and not args.smoke and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        best_val_cer = ckpt.get("best_val_cer", best_val_cer)
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        best_epoch = ckpt.get("best_epoch", best_epoch)
        bad_epochs = ckpt.get("bad_epochs", bad_epochs)
        elapsed_before = float(ckpt.get("elapsed_seconds", 0.0))
        start_epoch = int(ckpt["epoch"]) + 1
        rng = ckpt.get("rng")
        if rng:
            torch.set_rng_state(rng["torch"].cpu())
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        # Keep only log lines for epochs already covered by the checkpoint
        kept = []
        if log_file.exists():
            for line in log_file.read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(line).get("epoch", 0) < start_epoch:
                        kept.append(line)
                except json.JSONDecodeError:
                    pass
        log_file.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
        print(f"[RESUME] {last_ckpt_path} tapıldı -> epox {start_epoch}-dən davam edilir "
              f"(best epoch {best_epoch}, best CER {best_val_cer*100:.2f}%, bad_epochs {bad_epochs}).")
        log_fp = open(log_file, "a", encoding="utf-8")
    else:
        log_fp = open(log_file, "w", encoding="utf-8")

    # Save active configuration
    active_config = {
        "run_name": out_dir.name,
        "seed": seed,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "aug_mode": aug_mode,
        "random_padding": random_padding,
        "dynamic_width": dynamic_width,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "patience": patience,
        "hardware": hw_info
    }
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(active_config, f, default_flow_style=False)

    start_time = time.time()

    def atomic_save(obj, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    epoch = start_epoch - 1
    already_done = start_epoch > epochs or (start_epoch > 1 and bad_epochs >= patience)

    print("\n--- Training Progress ---")
    for epoch in range(start_epoch, epochs + 1):
        if already_done:
            epoch = start_epoch - 1
            break
        ep_t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_cer, val_wer, exact_cnt, total_cnt = evaluate(model, val_loader, criterion, device, vocab)
        ep_duration = time.time() - ep_t0

        exact_pct = round(exact_cnt / max(total_cnt, 1) * 100, 2)
        print(f"Epoch {epoch:2d}/{epochs:2d} ({ep_duration:4.1f}s) | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val CER: {val_cer*100:5.2f}% | Val WER: {val_wer*100:5.2f}% | "
              f"Exact: {exact_cnt}/{total_cnt} ({exact_pct}%)")

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_cer_pct": round(val_cer * 100, 3),
            "val_wer_pct": round(val_wer * 100, 3),
            "val_exact_count": exact_cnt,
            "val_total_count": total_cnt,
            "val_exact_pct": exact_pct,
            "epoch_seconds": round(ep_duration, 2)
        }
        # Check for best model (using val CER as primary metric, val loss as tiebreaker)
        is_best = (val_cer < best_val_cer) or (abs(val_cer - best_val_cer) < 1e-5 and val_loss < best_val_loss)
        elapsed_total = elapsed_before + (time.time() - start_time)
        if is_best:
            best_val_cer = val_cer
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            atomic_save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": ocr_cfg,
                "chars": vocab.chars,
                "val_cer": val_cer,
                "val_wer": val_wer,
                "val_loss": val_loss,
                "val_exact_pct": exact_pct,
                "train_seconds": elapsed_total,
                "active_config": active_config
            }, out_dir / "best.pt")
            print(f"  [*] Best model saved at epoch {epoch} (Val CER: {val_cer*100:.2f}%)")
        else:
            bad_epochs += 1

        # Save last checkpoint (full resume state) AFTER best.pt, then log the epoch
        atomic_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": ocr_cfg,
            "chars": vocab.chars,
            "val_cer": val_cer,
            "val_loss": val_loss,
            "best_val_cer": best_val_cer,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "elapsed_seconds": elapsed_total,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "active_config": active_config
        }, last_ckpt_path)
        log_fp.write(json.dumps(log_entry) + "\n")
        log_fp.flush()

        if bad_epochs >= patience:
            print(f"[!] Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
            break

    total_training_sec = elapsed_before + (time.time() - start_time)
    log_fp.close()

    # Final summary JSON
    summary = {
        "status": "COMPLETED",
        "run_name": out_dir.name,
        "seed": seed,
        "total_epochs": epoch,
        "best_epoch": best_epoch,
        "best_val_cer_pct": round(best_val_cer * 100, 3),
        "best_val_loss": round(best_val_loss, 5),
        "total_training_seconds": round(total_training_sec, 2),
        "hardware": hw_info,
        "active_config": active_config
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] Run Finished: {out_dir.name}")
    print(f"[+] Best Epoch: {best_epoch} | Best Val CER: {best_val_cer*100:.2f}% | Total Time: {total_training_sec/60:.2f} min")
    print(f"[+] Outputs saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
