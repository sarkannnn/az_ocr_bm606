"""Master Experiment Orchestrator for AzOCR-V2 GPU Training.

Orchestrates the 2x2 factorial matrix (Groups A, B, C, D) across 3 seeds (42, 43, 44)
as well as the Random Padding and Variable-Width experiments.

Usage:
  # 1. Smoke test (verifies CUDA, checkpoint saving, and pipeline in < 1 minute):
  python run_experiments.py --smoke

  # 2. Run full 12 factorial experiments (Groups A, B, C, D x 3 seeds):
  python run_experiments.py --all

  # 3. Run a specific group only:
  python run_experiments.py --group D --seeds 42 43 44

  # 4. Run additional experiments:
  python run_experiments.py --group padding
  python run_experiments.py --group variable_width
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


EXPERIMENT_MATRIX = {
    "A": {
        "name": "Group_A_V1Clean_BasicAug",
        "config": "configs/group_a.yaml",
        "description": "Group A (V1 10,241 clean samples, 8 fonts, V1 basic augmentation - Baseline)"
    },
    "B": {
        "name": "Group_B_V1Clean_PhotoAug",
        "config": "configs/group_b.yaml",
        "description": "Group B (V1 10,241 clean samples, 8 fonts, V2 realistic photo augmentation)"
    },
    "C": {
        "name": "Group_C_V2Clean_BasicAug",
        "config": "configs/group_c.yaml",
        "description": "Group C (V2 15,754 clean samples, 14 fonts, V1 basic augmentation)"
    },
    "D": {
        "name": "Group_D_V2Clean_PhotoAug",
        "config": "configs/group_d.yaml",
        "description": "Group D (V2 15,754 clean samples, 14 fonts, V2 realistic photo augmentation - Full V2)"
    },
    "padding": {
        "name": "Exp_RandomPadding_GroupD",
        "config": "configs/exp_random_padding.yaml",
        "description": "Random Border Padding (Group D + random 0-12 px border padding augmentation)"
    },
    "variable_width": {
        "name": "Exp_VariableWidth_GroupD",
        "config": "configs/exp_variable_width.yaml",
        "description": "Variable-Width Batching (Group D + dynamic canvas width collation)"
    }
}


def compile_experiment_summary(runs_dir: Path) -> Dict[str, Any]:
    """Compiles all completed run summaries into a unified report."""
    results_by_group: Dict[str, List[Dict[str, Any]]] = {}

    for summary_path in runs_dir.glob("*/run_summary.json"):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            run_name = data.get("run_name", "")
            # Find group
            group_key = None
            for k, meta in EXPERIMENT_MATRIX.items():
                if run_name.startswith(meta["name"]):
                    group_key = k
                    break
            if group_key is None:
                group_key = run_name.split("_seed")[0]

            if group_key not in results_by_group:
                results_by_group[group_key] = []
            results_by_group[group_key].append(data)
        except Exception as e:
            print(f"[!] Warning: Failed to read {summary_path}: {e}")

    summary_table = []
    for g_key, runs in results_by_group.items():
        cers = [r["best_val_cer_pct"] for r in runs]
        times = [r["total_training_seconds"] for r in runs]
        epochs = [r["best_epoch"] for r in runs]

        summary_table.append({
            "group": g_key,
            "description": EXPERIMENT_MATRIX.get(g_key, {}).get("description", g_key),
            "completed_runs": len(runs),
            "seeds": [r["seed"] for r in runs],
            "val_cer_mean_pct": round(float(np.mean(cers)), 2),
            "val_cer_std_pct": round(float(np.std(cers)), 2),
            "val_cer_min_pct": round(float(np.min(cers)), 2),
            "val_cer_max_pct": round(float(np.max(cers)), 2),
            "mean_best_epoch": round(float(np.mean(epochs)), 1),
            "mean_duration_minutes": round(float(np.mean(times)) / 60.0, 1)
        })

    out_json = {
        "status": "COMPLETED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups_summary": summary_table,
        "raw_runs": results_by_group
    }

    # Write summary JSON
    with open(runs_dir / "experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # Write summary Markdown
    md_path = runs_dir / "experiments_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AzOCR-V2 GPU Təlim Nəticələri Xülasəsi\n\n")
        f.write(f"**Tarix:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Tamamlanan Qruplar:** {len(summary_table)}  \n\n")
        f.write("## 1. 2x2 Faktorlu və Əlavə Təcrübələrin Müqayisəsi (3 Seed Ortalaması)\n\n")
        f.write("| Qrup / Eksperiment | Təsvir | Təkrarlar (Seed) | **Validasiya CER (%)** (Orta $\\pm$ Std) | Ən Yaxşı CER (%) | Orta Epox | Orta Müddət |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for row in summary_table:
            f.write(f"| **{row['group']}** | {row['description']} | {row['completed_runs']} ({row['seeds']}) | "
                    f"**{row['val_cer_mean_pct']:.2f}% $\\pm$ {row['val_cer_std_pct']:.2f}%** | "
                    f"**{row['val_cer_min_pct']:.2f}%** | {row['mean_best_epoch']} | {row['mean_duration_minutes']:.1f} dəq |\n")

        f.write("\n\n## 2. Metodoloji Qeydlər\n")
        f.write("- Qrup A vs B müqayisəsi: Veri sabit saxlanaraq (10,241 təmiz V1) yalnız augmentasiyanın effekti izolyasiya edilir.\n")
        f.write("- Qrup C vs D müqayisəsi: Veri sabit saxlanaraq (15,754 təmiz V2) yalnız augmentasiyanın effekti izolyasiya edilir.\n")
        f.write("- Qrup A vs C müqayisəsi: Augmentasiya sabit saxlanaraq (V1 basic) yalnız veri həcminin (10k -> 15.7k) effekti ölçülür.\n")
        f.write("- Qrup B vs D müqayisəsi: Augmentasiya sabit saxlanaraq (V2 photo) yalnız veri həcminin (10k -> 15.7k) effekti ölçülür.\n")

    print(f"\n[+] Unified summary written to {md_path}")
    return out_json


def check_cuda_diagnostics():
    """Runs a comprehensive CUDA and PyTorch diagnostic check."""
    import torch
    print("=" * 80)
    print("CUDA VƏ PYTORCH DİAQNOSTİKA DENETİMİ")
    print("=" * 80)
    print(f"PyTorch Versiyası:      {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA Mövcuddur:         {cuda_ok}")

    if not cuda_ok:
        print("\n[!] DİQQƏT: SİSTEMDƏ CUDA DƏSTƏYİ TAPILMADI!")
        print("Model hazırda CPU rejimində işləyəcək.")
        print("Tövsiyə:")
        print("1. NVIDIA sürücüsünün (versiya >= 525) quraşdırıldığını yoxlayın.")
        print("2. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 əmrini icra edin.")
        print("=" * 80)
        return False

    dev_cnt = torch.cuda.device_count()
    cur_dev = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(cur_dev)
    props = torch.cuda.get_device_properties(cur_dev)
    total_vram_gb = props.total_memory / (1024 ** 3)
    cuda_ver = torch.version.cuda

    print(f"NVIDIA GPU Sayı:        {dev_cnt}")
    print(f"Aktiv GPU Cihazı:       [{cur_dev}] {dev_name}")
    print(f"CUDA Runtime Versiyası: {cuda_ver}")
    print(f"Hesablama Bacarığı:     {props.major}.{props.minor}")
    print(f"Ümumi VRAM Həcmi:       {total_vram_gb:.2f} GB")

    # Real GPU tensor allocation & matmul test
    try:
        t0 = time.time()
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        alloc_vram_mb = torch.cuda.memory_allocated(cur_dev) / (1024 ** 2)
        matmul_time_ms = (time.time() - t0) * 1000.0
        print(f"Tensor Matmul Testi:    UĞURLU ({matmul_time_ms:.2f} ms, VRAM: {alloc_vram_mb:.1f} MB)")
        print("[+] CUDA mühiti tam funksionaldır və GPU təliminə hazırdır!")
    except Exception as e:
        print(f"[!] Tensor əməliyyatında xəta: {e}")
        return False

    print("=" * 80 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner for AzOCR-V2")
    parser.add_argument("--all", action="store_true", help="Run 12 factorial experiments (A, B, C, D across 3 seeds)")
    parser.add_argument("--all-14", action="store_true", help="Run all 12 factorial + 2 extra (Random Padding & Variable-Width)")
    parser.add_argument("--group", type=str, default=None, choices=["A", "B", "C", "D", "padding", "variable_width", "all", "extra"],
                        help="Select group to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds (default: 42 43 44)")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test on small subset")
    parser.add_argument("--cuda-check", action="store_true", help="Run CUDA diagnostics and exit")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Base directory to save run outputs")
    parser.add_argument("--summarize-only", action="store_true", help="Only generate summary from existing runs")
    args = parser.parse_args()

    if args.cuda_check:
        check_cuda_diagnostics()
        return

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        compile_experiment_summary(runs_dir)
        return

    # Check CUDA first: refuse to train on CPU
    if not check_cuda_diagnostics():
        print("[!] GPU hazır deyil, təlim başladılmadı.")
        sys.exit(2)

    # Determine which groups to run
    if args.all_14:
        groups_to_run = ["A", "B", "C", "D", "padding", "variable_width"]
    elif args.all or args.group == "all":
        groups_to_run = ["A", "B", "C", "D"]
    elif args.group == "extra":
        groups_to_run = ["padding", "variable_width"]
    elif args.group:
        groups_to_run = [args.group]
    elif args.smoke:
        groups_to_run = ["D"]
    else:
        print("Zəhmət olmasa parametr göstərin: --all, --all-14, --group <A|B|C|D|padding|variable_width>, və ya --smoke.")
        return

    seeds_to_run = [42] if args.smoke else args.seeds

    print("=" * 80)
    print("AzOCR-V2 GPU Training Matrix Orchestration")
    print(f"Groups to run: {groups_to_run}")
    print(f"Seeds: {seeds_to_run}")
    print(f"Smoke Test Mode: {args.smoke}")
    print(f"Output Directory: {runs_dir}")
    print("=" * 80)

    total_tasks = len(groups_to_run) * len(seeds_to_run)
    task_idx = 0

    overall_start = time.time()

    for g_key in groups_to_run:
        meta = EXPERIMENT_MATRIX[g_key]
        cfg_file = meta["config"]
        base_name = meta["name"]

        for seed in seeds_to_run:
            task_idx += 1
            run_name = f"{base_name}_seed{seed}"
            if args.smoke:
                run_name = f"smoke_test_{run_name}"

            out_run_dir = runs_dir / run_name

            summary_file = out_run_dir / "run_summary.json"
            if not args.smoke and summary_file.exists():
                try:
                    if json.loads(summary_file.read_text(encoding="utf-8")).get("status") == "COMPLETED":
                        print(f"\n[{task_idx}/{total_tasks}] SKIP (artıq tamamlanıb): {run_name}")
                        continue
                except (json.JSONDecodeError, OSError):
                    pass

            print(f"\n[{task_idx}/{total_tasks}] STARTING RUN: {run_name}")
            print(f"Description: {meta['description']}")

            cmd = [
                sys.executable,
                "train_gpu.py",
                "--config", cfg_file,
                "--out", str(out_run_dir),
                "--seed", str(seed)
            ]
            if args.smoke:
                cmd.append("--smoke")
            else:
                cmd.append("--resume")

            t0 = time.time()
            res = subprocess.run(cmd)
            elapsed = time.time() - t0

            if res.returncode != 0:
                print(f"[!] ERROR: Run {run_name} failed with exit code {res.returncode}")
                sys.exit(res.returncode)
            else:
                print(f"[+] Run {run_name} completed successfully in {elapsed/60:.2f} minutes.")

    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 80)
    print(f"[+] ALL REQUESTED EXPERIMENTS FINISHED! Total Time: {overall_elapsed/60:.2f} minutes.")
    print("=" * 80)

    # Automatically compile final summary
    compile_experiment_summary(runs_dir)


if __name__ == "__main__":
    main()
