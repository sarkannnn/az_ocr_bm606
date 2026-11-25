import os
import hashlib
import json
import torch
import pandas as pd

root = r'C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results'
runs_dir = os.path.join(root, 'runs')
repro_dir = os.path.join(root, 'reproducibility_check')

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

results_csv_path = os.path.join(root, 'results_per_run.csv')
df_runs = pd.read_csv(results_csv_path)

audit_records = []

# List all 18 official runs
expected_runs = [
    'Group_A_V1Clean_BasicAug_seed42',
    'Group_A_V1Clean_BasicAug_seed43',
    'Group_A_V1Clean_BasicAug_seed44',
    'Group_B_V1Clean_PhotoAug_seed42',
    'Group_B_V1Clean_PhotoAug_seed43',
    'Group_B_V1Clean_PhotoAug_seed44',
    'Group_C_V2Clean_BasicAug_seed42',
    'Group_C_V2Clean_BasicAug_seed43',
    'Group_C_V2Clean_BasicAug_seed44',
    'Group_D_V2Clean_PhotoAug_seed42',
    'Group_D_V2Clean_PhotoAug_seed43',
    'Group_D_V2Clean_PhotoAug_seed44',
    'Exp_RandomPadding_GroupD_seed42',
    'Exp_RandomPadding_GroupD_seed43',
    'Exp_RandomPadding_GroupD_seed44',
    'Exp_VariableWidth_GroupD_seed42',
    'Exp_VariableWidth_GroupD_seed43',
    'Exp_VariableWidth_GroupD_seed44'
]

print("Auditing all 18 official runs...")
for r_name in expected_runs:
    r_path = os.path.join(runs_dir, r_name)
    files = set(os.listdir(r_path)) if os.path.exists(r_path) else set()
    
    # Check expected files
    has_best = 'best.pt' in files
    has_last = 'last.pt' in files
    has_summary = 'run_summary.json' in files
    has_config = 'config.yaml' in files
    has_log = 'train_log.jsonl' in files
    
    # Load run_summary
    summary_data = {}
    if has_summary:
        with open(os.path.join(r_path, 'run_summary.json'), 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            
    # Check best.pt
    best_epoch = None
    best_val_cer = None
    best_val_wer = None
    best_sha256 = ""
    best_size = 0
    if has_best:
        pt_path = os.path.join(r_path, 'best.pt')
        best_size = os.path.getsize(pt_path)
        best_sha256 = get_sha256(pt_path)
        try:
            pt_data = torch.load(pt_path, map_location='cpu', weights_only=False)
            best_epoch = pt_data.get('epoch')
            best_val_cer = pt_data.get('val_cer')
            best_val_wer = pt_data.get('val_wer')
        except Exception as e:
            print(f"Error reading {r_name} best.pt:", e)

    # Check last.pt
    last_size = 0
    last_sha256 = ""
    last_epoch = None
    if has_last:
        last_path = os.path.join(r_path, 'last.pt')
        last_size = os.path.getsize(last_path)
        last_sha256 = get_sha256(last_path)
        try:
            pt_last = torch.load(last_path, map_location='cpu', weights_only=False)
            last_epoch = pt_last.get('epoch')
        except Exception as e:
            print(f"Error reading {r_name} last.pt:", e)
            
    # Check train_log.jsonl
    log_epochs = 0
    min_log_cer = 999
    min_log_epoch = -1
    if has_log:
        with open(os.path.join(r_path, 'train_log.jsonl'), 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    log_epochs += 1
                    ep = item.get('epoch')
                    cer = item.get('val_cer', 999)
                    if cer < min_log_cer:
                        min_log_cer = cer
                        min_log_epoch = ep
                        
    # Find matching row in results_per_run.csv
    csv_row = df_runs[df_runs['run'] == r_name]
    csv_val_cer = csv_row['val_cer_pct'].iloc[0] if len(csv_row) > 0 else None
    csv_best_ep = csv_row['best_epoch'].iloc[0] if len(csv_row) > 0 else None
    csv_early = csv_row['early_stopped'].iloc[0] if len(csv_row) > 0 else None
    csv_gpu = csv_row['gpu'].iloc[0] if len(csv_row) > 0 else None

    audit_records.append({
        'run': r_name,
        'has_all_5_files': has_best and has_last and has_summary and has_config and has_log,
        'best_size_mb': round(best_size / (1024 * 1024), 2),
        'last_size_mb': round(last_size / (1024 * 1024), 2),
        'pt_best_epoch': best_epoch,
        'summary_best_epoch': summary_data.get('best_epoch'),
        'log_min_cer_epoch': min_log_epoch,
        'csv_best_epoch': csv_best_ep,
        'pt_val_cer_pct': round(best_val_cer * 100, 3) if best_val_cer is not None else None,
        'summary_val_cer_pct': round(summary_data.get('best_val_cer', 0) * 100, 3) if summary_data.get('best_val_cer') is not None else None,
        'csv_val_cer_pct': csv_val_cer,
        'epochs_logged': log_epochs,
        'last_epoch': last_epoch,
        'early_stopped': csv_early,
        'gpu': csv_gpu,
        'best_sha256': best_sha256,
        'last_sha256': last_sha256
    })

# Also audit reproducibility run
repro_name = 'Exp_VariableWidth_GroupD_seed44_RTX2050'
repro_path = os.path.join(repro_dir, repro_name)
if os.path.exists(repro_path):
    pt_path = os.path.join(repro_path, 'best.pt')
    pt_data = torch.load(pt_path, map_location='cpu', weights_only=False)
    last_path = os.path.join(repro_path, 'last.pt')
    last_data = torch.load(last_path, map_location='cpu', weights_only=False)
    with open(os.path.join(repro_path, 'run_summary.json'), 'r') as f:
        repro_sum = json.load(f)
    csv_row = df_runs[df_runs['run'] == repro_name]
    audit_records.append({
        'run': repro_name + ' (REPRO_CHECK)',
        'has_all_5_files': True,
        'best_size_mb': round(os.path.getsize(pt_path) / (1024*1024), 2),
        'last_size_mb': round(os.path.getsize(last_path) / (1024*1024), 2),
        'pt_best_epoch': pt_data.get('epoch'),
        'summary_best_epoch': repro_sum.get('best_epoch'),
        'log_min_cer_epoch': None,
        'csv_best_epoch': csv_row['best_epoch'].iloc[0] if len(csv_row) > 0 else None,
        'pt_val_cer_pct': round(pt_data.get('val_cer', 0) * 100, 3),
        'summary_val_cer_pct': round(repro_sum.get('best_val_cer', 0) * 100, 3),
        'csv_val_cer_pct': csv_row['val_cer_pct'].iloc[0] if len(csv_row) > 0 else None,
        'epochs_logged': 30,
        'last_epoch': last_data.get('epoch'),
        'early_stopped': csv_row['early_stopped'].iloc[0] if len(csv_row) > 0 else None,
        'gpu': csv_row['gpu'].iloc[0] if len(csv_row) > 0 else None,
        'best_sha256': get_sha256(pt_path),
        'last_sha256': get_sha256(last_path)
    })

audit_df = pd.DataFrame(audit_records)
print("\nAUDIT SUMMARY TABLE:")
print(audit_df[['run', 'has_all_5_files', 'best_size_mb', 'pt_best_epoch', 'csv_best_epoch', 'pt_val_cer_pct', 'csv_val_cer_pct', 'early_stopped', 'gpu']].to_string())

audit_df.to_csv('results/real30_evaluation/gpu_runs_audit.csv', index=False)
print("Saved gpu_runs_audit.csv.")
