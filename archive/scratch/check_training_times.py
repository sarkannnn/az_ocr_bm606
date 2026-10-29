import glob, os, torch, pandas as pd

print("=== V2 Line Model ===")
ck_line_v2 = torch.load('runs/line_v2/best.pt', map_location='cpu')
sec = ck_line_v2.get('train_seconds')
print('best.pt epoch:', ck_line_v2.get('epoch'))
print('best.pt train_seconds:', sec, f"({sec/60:.2f} min)")
df_line_v2 = pd.read_csv('runs/line_v2/history.csv')
print('history.csv rows (epochs):', len(df_line_v2))

# Check logs
for lf in sorted(glob.glob('logs/*.log') + glob.glob('logs/*.txt')):
    if 'v2' in lf or 'line' in lf or 'word' in lf:
        with open(lf, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for l in lines[-10:]:
                if 'epoch' in l.lower() or 'train' in l.lower() or 'time' in l.lower() or 'early' in l.lower() or 'seconds' in l.lower():
                    print(f"Log {lf} tail: {l.strip()}")

print("\n=== Word V2 runs ===")
for wd in sorted(glob.glob('runs/word_v2*')):
    bp = os.path.join(wd, 'best.pt')
    hp = os.path.join(wd, 'history.csv')
    if os.path.exists(bp):
        ck = torch.load(bp, map_location='cpu')
        s = ck.get('train_seconds')
        print(f"{wd} best.pt: epoch={ck.get('epoch')}, train_seconds={s} ({s/60:.2f} min)")
    if os.path.exists(hp):
        df = pd.read_csv(hp)
        print(f"{wd} history.csv: {len(df)} epochs")

print("\n=== V1 Checkpoints for comparison ===")
for v1 in ['runs/word/best.pt', 'runs/line/best.pt']:
    if os.path.exists(v1):
        ck = torch.load(v1, map_location='cpu')
        s = ck.get('train_seconds')
        print(f"{v1}: epoch={ck.get('epoch')}, train_seconds={s} ({s/60:.2f} min)")
