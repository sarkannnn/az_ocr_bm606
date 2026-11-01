import os, sys, io, random, json, glob, hashlib
sys.path.insert(0, os.getcwd())
from pathlib import Path
out = io.open('paper/scratch/data_dump.txt', 'w', encoding='utf-8')
def w(*a): out.write(' '.join(str(x) for x in a) + '\n')
for d in ['synth_word','synth_line','synth_word_v2','synth_line_v2','diag_tahoma','diag_consola','diagw_tahoma','diagw_consola','sim_unseen','sim_ladder','sim_audit3','sim_audit4','smoke_word','corpus_word','corpus_line']:
    p = Path('data')/d
    for sp in ['train','val','test']:
        lp = p/sp/'labels.tsv'
        if lp.exists():
            n = sum(1 for l in open(lp, encoding='utf-8') if l.strip())
            w(d, sp, n)
        tp = p/(sp+'.txt')
        if tp.exists():
            ls = [l for l in open(tp, encoding='utf-8').read().split('\n') if l]
            w(d, sp+'.txt', len(ls))
    if p.exists(): w(d, 'listing', os.listdir(p)[:12])
from generate_data import font_ok
for folder in ['fonts','fonts_new','fonts_tahoma','fonts_consola']:
    paths = sorted(str(q) for q in Path(folder).rglob('*') if q.suffix.lower() in {'.ttf','.otf','.ttc'})
    good = [q for q in paths if font_ok(q)]
    w('FONTS', folder, 'found', len(paths), 'ok', len(good), [os.path.basename(x) for x in paths])
    if folder == 'fonts':
        rr = list(good)
        random.Random(42).shuffle(rr)
        n_test = max(1, round(0.2*len(rr)))
        w('  V1 test fonts', [os.path.basename(x) for x in rr[:n_test]], 'train/val fonts', [os.path.basename(x) for x in rr[n_test:]])
for f in ['runs/word_v2_20261027_180300/best.pt','runs/line_v2/best.pt','runs/word/best.pt','runs/line/best.pt']:
    if os.path.exists(f):
        w('SHA256', f, hashlib.sha256(open(f,'rb').read()).hexdigest(), os.path.getsize(f))
for f in glob.glob('runs/*/history.csv'):
    w('HIST', f)
w(os.listdir('runs/word'), os.listdir('runs/line'), os.listdir('runs/word_v2_20261027_180300'), os.listdir('runs/line_v2'))
out.close()
