import json, glob, io, os
out = io.open('paper/scratch/metrics_dump.txt', 'w', encoding='utf-8')
for d in ['word_synth','line_synth','word_v2_synth','line_v2_synth','tess_word_synth','tess_line_synth',
          'diag_tahoma','diag_consola','diagw_tahoma','diagw_consola','diag_tahoma_v2','diag_consola_v2','diagw_tahoma_v2','diagw_consola_v2',
          'tess_diag_tahoma','tess_diag_consola','tess_diagw_tahoma','tess_diagw_consola']:
    p = 'results/%s' % d
    out.write('##### %s files=%s\n' % (d, os.listdir(p)))
    mp = p + '/metrics.json'
    if os.path.exists(mp):
        j = json.load(open(mp, encoding='utf-8'))
        s = json.dumps(j, ensure_ascii=False)
        out.write(s[:1800] + '\n')
out.close()
