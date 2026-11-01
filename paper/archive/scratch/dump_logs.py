import glob, sys, io
out = io.open('paper/scratch/logs_dump.txt', 'w', encoding='utf-8')
for f in sorted(glob.glob('logs/*.log')):
    b = open(f, 'rb').read()
    t = b.decode('utf-16') if b[:2] in (b'\xff\xfe', b'\xfe\xff') else b.decode('utf-8', 'replace')
    out.write('===== %s\n%s\n' % (f, t.lstrip('\ufeff')))
out.close()
