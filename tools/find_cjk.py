import re, subprocess, sys
pat = re.compile(r'[　-〿一-鿿＀-￯⺀-⿿]')
bad = 0
# Hash-protected archive: listed in source_provenance.json and checked by scripts/05_report.py, so it is kept
# byte-identical (its 'note' field is in Chinese). See extensions/balanced/README.md.
EXEMPT = {'results/extensions/balanced_27464ba98305edbc/cache_reuse.json'}
for f in subprocess.check_output(['git', 'ls-files'], text=True).split():
    if f in EXEMPT:
        continue
    try:
        text = open(f, encoding='utf-8').read()
    except Exception:
        continue
    hits = [i + 1 for i, l in enumerate(text.split('\n')) if pat.search(l)]
    if hits:
        bad += len(hits)
        print(f'{f}: {len(hits)} lines, first at {hits[:5]}')
print('TOTAL lines with CJK:', bad)
sys.exit(1 if bad else 0)
