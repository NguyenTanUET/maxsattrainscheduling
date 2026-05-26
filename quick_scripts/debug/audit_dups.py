import re
from pathlib import Path
from collections import Counter

root = Path('D:/github/KLTN_TRP_Final')
labels = []
for tf in root.rglob('*.tex'):
    text = tf.read_text(encoding='utf-8')
    for m in re.finditer(r"\\label\{([^}]+)\}", text):
        labels.append((m.group(1), str(tf.relative_to(root))))

cnt = Counter(l for l, _ in labels)
dups = {l: [f for ll, f in labels if ll == l] for l, c in cnt.items() if c > 1}
if dups:
    print('DUPLICATE LABELS:')
    for l, files in dups.items():
        print(f'  {l!r}: {files}')
else:
    print('No duplicate labels.')
