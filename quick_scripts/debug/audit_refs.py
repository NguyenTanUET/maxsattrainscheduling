"""Cross-reference audit: labels vs references in a LaTeX project."""
import re
from pathlib import Path

root = Path('D:/github/KLTN_TRP_Final')
tex_files = sorted(root.rglob('*.tex'))

labels = {}
refs = {}

label_re = re.compile(r"\\label\{([^}]+)\}")
ref_re = re.compile(r"\\(?:ref|eqref|autoref|nameref|pageref|cref|Cref)\{([^}]+)\}")

for tf in tex_files:
    try:
        text = tf.read_text(encoding='utf-8')
    except Exception:
        continue
    rel = tf.relative_to(root)
    for i, line in enumerate(text.splitlines(), 1):
        for m in label_re.finditer(line):
            labels[m.group(1)] = (str(rel), i)
        for m in ref_re.finditer(line):
            for key in m.group(1).split(','):
                key = key.strip()
                if key:
                    refs.setdefault(key, []).append((str(rel), i))

print('=== DANGLING REFERENCES (ref to non-existent label) ===')
dangling = sorted(set(refs) - set(labels))
if not dangling:
    print('  (none)')
else:
    for r in dangling:
        sites = refs[r][:3]
        print(f'  {r!r}: referenced at {sites}')

print()
print('=== UNUSED non-eq/non-fig LABELS ===')
unused = sorted(set(labels) - set(refs))
for u in unused:
    if u.startswith(('eq:', 'fig:')):
        continue
    site = labels[u]
    print(f'  {u}  ({site[0]}:{site[1]})')

print()
print(f'Total: {len(labels)} labels, {len(refs)} unique refs.')
