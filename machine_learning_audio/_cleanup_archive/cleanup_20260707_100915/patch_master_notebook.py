import json
from pathlib import Path
p = Path('machine_learning_audio/00_master_audio_paper_pipeline.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
old = '''ROOT = Path.cwd()
AUDIO_DIR = ROOT / "machine_learning_audio"
assert AUDIO_DIR.exists(), f"Missing {AUDIO_DIR}"

print("ROOT:", ROOT)
print("AUDIO_DIR:", AUDIO_DIR)'''
new = '''CWD = Path.cwd()
if CWD.name == "machine_learning_audio":
    AUDIO_DIR = CWD
    ROOT = CWD.parent
else:
    ROOT = CWD
    AUDIO_DIR = ROOT / "machine_learning_audio"
assert AUDIO_DIR.exists(), f"Missing {AUDIO_DIR}"

print("ROOT:", ROOT)
print("AUDIO_DIR:", AUDIO_DIR)'''
changed = False
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if old in src:
            src = src.replace(old, new)
            cell['source'] = src.splitlines(True)
            changed = True
if not changed:
    print('snippet already patched or not found')
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('patched', p)
