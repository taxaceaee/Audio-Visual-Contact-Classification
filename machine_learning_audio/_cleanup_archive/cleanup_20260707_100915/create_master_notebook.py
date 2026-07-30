import json
from pathlib import Path

nb_path = Path('machine_learning_audio/00_master_audio_paper_pipeline.ipynb')

cells = []

def md(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':text.strip().splitlines(True)})

def code(text):
    cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.strip().splitlines(True)})

md(r'''
# Master Audio Paper Pipeline — Frozen Artifact Replay

This notebook validates the current audio-only paper-safe checkpoint from `machine_learning_audio`.

Mode used here: **Mode A: Frozen Artifact Replay**.

Purpose:
- validate final paper metrics without rebuilding all upstream models;
- inspect report and confusion matrix artifacts;
- export a compact validation summary for the conference paper.

This notebook does **not** retrain from raw audio. Full rebuild is Mode B and can drift across package versions, random seeds, and thread scheduling.
''')

code(r'''
from pathlib import Path
import json
import platform
import importlib
import numpy as np
import pandas as pd

ROOT = Path.cwd()
AUDIO_DIR = ROOT / "machine_learning_audio"
assert AUDIO_DIR.exists(), f"Missing {AUDIO_DIR}"

print("ROOT:", ROOT)
print("AUDIO_DIR:", AUDIO_DIR)
print("Python:", platform.python_version())

for name in ["numpy", "pandas", "sklearn", "joblib"]:
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'OK')}")
    except Exception as exc:
        print(f"{name}: MISSING {exc}")
''')

md('''
## 1. Locate Frozen Checkpoint Artifacts

The checkpoint README states that the final paper-safe run slug is `audio_lift_source_blend_select`.
''')

code(r'''
RUN = "audio_lift_source_blend_select"
paths = {
    "report": AUDIO_DIR / f"{RUN}_final_test_report.csv",
    "confusion": AUDIO_DIR / f"{RUN}_final_test_confusion_matrix.csv",
    "leaderboard": AUDIO_DIR / f"{RUN}_oof_leaderboard.csv",
    "selected": AUDIO_DIR / f"{RUN}_selected_without_test.json",
    "method_card": AUDIO_DIR / f"{RUN}_method_card_before_test.json",
    "protocol_summary": AUDIO_DIR / f"{RUN}_protocol_summary.json",
}

for key, path in paths.items():
    print(f"{key:16s}", "OK" if path.exists() else "MISSING", path)

missing = [key for key, path in paths.items() if not path.exists()]
assert not missing, f"Missing checkpoint artifacts: {missing}"
''')

md('''
## 2. Load Locked Metadata
''')

code(r'''
selected = json.loads(paths["selected"].read_text(encoding="utf-8"))
method_card = json.loads(paths["method_card"].read_text(encoding="utf-8"))
protocol_summary = json.loads(paths["protocol_summary"].read_text(encoding="utf-8"))

print("Selected lock:")
print(json.dumps(selected, indent=2, ensure_ascii=False)[:3000])
print("\nMethod card:")
print(json.dumps(method_card, indent=2, ensure_ascii=False)[:3000])
''')

md('''
## 3. Validate Final Report Metrics
''')

code(r'''
report = pd.read_csv(paths["report"])
display(report)

row = report.iloc[0].to_dict()
expected = {
    "accuracy_4class": 0.7967552951780081,
    "macro_f1_4class": 0.7026719927789447,
    "contact_macro_f1": 0.625050260344378,
    "binary_macro_f1": 0.9291164642187257,
}

print("Metric checks:")
for key, exp in expected.items():
    got = float(row[key])
    ok = abs(got - exp) < 1e-12
    print(f"{key:20s} got={got:.15f} expected={exp:.15f} ok={ok}")
    assert ok, f"{key} mismatch: got {got}, expected {exp}"

print("PASS: final report metrics match expected checkpoint values.")
''')

md('''
## 4. Validate Confusion Matrix and Recompute Metrics
''')

code(r'''
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

cm = pd.read_csv(paths["confusion"], index_col=0)
display(cm)

labels = ["ambient", "leaf", "trunk", "twig"]
contact_labels = [1, 2, 3]

y_true = []
y_pred = []
for i, true_label in enumerate(labels):
    for j, pred_label in enumerate(labels):
        count = int(cm.loc[true_label, pred_label])
        y_true.extend([i] * count)
        y_pred.extend([j] * count)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = accuracy_score(y_true, y_pred)
macro = f1_score(y_true, y_pred, labels=[0,1,2,3], average="macro", zero_division=0)
contact = f1_score(y_true, y_pred, labels=contact_labels, average="macro", zero_division=0)
binary = f1_score((y_true > 0).astype(int), (y_pred > 0).astype(int), labels=[0,1], average="macro", zero_division=0)

checks = {
    "accuracy_4class": acc,
    "macro_f1_4class": macro,
    "contact_macro_f1": contact,
    "binary_macro_f1": binary,
}

for key, got in checks.items():
    exp = expected[key]
    ok = abs(got - exp) < 1e-12
    print(f"{key:20s} recomputed={got:.15f} expected={exp:.15f} ok={ok}")
    assert ok, f"Recomputed {key} mismatch"

pr, rc, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], zero_division=0)
per_class = pd.DataFrame({"class": labels, "precision": pr, "recall": rc, "f1": f1, "support": support})
display(per_class)

print("PASS: confusion matrix recomputes final checkpoint metrics.")
''')

md('''
## 5. OOF Leaderboard and Selection Evidence
''')

code(r'''
leaderboard = pd.read_csv(paths["leaderboard"])
print("Leaderboard shape:", leaderboard.shape)
display(leaderboard.head(10))

if "selection_score" in leaderboard.columns:
    display(leaderboard.sort_values("selection_score", ascending=False).head(10))
''')

md('''
## 6. Export Paper Validation Summary
''')

code(r'''
summary = {
    "mode": "Frozen Artifact Replay",
    "run_slug": RUN,
    "split": row.get("split", "robot_test_final"),
    "audio_only": True,
    "uses_image_or_multimodal_features": False,
    "metrics": {key: float(row[key]) for key in expected},
    "n_samples_from_confusion_matrix": int(cm.to_numpy().sum()),
    "selected_lock": selected,
    "method_card": method_card,
}

out_json = AUDIO_DIR / "00_master_audio_paper_pipeline_validation_summary.json"
out_csv = AUDIO_DIR / "00_master_audio_paper_pipeline_per_class_metrics.csv"
Path(out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
per_class.to_csv(out_csv, index=False)

print("Wrote:", out_json)
print("Wrote:", out_csv)
print("VALIDATION PASS")
''')

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'}
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(nb_path)
