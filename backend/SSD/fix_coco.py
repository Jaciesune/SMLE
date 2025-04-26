import json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

# Ścieżki
train_path = "dataset/train/annotations/instances_train.json"
val_path   = "dataset/val/annotations/instances_val.json"

# Wczytujemy
train = load(train_path)
val   = load(val_path)

# 1. Dodajemy licenses = [] jeśli brakuje
for d in (train, val):
    if "licenses" not in d:
        d["licenses"] = []

# 2. Kopiujemy sekcję categories z train do val
if "categories" in train:
    val["categories"] = train["categories"]
else:
    raise RuntimeError("Brak sekcji 'categories' w instancjach treningowych!")

# Zapisujemy
save(train, train_path)
save(val,   val_path)

print("Poprawiono pliki COCO JSON.")
