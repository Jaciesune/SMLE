import os
import json
from PIL import Image

# Ścieżki do katalogów (dla zbioru treningowego)
#image_dir = "dataset/train/images"
#label_dir = "dataset/train/annotations"
#output_json = "dataset/train/annotations.json"

# Ścieżki do katalogu dla zbioru testowego (z annotacjami) - odpalic po uruchomieniu treningowego 
image_dir = "dataset/test/images"
label_dir = "dataset/test/annotations"
output_json = "dataset/test/annotations.json"

# Definicja kategorii (jedna klasa: "rura")
categories = [{"id": 1, "name": "rura"}]

# Tworzenie struktury COCO
coco_data = {"images": [], "annotations": [], "categories": categories}
annotation_id = 1

# Pobranie listy obrazów
image_files = sorted(os.listdir(image_dir))

for idx, image_file in enumerate(image_files):
    if not image_file.endswith(".jpg"):
        continue

    # Pobranie informacji o obrazie
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)
    width, height = image.size

    # Dodanie obrazu do COCO JSON
    image_id = idx + 1
    coco_data["images"].append({
        "id": image_id,
        "file_name": image_file,
        "width": width,
        "height": height
    })

    # Pobranie anotacji YOLO
    label_file = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0]) + 1  # COCO zaczyna numerację od 1
            x_center, y_center, w, h = map(float, parts[1:])

            # Przekształcenie YOLO do COCO bbox
            x = (x_center - w / 2) * width
            y = (y_center - h / 2) * height
            w = w * width
            h = h * height

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "bbox": [x, y, w, h],
                "category_id": class_id
            })
            annotation_id += 1

# Zapis do pliku JSON
with open(output_json, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"Konwersja YOLO → COCO zakończona! Zapisano do {output_json}")