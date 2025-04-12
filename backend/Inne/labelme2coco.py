import os
import json
import base64
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from pycocotools import mask as mask_utils
from io import BytesIO

# Ścieżki bazowe
INPUT_DIR = "./images_annotations_all"  # Katalog z obrazami i adnotacjami
OUTPUT_DIR = "./data"  # Katalog wyjściowy

# Procentowy podział na train/val
TRAIN_RATIO = 0.7

def decode_mask(base64_str):
    """Dekoduje maskę zapisaną w Base64 do postaci NumPy."""
    mask_bytes = base64.b64decode(base64_str)
    mask_image = Image.open(BytesIO(mask_bytes)).convert("L")
    return np.array(mask_image, dtype=np.uint8)

def normalize_image_path(image_path):
    """Normalizuje ścieżkę obrazu do samej nazwy pliku."""
    return os.path.basename(image_path)

def load_labelme_json(json_path):
    """Wczytuje plik JSON z LabelMe i normalizuje imagePath."""
    with open(json_path, "r") as f:
        data = json.load(f)
    data["imagePath"] = normalize_image_path(data["imagePath"])
    return data

def encode_rle(mask):
    """Koduje maskę binarną do formatu RLE."""
    mask = np.asfortranarray(mask)
    return mask_utils.encode(mask)

def convert_to_coco(data_list, output_path):
    """Konwertuje LabelMe do COCO JSON z maskami w formacie RLE."""
    coco = {
        "info": {"description": "Dataset converted from LabelMe"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_map = {}
    annotation_id = 1

    for image_id, data in enumerate(data_list):
        coco["images"].append({
            "id": image_id,
            "file_name": data["imagePath"],
            "width": data["imageWidth"],
            "height": data["imageHeight"]
        })

        for shape in data["shapes"]:
            label = shape["label"]
            if label not in category_map:
                category_map[label] = len(category_map) + 1
                coco["categories"].append({"id": category_map[label], "name": label})

            bbox = shape["points"]
            xmin, ymin = map(int, bbox[0])
            xmax, ymax = map(int, bbox[1])
            width, height = xmax - xmin, ymax - ymin

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_map[label],
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0
            }

            if "mask" in shape:
                mask_array = decode_mask(shape["mask"])
                rle = encode_rle(mask_array)
                annotation["segmentation"] = {
                    "counts": rle["counts"].decode("utf-8"),
                    "size": list(rle["size"])
                }

            coco["annotations"].append(annotation)
            annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=4)

def split_dataset():
    """Dzieli zbiór na train/val i konwertuje do formatu COCO JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Znajdź wszystkie pliki JSON w folderze images_annotations
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    if not json_files:
        print(f"Brak plików JSON w {INPUT_DIR}")
        return

    # Podziel na train i val
    train_files, val_files = train_test_split(json_files, train_size=TRAIN_RATIO, random_state=42)

    datasets = {
        "train": train_files,
        "val": val_files
    }

    for dataset, files in datasets.items():
        # Utwórz strukturę folderów
        os.makedirs(f"{OUTPUT_DIR}/{dataset}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{dataset}/annotations", exist_ok=True)

        data_list = []

        for json_file in files:
            json_path = os.path.join(INPUT_DIR, json_file)
            labelme_data = load_labelme_json(json_path)
            image_name = labelme_data["imagePath"]
            source_image_path = os.path.join(INPUT_DIR, image_name)

            if not os.path.exists(source_image_path):
                print(f"Brak obrazu w {source_image_path}, pomijam.")
                continue

            # Kopiuj obraz do folderu docelowego
            target_image_path = f"{OUTPUT_DIR}/{dataset}/images/{image_name}"
            shutil.copy(source_image_path, target_image_path)

            # Dodaj dane do listy
            data_list.append(labelme_data)

        # Zapisz plik COCO JSON
        output_json_path = f"{OUTPUT_DIR}/{dataset}/annotations/instances_{dataset}.json"
        convert_to_coco(data_list, output_json_path)

if __name__ == "__main__":
    split_dataset()