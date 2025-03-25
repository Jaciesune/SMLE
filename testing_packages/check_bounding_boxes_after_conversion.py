import json
import os
import cv2
import matplotlib.pyplot as plt

# Funkcja do wczytywania i sprawdzania bounding boxów
def ckeck_annotations(image_dir, annotations_path, output_dir):
    # Wczytanie danych COCO
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Pobranie listy wszystkich obrazów
    image_files = coco_data["images"]

    # Tworzenie katalogu wyjściowego (na obrazy z bounding boxami)
    os.makedirs(output_dir, exist_ok=True)

    # Przetwarzanie każdego obrazu
    for image_info in image_files:
        image_id = image_info["id"]
        image_file = image_info["file_name"]

        # Pobranie anotacji dla danego obrazu
        annotations = [a for a in coco_data["annotations"] if a["image_id"] == image_id]

        # Wczytanie obrazu
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Błąd: Nie można wczytać obrazu {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Narysowanie bounding boxów
        for ann in annotations:
            x, y, w, h = map(int, ann["bbox"])
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Zapis obrazu z bounding boxami do pliku
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(f"Wszystkie obrazy zostały sprawdzone i zapisane w folderze: {output_dir}")

# Sprawdzenie zbioru treningowego
ckeck_annotations(
    image_dir="dataset/train/images",
    annotations_path="dataset/train/annotations.json",
    output_dir="dataset/train/checked_images"
)

# Sprawdzenie zbioru testowego
ckeck_annotations(
    image_dir="dataset/test/images",
    annotations_path="dataset/test/annotations.json",
    output_dir="dataset/test/checked_images"
)