import os
import cv2
import numpy as np
from dataset import RuryDataset  # Zakładam, że Twój dataset.py jest w tym samym katalogu

def save_augmented_ground_truth_with_visualization(dataset_dir, output_dir, subset="train", num_augmentations=5):
    """
    Generuje augmentowany zestaw danych z nałożonymi maskami i bounding boxami, zapisując tylko obrazy.
    
    Args:
        dataset_dir (str): Ścieżka do oryginalnego katalogu danych.
        output_dir (str): Ścieżka do katalogu, gdzie zapisane będą obrazy z wizualizacją.
        subset (str): "train" lub "val" - zbiór do augmentacji.
        num_augmentations (int): Liczba augmentacji na obraz.
    """
    # Utwórz katalog wyjściowy
    output_image_dir = os.path.join(output_dir, subset, "images")
    os.makedirs(output_image_dir, exist_ok=True)

    # Wczytaj dataset
    dataset = RuryDataset(
        dataset_dir=dataset_dir,
        subset=subset,
        image_size=(1024, 1024),
        augment=True,
        num_augmentations=num_augmentations
    )

    print(f"Generowanie obrazów z maskami i bounding boxami dla zbioru {subset}...")

    # Iteruj przez wszystkie obrazy w dataset (włącznie z augmentacjami)
    for idx in range(len(dataset)):
        image, target = dataset[idx]

        # Oblicz oryginalny indeks i numer augmentacji
        orig_idx = idx // num_augmentations
        aug_idx = idx % num_augmentations

        orig_image_id = dataset.image_ids[orig_idx]
        orig_image_info = dataset.image_info[orig_image_id]

        # Nazwa pliku dla obrazu
        if aug_idx == 0:
            # Pierwsza kopia to oryginalny obraz
            new_filename = orig_image_info['file_name']
        else:
            # Kolejne kopie to augmentacje
            base_name, ext = os.path.splitext(orig_image_info['file_name'])
            new_filename = f"{base_name}_aug{aug_idx}{ext}"

        # Konwersja obrazu z tensoru na numpy (RGB)
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Nałóż bounding boxy i maski
        boxes = target["boxes"].numpy()
        masks = target["masks"].numpy()

        for box, mask in zip(boxes, masks):
            # Bounding box
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Zielony prostokąt

            # Maska
            mask_np = (mask > 0).astype(np.uint8) * 255  # Konwersja maski na wartości 0-255
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_np, contours, -1, (0, 0, 255), 2)  # Czerwone kontury maski

        # Zapisz obraz z wizualizacją
        image_path = os.path.join(output_image_dir, new_filename)
        cv2.imwrite(image_path, image_np)

    print(f"Zapisano obrazy z wizualizacją w: {output_image_dir}")
    print(f"Liczba zapisanych obrazów: {len(dataset)}")

if __name__ == "__main__":
    # Przykładowe użycie
    dataset_dir = "../data"  # Ścieżka do oryginalnych danych
    output_dir = "../data/test/data_augmented"  # Ścieżka do zapisu obrazów z wizualizacją
    num_augmentations = 5  # Liczba augmentacji na obraz

    save_augmented_ground_truth_with_visualization(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        subset="train",
        num_augmentations=num_augmentations
    )