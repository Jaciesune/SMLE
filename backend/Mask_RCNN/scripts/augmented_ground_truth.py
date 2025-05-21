import os
import json
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from pycocotools import mask as mask_utils
import logging
import psutil

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AugmentedGroundTruth:
    def __init__(self, dataset_dir, output_dir, num_augmentations=1, image_size=(1024, 1024), num_processes=4):
        """
        Args:
            dataset_dir (str): Ścieżka do katalogu z danymi (np. ../../data).
            output_dir (str): Ścieżka do katalogu wyjściowego dla augmentowanych obrazów.
            num_augmentations (int): Liczba augmentacji na obraz.
            image_size (tuple): Rozmiar obrazów (wysokość, szerokość).
            num_processes (int): Liczba procesów do równoległego przetwarzania.
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.num_augmentations = num_augmentations
        self.image_size = image_size
        self.num_processes = num_processes
        self.image_dir = os.path.join(dataset_dir, "train", "images")
        self.annotation_path = os.path.join(dataset_dir, "train", "annotations", "instances_train.json")

        # Sprawdzenie istnienia katalogów i plików
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Nie znaleziono katalogu: {self.image_dir}")
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Nie znaleziono pliku adnotacji: {self.annotation_path}")

        # Wczytanie adnotacji
        with open(self.annotation_path, 'r') as f:
            self.annotations = json.load(f)

        self.image_info = {img['id']: img for img in self.annotations['images']}
        self.image_ids = []

        # Filtracja obrazów z istniejącymi plikami i bboxami
        for img_id in self.image_info.keys():
            image_path = os.path.join(self.image_dir, self.image_info[img_id]['file_name'])
            if not os.path.exists(image_path):
                logger.warning("Pomijam obraz, plik nie istnieje: %s", image_path)
                continue
            anns = [a for a in self.annotations['annotations'] if a['image_id'] == img_id]
            if not anns:
                logger.warning("Pomijam obraz bez adnotacji: %s", image_path)
                continue
            has_bboxes = any('bbox' in ann and len(ann['bbox']) == 4 for ann in anns)
            if not has_bboxes:
                logger.warning("Pomijam obraz bez bboxów: %s", image_path)
                continue
            self.image_ids.append(img_id)

        logger.info("Załadowano %d obrazów z adnotacjami w %s", len(self.image_ids), self.image_dir)

        if not self.image_ids:
            raise ValueError(f"Brak obrazów z adnotacjami i bboxami w {self.image_dir}")

        # Pipeline augmentacji
        all_transforms = [
            # Geometryczne transformacje z SomeOf
            A.SomeOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.2),
                A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.4),
                A.Affine(
                    scale=(0.85, 1.15),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-30, 30),
                    shear=(-5, 5),
                    p=0.4
                ),
                A.RandomResizedCrop(
                    size=self.image_size,
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.33),
                    p=0.3
                ),
                A.Affine(shear=(-15, 15), p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=4, nb_cols=4, p=0.2),
                A.RandomScale(scale_limit=(-0.2, 0.2), p=0.3),
                A.CoarseDropout(
                    num_holes_range=(1, 5),  # Nowość: Zakres liczby dziur
                    fill=0,
                    fill_mask=0,
                    p=0.3
                ),  # Nowość: CoarseDropout z zakresami
            ], n=6, p=0.8),

            # Wizualne transformacje z SomeOf
            A.SomeOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.GaussNoise(p=0.3),
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True, p=0.2),
                A.Blur(blur_limit=(3, 7), p=0.2),
                A.MotionBlur(blur_limit=(3, 15), p=0.2),
                A.RandomFog(alpha_coef=0.2, p=0.1),
                A.RandomRain(brightness_coefficient=0.9, drop_length=20, p=0.1),
                A.RandomSnow(p=0.1),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    src_radius=150,
                    p=0.2
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_limit=(1, 3),
                    shadow_dimension=5,
                    p=0.3
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.Emboss(p=0.2),
                A.Sharpen(p=0.2),
                A.CLAHE(p=0.2),
                A.ImageCompression(quality_range=(50, 90), p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomToneCurve(scale=0.1, p=0.3),
                A.Solarize(p=0.2),
                A.Posterize(num_bits=(4, 8), p=0.2),
                A.Downscale(scale_range=(0.25,0.5), p=0.2),
                A.Superpixels(p_replace=0.1, n_segments=100, p=0.2),
            ], n=6, p=0.8),

            # Końcowe skalowanie
            A.Resize(height=image_size[1], width=image_size[0]),
        ]

        self.augment_transform = A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='coco',
                label_fields=['category_ids'],
                min_area=10,
                min_visibility=0.1
            ),
            additional_targets={'mask': 'mask'}
        )

        # Cache masek
        self.mask_cache = {}

    def decode_rle(self, segmentation, bbox, target_size):
        """Dekoduje RLE do maski binarnej i pozycjonuje ją w granicach bboxa"""
        rle_key = str(segmentation["counts"]) + str(segmentation["size"])
        if rle_key in self.mask_cache:
            return self.mask_cache[rle_key]

        rle = {"counts": segmentation["counts"].encode('utf-8'), "size": segmentation["size"]}
        mask = mask_utils.decode(rle)
        x, y, w, h = map(int, bbox)
        if w <= 0 or h <= 0:
            mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            full_mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
            x_end = min(x + w, target_size[1])
            y_end = min(y + h, target_size[0])
            full_mask[y:y_end, x:x_end] = mask[:y_end - y, :x_end - x]
            mask = full_mask

        self.mask_cache[rle_key] = mask
        return mask

    def overlay_mask_on_image(self, image, mask):
        """Nakłada maskę na obraz (czerwony kolor)"""
        image[mask > 0] = [255, 0, 0]
        return image

    def process_image(self, image_id):
        """
        Przetwarza pojedynczy obraz z augmentacjami i zapisuje wyniki.
        
        Args:
            image_id: ID obrazu z datasetu COCO.
        
        Returns:
            Tuple: (lista informacji o obrazie, lista adnotacji) dla augmentowanych danych.
        """
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        # Wczytanie obrazu
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Nie można wczytać obrazu: %s", image_path)
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        # Wczytanie adnotacji
        anns = [a for a in self.annotations['annotations'] if a['image_id'] == image_id]
        if not anns:
            logger.warning("Brak adnotacji dla obrazu: %s", image_path)
            return

        boxes = [ann['bbox'] for ann in anns]
        masks = [self.decode_rle(ann['segmentation'], ann['bbox'], (orig_height, orig_width))
                 if 'segmentation' in ann else None for ann in anns]
        labels = [ann['category_id'] for ann in anns]

        # Połącz maski w jedną
        combined_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        for mask, bbox in zip(masks, boxes):
            if mask is not None:
                x, y, w, h = map(int, bbox)
                x_end = min(x + w, orig_width)
                y_end = min(y + h, orig_height)
                combined_mask[y:y_end, x:x_end] |= mask[y:y_end, x:x_end]

        # Zapisz oryginalny obraz z maskami i bboxami
        base_name = os.path.splitext(image_info['file_name'])[0]
        vis_image = image.copy()
        for bbox in boxes:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        vis_image = self.overlay_mask_on_image(vis_image, combined_mask)
        output_path = os.path.join(self.output_dir, f"{base_name}_orig.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        # Generuj augmentacje
        for i in range(self.num_augmentations):
            aug_data = {
                'image': image,
                'bboxes': boxes,
                'category_ids': labels,
                'mask': combined_mask
            }
            augmented = self.augment_transform(**aug_data)
            aug_image = augmented['image']
            aug_boxes = augmented['bboxes']
            aug_mask = augmented['mask']
            aug_labels = augmented['category_ids']

            # Filtruj i przycinaj bboxy
            filtered_boxes = []
            filtered_labels = []
            height, width = aug_image.shape[:2]

            for bbox, label in zip(aug_boxes, aug_labels):
                x, y, w, h = bbox
                # Przycinanie współrzędnych do granic obrazu
                x_start = max(0, min(x, width))
                y_start = max(0, min(y, height))
                x_end = max(0, min(x + w, width))
                y_end = max(0, min(y + h, height))

                # Obliczenie nowej szerokości i wysokości
                new_w = x_end - x_start
                new_h = y_end - y_start

                # Sprawdzenie, czy bbox jest prawidłowy
                if new_w > 0 and new_h > 0:
                    filtered_boxes.append([x_start, y_start, new_w, new_h])
                    filtered_labels.append(label)

            if len(filtered_boxes) == 0:
                logger.warning(f"Obraz {image_path} (aug_{i}) nie ma bboxów po augmentacji")
                continue

            # Wizualizacja augmentowanego obrazu
            vis_aug_image = aug_image.copy()
            for bbox in filtered_boxes:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(vis_aug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            vis_aug_image = self.overlay_mask_on_image(vis_aug_image, aug_mask)

            # Zapisz augmentowany obraz
            output_path = os.path.join(self.output_dir, f"{base_name}_aug_{i}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(vis_aug_image, cv2.COLOR_RGB2BGR))

    def create_augmented_dataset(self):
        """Tworzy augmentowany zbiór danych z obrazami, maskami i bboxami"""
        os.makedirs(self.output_dir, exist_ok=True)

        # Automatyczne dostosowanie liczby procesów
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        cpu_count = os.cpu_count()
        num_processes = self.num_processes
        if available_memory < 4:
            num_processes = min(num_processes, max(1, cpu_count // 2))
            logger.warning("Mało pamięci systemowej (%.2f GB), zmniejszam num_processes do %d", available_memory, num_processes)
        else:
            num_processes = min(num_processes, cpu_count)

        logger.info("Używam %d procesów do augmentacji", num_processes)

        # Przetwarzanie obrazów
        for image_id in tqdm(self.image_ids, desc="Augmenting images"):
            self.process_image(image_id)

if __name__ == "__main__":
    dataset_dir = "../../data"
    output_dir = "../data/test/data_augmented"
    num_augmentations = 20

    augmenter = AugmentedGroundTruth(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        num_augmentations=num_augmentations,
        image_size=(1024, 1024)
    )
    augmenter.create_augmented_dataset()