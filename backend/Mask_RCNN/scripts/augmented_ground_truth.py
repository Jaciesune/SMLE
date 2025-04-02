import os
import json
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from pycocotools import mask as mask_utils

def load_coco_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    annotations_dict = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        file_name = img_info['file_name']
        if file_name not in annotations_dict:
            annotations_dict[file_name] = []
        annotations_dict[file_name].append(ann)
    
    return annotations_dict, coco_data['categories']

def decode_rle_segmentation(segmentation):
    """Dekoduje RLE do maski binarnej"""
    rle = {
        "counts": segmentation["counts"].encode('utf-8'),
        "size": segmentation["size"]
    }
    return mask_utils.decode(rle)

def combine_masks_to_image_size(masks, bboxes, image_shape):
    """Łączy wszystkie maski w jedną maskę o rozmiarze obrazu"""
    combined_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    
    for mask, bbox in zip(masks, bboxes):
        if mask is None:
            continue
        x, y, w, h = map(int, bbox)
        if w <= 0 or h <= 0:
            continue
        
        # Przeskaluj maskę do rozmiaru bboxa, jeśli to potrzebne
        mask_h, mask_w = mask.shape
        if mask_h != h or mask_w != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Umieść maskę w odpowiednim miejscu na pełnowymiarowej masce
        x_end = min(x + w, image_shape[1])
        y_end = min(y + h, image_shape[0])
        combined_mask[y:y_end, x:x_end] |= mask[:y_end - y, :x_end - x]
    
    return combined_mask

def overlay_mask_on_image(image, mask):
    """Nakłada jedną maskę na obraz"""
    image[mask > 0] = [255, 0, 0]  # Czerwony kolor dla maski
    return image

def get_augmentation_pipeline(image_height, image_width):
    """Tworzy pipeline augmentacji z dynamicznym przycinaniem"""
    crop_height = min(512, image_height)
    crop_width = min(512, image_width)
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Odbicie w pionie
        A.RandomBrightnessContrast(p=0.3),  # Losowa zmiana jasności i kontrastu
        A.Rotate(limit=30, p=0.5),  # Obrót o maksymalnie 30 stopni
        A.RandomCrop(height=crop_height, width=crop_width, p=0.3),  # Losowy przycięcie
        A.GaussNoise(p=0.2),  # Szum Gaussa
        
        # Dodatkowe transformacje
        A.Blur(blur_limit=(3, 7), p=0.2),  # Rozmazanie
        A.MedianBlur(blur_limit=5, p=0.1),  # Rozmazanie medianowe
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.2),  # Szum mnożący
        A.ChannelShuffle(p=0.1),  # Losowa permutacja kanałów
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),  # Zmiana kolorów
        A.ISONoise(p=0.1),  # Szum ISO
    ],
    bbox_params=A.BboxParams(
        format='coco', 
        label_fields=['category_ids'], 
        min_area=3,           # Minimalna powierzchnia bboxa
        min_visibility=0.1     # Minimalna widoczność bboxa
    ),
    additional_targets={'mask': 'mask'})

def augment_and_save(image_path, annotations, output_dir, num_augmentations, categories):
    # Wczytaj obraz
    image = cv2.imread(image_path)
    if image is None:
        print(f"Nie można wczytać obrazu: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Przygotuj dane do augmentacji
    bboxes = [ann['bbox'] for ann in annotations]
    category_ids = [ann['category_id'] for ann in annotations]
    masks = [decode_rle_segmentation(ann['segmentation']) if 'segmentation' in ann else None 
             for ann in annotations]
    
    # Połącz maski w jedną maskę o rozmiarze obrazu
    combined_mask = combine_masks_to_image_size(masks, bboxes, image.shape)
    
    # Przygotuj dane do augmentacji
    valid_bboxes = [bbox for bbox, mask in zip(bboxes, masks) if mask is not None]
    valid_category_ids = [cat_id for cat_id, mask in zip(category_ids, masks) if mask is not None]
    
    # Pipeline augmentacji
    aug_pipeline = get_augmentation_pipeline(image.shape[0], image.shape[1])
    
    # Oryginalna nazwa pliku
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Zapisz oryginalny obraz z adnotacjami
    output_image_path = os.path.join(output_dir, f"{base_name}_orig.jpg")
    vis_image = image.copy()
    for bbox in bboxes:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    vis_image = overlay_mask_on_image(vis_image, combined_mask)
    cv2.imwrite(output_image_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    # Generuj augmentacje
    for i in range(num_augmentations):
        aug_data = {
            'image': image,
            'bboxes': valid_bboxes,
            'category_ids': valid_category_ids,
            'mask': combined_mask
        }
        augmented = aug_pipeline(**aug_data)
        
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_mask = augmented['mask']
        aug_category_ids = augmented['category_ids']
        
        # Filtruj bboxy i maski wychodzące poza obraz
        filtered_bboxes = []
        filtered_category_ids = []
        filtered_mask = np.zeros_like(aug_mask)
        
        height, width = aug_image.shape[:2]
        
        for bbox, cat_id in zip(aug_bboxes, aug_category_ids):
            x, y, w, h = map(int, bbox)
            
            # Sprawdź czy bbox jest w granicach obrazu
            if (x >= width or y >= height or 
                x + w <= 0 or y + h <= 0):
                continue
                
            # Przytnij współrzędne do granic obrazu
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w > 0 and h > 0:
                filtered_bboxes.append([x, y, w, h])
                filtered_category_ids.append(cat_id)
        
        # Aktualizuj maskę tylko dla widocznych obszarów
        if len(filtered_bboxes) > 0:
            # Przytnij maskę do granic obrazu
            filtered_mask[:height, :width] = aug_mask[:height, :width]
        
        # Wizualizacja augmentowanego obrazu
        vis_aug_image = aug_image.copy()
        for bbox in filtered_bboxes:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(vis_aug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        vis_aug_image = overlay_mask_on_image(vis_aug_image, filtered_mask)
        
        # Zapisz augmentowany obraz
        output_image_path = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
        cv2.imwrite(output_image_path, cv2.cvtColor(vis_aug_image, cv2.COLOR_RGB2BGR))

def create_augmented_dataset(dataset_dir, output_dir, num_augmentations):
    images_dir = os.path.join(dataset_dir, "train", "images")
    annotation_path = os.path.join(dataset_dir, "train", "annotations", "coco.json")
    
    if not os.path.exists(images_dir) or not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Nie znaleziono katalogu {images_dir} lub pliku {annotation_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    annotations_dict, categories = load_coco_annotations(annotation_path)
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Augmenting images"):
        image_path = os.path.join(images_dir, image_file)
        if image_file in annotations_dict:
            augment_and_save(
                image_path=image_path,
                annotations=annotations_dict[image_file],
                output_dir=output_dir,
                num_augmentations=num_augmentations,
                categories=categories
            )
        else:
            print(f"Brak adnotacji dla {image_file}, pomijam.")

if __name__ == "__main__":
    dataset_dir = "../data"
    output_dir = "../data/test/data_augmented"
    num_augmentations = 20
    
    create_augmented_dataset(dataset_dir, output_dir, num_augmentations)