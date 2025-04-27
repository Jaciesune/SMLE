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
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.RandomCrop(height=crop_height, width=crop_width, p=0.3),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=(3, 7), p=0.2),
        A.MedianBlur(blur_limit=5, p=0.1),
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.2),
        A.ChannelShuffle(p=0.1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.ISONoise(p=0.1),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
        A.RandomResizedCrop(
            size=(crop_height, crop_width),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            p=0.5
        ),
        A.Affine(
            scale=(0.7, 1.3),
            translate_percent=(-0.2, 0.2),
            rotate=(-45, 45),
            shear=(-10, 10),
            p=0.5
        ),
        A.MotionBlur(blur_limit=(3, 15), p=0.2),
        A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.2),
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.3
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_limit=(-30, 30),
            num_flare_circles_lower=1,
            num_flare_circles_upper=3,
            src_radius=150,
            p=0.2
        ),
        A.Resize(height=image_height, width=image_width),
    ],
    bbox_params=A.BboxParams(
        format='coco', 
        label_fields=['category_ids'], 
        min_area=3,
        min_visibility=0.1
    ),
    additional_targets={'mask': 'mask'})

def augment_and_save(image_path, annotations, output_dir, num_augmentations, categories):
    # Wczytaj obraz
    image = cv2.imread(image_path)
    if image is None:
        print(f"Nie można wczytać obrazu: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image.shape[:2]
    
    # Przygotuj dane do augmentacji
    bboxes = [ann['bbox'] for ann in annotations]
    category_ids = [ann['category_id'] for ann in annotations]
    masks = [decode_rle_segmentation(ann['segmentation']) if 'segmentation' in ann else None 
             for ann in annotations]
    
    # Połącz maski w jedną maskę o rozmiarze obrazu
    combined_mask = combine_masks_to_image_size(masks, bboxes, image.shape)
    
    # Przygotuj dane do augmentacji
    # Zachowaj wszystkie bboxy i kategorie, nawet jeśli maska jest None
    valid_bboxes = bboxes
    valid_category_ids = category_ids
    # Jeśli nie ma żadnej maski, utwórz pustą maskę
    if not any(mask is not None for mask in masks):
        combined_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
    
    # Logowanie dla debugowania
    # print(f"Oryginalne bboxy dla {os.path.basename(image_path)}: {valid_bboxes}")
    
    # Pipeline augmentacji
    aug_pipeline = get_augmentation_pipeline(orig_height, orig_width)
    
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
        
        # Logowanie augmentowanych bboxów
        print(f"Augmentowane bboxy dla {base_name}_aug_{i}: {aug_bboxes}")
        
        # Filtruj bboxy
        filtered_bboxes = []
        filtered_category_ids = []
        filtered_mask = np.zeros_like(aug_mask)
        
        height, width = aug_image.shape[:2]
        
        for bbox, cat_id in zip(aug_bboxes, aug_category_ids):
            x, y, w, h = map(int, bbox)
            
            # Przytnij współrzędne do granic obrazu
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            x_end = max(0, min(x + w, width - 1))
            y_end = max(0, min(y + h, height - 1))
            
            # Oblicz nowe wymiary bboxa
            w = x_end - x
            h = y_end - y
            
            # Zachowaj bbox tylko jeśli ma dodatnią szerokość i wysokość
            if w > 0 and h > 0:
                filtered_bboxes.append([x, y, w, h])
                filtered_category_ids.append(cat_id)
            else:
                print(f"Odrzucono bbox w {base_name}_aug_{i}: x={x}, y={y}, w={w}, h={h}")
        
        # Aktualizuj maskę tylko dla widocznych obszarów
        if len(filtered_bboxes) > 0:
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
    annotation_path = os.path.join(dataset_dir, "train", "annotations", "instances_train.json")
    
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
    dataset_dir = "../../data"
    output_dir = "../data/test/data_augmented"
    num_augmentations = 20
    
    create_augmented_dataset(dataset_dir, output_dir, num_augmentations)