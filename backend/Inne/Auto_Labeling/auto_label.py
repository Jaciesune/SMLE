import os
import glob
import cv2
import torch
import numpy as np
import json
import base64
import shutil

# Konfiguracja
MODEL_PATH = "./_Mask_RCNN_models/model3.pth"  # Ścieżka do gotowego modelu
INPUT_DIR = "./_images_before"  # Katalog z obrazami wejściowymi (*.jpg)
OUTPUT_JSON_DIR = "./images_annotations"  # Katalog dla adnotacji JSON i skopiowanych obrazów
DEBUG_IMAGE_DIR = "./images_debug"  # Katalog dla obrazów z nałożonymi boxami i maskami
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Urządzenie (GPU/CPU)
CONFIDENCE_THRESHOLD = 0.7  # Obniżony próg pewności dla detekcji
MODEL_INPUT_SIZE = (1024, 1024)  # Wymiary, na których model był trenowany

# Wczytaj gotowy model
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.to(DEVICE)
model.eval()

def create_directories():
    """Tworzy katalogi wyjściowe, jeśli nie istnieją."""
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)

def preprocess_image(image_path):
    """Wczytuje, skaluje i przygotowuje obraz do predykcji."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image.shape[:2]
    print(f"Original dimensions for {image_path}: {orig_width}x{orig_height}")

    # Skaluj obraz do wymiarów, na których model był trenowany (1024x1024)
    scale_x = MODEL_INPUT_SIZE[0] / orig_width
    scale_y = MODEL_INPUT_SIZE[1] / orig_height
    image_resized = cv2.resize(image, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    print(f"Resized dimensions: {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]}")

    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    return image_tensor, image, orig_width, orig_height, scale_x, scale_y

def predict(image_tensor):
    """Wykonuje predykcję za pomocą modelu."""
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return predictions

def rescale_boxes(boxes, scale_x, scale_y, orig_width, orig_height):
    """Przeskalowuje bboxy z powrotem do oryginalnych wymiarów obrazu i koryguje krawędzie."""
    boxes = boxes.cpu().numpy()
    boxes[:, [0, 2]] /= scale_x  # Skaluj x_min i x_max
    boxes[:, [1, 3]] /= scale_y  # Skaluj y_min i y_max

    # Korekcja bboxów, aby sięgały krawędzi, jeśli są blisko
    edge_threshold = 10  # Piksele - jeśli bbox jest bliżej niż 10 pikseli od krawędzi, rozszerz go
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i]
        if x_min < edge_threshold:
            boxes[i][0] = 0
        if y_min < edge_threshold:
            boxes[i][1] = 0
        if (orig_width - x_max) < edge_threshold:
            boxes[i][2] = orig_width - 1
        if (orig_height - y_max) < edge_threshold:
            boxes[i][3] = orig_height - 1

    return boxes

def rescale_masks(masks, orig_width, orig_height):
    """Przeskalowuje maski z powrotem do oryginalnych wymiarów obrazu."""
    masks_resized = []
    for mask in masks:
        mask_np = (mask.cpu().numpy()[0] > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask_np, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        mask_resized = (mask_resized > 0.5).astype(np.uint8)
        masks_resized.append(mask_resized)
    return masks_resized

def crop_mask_to_bbox(mask, box, orig_width, orig_height, image_name, idx):
    """Przycina maskę do obszaru bboxa i zwraca maskę o wymiarach bboxa."""
    mask_np = mask  # [H, W]

    x_min, y_min, x_max, y_max = map(int, box)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(orig_width - 1, x_max)
    y_max = min(orig_height - 1, y_max)

    # Nie odrzucaj bboxów, które mają zerową szerokość/wysokość - spróbuj przyciąć widoczną część
    if x_max <= x_min or y_max <= y_min:
        print(f"Nieprawidłowy bbox dla {image_name}, obiekt {idx}: ({x_min}, {y_min}, {x_max}, {y_max}) - próbuję przyciąć widoczną część")
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(orig_width - 1, x_max), min(orig_height - 1, y_max)
        if x_max <= x_min or y_max <= y_min:
            return None

    cropped_mask = mask_np[y_min:y_max, x_min:x_max]
    return cropped_mask

def encode_mask_to_base64(cropped_mask):
    """Koduje przyciętą maskę do formatu base64 (PNG)."""
    success, encoded_image = cv2.imencode('.png', cropped_mask * 255)  # LabelMe oczekuje 0/255
    if success:
        return base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    return None

def save_image_with_annotations(image_name, image, predictions, orig_width, orig_height, scale_x, scale_y):
    """Zapisuje obraz z nałożonymi bounding boxami i maskami w folderze images_debug."""
    annotated_image = image.copy()
    boxes = rescale_boxes(predictions['boxes'], scale_x, scale_y, orig_width, orig_height)
    masks = rescale_masks(predictions.get('masks', []), orig_width, orig_height)

    for idx, (box, score, mask) in enumerate(zip(boxes, predictions['scores'], masks)):
        print(f"Score for {image_name}: {score:.2f}")
        if score >= CONFIDENCE_THRESHOLD:
            x_min, y_min, x_max, y_max = map(int, box)
            print(f"Raw bbox for {image_name}: ({x_min}, {y_min}, {x_max}, {y_max})")
            # Usunięto margines - bboxy są używane bez rozszerzania
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.putText(annotated_image, f"Score: {score:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Maska - przycinamy do bboxa
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(orig_width, x_max)
            y_max = min(orig_height, y_max)

            if x_max <= x_min or y_max <= y_min:
                continue

            cropped_mask = mask[y_min:y_max, x_min:x_max]
            full_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            full_mask[y_min:y_max, x_min:x_max] = cropped_mask

            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_image, contours, -1, (0, 0, 255), 2)

    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    image_path = os.path.join(DEBUG_IMAGE_DIR, f"{image_name}_annotated.jpg")
    cv2.imwrite(image_path, annotated_image_bgr)

def save_labelme_json(image_path, image_name, predictions, orig_width, orig_height, scale_x, scale_y):
    """Zapisuje adnotacje w formacie JSON zgodnym z LabelMe."""
    json_data = {
        "version": "5.8.1",
        "flags": {},
        "shapes": [],
        "imagePath": f"{image_name}.jpg",
        "imageData": None,
        "imageHeight": orig_height,
        "imageWidth": orig_width
    }

    boxes = rescale_boxes(predictions['boxes'], scale_x, scale_y, orig_width, orig_height)
    masks = rescale_masks(predictions.get('masks', []), orig_width, orig_height)

    for idx, (box, score, mask) in enumerate(zip(boxes, predictions['scores'], masks)):
        if score >= CONFIDENCE_THRESHOLD:
            x_min, y_min, x_max, y_max = box
            # Usunięto margines - bboxy są używane bez rozszerzania
            cropped_mask = crop_mask_to_bbox(mask, box, orig_width, orig_height, image_name, idx)
            if cropped_mask is None:
                continue

            mask_base64 = encode_mask_to_base64(cropped_mask)
            if mask_base64 is None:
                print(f"Nie udało się zakodować maski dla {image_name}, obiekt {idx}")
                continue

            shape = {
                "label": "pipe",
                "points": [
                    [float(x_min), float(y_min)],
                    [float(x_max), float(y_max)]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "mask",
                "flags": {},
                "mask": mask_base64
            }
            json_data["shapes"].append(shape)

    json_path = os.path.join(OUTPUT_JSON_DIR, f"{image_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

def auto_label_images():
    """Główna funkcja do automatycznego labelowania obrazów."""
    create_directories()
    image_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))

    if not image_files:
        print(f"Brak obrazów w {INPUT_DIR}")
        return

    print(f"Znaleziono {len(image_files)} obrazów do labelowania...")

    for idx, image_path in enumerate(image_files):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Przetwarzanie: {image_name} ({idx + 1}/{len(image_files)})")

        output_image_path = os.path.join(OUTPUT_JSON_DIR, f"{image_name}.jpg")
        shutil.copy(image_path, output_image_path)

        image_tensor, orig_image, orig_width, orig_height, scale_x, scale_y = preprocess_image(image_path)
        predictions = predict(image_tensor)
        save_labelme_json(image_path, image_name, predictions, orig_width, orig_height, scale_x, scale_y)
        save_image_with_annotations(image_name, orig_image, predictions, orig_width, orig_height, scale_x, scale_y)

    print("Labelowanie zakończone!")

if __name__ == "__main__":
    auto_label_images()