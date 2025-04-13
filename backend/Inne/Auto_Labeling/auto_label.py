import torch
import torchvision.transforms as T
import torchvision.models.detection
import os
import cv2
from PIL import Image
import numpy as np
import time
import sys
import argparse
import glob
import json
import base64
import shutil

# Ścieżki i parametry
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 1000
DETECTIONS_PER_IMAGE = 500
NUM_CLASSES = 2
MODEL_INPUT_SIZE = (1024, 1024)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    print(f"Sprawdzam model pod ścieżką: {model_path}", flush=True)
    if not model_path.endswith('_checkpoint.pth'):
        print(f"Błąd: Ścieżka do modelu {model_path} nie kończy się na _checkpoint.pth.", flush=True)
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Błąd: Plik modelu {model_path} nie istnieje.", flush=True)
        sys.exit(1)

    print(f"Tworzę instancję modelu Mask R-CNN...", flush=True)
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_channels=256, dim_reduced=256, num_classes=NUM_CLASSES
        )
    except Exception as e:
        print(f"Błąd podczas tworzenia modelu: {e}", flush=True)
        sys.exit(1)

    start_time = time.time()
    print(f"Wczytuję checkpoint z {model_path}...", flush=True)
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Błąd podczas wczytywania checkpointu: {e}", flush=True)
        sys.exit(1)

    print(f"Konfiguruję parametry modelu...", flush=True)
    try:
        if isinstance(model.rpn.pre_nms_top_n, dict):
            model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
            model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
            model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
            model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
        model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD
        model.roi_heads.detections_per_img = DETECTIONS_PER_IMAGE
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Błąd podczas konfiguracji modelu: {e}", flush=True)
        sys.exit(1)

    end_time = time.time()
    print(f"Czas wczytywania modelu: {end_time - start_time:.2f} sekund", flush=True)
    return model

def preprocess_image(image_path):
    print(f"Przetwarzam obraz: {image_path}", flush=True)
    start_time = time.time()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Błąd podczas otwierania obrazu {image_path}: {e}", flush=True)
        raise
    original_size = image.size
    print(f"Wymiary oryginalne dla {image_path}: {original_size[0]}x{original_size[1]}", flush=True)

    transform = T.Compose([
        T.Resize(MODEL_INPUT_SIZE),
        T.ToTensor()
    ])
    tensor = transform(image).to(DEVICE)
    end_time = time.time()
    print(f"Czas przetwarzania obrazu {image_path}: {end_time - start_time:.2f} sekund", flush=True)
    return [tensor], image, original_size[0], original_size[1]

def predict(model, image_tensor_list):
    print(f"Rozpoczynam detekcję...", flush=True)
    start_time = time.time()
    try:
        with torch.no_grad():
            predictions = model(image_tensor_list)[0]
    except Exception as e:
        print(f"Błąd podczas detekcji: {e}", flush=True)
        raise
    end_time = time.time()
    print(f"Czas detekcji: {end_time - start_time:.2f} sekund", flush=True)
    return predictions

def rescale_boxes(boxes, orig_width, orig_height):
    boxes = boxes.cpu().numpy()
    scale_x = orig_width / MODEL_INPUT_SIZE[0]
    scale_y = orig_height / MODEL_INPUT_SIZE[1]
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    edge_threshold = 10
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
    masks_resized = []
    for mask in masks:
        mask_np = (mask.cpu().numpy()[0] > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask_np, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        mask_resized = (mask_resized > 0.5).astype(np.uint8)
        masks_resized.append(mask_resized)
    return masks_resized

def crop_mask_to_bbox(mask, box, orig_width, orig_height, image_name, idx):
    mask_np = mask
    x_min, y_min, x_max, y_max = map(int, box)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(orig_width - 1, x_max)
    y_max = min(orig_height - 1, y_max)

    if x_max <= x_min or y_max <= y_min:
        print(f"Nieprawidłowy bbox dla {image_name}, obiekt {idx}: ({x_min}, {y_min}, {x_max}, {y_max})", flush=True)
        return None
    cropped_mask = mask_np[y_min:y_max, x_min:x_max]
    return cropped_mask

def encode_mask_to_base64(cropped_mask):
    success, encoded_image = cv2.imencode('.png', cropped_mask * 255)
    if success:
        return base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    print("Błąd podczas kodowania maski do base64.", flush=True)
    return None

def save_image_with_annotations(image_name, image, predictions, orig_width, orig_height, debug_dir):
    print(f"Rysuję wyniki detekcji dla {image_name}...", flush=True)
    image_np = np.array(image)
    boxes = rescale_boxes(predictions['boxes'], orig_width, orig_height)
    masks = rescale_masks(predictions.get('masks', []), orig_width, orig_height)

    detections_count = 0
    for idx, (box, score, mask) in enumerate(zip(boxes, predictions['scores'], masks)):
        if score >= CONFIDENCE_THRESHOLD:
            x_min, y_min, x_max, y_max = map(int, box)
            # Usunięto: print(f"Raw bbox dla {image_name}: ({x_min}, {y_min}, {x_max}, {y_max})", flush=True)
            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.putText(image_np, f"Score: {score:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(orig_width, x_max)
            y_max = min(orig_height, y_max)

            if x_max <= x_min or y_max <= y_min:
                print(f"Pominięto nieprawidłowy bbox dla {image_name}, obiekt {idx}.", flush=True)
                continue

            cropped_mask = mask[y_min:y_max, x_min:x_max]
            full_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            full_mask[y_min:y_max, x_min:x_max] = cropped_mask

            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_np, contours, -1, (0, 0, 255), 2)

            detections_count += 1

    image_path = os.path.join(debug_dir, f"{image_name}_annotated.jpg")
    try:
        success = cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        if success:
            print(f"Zapisano obraz z adnotacjami do: {image_path}", flush=True)
        else:
            print(f"Błąd: Nie udało się zapisać obrazu {image_path}.", flush=True)
    except Exception as e:
        print(f"Błąd podczas zapisywania obrazu z adnotacjami {image_path}: {e}", flush=True)
    return detections_count

def save_labelme_json(image_path, image_name, predictions, orig_width, orig_height, output_dir):
    output_image_path = os.path.join(output_dir, f"{image_name}.jpg")
    try:
        shutil.copy(image_path, output_image_path)
        print(f"Skopiowano obraz do: {output_image_path}", flush=True)
    except Exception as e:
        print(f"Błąd podczas kopiowania obrazu do {output_image_path}: {e}", flush=True)
        return

    json_data = {
        "version": "5.8.1",
        "flags": {},
        "shapes": [],
        "imagePath": f"{image_name}.jpg",
        "imageData": None,
        "imageHeight": orig_height,
        "imageWidth": orig_width
    }

    boxes = rescale_boxes(predictions['boxes'], orig_width, orig_height)
    masks = rescale_masks(predictions.get('masks', []), orig_width, orig_height)

    for idx, (box, score, mask) in enumerate(zip(boxes, predictions['scores'], masks)):
        if score >= CONFIDENCE_THRESHOLD:
            x_min, y_min, x_max, y_max = box
            cropped_mask = crop_mask_to_bbox(mask, box, orig_width, orig_height, image_name, idx)
            if cropped_mask is None:
                print(f"Pominięto maskę dla {image_name}, obiekt {idx}.", flush=True)
                continue

            mask_base64 = encode_mask_to_base64(cropped_mask)
            if mask_base64 is None:
                print(f"Nie udało się zakodować maski dla {image_name}, obiekt {idx}.", flush=True)
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

    json_path = os.path.join(output_dir, f"{image_name}.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"Zapisano adnotacje JSON do: {json_path}", flush=True)
    except Exception as e:
        print(f"Błąd podczas zapisywania JSON do {json_path}: {e}", flush=True)

def main():
    print("Rozpoczynam automatyczne labelowanie...", flush=True)
    parser = argparse.ArgumentParser(description="Automatyczne labelowanie katalogu zdjęć przy użyciu Mask R-CNN")
    parser.add_argument("--input_dir", type=str, required=True, help="Ścieżka do katalogu z obrazami wejściowymi")
    parser.add_argument("--output_dir", type=str, required=True, help="Ścieżka do katalogu na obrazy z detekcjami")
    parser.add_argument("--debug_dir", type=str, required=True, help="Ścieżka do katalogu na obrazy z adnotacjami")
    parser.add_argument("--model_path", type=str, required=True, help="Ścieżka do pliku modelu z końcówką _checkpoint.pth")
    args = parser.parse_args()

    print(f"Argumenty: input_dir={args.input_dir}, output_dir={args.output_dir}, debug_dir={args.debug_dir}, model_path={args.model_path}", flush=True)

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.debug_dir, exist_ok=True)
    except Exception as e:
        print(f"Błąd podczas tworzenia katalogów: {e}", flush=True)
        sys.exit(1)

    try:
        model = load_model(args.model_path)
    except Exception as e:
        print(f"Błąd wczytywania modelu: {e}", flush=True)
        sys.exit(1)

    image_paths = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    if not image_paths:
        print(f"Brak obrazów .jpg w katalogu {args.input_dir}", flush=True)
        sys.exit(1)

    total_detections = 0
    processed_images = 0
    for image_path in image_paths:
        print(f"Przetwarzam obraz: {image_path}", flush=True)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        try:
            image_tensor_list, image, orig_width, orig_height = preprocess_image(image_path)
            predictions = predict(model, image_tensor_list)
            save_labelme_json(image_path, image_name, predictions, orig_width, orig_height, args.output_dir)
            detections_count = save_image_with_annotations(image_name, image, predictions, orig_width, orig_height, args.debug_dir)
            total_detections += detections_count
            processed_images += 1
            print(f"Detections dla {image_path}: {detections_count}", flush=True)
        except Exception as e:
            print(f"Błąd podczas przetwarzania obrazu {image_path}: {e}", flush=True)
            continue

    output_files = os.listdir(args.output_dir)
    debug_files = os.listdir(args.debug_dir)
    print(f"Zawartość katalogu wyjściowego {args.output_dir}: {output_files}", flush=True)
    print(f"Zawartość katalogu debug {args.debug_dir}: {debug_files}", flush=True)

    if not output_files:
        print(f"Ostrzeżenie: Katalog wyjściowy {args.output_dir} jest pusty!", flush=True)
    if not debug_files:
        print(f"Ostrzeżenie: Katalog debug {args.debug_dir} jest pusty!", flush=True)

    print(f"Przetworzono obrazów: {processed_images}, Całkowita liczba detekcji: {total_detections}", flush=True)
    print("Automatyczne labelowanie zakończone!", flush=True)

if __name__ == "__main__":
    main()