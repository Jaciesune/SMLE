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
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ścieżki i parametry
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 1000
DETECTIONS_PER_IMAGE = 500
NUM_CLASSES = 2
MODEL_INPUT_SIZE = (1024, 1024)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    logger.debug(f"Sprawdzam model pod ścieżką: {model_path}")
    if not model_path.endswith('_checkpoint.pth'):
        logger.error(f"Ścieżka do modelu {model_path} nie kończy się na _checkpoint.pth.")
        sys.exit(1)

    if not os.path.exists(model_path):
        logger.error(f"Plik modelu {model_path} nie istnieje.")
        sys.exit(1)

    logger.debug(f"Tworzę instancję modelu Mask R-CNN...")
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_channels=256, dim_reduced=256, num_classes=NUM_CLASSES
        )
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modelu: {e}")
        sys.exit(1)

    start_time = time.time()
    logger.debug(f"Wczytuję checkpoint z {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania checkpointu: {e}")
        sys.exit(1)

    logger.debug(f"Konfiguruję parametry modelu...")
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
        logger.error(f"Błąd podczas konfiguracji modelu: {e}")
        sys.exit(1)

    end_time = time.time()
    logger.debug(f"Czas wczytywania modelu: {end_time - start_time:.2f} sekund")
    return model

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Błąd podczas otwierania obrazu {image_path}: {e}")
        raise
    original_size = image.size

    transform = T.Compose([
        T.Resize(MODEL_INPUT_SIZE),
        T.ToTensor()
    ])
    tensor = transform(image).to(DEVICE)
    return [tensor], image, original_size[0], original_size[1]

def predict(model, image_tensor_list):
    try:
        with torch.no_grad():
            predictions = model(image_tensor_list)[0]
    except Exception as e:
        logger.error(f"Błąd podczas detekcji: {e}")
        raise
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
        logger.warning(f"Nieprawidłowy bbox dla {image_name}, obiekt {idx}: ({x_min}, {y_min}, {x_max}, {y_max})")
        return None
    cropped_mask = mask_np[y_min:y_max, x_min:x_max]
    return cropped_mask

def encode_mask_to_base64(cropped_mask):
    success, encoded_image = cv2.imencode('.png', cropped_mask * 255)
    if success:
        return base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    logger.error("Błąd podczas kodowania maski do base64.")
    return None

def save_labelme_json(image_path, image_name, predictions, orig_width, orig_height, output_dir, custom_label):
    output_image_path = os.path.join(output_dir, f"{image_name}.jpg")
    try:
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        success = cv2.imwrite(output_image_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        if not success:
            logger.error(f"Nie udało się zapisać obrazu {output_image_path}.")
            return
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania obrazu do {output_image_path}: {e}")
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
    labels = predictions['labels'].cpu().numpy()

    for idx, (box, score, mask, label_idx) in enumerate(zip(boxes, predictions['scores'], masks, labels)):
        if score >= CONFIDENCE_THRESHOLD:
            x_min, y_min, x_max, y_max = box
            cropped_mask = crop_mask_to_bbox(mask, box, orig_width, orig_height, image_name, idx)
            if cropped_mask is None:
                logger.warning(f"Pominięto maskę dla {image_name}, obiekt {idx}.")
                continue

            mask_base64 = encode_mask_to_base64(cropped_mask)
            if mask_base64 is None:
                logger.warning(f"Nie udało się zakodować maski dla {image_name}, obiekt {idx}.")
                continue

            if label_idx == 0:
                logger.debug(f"Pominięto obiekt {idx} w {image_name} - etykieta to 'background'.")
                continue

            shape = {
                "label": custom_label,
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
            # logger.debug(f"Dodano kształt dla {image_name}, obiekt {idx}: etykieta={custom_label}, score={score:.2f}")

    json_path = os.path.join(output_dir, f"{image_name}.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        logger.debug(f"Zapisano adnotacje JSON do: {json_path}")
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania JSON do {json_path}: {e}")

def main():
    logger.info("Rozpoczynam automatyczne labelowanie...")
    parser = argparse.ArgumentParser(description="Automatyczne labelowanie katalogu zdjęć przy użyciu Mask R-CNN")
    parser.add_argument("--input_dir", type=str, required=True, help="Ścieżka do katalogu z obrazami wejściowymi")
    parser.add_argument("--output_dir", type=str, required=True, help="Ścieżka do katalogu na obrazy z detekcjami")
    parser.add_argument("--debug_dir", type=str, default="", help="Ścieżka do katalogu na obrazy z adnotacjami (opcjonalne)")
    parser.add_argument("--model_path", type=str, required=True, help="Ścieżka do pliku modelu z końcówką _checkpoint.pth")
    parser.add_argument("--custom_label", type=str, required=True, help="Etykieta do użycia dla wykrytych obiektów")
    args = parser.parse_args()

    logger.debug(f"Argumenty: input_dir={args.input_dir}, output_dir={args.output_dir}, debug_dir={args.debug_dir}, model_path={args.model_path}, custom_label={args.custom_label}")

    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia katalogu wyjściowego: {e}")
        sys.exit(1)

    try:
        model = load_model(args.model_path)
    except Exception as e:
        logger.error(f"Błąd wczytywania modelu: {e}")
        sys.exit(1)

    image_paths = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    if not image_paths:
        logger.error(f"Brak obrazów .jpg w katalogu {args.input_dir}")
        sys.exit(1)

    total_detections = 0
    processed_images = 0
    for image_path in image_paths:
        logger.debug(f"Przetwarzam obraz: {image_path}")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        try:
            image_tensor_list, image, orig_width, orig_height = preprocess_image(image_path)
            predictions = predict(model, image_tensor_list)
            save_labelme_json(image_path, image_name, predictions, orig_width, orig_height, args.output_dir, args.custom_label)
            total_detections += len([s for s in predictions['scores'] if s >= CONFIDENCE_THRESHOLD])
            processed_images += 1
            logger.debug(f"Detections dla {image_path}: {len([s for s in predictions['scores'] if s >= CONFIDENCE_THRESHOLD])}")
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania obrazu {image_path}: {e}")
            continue

    output_files = os.listdir(args.output_dir)
    logger.debug(f"Zawartość katalogu wyjściowego {args.output_dir}: {output_files}")

    if not output_files:
        logger.warning(f"Katalog wyjściowy {args.output_dir} jest pusty!")

    logger.info(f"Przetworzono obrazów: {processed_images}, Całkowita liczba detekcji: {total_detections}")
    logger.info("Automatyczne labelowanie zakończone!")

if __name__ == "__main__":
    main()