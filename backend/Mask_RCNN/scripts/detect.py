import torch
import torchvision.transforms.v2 as T  # Zmiana na v2
import torchvision.models.detection
import os
import cv2
from PIL import Image
import numpy as np
import time
import sys

# Ścieżki
RESULTS_PATH = "/app/backend/Mask_RCNN/data/detectes"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Parametry wykrywania
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 1000
DETECTIONS_PER_IMAGE = 500
NUM_CLASSES = 2
MODEL_INPUT_SIZE = (1024, 1024)

# Ładowanie modelu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    if not model_path.endswith('_checkpoint.pth'):
        print(f"Błąd: Ścieżka do modelu {model_path} nie kończy się na _checkpoint.pth.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Błąd: Plik modelu {model_path} nie istnieje.")
        sys.exit(1)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=256, dim_reduced=256, num_classes=NUM_CLASSES
    )

    start_time = time.time()
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if isinstance(model.rpn.pre_nms_top_n, dict):
        model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
    model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD
    model.roi_heads.detections_per_img = DETECTIONS_PER_IMAGE

    model.to(DEVICE)
    model.eval()
    end_time = time.time()
    print(f"Czas wczytywania modelu: {end_time - start_time:.2f} sekund")
    return model

# Przetwarzanie obrazu
def preprocess_image(image_path):
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    transform = T.Compose([
        T.Resize(MODEL_INPUT_SIZE, antialias=True),  # Dodano antialias=True dla lepszej jakości
        T.ToImage(),  # Konwersja do formatu obrazu
        T.ToDtype(torch.float32, scale=True),  # Normalizacja do [0, 1]
    ])
    tensor = transform(image).to(DEVICE)
    end_time = time.time()
    print(f"Czas przetwarzania obrazu {image_path}: {end_time - start_time:.2f} sekund")
    return [tensor], image, original_size 

# Detekcja obiektów
def detect_objects(model, image_tensor_list):
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image_tensor_list)[0]
    end_time = time.time()
    print(f"Czas detekcji: {end_time - start_time:.2f} sekund")
    return predictions

# Rysowanie wyników
def draw_results(image, predictions, original_size):
    image_np = np.array(image)
    image_np = cv2.resize(image_np, original_size)
    
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    masks = predictions['masks'].cpu().numpy() if 'masks' in predictions else None

    valid_indices = scores >= CONFIDENCE_THRESHOLD
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]
    if masks is not None:
        masks = masks[valid_indices]

    orig_width, orig_height = original_size
    model_width, model_height = MODEL_INPUT_SIZE
    scale_x = orig_width / model_width
    scale_y = orig_height / model_height

    scaled_boxes = boxes.copy()
    scaled_boxes[:, 0] *= scale_x
    scaled_boxes[:, 1] *= scale_y
    scaled_boxes[:, 2] *= scale_x
    scaled_boxes[:, 3] *= scale_y

    detections_count = 0
    for i, (box, score, label) in enumerate(zip(scaled_boxes, scores, labels)):
        if detections_count >= DETECTIONS_PER_IMAGE:
            break
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if masks is not None:
            mask = masks[i, 0] > 0.5
            mask = cv2.resize(mask.astype(np.uint8), original_size) * 255
            image_np[mask > 0] = [0, 0, 255]
        
        detections_count += 1

    return image_np, detections_count

# Zapis wyniku
def save_result(image_np, image_path):
    result_image_name = os.path.splitext(os.path.basename(image_path))[0] + "_detected.jpg"
    result_path = os.path.join(RESULTS_PATH, result_image_name)
    print(f"Zapisuję wynik detekcji do: {result_path}")
    cv2.imwrite(result_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return result_path

# Główna funkcja
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Użycie: python detect.py <ścieżka_do_obrazu> <ścieżka_do_modelu>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]

    model = load_model(model_path)
    image_tensor_list, image, original_size = preprocess_image(image_path)
    predictions = detect_objects(model, image_tensor_list)
    image_with_detections, detections_count = draw_results(image, predictions, original_size)
    result_path = save_result(image_with_detections, image_path)
    print(f"Detections: {detections_count}")