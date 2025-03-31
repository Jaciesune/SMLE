import torch
import torchvision.models.detection
import torchvision.transforms as T
import os
import cv2
from PIL import Image
import numpy as np
import time  # Dodajemy moduł time

# Ścieżki
MODEL_PATH = "../models/train_1.pth"
TEST_IMAGES_PATH = "../data/test/images"
RESULTS_PATH = "../data/test/results"

# Parametry wykrywania
CONFIDENCE_THRESHOLD = 0.7
DETECTIONS_PER_IMG = 500  # Maksymalna liczba predykcji na obraz

# Ładowanie modelu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=256, dim_reduced=256, num_classes=num_classes
    )
    model.to(device)
    return model

def load_model(model_path):
    start_time = time.time()  # Początek pomiaru
    model = get_model(num_classes=2, device=DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Wczytano state_dict z checkpointu: {model_path}")
    else:
        model = checkpoint
        print(f"Wczytano cały model: {model_path}")
    
    model.roi_heads.detections_per_img = DETECTIONS_PER_IMG
    model.eval()
    end_time = time.time()  # Koniec pomiaru
    print(f"Czas wczytywania modelu: {end_time - start_time:.2f} sekund")
    return model

# Przetwarzanie obrazu
def preprocess_image(image_path):
    start_time = time.time()  # Początek pomiaru
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor()
    ])
    tensor = transform(image).to(DEVICE)
    end_time = time.time()  # Koniec pomiaru
    print(f"Czas przetwarzania obrazu {image_path}: {end_time - start_time:.2f} sekund")
    return [tensor], image, original_size 

# Detekcja obiektów
def detect_objects(model, image_tensor_list):
    start_time = time.time()  # Początek pomiaru
    with torch.no_grad():
        predictions = model(image_tensor_list)[0]
    end_time = time.time()  # Koniec pomiaru
    print(f"Czas detekcji: {end_time - start_time:.2f} sekund")
    return predictions

# Wizualizacja wyników
def draw_predictions(image, predictions, confidence_threshold, original_size, min_area=1, min_visibility=0.01):
    start_time = time.time()  # Początek pomiaru
    original_width, original_height = original_size
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detections_count = 0

    for box, score, mask in zip(predictions["boxes"], predictions["scores"], predictions["masks"]):
        if score > confidence_threshold:
            x1, y1, x2, y2 = map(int, (box * torch.tensor([original_width/1024, original_height/1024, 
                                             original_width/1024, original_height/1024]).to(box.device)).tolist())

            mask_np = mask[0].cpu().numpy()
            mask_resized = cv2.resize(mask_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            mask_thresholded = ((mask_resized > 0.5) * 255).astype(np.uint8)

            mask_roi = mask_thresholded[y1:y2, x1:x2]
            area = np.sum(mask_roi > 0)
            if area < min_area or (area / ((x2-x1)*(y2-y1)+1e-6)) < min_visibility:
                continue

            mask_color = np.zeros_like(image)
            try:
                mask_color[y1:y2, x1:x2, 2] = mask_roi
                image = cv2.addWeighted(image, 1, mask_color, 0.5, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detections_count += 1
            except ValueError:
                continue

    end_time = time.time()  # Koniec pomiaru
    print(f"Liczba wykrytych obiektów: {detections_count}")
    print(f"Czas wizualizacji: {end_time - start_time:.2f} sekund")
    return image

# Tworzenie katalogu wyników
os.makedirs(RESULTS_PATH, exist_ok=True)

# Wczytanie modelu
start_total_time = time.time()  # Początek całkowitego pomiaru
model = load_model(MODEL_PATH)

# Przetwarzanie obrazów testowych
for image_name in os.listdir(TEST_IMAGES_PATH):
    image_path = os.path.join(TEST_IMAGES_PATH, image_name)
    
    # Przetwarzanie obrazu
    image_tensor_list, original_image, original_size = preprocess_image(image_path)
    
    # Detekcja
    predictions = detect_objects(model, image_tensor_list)
    
    # Wizualizacja i zapis
    start_save_time = time.time()  # Początek zapisu
    result_image = draw_predictions(original_image, predictions, CONFIDENCE_THRESHOLD, original_size, min_area=1, min_visibility=0.01)
    result_path = os.path.join(RESULTS_PATH, image_name)
    cv2.imwrite(result_path, result_image)
    end_save_time = time.time()  # Koniec zapisu
    print(f"Czas zapisu wyniku {result_path}: {end_save_time - start_save_time:.2f} sekund")
    print(f"Zapisano wynik: {result_path}")

end_total_time = time.time()  # Koniec całkowitego pomiaru
print(f"Całkowity czas wykrywania: {end_total_time - start_total_time:.2f} sekund")
print("Wykrywanie zakończone!")