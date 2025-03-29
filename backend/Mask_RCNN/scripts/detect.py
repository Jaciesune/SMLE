import torch
import torchvision.transforms as T
import os
import cv2
from PIL import Image
import numpy as np

# Ścieżki
MODEL_PATH = "../models/mask_rcnn_50epok_10eugment.pth"
TEST_IMAGES_PATH = "../data/test/images"
RESULTS_PATH = "../data/test/results"

# Parametry wykrywania
CONFIDENCE_THRESHOLD = 0.5
DETECTIONS_PER_IMG = 500  # Maksymalna liczba predykcji na obraz

# Ładowanie modelu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)  # Załaduj model na właściwe urządzenie
    model.to(DEVICE)  # Przenieś model na GPU/CPU
    model.roi_heads.detections_per_img = DETECTIONS_PER_IMG  # Ustaw maksymalną liczbę detekcji na obraz
    model.eval()
    return model

# Przetwarzanie obrazu
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Zapamiętaj oryginalny rozmiar (szerokość, wysokość)

    transform = T.Compose([
        T.Resize((1024, 1024)),  # Zmiana rozmiaru obrazu
        T.ToTensor()
    ])
    tensor = transform(image).to(DEVICE)
    return [tensor], image, original_size 

# Detekcja obiektów
def detect_objects(model, image_tensor_list):
    with torch.no_grad():
        predictions = model(image_tensor_list)[0]  # Model oczekuje listy tensorów
    return predictions

# Wizualizacja wyników
def draw_predictions(image, predictions, confidence_threshold, original_size, min_area=1, min_visibility=0.01):
    original_width, original_height = original_size
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detections_count = 0

    for box, score, mask in zip(predictions["boxes"], predictions["scores"], predictions["masks"]):
        if score > confidence_threshold:
            # Przeskalowanie współrzędnych bboxa
            x1, y1, x2, y2 = map(int, (box * torch.tensor([original_width/1024, original_height/1024, 
                                             original_width/1024, original_height/1024]).to(box.device)).tolist())

            # Przygotowanie maski
            mask_np = mask[0].cpu().numpy()
            mask_resized = cv2.resize(mask_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            mask_thresholded = ((mask_resized > 0.5) * 255).astype(np.uint8)

            # Wycinanie obszaru maski odpowiadającego bboxowi
            mask_roi = mask_thresholded[y1:y2, x1:x2]
            
            # Filtracja małych detekcji
            area = np.sum(mask_roi > 0)
            if area < min_area or (area / ((x2-x1)*(y2-y1)+1e-6)) < min_visibility:
                continue

            # Nakładanie maski
            mask_color = np.zeros_like(image)
            try:
                mask_color[y1:y2, x1:x2, 2] = mask_roi  # Kanał czerwony
                image = cv2.addWeighted(image, 1, mask_color, 0.5, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detections_count += 1
            except ValueError:
                continue

    print(f"Liczba wykrytych obiektów: {detections_count}")
    return image




# Tworzenie katalogu wyników
os.makedirs(RESULTS_PATH, exist_ok=True)

# Wczytanie modelu
model = load_model(MODEL_PATH)

# Przetwarzanie obrazów testowych
for image_name in os.listdir(TEST_IMAGES_PATH):
    image_path = os.path.join(TEST_IMAGES_PATH, image_name)
    image_tensor_list, original_image, original_size = preprocess_image(image_path)
    predictions = detect_objects(model, image_tensor_list)
    result_image = draw_predictions(original_image, predictions, CONFIDENCE_THRESHOLD, original_size, min_area=1, min_visibility=0.01)

    # Zapis wyników
    result_path = os.path.join(RESULTS_PATH, image_name)
    cv2.imwrite(result_path, result_image)
    print(f"Zapisano wynik: {result_path}")

print("Wykrywanie zakończone!")
