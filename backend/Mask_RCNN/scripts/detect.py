"""
Skrypt detekcji obiektów i segmentacji instancji z wykorzystaniem Mask R-CNN

Ten skrypt umożliwia wykrywanie obiektów na pojedynczych obrazach przy użyciu 
wytrenowanego modelu Mask R-CNN. Implementuje pełen potok detekcji: od wczytania 
modelu, przez przetwarzanie obrazu, detekcję obiektów, aż po wizualizację 
i zapis wyników.
"""

#######################
# Importy bibliotek
#######################
import torch                                # Framework PyTorch
import torchvision.transforms.v2 as T       # Transformacje obrazów (wersja v2)
import torchvision.models.detection         # Modele detekcji obiektów
import os                                   # Operacje na systemie plików
import cv2                                  # OpenCV do operacji na obrazach
from PIL import Image                       # Biblioteka do manipulacji obrazami
import numpy as np                          # Operacje na tablicach numerycznych
import time                                 # Pomiar czasu wykonania
import sys                                  # Operacje systemowe i argumenty

#######################
# Konfiguracja
#######################
# Ścieżki
RESULTS_PATH = "/app/backend/Mask_RCNN/data/detectes"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Parametry wykrywania
CONFIDENCE_THRESHOLD = 0.5     # Próg pewności dla detekcji
NMS_THRESHOLD = 1600           # Próg dla Non-Maximum Suppression
DETECTIONS_PER_IMAGE = 800     # Maksymalna liczba detekcji na obraz
NUM_CLASSES = 2                # Liczba klas (tło + 1 klasa obiektów)
MODEL_INPUT_SIZE = (1024, 1024)  # Rozmiar wejściowy obrazu dla modelu

# Ładowanie modelu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################
# Funkcje
#######################
def load_model(model_path):
    """
    Ładuje i konfiguruje model Mask R-CNN z pliku checkpointu.
    
    Args:
        model_path (str): Ścieżka do pliku checkpointu modelu.
        
    Returns:
        torch.nn.Module: Wczytany model w trybie ewaluacji.
        
    Raises:
        SystemExit: Gdy plik modelu nie istnieje lub ma niepoprawny format.
    """
    if not model_path.endswith('_checkpoint.pth'):
        print(f"Błąd: Ścieżka do modelu {model_path} nie kończy się na _checkpoint.pth.")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Błąd: Plik modelu {model_path} nie istnieje.")
        sys.exit(1)

    # Inicjalizacja modelu z predefiniowanymi wagami
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    
    # Dostosowanie głowicy klasyfikatora dla odpowiedniej liczby klas
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
    
    # Dostosowanie głowicy predyktora masek
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=256, dim_reduced=256, num_classes=NUM_CLASSES
    )

    # Wczytanie wag z checkpointu
    start_time = time.time()
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Konfiguracja parametrów detekcji
    if isinstance(model.rpn.pre_nms_top_n, dict):
        model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
        model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
    model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD
    model.roi_heads.detections_per_img = DETECTIONS_PER_IMAGE

    # Przeniesienie modelu na odpowiednie urządzenie i przełączenie w tryb ewaluacji
    model.to(DEVICE)
    model.eval()
    end_time = time.time()
    print(f"Czas wczytywania modelu: {end_time - start_time:.2f} sekund")
    return model

def preprocess_image(image_path):
    """
    Wczytuje i przygotowuje obraz do przetworzenia przez model.
    
    Args:
        image_path (str): Ścieżka do pliku obrazu.
        
    Returns:
        tuple: (tensor obrazu, oryginalny obraz, oryginalny rozmiar)
    """
    start_time = time.time()
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # Transformacja obrazu do formatu wymaganego przez model
    transform = T.Compose([
        T.Resize(MODEL_INPUT_SIZE, antialias=True),  # Dodano antialias=True dla lepszej jakości
        T.ToImage(),  # Konwersja do formatu obrazu
        T.ToDtype(torch.float32, scale=True),  # Normalizacja do [0, 1]
    ])
    tensor = transform(image).to(DEVICE)
    end_time = time.time()
    print(f"Czas przetwarzania obrazu {image_path}: {end_time - start_time:.2f} sekund")
    return [tensor], image, original_size 

def detect_objects(model, image_tensor_list):
    """
    Wykonuje detekcję obiektów na obrazie.
    
    Args:
        model: Model Mask R-CNN.
        image_tensor_list: Lista tensorów obrazu.
        
    Returns:
        dict: Słownik z wynikami detekcji (boxes, masks, labels, scores).
    """
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image_tensor_list)[0]
    end_time = time.time()
    print(f"Czas detekcji: {end_time - start_time:.2f} sekund")
    return predictions

def draw_results(image, predictions, original_size):
    """
    Rysuje wyniki detekcji na obrazie.
    
    Args:
        image: Oryginalny obraz PIL.
        predictions: Słownik z przewidywaniami modelu.
        original_size: Oryginalny rozmiar obrazu (szerokość, wysokość).
        
    Returns:
        tuple: (obraz z detekcjami, liczba detekcji)
    """
    # Konwersja obrazu PIL do tablicy NumPy i przywrócenie oryginalnego rozmiaru
    image_np = np.array(image)
    image_np = cv2.resize(image_np, original_size)
    
    # Ekstrakcja wyników detekcji
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    masks = predictions['masks'].cpu().numpy() if 'masks' in predictions else None

    # Filtrowanie detekcji na podstawie progu pewności
    valid_indices = scores >= CONFIDENCE_THRESHOLD
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]
    if masks is not None:
        masks = masks[valid_indices]

    # Obliczenie współczynników skalowania dla ramek
    orig_width, orig_height = original_size
    model_width, model_height = MODEL_INPUT_SIZE
    scale_x = orig_width / model_width
    scale_y = orig_height / model_height

    # Skalowanie ramek do oryginalnego rozmiaru obrazu
    scaled_boxes = boxes.copy()
    scaled_boxes[:, 0] *= scale_x  # x1
    scaled_boxes[:, 1] *= scale_y  # y1
    scaled_boxes[:, 2] *= scale_x  # x2
    scaled_boxes[:, 3] *= scale_y  # y2

    # Rysowanie ramek i masek
    detections_count = 0
    for i, (box, score, label) in enumerate(zip(scaled_boxes, scores, labels)):
        # Ograniczenie liczby wizualizowanych detekcji
        if detections_count >= DETECTIONS_PER_IMAGE:
            break
            
        # Rysowanie ramki ograniczającej
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Nakładanie maski segmentacji (jeśli dostępna)
        if masks is not None:
            mask = masks[i, 0] > 0.5
            mask = cv2.resize(mask.astype(np.uint8), original_size) * 255
            image_np[mask > 0] = [0, 0, 255]  # Nakładanie maski z kolorem czerwonym
        
        detections_count += 1

    return image_np, detections_count

def save_result(image_np, image_path):
    """
    Zapisuje obraz z detekcjami do pliku.
    
    Args:
        image_np: Obraz z detekcjami w formacie NumPy.
        image_path: Ścieżka do oryginalnego obrazu.
        
    Returns:
        str: Ścieżka do zapisanego pliku.
    """
    result_image_name = os.path.splitext(os.path.basename(image_path))[0] + "_detected.jpg"
    result_path = os.path.join(RESULTS_PATH, result_image_name)
    print(f"Zapisuję wynik detekcji do: {result_path}")
    cv2.imwrite(result_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return result_path

#######################
# Punkt wejścia programu
#######################
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