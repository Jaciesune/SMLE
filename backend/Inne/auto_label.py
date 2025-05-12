"""
Moduł automatycznego oznaczania obrazów (Auto-Labeling)

Ten skrypt umożliwia automatyczne oznaczanie obrazów przy użyciu modelu Mask R-CNN.
Wykrywa obiekty na obrazach i generuje adnotacje w formacie LabelMe,
które mogą być później wykorzystane do treningu lub wizualizacji.
"""

#######################
# Importy bibliotek
#######################
import torch                               # Biblioteka PyTorch do obsługi modeli głębokich sieci neuronowych
import torchvision.transforms as T         # Transformacje obrazów
import torchvision.models.detection        # Modele detekcji obiektów
import os                                  # Do operacji na systemie plików
import cv2                                 # OpenCV do operacji na obrazach
from PIL import Image                      # Biblioteka do operacji na obrazach
import numpy as np                         # Do operacji numerycznych
import time                                # Do pomiaru czasu wykonania
import sys                                 # Do operacji systemowych
import argparse                            # Do parsowania argumentów wiersza poleceń
import glob                                # Do wyszukiwania plików według wzorca
import json                                # Do operacji na plikach JSON
import base64                              # Do kodowania masek w formacie base64
import logging                             # Do logowania informacji

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#######################
# Parametry konfiguracyjne
#######################
CONFIDENCE_THRESHOLD = 0.7     # Minimalny próg pewności dla detekcji
NMS_THRESHOLD = 1000           # Próg dla Non-Maximum Suppression
DETECTIONS_PER_IMAGE = 500     # Maksymalna liczba detekcji na obraz
NUM_CLASSES = 2                # Liczba klas (tło + 1 klasa obiektów)
MODEL_INPUT_SIZE = (1024, 1024)  # Rozmiar wejściowy obrazu dla modelu

# Urządzenie obliczeniowe (GPU lub CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################
# Funkcje związane z modelem
#######################
def load_model(model_path):
    """
    Ładuje model Mask R-CNN z pliku checkpointu.
    
    Args:
        model_path (str): Ścieżka do pliku checkpointu modelu.
        
    Returns:
        torch.nn.Module: Wczytany model w trybie ewaluacji.
        
    Raises:
        SystemExit: Gdy wystąpi błąd podczas wczytywania modelu.
    """
    logger.debug(f"Sprawdzam model pod ścieżką: {model_path}")
    # Walidacja ścieżki do modelu
    if not model_path.endswith('_checkpoint.pth'):
        logger.error(f"Ścieżka do modelu {model_path} nie kończy się na _checkpoint.pth.")
        sys.exit(1)

    if not os.path.exists(model_path):
        logger.error(f"Plik modelu {model_path} nie istnieje.")
        sys.exit(1)

    logger.debug(f"Tworzę instancję modelu Mask R-CNN...")
    try:
        # Inicjalizacja modelu z predefiniowanymi wagami
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        
        # Dostosowanie głowicy klasyfikatora dla odpowiedniej liczby klas
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
        
        # Dostosowanie głowicy predyktora masek
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_channels=256, dim_reduced=256, num_classes=NUM_CLASSES
        )
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia modelu: {e}")
        sys.exit(1)

    # Wczytanie wag z checkpointu
    start_time = time.time()
    logger.debug(f"Wczytuję checkpoint z {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania checkpointu: {e}")
        sys.exit(1)

    # Konfiguracja parametrów modelu
    logger.debug(f"Konfiguruję parametry modelu...")
    try:
        # Ustawienie parametrów RPN (Region Proposal Network)
        if isinstance(model.rpn.pre_nms_top_n, dict):
            model.rpn.pre_nms_top_n["training"] = NMS_THRESHOLD
            model.rpn.pre_nms_top_n["testing"] = NMS_THRESHOLD
            model.rpn.post_nms_top_n["training"] = NMS_THRESHOLD
            model.rpn.post_nms_top_n["testing"] = NMS_THRESHOLD
            
        # Ustawienie progu pewności i liczby detekcji
        model.roi_heads.score_thresh = CONFIDENCE_THRESHOLD
        model.roi_heads.detections_per_img = DETECTIONS_PER_IMAGE
        
        # Przeniesienie modelu na odpowiednie urządzenie i przełączenie w tryb ewaluacji
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        logger.error(f"Błąd podczas konfiguracji modelu: {e}")
        sys.exit(1)

    end_time = time.time()
    logger.debug(f"Czas wczytywania modelu: {end_time - start_time:.2f} sekund")
    return model

#######################
# Funkcje przetwarzania obrazów
#######################
def preprocess_image(image_path):
    """
    Wczytuje obraz i przygotowuje go do przetworzenia przez model.
    
    Args:
        image_path (str): Ścieżka do pliku obrazu.
        
    Returns:
        tuple: (tensor obrazu, oryginalny obraz, szerokość, wysokość)
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Błąd podczas otwierania obrazu {image_path}: {e}")
        raise
    original_size = image.size

    # Transformacja obrazu do formatu wymaganego przez model
    transform = T.Compose([
        T.Resize(MODEL_INPUT_SIZE),  # Zmiana rozmiaru
        T.ToTensor()                 # Konwersja na tensor
    ])
    tensor = transform(image).to(DEVICE)
    return [tensor], image, original_size[0], original_size[1]

def predict(model, image_tensor_list):
    """
    Wykonuje detekcję obiektów na obrazie.
    
    Args:
        model: Model Mask R-CNN.
        image_tensor_list: Lista tensorów obrazu.
        
    Returns:
        dict: Słownik z wynikami detekcji (boxes, masks, labels, scores).
    """
    try:
        with torch.no_grad():
            predictions = model(image_tensor_list)[0]
    except Exception as e:
        logger.error(f"Błąd podczas detekcji: {e}")
        raise
    return predictions

#######################
# Funkcje post-processingu
#######################
def rescale_boxes(boxes, orig_width, orig_height):
    """
    Skaluje współrzędne ramek z powrotem do rozmiaru oryginalnego obrazu.
    
    Args:
        boxes: Tensor z ramkami w formacie [x_min, y_min, x_max, y_max].
        orig_width: Szerokość oryginalnego obrazu.
        orig_height: Wysokość oryginalnego obrazu.
        
    Returns:
        numpy.ndarray: Przeskalowane ramki.
    """
    boxes = boxes.cpu().numpy()
    
    # Obliczenie współczynników skalowania
    scale_x = orig_width / MODEL_INPUT_SIZE[0]
    scale_y = orig_height / MODEL_INPUT_SIZE[1]
    
    # Skalowanie współrzędnych
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # Korekta ramek na krawędziach obrazu
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
    """
    Skaluje maski segmentacji do rozmiaru oryginalnego obrazu.
    
    Args:
        masks: Lista masek segmentacji.
        orig_width: Szerokość oryginalnego obrazu.
        orig_height: Wysokość oryginalnego obrazu.
        
    Returns:
        list: Lista przeskalowanych masek.
    """
    masks_resized = []
    for mask in masks:
        # Konwersja maski na numpy i binaryzacja
        mask_np = (mask.cpu().numpy()[0] > 0.5).astype(np.uint8)
        
        # Zmiana rozmiaru maski
        mask_resized = cv2.resize(mask_np, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        
        # Ponowna binaryzacja
        mask_resized = (mask_resized > 0.5).astype(np.uint8)
        masks_resized.append(mask_resized)
    return masks_resized

def crop_mask_to_bbox(mask, box, orig_width, orig_height, image_name, idx):
    """
    Przycina maskę do ramki ograniczającej.
    
    Args:
        mask: Maska segmentacji.
        box: Współrzędne ramki [x_min, y_min, x_max, y_max].
        orig_width: Szerokość oryginalnego obrazu.
        orig_height: Wysokość oryginalnego obrazu.
        image_name: Nazwa obrazu (do logowania).
        idx: Indeks obiektu (do logowania).
        
    Returns:
        numpy.ndarray: Przycięta maska lub None w przypadku błędu.
    """
    mask_np = mask
    x_min, y_min, x_max, y_max = map(int, box)
    
    # Ograniczenie współrzędnych do granic obrazu
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(orig_width - 1, x_max)
    y_max = min(orig_height - 1, y_max)

    # Sprawdzenie poprawności ramki
    if x_max <= x_min or y_max <= y_min:
        logger.warning(f"Nieprawidłowy bbox dla {image_name}, obiekt {idx}: ({x_min}, {y_min}, {x_max}, {y_max})")
        return None
        
    # Przycięcie maski do ramki
    cropped_mask = mask_np[y_min:y_max, x_min:x_max]
    return cropped_mask

def encode_mask_to_base64(cropped_mask):
    """
    Koduje maskę do formatu base64 dla pliku JSON LabelMe.
    
    Args:
        cropped_mask: Przycięta maska binarowa.
        
    Returns:
        str: Zakodowana maska w formacie base64 lub None w przypadku błędu.
    """
    success, encoded_image = cv2.imencode('.png', cropped_mask * 255)
    if success:
        return base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    logger.error("Błąd podczas kodowania maski do base64.")
    return None

#######################
# Funkcje generowania wyjścia
#######################
def save_labelme_json(image_path, image_name, predictions, orig_width, orig_height, output_dir, custom_label):
    """
    Zapisuje wyniki detekcji w formacie JSON kompatybilnym z LabelMe.
    
    Args:
        image_path: Ścieżka do oryginalnego obrazu.
        image_name: Nazwa obrazu (bez rozszerzenia).
        predictions: Wyniki detekcji z modelu.
        orig_width: Szerokość oryginalnego obrazu.
        orig_height: Wysokość oryginalnego obrazu.
        output_dir: Katalog do zapisania wyników.
        custom_label: Niestandardowa etykieta dla wszystkich detekcji.
    """
    # Kopiowanie oryginalnego obrazu do katalogu wyjściowego
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

    # Przygotowanie struktury JSON dla LabelMe
    json_data = {
        "version": "5.8.1",
        "flags": {},
        "shapes": [],
        "imagePath": f"{image_name}.jpg",
        "imageData": None,
        "imageHeight": orig_height,
        "imageWidth": orig_width
    }

    # Przeskalowanie wyników do oryginalnego rozmiaru obrazu
    boxes = rescale_boxes(predictions['boxes'], orig_width, orig_height)
    masks = rescale_masks(predictions.get('masks', []), orig_width, orig_height)
    labels = predictions['labels'].cpu().numpy()

    # Dodanie każdej detekcji do pliku JSON
    for idx, (box, score, mask, label_idx) in enumerate(zip(boxes, predictions['scores'], masks, labels)):
        if score >= CONFIDENCE_THRESHOLD:
            # Ekstrakcja współrzędnych ramki
            x_min, y_min, x_max, y_max = box
            
            # Przycięcie maski do ramki
            cropped_mask = crop_mask_to_bbox(mask, box, orig_width, orig_height, image_name, idx)
            if cropped_mask is None:
                logger.warning(f"Pominięto maskę dla {image_name}, obiekt {idx}.")
                continue

            # Kodowanie maski do base64
            mask_base64 = encode_mask_to_base64(cropped_mask)
            if mask_base64 is None:
                logger.warning(f"Nie udało się zakodować maski dla {image_name}, obiekt {idx}.")
                continue

            # Pomijamy detekcje tła
            if label_idx == 0:
                logger.debug(f"Pominięto obiekt {idx} w {image_name} - etykieta to 'background'.")
                continue

            # Tworzenie informacji o kształcie
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

    # Zapisanie pliku JSON
    json_path = os.path.join(output_dir, f"{image_name}.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        logger.debug(f"Zapisano adnotacje JSON do: {json_path}")
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania JSON do {json_path}: {e}")

#######################
# Funkcja główna
#######################
def main():
    """
    Główna funkcja skryptu auto_label.py.
    
    Parsuje argumenty wiersza poleceń, ładuje model i przetwarza wszystkie obrazy
    w katalogu wejściowym, generując pliki LabelMe w katalogu wyjściowym.
    """
    logger.info("Rozpoczynam automatyczne labelowanie...")
    
    # Parsowanie argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description="Automatyczne labelowanie katalogu zdjęć przy użyciu Mask R-CNN")
    parser.add_argument("--input_dir", type=str, required=True, help="Ścieżka do katalogu z obrazami wejściowymi")
    parser.add_argument("--output_dir", type=str, required=True, help="Ścieżka do katalogu na obrazy z detekcjami")
    parser.add_argument("--debug_dir", type=str, default="", help="Ścieżka do katalogu na obrazy z adnotacjami (opcjonalne)")
    parser.add_argument("--model_path", type=str, required=True, help="Ścieżka do pliku modelu z końcówką _checkpoint.pth")
    parser.add_argument("--custom_label", type=str, required=True, help="Etykieta do użycia dla wykrytych obiektów")
    args = parser.parse_args()

    logger.debug(f"Argumenty: input_dir={args.input_dir}, output_dir={args.output_dir}, debug_dir={args.debug_dir}, model_path={args.model_path}, custom_label={args.custom_label}")

    # Tworzenie katalogu wyjściowego
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia katalogu wyjściowego: {e}")
        sys.exit(1)

    # Wczytanie modelu
    try:
        model = load_model(args.model_path)
    except Exception as e:
        logger.error(f"Błąd wczytywania modelu: {e}")
        sys.exit(1)

    # Wyszukanie obrazów w katalogu wejściowym
    image_paths = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    if not image_paths:
        logger.error(f"Brak obrazów .jpg w katalogu {args.input_dir}")
        sys.exit(1)

    # Przetwarzanie obrazów
    total_detections = 0
    processed_images = 0
    for image_path in image_paths:
        logger.debug(f"Przetwarzam obraz: {image_path}")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        try:
            # Wczytanie i przygotowanie obrazu
            image_tensor_list, image, orig_width, orig_height = preprocess_image(image_path)
            
            # Wykonanie detekcji
            predictions = predict(model, image_tensor_list)
            
            # Zapisanie wyników w formacie LabelMe
            save_labelme_json(image_path, image_name, predictions, orig_width, orig_height, args.output_dir, args.custom_label)
            
            # Zliczenie detekcji i przetworzonych obrazów
            total_detections += len([s for s in predictions['scores'] if s >= CONFIDENCE_THRESHOLD])
            processed_images += 1
            logger.debug(f"Detections dla {image_path}: {len([s for s in predictions['scores'] if s >= CONFIDENCE_THRESHOLD])}")
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania obrazu {image_path}: {e}")
            continue

    # Weryfikacja wyników
    output_files = os.listdir(args.output_dir)
    logger.debug(f"Zawartość katalogu wyjściowego {args.output_dir}: {output_files}")

    if not output_files:
        logger.warning(f"Katalog wyjściowy {args.output_dir} jest pusty!")

    # Podsumowanie
    logger.info(f"Przetworzono obrazów: {processed_images}, Całkowita liczba detekcji: {total_detections}")
    logger.info("Automatyczne labelowanie zakończone!")