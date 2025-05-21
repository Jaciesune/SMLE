"""
Skrypt testowania modelu Faster R-CNN

Ten skrypt umożliwia testowanie wytrenowanego modelu Faster R-CNN na pojedynczych
obrazach. Implementuje pełen potok inferencji, w tym ładowanie modelu, przetwarzanie
obrazu, detekcję obiektów, filtrowanie wyników za pomocą Non-Maximum Suppression (NMS),
wizualizację detekcji i zapis wyników.
"""

#######################
# Importy bibliotek
#######################
import os                # Do operacji na systemie plików
import io                # Do operacji wejścia/wyjścia
import sys               # Do operacji systemowych
import torch             # Framework PyTorch do inferencji modeli głębokich sieci neuronowych
from torchvision import transforms  # Transformacje obrazów
from torchvision.ops import nms     # Implementacja Non-Maximum Suppression
from PIL import Image, ImageDraw, ImageFont  # Biblioteki do operacji na obrazach i rysowania
import argparse          # Do parsowania argumentów wiersza poleceń
import logging           # Do logowania informacji i błędów

# Importy własnych modułów
from model import get_model        # Funkcja inicjalizacji modelu
from config import CONFIDENCE_THRESHOLD  # Parametr konfiguracyjny

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Wymuszenie UTF-8 z fallbackiem na błędy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

#######################
# Funkcje pomocnicze
#######################
def load_model(model_path, device, num_classes=2):
    """
    Ładuje model Faster R-CNN z checkpointa, używając get_model z model.py.
    
    Wczytuje wcześniej wytrenowany model z pliku checkpointa i przygotowuje
    go do inferencji, przenosząc na odpowiednie urządzenie (CPU/GPU).
    
    Args:
        model_path (str): Ścieżka do pliku checkpointa modelu.
        device (torch.device): Urządzenie, na którym ma działać model (CPU/GPU).
        num_classes (int): Liczba klas (wraz z tłem). Domyślnie 2.
        
    Returns:
        torch.nn.Module: Wczytany model w trybie ewaluacji.
        
    Raises:
        SystemExit: Gdy plik modelu nie istnieje lub ma niepoprawny format.
    """
    if not model_path.endswith('_checkpoint.pth'):
        logger.error(f"Ścieżka do modelu {model_path} nie kończy się na _checkpoint.pth.")
        sys.exit(1)

    if not os.path.isfile(model_path):
        logger.error(f"Plik modelu {model_path} nie istnieje.")
        sys.exit(1)

    model = get_model(num_classes=num_classes, device=device)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Wczytano model z {model_path}")
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania modelu: {e}")
        sys.exit(1)

    model.eval()
    return model

def load_image(image_path):
    """
    Ładuje obraz i konwertuje na tensor.
    
    Args:
        image_path (str): Ścieżka do pliku obrazu.
        
    Returns:
        tuple: Para (obraz PIL, tensor obrazu).
        
    Raises:
        SystemExit: Gdy plik obrazu nie istnieje lub ma niepoprawny format.
    """
    if not os.path.isfile(image_path):
        logger.error(f"Obraz {image_path} nie istnieje.")
        sys.exit(1)

    try:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.ToTensor()
        return image, transform(image)
    except Exception as e:
        logger.error(f"Błąd podczas wczytywania obrazu {image_path}: {e}")
        sys.exit(1)

def draw_predictions(image, boxes, labels, scores, threshold):
    """
    Rysuje predykcje na obrazie.
    
    Nakłada na obraz ramki ograniczające (bounding boxes) dla wykrytych obiektów
    wraz z etykietami klas i wartościami pewności detekcji.
    
    Args:
        image (PIL.Image.Image): Oryginalny obraz PIL.
        boxes (torch.Tensor): Tensor ramek ograniczających w formacie [x1, y1, x2, y2].
        labels (torch.Tensor): Tensor etykiet klas dla każdej ramki.
        scores (torch.Tensor): Tensor wartości pewności (confidence) dla każdej ramki.
        threshold (float): Minimalny próg pewności do wyświetlenia detekcji.
        
    Returns:
        PIL.Image.Image: Obraz z narysowanymi detekcjami.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        logger.warning("Nie znaleziono fontu arial.ttf, używam domyślnego.")
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            box = box.tolist()
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], max(0, box[1] - 15)), f"Class {label}: {score:.2f}", fill="red", font=font)
    return image

#######################
# Główna funkcja
#######################
def main():
    """
    Główna funkcja skryptu testowania.
    
    Parsuje argumenty, ładuje model i obraz, przeprowadza inferencję,
    filtruje wyniki i zapisuje obraz z detekcjami.
    """
    #######################
    # Parsowanie argumentów
    #######################
    parser = argparse.ArgumentParser(description="Testowanie modelu Faster R-CNN")
    parser.add_argument("--image_path", required=True, help="Ścieżka do obrazu do testowania")
    parser.add_argument("--model_path", required=True, help="Ścieżka do wytrenowanego modelu (np. model_checkpoint.pth)")
    parser.add_argument("--output_dir", default="test", help="Folder zapisu wyników")
    parser.add_argument("--num_classes", type=int, default=2, help="Liczba klas (łącznie z tłem)")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Minimalny próg ufności dla predykcji")
    args = parser.parse_args()

    logger.info(f"Argumenty: image_path={args.image_path}, model_path={args.model_path}, output_dir={args.output_dir}, num_classes={args.num_classes}, threshold={args.threshold}")

    #######################
    # Inicjalizacja urządzenia
    #######################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Używane urządzenie: {device}")

    #######################
    # Ładowanie modelu i obrazu
    #######################
    logger.info(f"Ładowanie modelu z: {args.model_path}")
    model = load_model(args.model_path, device, num_classes=args.num_classes)

    logger.info(f"Ładowanie obrazu: {args.image_path}")
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    image_pil, image_tensor = load_image(args.image_path)

    #######################
    # Detekcja obiektów
    #######################
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])

    boxes = prediction[0]['boxes'].cpu()
    labels = prediction[0]['labels'].cpu()
    scores = prediction[0]['scores'].cpu()

    #######################
    # Filtrowanie wyników
    #######################
    # Non-Maximum Suppression do usuwania nakładających się wykryć
    keep_indices = nms(boxes, scores, iou_threshold=0.25)
    boxes = boxes[keep_indices].cpu()
    scores = scores[keep_indices].cpu()
    labels = labels[keep_indices].cpu()

    # Filtrowanie ramek po wielkości (odrzucenie zbyt małych i zbyt dużych)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    mean_area = areas.mean() if len(areas) > 0 else 0
    min_area = 0.3 * mean_area
    max_area = 3.0 * mean_area

    valid_indices = (areas >= min_area) & (areas <= max_area)
    boxes = boxes[valid_indices].cpu()
    scores = scores[valid_indices].cpu()
    labels = labels[valid_indices].cpu()

    # Debugowanie wartości scores i areas
    logger.debug(f"Liczba bboxów po NMS: {len(keep_indices)}")
    logger.debug(f"Liczba bboxów po filtrowaniu powierzchni: {len(boxes)}")
    logger.debug(f"Wyniki ufności (scores): {scores.tolist()}")
    logger.debug(f"Powierzchnie bboxów (areas): {areas.tolist()}")

    #######################
    # Wizualizacja i zapis wyników
    #######################
    annotated_image = draw_predictions(image_pil.copy(), boxes, labels, scores, args.threshold)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{image_name}_detected.jpg")
    annotated_image.save(save_path)
    logger.info(f"Zapisano obraz z wykryciami do {save_path}")

    num_detections = (scores >= args.threshold).sum().item()
    logger.info(f"Liczba detekcji: {num_detections}")

    # Wypisanie liczby detekcji w formacie zgodnym z Mask R-CNN
    print(f"Detections: {num_detections}")

#######################
# Punkt wejścia programu
#######################
if __name__ == "__main__":
    main()