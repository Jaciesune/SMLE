"""
Moduł walidacji dla modelu Faster R-CNN

Ten moduł implementuje funkcję walidacji modelu detekcji obiektów Faster R-CNN,
obliczając metryki wydajności na zbiorze walidacyjnym zgodnie ze standardem
COCO (Common Objects in Context). Dostarcza informacje o jakości modelu,
w tym mAP (mean Average Precision) i Recall, oraz generuje wizualizacje
wykrytych obiektów.
"""

#######################
# Importy bibliotek
#######################
import os                  # Do operacji na systemie plików
import sys                 # Do operacji systemowych
import torch               # Framework PyTorch do głębokich sieci neuronowych
import cv2                 # OpenCV do operacji na obrazach i wizualizacji
import numpy as np         # Do operacji numerycznych i manipulacji tablicami
from tqdm import tqdm                    # Pasek postępu
from pycocotools.cocoeval import COCOeval  # Ewaluator metryk COCO
from config import (                     # Import parametrów konfiguracyjnych
    CONFIDENCE_THRESHOLD,                # Minimalny próg pewności dla detekcji
    MIN_ASPECT_RATIO,                    # Minimalny stosunek szerokości do wysokości
    MAX_ASPECT_RATIO,                    # Maksymalny stosunek szerokości do wysokości
    MIN_BOX_AREA_RATIO,                  # Minimalny stosunek powierzchni ramki do obrazu
    MAX_BOX_AREA_RATIO                   # Maksymalny stosunek powierzchni ramki do obrazu
)
import logging                           # Do logowania informacji i błędów

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Wyłączenie debugowania dla Pillow
logging.getLogger('PIL').setLevel(logging.WARNING)

# Wymuszenie kodowania UTF-8 dla stdout
sys.stdout.reconfigure(encoding='utf-8')

#######################
# Funkcja walidacji
#######################
def validate_model(model, dataloader, device, epoch, model_name):
    """
    Przeprowadza walidację modelu na zbiorze walidacyjnym i oblicza metryki COCO.
    
    Funkcja:
    - Wykonuje inferencję modelu na całym zbiorze walidacyjnym
    - Filtruje wyniki na podstawie progów pewności, proporcji i rozmiaru
    - Wizualizuje wyniki detekcji dla celów debugowania
    - Oblicza metryki COCO (mAP@0.5, mAP@0.5:0.95, precision, recall)
    
    Args:
        model (torch.nn.Module): Model Faster R-CNN do walidacji
        dataloader (torch.utils.data.DataLoader): Loader danych walidacyjnych
        device (torch.device): Urządzenie, na którym ma odbywać się walidacja
        epoch (int): Numer aktualnej epoki treningu
        model_name (str): Nazwa modelu (używana do nazewnictwa plików)
        
    Returns:
        tuple: Krotka zawierająca:
            - val_loss (float): Wartość funkcji straty (1 - mAP@0.5:0.95)
            - pred_count (int): Liczba wszystkich wykrytych obiektów
            - gt_count (int): Liczba obiektów ground truth
            - map_50_95 (float): Średnia wartość mAP dla progów IoU od 0.5 do 0.95
            - map_50 (float): Wartość mAP dla progu IoU 0.5
            - precision (float): Precyzja
            - recall (float): Czułość
    """
    #######################
    # Przygotowanie walidacji
    #######################
    model.eval()                 # Przełączenie modelu w tryb ewaluacji
    predictions = []             # Lista do zbierania predykcji
    total_loss = 0.0             # Inicjalizacja sumy strat

    coco_gt = dataloader.dataset.coco  # Obiekt COCO z adnotacjami ground truth
    image_ids = []               # Lista przetworzonych identyfikatorów obrazów
    pred_count = 0               # Licznik predykcji
    gt_count = len(coco_gt.getAnnIds())  # Liczba adnotacji ground truth

    logger.info(f"Rozpoczynanie walidacji dla epoki {epoch}, model: {model_name}")
    logger.info(f"Liczba adnotacji GT: {gt_count}")

    #######################
    # Przeprowadzenie walidacji
    #######################
    for images, targets in tqdm(dataloader, desc="Walidacja"):
        # Przeniesienie danych na urządzenie
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Inferencja modelu w trybie mixed precision
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = model(images)

        #######################
        # Przetwarzanie wyników dla każdego obrazu
        #######################
        for i, output in enumerate(outputs):
            logger.debug(f"Epoka {epoch}, obraz {i} -> liczba predykcji (raw): {len(output['boxes'])}")

            # Pobranie wyników detekcji
            boxes = output["boxes"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            labels = output["labels"].detach().cpu().tolist()
            image_id = targets[i]["image_id"].cpu().item()

            #######################
            # Filtrowanie wyników
            #######################
            for box, score, label in zip(boxes, scores, labels):
                # Filtrowanie po pewności
                if score < CONFIDENCE_THRESHOLD:
                    continue

                # Obliczenie parametrów geometrycznych ramki
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                image_area = images[i].shape[1] * images[i].shape[2]
                area_ratio = area / image_area

                # Filtrowanie po proporcjach i rozmiarze
                if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                    continue
                if not (MIN_BOX_AREA_RATIO <= area_ratio <= MAX_BOX_AREA_RATIO):
                    continue

                # Dodanie zaakceptowanej predykcji
                pred = {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, width, height],
                    "score": float(score)
                }
                predictions.append(pred)

                # Debugowanie predykcji
                if image_id not in coco_gt.imgs:
                    logger.warning(f"Nieznane image_id w predykcji: {image_id}")
                if pred["category_id"] not in coco_gt.cats:
                    logger.warning(f"Nieznane category_id w predykcji: {pred['category_id']}")
                if any(x < 0 for x in pred["bbox"]):
                    logger.warning(f"Ujemne wartości w bbox: {pred['bbox']}")

            pred_count += len(scores)
            image_ids.append(image_id)

    #######################
    # Obsługa braku predykcji
    #######################
    if len(predictions) == 0:
        logger.warning("Brak predykcji - pomijam COCOeval.")
        return 1.0, 0, gt_count, 0.0, 0.0, 0.0, 0.0

    #######################
    # Wizualizacja wyników do debugowania
    #######################
    debug_dir = f"/app/backend/FasterRCNN/logs/val/{model_name}/debug_preds"
    os.makedirs(debug_dir, exist_ok=True)

    for i, image in enumerate(images):
        # Konwersja tensora obrazu na obraz NumPy
        image_np = image.permute(1, 2, 0).cpu().numpy() * 255
        image_np = image_np.astype(np.uint8).copy()
        image_id = targets[i]["image_id"].cpu().item()
        image_name = f"img_{image_id}_ep{epoch}.jpg"

        # Rysowanie ramek dla celów debugowania
        for box, score in zip(outputs[i]["boxes"], outputs[i]["scores"]):
            if score < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image_np, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Zapisanie obrazu z naniesionymi detekcjami
        cv2.imwrite(os.path.join(debug_dir, image_name), image_np)
        logger.debug(f"Zapisano obraz debugowania: {os.path.join(debug_dir, image_name)}")

    #######################
    # Obliczenie metryk COCO
    #######################
    try:
        # Bezpośrednie użycie predictions w COCOeval
        coco_dt = coco_gt.loadRes(predictions)  # Przekazanie listy predykcji bezpośrednio
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Wyciągnięcie metryk
        map_50_95 = coco_eval.stats[0] if coco_eval.stats is not None else 0.0
        map_50 = coco_eval.stats[1] if coco_eval.stats is not None else 0.0
        precision = coco_eval.stats[6] if coco_eval.stats is not None else 0.0
        recall = coco_eval.stats[8] if coco_eval.stats is not None else 0.0

        logger.info(f"Metryki COCO - mAP@0.5: {map_50:.4f} | mAP@0.5:0.95: {map_50_95:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    except Exception as e:
        logger.error(f"Błąd podczas obliczania metryk COCO: {str(e)}")
        map_50_95, map_50, precision, recall = 0.0, 0.0, 0.0, 0.0

    # Obliczenie straty walidacyjnej jako dopełnienie mAP
    val_loss = 1 - map_50_95
    return val_loss, pred_count, gt_count, map_50_95, map_50, precision, recall