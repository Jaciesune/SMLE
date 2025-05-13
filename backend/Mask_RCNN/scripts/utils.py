"""
Moduł narzędziowy dla treningu i walidacji modelu Mask R-CNN

Ten moduł implementuje funkcje pomocnicze do treningu i walidacji modeli segmentacji instancji,
w tym przetwarzanie z mixed precision, zaawansowaną walidację z metrykami COCO,
wizualizację wyników oraz zarządzanie zasobami systemowymi.
"""

#######################
# Importy bibliotek
#######################
import torch                              # Framework PyTorch
from torch.amp import autocast, GradScaler  # Do treningu z mixed precision
import cv2                                # OpenCV do operacji na obrazach
import numpy as np                        # Do operacji numerycznych
import os                                 # Operacje na systemie plików
from pycocotools.coco import COCO         # Narzędzia do obsługi formatu COCO
from pycocotools import mask as coco_mask  # Operacje na maskach COCO
from pycocotools.cocoeval import COCOeval  # Do ewaluacji modeli z metrykami COCO
import gc                                 # Garbage collector do zarządzania pamięcią
import psutil                             # Do monitorowania zasobów systemowych
import shutil                             # Do operacji na plikach
import logging                            # Do logowania informacji

#######################
# Konfiguracja logowania
#######################
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trenuje model przez jedną epokę z użyciem mixed precision (AMP), bez akumulacji gradientów i grad_clip.
    
    Funkcja implementuje pełną pętlę treningową z wykorzystaniem:
    - Mixed precision training dla lepszej wydajności
    - Automatycznego skalowania gradientów za pomocą GradScaler
    - Zarządzania pamięcią poprzez ręczne wywoływanie garbage collectora
    - Zamykania procesów DataLoader po epoce
    
    Args:
        model: Model PyTorch do treningu
        dataloader: DataLoader z danymi treningowymi
        optimizer: Optymalizator (np. SGD)
        device: Urządzenie (CPU/GPU)
        epoch: Numer bieżącej epoki
        
    Returns:
        float: Średnia strata dla epoki
    """
    scaler = GradScaler()
    model.train()
    całkowita_strata = 0

    print(f"\nEpoka {epoch}... (Batchy: {len(dataloader)})")

    for batch_idx, (obrazy, cele) in enumerate(dataloader):
        # Monitorowanie pamięci RAM i /dev/shm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Przeniesienie danych na urządzenie z odpowiednimi typami danych
        obrazy = [img.to(device, dtype=torch.bfloat16) for img in obrazy]
        nowe_cele = [
            {
                k: v.to(device, dtype=torch.float32) if k == 'boxes' else 
                  v.to(device, dtype=torch.bfloat16) if k == 'masks' else 
                  v.to(device) 
                for k, v in t.items()
            } 
            for t in cele
        ]

        optimizer.zero_grad(set_to_none=True)  # Reset gradientów przed każdym batch-em

        # Forward pass z mixed precision
        with autocast(device_type='cuda', dtype=torch.bfloat16):  # Poprawione użycie autocast
            słownik_strat = model(obrazy, nowe_cele)
            strata = sum(strata for strata in słownik_strat.values())

        # Skalowanie straty i wsteczne propagowanie
        scaler.scale(strata).backward()

        # Aktualizacja wag bez grad_clip
        scaler.step(optimizer)
        scaler.update()

        całkowita_strata += strata.item()
        print(f"Batch {batch_idx+1}/{len(dataloader)} - Strata: {strata.item():.4f}")

        # Czyszczenie zmiennych dla oszczędności pamięci
        del obrazy, nowe_cele, słownik_strat, strata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Zamknij procesy DataLoader po epoce
    if hasattr(dataloader.dataset, '_shut_down_workers'):
        dataloader.dataset._shut_down_workers()
        logger.info(f"Zamknięto procesy DataLoader po epoce {epoch}")

    return całkowita_strata / len(dataloader) if len(dataloader) > 0 else 0

def decode_rle_segmentation(segmentation):
    """
    Dekoduje segmentację z formatu RLE (Run-Length Encoding) do maski binarnej.
    
    Args:
        segmentation (dict): Słownik z danymi RLE w formacie COCO
        
    Returns:
        numpy.ndarray: Zdekodowana maska binarna
    """
    rle = {
        "counts": segmentation["counts"].encode('utf-8'),
        "size": segmentation["size"]
    }
    return coco_mask.decode(rle)

def create_full_mask(masks, bboxes, image_shape):
    """
    Tworzy pełną maskę z listy masek i ramek ograniczających.
    
    Funkcja łączy poszczególne maski w jedną maskę o wymiarach całego obrazu,
    uwzględniając pozycję każdej maski określoną przez ramkę ograniczającą.
    
    Args:
        masks (list): Lista masek binarnych dla każdego obiektu
        bboxes (list): Lista ramek ograniczających [x, y, w, h]
        image_shape (tuple): Kształt obrazu wyjściowego (h, w)
        
    Returns:
        numpy.ndarray: Połączona maska binarna
    """
    full_mask = np.zeros(image_shape, dtype=np.uint8)
    
    for mask, bbox in zip(masks, bboxes):
        if mask is None:
            continue
        
        if mask.dtype == bool:
            mask = mask.astype(np.uint8)
        
        x, y, w, h = map(int, bbox)
        if w <= 0 or h <= 0:
            continue
        
        mask_h, mask_w = mask.shape
        if mask_h != h or mask_w != w:
            try:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            except cv2.error as e:
                print(f"Błąd podczas resize maski: {e}. Pomijam tę maskę.")
                continue
        
        x_end = min(x + w, image_shape[1])
        y_end = min(y + h, image_shape[0])
        if x_end <= x or y_end <= y:
            continue
        
        full_mask[y:y_end, x:x_end] |= mask[:y_end - y, :x_end - x]
    
    return full_mask

def validate_model(model, dataloader, device, epoch, nazwa_modelu, ścieżka_coco_gt):
    """
    Waliduje model, oblicza metryki (mAP) na podstawie IoU pełnych masek i zapisuje wizualizacje.
    
    Funkcja wykonuje następujące kroki:
    1. Przeprowadza inferencję modelu na zbiorze walidacyjnym
    2. Konwertuje wyniki do formatu COCO do obliczeń metryk
    3. Oblicza IoU dla pełnych masek i ramek ograniczających (bbox)
    4. Generuje i zapisuje wizualizacje predykcji do analizy jakościowej
    5. Oblicza metryki COCO (mAP) dla ramek i segmentacji
    
    Args:
        model: Model PyTorch do walidacji
        dataloader: DataLoader z danymi walidacyjnymi
        device: Urządzenie (CPU/GPU)
        epoch: Numer bieżącej epoki
        nazwa_modelu: Nazwa modelu dla zapisu wyników
        ścieżka_coco_gt: Ścieżka do pliku COCO z ground truth
        
    Returns:
        tuple: (średnia_strata_walidacyjna, liczba_predykcji, liczba_gt, mAP_bbox, mAP_seg)
    """
    model.eval()
    całkowita_strata_walidacyjna = 0
    całkowita_liczba_predykcji = 0
    całkowita_liczba_gt = 0
    pełne_iou_sum = 0
    pełne_iou_count = 0
    ścieżka_zapisu = f"/app/backend/Mask_RCNN/logs/val/{nazwa_modelu}/epoch_{epoch:02d}"
    ścieżka_zapisu_gt = f"/app/backend/Mask_RCNN/logs/val/{nazwa_modelu}/gt_image"
    os.makedirs(ścieżka_zapisu, exist_ok=True)

    # Wczytanie adnotacji ground truth
    coco_gt = COCO(ścieżka_coco_gt)
    coco_dt = []
    nowy_rozmiar = (1024, 1024)

    # Skalowanie adnotacji ground truth do rozmiaru wejściowego modelu
    orig_sizes = {img["id"]: (img["height"], img["width"]) for img in coco_gt.loadImgs(coco_gt.getImgIds())}
    ann_ids = coco_gt.getAnnIds(imgIds=coco_gt.getImgIds())
    anns = coco_gt.loadAnns(ann_ids)
    anns_by_image = {}
    for ann in anns:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Przeskalowanie adnotacji ground truth
    for img_id, annotations in anns_by_image.items():
        orig_h, orig_w = orig_sizes[img_id]
        scale_x = nowy_rozmiar[1] / orig_w
        scale_y = nowy_rozmiar[0] / orig_h
        for ann in annotations:
            bbox = ann["bbox"]
            scaled_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]
            ann["bbox"] = scaled_bbox
            mask = decode_rle_segmentation(ann["segmentation"])
            w, h = int(scaled_bbox[2]), int(scaled_bbox[3])
            if w > 0 and h > 0:
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                rle = coco_mask.encode(np.asfortranarray(mask_resized.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                ann["segmentation"] = rle
                ann["area"] = int(np.sum(mask_resized))
            else:
                print(f"Pomijam maskę dla img_id {img_id}, bbox o zerowych wymiarach: {scaled_bbox}")

    coco_gt.dataset["annotations"] = anns
    coco_gt.createIndex()

    # Generowanie obrazów ground truth (tylko raz, jeśli folder nie istnieje)
    if not os.path.exists(ścieżka_zapisu_gt):
        os.makedirs(ścieżka_zapisu_gt, exist_ok=True)
        print("Generowanie gt_image jednorazowo...")
        for idx, (obrazy, cele) in enumerate(dataloader):
            for i, (obraz, cel) in enumerate(zip(obrazy, cele)):
                obraz_np = (obraz.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                obraz_np = cv2.cvtColor(obraz_np, cv2.COLOR_RGB2BGR)
                if "image_id" not in cel:
                    print(f"Batch {idx}, Obraz {i}: Brak image_id, pomijam.")
                    continue
                id_obrazu = int(cel["image_id"].cpu().numpy()[0])
                gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=id_obrazu))
                gt_masks = [decode_rle_segmentation(ann["segmentation"]) for ann in gt_anns]
                gt_bboxes = [ann["bbox"] for ann in gt_anns]
                full_gt_mask = create_full_mask(gt_masks, gt_bboxes, nowy_rozmiar)
                gt_image = obraz_np.copy()
                for ann in gt_anns:
                    x, y, w, h = map(int, ann["bbox"])
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    cv2.rectangle(gt_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    mask = decode_rle_segmentation(ann["segmentation"])
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    full_mask = np.zeros_like(obraz_np[:, :, 0], dtype=np.uint8)
                    full_mask[y:y + h, x:x + w] = mask
                    gt_image[full_mask > 0] = [255, 0, 0]
                cv2.imwrite(f"{ścieżka_zapisu_gt}/gt_image_{idx}_{i}.png", gt_image)

    print(f"Walidacja epoki {epoch}...")

    # Pętla walidacyjna
    for idx, (obrazy, cele) in enumerate(dataloader):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Przeniesienie danych na urządzenie
        obrazy = [img.to(device, dtype=torch.bfloat16) for img in obrazy]
        nowe_cele = [
            {
                k: v.to(device, dtype=torch.float32) if k == 'boxes' else 
                  v.to(device, dtype=torch.bfloat16) if k == 'masks' else 
                  v.to(device) 
                for k, v in t.items()
            } 
            for t in cele
        ]

        # Obliczenie straty walidacyjnej i wykonanie inferencji
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            model.train()
            with torch.no_grad():
                słownik_strat = model(obrazy, nowe_cele)
                strata_walidacyjna = sum(strata for strata in słownik_strat.values())
            całkowita_strata_walidacyjna += strata_walidacyjna.item()

            model.eval()
            wyniki = model(obrazy)

        # Przetwarzanie wyników dla każdego obrazu
        for i, (obraz, wynik, cel) in enumerate(zip(obrazy, wyniki, cele)):
            obraz_np = (obraz.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            obraz_np = cv2.cvtColor(obraz_np, cv2.COLOR_RGB2BGR)
            liczba_gt = cel["boxes"].shape[0] if "boxes" in cel and cel["boxes"].numel() > 0 else 0
            całkowita_liczba_gt += liczba_gt

            if liczba_gt == 0 or "image_id" not in cel:
                print(f"Batch {idx}, Obraz {i}: Brak adnotacji GT lub image_id, pomijam.")
                continue
            id_obrazu = int(cel["image_id"].cpu().numpy()[0])

            # Ekstrakcja predykcji modelu
            pred_masks = [maska.detach().cpu().numpy()[0] > 0.5 for maska in wynik["masks"]]
            pred_bboxes = [box.detach().cpu().numpy() for box in wynik["boxes"]]
            pred_scores = wynik["scores"].detach().cpu().numpy()
            pred_labels = wynik["labels"].detach().cpu().numpy()

            # Tworzenie pełnej maski predykcji
            full_pred_mask = np.zeros(nowy_rozmiar, dtype=np.uint8)
            for mask in pred_masks:
                full_pred_mask |= mask.astype(np.uint8)

            # Filtrowanie predykcji według progu pewności
            filtered_preds = [
                (box, mask, score, label)
                for box, mask, score, label in zip(pred_bboxes, pred_masks, pred_scores, pred_labels)
                if score >= 0.1
            ]
            liczba_predykcji = len(filtered_preds)
            całkowita_liczba_predykcji += liczba_predykcji

            # Konwersja predykcji do formatu COCO
            for box, mask, score, label in filtered_preds:
                x1, y1, x2, y2 = map(int, box)
                szerokość = max(0, x2 - x1)
                wysokość = max(0, y2 - y1)
                if szerokość > 0 and wysokość > 0:
                    mask_cropped = mask[y1:y1 + wysokość, x1:x1 + szerokość]
                    rle = coco_mask.encode(np.asfortranarray(mask_cropped.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    coco_dt.append({
                        "image_id": id_obrazu,
                        "category_id": int(label),
                        "bbox": [x1, y1, szerokość, wysokość],
                        "score": float(score),
                        "segmentation": rle
                    })

            # Wczytanie adnotacji ground truth dla bieżącego obrazu
            gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=id_obrazu))
            gt_masks = [decode_rle_segmentation(ann["segmentation"]) for ann in gt_anns]
            gt_bboxes = [ann["bbox"] for ann in gt_anns]
            full_gt_mask = create_full_mask(gt_masks, gt_bboxes, nowy_rozmiar)

            # Generowanie wizualizacji predykcji
            pred_image = obraz_np.copy()
            for box, mask, score, label in filtered_preds:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(pred_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                pred_image[mask > 0] = [0, 0, 255]
            cv2.imwrite(f"{ścieżka_zapisu}/pred_image_{idx}_{i}.png", pred_image)

            # Obliczenie IoU dla pełnych masek
            if full_pred_mask.sum() > 0 and full_gt_mask.sum() > 0:
                intersection = np.logical_and(full_pred_mask, full_gt_mask).sum()
                union = np.logical_or(full_pred_mask, full_gt_mask).sum()
                full_iou = intersection / union if union > 0 else 0
                pełne_iou_sum += full_iou
                pełne_iou_count += 1
                print(f"Batch {idx}, Obraz {i}, Predykcje: {liczba_predykcji}, Liczba GT: {liczba_gt}, Ratio: {liczba_predykcji/liczba_gt:.2f}, IoU pełnych masek: {full_iou:.3f}")
            else:
                print(f"Batch {idx}, Obraz {i}, Predykcje: {liczba_predykcji}, Liczba GT: {liczba_gt}, Ratio: {liczba_predykcji/liczba_gt:.2f}, IoU pełnych masek: 0.000 (puste maski)")

            # Zarządzanie pamięcią
            del obraz_np, pred_masks, pred_bboxes, pred_scores, pred_labels, full_pred_mask, gt_masks, gt_bboxes, full_gt_mask, pred_image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Obliczenie metryk COCO
    if len(coco_dt) == 0:
        print("Brak predykcji powyżej progu.")
        mAP_bbox = 0.0
        mAP_seg = 0.0
    else:
        coco_dt = coco_gt.loadRes(coco_dt)
        coco_eval_bbox = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        coco_eval_bbox.summarize()
        mAP_bbox = coco_eval_bbox.stats[0]
        mAP_seg = pełne_iou_sum / pełne_iou_count if pełne_iou_count > 0 else 0

    # Podsumowanie wyników walidacji
    średnia_strata_walidacyjna = całkowita_strata_walidacyjna / len(dataloader) if len(dataloader) > 0 else 0
    print(f"Średnia strata walidacyjna: {średnia_strata_walidacyjna:.4f}")
    print(f"Całkowita liczba predykcji: {całkowita_liczba_predykcji}, GT: {całkowita_liczba_gt}")
    print(f"mAP_bbox: {mAP_bbox:.4f}, mAP_seg (IoU pełnych masek): {mAP_seg:.4f}")
    return średnia_strata_walidacyjna, całkowita_liczba_predykcji, całkowita_liczba_gt, mAP_bbox, mAP_seg

def custom_collate_fn(batch):
    """
    Funkcja łącząca próbki w batche dla DataLoadera.
    
    Args:
        batch: Lista próbek zwróconych przez __getitem__
        
    Returns:
        tuple: Próbki złączone w batch
    """
    return tuple(zip(*batch))

def estimate_batch_size(image_size, max_objects, max_batch_size=16, min_batch_size=1, use_amp=True, is_training=True):
    """
    Estymuje optymalny batch size na podstawie dostępnych zasobów systemu.
    
    Uwzględnia dostępną pamięć RAM i VRAM, liczbę obiektów w obrazach oraz
    dodatkowe czynniki, takie jak użycie mixed precision i tryb (trening/inferencja).
    
    Args:
        image_size (tuple): Rozmiar obrazu (wysokość, szerokość)
        max_objects (int): Maksymalna liczba obiektów na obraz
        max_batch_size (int): Maksymalny dozwolony batch size
        min_batch_size (int): Minimalny batch size
        use_amp (bool): Czy używać mixed precision
        is_training (bool): Czy estymacja dotyczy treningu
        
    Returns:
        int: Estymowany batch size
    """
    if image_size is None:
        raise ValueError("image_size musi być podane jako argument w estimate_batch_size")

    # Obliczanie zużycia pamięci dla jednego obrazu
    image_memory = image_size[0] * image_size[1] * 3 * 4  # Obraz RGB (float32)
    masks_memory_per_image = max_objects * image_size[0] * image_size[1] * 1  # Maski (uint8)
    activations_memory_per_image = 0.5 * 1024 ** 3  # Aktywacje
    model_memory = 0.6 * 1024 ** 3  # Model
    cuda_overhead = 1.0 * 1024 ** 3  # Overhead CUDA
    
    # Współczynnik oszczędności pamięci przy użyciu mixed precision
    amp_factor = 0.6 if use_amp else 1.0
    memory_per_image_gpu = (image_memory + masks_memory_per_image + activations_memory_per_image) * amp_factor

    # Dodatkowy narzut dla treningu
    if is_training:
        model_memory *= 3  # Gradienty i optymalizator
        memory_per_image_gpu *= 1.5  # Dodatkowe zużycie podczas treningu

    # Analiza dostępnej pamięci RAM
    available_memory = psutil.virtual_memory().available
    cpu_count = os.cpu_count()
    usable_memory = available_memory * 0.7  # Używamy 70% dostępnej pamięci
    max_images_by_memory = int(usable_memory // (image_memory + masks_memory_per_image))
    max_images_by_cpu = cpu_count

    # Analiza dostępnej pamięci GPU
    max_images_by_gpu = max_batch_size
    if torch.cuda.is_available():
        try:
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_free = gpu_memory_total - torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            usable_gpu_memory = gpu_memory_free * 0.7  # Używamy 70% wolnej pamięci GPU
            total_memory_per_image = memory_per_image_gpu + (model_memory + cuda_overhead) / max_batch_size
            max_images_by_gpu = int(usable_gpu_memory // total_memory_per_image)
            logger.info("Dostępna pamięć GPU: %.2f GB, szacowane zużycie na obraz: %.2f MB, model: %.2f GB, max obiektów: %d",
                        gpu_memory_free / (1024 ** 3), total_memory_per_image / (1024 ** 2), model_memory / (1024 ** 3), max_objects)
        except Exception as e:
            logger.warning("Nie można sprawdzić pamięci GPU: %s. Używam konserwatywnego batch_size.", str(e))
            max_images_by_gpu = 1

    # Wyznaczenie finalnego batch size
    max_possible_batch_size = min(max_images_by_memory, max_images_by_cpu, max_images_by_gpu, max_batch_size)
    batch_size = max(min_batch_size, min(max_possible_batch_size, max_batch_size))

    logger.info("Estymowany batch_size: %d (pamięć RAM: %.2f GB, pamięć GPU: %.2f GB, CPU: %d rdzeni, AMP: %s, training: %s, max obiektów: %d)",
                batch_size, available_memory / (1024 ** 3), gpu_memory_free / (1024 ** 3) if torch.cuda.is_available() else 0,
                cpu_count, use_amp, is_training, max_objects)
    return batch_size