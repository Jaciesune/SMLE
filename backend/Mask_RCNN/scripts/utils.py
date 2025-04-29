import torch
from torch.amp import autocast, GradScaler
import cv2
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from pycocotools.cocoeval import COCOeval
import gc

# Funkcja do treningu jednej epoki
def train_one_epoch(model, dataloader, optimizer, device, epoch, accumulation_steps=8):
    """
    Trenuje model przez jedną epokę z użyciem mixed precision (AMP) i akumulacji gradientów.
    
    Args:
        model: Model PyTorch do treningu
        dataloader: DataLoader z danymi treningowymi
        optimizer: Optymalizator (np. SGD)
        device: Urządzenie (CPU/GPU)
        epoch: Numer bieżącej epoki
        accumulation_steps: Liczba kroków akumulacji gradientów
    Returns:
        Średnia strata dla epoki
    """
    scaler = GradScaler('cuda')
    model.train()
    całkowita_strata = 0

    print(f"\nEpoka {epoch}... (Batchy: {len(dataloader)})")
    optimizer.zero_grad()

    for batch_idx, (obrazy, cele) in enumerate(dataloader):
        # Zwolnienie pamięci GPU i synchronizacja
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        obrazy = [img.to(device) for img in obrazy]
        nowe_cele = [{k: v.to(device) for k, v in t.items()} for t in cele]

        with autocast('cuda'):
            słownik_strat = model(obrazy, nowe_cele)
            strata = sum(strata for strata in słownik_strat.values()) / accumulation_steps

        scaler.scale(strata).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        całkowita_strata += strata.item() * accumulation_steps
        print(f"Batch {batch_idx+1}/{len(dataloader)} - Strata: {strata.item() * accumulation_steps:.4f}")

        # Zwolnienie pamięci RAM po każdym batchu
        del obrazy, nowe_cele, słownik_strat, strata
        gc.collect()

    # Ostatni krok, jeśli liczba batchy nie jest podzielna przez accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return całkowita_strata / len(dataloader)

# Funkcja do dekodowania RLE
def decode_rle_segmentation(segmentation):
    """
    Dekoduje segmentację w formacie RLE do maski binarnej.
    
    Args:
        segmentation: Słownik z kluczami 'counts' i 'size'
    Returns:
        Maska binarna jako numpy array
    """
    rle = {
        "counts": segmentation["counts"].encode('utf-8'),
        "size": segmentation["size"]
    }
    return coco_mask.decode(rle)

# Funkcja do tworzenia pełnej maski
def create_full_mask(masks, bboxes, image_shape):
    """
    Tworzy pełną maskę o rozmiarze obrazu, umieszczając maski w odpowiednich miejscach bboxów.
    
    Args:
        masks: Lista masek binarnych (rozmiar bboxa)
        bboxes: Lista bboxów w formacie [x_min, y_min, szerokość, wysokość]
        image_shape: Tuple (wysokość, szerokość) obrazu
    Returns:
        Pełna maska binarna jako numpy array
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

# Funkcja walidacji modelu
def validate_model(model, dataloader, device, epoch, nazwa_modelu, ścieżka_coco_gt):
    """
    Waliduje model, oblicza metryki (mAP) na podstawie IoU pełnych masek i zapisuje gt_image oraz pred_image.
    
    Args:
        model: Model PyTorch do walidacji
        dataloader: DataLoader z danymi walidacyjnymi
        device: Urządzenie (CPU/GPU)
        epoch: Numer bieżącej epoki
        nazwa_modelu: Nazwa modelu dla zapisu wyników
        ścieżka_coco_gt: Ścieżka do pliku COCO z ground truth
    Returns:
        średnia_strata_walidacyjna, liczba_predykcji, liczba_gt, mAP_bbox, mAP_seg
    """
    model.eval()
    całkowita_strata_walidacyjna = 0
    całkowita_liczba_predykcji = 0
    całkowita_liczba_gt = 0
    pełne_iou_sum = 0
    pełne_iou_count = 0
    # Ścieżki dostosowane do struktury Dockera
    ścieżka_zapisu = f"/app/backend/Mask_RCNN/logs/val/{nazwa_modelu}/epoch_{epoch:02d}"
    ścieżka_zapisu_gt = f"/app/backend/Mask_RCNN/logs/val/{nazwa_modelu}/gt_image"
    os.makedirs(ścieżka_zapisu, exist_ok=True)

    # Wczytanie danych ground truth
    coco_gt = COCO(ścieżka_coco_gt)
    coco_dt = []
    nowy_rozmiar = (1024, 1024)  # Docelowy rozmiar - może wymagać dostosowania, jeśli zmienisz image_size

    # Przeskalowanie adnotacji ground truth
    orig_sizes = {img["id"]: (img["height"], img["width"]) for img in coco_gt.loadImgs(coco_gt.getImgIds())}
    ann_ids = coco_gt.getAnnIds(imgIds=coco_gt.getImgIds())
    anns = coco_gt.loadAnns(ann_ids)
    anns_by_image = {}
    for ann in anns:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    for img_id, annotations in anns_by_image.items():
        orig_h, orig_w = orig_sizes[img_id]
        scale_x = nowy_rozmiar[1] / orig_w
        scale_y = nowy_rozmiar[0] / orig_h

        for ann in annotations:
            bbox = ann["bbox"]
            bbox[0] *= scale_x  # x_min
            bbox[1] *= scale_y  # y_min
            bbox[2] *= scale_x  # szerokość
            bbox[3] *= scale_y  # wysokość
            ann["bbox"] = bbox

            mask = decode_rle_segmentation(ann["segmentation"])
            mask_resized = cv2.resize(mask, (int(bbox[2]), int(bbox[3])), interpolation=cv2.INTER_NEAREST)
            rle = coco_mask.encode(np.asfortranarray(mask_resized.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            ann["segmentation"] = rle
            ann["area"] = int(np.sum(mask_resized))

    coco_gt.dataset["annotations"] = anns
    coco_gt.createIndex()

    # Jednorazowe generowanie gt_image, jeśli folder nie istnieje
    if not os.path.exists(ścieżka_zapisu_gt):
        os.makedirs(ścieżka_zapisu_gt, exist_ok=True)
        print("Generowanie gt_image jednorazowo...")
        for idx, (obrazy, cele) in enumerate(dataloader):
            for i, (obraz, cel) in enumerate(zip(obrazy, cele)):
                obraz_np = (obraz.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                obraz_np = cv2.cvtColor(obraz_np, cv2.COLOR_RGB2BGR)

                if "image_id" not in cel:
                    print(f"Batch {idx}, Obraz {i}: Brak image_id, pomijam.")
                    continue
                id_obrazu = int(cel["image_id"].cpu().numpy()[0])

                # Pobieranie GT i tworzenie pełnej maski GT
                gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=id_obrazu))
                gt_masks = [decode_rle_segmentation(ann["segmentation"]) for ann in gt_anns]
                gt_bboxes = [ann["bbox"] for ann in gt_anns]
                full_gt_mask = create_full_mask(gt_masks, gt_bboxes, nowy_rozmiar)

                # Tworzenie gt_image z boxami i maskami
                gt_image = obraz_np.copy()
                for ann in gt_anns:
                    x, y, w, h = map(int, ann["bbox"])
                    cv2.rectangle(gt_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Niebieski obrys
                    mask = decode_rle_segmentation(ann["segmentation"])
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    full_mask = np.zeros_like(obraz_np[:, :, 0], dtype=np.uint8)
                    full_mask[y:y + h, x:x + w] = mask
                    gt_image[full_mask > 0] = [255, 0, 0]  # Czerwony dla maski GT

                # Zapis gt_image
                cv2.imwrite(f"{ścieżka_zapisu_gt}/gt_image_{idx}_{i}.png", gt_image)

    print(f"Walidacja epoki {epoch}...")

    for idx, (obrazy, cele) in enumerate(dataloader):
        # Zwolnienie pamięci GPU i synchronizacja
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        obrazy = [img.to(device) for img in obrazy]
        nowe_cele = [{k: v.to(device) for k, v in t.items()} for t in cele]

        # Obliczenie straty w trybie treningowym
        model.train()
        with torch.no_grad():
            with autocast('cuda'):
                słownik_strat = model(obrazy, nowe_cele)
                strata_walidacyjna = sum(strata for strata in słownik_strat.values())
        całkowita_strata_walidacyjna += strata_walidacyjna.item()

        # Predykcje w trybie ewaluacji
        model.eval()
        with torch.no_grad():
            wyniki = model(obrazy)

        for i, (obraz, wynik, cel) in enumerate(zip(obrazy, wyniki, cele)):
            obraz_np = (obraz.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            obraz_np = cv2.cvtColor(obraz_np, cv2.COLOR_RGB2BGR)

            liczba_gt = cel["boxes"].shape[0] if "boxes" in cel and cel["boxes"].numel() > 0 else 0
            całkowita_liczba_gt += liczba_gt

            if liczba_gt == 0 or "image_id" not in cel:
                print(f"Batch {idx}, Obraz {i}: Brak adnotacji GT lub image_id, pomijam.")
                continue
            id_obrazu = int(cel["image_id"].cpu().numpy()[0])

            # Przygotowanie predykcji
            pred_masks = [maska.cpu().numpy()[0] > 0.5 for maska in wynik["masks"]]
            pred_bboxes = [box.cpu().numpy() for box in wynik["boxes"]]
            pred_scores = wynik["scores"].cpu().numpy()
            pred_labels = wynik["labels"].cpu().numpy()

            # Tworzenie pełnej maski predykcji
            full_pred_mask = np.zeros(nowy_rozmiar, dtype=np.uint8)
            for mask in pred_masks:
                full_pred_mask |= mask.astype(np.uint8)  # Łączenie wszystkich masek w pełnym rozmiarze

            # Filtrowanie predykcji
            filtered_preds = [
                (box, mask, score, label)
                for box, mask, score, label in zip(pred_bboxes, pred_masks, pred_scores, pred_labels)
                if score >= 0.1  # Obniżony próg
            ]
            liczba_predykcji = len(filtered_preds)
            całkowita_liczba_predykcji += liczba_predykcji

            # Tworzenie coco_dt dla bboxów
            for box, mask, score, label in filtered_preds:
                x_min, y_min, x_max, y_max = map(int, box)
                szerokość = max(0, x_max - x_min)
                wysokość = max(0, y_max - y_min)
                if szerokość > 0 and wysokość > 0:
                    mask_cropped = mask[y_min:y_min + wysokość, x_min:x_min + szerokość]
                    rle = coco_mask.encode(np.asfortranarray(mask_cropped.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    coco_dt.append({
                        "image_id": id_obrazu,
                        "category_id": int(label),
                        "bbox": [x_min, y_min, szerokość, wysokość],
                        "score": float(score),
                        "segmentation": rle
                    })

            # Pobieranie GT i tworzenie pełnej maski GT (tylko do obliczeń IoU)
            gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=id_obrazu))
            gt_masks = [decode_rle_segmentation(ann["segmentation"]) for ann in gt_anns]
            gt_bboxes = [ann["bbox"] for ann in gt_anns]
            full_gt_mask = create_full_mask(gt_masks, gt_bboxes, nowy_rozmiar)

            # Tworzenie pred_image z boxami i maskami
            pred_image = obraz_np.copy()
            for box, mask, score, label in filtered_preds:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(pred_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Zielony obrys
                pred_image[mask > 0] = [0, 0, 255]  # Czerwony dla maski predykcji

            # Zapis pred_image
            cv2.imwrite(f"{ścieżka_zapisu}/pred_image_{idx}_{i}.png", pred_image)

            # Obliczanie IoU dla pełnych masek
            if full_pred_mask.sum() > 0 and full_gt_mask.sum() > 0:
                intersection = np.logical_and(full_pred_mask, full_gt_mask).sum()
                union = np.logical_or(full_pred_mask, full_gt_mask).sum()
                full_iou = intersection / union if union > 0 else 0
                pełne_iou_sum += full_iou
                pełne_iou_count += 1
                print(f"Batch {idx}, Obraz {i}, Predykcje: {liczba_predykcji}, Liczba GT: {liczba_gt}, Ratio: {liczba_predykcji/liczba_gt}, IoU pełnych masek: {full_iou:.3f}")
            else:
                print(f"Batch {idx}, Obraz {i}, Predykcje: {liczba_predykcji}, Liczba GT: {liczba_gt}, Ratio: {liczba_predykcji/liczba_gt}, IoU pełnych masek: 0.000 (puste maski)")

    # Ocena COCO dla bboxów
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

        # mAP_seg jako średnie IoU pełnych masek
        mAP_seg = pełne_iou_sum / pełne_iou_count if pełne_iou_count > 0 else 0

    średnia_strata_walidacyjna = całkowita_strata_walidacyjna / len(dataloader) if len(dataloader) > 0 else 0
    print(f"Średnia strata walidacyjna: {średnia_strata_walidacyjna:.4f}")
    print(f"Całkowita liczba predykcji: {całkowita_liczba_predykcji}, GT: {całkowita_liczba_gt}")
    print(f"mAP_bbox: {mAP_bbox:.4f}, mAP_seg (IoU pełnych masek): {mAP_seg:.4f}")
    return średnia_strata_walidacyjna, całkowita_liczba_predykcji, całkowita_liczba_gt, mAP_bbox, mAP_seg