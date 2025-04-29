# val_utils.py
import os
import sys
import torch
import cv2
import numpy as np
from utils import filter_and_draw_boxes
from pycocotools.coco import COCO
import json
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from config import (
    CONFIDENCE_THRESHOLD,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    MIN_BOX_AREA_RATIO,
    MAX_BOX_AREA_RATIO
)
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.stdout.reconfigure(encoding='utf-8')

def validate_model(model, dataloader, device, epoch, model_name):
    model.eval()
    predictions = []
    total_loss = 0.0

    coco_gt = dataloader.dataset.coco
    image_ids = []
    pred_count = 0
    gt_count = len(coco_gt.getAnnIds())

    logger.info(f"Rozpoczynanie walidacji dla epoki {epoch}, model: {model_name}")
    logger.info(f"Liczba adnotacji GT: {gt_count}")

    for images, targets in tqdm(dataloader, desc="Walidacja"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(images)

        for i, output in enumerate(outputs):
            logger.debug(f"Epoka {epoch}, obraz {i} -> liczba predykcji (raw): {len(output['boxes'])}")

            boxes = output["boxes"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            labels = output["labels"].detach().cpu().tolist()
            image_id = targets[i]["image_id"].cpu().item()

            for box, score, label in zip(boxes, scores, labels):
                if score < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                image_area = images[i].shape[1] * images[i].shape[2]
                area_ratio = area / image_area

                if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                    continue
                if not (MIN_BOX_AREA_RATIO <= area_ratio <= MAX_BOX_AREA_RATIO):
                    continue

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

    # Zapis predykcji do pliku
    result_file = f"val/{model_name}/epoch_{epoch}_predictions.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w', encoding="utf-8") as f:
        json.dump(predictions, f)
    logger.info(f"Zapisano predykcje do: {result_file}, liczba predykcji: {len(predictions)}")

    if len(predictions) == 0:
        logger.warning("Brak predykcji - pomijam COCOeval.")
        return 1.0, 0, gt_count, 0.0, 0.0, 0.0, 0.0

    # Rysowanie wyników 
    debug_dir = f"val/{model_name}/debug_preds"
    os.makedirs(debug_dir, exist_ok=True)

    for i, image in enumerate(images):
        image_np = image.permute(1, 2, 0).cpu().numpy() * 255
        image_np = image_np.astype(np.uint8).copy()
        image_id = targets[i]["image_id"].cpu().item()
        image_name = f"img_{image_id}_ep{epoch}.jpg"

        for box, score in zip(outputs[i]["boxes"], outputs[i]["scores"]):
            if score < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image_np, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(os.path.join(debug_dir, image_name), image_np)
        logger.debug(f"Zapisano obraz debugowania: {os.path.join(debug_dir, image_name)}")

    # Ocena mAP
    try:
        coco_dt = coco_gt.loadRes(result_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        map_50_95 = coco_eval.stats[0] if coco_eval.stats is not None else 0.0
        map_50 = coco_eval.stats[1] if coco_eval.stats is not None else 0.0
        precision = coco_eval.stats[6] if coco_eval.stats is not None else 0.0
        recall = coco_eval.stats[8] if coco_eval.stats is not None else 0.0

        logger.info(f"Metryki COCO - mAP@0.5: {map_50:.4f} | mAP@0.5:0.95: {map_50_95:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    except Exception as e:
        logger.error(f"Błąd podczas obliczania metryk COCO: {str(e)}")
        map_50_95, map_50, precision, recall = 0.0, 0.0, 0.0, 0.0

    val_loss = 1 - map_50_95
    return val_loss, pred_count, gt_count, map_50_95, map_50, precision, recall