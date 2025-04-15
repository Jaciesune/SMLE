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

sys.stdout.reconfigure(encoding='utf-8')

def validate_model(model, dataloader, device, epoch, model_name):
    model.eval()
    predictions = []
    total_loss = 0.0

    coco_gt = dataloader.dataset.coco
    image_ids = []
    pred_count = 0
    gt_count = len(coco_gt.getAnnIds())

    for images, targets in tqdm(dataloader, desc="Walidacja"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)

        for i, output in enumerate(outputs):
            print(f"[DEBUG] Epoka {epoch}, obraz {i} -> liczba predykcji (raw): {len(output['boxes'])}")

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

                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, width, height],
                    "score": float(score)
                })

            pred_count += len(scores)
            image_ids.append(image_id)

    # Zapis predykcji do pliku
    result_file = f"val/{model_name}/epoch_{epoch}_predictions.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w', encoding="utf-8") as f:
        json.dump(predictions, f)

    if len(predictions) == 0:
        print("Brak predykcji - pomijam COCOeval.")
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

    # Ocena mAP
    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50_95 = coco_eval.stats[0]
    map_50 = coco_eval.stats[1]
    precision = coco_eval.stats[6]
    recall = coco_eval.stats[8]

    print(f"Metryki COCO - mAP@0.5: {map_50:.4f} | mAP@0.5:0.95: {map_50_95:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    val_loss = 1 - map_50_95
    return val_loss, pred_count, gt_count, map_50_95, map_50, precision, recall