import os
import torch
import cv2
import numpy as np
from utils import filter_and_draw_boxes
from pycocotools.coco import COCO
import json
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval


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
            boxes = output["boxes"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            labels = output["labels"].detach().cpu().tolist()
            image_id = targets[i]["image_id"].cpu().item()

            for box, score, label in zip(boxes, scores, labels):
                if score < 0.05:
                    continue
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
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
    with open(result_file, 'w') as f:
        json.dump(predictions, f)

    # Sprawdzenie czy są jakiekolwiek predykcje
    if len(predictions) == 0:
        print("Brak predykcji – pomijam COCOeval.")
        return 1.0, 0, gt_count, 0.0, 0.0, 0.0, 0.0


    # Ocena mAP itd.
    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50_95 = coco_eval.stats[0]  # mAP@[.5:.95]
    map_50 = coco_eval.stats[1]     # mAP@0.5
    precision = coco_eval.stats[6]  # AR@1
    recall = coco_eval.stats[8]     # AR@10

    print(f"Metryki COCO - mAP@0.5: {map_50:.4f} | mAP@0.5:0.95: {map_50_95:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    val_loss = 1 - map_50_95
    return val_loss, pred_count, gt_count, map_50_95, map_50, precision, recall