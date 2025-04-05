import numpy as np
import cv2
import os
from ssd.structures.container import Container
import torch


def compute_iou(box1, box2):
    """Oblicza IoU między dwoma boxami: [xmin, ymin, xmax, ymax]."""
    x1, y1, x2, y2 = box1
    x1_g, y1_g, x2_g, y2_g = box2

    xi1 = max(x1, x1_g)
    yi1 = max(y1, y1_g)
    xi2 = min(x2, x2_g)
    yi2 = min(y2, y2_g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_g - x1_g) * (y2_g - y1_g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def draw_boxes(image, boxes, labels, color=(0, 255, 0), thickness=2):
    """Rysuje bounding boxy na obrazie."""
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.putText(image, f"pipe_{label}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def evaluate(dataset, predictions, output_dir, **kwargs):
    """Ewaluacja dla datasetu 'pipes' z zapisem oznaczonych zdjęć."""
    print(f"Evaluating dataset: {dataset.__class__.__name__}")
    num_images = len(dataset)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    iou_threshold = 0.5

    # Folder na obrazy z predykcjami
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for i in range(num_images):
        pred = predictions[i]  # {"boxes": ..., "labels": ..., "scores": ...}
        gt = dataset[i][1]     # {"boxes": ..., "labels": ...}
        
        # Wczytaj obraz z datasetu
        img = dataset[i][0]  # Pierwszy element z __getitem__
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().transpose(1, 2, 0)  # Z CHW do HWC
            img = (img * 255).astype(np.uint8)  # Skalowanie z [0,1] do [0,255]
        else:
            img = img.astype(np.uint8).copy()  # Jeśli już NumPy

        # Ground truth
        gt_boxes = gt["boxes"] if isinstance(gt["boxes"], np.ndarray) else gt["boxes"].cpu().numpy()
        gt_labels = gt["labels"] if isinstance(gt["labels"], np.ndarray) else gt["labels"].cpu().numpy()

        # Predykcje (już w NumPy z inference.py)
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        # Filtruj predykcje z niskim score
        mask = pred_scores > 0.5
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        # Rysuj predykcje (zielone) i ground truth (czerwone)
        img = draw_boxes(img, pred_boxes, pred_labels, color=(0, 255, 0))  # Zielone dla predykcji
        img = draw_boxes(img, gt_boxes, gt_labels, color=(0, 0, 255))     # Czerwone dla GT

        # Zapisz obraz
        img_id = dataset[i][2]  # ID obrazu (index)
        cv2.imwrite(os.path.join(pred_dir, f"image_{img_id:04d}_pred.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Liczenie metryk
        matched = set()
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched.add(best_gt_idx)
            else:
                false_positives += 1

        false_negatives += len(gt_boxes) - len(matched)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    print(f"True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    return {"precision": precision, "recall": recall}